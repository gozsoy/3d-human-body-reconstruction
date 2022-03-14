from abc import ABC
import warnings

import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.nn import SuperGATConv
from torch_geometric.data import Data, Batch

from .body_model import BodyModel
from torchvision import models
from .augment import AugmentPipe


class BaseModel(nn.Module, ABC):
    def __init__(self, cfg):
        super().__init__()

        self.cfg = cfg

        # parameters
        self.bm_path = cfg["data"]["bm_path"]
        self.in_ch = cfg["model"].get("in_ch", 3)
        self.out_ch = cfg["model"].get("out_ch", 70)
        self.img_resolution = cfg["data"]["resy"], cfg["data"]["resx"]

        self.device = cfg.get("device", "cuda")
        self.batch_size = cfg["training"]["batch_size"]

        # body_model
        self.body_model = BodyModel(
            bm_path=self.bm_path, num_betas=10, batch_size=self.batch_size,
        ).to(device=self.device)

    @staticmethod
    def create_model(cfg):
        model_name = cfg["model"]["name"]
        if model_name == "conv":
            model = ConvModel(cfg)
        elif model_name == "convgat":
            model = ConvGATModel(cfg)
        else:
            raise Exception(f"Model `{model_name}` is not defined.")

        return model

    def get_vertices(self, root_loc, root_orient, betas, pose_body, pose_hand):
        """ Fwd pass through the parametric body model to obtain mesh vertices.

        Args:
            root_loc (torch.Tensor): Root location (B, 10).
            root_orient (torch.Tensor): Root orientation (B, 3).
            betas (torch.Tensor): Shape coefficients (B, 10).
            pose_body (torch.Tensor): Body joint rotations (B, 21*3).
            pose_hand (torch.Tensor): Hand joint rotations (B, 2*3).

        Returns:
            mesh vertices (torch.Tensor): (B, 6890, 3)
        """
        body = self.body_model(
            trans=root_loc,
            root_orient=root_orient,
            pose_body=pose_body,
            pose_hand=pose_hand,
            betas=betas,
        )
        vertices = body.v
        return vertices


class ConvModel(BaseModel):
    def __init__(self, cfg):
        super().__init__(cfg)

        self.backbone_f_len = cfg["model"].get("backbone_f_len", 500)
        self._build_net()

    def _build_net(self):
        """ Creates NNs. """
        fc_in_ch = 1 * (self.img_resolution[0] // 2 ** 3) * (self.img_resolution[1] // 2 ** 3)
        self.backbone = nn.Sequential(
            nn.Conv2d(self.in_ch, 5, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
            nn.ReLU(),
            nn.Conv2d(5, 5, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
            nn.ReLU(),
            nn.Conv2d(5, 5, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
            nn.ReLU(),
            nn.Conv2d(5, 1, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
            nn.ReLU(),
            # flattening
            nn.Flatten(),
            nn.Linear(fc_in_ch, self.backbone_f_len),
        )

        self.nn_root_orient = nn.Linear(self.backbone_f_len, 3)
        self.nn_betas = nn.Linear(self.backbone_f_len, 10)
        self.nn_pose_body = nn.Linear(self.backbone_f_len, 63)
        self.nn_pose_hand = nn.Linear(self.backbone_f_len, 6)

    def forward(self, input_data):
        """ Fwd pass.

        Returns (dict):
            mesh vertices (torch.Tensor): (B, 6890, 3)
        """
        image_crop = input_data["image_crop"]
        root_loc = input_data["root_loc"]

        img_encoding = self.backbone(image_crop)

        # regress parameters
        root_orient = self.nn_root_orient(img_encoding)
        betas = self.nn_betas(img_encoding)
        pose_body = self.nn_pose_body(img_encoding)
        pose_hand = self.nn_pose_hand(img_encoding)

        # regress vertices
        vertices = self.get_vertices(root_loc, root_orient, betas, pose_body, pose_hand)
        predictions = {
            "vertices": vertices,
            "root_loc": root_loc,
            "root_orient": root_orient,
            "betas": betas,
            "pose_body": pose_body,
            "pose_hand": pose_hand,
        }
        return predictions


class Linear(nn.Module):
    def __init__(self, linear_size, p_dropout=0.0):
        super(Linear, self).__init__()
        self.l_size = linear_size

        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p_dropout)

        self.w1 = nn.Linear(self.l_size, self.l_size)
        self.batch_norm1 = nn.BatchNorm1d(self.l_size)

        self.w2 = nn.Linear(self.l_size, self.l_size)
        self.batch_norm2 = nn.BatchNorm1d(self.l_size)

    def forward(self, x):
        y = self.relu(x)
        y = self.batch_norm1(y)
        y = self.dropout(y)
        y = self.w1(y)

        y = self.relu(y)
        y = self.batch_norm2(y)
        y = self.dropout(y)
        y = self.w2(y)

        out = x + y

        return out


class ConvGATModel(BaseModel):
    def __init__(
        self,
        cfg,
        graph_depth=3,
        graph_size=64,
        attention_heads=4,
        linear_depth=5,
        linear_dropout=0.05, # https://arxiv.org/abs/1905.05928v1
        linear_size=128,
    ):
        """
        betas: Shape coefficients (10-dimensional vector).
        root_orient: Root orientation of a human body (3-dimensional vector).
        pose_body: Per-joint axis-angle (21*3-dimensional vector, where 21 indicates the number of joints).
        pose_hand: Axis-angles for two hands (2*3-dimensional vector).

        => 24-node graph, 10-dim beta output
        """
        super(ConvGATModel, self).__init__(cfg)

        # See https://github.com/pytorch/pytorch/issues/49285
        warnings.filterwarnings("ignore", message="Setting attributes on ParameterList is not supported.")

        device = cfg["device"]
        
        self.image_mean = torch.tensor([0.485, 0.456, 0.406], device=device).reshape([1, -1, 1, 1])
        self.image_std = torch.tensor([0.229, 0.224, 0.225], device=device).reshape([1, -1, 1, 1])

        self.augmentor = AugmentPipe(
            xflip=1, rotate90=1, xint=1, scale=1, rotate=1,
            aniso=0, xfrac=1, brightness=1, contrast=1, lumaflip=1,
            hue=1, saturation=1, cutout = 1, cutout_size=0.25
        )

        resnet_full = models.resnext50_32x4d(pretrained=True)
        modules = list(resnet_full.children())[:-1]
        self.resnet = nn.Sequential(*modules)

        self.resnet_feature_count = 2048 
        self.node_count = 24

        self.distributor = nn.Linear(2048, graph_size * 2 * attention_heads)

        graph_convs = [
            SuperGATConv(
                graph_size * 2 * attention_heads,
                graph_size,
                heads=attention_heads,
                dropout=0.25,
                attention_type="SD",
                is_undirected=True,
            )
            for _ in range(graph_depth)
        ] 
        init_res_conn = [
            nn.Linear(graph_size * 2 * attention_heads, graph_size * 2 * attention_heads)
            for _ in range(graph_depth)
        ]
        self.graph_convs = nn.ModuleList(graph_convs)
        self.init_res_conn = nn.ModuleList(init_res_conn)

        self.nodes_linear = nn.Linear(self.node_count * graph_size * 2 * attention_heads, self.node_count * 3)

        self.betas_linear = nn.Sequential(
            nn.Linear(self.resnet_feature_count, linear_size),
            *[Linear(linear_size, linear_dropout) for _ in range(linear_depth - 1)],
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(linear_size),
            nn.Linear(linear_size, 10),
        )

        undirected_edge_list = [
            [0, 1],
            [0, 2],
            [1, 2],
            [1, 4],
            [2, 5],
            [4, 7],
            [5, 8],
            [4, 5],
            [7, 10],
            [8, 11],
            [7, 8],
            [10, 11],
            [0, 3],
            [3, 6],
            [6, 9],
            [9, 12],
            [12, 15],
            [9, 13],
            [13, 16],
            [16, 18],
            [18, 20],
            [20, 22],
            [9, 14],
            [14, 17],
            [17, 19],
            [19, 21],
            [21, 23],
            [13, 14],
            [16, 17],
            [18, 19],
            [20, 21],
            [22, 23],
        ]

        full_edge_list = undirected_edge_list + [(j, i) for i, j in undirected_edge_list]
        self.edge_index = (
            torch.tensor(full_edge_list, dtype=torch.long).t().contiguous().to(device=device)
        )

    def forward(self, input_data):

        image_crop = input_data["image_crop"] # range: 0-1
        root_loc = input_data["root_loc"]

        batch_size = image_crop.shape[0]

        if self.training:
            image_crop = self.augmentor(image_crop * 2 - 1)
            image_crop = (image_crop + 1) / 2

        image_crop = (image_crop - self.image_mean) / self.image_std
        image_resnet = self.resnet(image_crop).view(batch_size, self.resnet_feature_count)
        image_enc = image_resnet.view(batch_size, 1, self.resnet_feature_count).expand(
            -1, self.node_count, -1
        )
        image_enc = F.dropout(image_enc, 0.25, training=self.training)

        node_encodings = self.distributor(image_enc)
        graph_batch = Batch.from_data_list(
            [Data(x=nodes, edge_index=self.edge_index) for nodes in node_encodings]
        )

        x = x_0 = graph_batch.x
        x_mem = x[:, : x.shape[1] // 2]
        att_loss = 0
        for conv, init_res in zip(self.graph_convs, self.init_res_conn):
            x = conv(x, graph_batch.edge_index, batch=graph_batch.batch)
            x_mem_tmp = x
            x = torch.cat([x_mem, x], dim=1) + init_res(x_0)
            x_mem = x_mem_tmp
            x = F.leaky_relu(x, inplace=True)
            att_loss += conv.get_attention_loss()

        rebatched_output_graph = x.view(batch_size, -1)
        nodes = self.nodes_linear(rebatched_output_graph)

        betas = self.betas_linear(image_resnet)

        root_orient = nodes[:, :3]
        pose_body = nodes[:, 3 : 22 * 3]
        pose_hand = nodes[:, 22 * 3 :]

        # regress vertices
        vertices = self.get_vertices(root_loc, root_orient, betas, pose_body, pose_hand)
        predictions = {
            "vertices": vertices,
            "root_loc": root_loc,
            "root_orient": root_orient,
            "betas": betas,
            "pose_body": pose_body,
            "pose_hand": pose_hand,
        }
        return predictions, att_loss
