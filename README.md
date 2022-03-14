# Project 3 - Dicemen

## Installation

1. Create the environment
    ```bash
    conda env create -f environment.yml
    ```
2. Activate the environment
    ```bash
    conda activate mp_project3
    ```
3. Run the following commands to install the remaining dependencies about pytorch-geometric in to the activated environment
    ```bash
    pip install torch-scatter -f https://pytorch-geometric.com/whl/torch-1.7.0+cu101.html
    pip install torch-sparse -f https://pytorch-geometric.com/whl/torch-1.7.0+cu101.html
    pip install torch-cluster -f https://pytorch-geometric.com/whl/torch-1.7.0+cu101.html
    pip install torch-spline-conv -f https://pytorch-geometric.com/whl/torch-1.7.0+cu101.html
    pip install torch-geometric
    ```
    Note: 
    If pip commands give error, it is necessary to follow these steps below:
    ```bash
    wget https://pytorch-geometric.com/whl/torch-1.7.0+cu101/torch_cluster-1.5.9-cp38-cp38-linux_x86_64.whl
    wget https://pytorch-geometric.com/whl/torch-1.7.0+cu101/torch_scatter-2.0.6-cp38-cp38-linux_x86_64.whl
    wget https://pytorch-geometric.com/whl/torch-1.7.0+cu101/torch_sparse-0.6.9-cp38-cp38-linux_x86_64.whl
    wget https://pytorch-geometric.com/whl/torch-1.7.0+cu101/torch_spline_conv-1.2.1-cp38-cp38-linux_x86_64.whl
    pip install torch-geometric
    pip install wheel
    pip install torch_cluster-1.5.9-cp38-cp38-linux_x86_64.whl
    pip install torch_scatter-2.0.6-cp38-cp38-linux_x86_64.whl
    pip install torch_sparse-0.6.9-cp38-cp38-linux_x86_64.whl
    pip install torch_spline_conv-1.2.1-cp38-cp38-linux_x86_64.whl
    ```
4. Submit the training task to GPU with the following command (indicated time necessary to reproduce results)
    ```bash
    cd codebase/
    bsub -n 4 -W 24:00 -o sample_test -R "rusage[mem=4096, ngpus_excl_p=1]" -R "select[gpu_mtotal0>=10240]" python train.py ../configs/convgatadv.yaml
    ```
5. Submit the prediction task to GPU with the following command
    ```bash
    bsub -n 4 -W 2:00 -o sample_test -R "rusage[mem=4096, ngpus_excl_p=1]" -R "select[gpu_mtotal0>=10240]" python test.py ../configs/convgatadv.yaml --gen_model_file gen_model_200000.pt --disc_model_file disc_model_200000.pt
    ```
Results will be saved under a directory with the same name as the model (NOT in the top directory as the vanilla code did).