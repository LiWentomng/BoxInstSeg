
## Environments

- Linux or macOS
- Python 3.6+  (Python 3.8 in our envs)
- PyTorch 1.7+ (1.9.0 in our envs)
- CUDA 9.2+ (CUDA 11.1 in our envs)
- GCC 5+
- mmdet==2.25.0
- mmcv-full==1.5.0 ([MMCV](https://mmcv.readthedocs.io/en/latest/#installation))

## Installation

1. Create a conda virtual environment and activate it.

    ```shell
    conda create -n boxinstseg python=3.8 -y
    conda activate boxinstseg
    ```

2. Install PyTorch and torchvision following the [official instructions](https://pytorch.org/get-started/previous-versions/), e.g.,
    
    ```shell
    pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html

   ```
    Note: Make sure that your compilation CUDA version and runtime CUDA version match. 


3. Install mmcv using mim

    ```shell
    pip install -U openmim
    mim install mmcv-full==1.5.0
   ```

4. Clone the this repository.

    ```shell
    git clone https://github.com/LiWentomng/BoxInstSeg.git
    cd BoxInstSeg
    ```

5. Install build requirements and then install MMDetection.

    ```shell
    bash setup.sh   #compile the whole envs 
    ```
   Or 
    ```shell
    pip install -v -e .
    # "-v" means verbose, or more output
    # "-e" means installing a project in editable mode,

    cd mmdet/ops/pairwise #compiling for the opeartion using in boxinst
    python setup.py build develop 

    cd mmdet/ops/tree_filter #compiling for the opeartion using in boxlevelset
    python setup.py build develop
    ```


