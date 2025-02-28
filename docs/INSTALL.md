# Installation

### Acknowledgement: This readme file for installing datasets is modified from [MaPLe's](https://github.com/muzairkhattak/multimodal-prompt-learning) official repository.

This codebase is tested on Ubuntu 22.04.2 LTS with python 3.10. Follow the below steps to create environment and install dependencies.

* Setup conda environment (recommended).
```bash
# Create a conda environment
conda create -y -n domain-forgetting python=3.10

# Activate the environment
conda activate domain-forgetting

# Install torch (requires version >= 1.8.1) and torchvision
# Please refer to https://pytorch.org/ if you need a different cuda version
pip install torch==2.5.0 torchvision==0.20.0 torchaudio==2.5.0 --index-url https://download.pytorch.org/whl/cu124
```

* Install dassl library.
```bash
# Instructions borrowed from https://github.com/KaiyangZhou/Dassl.pytorch#installation

# Clone this repo
git clone https://github.com/KaiyangZhou/Dassl.pytorch.git
cd Dassl.pytorch/

# Install dependencies
pip install -r requirements.txt

# Install this library (no need to re-build if the source code is modified)
python setup.py develop
cd ..
```

* Clone Domain-Forgetting code repository and install requirements
```bash
# Clone Domain-Forgetting code base
git clone https://github.com/510yuta/Domain-Forgetting.git

cd Domain-Forgetting/
# Install requirements

pip install -r requirements.txt

# Update setuptools package 
pip install setuptools==59.5.0
pip install opencv-python
pip install matplotlib
pip install einops
pip install segment_anything
```
