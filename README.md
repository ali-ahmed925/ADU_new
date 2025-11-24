# (NeurIPS 2025 Spotlight) Approximate Domain Unlearning for Vision-Language Models 

## Overview
This is the official code repository for the paper [Approximate Domain Unlearning for Vision-Language Models](https://kodaikawamura.github.io/Domain_Unlearning/) accepted at NeurIPS 2025 as a spotlight.

## Installation 
* Setup conda environment (recommended).

```bash
# Create a conda environment
conda env create -f environment.yml

# Activate the environment
conda activate adu

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

* Install requirements
```bash
# Install requirements

pip install -r requirements.txt

```

* Prepare Datasets
We followed the instructions in [dassl.pytorch](https://github.com/KaiyangZhou/Dassl.pytorch/blob/master/DATASETS.md) to prepare the datasets. Please refer to the instructions there.

## How to Run
You can run the training and evaluation using the provided `main.sh <GPU_ID>` script. Please modify the parameters in the script as needed.

