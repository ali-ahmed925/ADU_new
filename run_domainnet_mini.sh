#!/bin/bash
# Reproduces ADU on DomainNetMini with paper-exact hyperparameters:
#   gamma=30 (DDL CE weight), lambda=10 (MMD weight), 50 epochs
# Usage: ./run_domainnet_mini.sh <GPU_ID> <DATA_ROOT> <OUTPUT_DIR>

CUDA_DEVICE=${1:-0}
DATA_ROOT=${2:-"/path/to/datasets"}
OUTPUT_DIR=${3:-"./output/domainnet_mini"}

DOMAIN_WEIGHT=30   # gamma in paper: weight for domain classifier CE loss
MMD_WEIGHT=10      # lambda in paper: weight for MMD^2 loss (negated in L_domain)

CUDA_VISIBLE_DEVICES=${CUDA_DEVICE} python train_loop.py \
    --root "${DATA_ROOT}" \
    --trainer IVLP_VL_Adapter_Prompt \
    --dataset-config-file configs/datasets/domainnet_mini_df.yaml \
    --config-file configs/trainers/vit_b16_ep50.yaml \
    --output-dir "${OUTPUT_DIR}" \
    --num_shots 8 \
    --dataset_name domainnet_mini_df \
    --domainloss_weight ${DOMAIN_WEIGHT} \
    --mmd_weight ${MMD_WEIGHT} \
    --use_domain_cls_loss \
    --is_domain_divided \
    --forget_domains sketch
