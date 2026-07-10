#!/bin/bash
# Reproduces ADU on DomainNetMini-126 with paper-exact hyperparameters:
#   gamma=30, lambda=10, 50 epochs, 8-shot, batch=8, 3 seeds.
#
# Usage (forget only sketch — matches our EBM comparison):
#   ./run_domainnet_mini.sh 0 ./output/paper_sketch sketch
#
# Usage (full Table 1 reproduction — all forget combinations, ~73h on laptop):
#   ./run_domainnet_mini.sh 0 ./output/paper_all
#
# GPU_ID: default 0
# OUTPUT_DIR: default ./output/paper_domainnet_mini
# FORGET: if provided, runs only that specific forget combination

CUDA_DEVICE=${1:-0}
OUTPUT_DIR=${2:-"./output/paper_domainnet_mini"}
shift 2
FORGET_FILTER="$@"   # e.g. "sketch" or "clipart painting" or empty for full run

DATA_ROOT="/home/owais/machine unlearning/ebm_unlearning/data/domainnet"
DOMAIN_WEIGHT=30   # gamma
MMD_WEIGHT=10      # lambda

if [ -n "$FORGET_FILTER" ]; then
    FILTER_ARG="--run_forget_domains ${FORGET_FILTER}"
    echo "Running forget combination: [${FORGET_FILTER}], 3 seeds"
else
    FILTER_ARG=""
    echo "Running FULL power-set (all 14 combinations × 3 seeds)"
fi

CUDA_VISIBLE_DEVICES=${CUDA_DEVICE} conda run -n myn_again \
    python train_loop.py \
    --root "${DATA_ROOT}" \
    --trainer IVLP_VL_Adapter_Prompt \
    --dataset-config-file configs/datasets/domainnet_mini_paper_df.yaml \
    --config-file configs/trainers/vit_b16_ep50.yaml \
    --output-dir "${OUTPUT_DIR}" \
    --num_shots 8 \
    --dataset_name domainnet_mini_paper_df \
    --domainloss_weight ${DOMAIN_WEIGHT} \
    --mmd_weight ${MMD_WEIGHT} \
    --use_domain_cls_loss \
    --is_domain_divided \
    ${FILTER_ARG}
