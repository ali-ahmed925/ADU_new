#!/bin/bash

CUDA_DEVICE=$1
DATASET=office_home_df # [office_home_df, domainnet_mini_df, domainnet_df] 
SEED=1
CFG=vit_b16_ep50
SHOTS=8
TRAINER=IVLP_VL_Adapter_Prompt

DOMAIN_WEIGHT=30
MMD=0.5

DIR=/path/to/directory/to/save/results


echo "Run this job and save the output to ${DIR}"

# 実行コマンド
CUDA_VISIBLE_DEVICES=${CUDA_DEVICE} python train_loop.py \
    --root /nas/data/gotoyuta/Dataset/ \
    --seed ${SEED} \
    --trainer ${TRAINER}  \
    --dataset-config-file configs/datasets/${DATASET}.yaml \
    --config-file configs/trainers/${CFG}.yaml \
    --output-dir ${DIR} \
    --num_shots ${SHOTS} \
    --dataset_name ${DATASET} \
    --mmd_weight ${MMD} \
    --domainloss_weight ${DOMAIN_WEIGHT} \
    --is_domain_divided \
    --use_domain_cls_loss \