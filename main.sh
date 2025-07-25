#!/bin/bash

CUDA_DEVICE=$1
DATASET=office_home_df # [office_home_df, domainnet_mini_df, ImageNetDF] 
SEED=1
CFG=vit_b16_ep50
NCTX=4
DEPTH_VISION=9
DEPTH_TEXT=9
SHOTS=8
DATASETSEED=0
TRAINER=IVLP_VL_Adapter_Prompt

IS_DOMAIN_DIVIDED_FLAG=""
USE_DOMAIN_CLS_LOSS_FLAG=""
USE_NEAREST_NEIGHBOR_LOSS_FLAG=""

USE_VISION_ADAPTER=False
USE_TEXT_ADAPTER=False

#　頻繁に変えるargs
EXPNAME=rebuttal
SUBEXPNAME=imbalanced_domain

DOMAIN_WEIGHT=30
MMD=0.5
BASELINE=False
## imbalanced_domain labels
DROP_RATE=0.0
DROP_DOMAIN_IDX=1

if [ "$BASELINE" = "False" ]; then
    USE_DOMAIN_CLS_LOSS=true  # DDL
    USE_CROSSATTENTION_FLAG="True"  # InstaPG
else
    USE_DOMAIN_CLS_LOSS=false  # DDL
    USE_CROSSATTENTION_FLAG="False"  # InstaPG
fi

# 実行ディレクトリ
DIR=/nas/data/kawamura/ADU/${EXPNAME}/${SUBEXPNAME}/${DATASET}/

echo "Run this job and save the output to ${DIR}"

# 実行コマンド
CUDA_VISIBLE_DEVICES=${CUDA_DEVICE} python train_loop.py \
    --root /nas/data/gotoyuta/Dataset/ \
    --seed ${SEED} \
    --trainer ${TRAINER}  \
    --dataset-config-file configs/datasets/${DATASET}.yaml \
    --config-file configs/trainers/IVLP/${CFG}.yaml \
    --forget_domains "${DOMAIN_LIST[@]}" \
    --output-dir ${DIR} \
    --num_shots ${SHOTS} \
    --dataset_name ${DATASET} \
    --dataset_seed ${DATASETSEED} \
    --experiment_name ${EXPNAME} \
    --sub_experiment_name ${SUBEXPNAME} \
    --mmd_weight ${MMD} \
    --domainloss_weight ${DOMAIN_WEIGHT} \
    --drop_rate ${DROP_RATE} \
    --drop_domain_idx ${DROP_DOMAIN_IDX} \
    --is_domain_divided \
    --use_domain_cls_loss \
    ${USE_DOMAIN_CLS_LOSS_FLAG} \
    ${USE_NEAREST_NEIGHBOR_LOSS_FLAG} \
    TRAINER.IVLP.PROMPT_DEPTH_VISION ${DEPTH_VISION} \
    TRAINER.IVLP.N_CTX_VISION ${NCTX} \
    TRAINER.IVLP.PROMPT_DEPTH_TEXT ${DEPTH_TEXT} \
    TRAINER.IVLP.N_CTX_TEXT ${NCTX} \
    USE_CROSSATTENTION ${USE_CROSSATTENTION_FLAG} \
    INSERT_LAYER_ATTN ${DEPTH_VISION} \
    USE_TEXT_ADAPTER ${USE_TEXT_ADAPTER} \
    USE_VISION_ADAPTER ${USE_VISION_ADAPTER}