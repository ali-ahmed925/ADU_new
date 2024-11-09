#!/bin/bash
export CUDA_VISIBLE_DEVICES=$1
#cd ../..
# custom config
DATASET=$2
SEED=$3
CFG=$4
NCTX=$5 # 8
DEPTH_VISION=$6 # 9
TOPK=$7
IS_BLOCK_SHUFFLE=$8 # true 0 false 1
GRID=$9
# 7番目以降の引数をアンダースコアでつなげる
DOMAIN_LIST=("${@:10}")

DOMAIN_SEC=$(IFS=-; echo "${DOMAIN_LIST[*]}")
DOMAIN_COUNT=${#DOMAIN_LIST[@]}
TODAY=$(date +"%Y%m%d_%H%M%S")

DATA="/nas/data/gotoyuta/Dataset/"
TRAINER=VPT_Local_w_DC_Divided
# if $IS_BLOCK_SHUFFLE; then
#     TRAINER=
# else 
    
# fi 

if $IS_BLOCK_SHUFFLE; then
    DIR=/nas/data/gotoyuta/Result_Domain_Forgetting/${DATASET}/${TRAINER}/FORGET_DOMAIN${DOMAIN_COUNT}/${DOMAIN_SEC}/${CFG}/nctx-vision${NCTX}_prmpt-depth${DEPTH_VISION}_topk${TOPK}_BLOCK-SHUFFLE_GRID${GRID}/seed${SEED}/${TODAY}
    echo "Run this job and save the output to ${DIR}"
    
    python train.py \
    --root ${DATA} \
    --seed ${SEED} \
    --trainer ${TRAINER} \
    --dataset-config-file configs/datasets/${DATASET}.yaml \
    --config-file configs/trainers/VPT/${CFG}.yaml \
    --forget_domains "${DOMAIN_LIST[@]}" \
    --output-dir ${DIR} \
    --is_block_shuffle \
    --topk ${TOPK} \
    --grid_num ${GRID} \
    TRAINER.VPT.PROMPT_DEPTH_VISION ${DEPTH_VISION} \
    TRAINER.VPT.N_CTX_VISION ${NCTX} 
else
    DIR=/nas/data/gotoyuta/Result_Domain_Forgetting/${DATASET}/${TRAINER}/FORGET_DOMAIN${DOMAIN_COUNT}/${DOMAIN_SEC}/${CFG}/nctx-vision${NCTX}_prmpt-depth${DEPTH_VISION}_topk${TOPK}/seed${SEED}/${TODAY}
    echo "Run this job and save the output to ${DIR}"

    python train.py \
    --root ${DATA} \
    --seed ${SEED} \
    --trainer ${TRAINER} \
    --dataset-config-file configs/datasets/${DATASET}.yaml \
    --config-file configs/trainers/VPT/${CFG}.yaml \
    --forget_domains "${DOMAIN_LIST[@]}" \
    --output-dir ${DIR} \
    --topk ${TOPK} \
    TRAINER.VPT.PROMPT_DEPTH_VISION ${DEPTH_VISION} \
    TRAINER.VPT.N_CTX_VISION ${NCTX} \
    # TRAINER.${TRAINER}. ${NCTX} \
fi