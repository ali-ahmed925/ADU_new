#!/bin/bash
export CUDA_VISIBLE_DEVICES=$1
#cd ../..

# custom config
DATA="/nas/data/gotoyuta/Dataset/"
TRAINER=VPT_Local

DATASET=$2
SEED=$3
CFG=$4
NCTX=$5 # 8
DEPTH_VISION=$6 # 9
TOPK=$7
# 7番目以降の引数をアンダースコアでつなげる
DOMAIN_LIST=("${@:8}")

DOMAIN_SEC=$(IFS=-; echo "${DOMAIN_LIST[*]}")
DOMAIN_COUNT=${#DOMAIN_LIST[@]}
TODAY=$(date +"%Y%m%d_%H%M%S")

DIR=/nas/data/gotoyuta/Result_Domain_Forgetting/${DATASET}/${TRAINER}/FORGET_DOMAIN${DOMAIN_COUNT}/${DOMAIN_SEC}/${CFG}/nctx-vision${NCTX}_prmpt-depth${DEPTH_VISION}_topk${TOPK}/seed${SEED}/${TODAY}
if [ -d "$DIR" ]; then
    echo "Results are available in ${DIR}."
else
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