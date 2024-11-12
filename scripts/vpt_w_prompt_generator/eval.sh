#!/bin/bash
export CUDA_VISIBLE_DEVICES=$1
#cd ../..

# custom config
DATA="/nas/data/gotoyuta/Dataset/"
TRAINER=VPT_w_NNL_Local_PromptGenerator

DATASET=$2
SEED=$3
CFG=$4
NCTX=$5 # 8
DEPTH_VISION=$6 # 9

DIR=$7
TOPK=$8
# 7番目以降の引数をアンダースコアでつなげる
DOMAIN_LIST=("${@:9}")

DOMAIN_SEC=$(IFS=-; echo "${DOMAIN_LIST[*]}")
DOMAIN_COUNT=${#DOMAIN_LIST[@]}
TODAY=$(date +"%Y%m%d_%H%M%S")


# if [ -d "$DIR" ]; then
#     echo "Results are available in ${DIR}."
#else
echo "Run this job and save the output to ${DIR}"

python train.py \
    --root ${DATA} \
    --seed ${SEED} \
    --trainer ${TRAINER} \
    --dataset-config-file configs/datasets/${DATASET}.yaml \
    --config-file configs/trainers/VPT/${CFG}.yaml \
    --forget_domains "${DOMAIN_LIST[@]}" \
    --output-dir ${DIR} \
    --model-dir ${DIR} \
    --eval-only \
    --topk ${TOPK} \
    --load-epoch 50 \
    TRAINER.VPT.PROMPT_DEPTH_VISION ${DEPTH_VISION} \
    TRAINER.VPT.N_CTX_VISION ${NCTX} \
    TRAINER.COOP.N_CTX ${NCTX}
    # TRAINER.${TRAINER}. ${NCTX} \
# fi