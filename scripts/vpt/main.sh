#!/bin/bash
export CUDA_VISIBLE_DEVICES=$1
#cd ../..

# custom config
DATA="/nas/data/gotoyuta/Dataset/"
TRAINER=VPT

DATASET=$2
SEED=$3
CFG=$4
NCTX=$5 # 8
DEPTH_VISION=$6 # 9
USE_DOMAIN_CLS_LOSS=${7}
USE_NEAREST_NEIGHBOR_LOSS=${8}
IS_DOMAIN_DIVIDED=${9}
# 7番目以降の引数をアンダースコアでつなげる
DOMAIN_LIST=("${@:10}")

DOMAIN_SEC=$(IFS=-; echo "${DOMAIN_LIST[*]}")
DOMAIN_COUNT=${#DOMAIN_LIST[@]}
TODAY=$(date +"%Y%m%d_%H%M%S")
SHOTS=16

# フラグでCLIオプションを切り替え
IS_DOMAIN_DIVIDED_FLAG=""
USE_DOMAIN_CLS_LOSS_FLAG=""
USE_NEAREST_NEIGHBOR_LOSS_FLAG=""

# --is_domain_divided を ON にするか
if [ "$IS_DOMAIN_DIVIDED" = "true" ]; then
    IS_DOMAIN_DIVIDED_FLAG="--is_domain_divided"
fi

# --use_domain_cls_loss を ON にするか
if [ "$USE_DOMAIN_CLS_LOSS" = "true" ]; then
    USE_DOMAIN_CLS_LOSS_FLAG="--use_domain_cls_loss"
fi

# --use_nearest_neighbor_loss を ON にするか
if [ "$USE_NEAREST_NEIGHBOR_LOSS" = "true" ]; then
    USE_NEAREST_NEIGHBOR_LOSS_FLAG="--use_nearest_neighbor_loss"
fi

DIR=/nas/data/gotoyuta/Result_Domain_Forgetting/${DATASET}/${TRAINER}/FORGET_DOMAIN${DOMAIN_COUNT}/${DOMAIN_SEC}/${CFG}/nctx-vision${NCTX}_prmpt-depth${DEPTH_VISION}_shots${SHOTS}_nnl-${USE_NEAREST_NEIGHBOR_LOSS}_dclsl-${USE_DOMAIN_CLS_LOSS}_divided-${IS_DOMAIN_DIVIDED}/seed${SEED}/${TODAY}
CSV_FILE_PATH=/nas/data/gotoyuta/Result_Domain_Forgetting/${DATASET}/${TRAINER}/FORGET_DOMAIN${DOMAIN_COUNT}/${CFG}_nctx${NCTX}_prmpt-depth${DEPTH_VISION}__shots${SHOTS}_nnl-${USE_NEAREST_NEIGHBOR_LOSS}_dclsl-${USE_DOMAIN_CLS_LOSS}_divided-${IS_DOMAIN_DIVIDED}_seed${SEED}.csv
if [ -d "$DIR" ]; then
    echo "Results are available in ${DIR}."
else
    echo "Run this job and save the output to ${DIR}"

    python train.py \
    --root ${DATA} \
    --seed ${SEED} \
    --trainer ${TRAINER} \
    --dataset-config-file configs/datasets/${DATASET}.yaml \
    --config-file configs/trainers/${TRAINER}/${CFG}.yaml \
    --forget_domains "${DOMAIN_LIST[@]}" \
    --output-dir ${DIR} \
    --num_shots ${SHOTS} \
    ${IS_DOMAIN_DIVIDED_FLAG} \
    ${USE_DOMAIN_CLS_LOSS_FLAG} \
    ${USE_NEAREST_NEIGHBOR_LOSS_FLAG} \
    --csv_file_path ${CSV_FILE_PATH} \
    TRAINER.${TRAINER}.PROMPT_DEPTH_VISION ${DEPTH_VISION} \
    TRAINER.${TRAINER}.N_CTX_VISION ${NCTX} \
    # TRAINER.${TRAINER}. ${NCTX} \
fi