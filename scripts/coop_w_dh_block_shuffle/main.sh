#!/bin/bash

#cd ../..
export CUDA_VISIBLE_DEVICES=$1
# custom config
DATA="/nas/data/gotoyuta/Dataset/"
TRAINER=CoOp_w_DH

DATASET=$2 # ex.) office_home_df
CFG=$3  # config file
CTP=$4  # class token position (end or middle or front)
NCTX=$5  # number of context tokens
# SHOTS=$5  # number of shots (1, 2, 4, 8, 16)
CSC=$6 # class-specific context (False or True)

# ROOT_DIR=$7 # 
GRID_NUM=$7
# 7番目以降の引数をアンダースコアでつなげる
DOMAIN_LIST=("${@:8}")
# DOMAIN_LIST=${@:7:($# - 6)}
DOMAIN_SEC=$(IFS=-; echo "${DOMAIN_LIST[*]}")

# DOMAIN_LIST の要素数をカウント
# DOMAIN_COUNT=$(echo "$DOMAIN_LIST" | wc -w)
DOMAIN_COUNT=${#DOMAIN_LIST[@]}
TODAY=$(date +"%Y%m%d_%H%M%S")

for SEED in 1
do
    DIR=/nas/data/gotoyuta/Result_Domain_Forgetting/${DATASET}/${TRAINER}/FORGET_DOMAIN${DOMAIN_COUNT}/${DOMAIN_SEC}/${CFG}/nctx${NCTX}_csc${CSC}_ctp${CTP}_gridnum${GRID_NUM}/seed${SEED}/${TODAY}
    # DIR=./test/${TODAY}
    if [ -d "$DIR" ]; then
        echo "Results are available in ${DIR}. Skip this job"
    else
        echo "Run this job and save the output to ${DIR}"
        python train.py \
        --root ${DATA} \
        --seed ${SEED} \
        --trainer ${TRAINER} \
        --config-file configs/trainers/${TRAINER}/${CFG}.yaml \
        --dataset-config-file configs/datasets/${DATASET}.yaml \
        --forget_domains "${DOMAIN_LIST[@]}" \
        --is_block_shuffle \
        --grid_num $GRID_NUM \
        --output-dir ${DIR} \
        TRAINER.COOP.N_CTX ${NCTX} \
        TRAINER.COOP.CSC ${CSC} \
        TRAINER.COOP.CLASS_TOKEN_POSITION ${CTP} \
        # DATASET.NUM_SHOTS ${SHOTS}
    fi
done