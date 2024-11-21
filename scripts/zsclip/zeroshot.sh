#!/bin/bash

#cd ../..

# custom config
DATA="/nas/data/gotoyuta/Dataset/"
TRAINER=ZeroshotCLIP
DATASET=$1
CFG=$2  # rn50, rn101, vit_b32 or vit_b16
gpuid=$3
DOMAIN_LIST=("${@:4}")
DOMAIN_SEC=$(IFS=-; echo "${DOMAIN_LIST[*]}")
DOMAIN_COUNT=${#DOMAIN_LIST[@]}
TODAY=$(date +"%Y%m%d_%H%M%S")
SHOTS=16

DIR=/nas/data/gotoyuta/Result_Domain_Forgetting/${DATASET}/${TRAINER}/FORGET_DOMAIN${DOMAIN_COUNT}/${DOMAIN_SEC}/${TODAY}
CSV_FILE_PATH=/nas/data/gotoyuta/Result_Domain_Forgetting/${DATASET}/${TRAINER}/FORGET_DOMAIN${DOMAIN_COUNT}/zero-shot.csv

CUDA_VISIBLE_DEVICES=${gpuid} python train.py \
    --root ${DATA} \
    --trainer ${TRAINER} \
    --dataset-config-file configs/datasets/${DATASET}.yaml \
    --config-file configs/trainers/CoOp/${CFG}.yaml \
    --forget_domains "${DOMAIN_LIST[@]}" \
    --output-dir ${DIR} \
    --csv_file_path ${CSV_FILE_PATH} \
    --eval-only \
    