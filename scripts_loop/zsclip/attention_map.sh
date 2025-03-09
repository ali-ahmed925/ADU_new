#!/bin/bash
export CUDA_VISIBLE_DEVICES=$1
#cd ../..

# custom config
DATA="/nas/data/gotoyuta/Dataset/"
TRAINER=ZeroshotCLIP

DATASET=$2
SEED=$3

CFG=$4 # vit_b16_ep50
SHOTS=${5}
DATASETSEED=${6}
FORGET_DOMAIN=${7}
DIR=${8}

# CSV_FILE_PATH=/nas/data/gotoyuta/Result_Domain_Forgetting/${DATASET}/${TRAINER}/SHOTS${SHOTS}/FORGET_DOMAIN${DOMAIN_COUNT}/${CFG}_CROSS_ATTENTION_nctx${NCTX}_prmpt-depth${DEPTH_VISION}_prtmp-txt${DEPTH_TEXT}_shots${SHOTS}_nnl${USE_NEAREST_NEIGHBOR_LOSS}_dclsl${USE_DOMAIN_CLS_LOSS}_divided${IS_DOMAIN_DIVIDED}_seed${SEED}.csv


echo "Run this job and save the output to ${DIR}"
python main_attention_map.py \
    --root ${DATA} \
    --seed ${SEED} \
    --trainer ${TRAINER} \
    --dataset-config-file configs/datasets/${DATASET}.yaml \
    --config-file configs/trainers/ZeroshotCLIP_Local/${CFG}.yaml \
    --forget_domains $FORGET_DOMAIN \
    --output-dir ${DIR} \
    --num_shots ${SHOTS} \
    --dataset_name ${DATASET} \
    --dataset_seed ${DATASETSEED} \
    # USE_KLDIV_PENALTY ${KLDIV} \
    # ONLY_KLDIV_FOR_PRV ${USE_KL_DIV_ONLY_PRV_FRAG}


