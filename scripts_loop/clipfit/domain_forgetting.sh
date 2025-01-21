#!/bin/bash
export CUDA_VISIBLE_DEVICES=$1
#cd ../..

# custom config
DATA="/nas/data/gotoyuta/Dataset/"
TRAINER=ClipFit_DF

DATASET=$2
SEED=$3

CFG=$4 # vit_b16_ep50
SHOTS=${5}
EXPNAME=${6}
DATASETSEED=${7}
SUBEXPNAME=ORG



# フラグでCLIオプションを切り替え
IS_DOMAIN_DIVIDED_FLAG=""
USE_DOMAIN_CLS_LOSS_FLAG=""
USE_NEAREST_NEIGHBOR_LOSS_FLAG=""
USE_CROSSATTENTION_FLAG=False

DIR=/nas/data/gotoyuta/Result_Domain_Forgetting_Loop/${DATASET}/${TRAINER}/SHOTS${SHOTS}/${CFG}
# CSV_FILE_PATH=/nas/data/gotoyuta/Result_Domain_Forgetting/${DATASET}/${TRAINER}/SHOTS${SHOTS}/FORGET_DOMAIN${DOMAIN_COUNT}/${CFG}_CROSS_ATTENTION_nctx${NCTX}_prmpt-depth${DEPTH_VISION}_prtmp-txt${DEPTH_TEXT}_shots${SHOTS}_nnl${USE_NEAREST_NEIGHBOR_LOSS}_dclsl${USE_DOMAIN_CLS_LOSS}_divided${IS_DOMAIN_DIVIDED}_seed${SEED}.csv


echo "Run this job and save the output to ${DIR}"
python train_loop.py \
    --root ${DATA} \
    --seed ${SEED} \
    --trainer ${TRAINER} \
    --dataset-config-file configs/datasets/${DATASET}.yaml \
    --config-file configs/trainers/IVLP/${CFG}.yaml \
    --forget_domains "${DOMAIN_LIST[@]}" \
    --output-dir ${DIR} \
    --num_shots ${SHOTS} \
    --dataset_name ${DATASET} \
    --dataset_seed ${DATASETSEED} \
    --experiment_name ${EXPNAME} \
    --sub_experiment_name ${SUBEXPNAME} \
