#!/bin/bash
export CUDA_VISIBLE_DEVICES=$1
#cd ../..

# custom config
DATA="/nas/data/gotoyuta/Dataset/"
TRAINER=IVLP_VL_Adapter

DATASET=$2
SEED=$3

CFG=$4 # vit_b16_ep50
NCTX=$5
DEPTH_VISION=$6
DEPTH_TEXT=$7
USE_VISION_ADAPTER=${8}
USE_TEXT_ADAPTER=${9}
SHOTS=${10}
EXPNAME=${11}
DATASETSEED=${12}
# SUBEXPNAME=DC-${USE_DOMAIN_CLS_LOSS}_NN-${USE_NEAREST_NEIGHBOR_LOSS}_DIV-${IS_DOMAIN_DIVIDED}_InstPG-${USE_CROSSATTENTION}
SUBEXPNAME=VPrompt-${DEPTH_VISION}_TPrompt${DEPTH_TEXT}-VAdapter${USE_VISION_ADAPTER}-TAdapter${USE_TEXT_ADAPTER}
DIR=/nas/data/gotoyuta/Result_Domain_Forgetting_Loop/${DATASET}/${TRAINER}/SHOTS${SHOTS}/${CFG}/
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
    USE_VISION_ADAPTER ${USE_VISION_ADAPTER}\
    USE_TEXT_ADAPTER ${USE_TEXT_ADAPTER} \
    TRAINER.IVLP.PROMPT_DEPTH_VISION ${DEPTH_VISION} \
    TRAINER.IVLP.N_CTX_VISION ${NCTX} \
    TRAINER.IVLP.PROMPT_DEPTH_TEXT ${DEPTH_TEXT} \
    TRAINER.IVLP.N_CTX_TEXT ${NCTX} \

