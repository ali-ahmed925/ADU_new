#!/bin/bash

#cd ../..
export CUDA_VISIBLE_DEVICES=3
# custom config
DATA="/nas/data/gotoyuta/Dataset/"
TRAINER=CoOpDomainSpecific

DATASET=$1 # office_home_df_domain.py
CFG=$2  # config file
CTP=$3  # class token position (end or middle or front)
NCTX=$4  # number of context tokens
# SHOTS=$5  # number of shots (1, 2, 4, 8, 16)
CSC=$5  # class-specific context (False or True)

for SEED in 1
do
    DIR=output/${DATASET}/${TRAINER}/${CFG}/nctx${NCTX}_csc${CSC}_ctp${CTP}/seed${SEED}
    if [ -d "$DIR" ]; then
        echo "Results are available in ${DIR}. Skip this job"
    else
        echo "Run this job and save the output to ${DIR}"
        python train.py \
        --root ${DATA} \
        --seed ${SEED} \
        --trainer ${TRAINER} \
        --dataset-config-file configs/datasets/${DATASET}.yaml \
        --config-file configs/trainers/${TRAINER}/${CFG}.yaml \
        --output-dir ${DIR} \
        TRAINER.COOP.N_CTX ${NCTX} \
        TRAINER.COOP.CSC ${CSC} \
        TRAINER.COOP.CLASS_TOKEN_POSITION ${CTP} \
        # DATASET.NUM_SHOTS ${SHOTS}
    fi
done