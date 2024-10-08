#!/bin/bash
export CUDA_VISIBLE_DEVICES=$1
#cd ../..

# custom config
DATA="/nas/data/gotoyuta/Dataset/"
TRAINER=IVLP

DATASET=$2
SEED=$3

CFG=$4 # vit_b16_ep50
NCTX=$5
DEPTH_VISION=$6
DEPTH_TEXT=$7

DOMAIN_LIST=("${@:8}")

DOMAIN_SEC=$(IFS=-; echo "${DOMAIN_LIST[*]}")
DOMAIN_COUNT=${#DOMAIN_LIST[@]}
TODAY=$(date +"%Y%m%d_%H%M%S")
# SHOTS=16


# DIR=output/${DATASET}/${TRAINER}/${CFG}/seed${SEED}
DIR=/nas/data/gotoyuta/Result_Domain_Forgetting/${DATASET}/${TRAINER}/FORGET_DOMAIN${DOMAIN_COUNT}/${DOMAIN_SEC}/${CFG}/nctx${NCTX}_prmpt-depth${DEPTH_VISION}_prtmp-txt${DEPTH_TEXT}/seed${SEED}/${TODAY}
if [ -d "$DIR" ]; then
    echo "Results are available in ${DIR}. Resuming..."
    # python train.py \
    # --root ${DATA} \
    # --seed ${SEED} \
    # --trainer ${TRAINER} \
    # --dataset-config-file configs/datasets/${DATASET}.yaml \
    # --config-file configs/trainers/${TRAINER}/${CFG}.yaml \
    # --output-dir ${DIR} \
    # DATASET.NUM_SHOTS ${SHOTS} \
    # DATASET.SUBSAMPLE_CLASSES base
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
    TRAINER.${TRAINER}.PROMPT_DEPTH_VISION ${DEPTH_VISION} \
    TRAINER.${TRAINER}.N_CTX_VISION ${NCTX} \
    TRAINER.${TRAINER}.PROMPT_DEPTH_TEXT ${DEPTH_TEXT} \
    TRAINER.${TRAINER}.N_CTX_TEXT ${NCTX} \
    # DATASET.NUM_SHOTS ${SHOTS} \
    # DATASET.SUBSAMPLE_CLASSES base
fi