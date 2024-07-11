#!/bin/bash
export CUDA_VISIBLE_DEVICES=3
#cd ../..

# custom config
DATA="/nas/data/gotoyuta/Dataset/"
TRAINER=MaPLe

DATASET=$1
SEED=$2

CFG=vit_b16_c2_ep50_batch128_8ctx
SHOTS=16


DIR=output/${DATASET}/${TRAINER}/${CFG}/seed${SEED}
if [ -d "$DIR" ]; then
    echo "Results are available in ${DIR}. Resuming..."
    python train.py \
    --root ${DATA} \
    --seed ${SEED} \
    --trainer ${TRAINER} \
    --dataset-config-file configs/datasets/${DATASET}.yaml \
    --config-file configs/trainers/${TRAINER}/${CFG}.yaml \
    --output-dir ${DIR} \
    DATASET.NUM_SHOTS ${SHOTS} \
    DATASET.SUBSAMPLE_CLASSES base
else
    echo "Run this job and save the output to ${DIR}"
    python train.py \
    --root ${DATA} \
    --seed ${SEED} \
    --trainer ${TRAINER} \
    --dataset-config-file configs/datasets/${DATASET}.yaml \
    --config-file configs/trainers/${TRAINER}/${CFG}.yaml \
    --output-dir ${DIR} \
    # DATASET.NUM_SHOTS ${SHOTS} \
    # DATASET.SUBSAMPLE_CLASSES base
fi
