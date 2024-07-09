#!/bin/bash
export CUDA_VISIBLE_DEVICES=$1
#cd ../..

# custom config
DATA="/nas/data/gotoyuta/Dataset/"
TRAINER=VPT

DATASET=$2
SEED=$3
CFG=$4
# CFG=vit_b16_c2_ep50_batch128_8_depthvision1
# SHOTS=16

DIR=output/${DATASET}/${TRAINER}/${CFG}/seed${SEED}
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
    --output-dir ${DIR} \
    # TRAINER.${TRAINER}.PROMPT_DEPTH_VISION=${DEPTH_VISION} 
fi