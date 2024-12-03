#!/bin/bash
export CUDA_VISIBLE_DEVICES=$1
#cd ../..

# custom config
DATA="/nas/data/gotoyuta/Dataset/"
TRAINER=IVLP_VL_Adapter

DATASET=office_home_df
SEED=1

CFG=vit_b16_ep50 # vit_b16_ep50
NCTX=8
DEPTH_VISION=9
DEPTH_TEXT=1
DIR=$2
DOMAIN_LIST=("${@:3}")

DOMAIN_SEC=$(IFS=-; echo "${DOMAIN_LIST[*]}")
DOMAIN_COUNT=${#DOMAIN_LIST[@]}
TODAY=$(date +"%Y%m%d_%H%M%S")
# SHOTS=16


# DIR=output/${DATASET}/${TRAINER}/${CFG}/seed${SEED}
# DIR=/nas/data/gotoyuta/Result_Domain_Forgetting/${DATASET}/${TRAINER}/FORGET_DOMAIN${DOMAIN_COUNT}/${DOMAIN_SEC}/${CFG}/nctx${NCTX}_prmpt-depth${DEPTH_VISION}_prtmp-txt${DEPTH_TEXT}/seed${SEED}/${TODAY}

echo "Run this job and save the output to ${DIR}"
python train.py \
    --root ${DATA} \
    --seed ${SEED} \
    --trainer ${TRAINER} \
    --dataset-config-file configs/datasets/${DATASET}.yaml \
    --config-file configs/trainers/IVLP/${CFG}.yaml \
    --forget_domains "${DOMAIN_LIST[@]}" \
    --output-dir ${DIR} \
    --eval-only \
    --load-epoch 50\
    --model-dir ${DIR} \
    --use_domain_cls_loss \
    --use_nearest_neighbor_loss \
    --is_domain_divided \
    TRAINER.IVLP.PROMPT_DEPTH_VISION ${DEPTH_VISION} \
    TRAINER.IVLP.N_CTX_VISION ${NCTX} \
    TRAINER.IVLP.PROMPT_DEPTH_TEXT ${DEPTH_TEXT} \
    TRAINER.IVLP.N_CTX_TEXT ${NCTX} \
# DATASET.NUM_SHOTS ${SHOTS} \
# DATASET.SUBSAMPLE_CLASSES base
