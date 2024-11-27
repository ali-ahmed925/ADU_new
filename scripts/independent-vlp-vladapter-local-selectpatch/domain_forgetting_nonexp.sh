#!/bin/bash
export CUDA_VISIBLE_DEVICES=$1
#cd ../..

# custom config
DATA="/nas/data/gotoyuta/Dataset/"
TRAINER=IVLP_VL_Adapter_Local

DATASET=$2
SEED=$3

CFG=$4 # vit_b16_ep50
NCTX=$5
DEPTH_VISION=$6
DEPTH_TEXT=$7
USE_DOMAIN_CLS_LOSS=${8}
USE_NEAREST_NEIGHBOR_LOSS=${9}
IS_DOMAIN_DIVIDED=${10}
MASKED_CLS=${11}
MASKED_NN=${12}
ENTROPY_MASK=${13}
BLOCK_SHUFFLE_SELECTION=${14}
BLOCK_SHUFFLE_SELECTION_NON_EXP=${15}
TOPK=${16}
EXPERIMENT=${17}
GRID=${18}
DOMAIN_LIST=("${@:19}")

DOMAIN_SEC=$(IFS=-; echo "${DOMAIN_LIST[*]}")
DOMAIN_COUNT=${#DOMAIN_LIST[@]}
TODAY=$(date +"%Y%m%d_%H%M%S")
SHOTS=16


# フラグでCLIオプションを切り替え
IS_DOMAIN_DIVIDED_FLAG=""
USE_DOMAIN_CLS_LOSS_FLAG=""
USE_NEAREST_NEIGHBOR_LOSS_FLAG=""
MASKED_CLS_FLAG=""
MASKED_NN_FLAG=""
ENTROPY_MASK_FLAG=""
BLOCK_SHUFFLE_SELECTION_FLAG=""
BLOCK_SHUFFLE_SELECTION_NON_EXP_FLAG=""

# --is_domain_divided を ON にするか
if [ "$IS_DOMAIN_DIVIDED" = "true" ]; then
    IS_DOMAIN_DIVIDED_FLAG="--is_domain_divided"
fi

# --use_domain_cls_loss を ON にするか
if [ "$USE_DOMAIN_CLS_LOSS" = "true" ]; then
    USE_DOMAIN_CLS_LOSS_FLAG="--use_domain_cls_loss"
fi

# --use_nearest_neighbor_loss を ON にするか
if [ "$USE_NEAREST_NEIGHBOR_LOSS" = "true" ]; then
    USE_NEAREST_NEIGHBOR_LOSS_FLAG="--use_nearest_neighbor_loss"
fi

if [ "$MASKED_CLS" = "true" ]; then
    MASKED_CLS_FLAG="--masked_dc"
fi

if [ "$MASKED_NN" = "true" ]; then
    MASKED_NN_FLAG="--masked_nn"
fi

if [ "$ENTROPY_MASK" = "true" ];then
    ENTROPY_MASK_FLAG="--entropy_mask"
fi

if [ "$BLOCK_SHUFFLE_SELECTION" = "true" ];then
    BLOCK_SHUFFLE_SELECTION_FLAG="--block_shuffle_selection"
fi

if [ "$BLOCK_SHUFFLE_SELECTION_NON_EXP" = "true" ];then
    BLOCK_SHUFFLE_SELECTION_NON_EXP_FLAG="--block_shuffle_selection_nonexp"
fi

DIR=/nas/data/gotoyuta/Result_Domain_Forgetting/${DATASET}/${TRAINER}/${EXPERIMENT}/FORGET_DOMAIN${DOMAIN_COUNT}/${DOMAIN_SEC}/${CFG}/nctx${NCTX}_prmpt-depth${DEPTH_VISION}_prtmp-txt${DEPTH_TEXT}_shots${SHOTS}_nnl-${USE_NEAREST_NEIGHBOR_LOSS}_masked-${MASKED_NN}_dclsl-${USE_DOMAIN_CLS_LOSS}_masked-${MASKED_CLS}_divided-${IS_DOMAIN_DIVIDED}_topk-${TOPK}_grid${GRID}/seed${SEED}/${TODAY}
CSV_FILE_PATH=/nas/data/gotoyuta/Result_Domain_Forgetting/${DATASET}/${TRAINER}/${EXPERIMENT}/FORGET_DOMAIN${DOMAIN_COUNT}/${CFG}_nctx${NCTX}_prmpt-depth${DEPTH_VISION}_prtmp-txt${DEPTH_TEXT}_shots${SHOTS}_nnl-${USE_NEAREST_NEIGHBOR_LOSS}_masked-${MASKED_NN}_dclsl-${USE_DOMAIN_CLS_LOSS}-masked-${MASKED_CLS}_divided-${IS_DOMAIN_DIVIDED}_topk-${TOPK}_grid${GRID}_seed${SEED}.csv

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
    --config-file configs/trainers/IVLP/${CFG}.yaml \
    --forget_domains "${DOMAIN_LIST[@]}" \
    --output-dir ${DIR} \
    --num_shots ${SHOTS} \
    --topk ${TOPK} \
    --grid_num ${GRID} \
    ${IS_DOMAIN_DIVIDED_FLAG} \
    ${USE_DOMAIN_CLS_LOSS_FLAG} \
    ${USE_NEAREST_NEIGHBOR_LOSS_FLAG} \
    ${MASKED_CLS_FLAG} \
    ${MASKED_NN_FLAG} \
    ${ENTROPY_MASK_FLAG} \
    ${BLOCK_SHUFFLE_SELECTION_FLAG} \
    ${BLOCK_SHUFFLE_SELECTION_NON_EXP_FLAG} \
    --csv_file_path ${CSV_FILE_PATH} \
    TRAINER.IVLP.PROMPT_DEPTH_VISION ${DEPTH_VISION} \
    TRAINER.IVLP.N_CTX_VISION ${NCTX} \
    TRAINER.IVLP.PROMPT_DEPTH_TEXT ${DEPTH_TEXT} \
    TRAINER.IVLP.N_CTX_TEXT ${NCTX} \

fi
