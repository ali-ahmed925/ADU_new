#!/bin/bash
export CUDA_VISIBLE_DEVICES=$1
#cd ../..

# custom config
DATA="/nas/data/gotoyuta/Dataset/"
TRAINER=IVLP_VL_Adapter_Prompt

DATASET=$2
SEED=$3

CFG=$4 # vit_b16_ep50
NCTX=$5
DEPTH_VISION=$6
DEPTH_TEXT=$7
USE_DOMAIN_CLS_LOSS=${8}
USE_NEAREST_NEIGHBOR_LOSS=${9}
IS_DOMAIN_DIVIDED=${10}
SHOTS=${11}
EXPNAME=${12}
USE_CROSSATTENTION=${13}
DATASETSEED=${14}
SUBEXPNAME=DC-${USE_DOMAIN_CLS_LOSS}_NN-${USE_NEAREST_NEIGHBOR_LOSS}_DIV-${IS_DOMAIN_DIVIDED}_InstPG-${USE_CROSSATTENTION}



# フラグでCLIオプションを切り替え
IS_DOMAIN_DIVIDED_FLAG=""
USE_DOMAIN_CLS_LOSS_FLAG=""
USE_NEAREST_NEIGHBOR_LOSS_FLAG=""
USE_CROSSATTENTION_FLAG=False

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

if [ "$USE_CROSSATTENTION" = "true" ]; then
    USE_CROSSATTENTION_FLAG="True"
fi

DIR=/nas/data/gotoyuta/Result_Domain_Forgetting_Loop/${DATASET}/${TRAINER}/SHOTS${SHOTS}/${CFG}/nctx${NCTX}_vision-depth${DEPTH_VISION}_text-depth${DEPTH_TEXT}
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
    ${IS_DOMAIN_DIVIDED_FLAG} \
    ${USE_DOMAIN_CLS_LOSS_FLAG} \
    ${USE_NEAREST_NEIGHBOR_LOSS_FLAG} \
    TRAINER.IVLP.PROMPT_DEPTH_VISION ${DEPTH_VISION} \
    TRAINER.IVLP.N_CTX_VISION ${NCTX} \
    TRAINER.IVLP.PROMPT_DEPTH_TEXT ${DEPTH_TEXT} \
    TRAINER.IVLP.N_CTX_TEXT ${NCTX} \
    USE_CROSSATTENTION ${USE_CROSSATTENTION_FLAG} \
    INSERT_LAYER_ATTN ${DEPTH_VISION}

