#!/bin/bash

# 引数の取得
CUDA_DEVICE=$1
DATASET=domainnet_df
SEED=1
CFG=vit_b16_ep50
NCTX=4
DEPTH_VISION=9
DEPTH_TEXT=9
SHOTS=8
EXPNAME=BBF
DATASETSEED=0
TRAINER=IVLP_VL_Adapter_Prompt

# 固定アダプターの有無（必要ならループで回してもOK）
USE_VISION_ADAPTER=False
USE_TEXT_ADAPTER=False

# true/false のフルアブレーション
for data in $DATASET ;do
for USE_DOMAIN_CLS_LOSS in true; do 
    for USE_NEAREST_NEIGHBOR_LOSS in false; do
        for IS_DOMAIN_DIVIDED in true; do
            for USE_CROSSATTENTION in true; do
            for DOMAIN_WEIGHT in 0.0 10.0 30.0 50.0 100.0; do
            for MMD in 0.0 10.0 20.0 30.0 50.0 100.0 ; do


                # 各フラグに対応するCLIオプションの設定
                IS_DOMAIN_DIVIDED_FLAG=""
                USE_DOMAIN_CLS_LOSS_FLAG=""
                USE_NEAREST_NEIGHBOR_LOSS_FAG=""
                USE_CROSSATTENTION_FLAG=False

                if [ "$IS_DOMAIN_DIVIDED" = "true" ]; then
                    IS_DOMAIN_DIVIDED_FLAG="--is_domain_divided"
                fi

                if [ "$USE_DOMAIN_CLS_LOSS" = "true" ]; then
                    USE_DOMAIN_CLS_LOSS_FLAG="--use_domain_cls_loss"
                fi

                if [ "$USE_NEAREST_NEIGHBOR_LOSS" = "true" ]; then
                    USE_NEAREST_NEIGHBOR_LOSS_FLAG="--use_nearest_neighbor_loss"
                fi

                if [ "$USE_CROSSATTENTION" = "true" ]; then
                    USE_CROSSATTENTION_FLAG="True"
                fi

                # サブ実験名
                SUBEXPNAME=MMD

                # 実行ディレクトリ
                DIR=/nas/data/kawamura/ADU/domainnet/domain_weight_${DOMAIN_WEIGHT}/mmd_weight_${MMD}/

                echo "Run this job and save the output to ${DIR}"

                # 実行コマンド
                CUDA_VISIBLE_DEVICES=${CUDA_DEVICE} python train_loop.py \
                    --root /nas/data/gotoyuta/Dataset/ \
                    --seed ${SEED} \
                    --trainer ${TRAINER}  \
                    --dataset-config-file configs/datasets/${data}.yaml \
                    --config-file configs/trainers/IVLP/${CFG}.yaml \
                    --forget_domains "${DOMAIN_LIST[@]}" \
                    --output-dir ${DIR} \
                    --num_shots ${SHOTS} \
                    --dataset_name ${data} \
                    --dataset_seed ${DATASETSEED} \
                    --experiment_name ${EXPNAME} \
                    --sub_experiment_name ${SUBEXPNAME} \
                    --mmd_weight ${MMD} \
                    --domainloss_weight ${DOMAIN_WEIGHT} \
                    ${IS_DOMAIN_DIVIDED_FLAG} \
                    ${USE_DOMAIN_CLS_LOSS_FLAG} \
                    ${USE_NEAREST_NEIGHBOR_LOSS_FLAG} \
                    TRAINER.IVLP.PROMPT_DEPTH_VISION ${DEPTH_VISION} \
                    TRAINER.IVLP.N_CTX_VISION ${NCTX} \
                    TRAINER.IVLP.PROMPT_DEPTH_TEXT ${DEPTH_TEXT} \
                    TRAINER.IVLP.N_CTX_TEXT ${NCTX} \
                    USE_CROSSATTENTION ${USE_CROSSATTENTION_FLAG} \
                    INSERT_LAYER_ATTN ${DEPTH_VISION} \
                    USE_TEXT_ADAPTER ${USE_TEXT_ADAPTER} \
                    USE_VISION_ADAPTER ${USE_VISION_ADAPTER}
            done
            done
            done
        done
    done
done
done