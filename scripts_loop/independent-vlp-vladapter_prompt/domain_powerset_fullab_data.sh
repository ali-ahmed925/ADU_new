#!/bin/bash
cd ../../
# 他の引数の設定（例: $1-$7, $11-$12 は適宜設定）
CUDA_DEVICE=$1
DATASET=$2
SEED=$3
CFG=$4
NCTX=$5
DEPTH_VISION=$6
DEPTH_TEXT=$7
SHOTS=${8}
EXPNAME=${9}
DATASETSEED=${10}

# true/false のフルアブレーション
for data in office_home_df domainnet_mini_df;do
for USE_DOMAIN_CLS_LOSS in true; do
    for USE_NEAREST_NEIGHBOR_LOSS in true; do
        for IS_DOMAIN_DIVIDED in true; do
            for USE_CROSSATTENTION in true; do
                # サブエクスペリメント名の設定
                SUBEXPNAME=DC-${USE_DOMAIN_CLS_LOSS}_NN-${USE_NEAREST_NEIGHBOR_LOSS}_DIV-${IS_DOMAIN_DIVIDED}_InstPG${USE_CROSSATTENTION}

                # 実行コマンド
                bash scripts_loop/independent-vlp-vladapter_prompt/domain_forgetting_fullab.sh $CUDA_DEVICE $DATASET $SEED $CFG $NCTX $DEPTH_VISION $DEPTH_TEXT \
                    $USE_DOMAIN_CLS_LOSS $USE_NEAREST_NEIGHBOR_LOSS $IS_DOMAIN_DIVIDED \
                    $SHOTS $EXPNAME $USE_CROSSATTENTION $DATASETSEED False False
            done
        done
    done
done
done
