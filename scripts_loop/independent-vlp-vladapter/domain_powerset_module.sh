#!/bin/bash
cd ../../
# 他の引数の設定（例: $1-$7, $11-$12 は適宜設定）
CUDA_DEVICE=$1
DATASET=$2
SEED=$3
CFG=$4
NCTX=$5
# DEPTH_VISION=$6
# DEPTH_TEXT=$7
SHOTS=${6}
EXPNAME=${7}
DATASETSEED=${8}

for DEPTH_TEXT in 0 1;do
    for DEPTH_VISION in 0 9;do 
        for USE_VISION_ADAPTER in False True;do
            for USE_TEXT_ADAPTER in False True;do
                if [ "$DEPTH_TEXT" -eq 0 ] && [ "$DEPTH_VISION" -eq 0 ] && [ "$USE_VISION_ADAPTER" = "False" ] && [ "$USE_TEXT_ADAPTER" = "False" ]; then
                    continue
                fi
                bash scripts_loop/independent-vlp-vladapter/domain_forgetting_module_ab.sh $CUDA_DEVICE $DATASET $SEED $CFG $NCTX \
                $DEPTH_VISION $DEPTH_TEXT $USE_VISION_ADAPTER $USE_TEXT_ADAPTER $SHOTS $EXPNAME $DATASETSEED
            done
        done
    done
done

# true/false のフルアブレーション
# for USE_DOMAIN_CLS_LOSS in false true; do
#     for USE_NEAREST_NEIGHBOR_LOSS in false true; do
#         for IS_DOMAIN_DIVIDED in false true; do
#             for USE_CROSSATTENTION in false true; do
#                 # サブエクスペリメント名の設定
#                 SUBEXPNAME=DC-${USE_DOMAIN_CLS_LOSS}_NN-${USE_NEAREST_NEIGHBOR_LOSS}_DIV-${IS_DOMAIN_DIVIDED}_InstPG${USE_CROSSATTENTION}

#                 # 実行コマンド
#                 bash scripts_loop/independent-vlp-vladapter_prompt/domain_forgetting_fullab.sh $CUDA_DEVICE $DATASET $SEED $CFG $NCTX $DEPTH_VISION $DEPTH_TEXT \
#                     $USE_DOMAIN_CLS_LOSS $USE_NEAREST_NEIGHBOR_LOSS $IS_DOMAIN_DIVIDED \
#                     $SHOTS $EXPNAME $USE_CROSSATTENTION $DATASETSEED
#             done
#         done
#     done
# done
