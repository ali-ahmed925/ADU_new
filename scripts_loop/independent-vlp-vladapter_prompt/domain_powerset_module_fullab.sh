#!/bin/bash
cd ../../
# 他の引数の設定（例: $1-$7, $11-$12 は適宜設定）
CUDA_DEVICE=$1
DATASET=$2
SEED=$3
CFG=$4
NCTX=$5
SHOTS=${6}
EXPNAME=${7}
DATASETSEED=${8}
IS_DOMAIN_DIVIDED=${9} # true false
USE_CROSSATTENTION=True
# true/false のフルアブレーション


for DEPTH_TEXT in 0 1;do 
    for DEPTH_VISION in 9;do
        for USE_VISION_ADAPTER in False True;do
            for USE_TEXT_ADAPTER in False True;do 
                if [ "$DEPTH_TEXT" -eq 0 ] && [ "$DEPTH_VISION" -eq 0 ] && [ "$USE_VISION_ADAPTER" = "False" ] && [ "$USE_TEXT_ADAPTER" = "False" ]; then
                    continue
                fi
                if [ "$DEPTH_TEXT" -eq 1 ] && [ "$DEPTH_VISION" -eq 9 ] && [ "$USE_VISION_ADAPTER" = "True" ] && [ "$USE_TEXT_ADAPTER" = "True" ]; then
                    continue
                fi
                bash scripts_loop/independent-vlp-vladapter_prompt/domain_forgetting_module_fullab.sh $CUDA_DEVICE $DATASET $SEED $CFG $NCTX $DEPTH_VISION $DEPTH_TEXT \
                    true true $IS_DOMAIN_DIVIDED \
                    $SHOTS $EXPNAME $USE_CROSSATTENTION $DATASETSEED $USE_VISION_ADAPTER $USE_VISION_ADAPTER
            done
        done
    done
done


# for USE_DOMAIN_CLS_LOSS in true; do
#     for USE_NEAREST_NEIGHBOR_LOSS in true; do
#         for IS_DOMAIN_DIVIDED in true; do
#             for USE_CROSSATTENTION in true; do
#             for USE_VISION_ADAPTER
#                 # サブエクスペリメント名の設定
#                 SUBEXPNAME=DC-${USE_DOMAIN_CLS_LOSS}_NN-${USE_NEAREST_NEIGHBOR_LOSS}_DIV-${IS_DOMAIN_DIVIDED}_InstPG${USE_CROSSATTENTION}

#                 # 実行コマンド
#                 bash scripts_loop/independent-vlp-vladapter_prompt/domain_forgetting_fullab.sh $CUDA_DEVICE $DATASET $SEED $CFG $NCTX $DEPTH_VISION $DEPTH_TEXT \
#                     $USE_DOMAIN_CLS_LOSS $USE_NEAREST_NEIGHBOR_LOSS $IS_DOMAIN_DIVIDED \
#                     $SHOTS $EXPNAME $USE_CROSSATTENTION $DATASETSEED False False
#             done
#         done
#     done
# done
