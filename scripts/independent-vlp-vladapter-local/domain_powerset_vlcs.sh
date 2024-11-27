#!/bin/bash
cd ../../
gpu_id=$1
vision_depth=$2
text_depth=$3
use_domain_cls_loss=$4
use_nearest_neighbor_loss=$5
is_divided=$6
seed=$7
bash scripts/independent-vlp-vladapter/domain_forgetting.sh $gpu_id visda17_df $seed vit_b16_ep50 8 $vision_depth $text_depth $use_domain_cls_loss $use_nearest_neighbor_loss $is_divided synthetic

bash scripts/independent-vlp-vladapter/domain_forgetting.sh $gpu_id visda17_df $seed vit_b16_ep50 8 $vision_depth $text_depth $use_domain_cls_loss $use_nearest_neighbor_loss $is_divided real

# domains=("caltech" "labelme" "pascal" "sun")
# for ((i = 1; i < 15; i++)); do
#   # バイナリ数として各組み合わせを選択
#   selected_domains=()
#   for ((j = 0; j < 4; j++)); do
#     if ((i & (1 << j))); then
#       selected_domains+=("${domains[j]}")
#     fi
#   done
#   # コマンドを実行
#   selected_domains_str=$(IFS=" "; echo "${selected_domains[*]}")
#   bash scripts/independent-vlp-vladapter/domain_forgetting.sh $gpu_id vlcs_df $seed vit_b16_ep50 8 $vision_depth $text_depth $use_domain_cls_loss $use_nearest_neighbor_loss $is_divided $selected_domains_str
#   # echo "${selected_domains[@]}"
# done