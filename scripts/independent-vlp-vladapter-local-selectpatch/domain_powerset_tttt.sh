#!/bin/bash
cd ../../
gpu_id=$1
vision_depth=$2 # 9
text_depth=$3 # 1
use_domain_cls_loss=$4 # true
use_nearest_neighbor_loss=$5 # true
is_divided=$6 # true
masked_cls=$7 # true
maksed_nn=$8 # true
entropy_mask=$9 # true 
block_shuffle_selection=${10} # false
topk=${11} # 190
experiment=${12} 
seed=${13} # 1
only_masked=${14} # True
grid=${15} # default 8

# domains=("clipart" "infograph" "painting" "quickdraw" "real" "sketch")
# for ((i = 1; i < 63; i++)); do
#   # バイナリ数として各組み合わせを選択
#   selected_domains=()
#   for ((j = 0; j < 6; j++)); do
#     if ((i & (1 << j))); then
#       selected_domains+=("${domains[j]}")
#     fi
#   done
#   # コマンドを実行
#   selected_domains_str=$(IFS=" "; echo "${selected_domains[*]}")
#   bash scripts/independent-vlp-vladapter/domain_forgetting.sh $gpu_id domainnet_df 1 vit_b16_ep50 8 $vision_depth $text_depth $use_domain_cls_loss $use_nearest_neighbor_loss $is_divided $selected_domains_str
# done

domains=("art" "clipart" "product" "real_world")
for ((i = 1; i < 15; i++)); do
  # バイナリ数として各組み合わせを選択
  selected_domains=()
  for ((j = 0; j < 4; j++)); do
    if ((i & (1 << j))); then
      selected_domains+=("${domains[j]}")
    fi
  done
  # コマンドを実行
  selected_domains_str=$(IFS=" "; echo "${selected_domains[*]}")
  bash scripts/independent-vlp-vladapter-local-selectpatch/domain_forgetting_grid.sh $gpu_id office_home_df $seed vit_b16_ep50 8 $vision_depth $text_depth $use_domain_cls_loss $use_nearest_neighbor_loss $is_divided $masked_cls $maksed_nn $entropy_mask $block_shuffle_selection $topk $experiment $only_masked $grid $selected_domains_str
  # echo "${selected_domains[@]}"
done

domains=("clipart" "painting" "real" "sketch")
for ((i = 1; i < 15; i++)); do
  # バイナリ数として各組み合わせを選択
  selected_domains=()
  for ((j = 0; j < 4; j++)); do
    if ((i & (1 << j))); then
      selected_domains+=("${domains[j]}")
    fi
  done
  # コマンドを実行
  selected_domains_str=$(IFS=" "; echo "${selected_domains[*]}")
  bash scripts/independent-vlp-vladapter-local-selectpatch/domain_forgetting_grid.sh $gpu_id domainnet_mini_df $seed vit_b16_ep50 8 $vision_depth $text_depth $use_domain_cls_loss $use_nearest_neighbor_loss $is_divided $masked_cls $maksed_nn $entropy_mask $block_shuffle_selection $topk $experiment $only_masked $grid $selected_domains_str
  # echo "${selected_domains[@]}"
done

bash scripts/independent-vlp-vladapter-local-selectpatch/domain_forgetting_grid.sh $gpu_id visda17_df $seed vit_b16_ep50 8 $vision_depth $text_depth $use_domain_cls_loss $use_nearest_neighbor_loss false $masked_cls $maksed_nn $entropy_mask $block_shuffle_selection $topk $experiment $only_masked $grid synthetic

bash scripts/independent-vlp-vladapter-local-selectpatch/domain_forgetting_grid.sh $gpu_id visda17_df $seed vit_b16_ep50 8 $vision_depth $text_depth $use_domain_cls_loss $use_nearest_neighbor_loss false $masked_cls $maksed_nn $entropy_mask $block_shuffle_selection $topk $experiment $only_masked $grid real
