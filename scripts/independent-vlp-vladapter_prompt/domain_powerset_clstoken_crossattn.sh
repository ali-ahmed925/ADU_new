#!/bin/bash
cd ../../
gpu_id=$1
vision_depth=$2
text_depth=$3
use_domain_cls_loss=$4
use_nearest_neighbor_loss=$5
is_divided=$6
seed=$7
shots=$8

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
  bash scripts/independent-vlp-vladapter_prompt/domain_forgetting_clstoken_crossattn.sh $gpu_id office_home_df $seed vit_b16_ep50 8 $vision_depth $text_depth $use_domain_cls_loss $use_nearest_neighbor_loss $is_divided $shots $selected_domains_str
  # echo "${selected_domains[@]}"
done

# domains=("cartoon" "art_painting" "sketch" "photo")
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
#   bash scripts/independent-vlp-vladapter/domain_forgetting.sh $gpu_id pacs_df $seed vit_b16_ep50 8 $vision_depth $text_depth $use_domain_cls_loss $use_nearest_neighbor_loss $is_divided $shots $selected_domains_str
#   # echo "${selected_domains[@]}"
# done

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
  bash scripts/independent-vlp-vladapter_prompt/domain_forgetting_clstoken_crossattn.sh $gpu_id domainnet_mini_df $seed vit_b16_ep50 8 $vision_depth $text_depth $use_domain_cls_loss $use_nearest_neighbor_loss $is_divided $shots $selected_domains_str
  # bash scripts/vpt_with_dc/main.sh 0 domainnet_mini_df 1 vit_b16_ep50 8 9 $selected_domains_str
  # echo "${selected_domains[@]}"
done

bash scripts/independent-vlp-vladapter_prompt/domain_forgetting_clstoken_crossattn.sh $gpu_id visda17_df $seed vit_b16_ep50 8 $vision_depth $text_depth $use_domain_cls_loss $use_nearest_neighbor_loss $is_divided $shots synthetic
bash scripts/independent-vlp-vladapter_prompt/domain_forgetting_clstoken_crossattn.sh $gpu_id visda17_df $seed vit_b16_ep50 8 $vision_depth $text_depth $use_domain_cls_loss $use_nearest_neighbor_loss $is_divided $shots real

# bash scripts/independent-vlp-vladapter-local-selectpatch/domain_forgetting_grid.sh $gpu_id visda17_df $seed vit_b16_ep50 8 $vision_depth $text_depth $use_domain_cls_loss $use_nearest_neighbor_loss false $masked_cls $maksed_nn $entropy_mask $block_shuffle_selection $topk $experiment $only_masked $grid synthetic

# bash scripts/independent-vlp-vladapter-local-selectpatch/domain_forgetting_grid.sh $gpu_id visda17_df $seed vit_b16_ep50 8 $vision_depth $text_depth $use_domain_cls_loss $use_nearest_neighbor_loss false $masked_cls $maksed_nn $entropy_mask $block_shuffle_selection $topk $experiment $only_masked $grid real