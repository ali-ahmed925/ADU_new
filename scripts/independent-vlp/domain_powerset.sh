#!/bin/bash
cd ../../
gpu_id=$1
vision_depth=$2
text_depth=$3

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
  bash scripts/independent-vlp/domain_forgetting.sh $gpu_id office_home_df 0 vit_b16_ep50 8 $vision_depth $text_depth $selected_domains_str
  # echo "${selected_domains[@]}"
done

domains=("cartoon" "art_painting" "sketch" "photo")
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
  bash scripts/independent-vlp/domain_forgetting.sh $gpu_id pacs_df 0 vit_b16_ep50 8 $vision_depth $text_depth $selected_domains_str
  # echo "${selected_domains[@]}"
done