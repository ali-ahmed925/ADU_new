#!/bin/bash
cd ../../
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
  bash scripts/vpt/main.sh 1 pacs_df 1 vit_b16_ep50 8 9 $selected_domains_str
  # echo "${selected_domains[@]}"
done
domains=("art" "clipart" "product" "real_world")
for ((i = 3; i < 15; i++)); do
  # バイナリ数として各組み合わせを選択
  selected_domains=()
  for ((j = 0; j < 4; j++)); do
    if ((i & (1 << j))); then
      selected_domains+=("${domains[j]}")
    fi
  done
  # コマンドを実行
  selected_domains_str=$(IFS=" "; echo "${selected_domains[*]}")
  bash scripts/vpt/main.sh 1 office_home_df 1 vit_b16_ep50 8 9 $selected_domains_str
  # echo "${selected_domains[@]}"
done