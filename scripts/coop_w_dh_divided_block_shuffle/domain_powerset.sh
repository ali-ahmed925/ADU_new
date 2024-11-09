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
  bash scripts/coop_w_dh_divided/main.sh 2 pacs_df vit_b16_ep50 end 8 False $selected_domains_str 
  bash scripts/coop_w_dh_divided_block_shuffled/main.sh 2 pacs_df vit_b16_ep50 end 8 False $selected_domains_str 
  bash scripts/coop_w_dh_block_shuffled/main.sh 2 pacs_df vit_b16_ep50 end 8 False $selected_domains_str 
  # echo "${selected_domains[@]}"
done
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
  bash scripts/coop_w_dh_divided/main.sh 2 office_home_df vit_b16_ep50 end 8 False $selected_domains_str 
  bash scripts/coop_w_dh_divided_block_shuffle/main.sh 2 office_home_df vit_b16_ep50 end 8 False $selected_domains_str 
  bash scripts/coop_w_dh_block_shuffle/main.sh 2 office_home_df vit_b16_ep50 end 8 False $selected_domains_str 
  # echo "${selected_domains[@]}"
done
