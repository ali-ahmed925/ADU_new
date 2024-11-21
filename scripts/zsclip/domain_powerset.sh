#!/bin/bash
cd ../../
gpu_id=$1

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
  bash scripts/zsclip/zeroshot.sh office_home_df vit_b16 $gpu_id $selected_domains_str
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
  bash scripts/zsclip/zeroshot.sh pacs_df vit_b16 $gpu_id $selected_domains_str
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
  bash scripts/zsclip/zeroshot.sh domainnet_mini_df vit_b16 $gpu_id $selected_domains_str
  # echo "${selected_domains[@]}"
done
