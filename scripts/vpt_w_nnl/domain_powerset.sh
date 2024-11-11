#!/bin/bash
cd ../../
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
  bash scripts/vpt_w_nnl/main.sh 3 domainnet_mini_df 1 vit_b16_ep50 8 9 $selected_domains_str
  # bash scripts/vpt_with_dc/main.sh 0 domainnet_mini_df 1 vit_b16_ep50 8 9 $selected_domains_str
  # echo "${selected_domains[@]}"
done
# domains=("clipart" "infograph" "painting" "quickdraw" "real" "sketch")
# for ((i = 1; i < 64; i++)); do
#   # バイナリ数として各組み合わせを選択
#   selected_domains=()
#   for ((j = 0; j < 6; j++)); do
#     if ((i & (1 << j))); then
#       selected_domains+=("${domains[j]}")
#     fi
#   done
#   # コマンドを実行
#   selected_domains_str=$(IFS=" "; echo "${selected_domains[*]}")
#   bash scripts/vpt/main.sh 2 domainnet_df 1 vit_b16_ep50 8 9 $selected_domains_str
#   bash scripts/vpt_with_dc/main.sh 2 domainnet_df 1 vit_b16_ep50 8 9 $selected_domains_str
#   # bash scripts/vpt_with_dc/main.sh 1 domainnet_df 1 vit_b16_ep50 8 9 $selected_domains_str
#   # bash scripts/vpt_with_dc/main.sh 1 domainnet_df 1 vit_b16_ep50 8 9 $selected_domains_str
#   # echo "${selected_domains[@]}"
# done

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
#   bash scripts/vpt_w_nnl/main.sh 3 pacs_df 1 vit_b16_ep50 8 9 $selected_domains_str
#   # echo "${selected_domains[@]}"
# done

# domains=("art" "clipart" "product" "real_world")
# for ((i = 3; i < 15; i++)); do
#   # バイナリ数として各組み合わせを選択
#   selected_domains=()
#   for ((j = 0; j < 4; j++)); do
#     if ((i & (1 << j))); then
#       selected_domains+=("${domains[j]}")
#     fi
#   done
#   # コマンドを実行
#   selected_domains_str=$(IFS=" "; echo "${selected_domains[*]}")
#   bash scripts/vpt/main.sh 0 pacs_df 0 vit_b16_ep50 8 9 $selected_domains_str
#   # echo "${selected_domains[@]}"
# done
# domains=("art" "clipart" "product" "real_world")
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
#   bash scripts/vpt/main.sh 3 office_home_df 1 vit_b16_ep50 8 9 $selected_domains_str
#   # echo "${selected_domains[@]}"
# done