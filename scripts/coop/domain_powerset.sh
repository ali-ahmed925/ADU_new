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
  # bash scripts/coop_with_adapter/main.sh $gpu_id office_home_df vit_b16_ep50 end 8 False $selected_domains_str
  bash scripts/coop/main.sh $gpu_id office_home_df 1 vit_b16_ep50 end 8 False $selected_domains_str
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
#   # bash scripts/coop_with_adapter/main.sh $gpu_id pacs_df vit_b16_ep50 end 8 False $selected_domains_str
#   bash scripts/coop/main.sh $gpu_id pacs_df 1 vit_b16_ep50 8 9 $selected_domains_str
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
  # bash scripts/coop_with_adapter/main.sh $gpu_id domainnet_mini_df vit_b16_ep50 end 8 False $selected_domains_str
  bash scripts/coop/main.sh $gpu_id domainnet_mini_df 1 vit_b16_ep50 end 8 False $selected_domains_str
  # bash scripts/vpt_with_dc/main.sh 0 domainnet_mini_df 1 vit_b16_ep50 8 9 $selected_domains_str
  # echo "${selected_domains[@]}"
done

bash scripts/coop/main.sh $gpu_id visda17_df 1 vit_b16_ep50 end 8 False synthetic
bash scripts/coop/main.sh $gpu_id visda17_df 1 vit_b16_ep50 end 8 False real


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
#   bash scripts/coop/main.sh 3 pacs_df 1 vit_b16_ep50 8 9 $selected_domains_str
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
#   bash scripts/coop/main.sh 3 office_home_df 1 vit_b16_ep50 8 9 $selected_domains_str
#   # echo "${selected_domains[@]}"
# done