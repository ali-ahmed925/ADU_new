# gpuid shots
for pd in 1 2 3 4 5 6 7 8 9 10 12;do
bash ddl_shots.sh $1 imagenet_df 1 vit_b16_ep50 8 $pd 0 $2 FullAblation 1
done