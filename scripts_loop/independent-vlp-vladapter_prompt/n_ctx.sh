for n_ctx in 4 2 1 16;do
    bash domain_powerset_fullab_data.sh $1 office_home_df 1 vit_b16_ep50 ${n_ctx} 9 0 16 FullAblation 1
done