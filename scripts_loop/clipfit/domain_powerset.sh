gpuid=${1}
# datasetname=${2}
expname=Baseline

cd ../../

bash scripts_loop/clipfit/domain_forgetting.sh $gpuid imagenet_df 1 vit_b16_ep1 16 Baseline 1 
bash scripts_loop/clipfit/domain_forgetting.sh $gpuid office_home_df 1 vit_b16_ep1 16 Baseline 1 
bash scripts_loop/clipfit/domain_forgetting.sh $gpuid domainnet_mini_df 1 vit_b16_ep1 16 Baseline 1 
# bash scripts_loop/clipfit/domain_forgetting.sh $gpuid imagenet_df 1 vit_b16_ep50 16 Baseline 1 