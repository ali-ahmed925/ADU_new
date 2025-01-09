cd ../../
gpu_id=$1
shots=16
vision_depth=9
text_depth=1
use_domain_cls_loss=true
use_nearest_neighbor_loss=true
is_domain_divided=true
expname=${2}
subexpname=Insert-Vision-Layer

for layer in 1 2 3 4 5 6 7 8 9 10 11 12;do
vision_depth=$layer
bash scripts_loop/independent-vlp-vladapter_prompt/domain_forgetting_wo_Cross.sh $gpu_id office_home_df 1 vit_b16_ep50 8 $vision_depth $text_depth $use_domain_cls_loss $use_nearest_neighbor_loss $is_domain_divided ${shots} ${expname} ${subexpname}${layer}
done