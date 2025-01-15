cd ../../
gpu_id=$1
shots=16
vision_depth=9
text_depth=1
use_domain_cls_loss=true
use_nearest_neighbor_loss=true
is_domain_divided=false
expname=${2}
subexpname=Insert-Vision-Layer
DATASET=${3}
DATASETSEED=${4}

for layer in 7 8 10 11 12;do
vision_depth=$layer
bash scripts_loop/independent-vlp-vladapter_prompt/domain_forgetting_SingleCross.sh $gpu_id $DATASET 1 vit_b16_ep50 8 $vision_depth $text_depth $use_domain_cls_loss $use_nearest_neighbor_loss $is_domain_divided ${shots} ${expname} ${subexpname}${layer} ${DATASETSEED}
done