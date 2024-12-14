gpu_id=$1
topk=$2
grid=${3}

bash ./domain_powerset.sh $gpu_id 9 1 true true true 1 16 $topk $grid