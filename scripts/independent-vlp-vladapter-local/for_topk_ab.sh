gpu_id=$1
topk=190

bash domain_powerset_cpy.sh $gpu_id 9 1 true true true false true true false $topk tin 0 1 1
bash domain_powerset_cpy.sh $gpu_id 9 1 true true true true false true false $topk tin 0 1 1
bash domain_powerset_cpy.sh $gpu_id 9 1 true true true true true true false $topk tin 0 1 1
