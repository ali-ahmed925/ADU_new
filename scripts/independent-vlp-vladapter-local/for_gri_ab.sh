gpu_id=$1
masked_cls=$2
masked_nn=$3
experiment=$4
topk=180

for grid in 8 18 36 48 ;do
    bash domain_powerset.sh $gpu_id 9 1 true true true $masked_cls $masked_nn false true $topk $experiment $grid 1
done