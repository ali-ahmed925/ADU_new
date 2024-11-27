gpu_id=$1
masked_cls=$2
masked_nn=$3
experiment=$4
topk=190

for grid in 36 ;do
    bash domain_powerset_nonexp.sh $gpu_id 9 1 true true true $masked_cls $masked_nn false false true $topk $experiment $grid 1
done