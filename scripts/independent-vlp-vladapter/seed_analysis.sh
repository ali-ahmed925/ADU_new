gpu_id=$1
seed=$2

# bash ./domain_powerset.sh $gpu_id 9 1 true true true $seed 16
# bash ./domain_powerset.sh $gpu_id 9 1 false true true $seed 16
bash ./domain_powerset.sh $gpu_id 9 1 true false true $seed 16