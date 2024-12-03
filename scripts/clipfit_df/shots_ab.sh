gpu_id=$1
seed=$2
for shots in 2 4 8 32;do
    bash domain_powerset.sh $gpu_id $seed $shots
done