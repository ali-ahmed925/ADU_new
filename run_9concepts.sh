#!/usr/bin/env bash
# =============================================================================
# First pass at killing n=1: the 9 pre-registered concepts, ONE seed,
# our method + the NegGrad baseline. Config is identical to run E
# (50 epochs, batch 32, forget pool 100, full pool forwarded each step).
#
#   ./run_9concepts.sh            # ours + neggrad   (18 runs, ~9h)
#   METHODS=ours ./run_9concepts.sh   # ours only    (9 runs,  ~5.5h)
#
# Resumable: any cell whose checkpoint already exists is skipped.
# =============================================================================
set -u

PY=${PY:-python}
ROOT=${ROOT:-"/home/ai/machine unlearning/ebm_unlearning/data/domainnet"}
OUT=${OUT:-"$HOME/adu_results/n9"}
CFG=configs/trainers/vit_b16_ep50_bs32_concept.yaml
DCFG=configs/datasets/domainnet_mini_paper_df.yaml
EPOCH=50
SEED=${SEED:-1}
BS=${BS:-32}
CHUNK=${CHUNK:-0}          # 0 = forward the whole pool each step, as in run E
POOL=${POOL:-100}
FD=${FD:-sketch}
METHODS=${METHODS:-"ours neggrad"}

# pre-registered by select_concepts.py (seed 0, 3 per proximity tercile)
CONCEPTS=${CONCEPTS:-"dog fish squirrel skateboard flamingo spider vase leaf The_Eiffel_Tower"}

OURS="--forget_loss_type suppress_marg --forget_weight 1.0 --flat_weight 1.0 \
--marg_weight 1.0 --forget_pool_size $POOL --suppress_cap 6 --exclude_forget_class_from_retain"
NEGGRAD="--forget_loss_type neggrad --exclude_forget_class_from_retain"

neighbor_of () {
  case "$1" in
    dog) echo cat ;;           fish) echo bird ;;     squirrel) echo monkey ;;
    skateboard) echo guitar ;; flamingo) echo swan ;; spider) echo ant ;;
    vase) echo flower ;;       leaf) echo feather ;;  The_Eiffel_Tower) echo castle ;;
    tiger) echo lion ;;        *) echo lion ;;
  esac
}

mkdir -p "$OUT"
START=$(date +%s)
TOTAL=0; DONE=0
for c in $CONCEPTS; do for m in $METHODS; do TOTAL=$((TOTAL+1)); done; done
echo "=== $TOTAL cells | seed $SEED | out=$OUT ==="

for c in $CONCEPTS; do
  nb=$(neighbor_of "$c")
  for m in $METHODS; do
    DONE=$((DONE+1))
    run="$OUT/${m}__${c}__s${SEED}"
    case "$m" in ours) ARGS="$OURS" ;; neggrad) ARGS="$NEGGRAD" ;; *) echo "unknown $m"; exit 1 ;; esac

    # any saved checkpoint counts -- training may stop early, in which case the
    # file is model.pth.tar-<N> with N < EPOCH, not model.pth.tar-50.
    have_ckpt () { ls "$run"/seed*/ForgetDomain1/$FD/*/VLPromptLearner/model.pth.tar-* >/dev/null 2>&1; }

    if have_ckpt; then
      echo "[$DONE/$TOTAL] skip $m/$c (checkpoint exists)"
    else
      echo "[$DONE/$TOTAL] TRAIN $m / $c  ($(date +%H:%M))"
      chunk_arg=""; [ "$CHUNK" != "0" ] && chunk_arg="--forget_chunk $CHUNK"
      # record the exact invocation at the top of the log for reproducibility
      { echo "# $(date -Is)"; echo "# method=$m concept=$c seed=$SEED neighbor=$nb"; \
        echo "# args: $ARGS $chunk_arg BATCH_SIZE=$BS CHUNK=$CHUNK POOL=$POOL"; } \
        > "$run.train.log"
      CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
      $PY train_loop.py \
          --root "$ROOT" --trainer IVLP_VL_Adapter_Prompt \
          --dataset-config-file $DCFG --config-file $CFG \
          --output-dir "$run" \
          --num_shots 8 --dataset_name domainnet_mini_paper_df \
          --is_domain_divided --seeds "$SEED" \
          --run_forget_domains $FD --forget_classes "$c" \
          $ARGS $chunk_arg \
          DATALOADER.TRAIN_X.BATCH_SIZE $BS \
          >> "$run.train.log" 2>&1 \
        || { echo "  !! training FAILED, see $run.train.log"; continue; }
    fi

    # pick the highest-epoch checkpoint. Sort on the numeric suffix only, so the
    # result does not depend on hyphens elsewhere in the path. Note a run dir can
    # hold several timestamp folders if earlier attempts crashed; only ones that
    # actually saved a model are considered here.
    ckfile=$(ls "$run"/seed*/ForgetDomain1/$FD/*/VLPromptLearner/model.pth.tar-* 2>/dev/null \
             | awk -F 'model.pth.tar-' '{print $2"\t"$0}' | sort -n | tail -1 | cut -f2-)
    [ -z "$ckfile" ] && { echo "  !! NO CHECKPOINT for $m/$c -- skipping eval"; continue; }
    ck=$(dirname "$(dirname "$ckfile")")
    ep=${ckfile##*model.pth.tar-}
    [ "$ep" != "$EPOCH" ] && echo "  ** note: $m/$c stopped early, using epoch $ep"

    echo "        EVAL  $m / $c  (epoch $ep)"
    CUDA_VISIBLE_DEVICES=0 $PY eval_concept.py --root "$ROOT" --ckpt-dir "$ck" \
        --load-epoch "$ep" --forget-domain $FD --forget-class "$c" --neighbor "$nb" \
        --seed "$SEED" --num-shots 8 > "$run.concept.log" 2>&1 \
      || echo "  !! eval_concept failed for $m/$c"
    CUDA_VISIBLE_DEVICES=0 $PY eval_detect.py --root "$ROOT" --ckpt-dir "$ck" \
        --load-epoch "$ep" --forget-domain $FD --forget-class "$c" \
        --seed "$SEED" --num-shots 8 --tag "$m/$c" > "$run.detect.log" 2>&1 \
      || echo "  !! eval_detect failed for $m/$c"

    ELAPSED=$(( $(date +%s) - START ))
    echo "        done. elapsed ${ELAPSED}s, $((TOTAL-DONE)) cells left"
  done
done

echo ""
echo "=== all cells finished in $(( ($(date +%s) - START) / 60 )) min ==="
$PY collect_results.py --out "$OUT" 2>/dev/null || \
  echo "run: python collect_results.py --out $OUT"
