#!/usr/bin/env bash
# =============================================================================
# Experiment driver. Runs training + all three evaluations for each cell and
# appends a one-line record per run to $OUT/grid_results.csv.
#
#   ./run_grid.sh phaseA     critical gaps, tiger only  (5 runs,  ~7h)
#   ./run_grid.sh grid       6 concepts x 3 seeds x 3 methods (54 runs)
#   ./run_grid.sh evalonly   re-run evaluations over existing checkpoints
#
# Edit PY / ROOT / OUT for the machine you are on.
# =============================================================================
set -u

PY=${PY:-python}
ROOT=${ROOT:-"/home/ai/machine unlearning/ebm_unlearning/data/domainnet"}
OUT=${OUT:-"$HOME/adu_results/grid"}
CFG=configs/trainers/vit_b16_ep50_bs32_concept.yaml
DCFG=configs/datasets/domainnet_mini_paper_df.yaml
EPOCH=50
# per-step memory knobs: leave empty on a big GPU to use the full pool each step
CHUNK=${CHUNK:-0}
BS=${BS:-32}

# Evaluation set produced by select_concepts.py (seed 0, 3 per proximity tercile).
# Do not edit by hand -- re-run the selector if the rule changes.
CONCEPTS=${CONCEPTS:-"dog fish squirrel skateboard flamingo spider vase leaf The_Eiffel_Tower"}
SEEDS=${SEEDS:-"1 2 3"}
SEEDS_CONTROL=${SEEDS_CONTROL:-"1"}   # retain-only is a control: fewer seeds needed
FORGET_DOMAIN=${FORGET_DOMAIN:-sketch}
POOL=${POOL:-100}

mkdir -p "$OUT"
CSV="$OUT/grid_results.csv"
[ -f "$CSV" ] || echo "method,concept,seed,tag,run_dir" > "$CSV"

# ---------------------------------------------------------------- one run
# train_one <tag> <concept> <seed> <extra train args...>
train_one () {
  local tag=$1 concept=$2 seed=$3; shift 3
  local run="$OUT/${tag}__${concept}__s${seed}"
  if [ -d "$run" ] && ls "$run"/seed*/ForgetDomain1/$FORGET_DOMAIN/*/VLPromptLearner/model.pth.tar-$EPOCH >/dev/null 2>&1; then
    echo "[skip] $tag $concept s$seed (checkpoint exists)"; return 0
  fi
  echo "=== TRAIN $tag | $concept | seed $seed ==="
  local chunk_arg=""
  [ "$CHUNK" != "0" ] && chunk_arg="--forget_chunk $CHUNK"
  CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  $PY train_loop.py \
      --root "$ROOT" --trainer IVLP_VL_Adapter_Prompt \
      --dataset-config-file $DCFG --config-file $CFG \
      --output-dir "$run" \
      --num_shots 8 --dataset_name domainnet_mini_paper_df \
      --is_domain_divided --seeds "$seed" \
      --run_forget_domains $FORGET_DOMAIN --forget_classes "$concept" \
      "$@" $chunk_arg \
      DATALOADER.TRAIN_X.BATCH_SIZE $BS \
      > "$run.train.log" 2>&1
  echo "$tag,$concept,$seed,$tag,$run" >> "$CSV"
}

# nearest class in zero-shot CLIP text space, per select_concepts.py
neighbor_of () {
  case "$1" in
    dog) echo cat ;;            fish) echo bird ;;        squirrel) echo monkey ;;
    skateboard) echo guitar ;;  flamingo) echo swan ;;    spider) echo ant ;;
    vase) echo flower ;;        leaf) echo feather ;;     The_Eiffel_Tower) echo castle ;;
    tiger) echo lion ;;
    *) echo lion ;;             # fallback; update if the concept set changes
  esac
}

# eval_one <tag> <concept> <seed>
eval_one () {
  local tag=$1 concept=$2 seed=$3
  local nb; nb=$(neighbor_of "$concept")
  local run="$OUT/${tag}__${concept}__s${seed}"
  local ck
  ck=$(dirname "$(ls -d "$run"/seed*/ForgetDomain1/$FORGET_DOMAIN/*/VLPromptLearner 2>/dev/null | head -1)" 2>/dev/null)
  [ -z "$ck" ] && { echo "[warn] no checkpoint for $tag $concept s$seed"; return 1; }
  echo "=== EVAL $tag | $concept | seed $seed ==="
  CUDA_VISIBLE_DEVICES=0 $PY eval_concept.py --root "$ROOT" --ckpt-dir "$ck" \
      --load-epoch $EPOCH --forget-domain $FORGET_DOMAIN --forget-class "$concept" \
      --neighbor "$nb" --seed "$seed" --num-shots 8 > "$run.concept.log" 2>&1
  CUDA_VISIBLE_DEVICES=0 $PY eval_detect.py --root "$ROOT" --ckpt-dir "$ck" \
      --load-epoch $EPOCH --forget-domain $FORGET_DOMAIN --forget-class "$concept" \
      --seed "$seed" --num-shots 8 --tag "$tag/$concept/s$seed" > "$run.detect.log" 2>&1
  CUDA_VISIBLE_DEVICES=0 $PY diagnose_sink.py --root "$ROOT" --ckpt-dir "$ck" \
      --load-epoch $EPOCH --forget-domain $FORGET_DOMAIN --forget-class "$concept" \
      --seed "$seed" --num-shots 8 > "$run.sink.log" 2>&1
}

# ---------------------------------------------------------------- methods
OURS="--forget_loss_type suppress_marg --forget_weight 1.0 --flat_weight 1.0 --marg_weight 1.0 --forget_pool_size $POOL --exclude_forget_class_from_retain"
NEGGRAD="--forget_loss_type neggrad --exclude_forget_class_from_retain"
RETAINONLY="--forget_loss_type suppress_flat --forget_weight 0.0 --exclude_forget_class_from_retain"

case "${1:-}" in

  phaseA)   # critical gaps, tiger/seed1 only: controls + baseline + cap sweep
    train_one retain_only tiger 1 $RETAINONLY
    train_one neggrad     tiger 1 $NEGGRAD
    train_one cap_inf     tiger 1 $OURS --suppress_cap 1000000
    train_one cap8        tiger 1 $OURS --suppress_cap 8
    train_one ours        tiger 1 $OURS --suppress_cap 6
    for t in retain_only neggrad cap_inf cap8 ours; do eval_one $t tiger 1; done
    ;;

  grid)     # main table: ours + NegGrad over all concepts x seeds
    for c in $CONCEPTS; do for s in $SEEDS; do
      train_one ours    "$c" "$s" $OURS --suppress_cap 6
      train_one neggrad "$c" "$s" $NEGGRAD
      for t in ours neggrad; do eval_one $t "$c" "$s"; done
    done; done
    ;;

  control)  # retain-only: qualitative control, fewer seeds
    for c in $CONCEPTS; do for s in $SEEDS_CONTROL; do
      train_one retain_only "$c" "$s" $RETAINONLY
      eval_one retain_only "$c" "$s"
    done; done
    ;;

  evalonly)
    for d in "$OUT"/*__*__s*/; do
      b=$(basename "${d%/}"); t=${b%%__*}; rest=${b#*__}; c=${rest%%__*}; s=${rest##*s}
      eval_one "$t" "$c" "$s"
    done
    ;;

  *) echo "usage: $0 {phaseA|grid|control|evalonly}"; exit 1 ;;
esac

echo "=== done. logs + $CSV under $OUT ==="
