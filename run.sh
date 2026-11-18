#!/usr/bin/env bash
# One launcher for every task: training, checkpoint selection, export and evaluation.
#
#   ./run.sh <task> <mode> [args]
#
#   tasks   brain_r4 | brain_r8 | ffhq_box | ffhq_random
#   modes   train                  train from scratch into $RUNS/<task>
#           resume <checkpoint>    continue a training checkpoint (optimizer and lr state included)
#           select [run dir]       one-draw validation PSNR of every checkpoint of a run
#                                  (default $RUNS/<task>); prints the best one
#           export <checkpoint> [file]   weights-only copy, default $CKPTS/<released file>
#           eval [checkpoint]      test-set evaluation (default $CKPTS/<released file>)
#
#   Extra args go to train.py / evaluate.py and override the defaults below, e.g.
#       GPUS=0,1,2,3 ./run.sh ffhq_box train
#       ./run.sh ffhq_random eval --M 1 16 --n 100 --restore
#       ./run.sh brain_r8 train --set training.epochs=40
#
# Environment, [default]; a local.env next to this script is sourced first:
#   GPUS [0]                      training: one process per GPU; select / eval use the first GPU
#   BACKEND [nccl]                torch.distributed backend of multi-GPU training
#   PYTHON [python]
#   FASTMRI_CACHE [data/cache320] FFHQ_ROOT [data/ffhq256]
#   RUNS [runs]                   training runs         CKPTS [checkpoints]   released weights
#   OUT                           train / resume: the run directory; eval: save per-item CSVs and images here
set -euo pipefail
cd "$(dirname "$0")"
[[ -f local.env ]] && source local.env
: "${GPUS:=0}" "${BACKEND:=nccl}" "${PYTHON:=python}"
: "${FASTMRI_CACHE:=data/cache320}" "${FFHQ_ROOT:=data/ffhq256}" "${RUNS:=runs}" "${CKPTS:=checkpoints}"

usage() { awk 'NR > 1 && /^#/ { sub(/^# ?/, ""); print; next } NR > 1 { exit }' "$0"; exit 1; }
[[ $# -ge 2 ]] || usage
task=$1; mode=$2; shift 2

case "$task" in
    brain_r4|brain_r8)
        R=${task#brain_r}
        cfg=configs/brain.yaml
        sets=("data.cache_dir=$FASTMRI_CACHE" "mri.acceleration=$R")
        weights=nullflow_brain_R$R.pt
        eval_args=(--split test --ids data_splits/brain/test100.txt --M 1 16 --lpips)
        val_args=(--split val --ids data_splits/brain/val20.txt --M 1) ;;
    ffhq_box|ffhq_random)
        cfg=configs/$task.yaml
        sets=("data.root=$FFHQ_ROOT")
        weights=nullflow_$task.pt
        eval_args=(--split test --M 1 4 16 100 --lpips)
        val_args=(--split val --M 1) ;;
    *) echo "unknown task: $task"; usage ;;
esac

ngpu=$(awk -F, '{print NF}' <<< "$GPUS")
launch() {          # train.py on every GPU of $GPUS
    if (( ngpu > 1 )); then
        CUDA_VISIBLE_DEVICES=$GPUS "$PYTHON" -m torch.distributed.run --standalone --nproc_per_node="$ngpu" "$@"
    else
        CUDA_VISIBLE_DEVICES=$GPUS "$PYTHON" "$@"
    fi
}
one_gpu() { CUDA_VISIBLE_DEVICES=${GPUS%%,*} "$PYTHON" "$@"; }

case "$mode" in
    train)
        out=${OUT:-$RUNS/$task}
        launch train.py --config "$cfg" --out "$out" --set "${sets[@]}" "training.backend=$BACKEND" "$@" ;;
    resume)
        ck=${1:?resume needs a checkpoint}; shift
        out=${OUT:-$RUNS/$task}
        [[ -f $out/config.yaml ]] && cfg=$out/config.yaml      # the run's own settings, if it has any
        launch train.py --config "$cfg" --out "$out" --resume "$ck" --set "${sets[@]}" "training.backend=$BACKEND" "$@" ;;
    select)
        run=$RUNS/$task
        if [[ $# -gt 0 && $1 != -* ]]; then run=$1; shift; fi
        one_gpu evaluate.py --config "$cfg" --set "${sets[@]}" --ckpt "$run"/checkpoints/*.pt "${val_args[@]}" "$@" ;;
    export)
        ck=${1:?export needs a checkpoint}
        dst=${2:-$CKPTS/$weights}
        mkdir -p "$(dirname "$dst")"
        "$PYTHON" scripts/export_weights.py "$ck" "$dst" ;;
    eval)
        ck=$CKPTS/$weights
        if [[ $# -gt 0 && $1 != -* ]]; then ck=$1; shift; fi
        out_args=()
        [[ -n ${OUT:-} ]] && out_args=(--out "$OUT")
        one_gpu evaluate.py --config "$cfg" --set "${sets[@]}" --ckpt "$ck" "${eval_args[@]}" \
            ${out_args[@]+"${out_args[@]}"} "$@" ;;
    *) echo "unknown mode: $mode"; usage ;;
esac
