#!/bin/bash
# Launch 5 LLM judges in parallel, one per GPU.
# Usage: bash run_all_judges.sh [pairs_file=data/hs2_pairs.json] [out_dir=experiments/judges_hs2] [n=1000]
set -e
PAIRS=${1:-data/hs2_pairs.json}
OUTDIR=${2:-experiments/judges_hs2}
N=${3:-1000}
MODELS_ROOT=${MODELS_ROOT:-./models}
PYTHON=${PYTHON:-python}

mkdir -p "$OUTDIR"

# 5 open-source 7B/8B LLM judges (instruction-tuned, cross-family).
MODELS=(
    "$MODELS_ROOT/Qwen2.5-7B-Instruct:qwen:4096"
    "$MODELS_ROOT/Mistral-7B-Instruct-v0.3:mistral:4096"
    "$MODELS_ROOT/OLMo-7B-Instruct-hf:olmo:2048"
    "$MODELS_ROOT/granite-3.0-8b-instruct:granite:4096"
    "$MODELS_ROOT/falcon-7b-instruct:falcon:2048"
)

PIDS=()
for i in "${!MODELS[@]}"; do
    IFS=':' read -r model short max_len <<< "${MODELS[$i]}"
    out="$OUTDIR/${short}.json"
    logf="$OUTDIR/${short}.log"
    echo "[GPU $i] $(basename $model) -> $out (max_model_len=$max_len)"
    CUDA_VISIBLE_DEVICES=$i nohup $PYTHON src/llm_judge_infer.py \
        --model "$model" \
        --pairs_file "$PAIRS" \
        --output_file "$out" \
        --n_pairs "$N" \
        --max_model_len "$max_len" \
        > "$logf" 2>&1 &
    PIDS+=($!)
    sleep 5
done

echo "Launched PIDs: ${PIDS[@]}"
echo "Waiting for all judges to complete..."
for pid in "${PIDS[@]}"; do
    wait "$pid" && echo "  $pid: done" || echo "  $pid: FAILED"
done

echo "All done."
ls -la "$OUTDIR"/*.json 2>/dev/null
