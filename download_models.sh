#!/bin/bash
# Download the 5 open-source 7B/8B LLM judges used in this project.
# Total download: ~85 GB (all safetensors, BF16).
# Uses HF-Mirror as CDN if HF_ENDPOINT is set.
set -e

export HF_ENDPOINT=${HF_ENDPOINT:-https://hf-mirror.com}
export HF_HUB_DISABLE_XET=1
export HF_HUB_ENABLE_HF_TRANSFER=0

MODELS_ROOT=${MODELS_ROOT:-./models}
mkdir -p "$MODELS_ROOT"

MODELS=(
    "Qwen/Qwen2.5-7B-Instruct"
    "mistralai/Mistral-7B-Instruct-v0.3"
    "allenai/OLMo-7B-Instruct-hf"   # NOTE: use the -hf variant for vLLM compatibility
    "ibm-granite/granite-3.0-8b-instruct"
    "tiiuae/falcon-7b-instruct"
)

for m in "${MODELS[@]}"; do
    short=$(basename "$m")
    local_dir="$MODELS_ROOT/$short"
    if [ -f "$local_dir/config.json" ] && [ -f "$local_dir/tokenizer.json" ] \
       && ls "$local_dir"/*.safetensors >/dev/null 2>&1; then
        echo "[SKIP] $m already in $local_dir"
        continue
    fi
    mkdir -p "$local_dir"
    echo ""
    echo "================================================"
    echo "  $m -> $local_dir"
    echo "================================================"
    for attempt in 1 2 3; do
        if huggingface-cli download "$m" \
            --local-dir "$local_dir" \
            --include "*.safetensors" "*.json" "tokenizer*" "special_tokens*" \
            --max-workers 4; then
            echo "  Done: $(du -sh "$local_dir" | cut -f1)"
            break
        else
            echo "  attempt $attempt failed, retrying in 10s..."
            sleep 10
        fi
    done
done

echo ""
echo "=== All 5 models downloaded ==="
du -sh "$MODELS_ROOT"/*
