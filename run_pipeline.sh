#!/bin/bash
# End-to-end pipeline: build HelpSteer2 pairs → 5-judge inference → cross-judge analysis
# Usage: bash run_pipeline.sh [N_pairs=1000]
set -e
N=${1:-1000}
PROJECT_ROOT=${PROJECT_ROOT:-$(pwd)}
export PATH=${PYTHON_BIN_DIR:-/root/miniconda3/bin}:$PATH

cd "$PROJECT_ROOT"

echo "=== Step 1: Build pairs ==="
if [ -f data/hs2_pairs.json ]; then
    echo "  data/hs2_pairs.json exists, skip"
else
    python src/build_pairs.py --source helpsteer2 --n "$N" --output data/hs2_pairs.json
fi

echo "=== Step 2: 5-judge parallel inference ==="
bash run_all_judges.sh data/hs2_pairs.json experiments/judges_hs2 "$N"

echo "=== Step 3: Cross-judge analysis ==="
python src/analyze_partitioned_rstar.py \
    --judge_files experiments/judges_hs2/*.json \
    --output_dir experiments/analysis_hs2_5judges \
    --dataset_label "HelpSteer2 (5 judges)"

echo "=== Done ==="
cat experiments/analysis_hs2_5judges/cross_partition_stats.json
