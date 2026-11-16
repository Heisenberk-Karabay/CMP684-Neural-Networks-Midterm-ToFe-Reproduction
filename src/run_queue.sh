#!/usr/bin/env bash
# Experiment queue (most important first). A run is skipped if its results.json exists,
# so the queue can be restarted at any time. usage: bash src/run_queue.sh
cd "$(dirname "$0")/.."
PY=./.venv/Scripts/python.exe
export PYTHONIOENCODING=utf-8

run() {  # run <name> [train_tofe args...]
  name=$1; shift
  [ -f "runs/$name/results.json" ] && { echo "skip $name"; return; }
  echo "=== $name $* ($(date +%H:%M))"
  $PY -u src/train_tofe.py --name "$name" --epochs 5 "$@" 2>&1 | grep --line-buffered -E "^ep|^\{\"name|Error|Traceback"
}

[ -f runs/tofe_b50/analysis.json ] || $PY src/analyze_tofe.py tofe_b50 2>&1 | grep -v "symlinks\|HF_TOKEN\|Warning\|from_numpy"

run noreuse_b50   --no-reuse                     # Table V
run apx_identity  --approx identity              # Table VI
run tofe_b50_e15  --epochs 15                    # is it only under-training?
run tofe_b40      --budget 0.4                   # Fig. 6 trade-off
run tofe_b60      --budget 0.6
run tofe_b70      --budget 0.7
run fixed_50_25_12 --fixed-keep 0.5,0.25,0.125   # Table VII
run fixed_40_40_10 --fixed-keep 0.4,0.4,0.1
run fixed_33_33_33 --fixed-keep 0.33,0.33,0.33
run fixed_40_30_30 --fixed-keep 0.4,0.3,0.3
run sel_depth5    --selector-depth 5             # Table IV
run sel_depth7    --selector-depth 7
run sel_dynamicvit --selector dynamicvit
run apx_dwconv    --approx dwconv                # Table VI
run apx_block     --approx block
echo "queue finished $(date +%H:%M)"
