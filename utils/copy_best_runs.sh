#!/usr/bin/env bash
set -euo pipefail

SRC="/Users/arichter/Documents/GitHub/be-fair/mlzero"
DST="/Users/arichter/Documents/GitHub/be-fair/evaluation/mlzero"

mkdir -p "$DST"

for run_dir in "$SRC"/*/; do
    run_name="$(basename "$run_dir")"

    best_link="$(find "$run_dir" -maxdepth 1 -name 'best_run_*' -print -quit)"
    if [[ -z "$best_link" ]]; then
        echo "SKIP  $run_name (no best_run_* found)"
        continue
    fi

    idx="${best_link##*/best_run_}"
    if ! [[ "$idx" =~ ^[0-9]+$ ]]; then
        echo "SKIP  $run_name (best_run name has no integer: $(basename "$best_link"))"
        continue
    fi

    src_file="$run_dir/node_${idx}/generated_code.py"
    if [[ ! -f "$src_file" ]]; then
        echo "SKIP  $run_name (missing $src_file)"
        continue
    fi

    dst_dir="$DST/$run_name"
    mkdir -p "$dst_dir"
    cp "$src_file" "$dst_dir/generated_code.py"
    echo "COPY  $run_name  node_${idx}/generated_code.py"

    src_file2="$run_dir/node_${idx}/execution_script.sh"
    if [[ ! -f "$src_file" ]]; then
        echo "SKIP  $run_name (missing $src_file)"
        continue
    fi
    cp "$src_file2" "$dst_dir/execution_script.sh"
    echo "COPY  $run_name  node_${idx}/execution_script.sh"
done

echo "Done."