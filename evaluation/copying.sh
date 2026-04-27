#!/usr/bin/env bash
set -euo pipefail

LOGS_DIR="/Users/arichter/Documents/GitHub/be-fair/aide/logs"
WORKSPACES_DIR="/Users/arichter/Documents/GitHub/be-fair/aide/workspaces"
EVAL_DIR="/Users/arichter/Documents/GitHub/be-fair/evaluation/aide"

# Step 1: Iterate logs folders — create eval folder and copy best_solution.py
for log_folder in "$LOGS_DIR"/*/; do
    folder_name="$(basename "$log_folder")"
    dest="$EVAL_DIR/$folder_name"
    mkdir -p "$dest"

    src_py="$log_folder/best_solution.py"
    if [[ -f "$src_py" ]]; then
        cp "$src_py" "$dest/best_solution.py"
        echo "Copied best_solution.py -> $dest/"
    else
        echo "WARNING: No best_solution.py found in $log_folder"
    fi
done

# Step 2: Iterate workspaces folders — copy working/ contents into eval folder
for ws_folder in "$WORKSPACES_DIR"/*/; do
    folder_name="$(basename "$ws_folder")"
    dest="$EVAL_DIR/$folder_name/working"
    working_src="$ws_folder/working"

    if [[ -d "$working_src" ]]; then
        mkdir -p "$dest"
        cp -r "$working_src"/. "$dest/"
        echo "Copied working/ contents -> $dest/"
    else
        echo "WARNING: No working/ folder found in $ws_folder"
    fi
done

echo "Done."