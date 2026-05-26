import os
import pandas as pd

file_path = "/sc-scratch/sc-scratch-ikim-guidlight/be-fair/mlzero/basic_prompt_data/train.csv"

def print_columns(cols):
    n = len(cols)
    if n > 20:
        print("Columns:", list(cols[:10]) + ["..."] + list(cols[-10:]))
    else:
        print("Columns:", list(cols))

def truncate_cell(val):
    s = str(val)
    return s if len(s) <= 50 else s[:47] + "..."

try:
    df = pd.read_csv(file_path, low_memory=False)
    print_columns(df.columns)
    pd.set_option('display.max_colwidth', 50)
    print(df.head(3).to_string(index=False))
except Exception:
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read(768)
            print("Text preview:\n", content)
    except Exception as e:
        print(f"Could not open file. Type: {os.path.splitext(file_path)[1]}, Size: {os.path.getsize(file_path)} bytes")