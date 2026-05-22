import os
import pandas as pd

file_path = "/sc-scratch/sc-scratch-ikim-guidlight/be-fair/mlzero/addition_1_data/descriptions.txt"

def print_truncated(df, n=3, width=50):
    def trunc(val):
        s = str(val)
        return s if len(s) <= width else s[:width-3] + "..."
    pd.set_option('display.max_colwidth', width)
    print(df.head(n).applymap(trunc).to_string(index=False))

try:
    df = pd.read_csv(file_path, sep=None, engine='python')
    cols = list(df.columns)
    if len(cols) > 20:
        print("Columns:", cols[:10], "...", cols[-10:])
    else:
        print("Columns:", cols)
    print_truncated(df)
except Exception:
    try:
        with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
            content = f.read(768)
            print("Text preview:\n" + content)
    except Exception as e:
        print(f"Could not read file: {e}")