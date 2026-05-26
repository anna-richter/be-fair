import os
import pandas as pd

file_path = "/sc-scratch/sc-scratch-ikim-guidlight/be-fair/mlzero/basic_prompt_data/descriptions.txt"

def print_truncated(df, n=3, width=50):
    def trunc(val):
        s = str(val)
        return s if len(s) <= width else s[:width-3] + "..."
    rows = df.head(n).applymap(trunc)
    print(rows.to_string(index=False))

def show_columns(cols):
    if len(cols) > 20:
        first = list(cols[:10])
        last = list(cols[-10:])
        print("Columns:", first + ["..."] + last)
    else:
        print("Columns:", list(cols))

try:
    df = pd.read_csv(file_path, sep=None, engine='python')
    show_columns(df.columns)
    print_truncated(df)
except Exception:
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read(768)
            print(content)
    except Exception as e:
        print(f"Could not read file: {e}")