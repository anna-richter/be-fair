import os
import pandas as pd

file_path = "/sc-scratch/sc-scratch-ikim-guidlight/be-fair/mlzero/basic_prompt_data/test.csv"

def print_truncated(df, n=3, width=50):
    pd.set_option('display.max_colwidth', width)
    if df.shape[1] > 20:
        cols = list(df.columns[:10]) + list(df.columns[-10:])
        df = df[cols]
        print("Columns (first 10 + last 10):", cols)
    else:
        print("Columns:", list(df.columns))
    print(df.head(n).to_string(index=False))

try:
    df = pd.read_csv(file_path)
    print_truncated(df)
except Exception:
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read(768)
            print("Text file preview:\n", content)
    except Exception as e:
        print(f"Could not open file. Error: {e}")