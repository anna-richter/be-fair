import os
import pandas as pd

file_path = "/home/anri21/be-fair/mlzero/basic_prompt_data/mydataset.csv"

def print_columns(cols):
    n = len(cols)
    if n > 20:
        print("Columns (first 10):", list(cols[:10]))
        print("Columns (last 10):", list(cols[-10:]))
    else:
        print("Columns:", list(cols))

def print_rows(df):
    pd.set_option('display.max_colwidth', 50)
    rows = df.head(3)
    print(rows.to_string(index=False))

try:
    df = pd.read_csv(file_path, low_memory=False)
    print_columns(df.columns)
    print_rows(df)
except Exception:
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read(768)
            print("Text preview:\n", content)
    except Exception as e:
        print(f"Could not open file: {e}")