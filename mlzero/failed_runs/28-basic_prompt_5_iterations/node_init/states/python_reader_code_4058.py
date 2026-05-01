import os
import pandas as pd

file_path = "/home/anri21/be-fair/mlzero/basic_prompt_data/mydataset.csv"

def print_truncated(df, n=3, width=50):
    pd.set_option('display.max_colwidth', width)
    print(df.head(n).to_string(index=False))

def show_columns(cols):
    n = len(cols)
    if n > 20:
        print("Columns (first 10):", list(cols[:10]))
        print("Columns (last 10):", list(cols[-10:]))
    else:
        print("Columns:", list(cols))

try:
    df = pd.read_csv(file_path, low_memory=False)
    show_columns(df.columns)
    print_truncated(df)
except Exception:
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read(768)
            print(content)
    except Exception as e:
        stat = os.stat(file_path)
        print(f"File type: {os.path.splitext(file_path)[1]}, Size: {stat.st_size} bytes")