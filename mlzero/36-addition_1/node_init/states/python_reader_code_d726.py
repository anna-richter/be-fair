import os
import pandas as pd

file_path = "/home/anri21/be-fair/mlzero/addition_1_data/mydataset.csv"

def print_columns(cols):
    n = len(cols)
    if n > 20:
        first = list(cols[:10])
        last = list(cols[-10:])
        print("Columns (first 10):", first)
        print("Columns (last 10):", last)
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
            print("Text file preview:\n", content)
    except Exception as e:
        size = os.path.getsize(file_path)
        print(f"Could not read file. Size: {size} bytes. Type: unknown.")