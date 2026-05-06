import os
import pandas as pd

file_path = "/home/anri21/be-fair/mlzero/addition_1_data/mydataset.csv"

def display_columns(cols):
    n = len(cols)
    if n > 20:
        first = list(cols[:10])
        last = list(cols[-10:])
        print("Columns (first 10 + last 10 of {}):\n{}\n...\n{}".format(n, first, last))
    else:
        print("Columns:\n{}".format(list(cols)))

def display_rows(df):
    pd.set_option('display.max_colwidth', 50)
    rows = df.head(3)
    print("First rows:")
    print(rows.to_string(index=False))

try:
    df = pd.read_csv(file_path)
    display_columns(df.columns)
    display_rows(df)
except Exception:
    try:
        with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
            content = f.read(768)
            print("Text file preview:\n" + content)
    except Exception as e:
        print(f"Could not read file: {e}")