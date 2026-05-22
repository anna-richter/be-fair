import os
import pandas as pd

file_path = "/sc-scratch/sc-scratch-ikim-guidlight/be-fair/mlzero/addition_1_data/train.csv"

def display_tabular(df):
    cols = list(df.columns)
    if len(cols) > 20:
        display_cols = cols[:10] + cols[-10:]
    else:
        display_cols = cols
    print("Columns:", display_cols)
    pd.set_option('display.max_colwidth', 50)
    print(df[display_cols].head(3).to_string(index=False))

try:
    df = pd.read_csv(file_path)
    display_tabular(df)
except Exception:
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read(768)
            print("Text preview:\n", content)
    except Exception as e:
        print(f"Could not open file: {e}")