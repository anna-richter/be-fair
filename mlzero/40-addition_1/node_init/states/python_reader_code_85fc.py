import os
import pandas as pd

file_path = "/sc-scratch/sc-scratch-ikim-guidlight/be-fair/mlzero/addition_1_data/train.csv"

def display_columns(cols):
    n = len(cols)
    if n > 20:
        cols = list(cols[:10]) + ["..."] + list(cols[-10:])
    print("Columns:", cols)

def display_rows(df):
    pd.set_option('display.max_colwidth', 50)
    print(df.head(3).to_string(index=False))

try:
    df = pd.read_csv(file_path)
    display_columns(df.columns)
    display_rows(df)
except Exception:
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read(768)
            print(content)
    except Exception as e:
        print(f"Could not read file: {e}")