import os
import pandas as pd

file_path = "/sc-scratch/sc-scratch-ikim-guidlight/be-fair/mlzero/basic_prompt_data/test.csv"

def print_columns(cols):
    n = len(cols)
    if n > 20:
        cols_disp = list(cols[:10]) + ["..."] + list(cols[-10:])
    else:
        cols_disp = list(cols)
    print("Columns:", cols_disp)

def print_rows(df):
    pd.set_option('display.max_colwidth', 50)
    rows = df.head(3)
    print(rows.to_string(index=False))

def print_text(fp):
    with open(fp, 'r', encoding='utf-8', errors='replace') as f:
        txt = f.read(768)
        print(txt)

try:
    df = pd.read_csv(file_path, nrows=3)
    print_columns(df.columns)
    df_full = pd.read_csv(file_path)
    print_rows(df_full)
except Exception:
    print_text(file_path)