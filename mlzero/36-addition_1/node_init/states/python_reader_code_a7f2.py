import os
import pandas as pd

file_path = "/sc-scratch/sc-scratch-ikim-guidlight/be-fair/mlzero/addition_1_data/train.csv"

def analyze_csv(fp):
    try:
        df = pd.read_csv(fp, dtype=str, low_memory=False)
        cols = list(df.columns)
        n_cols = len(cols)
        if n_cols > 20:
            display_cols = cols[:10] + ["..."] + cols[-10:]
        else:
            display_cols = cols
        print("Columns:", display_cols)
        pd.set_option('display.max_colwidth', 50)
        print(df.head(3).to_string(index=False))
    except Exception:
        return False
    return True

def analyze_text(fp):
    with open(fp, 'r', encoding='utf-8', errors='ignore') as f:
        content = f.read(768)
        print(content)

if not analyze_csv(file_path):
    analyze_text(file_path)