import os
import pandas as pd

file_path = "/sc-scratch/sc-scratch-ikim-guidlight/be-fair/mlzero/addition_2_data/train.csv"

def display_tabular(df):
    cols = list(df.columns)
    n = len(cols)
    if n > 20:
        shown = cols[:10] + ['...'] + cols[-10:]
    else:
        shown = cols
    print("Columns:", shown)
    pd.set_option('display.max_colwidth', 50)
    print(df.head(3).to_string(index=False))

try:
    df = pd.read_csv(file_path)
    display_tabular(df)
except Exception:
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read(768)
            print(content)
    except Exception as e:
        print(f"Could not open file: {e}")