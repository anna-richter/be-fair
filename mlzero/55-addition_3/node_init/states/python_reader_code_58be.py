import os
import pandas as pd

file_path = "/sc-scratch/sc-scratch-ikim-guidlight/be-fair/mlzero/addition_3_data/test.csv"

def display_tabular(df):
    cols = list(df.columns)
    if len(cols) > 20:
        shown = cols[:10] + ['...'] + cols[-10:]
    else:
        shown = cols
    print("Columns:", shown)
    pd.set_option('display.max_colwidth', 50)
    print(df.head(3).to_string(index=False))

def display_text(fp):
    with open(fp, 'r', encoding='utf-8', errors='ignore') as f:
        content = f.read(768)
    print("Text preview:\n", content)

try:
    df = pd.read_csv(file_path)
    display_tabular(df)
except Exception:
    display_text(file_path)