import os
import pandas as pd

file_path = "/sc-scratch/sc-scratch-ikim-guidlight/be-fair/mlzero/basic_prompt_data/test.csv"

def truncate_cell(val, length=50):
    val = str(val)
    return val if len(val) <= length else val[:length] + "..."

def display_tabular(df):
    cols = list(df.columns)
    n_cols = len(cols)
    if n_cols > 20:
        show_cols = cols[:10] + cols[-10:]
    else:
        show_cols = cols
    print("Columns:", show_cols)
    rows = df.head(3)
    print(rows.applymap(lambda x: truncate_cell(x)).to_string(index=False))

def display_text(fp):
    with open(fp, 'r', encoding='utf-8', errors='ignore') as f:
        content = f.read(768)
    print(content)

try:
    df = pd.read_csv(file_path)
    display_tabular(df)
except Exception:
    display_text(file_path)