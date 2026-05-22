import os
import pandas as pd

file_path = "/sc-scratch/sc-scratch-ikim-guidlight/be-fair/mlzero/basic_prompt_data/test.csv"

def truncate_cell(val, length=50):
    val = str(val)
    return val if len(val) <= length else val[:47] + "..."

def display_tabular(df):
    cols = list(df.columns)
    if len(cols) > 20:
        shown = cols[:10] + ["..."] + cols[-10:]
    else:
        shown = cols
    print("Columns:", shown)
    rows = df.head(3)
    print("Rows:")
    for _, row in rows.iterrows():
        print([truncate_cell(row[c]) for c in df.columns])

def display_text(fp):
    with open(fp, 'r', encoding='utf-8', errors='ignore') as f:
        content = f.read(768)
    print("Text preview:\n", content)

try:
    df = pd.read_csv(file_path, dtype=str, nrows=3)
    df_full = pd.read_csv(file_path, dtype=str)
    display_tabular(df_full)
except Exception:
    try:
        display_text(file_path)
    except Exception as e:
        print(f"Could not read file: {e}")