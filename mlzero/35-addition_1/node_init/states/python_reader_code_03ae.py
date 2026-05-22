import os
import pandas as pd

file_path = "/sc-scratch/sc-scratch-ikim-guidlight/be-fair/mlzero/addition_1_data/test.csv"

def truncate_cell(val, length=50):
    val = str(val)
    return val if len(val) <= length else val[:length] + "..."

def show_tabular(df):
    cols = list(df.columns)
    if len(cols) > 20:
        display_cols = cols[:10] + cols[-10:]
    else:
        display_cols = cols
    print("Columns:", display_cols)
    pd.set_option('display.max_colwidth', 50)
    print(df[display_cols].head(3).applymap(truncate_cell).to_string(index=False))

def show_text(fp):
    with open(fp, 'r', encoding='utf-8', errors='replace') as f:
        content = f.read(768)
    print("Text preview:\n", content)

try:
    df = pd.read_csv(file_path)
    show_tabular(df)
except Exception:
    try:
        show_text(file_path)
    except Exception as e:
        print(f"Could not read file. Error: {e}")
        print(f"File size: {os.path.getsize(file_path)} bytes")