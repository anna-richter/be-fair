import os
import pandas as pd

file_path = "/sc-scratch/sc-scratch-ikim-guidlight/be-fair/mlzero/basic_prompt_data/test.csv"

def truncate_cell(val, maxlen=50):
    val = str(val)
    return val if len(val) <= maxlen else val[:47] + "..."

try:
    df = pd.read_csv(file_path)
    cols = list(df.columns)
    n_cols = len(cols)
    if n_cols > 20:
        display_cols = cols[:10] + cols[-10:]
    else:
        display_cols = cols
    print("Columns:", display_cols)
    pd.set_option('display.max_colwidth', 50)
    print(df[display_cols].head(3).applymap(truncate_cell).to_string(index=False))
except Exception:
    try:
        with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
            content = f.read(768)
            print("Text preview:\n", content)
    except Exception as e:
        print(f"Could not open file. Error: {e}")
        print(f"File size: {os.path.getsize(file_path)} bytes")