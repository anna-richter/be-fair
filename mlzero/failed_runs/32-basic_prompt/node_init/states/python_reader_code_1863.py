import os
import pandas as pd

file_path = "/home/anri21/be-fair/mlzero/basic_prompt_data/mydataset.csv"

def print_truncated(df, n=3, width=50):
    pd.set_option('display.max_colwidth', width)
    print(df.head(n).to_string(index=False))

try:
    df = pd.read_csv(file_path)
    cols = df.columns.tolist()
    n_cols = len(cols)
    if n_cols > 20:
        display_cols = cols[:10] + ['...'] + cols[-10:]
    else:
        display_cols = cols
    print("Columns:", display_cols)
    print_truncated(df)
except Exception:
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read(768)
            print("Text preview:\n", content)
    except Exception as e:
        size = os.path.getsize(file_path)
        print(f"File could not be read. Size: {size} bytes. Type: unknown.")