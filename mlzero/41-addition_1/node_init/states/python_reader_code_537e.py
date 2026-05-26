import os
import pandas as pd

file_path = "/sc-scratch/sc-scratch-ikim-guidlight/be-fair/mlzero/addition_1_data/descriptions.txt"

def print_truncated(s, maxlen=50):
    s = str(s)
    return s if len(s) <= maxlen else s[:maxlen-3] + "..."

def analyze_file(fp):
    try:
        # Try reading as tabular data
        df = pd.read_csv(fp, sep=None, engine='python', nrows=3)
        cols = list(df.columns)
        ncol = len(cols)
        if ncol > 20:
            display_cols = cols[:10] + cols[-10:]
        else:
            display_cols = cols
        print("Columns:", display_cols)
        print("Rows:")
        for _, row in df.iterrows():
            print([print_truncated(row[c]) for c in display_cols])
    except Exception:
        # Fallback: treat as text file
        try:
            with open(fp, 'r', encoding='utf-8', errors='replace') as f:
                content = f.read(768)
                print("Text preview:\n" + content)
        except Exception as e:
            print(f"Could not read file: {e}")

analyze_file(file_path)