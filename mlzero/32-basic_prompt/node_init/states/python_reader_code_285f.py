import os

file_path = "/sc-scratch/sc-scratch-ikim-guidlight/be-fair/mlzero/basic_prompt_data/descriptions.txt"

def print_trunc(s, l=50):
    s = str(s)
    return s if len(s) <= l else s[:l-3] + "..."

def analyze_text(fp):
    with open(fp, 'r', encoding='utf-8', errors='ignore') as f:
        content = f.read(768)
    print("First lines:\n" + content)

def analyze_tabular(fp):
    import pandas as pd
    try:
        df = pd.read_csv(fp, nrows=3)
    except Exception:
        try:
            df = pd.read_excel(fp, nrows=3)
        except Exception:
            analyze_text(fp)
            return
    cols = list(df.columns)
    if len(cols) > 20:
        cols_disp = cols[:10] + ["..."] + cols[-10:]
    else:
        cols_disp = cols
    print("Columns:", cols_disp)
    print("Rows:")
    for _, row in df.iterrows():
        print([print_trunc(row[c]) for c in cols_disp if c != "..."])

def main():
    ext = os.path.splitext(file_path)[1].lower()
    try:
        if ext in ['.csv', '.tsv', '.xls', '.xlsx', '.parquet']:
            analyze_tabular(file_path)
        else:
            analyze_text(file_path)
    except Exception:
        analyze_text(file_path)

main()