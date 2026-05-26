import os

file_path = "/sc-scratch/sc-scratch-ikim-guidlight/be-fair/mlzero/basic_prompt_data/descriptions.txt"

def print_trunc(s, l=50):
    s = str(s)
    return s if len(s) <= l else s[:l-3] + "..."

def analyze_text(fp):
    with open(fp, 'r', encoding='utf-8', errors='ignore') as f:
        content = f.read(768)
    print("First lines:\n" + content)

def analyze_tabular(fp, ext):
    import pandas as pd
    try:
        if ext == '.csv':
            df = pd.read_csv(fp)
        elif ext in ['.xls', '.xlsx']:
            df = pd.read_excel(fp)
        elif ext == '.parquet':
            df = pd.read_parquet(fp)
        else:
            raise Exception
        cols = list(df.columns)
        if len(cols) > 20:
            shown = cols[:10] + ['...'] + cols[-10:]
        else:
            shown = cols
        print("Columns:", shown)
        rows = df.head(3)
        print(rows.to_string(index=False, max_colwidth=50, line_width=768))
    except Exception:
        analyze_text(fp)

ext = os.path.splitext(file_path)[1].lower()
if ext in ['.csv', '.xls', '.xlsx', '.parquet']:
    analyze_tabular(file_path, ext)
else:
    analyze_text(file_path)