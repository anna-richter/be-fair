import os

file_path = "/sc-scratch/sc-scratch-ikim-guidlight/be-fair/mlzero/addition_1_data/descriptions.txt"

def print_truncated_lines(lines, max_chars=768):
    out = ''
    for line in lines:
        if len(out) + len(line) > max_chars:
            out += line[:max_chars - len(out)] + '...'
            break
        out += line
    print(out)

def analyze_file(fp):
    try:
        import pandas as pd
        _, ext = os.path.splitext(fp)
        if ext.lower() in ['.csv', '.tsv']:
            df = pd.read_csv(fp, sep=None, engine='python')
        elif ext.lower() in ['.xls', '.xlsx']:
            df = pd.read_excel(fp)
        elif ext.lower() == '.parquet':
            df = pd.read_parquet(fp)
        else:
            raise Exception("Not tabular")
        cols = list(df.columns)
        n = len(cols)
        if n > 20:
            display_cols = cols[:10] + ['...'] + cols[-10:]
        else:
            display_cols = cols
        print("Columns:", display_cols)
        pd.set_option('display.max_colwidth', 50)
        print(df.head(3).to_string(index=False))
    except Exception:
        try:
            with open(fp, 'r', encoding='utf-8', errors='ignore') as f:
                lines = f.readlines()
            print_truncated_lines(lines)
        except Exception as e:
            print(f"Could not read file: {e}")

analyze_file(file_path)