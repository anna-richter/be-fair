import os

file_path = "/sc-scratch/sc-scratch-ikim-guidlight/be-fair/mlzero/addition_1_data/descriptions.txt"

def print_truncated_lines(lines, max_chars=768):
    out = ''
    for line in lines:
        if len(out) + len(line) + 1 > max_chars:
            out += line[:max_chars - len(out)] + ('...' if len(line) > max_chars - len(out) else '')
            break
        out += line
    print(out[:max_chars])

def analyze_text_file(fp):
    with open(fp, 'r', encoding='utf-8', errors='replace') as f:
        lines = f.readlines()
    print_truncated_lines(lines)

def analyze_tabular_file(fp):
    import pandas as pd
    try:
        df = pd.read_csv(fp, nrows=3, dtype=str)
    except Exception:
        try:
            df = pd.read_table(fp, nrows=3, dtype=str)
        except Exception:
            analyze_text_file(fp)
            return
    cols = list(df.columns)
    if len(cols) > 20:
        display_cols = cols[:10] + ['...'] + cols[-10:]
    else:
        display_cols = cols
    print("Columns:", display_cols)
    def trunc(x): return str(x)[:50] + ('...' if len(str(x)) > 50 else '')
    print(df.applymap(trunc).to_string(index=False))

try:
    ext = os.path.splitext(file_path)[1].lower()
    if ext in ['.csv', '.tsv', '.txt']:
        analyze_tabular_file(file_path)
    else:
        analyze_text_file(file_path)
except Exception:
    print("File could not be read or is binary/empty.")