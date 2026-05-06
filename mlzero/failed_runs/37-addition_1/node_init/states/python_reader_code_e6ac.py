import os

file_path = "/home/anri21/be-fair/mlzero/addition_1_data/descriptions.txt"

def print_truncated_lines(lines, max_chars=768):
    out = ''
    for line in lines:
        if len(out) + len(line) + 1 > max_chars:
            out += line[:max_chars - len(out)] + ('...' if len(line) > max_chars - len(out) else '')
            break
        out += line
    print(out)

def analyze_file(fp):
    try:
        import pandas as pd
        ext = os.path.splitext(fp)[1].lower()
        if ext in ['.csv', '.tsv', '.txt']:
            try:
                df = pd.read_csv(fp, sep=None, engine='python')
            except Exception:
                raise
        elif ext in ['.xls', '.xlsx']:
            df = pd.read_excel(fp)
        elif ext in ['.parquet']:
            df = pd.read_parquet(fp)
        else:
            raise Exception
        cols = list(df.columns)
        if len(cols) > 20:
            display_cols = cols[:10] + ['...'] + cols[-10:]
        else:
            display_cols = cols
        print("Columns:", display_cols)
        pd.set_option('display.max_colwidth', 50)
        print(df.head(3).to_string(index=False))
    except Exception:
        try:
            with open(fp, 'r', encoding='utf-8', errors='replace') as f:
                lines = f.readlines()
            print_truncated_lines(lines)
        except Exception as e:
            print(f"Could not read file: {e}")

analyze_file(file_path)