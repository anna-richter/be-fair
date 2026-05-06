import os

file_path = "/home/anri21/be-fair/mlzero/basic_prompt_data/descriptions.txt"

def analyze_file(fp):
    try:
        size = os.path.getsize(fp)
        if size == 0:
            print("File is empty.")
            return
        # Try tabular formats
        try:
            import pandas as pd
            ext = os.path.splitext(fp)[1].lower()
            if ext in ['.csv', '.tsv']:
                df = pd.read_csv(fp, sep=',' if ext=='.csv' else '\t')
            elif ext in ['.xlsx', '.xls']:
                df = pd.read_excel(fp)
            elif ext == '.parquet':
                df = pd.read_parquet(fp)
            else:
                raise Exception
            cols = list(df.columns)
            n = len(cols)
            if n > 20:
                shown = cols[:10] + ['...'] + cols[-10:]
            else:
                shown = cols
            print("Columns:", shown)
            def trunc(x): return str(x)[:50] + ('...' if len(str(x)) > 50 else '')
            print(df.head(3).applymap(trunc).to_string(index=False))
            return
        except Exception:
            pass
        # Treat as text file
        with open(fp, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read(768)
            print(content)
    except Exception as e:
        print(f"Could not analyze file: {e}")

analyze_file(file_path)