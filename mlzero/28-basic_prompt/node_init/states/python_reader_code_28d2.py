import os

file_path = "/home/anri21/be-fair/mlzero/basic_prompt_data/descriptions.txt"

def analyze_file(fp):
    max_output = 768
    try:
        import pandas as pd
        ext = os.path.splitext(fp)[1].lower()
        if ext in ['.csv', '.tsv']:
            df = pd.read_csv(fp, sep=',' if ext=='.csv' else '\t')
        elif ext in ['.xlsx', '.xls']:
            df = pd.read_excel(fp)
        elif ext in ['.parquet']:
            df = pd.read_parquet(fp)
        else:
            raise Exception("Not tabular")
        cols = list(df.columns)
        if len(cols) > 20:
            col_str = ', '.join(cols[:10]) + ', ..., ' + ', '.join(cols[-10:])
        else:
            col_str = ', '.join(cols)
        print("Columns:", col_str)
        pd.set_option('display.max_colwidth', 50)
        print(df.head(3).to_string(index=False))
    except Exception:
        try:
            with open(fp, 'r', encoding='utf-8', errors='replace') as f:
                content = f.read(768)
                lines = content.splitlines()
                preview = '\n'.join(lines[:min(10, len(lines))])
                print(preview[:max_output])
        except Exception as e:
            print(f"Could not read file: {e}")

analyze_file(file_path)