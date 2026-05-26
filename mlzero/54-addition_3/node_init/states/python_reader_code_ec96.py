import os

file_path = "/sc-scratch/sc-scratch-ikim-guidlight/be-fair/mlzero/addition_3_data/descriptions.txt"

def display_text_file(fp, max_chars=768):
    with open(fp, 'r', encoding='utf-8', errors='replace') as f:
        content = f.read(max_chars)
        print(content)

def main():
    try:
        import pandas as pd
        try:
            df = pd.read_csv(file_path, sep=None, engine='python', nrows=3)
            cols = list(df.columns)
            n = len(cols)
            if n > 20:
                display_cols = cols[:10] + cols[-10:]
            else:
                display_cols = cols
            print("Columns:", display_cols)
            pd.set_option('display.max_colwidth', 50)
            print(df.head(3).to_string(index=False))
            return
        except Exception:
            pass
        display_text_file(file_path)
    except Exception as e:
        print(f"Could not open file: {e}")

main()