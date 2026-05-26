# Agent Task: Prepare Evaluation Scripts (mlzero)

## What you need to do

For every subfolder in `/Users/arichter/Documents/GitHub/be-fair/evaluation/mlzero/` that contains **both** `generated_code.py` and `execution_script.sh`:

1. Copy `generated_code.py` → `{foldername}_generated_code_DDI.py` in the **same subfolder**.
2. Copy `execution_script.sh` → `{foldername}_execution_script_DDI.sh` in the **same subfolder**.
3. Apply 4 minimal patches to the Python copy and 1 minimal patch to the shell copy. **All changes go into the copies only — never modify the originals.**

Example: for the folder `28-basic_prompt`, copy:

- `28-basic_prompt/generated_code.py` → `28-basic_prompt/28-basic_prompt_generated_code_DDI.py`
- `28-basic_prompt/execution_script.sh` → `28-basic_prompt/28-basic_prompt_execution_script_DDI.sh`

Then apply the patches described below to the two `_DDI` copies.

**Mode required:** Agent mode (not Ask or Plan mode) — you will be making file edits.

> **Important:** never touch `generated_code.py` or `execution_script.sh` themselves. Every edit goes into the `_DDI` copies.

---

## Repository context

The 26 subfolders to process (folders `44-*` and `51-*` are intentionally missing — skip them):

```
evaluation/mlzero/
  28-basic_prompt/   29-basic_prompt/   30-basic_prompt/   31-basic_prompt/
  32-basic_prompt/   33-basic_prompt/   34-basic_prompt/
  35-addition_1/     36-addition_1/     37-addition_1/     38-addition_1/
  39-addition_1/     40-addition_1/     41-addition_1/
  42-addition_2/     43-addition_2/     45-addition_2/     46-addition_2/
  47-addition_2/     48-addition_2/
  49-addition_3/     50-addition_3/     52-addition_3/     53-addition_3/
  54-addition_3/     55-addition_3/
```

Each subfolder already contains a `generated_code.py` (a self-contained training+prediction script that reads `train.csv` + `test.csv` from a `DATA_DIR` and writes a `results.csv` to an `OUTPUT_DIR`) and an `execution_script.sh` (a SLURM-friendly wrapper that creates the conda env, installs requirements, and runs the Python script).

---

## Reference paths

| Name | Absolute path |
|------|---------------|
| Training + test data (pre-formatted by the user) | `/sc-scratch/sc-scratch-ikim-guidlight/be-fair/evaluation/evaluation_data/` |
| Output directory (per folder) | `/sc-scratch/sc-scratch-ikim-guidlight/be-fair/evaluation/mlzero/{foldername}` |

The `evaluation_data/` directory contains `train.csv`, `test.csv`, and a `MyImages/` subfolder. Both CSVs have an `image_name` column listing names **without extension** (e.g. `000001`).

- **Training images** stay `.jpg` (untouched).
- **DDI test images on disk are `.png`** (e.g. `000001.png`). Therefore the **test** image-path construction in each Python script must be patched to use `.png`, while the train side stays `.jpg`.

The new prediction file `DDI_predictions.csv` is written into the per-folder output directory. It has exactly 2 columns:

- `DDI_file` — `image_name` + `.png` (e.g. `000001.png`)
- `predicted_probability` — float in [0, 1], the probability of malignancy

---

## Editing conventions

### Rule — mark every change

Wrap every changed block in `# start change` / `# end change`. Comment out the old line instead of deleting it. This applies to both Python and shell files (both use `#` for comments — the markers are identical).

```python
# start change
# OLD_VAR = "/sc-scratch/.../mlzero/basic_prompt_data"  # original
OLD_VAR = "/sc-scratch/sc-scratch-ikim-guidlight/be-fair/evaluation/evaluation_data/"
# end change
```

```bash
# start change
# PY_SCRIPT="/sc-scratch/.../mlzero/28-basic_prompt/node_0/generated_code.py"  # original
PY_SCRIPT="/sc-scratch/sc-scratch-ikim-guidlight/be-fair/evaluation/mlzero/28-basic_prompt/28-basic_prompt_generated_code_DDI.py"
# end change
```

---

## The 5 patches

Patches 1–4 are applied to `{foldername}_generated_code_DDI.py`. Patch 5 is the only change to `{foldername}_execution_script_DDI.sh`.

### Patch 1 — `DATA_DIR`

Comment out the original `DATA_DIR = ".../basic_prompt_data"` (or `addition_1_data`, `addition_2_data`, `addition_3_data`) and set it to the evaluation data path.

```python
# start change
# DATA_DIR = "/sc-scratch/sc-scratch-ikim-guidlight/be-fair/mlzero/<original_data_name>"  # original
DATA_DIR = "/sc-scratch/sc-scratch-ikim-guidlight/be-fair/evaluation/evaluation_data/"
# end change
```

Reproduce the exact original line in the commented-out line (read it from the script). Most scripts define `DATA_DIR` at module level near the top; a few (30, 41, 46) define it inside the `if __name__ == "__main__":` block — the per-script section below tells you exactly where.

### Patch 2 — `OUTPUT_DIR`

Comment out the original and replace with `/sc-scratch/.../evaluation/mlzero/{foldername}` (no trailing `/output`, no `/node_N/`).

```python
# start change
# OUTPUT_DIR = "/sc-scratch/sc-scratch-ikim-guidlight/be-fair/mlzero/{foldername}/node_N/output"  # original
OUTPUT_DIR = "/sc-scratch/sc-scratch-ikim-guidlight/be-fair/evaluation/mlzero/{foldername}"
# end change
```

Substitute `{foldername}` literally (e.g. `28-basic_prompt`). The original `node_N` value differs per folder — preserve whatever was there in the commented-out line.

### Patch 3 — Test image extension (`.jpg` → `.png`)

This patches **only the test set's** image-path construction. The training side stays `.jpg` and is left untouched.

**Generic template** (use when the script's test-image line calls a helper or a function that hardcodes `.jpg`):

```python
# start change
# test['image'] = test['image_name'].apply(get_image_path)  # original (.jpg)
test['image'] = test['image_name'].apply(
    lambda x: os.path.abspath(os.path.join(IMG_DIR, f"{x}.png"))
)
# end change
```

Adapt the variable names per script:

- `test` vs `test_df` — use whichever the script uses.
- `image` vs `image_path` — use the column name the script writes to.
- `IMG_DIR` vs `IMAGE_DIR` vs `IMAGES_DIR` — use whatever the script declares.

**Inline-lambda variant** (scripts **28, 34, 35, 40** already use an inline lambda with the literal `.jpg`). For these the patch is one character on the test line:

```python
# start change
# test['image'] = test['image_name'].apply(lambda x: os.path.join(IMG_DIR, f"{x}.jpg"))  # original (.jpg)
test['image'] = test['image_name'].apply(lambda x: os.path.join(IMG_DIR, f"{x}.png"))
# end change
```

**Helper-overwrites-`image_name` variant** (scripts **36, 45**): the helper mutates the `image_name` column itself rather than adding a new `image` column. The original `image_name` values are then lost. After applying the helper, swap the extension on the resulting path **and** save the original names from a fresh re-read of `test.csv` so Patch 4 still has access to them. The per-script section spells this out.

**Helper-inside-`prepare_test_data` variant** (scripts **33, 46, 50**): the test image column is set inside a helper function. Patch the line **inside the helper** (it is only called for the test side, so the train path is not affected). The per-script section gives the exact line number.

### Patch 4 — DDI save block

Insert this block **immediately after** the script's prediction step (where the malignancy probability is computed) and **before** any validation/assert/output-CSV code. Adapt variable names per script.

**Generic template:**

```python
# start change
_ddi_df = pd.DataFrame({
    "DDI_file": test["image_name"].astype(str) + ".png",
    "predicted_probability": <PROB_VAR>.astype(float),
}).reset_index(drop=True)
_ddi_df.to_csv(os.path.join(OUTPUT_DIR, "DDI_predictions.csv"), index=False)
# end change
```

Where:

- `<PROB_VAR>` is the per-script numpy array, Series, or list that holds the probability of malignancy. See the per-script section below for the exact name.
- Use `test_df` instead of `test` where the script uses that variable name.
- If `<PROB_VAR>` is a Python `list` (e.g. script 29), wrap it: `np.asarray(<PROB_VAR>)`.
- If the test DataFrame's `image_name` column has been mutated into an absolute path (scripts **36** and **45**), build `DDI_file` from a fresh re-read of `test.csv` — see those two sections.

For pandas Series whose values are floats already (e.g. `proba["malignant"]`), `.astype(float)` is still fine.

### Patch 5 — `PY_SCRIPT` line in the shell copy

The **only** change to `{foldername}_execution_script_DDI.sh`. Comment out the original `PY_SCRIPT=` line and replace it with the path to the new `_DDI` Python copy. No other line in the shell file (env setup, requirements installs, conda activation, etc.) is touched.

```bash
# start change
# PY_SCRIPT="/sc-scratch/sc-scratch-ikim-guidlight/be-fair/mlzero/{foldername}/node_N/generated_code.py"  # original
PY_SCRIPT="/sc-scratch/sc-scratch-ikim-guidlight/be-fair/evaluation/mlzero/{foldername}/{foldername}_generated_code_DDI.py"
# end change
```

The original `node_N` value varies per folder (e.g. `node_0`, `node_13`, `node_3`, …). Read the original `PY_SCRIPT=` line from the file and preserve it verbatim in the commented-out line. The per-script section below also lists the exact original path for cross-checking.

---

## Per-script instructions

For each script, line numbers refer to the **original** `generated_code.py`. Apply the patches to the copied `_DDI.py` file; the line numbers in the copy will be identical until the first insertion (Patch 4 shifts later lines).

---

### 28-basic_prompt

- **Patch 1** — `DATA_DIR` at L37. Original: `DATA_DIR = "/sc-scratch/sc-scratch-ikim-guidlight/be-fair/mlzero/basic_prompt_data"`.
- **Patch 2** — `OUTPUT_DIR` at L41. Original: `OUTPUT_DIR = "/sc-scratch/sc-scratch-ikim-guidlight/be-fair/mlzero/28-basic_prompt/node_0/output"`.
- **Patch 3** — Test image at L88. Inline-lambda variant:
  ```python
  # start change
  # test['image'] = test['image_name'].apply(lambda x: os.path.join(IMG_DIR, f"{x}.jpg"))  # original (.jpg)
  test['image'] = test['image_name'].apply(lambda x: os.path.join(IMG_DIR, f"{x}.png"))
  # end change
  ```
- **Patch 4** — Insert after L133 (`malignancy_probs = proba[1].values`), before L138 (`results_df = ...`). `<PROB_VAR>` = `malignancy_probs` (numpy array). Use `test` and `test["image_name"]`.
- **Patch 5** — Shell PY_SCRIPT original: `PY_SCRIPT="/sc-scratch/sc-scratch-ikim-guidlight/be-fair/mlzero/28-basic_prompt/node_0/generated_code.py"`.

---

### 29-basic_prompt

- **Patch 1** — `DATA_DIR` at L40. Original: `DATA_DIR = "/sc-scratch/sc-scratch-ikim-guidlight/be-fair/mlzero/basic_prompt_data"`.
- **Patch 2** — `OUTPUT_DIR` at L41. Original: `OUTPUT_DIR = "/sc-scratch/sc-scratch-ikim-guidlight/be-fair/mlzero/29-basic_prompt/node_13/output"`.
- **Patch 3** — Test image at L109. The script uses helper `get_image_path` (L50–52) which hardcodes `.jpg` and is shared with train (L108). Override the test line directly:
  ```python
  # start change
  # test_df['image'] = test_df['image_name'].apply(get_image_path)  # original (.jpg)
  test_df['image'] = test_df['image_name'].apply(
      lambda x: os.path.join(IMAGES_DIR, f"{x}.png")
  )
  # end change
  ```
- **Patch 4** — Insert after the test-prediction loop ends (after L226, `test_probs.extend(probs)` finishes) and before L228 (`results_df = pd.DataFrame(...)`). `test_probs` is a Python list — wrap with `np.asarray(test_probs)`. Uses `test_df` and `test_df["image_name"]`.
- **Patch 5** — Shell PY_SCRIPT original: `PY_SCRIPT="/sc-scratch/sc-scratch-ikim-guidlight/be-fair/mlzero/29-basic_prompt/node_13/generated_code.py"`.

---

### 30-basic_prompt

- **Patch 1** — `DATA_DIR` at L51 (inside `if __name__ == "__main__":`). Original: `DATA_DIR = "/sc-scratch/sc-scratch-ikim-guidlight/be-fair/mlzero/basic_prompt_data"`.
- **Patch 2** — `OUTPUT_DIR` at L55. Original: `OUTPUT_DIR = "/sc-scratch/sc-scratch-ikim-guidlight/be-fair/mlzero/30-basic_prompt/node_0/output"`.
- **Patch 3** — Test image at L84. The script defines a local helper `image_name_to_path` (L77–81) used by both train (L83) and test (L84). Override the test line:
  ```python
  # start change
  # test['image'] = test['image_name'].apply(image_name_to_path)  # original (.jpg)
  test['image'] = test['image_name'].apply(
      lambda x: os.path.join(IMG_DIR, (x if x.endswith('.png') else x + '.png'))
  )
  # end change
  ```
  Note there's a second occurrence at L129 (`test_for_pred['image'] = test_for_pred['image_name'].apply(image_name_to_path)`) — patch it identically:
  ```python
  # start change
  # test_for_pred['image'] = test_for_pred['image_name'].apply(image_name_to_path)  # original (.jpg)
  test_for_pred['image'] = test_for_pred['image_name'].apply(
      lambda x: os.path.join(IMG_DIR, (x if x.endswith('.png') else x + '.png'))
  )
  # end change
  ```
- **Patch 4** — Insert after L148 (`malignancy_prob = proba.iloc[:, -1].values` or the branch that sets `malignancy_prob`), before L154 (`output_df = test[['image_name']].copy()`). `<PROB_VAR>` = `malignancy_prob`. Uses `test` and `test["image_name"]`.
- **Patch 5** — Shell PY_SCRIPT original: `PY_SCRIPT="/sc-scratch/sc-scratch-ikim-guidlight/be-fair/mlzero/30-basic_prompt/node_0/generated_code.py"`.

---

### 31-basic_prompt

- **Patch 1** — `DATA_DIR` at L41. Original: `DATA_DIR = "/sc-scratch/sc-scratch-ikim-guidlight/be-fair/mlzero/basic_prompt_data"`.
- **Patch 2** — `OUTPUT_DIR` at L45. Original: `OUTPUT_DIR = "/sc-scratch/sc-scratch-ikim-guidlight/be-fair/mlzero/31-basic_prompt/node_3/output"`.
- **Patch 3** — Test image at L167. The script uses `_ensure_absolute_image_paths(df, image_col)` (L101–104) which calls `_get_image_path` (L60–62), hardcoding `.jpg`. Shared with train at L166. Override the test line:
  ```python
  # start change
  # test_df = _ensure_absolute_image_paths(test_df, image_col)  # original (.jpg)
  test_df = test_df.copy()
  test_df[image_col] = test_df[image_col].apply(
      lambda x: os.path.abspath(os.path.join(IMG_DIR, f"{x}.png"))
  )
  # end change
  ```
- **Patch 4** — Insert after L206 (`malignancy_probs = proba_df["1"].values`), before L211 (`output_df = test_df.copy()`). `<PROB_VAR>` = `malignancy_probs`. Uses `test_df` and `test_df["image_name"]`.
- **Patch 5** — Shell PY_SCRIPT original: `PY_SCRIPT="/sc-scratch/sc-scratch-ikim-guidlight/be-fair/mlzero/31-basic_prompt/node_3/generated_code.py"`.

---

### 32-basic_prompt

- **Patch 1** — `DATA_DIR` at L42. Original: `DATA_DIR = "/sc-scratch/sc-scratch-ikim-guidlight/be-fair/mlzero/basic_prompt_data"`.
- **Patch 2** — `OUTPUT_DIR` at L46. Original: `OUTPUT_DIR = "/sc-scratch/sc-scratch-ikim-guidlight/be-fair/mlzero/32-basic_prompt/node_18/output"`.
- **Patch 3** — Test image at L83. The script's `get_image_path` (L61–63) is shared with train (L82). Override the test line:
  ```python
  # start change
  # test['image'] = test['image_name'].apply(get_image_path)  # original (.jpg)
  test['image'] = test['image_name'].apply(
      lambda x: os.path.join(IMG_DIR, f"{x}.png")
  )
  # end change
  ```
- **Patch 4** — Insert after L160 (the branch ending with `malignancy_prob = np.array(proba)`), before L165 (`results_df = test[['image_name']].copy()`). `<PROB_VAR>` = `malignancy_prob`. Uses `test` and `test["image_name"]`.
- **Patch 5** — Shell PY_SCRIPT original: `PY_SCRIPT="/sc-scratch/sc-scratch-ikim-guidlight/be-fair/mlzero/32-basic_prompt/node_18/generated_code.py"`.

---

### 33-basic_prompt

- **Patch 1** — `DATA_DIR` at L40. Original: `DATA_DIR = "/sc-scratch/sc-scratch-ikim-guidlight/be-fair/mlzero/basic_prompt_data"`.
- **Patch 2** — `OUTPUT_DIR` at L44. Original: `OUTPUT_DIR = "/sc-scratch/sc-scratch-ikim-guidlight/be-fair/mlzero/33-basic_prompt/node_8/output"`.
- **Patch 3** — Test image **inside `prepare_test_data`** at L118. The function `prepare_train_data` (L89) calls `get_image_abs_path` separately at L104, so patching only the line inside `prepare_test_data` does not affect train. The helper `get_image_abs_path` (L83–87) appends `.jpg`. Patch L118:
  ```python
  # start change
  # df['image'] = df['image_name'].apply(get_image_abs_path)  # original (.jpg)
  df['image'] = df['image_name'].apply(
      lambda x: os.path.abspath(os.path.join(IMG_DIR, f"{x}.png"))
  )
  # end change
  ```
- **Patch 4** — Insert after L235 (`malignancy_proba = proba_df.iloc[:, 1].values`), before L238 (`save_predictions_with_format(...)`). `<PROB_VAR>` = `malignancy_proba`. Uses `test_df` post-`prepare_test_data` (image_name column was dropped at L120 inside the helper — re-read the test CSV for image_names):
  ```python
  # start change
  _orig_test_names = pd.read_csv(TEST_CSV)["image_name"].astype(str).reset_index(drop=True)
  _ddi_df = pd.DataFrame({
      "DDI_file": _orig_test_names + ".png",
      "predicted_probability": malignancy_proba.astype(float),
  }).reset_index(drop=True)
  _ddi_df.to_csv(os.path.join(OUTPUT_DIR, "DDI_predictions.csv"), index=False)
  # end change
  ```
- **Patch 5** — Shell PY_SCRIPT original: `PY_SCRIPT="/sc-scratch/sc-scratch-ikim-guidlight/be-fair/mlzero/33-basic_prompt/node_8/generated_code.py"`.

---

### 34-basic_prompt

- **Patch 1** — `DATA_DIR` at L46 (inside `if __name__ == "__main__":`). Original: `DATA_DIR = "/sc-scratch/sc-scratch-ikim-guidlight/be-fair/mlzero/basic_prompt_data"`.
- **Patch 2** — `OUTPUT_DIR` at L48. Original: `OUTPUT_DIR = "/sc-scratch/sc-scratch-ikim-guidlight/be-fair/mlzero/34-basic_prompt/node_3/output"`.
- **Patch 3** — Test image at L72. Inline-lambda variant calling helper `get_abs_image_path(x, image_dir)` (L41–42); the helper is shared, so patch only the test line:
  ```python
  # start change
  # test_df['image'] = test_df['image_name'].apply(lambda x: get_abs_image_path(x, IMAGE_DIR))  # original (.jpg)
  test_df['image'] = test_df['image_name'].apply(
      lambda x: os.path.abspath(os.path.join(IMAGE_DIR, f"{x}.png"))
  )
  # end change
  ```
- **Patch 4** — Insert after L129 (`malignancy_prob = np.array(proba).reshape(-1)`), before L134 (`output_df = test_df.copy()`). `<PROB_VAR>` = `malignancy_prob`. Uses `test_df` and `test_df["image_name"]`.
- **Patch 5** — Shell PY_SCRIPT original: `PY_SCRIPT="/sc-scratch/sc-scratch-ikim-guidlight/be-fair/mlzero/34-basic_prompt/node_3/generated_code.py"`.

---

### 35-addition_1

- **Patch 1** — `DATA_DIR` at L40. Original: `DATA_DIR = "/sc-scratch/sc-scratch-ikim-guidlight/be-fair/mlzero/addition_1_data"`.
- **Patch 2** — `OUTPUT_DIR` at L41. Original: `OUTPUT_DIR = "/sc-scratch/sc-scratch-ikim-guidlight/be-fair/mlzero/35-addition_1/node_0/output"`.
- **Patch 3** — Test image at L75. Inline-lambda variant. Column used is `image_path` (not `image`):
  ```python
  # start change
  # test['image_path'] = test['image_name'].apply(lambda x: os.path.join(IMAGES_DIR, f"{x}.jpg"))  # original (.jpg)
  test['image_path'] = test['image_name'].apply(lambda x: os.path.join(IMAGES_DIR, f"{x}.png"))
  # end change
  ```
- **Patch 4** — Insert after L121 (`malignancy_probs = proba_df[1].values`), before L128 (`results = test[['image_name']].copy()`). `<PROB_VAR>` = `malignancy_probs`. Uses `test` and `test["image_name"]`.
- **Patch 5** — Shell PY_SCRIPT original: `PY_SCRIPT="/sc-scratch/sc-scratch-ikim-guidlight/be-fair/mlzero/35-addition_1/node_0/generated_code.py"`.

---

### 36-addition_1

- **Patch 1** — `DATA_DIR` at L53. Original: `DATA_DIR = "/sc-scratch/sc-scratch-ikim-guidlight/be-fair/mlzero/addition_1_data"`.
- **Patch 2** — `OUTPUT_DIR` at L57. Original: `OUTPUT_DIR = "/sc-scratch/sc-scratch-ikim-guidlight/be-fair/mlzero/36-addition_1/node_8/output"`.
- **Patch 3** — Test image: the helper `map_image_name_to_path` (L66–70) **overwrites** the `image_name` column itself with the absolute `.jpg` path. It is called inside `prepare_test_data` (L100) and `prepare_train_data` (L81). Don't patch the helper (it's shared). Instead, insert an override **after** `test_df, test_orig_index = prepare_test_data(...)` on L157 — swap the extension to `.png`:
  ```python
  # start change
  test_df["image_name"] = test_df["image_name"].apply(
      lambda p: os.path.splitext(p)[0] + ".png"
  )
  # end change
  ```
- **Patch 4** — Insert after L209 (`test_pred_df.index = test_orig_index`), before L212 (`results_path = save_results(...)`). The `image_name` column was destroyed by the helper, so re-read the test CSV for original names:
  ```python
  # start change
  _orig_test_names = pd.read_csv(TEST_CSV)["image_name"].astype(str).reset_index(drop=True)
  _ddi_df = pd.DataFrame({
      "DDI_file": _orig_test_names + ".png",
      "predicted_probability": test_pred_df["label"].astype(float).values,
  }).reset_index(drop=True)
  _ddi_df.to_csv(os.path.join(OUTPUT_DIR, "DDI_predictions.csv"), index=False)
  # end change
  ```
  (`test_pred_df["label"]` was set immediately above to the per-class-1 probability.)
- **Patch 5** — Shell PY_SCRIPT original: `PY_SCRIPT="/sc-scratch/sc-scratch-ikim-guidlight/be-fair/mlzero/36-addition_1/node_8/generated_code.py"`.

---

### 37-addition_1

- **Patch 1** — `DATA_DIR` at L36. Original: `DATA_DIR = "/sc-scratch/sc-scratch-ikim-guidlight/be-fair/mlzero/addition_1_data"`.
- **Patch 2** — `OUTPUT_DIR` at L37. Original: `OUTPUT_DIR = "/sc-scratch/sc-scratch-ikim-guidlight/be-fair/mlzero/37-addition_1/node_5/output"`.
- **Patch 3** — Test image at L74. Helper `get_image_path` (L50–52) is shared with train (L73). Override the test line:
  ```python
  # start change
  # test['image'] = test['image_name'].apply(get_image_path)  # original (.jpg)
  test['image'] = test['image_name'].apply(
      lambda x: os.path.abspath(os.path.join(IMAGES_DIR, f"{x}.png"))
  )
  # end change
  ```
- **Patch 4** — Insert after L153 (`malignancy_proba = proba[:, 1]`), before L155 (`results = test[['image_name']].copy()`). `<PROB_VAR>` = `malignancy_proba`. Uses `test` and `test["image_name"]`.
- **Patch 5** — Shell PY_SCRIPT original: `PY_SCRIPT="/sc-scratch/sc-scratch-ikim-guidlight/be-fair/mlzero/37-addition_1/node_5/generated_code.py"`.

---

### 38-addition_1

- **Patch 1** — `DATA_DIR` at L33. Original: `DATA_DIR = "/sc-scratch/sc-scratch-ikim-guidlight/be-fair/mlzero/addition_1_data"`.
- **Patch 2** — `OUTPUT_DIR` at L37. Original: `OUTPUT_DIR = "/sc-scratch/sc-scratch-ikim-guidlight/be-fair/mlzero/38-addition_1/node_6/output"`.
- **Patch 3** — Test image at L64. The script defines a local helper `image_name_to_path` (L60–61) inside `__main__`, used by both train (L63) and test (L64). Override only the test line:
  ```python
  # start change
  # test['image'] = test['image_name'].apply(image_name_to_path)  # original (.jpg)
  test['image'] = test['image_name'].apply(
      lambda x: os.path.abspath(os.path.join(IMG_DIR, f"{x}.png"))
  )
  # end change
  ```
- **Patch 4** — Insert after L138 (`malignancy_prob = proba.iloc[:, -1].values`), before L140 (`results_df = test[['image_name']].copy()`). `<PROB_VAR>` = `malignancy_prob`. Uses `test` and `test["image_name"]`.
- **Patch 5** — Shell PY_SCRIPT original: `PY_SCRIPT="/sc-scratch/sc-scratch-ikim-guidlight/be-fair/mlzero/38-addition_1/node_6/generated_code.py"`.

---

### 39-addition_1

- **Patch 1** — `DATA_DIR` at L42 (inside `if __name__ == "__main__":`). Original: `DATA_DIR = "/sc-scratch/sc-scratch-ikim-guidlight/be-fair/mlzero/addition_1_data"`.
- **Patch 2** — `OUTPUT_DIR` at L43. Original: `OUTPUT_DIR = "/sc-scratch/sc-scratch-ikim-guidlight/be-fair/mlzero/39-addition_1/node_3/output"`.
- **Patch 3** — Test image at L76. Local helper `image_name_to_path` (L72–73) inside `__main__`, used by both train (L75) and test (L76). Override the test line:
  ```python
  # start change
  # test["image"] = test["image_name"].apply(image_name_to_path)  # original (.jpg)
  test["image"] = test["image_name"].apply(
      lambda x: os.path.abspath(os.path.join(IMAGE_DIR, f"{x}.png"))
  )
  # end change
  ```
- **Patch 4** — Insert after L150 (`malignancy_probs = proba_df[1].values`), before L158 (`results = test[["image_name"]].copy()`). `<PROB_VAR>` = `malignancy_probs`. Uses `test` and `test["image_name"]` (preserved through the reorder at L90).
- **Patch 5** — Shell PY_SCRIPT original: `PY_SCRIPT="/sc-scratch/sc-scratch-ikim-guidlight/be-fair/mlzero/39-addition_1/node_3/generated_code.py"`.

---

### 40-addition_1

- **Patch 1** — `DATA_DIR` at L48. Original: `DATA_DIR = "/sc-scratch/sc-scratch-ikim-guidlight/be-fair/mlzero/addition_1_data"`.
- **Patch 2** — `OUTPUT_DIR` at L52. Original: `OUTPUT_DIR = "/sc-scratch/sc-scratch-ikim-guidlight/be-fair/mlzero/40-addition_1/node_10/output"`.
- **Patch 3** — Test image at L102. Inline-lambda variant:
  ```python
  # start change
  # test['image'] = test['image_name'].apply(lambda x: os.path.abspath(os.path.join(IMG_DIR, f"{x}.jpg")))  # original (.jpg)
  test['image'] = test['image_name'].apply(
      lambda x: os.path.abspath(os.path.join(IMG_DIR, f"{x}.png"))
  )
  # end change
  ```
- **Patch 4** — Insert after L187 (`test_malignant_proba = test_pred_proba[str(1)].values`), before L190 (`output_df = test.copy()`). `<PROB_VAR>` = `test_malignant_proba`. Uses `test` and `test["image_name"]`.
- **Patch 5** — Shell PY_SCRIPT original: `PY_SCRIPT="/sc-scratch/sc-scratch-ikim-guidlight/be-fair/mlzero/40-addition_1/node_10/generated_code.py"`.

---

### 41-addition_1

- **Patch 1** — `DATA_DIR` at L56 (inside `if __name__ == "__main__":`). Original: `DATA_DIR = "/sc-scratch/sc-scratch-ikim-guidlight/be-fair/mlzero/addition_1_data"`.
- **Patch 2** — `OUTPUT_DIR` at L57. Original: `OUTPUT_DIR = "/sc-scratch/sc-scratch-ikim-guidlight/be-fair/mlzero/41-addition_1/node_4/output"`.
- **Patch 3** — Test image at L90. Local helper `image_name_to_path` (L86–87), shared. Override the test line:
  ```python
  # start change
  # test['image'] = test['image_name'].apply(image_name_to_path)  # original (.jpg)
  test['image'] = test['image_name'].apply(
      lambda x: os.path.abspath(os.path.join(IMAGE_DIR, f"{x}.png"))
  )
  # end change
  ```
- **Patch 4** — Insert after L171 (`malignancy_prob = proba.iloc[:, -1]`), before L174 (`output_df = test[['image_name']].copy()`). `<PROB_VAR>` = `malignancy_prob` (pandas Series — `.values` first or just use `.astype(float)`). Uses `test` and `test["image_name"]` (preserved by the reorder at L100):
  ```python
  # start change
  _ddi_df = pd.DataFrame({
      "DDI_file": test["image_name"].astype(str) + ".png",
      "predicted_probability": malignancy_prob.astype(float).values,
  }).reset_index(drop=True)
  _ddi_df.to_csv(os.path.join(OUTPUT_DIR, "DDI_predictions.csv"), index=False)
  # end change
  ```
- **Patch 5** — Shell PY_SCRIPT original: `PY_SCRIPT="/sc-scratch/sc-scratch-ikim-guidlight/be-fair/mlzero/41-addition_1/node_4/generated_code.py"`.

---

### 42-addition_2

- **Patch 1** — `DATA_DIR` at L31. Original: `DATA_DIR = "/sc-scratch/sc-scratch-ikim-guidlight/be-fair/mlzero/addition_2_data"`.
- **Patch 2** — `OUTPUT_DIR` at L35. Original: `OUTPUT_DIR = "/sc-scratch/sc-scratch-ikim-guidlight/be-fair/mlzero/42-addition_2/node_11/output"`.
- **Patch 3** — Test image at L62. Local helper `image_path_fn` (L59–60), shared with train (L61). Override the test line:
  ```python
  # start change
  # test['image'] = test['image_name'].apply(image_path_fn)  # original (.jpg)
  test['image'] = test['image_name'].apply(
      lambda x: os.path.abspath(os.path.join(IMAGE_DIR, f"{x}.png"))
  )
  # end change
  ```
- **Patch 4** — Insert after L130 (`malignancy_prob = y_pred_proba.iloc[:, 1].values`), before L133 (`results = test.copy()`). `<PROB_VAR>` = `malignancy_prob`. Uses `test` and `test["image_name"]`.
- **Patch 5** — Shell PY_SCRIPT original: `PY_SCRIPT="/sc-scratch/sc-scratch-ikim-guidlight/be-fair/mlzero/42-addition_2/node_11/generated_code.py"`.

---

### 43-addition_2

- **Patch 1** — `DATA_DIR` at L33 (inside `if __name__ == "__main__":`). Original: `DATA_DIR = "/sc-scratch/sc-scratch-ikim-guidlight/be-fair/mlzero/addition_2_data"`.
- **Patch 2** — `OUTPUT_DIR` at L37. Original: `OUTPUT_DIR = "/sc-scratch/sc-scratch-ikim-guidlight/be-fair/mlzero/43-addition_2/node_11/output"`.
- **Patch 3** — Test image at L63. Local helper `image_name_to_path` (L59–60), shared with train (L62). Override the test line:
  ```python
  # start change
  # test["image"] = test["image_name"].apply(image_name_to_path)  # original (.jpg)
  test["image"] = test["image_name"].apply(
      lambda x: os.path.abspath(os.path.join(IMG_DIR, f"{x}.png"))
  )
  # end change
  ```
- **Patch 4** — Quirk: this script uses string labels (`problem_type="classification"`). The malignancy probability lives in `proba["malignant"]`. Insert after L150 (`output_df["label"] = proba["malignant"].astype(float)`), before L153 (`assert len(output_df) == len(test)`):
  ```python
  # start change
  _ddi_df = pd.DataFrame({
      "DDI_file": test["image_name"].astype(str) + ".png",
      "predicted_probability": proba["malignant"].astype(float).values,
  }).reset_index(drop=True)
  _ddi_df.to_csv(os.path.join(OUTPUT_DIR, "DDI_predictions.csv"), index=False)
  # end change
  ```
- **Patch 5** — Shell PY_SCRIPT original: `PY_SCRIPT="/sc-scratch/sc-scratch-ikim-guidlight/be-fair/mlzero/43-addition_2/node_11/generated_code.py"`.

---

### 45-addition_2

- **Patch 1** — `DATA_DIR` at L40. Original: `DATA_DIR = "/sc-scratch/sc-scratch-ikim-guidlight/be-fair/mlzero/addition_2_data"`.
- **Patch 2** — `OUTPUT_DIR` at L44. Original: `OUTPUT_DIR = "/sc-scratch/sc-scratch-ikim-guidlight/be-fair/mlzero/45-addition_2/node_0/output"`.
- **Patch 3** — Test image at L122. Like script 36, the helper `map_image_names_to_paths` (L55–59) **overwrites** the `image_name` column itself, hardcoding `.jpg`. The same helper is used by train at L121. Don't patch the helper; instead, insert an extension swap right after L122:
  ```python
  # start change
  test_df["image_name"] = test_df["image_name"].apply(
      lambda p: os.path.splitext(p)[0] + ".png"
  )
  # end change
  ```
- **Patch 4** — Insert after L194 (`malignancy_proba = proba.iloc[:, -1].values`), before L200 (`results_df = pd.DataFrame({...}, index=orig_indices)`). The `image_name` column was destroyed — re-read `test.csv` to recover names:
  ```python
  # start change
  _orig_test_names = pd.read_csv(TEST_CSV)["image_name"].astype(str).reset_index(drop=True)
  _ddi_df = pd.DataFrame({
      "DDI_file": _orig_test_names + ".png",
      "predicted_probability": malignancy_proba.astype(float),
  }).reset_index(drop=True)
  _ddi_df.to_csv(os.path.join(OUTPUT_DIR, "DDI_predictions.csv"), index=False)
  # end change
  ```
- **Patch 5** — Shell PY_SCRIPT original: `PY_SCRIPT="/sc-scratch/sc-scratch-ikim-guidlight/be-fair/mlzero/45-addition_2/node_0/generated_code.py"`.

---

### 46-addition_2

- **Patch 1** — `DATA_DIR` at L183 (inside `if __name__ == "__main__":`). Original: `DATA_DIR = "/sc-scratch/sc-scratch-ikim-guidlight/be-fair/mlzero/addition_2_data"`.
- **Patch 2** — `OUTPUT_DIR` at L184. Original: `OUTPUT_DIR = "/sc-scratch/sc-scratch-ikim-guidlight/be-fair/mlzero/46-addition_2/node_4/output"`.
- **Patch 3** — Test image inside `prepare_test_data` at L79. The helper `get_image_path` (L27–29) hardcodes `.jpg` and is shared (train uses it via `prepare_train_data` at L72). Patch only the line inside `prepare_test_data` (it never runs on train data):
  ```python
  # start change
  # test_df["image"] = test_df["image_name"].apply(lambda x: get_image_path(x, image_dir))  # original (.jpg)
  test_df["image"] = test_df["image_name"].apply(
      lambda x: os.path.abspath(os.path.join(image_dir, f"{x}.png"))
  )
  # end change
  ```
- **Patch 4** — Insert after L276 (`proba = clf.predict_proba(X_test)[:, 1]`), before L279 (`ext, sep = infer_test_file_format(TEST_CSV)`). `<PROB_VAR>` = `proba`. Uses `test_df` and `test_df["image_name"]` (preserved by `prepare_test_data`, which only adds a separate `image` column).
- **Patch 5** — Shell PY_SCRIPT original: `PY_SCRIPT="/sc-scratch/sc-scratch-ikim-guidlight/be-fair/mlzero/46-addition_2/node_4/generated_code.py"`.

---

### 47-addition_2

- **Patch 1** — `DATA_DIR` at L34. Original: `DATA_DIR = "/sc-scratch/sc-scratch-ikim-guidlight/be-fair/mlzero/addition_2_data"`.
- **Patch 2** — `OUTPUT_DIR` at L38. Original: `OUTPUT_DIR = "/sc-scratch/sc-scratch-ikim-guidlight/be-fair/mlzero/47-addition_2/node_16/output"`.
- **Patch 3** — Test image at L80. Helper `get_image_path` (L42–44) is shared with train (L79). Override the test line:
  ```python
  # start change
  # test['image'] = test['image_name'].apply(get_image_path)  # original (.jpg)
  test['image'] = test['image_name'].apply(
      lambda x: os.path.abspath(os.path.join(IMG_DIR, f"{x}.png"))
  )
  # end change
  ```
- **Patch 4** — Quirk: this script uses string labels (`problem_type="classification"`). The malignancy probability lives in `proba_df["malignant"]`, captured at L158 as `malignancy_proba`. Insert after L160 (`malignancy_proba = proba_df.iloc[:, -1]`), before L164 (`results = test[['image_name']].copy()`):
  ```python
  # start change
  _ddi_df = pd.DataFrame({
      "DDI_file": test["image_name"].astype(str) + ".png",
      "predicted_probability": malignancy_proba.astype(float).values,
  }).reset_index(drop=True)
  _ddi_df.to_csv(os.path.join(OUTPUT_DIR, "DDI_predictions.csv"), index=False)
  # end change
  ```
- **Patch 5** — Shell PY_SCRIPT original: `PY_SCRIPT="/sc-scratch/sc-scratch-ikim-guidlight/be-fair/mlzero/47-addition_2/node_16/generated_code.py"`.

---

### 48-addition_2 — SPECIAL CASE

This script is the only one where the test image path **cannot** be patched in a single line. The helper `get_image_path` (L57–59) is called **inside** `extract_features_df` (L86) which is used **identically for train, val, and test** (L139, L143, L147). The cleanest minimal fix is to parameterize the extension and pass `ext=".png"` **only** at the test call site.

- **Patch 1** — `DATA_DIR` at L47. Original: `DATA_DIR = "/sc-scratch/sc-scratch-ikim-guidlight/be-fair/mlzero/addition_2_data"`.
- **Patch 2** — `OUTPUT_DIR` at L51. Original: `OUTPUT_DIR = "/sc-scratch/sc-scratch-ikim-guidlight/be-fair/mlzero/48-addition_2/node_4/output"`.
- **Patch 3** — three-part patch:

  (a) Add an `ext` kwarg to `get_image_path` (L57–59):
  ```python
  # start change
  # def get_image_path(image_name):
  #     # Returns absolute path to image file given image_name (without extension)
  #     return os.path.join(IMAGE_DIR, f"{image_name}.jpg")
  def get_image_path(image_name, ext=".jpg"):
      return os.path.join(IMAGE_DIR, f"{image_name}{ext}")
  # end change
  ```

  (b) Add an `ext` kwarg to `extract_features_df` and forward it (L80, L86):
  ```python
  # start change
  # def extract_features_df(df, resize=(128, 128)):
  def extract_features_df(df, resize=(128, 128), ext=".jpg"):
  # end change
  ```
  and inside it (L86):
  ```python
  # start change
  # image_paths = df['image_name'].apply(get_image_path).tolist()
  image_paths = df['image_name'].apply(lambda x: get_image_path(x, ext=ext)).tolist()
  # end change
  ```

  (c) At the **test** call site only (L147), pass `ext=".png"`:
  ```python
  # start change
  # X_test = extract_features_df(test)  # original (.jpg)
  X_test = extract_features_df(test, ext=".png")
  # end change
  ```
  Leave L139 (`X_train = extract_features_df(train_data)`) and L143 (`X_val = extract_features_df(val_data)`) **unchanged** — they default to `.jpg`.

- **Patch 4** — Insert after L172 (`proba = clf.predict_proba(X_test)[:, 1]`), before L175 (`results_df = test[['image_name']].copy()`). `<PROB_VAR>` = `proba`. Uses `test` and `test["image_name"]`.
- **Patch 5** — Shell PY_SCRIPT original: `PY_SCRIPT="/sc-scratch/sc-scratch-ikim-guidlight/be-fair/mlzero/48-addition_2/node_4/generated_code.py"`.

---

### 49-addition_3

- **Patch 1** — `DATA_DIR` at L42. Original: `DATA_DIR = "/sc-scratch/sc-scratch-ikim-guidlight/be-fair/mlzero/addition_3_data"`.
- **Patch 2** — `OUTPUT_DIR` at L46. Original: `OUTPUT_DIR = "/sc-scratch/sc-scratch-ikim-guidlight/be-fair/mlzero/49-addition_3/node_7/output"`.
- **Patch 3** — Test image at L187. Helper `get_image_path` (L51–53) is shared with train (L186). Override the test line:
  ```python
  # start change
  # test_df["image"] = test_df["image_name"].apply(get_image_path)  # original (.jpg)
  test_df["image"] = test_df["image_name"].apply(
      lambda x: os.path.abspath(os.path.join(IMG_DIR, f"{x}.png"))
  )
  # end change
  ```
- **Patch 4** — Insert after L266 (`test_pred = np.clip(test_pred, 0, 1)`), before L269 (`result_df = test_df_orig.copy()`). `<PROB_VAR>` = `test_pred`. Uses `test_df` and `test_df["image_name"]` (or equivalently `test_df_orig["image_name"]`).
- **Patch 5** — Shell PY_SCRIPT original: `PY_SCRIPT="/sc-scratch/sc-scratch-ikim-guidlight/be-fair/mlzero/49-addition_3/node_7/generated_code.py"`.

---

### 50-addition_3

- **Patch 1** — `DATA_DIR` at L38. Original: `DATA_DIR = "/sc-scratch/sc-scratch-ikim-guidlight/be-fair/mlzero/addition_3_data"`.
- **Patch 2** — `OUTPUT_DIR` at L39. Original: `OUTPUT_DIR = "/sc-scratch/sc-scratch-ikim-guidlight/be-fair/mlzero/50-addition_3/node_5/output"`.
- **Patch 3** — Test image **inside `preprocess_test`** at L69. Helper `get_image_path` (L44–46) is shared with `preprocess_train` (L57). Patch only the line inside `preprocess_test`:
  ```python
  # start change
  # test_df['image'] = test_df['image_name'].apply(get_image_path)  # original (.jpg)
  test_df['image'] = test_df['image_name'].apply(
      lambda x: os.path.join(IMAGES_DIR, f"{x}.png")
  )
  # end change
  ```
- **Patch 4** — Insert after L197 (`malignancy_prob = test_pred_proba["1"]`), before L200 (`pred_df = pd.DataFrame({...})`). `<PROB_VAR>` = `malignancy_prob` (pandas Series). Uses `test_df` and `test_df["image_name"]`:
  ```python
  # start change
  _ddi_df = pd.DataFrame({
      "DDI_file": test_df["image_name"].astype(str) + ".png",
      "predicted_probability": malignancy_prob.astype(float).values,
  }).reset_index(drop=True)
  _ddi_df.to_csv(os.path.join(OUTPUT_DIR, "DDI_predictions.csv"), index=False)
  # end change
  ```
- **Patch 5** — Shell PY_SCRIPT original: `PY_SCRIPT="/sc-scratch/sc-scratch-ikim-guidlight/be-fair/mlzero/50-addition_3/node_5/generated_code.py"`.

---

### 52-addition_3

- **Patch 1** — `DATA_DIR` at L56. Original: `DATA_DIR = "/sc-scratch/sc-scratch-ikim-guidlight/be-fair/mlzero/addition_3_data"`.
- **Patch 2** — `OUTPUT_DIR` at L60. Original: `OUTPUT_DIR = "/sc-scratch/sc-scratch-ikim-guidlight/be-fair/mlzero/52-addition_3/node_7/output"`.
- **Patch 3** — Test image at L87. Local helper `image_path_fn` (L83–84) inside `if __name__ == "__main__":`, shared with train (L86). Override the test line:
  ```python
  # start change
  # test['image'] = test['image_name'].apply(image_path_fn)  # original (.jpg)
  test['image'] = test['image_name'].apply(
      lambda x: os.path.abspath(os.path.join(IMG_DIR, f"{x}.png"))
  )
  # end change
  ```
- **Patch 4** — Insert after L160 (`malignancy_prob = proba.iloc[:, 1].values`), before L164 (`results = test.copy()`). `<PROB_VAR>` = `malignancy_prob`. Uses `test` and `test["image_name"]`.
- **Patch 5** — Shell PY_SCRIPT original: `PY_SCRIPT="/sc-scratch/sc-scratch-ikim-guidlight/be-fair/mlzero/52-addition_3/node_7/generated_code.py"`.

---

### 53-addition_3

- **Patch 1** — `DATA_DIR` at L48. Original: `DATA_DIR = "/sc-scratch/sc-scratch-ikim-guidlight/be-fair/mlzero/addition_3_data"`.
- **Patch 2** — `OUTPUT_DIR` at L49. Original: `OUTPUT_DIR = "/sc-scratch/sc-scratch-ikim-guidlight/be-fair/mlzero/53-addition_3/node_18/output"` (note: the `execution_script.sh` references `node_19`, but the Python script's own internal path string uses `node_18` — preserve whatever appears on line 49 of the Python file in the commented-out original line).
- **Patch 3** — Test image at L118. Helper `get_absolute_image_path` (L54–56) is shared with train (L117). Override the test line:
  ```python
  # start change
  # test_df["image"] = test_df["image_name"].apply(get_absolute_image_path)  # original (.jpg)
  test_df["image"] = test_df["image_name"].apply(
      lambda x: os.path.abspath(os.path.join(IMAGES_DIR, f"{x}.png"))
  )
  # end change
  ```
- **Patch 4** — Insert after L188 (`test_pred_df[label_col] = proba["1"]`), before L191 (`output_df = test_pred_df[[label_col]].copy()`). `<PROB_VAR>` = `test_pred_df["label"]` (assigned by the branch above). Uses `test_df` for the original `image_name`:
  ```python
  # start change
  _ddi_df = pd.DataFrame({
      "DDI_file": test_df["image_name"].astype(str) + ".png",
      "predicted_probability": test_pred_df["label"].astype(float).values,
  }).reset_index(drop=True)
  _ddi_df.to_csv(os.path.join(OUTPUT_DIR, "DDI_predictions.csv"), index=False)
  # end change
  ```
- **Patch 5** — Shell PY_SCRIPT original (read from `execution_script.sh`): `PY_SCRIPT="/sc-scratch/sc-scratch-ikim-guidlight/be-fair/mlzero/53-addition_3/node_19/generated_code.py"`.

---

### 54-addition_3

- **Patch 1** — `DATA_DIR` at L37. Original: `DATA_DIR = "/sc-scratch/sc-scratch-ikim-guidlight/be-fair/mlzero/addition_3_data"`.
- **Patch 2** — `OUTPUT_DIR` at L38. Original: `OUTPUT_DIR = "/sc-scratch/sc-scratch-ikim-guidlight/be-fair/mlzero/54-addition_3/node_15/output"`.
- **Patch 3** — Test image at L182. Helper `get_image_path` (L49–51) is shared with train (L181). Override the test line:
  ```python
  # start change
  # test['image'] = test['image_name'].apply(get_image_path)  # original (.jpg)
  test['image'] = test['image_name'].apply(
      lambda x: os.path.join(IMAGES_DIR, f"{x}.png")
  )
  # end change
  ```
- **Patch 4** — Insert after L244 (`malignancy_prob = predict_proba(model, test_loader, DEVICE)`), before L247 (`results = test[['image_name']].copy()`). `<PROB_VAR>` = `malignancy_prob` (numpy array). Uses `test` and `test["image_name"]`.
- **Patch 5** — Shell PY_SCRIPT original: `PY_SCRIPT="/sc-scratch/sc-scratch-ikim-guidlight/be-fair/mlzero/54-addition_3/node_15/generated_code.py"`.

---

### 55-addition_3

- **Patch 1** — `DATA_DIR` at L35. Original: `DATA_DIR = "/sc-scratch/sc-scratch-ikim-guidlight/be-fair/mlzero/addition_3_data"`.
- **Patch 2** — `OUTPUT_DIR` at L36. Original: `OUTPUT_DIR = "/sc-scratch/sc-scratch-ikim-guidlight/be-fair/mlzero/55-addition_3/node_7/output"`.
- **Patch 3** — Test image at L172. Helper `get_absolute_image_path` (L45–47) is shared with train (L171). Override the test line:
  ```python
  # start change
  # test["image"] = test["image_name"].apply(get_absolute_image_path)  # original (.jpg)
  test["image"] = test["image_name"].apply(
      lambda x: os.path.abspath(os.path.join(IMAGE_DIR, f"{x}.png"))
  )
  # end change
  ```
- **Patch 4** — Insert after L246 (`malignancy_prob = model.predict_proba(X_test)[:, 1]`), before L249 (`results = test[["image_name"]].copy()`). `<PROB_VAR>` = `malignancy_prob`. Uses `test` and `test["image_name"]`.
- **Patch 5** — Shell PY_SCRIPT original: `PY_SCRIPT="/sc-scratch/sc-scratch-ikim-guidlight/be-fair/mlzero/55-addition_3/node_7/generated_code.py"`.

---

## Summary checklist

| Folder | Test DataFrame | Prob variable | Quirks / notes |
|--------|----------------|---------------|----------------|
| 28-basic_prompt | `test` | `malignancy_probs` (np array) | inline-lambda Patch 3 |
| 29-basic_prompt | `test_df` | `test_probs` (Python list — wrap with `np.asarray`) | shared helper |
| 30-basic_prompt | `test` / `test_for_pred` | `malignancy_prob` (np array) | Patch 3 needs to be applied at two call sites (L84 and L129) |
| 31-basic_prompt | `test_df` | `malignancy_probs` (np array) | shared helper `_ensure_absolute_image_paths` |
| 32-basic_prompt | `test` | `malignancy_prob` (np array) | shared helper |
| 33-basic_prompt | `test_df` | `malignancy_proba` (np array) | `image_name` column dropped inside `prepare_test_data` — re-read `test.csv` for DDI save |
| 34-basic_prompt | `test_df` | `malignancy_prob` (np array) | inline-lambda Patch 3 |
| 35-addition_1 | `test` | `malignancy_probs` (np array) | column is `image_path` (not `image`); inline-lambda Patch 3 |
| 36-addition_1 | `test_df` (via `test_pred_df`) | `test_pred_df["label"]` (after L200) | helper **overwrites** `image_name` — re-read `test.csv` for DDI save |
| 37-addition_1 | `test` | `malignancy_proba` (np array) | shared helper |
| 38-addition_1 | `test` | `malignancy_prob` (np array) | local helper |
| 39-addition_1 | `test` | `malignancy_probs` (np array) | local helper; `test` is reindexed but keeps `image_name` |
| 40-addition_1 | `test` | `test_malignant_proba` (np array) | inline-lambda Patch 3 |
| 41-addition_1 | `test` | `malignancy_prob` (Series — `.astype(float).values`) | local helper |
| 42-addition_2 | `test` | `malignancy_prob` (np array) | local helper |
| 43-addition_2 | `test` | `proba["malignant"]` (Series) | string labels; `problem_type="classification"` |
| 45-addition_2 | `test_df` | `malignancy_proba` (np array) | helper **overwrites** `image_name` — re-read `test.csv` for DDI save |
| 46-addition_2 | `test_df` | `proba` (np array, single-column probabilities) | Patch 3 inside `prepare_test_data` |
| 47-addition_2 | `test` | `malignancy_proba` (Series) | string labels; `problem_type="classification"` |
| **48-addition_2** | `test` | `proba` (np array) | **SPECIAL CASE** — `extract_features_df`/`get_image_path` parameterized with `ext` kwarg; pass `ext=".png"` only at the test call site |
| 49-addition_3 | `test_df` | `test_pred` (np array) | shared helper |
| 50-addition_3 | `test_df` | `malignancy_prob` (Series) | Patch 3 inside `preprocess_test` |
| 52-addition_3 | `test` | `malignancy_prob` (np array) | local helper |
| 53-addition_3 | `test_df` | `test_pred_df["label"]` (Series after assignment) | shell script references `node_19` but the Python file's own constant says `node_18` — preserve verbatim per file |
| 54-addition_3 | `test` | `malignancy_prob` (np array) | shared helper |
| 55-addition_3 | `test` | `malignancy_prob` (np array) | shared helper |

---

## Out of scope (flagged here, not implemented)

- Setting up the contents of `/sc-scratch/sc-scratch-ikim-guidlight/be-fair/evaluation/evaluation_data/` (the user prepares this directory; `train.csv`, `test.csv`, and `MyImages/` are assumed to already exist there).
- Any line of `execution_script.sh` other than `PY_SCRIPT=` — env setup, requirements installs, conda activation, etc. all stay untouched.
- Validation that the patched scripts actually run on the cluster.
