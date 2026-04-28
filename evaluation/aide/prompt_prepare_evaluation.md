# Agent Task: Prepare Evaluation Scripts

## What you need to do

For every subfolder in `/Users/arichter/Documents/GitHub/be-fair/evaluation/aide/` that contains a `best_solution.py`, first copy that file to a new file named `{foldername}_best_solution_DDI.py` in the **same subfolder**. All changes described in this document must be applied **only to the copy**. Never modify the original `best_solution.py`.

Example: for the folder `0-basic_prompt`, copy `0-basic_prompt/best_solution.py` → `0-basic_prompt/0-basic_prompt_best_solution_DDI.py`, then apply all changes to `0-basic_prompt/0-basic_prompt_best_solution_DDI.py`.

Then edit each `{foldername}_best_solution_DDI.py` so that it:

1. Reads training data from the correct absolute paths (instead of the relative `./input/` placeholders left by the AIDE training framework).
2. Calls `predict()` on the DDI test image folder after training.
3. Saves predictions to `./DDI_predictions.csv` in the script's own directory with exactly 2 columns: `DDI_file` and `predicted_probability` (float 0–1), rows ordered by `sorted(os.listdir(DDI_folder))`.

**Mode required:** Agent mode (not Ask or Plan mode) — you will be making file edits.

> **Important:** never touch `best_solution.py` itself. Every edit goes into the `{foldername}_best_solution_DDI.py` copy.

---

## Repository context

Scripts live at:
```
evaluation/aide/
  0-basic_prompt/best_solution.py
  1-basic_prompt/best_solution.py
  2-basic_prompt/best_solution.py
  3-basic_prompt/best_solution.py
  4-basic_prompt/best_solution.py
  5-basic_prompt/best_solution.py
  6-basic_prompt/best_solution.py
  7-addition_1/best_solution.py
  8-addition_1/best_solution.py
  9-addition_1/best_solution.py
  10-addition_1/best_solution.py
  11-addition_1/best_solution.py
  12-addition_1/best_solution.py
  13-addition_1/best_solution.py
  14-addition_2/best_solution.py
  15-addition_2/best_solution.py
  16-addition_2/best_solution.py
  17-addition_2/best_solution.py
  18-addition_2/best_solution.py
  19-addition_2/best_solution.py
  20-addition_2/best_solution.py
  21-addition_3/best_solution.py
  22-addition_3/best_solution.py
  23-addition_3/best_solution.py
  24-addition_3/best_solution.py   ← INCOMPLETE, skip or handle manually
  25-addition_3/best_solution.py
  26-addition_3/best_solution.py
  27-addition_3/best_solution.py
```

Each script is run by SLURM via `run_evaluation.sh` which does `cd N-scriptname && python best_solution.py`. The working directory when the script runs is the script's own subfolder, so `./working/` and `./DDI_predictions.csv` are relative to e.g. `evaluation/aide/0-basic_prompt/`.

---

## Reference paths

| Name | Absolute path |
|------|--------------|
| Training CSV | `/home/anri21/be-fair/aide/MyData/mydataset.csv` |
| Training images | `/home/anri21/be-fair/aide/MyData/MyImages` |
| DDI test images | `/home/anri21/be-fair/evaluation/DDI/images` |

DDI images are all `.png` files (`000001.png`, `000002.png`, …). **Any `predict()` function that only filters for `.jpg` will return empty results and must be fixed.**

---

## Editing conventions

### Rule 1 — mark every change
Wrap every changed block in `# start change` / `# end change`. Comment out the old line rather than deleting it.

```python
# start change
# OLD_VAR = "input/mydataset.csv"  # original
OLD_VAR = "/home/anri21/be-fair/aide/MyData/mydataset.csv"
# end change
```

### Rule 2 — extension filter fix
For any `predict()` whose file-listing only accepts `.jpg`, also accept `.png`. Example:

```python
# start change
# if fname.lower().endswith(".jpg"):  # original
if fname.lower().endswith((".jpg", ".png")):
# end change
```

The same pattern applies inside list comprehensions:
```python
# start change
# [f for f in os.listdir(folder) if f.lower().endswith(".jpg")]  # original
[f for f in os.listdir(folder) if f.lower().endswith((".jpg", ".png"))]
# end change
```

### Rule 3 — DDI save templates
After calling `predict()` with the DDI folder, build and save `./DDI_predictions.csv`.
Use the template that matches what `predict()` returns.
Substitute the literal path `"/home/anri21/be-fair/evaluation/DDI/images"` for `TEST_IMG`.

**Template D** — `predict()` returns a `dict` with filename-WITH-extension keys (`000001.png`):
```python
# start change
_ddi_files = sorted(os.listdir("/home/anri21/be-fair/evaluation/DDI/images"))
_preds = predict("/home/anri21/be-fair/evaluation/DDI/images")
pd.DataFrame({
    "DDI_file": _ddi_files,
    "predicted_probability": [float(_preds[f]) for f in _ddi_files]
}).to_csv("./DDI_predictions.csv", index=False)
# end change
```

**Template D2** — `predict()` returns a `dict` with filename-WITHOUT-extension keys (`000001`):
```python
# start change
_ddi_files = sorted(os.listdir("/home/anri21/be-fair/evaluation/DDI/images"))
_preds = predict("/home/anri21/be-fair/evaluation/DDI/images")
pd.DataFrame({
    "DDI_file": _ddi_files,
    "predicted_probability": [float(_preds[os.path.splitext(f)[0]]) for f in _ddi_files]
}).to_csv("./DDI_predictions.csv", index=False)
# end change
```

**Template F** — `predict()` returns a `DataFrame` whose first column is the filename WITH extension:
```python
# start change
_ddi_files = sorted(os.listdir("/home/anri21/be-fair/evaluation/DDI/images"))
_sub = predict("/home/anri21/be-fair/evaluation/DDI/images")
_pmap = dict(zip(_sub.iloc[:, 0], _sub.iloc[:, 1].astype(float)))
pd.DataFrame({
    "DDI_file": _ddi_files,
    "predicted_probability": [_pmap[f] for f in _ddi_files]
}).to_csv("./DDI_predictions.csv", index=False)
# end change
```

**Template F2** — `predict()` returns a `DataFrame` whose first column is the filename WITHOUT extension:
```python
# start change
_ddi_files = sorted(os.listdir("/home/anri21/be-fair/evaluation/DDI/images"))
_sub = predict("/home/anri21/be-fair/evaluation/DDI/images")
_pmap = dict(zip(_sub.iloc[:, 0], _sub.iloc[:, 1].astype(float)))
pd.DataFrame({
    "DDI_file": _ddi_files,
    "predicted_probability": [_pmap[os.path.splitext(f)[0]] for f in _ddi_files]
}).to_csv("./DDI_predictions.csv", index=False)
# end change
```

---

## Per-script instructions

Scripts marked **[ext fix]** require the extension filter fix inside `predict()` (Rule 2).
All scripts require the training path replacement (Rule 1) and a DDI save block (Rule 3).

---

### 0-basic_prompt/best_solution.py

**Training paths to replace:**
- `DATA_CSV = "input/mydataset.csv"` (near top of `main()`)
- `IMG_DIR = "input/MyImages"` (near top of `main()`)

**[ext fix]** Inside `predict()`, the line:
```python
files = sorted([f for f in os.listdir(folder_path) if f.lower().endswith(".jpg")])
```
→ change to `endswith((".jpg", ".png"))`.

**Predict call change:** `predict()` has signature `predict(folder_path, model, device, mean, std)`. Near the end of `main()`, change:
```python
predict(IMG_DIR, model, DEVICE, mean, std)
```
to `predict("/home/anri21/be-fair/evaluation/DDI/images", model, DEVICE, mean, std)`.

`predict()` already saves `working/submission.csv` internally. Add **Template F2** after the call.
(`predict()` returns DataFrame, first col = `image_name` WITHOUT ext, second col = `malignancy_prob`)

---

### 1-basic_prompt/best_solution.py

**Training paths to replace:**
- `DATA_CSV = "./input/mydataset.csv"` (module-level constant)
- `IMAGES_DIR = "./input/MyImages"` (module-level constant)

**[ext fix]** Inside `predict()`, the line:
```python
img_files = [f for f in os.listdir(image_folder) if f.lower().endswith(".jpg")]
```
→ change to `endswith((".jpg", ".png"))`.

**Predict call:** No predict call exists. After `train_and_evaluate()` at the bottom of `__main__`, add **Template D**.
(`predict()` returns dict, keys = filenames WITH ext)

---

### 2-basic_prompt/best_solution.py

**Training paths to replace:**
- `df = pd.read_csv("input/mydataset.csv")` (inside `main()`)
- `img_dir = os.path.join("input", "MyImages")` (inside `main()`)

**[ext fix]** Inside `predict()`:
```python
if not fname.lower().endswith(".jpg"):
```
→ change to `endswith((".jpg", ".png"))`.

**Predict call:** No predict call. After `main()` call in `__main__`, add **Template F**.
(`predict()` returns DataFrame, first col = `image_name` WITH ext, second col = `malignancy_prob`)

---

### 3-basic_prompt/best_solution.py

**Training paths to replace:**
- `df = pd.read_csv("input/mydataset.csv")` (inside `main()`)
- `img_dir = "input/MyImages"` (inside `main()`)

**[ext fix]** Inside `predict()`:
```python
if not fname.lower().endswith(".jpg"):
```
→ change to `endswith((".jpg", ".png"))`.

**Predict call:** No predict call. After `main()` call in `__main__`, add **Template D**.
(`predict()` returns dict, keys = filenames WITH ext; model saves to `working/model.pt`)

---

### 4-basic_prompt/best_solution.py

**Training paths to replace:**
- `data_csv = "./input/mydataset.csv"` (inside `main()`)
- `img_dir = "./input/MyImages"` (inside `main()`)

**[ext fix]** Inside `predict()`, change the `endswith(".jpg")` filter to `endswith((".jpg", ".png"))`.

**Predict call:** Near end of `main()` there is:
```python
test_folder = "./input/test_images"
if os.path.isdir(test_folder):
    df_sub = predict(model_path, test_folder, device=device)
    df_sub.to_csv("./working/submission.csv", index=False)
```
Change to (unconditional, wrapped in `# start change` / `# end change`):
```python
_ddi_files = sorted(os.listdir("/home/anri21/be-fair/evaluation/DDI/images"))
_sub = predict(model_path, "/home/anri21/be-fair/evaluation/DDI/images", device=device)
_pmap = dict(zip(_sub.iloc[:, 0], _sub.iloc[:, 1].astype(float)))
pd.DataFrame({
    "DDI_file": _ddi_files,
    "predicted_probability": [_pmap[os.path.splitext(f)[0]] for f in _ddi_files]
}).to_csv("./DDI_predictions.csv", index=False)
```
(`predict()` returns DataFrame, first col = `image_name` WITHOUT ext via `fname.rsplit(".", 1)[0]`)
This is **Template F2** adapted for the `predict(model_path, folder, device)` signature.

---

### 5-basic_prompt/best_solution.py

**Training paths to replace:**
- `csv_path = "input/mydataset.csv"` (inside `main()`)
- `img_dir = "input/MyImages"` (inside `main()`)

No ext fix needed — `predict()` already handles `(".jpg", ".png", ".jpeg")`.

**Predict call:** No predict call at end (model saves to `working/model.pth`). After `main()` call in `__main__`, add **Template F2**.
(`predict()` returns DataFrame, first col = `image_name` WITHOUT ext, also internally saves `working/submission.csv`)

---

### 6-basic_prompt/best_solution.py

**Training paths to replace:**
- `csv_path = os.path.join("input", "mydataset.csv")` (inside `main()`)
- `img_dir = os.path.join("input", "MyImages")` (inside `main()`)

No ext fix needed — `predict()` already handles `(".jpg", ".jpeg", ".png")`.

**Predict call:** Near end of `main()`:
```python
test_folder = os.path.join("input", "test")
if os.path.exists(test_folder):
    sub = predict(test_folder)
    sub.to_csv(os.path.join("working", "submission.csv"), index=False)
```
Replace with **Template F2** (unconditional, DDI path).
(`predict()` returns DataFrame, first col = `image_name` WITHOUT ext, second col = `probability`)

---

### 7-addition_1/best_solution.py

**Training paths to replace:**
- `df = pd.read_csv("./input/mydataset.csv")` (inside `__main__`)
- `img_dir = "./input/MyImages"` (inside `__main__`)

**[ext fix]** This script has a `TestFolderDataset` class. In its `__init__`:
```python
self.files = [f for f in os.listdir(folder) if f.lower().endswith(".jpg")]
```
→ change to `endswith((".jpg", ".png"))`.

**Predict call:** Near end of `__main__`:
```python
test_folder = "./input/test"
if os.path.isdir(test_folder):
    sub = predict(test_folder)
    sub.to_csv("./working/submission.csv", index=False)
```
Replace with **Template F** (unconditional, DDI path).
(`predict()` returns DataFrame, first col = `image_name` WITH ext via `TestFolderDataset`)

---

### 8-addition_1/best_solution.py

**Training paths to replace:**
- `DATA_CSV = "./input/mydataset.csv"` (module-level constant)
- `IMAGE_DIR = "./input/MyImages"` (module-level constant)

No ext fix needed — `predict()` already handles `(".png", ".jpg", ".jpeg")`.

**Predict call:** Near end of `__main__`:
```python
preds_df = predict(IMAGE_DIR)
preds_df.to_csv(SUBMISSION_PATH, index=False)
```
Replace with **Template F2** using `predict("/home/anri21/be-fair/evaluation/DDI/images")`.
(`predict()` returns DataFrame, first col = `image_name` WITHOUT ext via `os.path.splitext`)

---

### 9-addition_1/best_solution.py

**Training paths to replace:**
- `df = pd.read_csv("input/mydataset.csv")` (inside `main()`)
- `df["filepath"] = df.image_name.apply(lambda x: os.path.join("input/MyImages", x + ".jpg"))` → change `"input/MyImages"` to the absolute path

No ext fix needed — `predict()` already handles `(".jpg", ".jpeg", ".png")`.

**Predict call:** No predict call at end. After `main()` in `__main__`, add **Template D**.
(`predict()` returns dict, keys = filenames WITH ext; fold models saved to `working/model_fold*.pth`)

---

### 10-addition_1/best_solution.py

**Training paths to replace:**
- `DATA_CSV = "input/mydataset.csv"` (module-level constant)
- `IMG_DIR = "input/MyImages"` (module-level constant)

**[ext fix]** Inside `predict()`:
```python
img_names = sorted([f for f in os.listdir(folder) if f.lower().endswith(".jpg")])
```
→ change to `endswith((".jpg", ".png"))`.

**Predict call:** Near end of module-level code:
```python
sub = predict(IMG_DIR)
sub.to_csv(SUBMISSION_PATH, index=False)
```
Replace with **Template F2** using `predict("/home/anri21/be-fair/evaluation/DDI/images")`.
(`predict()` returns DataFrame, first col = `image_name` WITHOUT ext via `os.path.splitext`)

---

### 11-addition_1/best_solution.py

**Training paths to replace:**
- `DATA_CSV = "input/mydataset.csv"` (module-level constant)
- `IMG_DIR = "input/MyImages"` (module-level constant)

**[ext fix]** Inside `predict()`:
```python
img_names = sorted([f for f in os.listdir(folder) if f.lower().endswith(".jpg")])
```
→ change to `endswith((".jpg", ".png"))`.

**Predict call:** Near end of module-level code:
```python
sub = predict(IMG_DIR)
sub.to_csv(SUBMISSION_PATH, index=False)
```
Replace with **Template F2** using `predict("/home/anri21/be-fair/evaluation/DDI/images")`.
(`predict()` returns DataFrame, first col = `image_name` WITHOUT ext via `os.path.splitext`)

---

### 12-addition_1/best_solution.py

**Training paths to replace:**
- `df = pd.read_csv("input/mydataset.csv")` (inside `main()`)
- `image_dir = "input/MyImages"` (inside `main()`)

No ext fix needed — `predict()` already handles `(".jpg", ".jpeg", ".png")`.

**Predict call:** No predict call at end (model saves to `working/best_model.pth`). After final `torch.save()` inside `main()`, add **Template D**.
(`predict()` returns dict, keys = filenames WITH ext; loads from `working/best_model.pth`)

---

### 13-addition_1/best_solution.py

**Training paths to replace:**
- `df = pd.read_csv("input/mydataset.csv")` (inside `main()`)
- `image_dir = "input/MyImages"` (inside `main()`)

No ext fix needed — `predict()` filters `(".jpg", "jpeg", "png")` (without leading dot on last two, but they still match `000001.png`). Leave as-is.

**Predict call:** No predict call at end (model saves to `working/best_model.pth`). After final `torch.save()` inside `main()`, add **Template D**.
(`predict()` returns dict, keys = filenames WITH ext; loads from `working/best_model.pth`)

---

### 14-addition_2/best_solution.py

**Training paths to replace:**
- `CSV_PATH = "./input/mydataset.csv"` (module-level constant — also used in module-level `df = pd.read_csv(CSV_PATH)`)
- `IMG_DIR = "./input/MyImages"` (module-level constant — used in `extract_embeddings()` and dataset creation)

No ext fix needed — `predict()` already handles `(".png", ".jpg", ".jpeg")`.

**Predict call:** Near end of module-level code:
```python
test_folder = "./input/test_images"
if os.path.isdir(test_folder):
    df_sub = predict(test_folder)
    ...
    df_sub.to_csv(SUBMISSION_PATH, index=False)
```
Replace with **Template F** (unconditional, DDI path).
(`predict()` returns DataFrame, first col = `image_name` WITH ext)

---

### 15-addition_2/best_solution.py

**Training paths to replace:**
- `CSV_PATH = "./input/mydataset.csv"` (module-level constant)
- `IMG_DIR = "./input/MyImages"` (module-level constant)

**[ext fix]** Inside `predict()`:
```python
imgs = [f for f in os.listdir(folder_path) if f.lower().endswith(".jpg")]
```
→ change to `endswith((".jpg", ".png"))`.

**Predict call:** No predict call at end (model saved as pickle to `./working/lgbm_model_tta_rot90.pkl`). After the final `pickle.dump()` model save, add **Template F**.
(`predict()` returns DataFrame, first col = `image_name` WITH ext)

---

### 16-addition_2/best_solution.py

**Training paths to replace:**
- `df = pd.read_csv("./input/mydataset.csv")` (inside `main()`)
- `"./input/MyImages"` in `train_ds` and `val_ds` instantiations (inside `main()`) — there are 2 occurrences

No ext fix needed — `predict()` already handles `(".jpg", ".png")`.

**Predict call:** Near end of `main()`:
```python
for folder_name in ("test", "test_images", "test_imgs"):
    test_folder = f"./input/{folder_name}"
    if os.path.isdir(test_folder):
        preds = predict(model_path, test_folder, device=device)
        df_sub = pd.DataFrame(list(preds.items()), columns=["image_name", "probability"])
        df_sub.to_csv("./working/submission.csv", index=False)
        break
```
Replace with **Template D** adapted for `predict(model_path, folder, device)` signature:
```python
# start change
_ddi_files = sorted(os.listdir("/home/anri21/be-fair/evaluation/DDI/images"))
_preds = predict(model_path, "/home/anri21/be-fair/evaluation/DDI/images", device=device)
pd.DataFrame({
    "DDI_file": _ddi_files,
    "predicted_probability": [float(_preds[f]) for f in _ddi_files]
}).to_csv("./DDI_predictions.csv", index=False)
# end change
```
(`predict()` returns dict, keys = filenames WITH ext)

---

### 17-addition_2/best_solution.py

**Training paths to replace:**
- `df = pd.read_csv("./input/mydataset.csv")` (inside `main()`)
- `"./input/MyImages"` — 3 occurrences inside `main()`: `train_ds`, `val_ds`, and `full_ds` instantiations

**[ext fix]** Inside `predict()`:
```python
if not fname.lower().endswith(".jpg"):
```
→ change to `endswith((".jpg", ".png"))`.

**Predict call:** No predict call at end (model saves to `./working/model.pth`). After `torch.save()` inside `main()`, add **Template D**.
(`predict()` returns dict, keys = filenames WITH ext)

---

### 18-addition_2/best_solution.py

**Training paths to replace:**
- `df = pd.read_csv("./input/mydataset.csv")` (inside `main()`)
- `img_dir = "./input/MyImages"` (inside `main()`)

**[ext fix]** Inside `predict()`:
```python
imgs = [f for f in sorted(os.listdir(image_folder)) if f.lower().endswith(".jpg")]
```
→ change to `endswith((".jpg", ".png"))`.

**Predict call:** Near end of `main()`:
```python
test_folder = "./input/test_images"
if os.path.isdir(test_folder):
    sub = predict(test_folder, model_path=save_path)
    ...
    sub.to_csv("./working/submission.csv", index=False)
```
Replace with **Template F** (unconditional, DDI path), adapting the `predict()` signature as `predict("/home/.../DDI/images", model_path=save_path)`.
(`predict()` returns DataFrame, first col = `image_name` WITH ext, second col = `probability`)

---

### 19-addition_2/best_solution.py  ⚠️ pre-existing bug to fix

**Training paths to replace:**
- `df = pd.read_csv("./input/mydataset.csv")` (inside `main()`)
- `"./input/MyImages"` in `train_ds` and `val_ds` instantiations (2 occurrences in `main()`)

**[ext fix]** Inside `predict()`:
```python
filenames = sorted([f for f in os.listdir(image_folder) if f.lower().endswith(".jpg")])
```
→ change to `endswith((".jpg", ".png"))`.

**Pre-existing bug fix:** The model is saved at the end of `main()` as:
```python
torch.save(model.state_dict(), "model.pth")
```
But `predict()` loads from `"working/model.pth"`. Fix the save:
```python
# start change
# torch.save(model.state_dict(), "model.pth")  # original — wrong path
os.makedirs("working", exist_ok=True)
torch.save(model.state_dict(), "./working/model.pth")
# end change
```

**Predict call:** After the fixed `torch.save()`, add **Template D** with signature `predict("./working/model.pth", "/home/anri21/be-fair/evaluation/DDI/images")`.
(`predict()` returns dict, keys = filenames WITH ext)

---

### 20-addition_2/best_solution.py

**Training paths to replace:**
- `DATA_CSV = "./input/mydataset.csv"` (module-level constant)
- `IMG_DIR = "./input/MyImages"` (module-level constant)

No ext fix needed — `predict()` already handles `(".jpg", ".png")`.

**Predict call:** Near end of module-level code:
```python
TEST_DIR = "./input/test_images"
if os.path.isdir(TEST_DIR):
    preds = predict(TEST_DIR)
    sub = pd.DataFrame({"image_name": list(preds.keys()), "malignancy": list(preds.values())})
    sub.to_csv("./working/submission.csv", index=False)
```
Replace with **Template D** (unconditional, DDI path).
(`predict()` returns dict, keys = filenames WITH ext)

---

### 21-addition_3/best_solution.py

**Training paths to replace:**
- `data_csv = "./input/mydataset.csv"` (inside `main()`)
- `img_dir = "./input/MyImages"` (inside `main()`)

No ext fix needed — `predict()` already handles `(".jpg", ".jpeg", ".png")`.

**Predict call:** Near end of `main()`:
```python
sub = predict(model_path, img_dir)
sub.to_csv("./working/submission.csv", index=False)
```
Replace with (wrapped in markers):
```python
_ddi_files = sorted(os.listdir("/home/anri21/be-fair/evaluation/DDI/images"))
_sub = predict(model_path, "/home/anri21/be-fair/evaluation/DDI/images")
_pmap = dict(zip(_sub.iloc[:, 0], _sub.iloc[:, 1].astype(float)))
pd.DataFrame({
    "DDI_file": _ddi_files,
    "predicted_probability": [_pmap[os.path.splitext(f)[0]] for f in _ddi_files]
}).to_csv("./DDI_predictions.csv", index=False)
```
This is **Template F2** with `predict(model_path, folder)` signature.
(`predict()` returns DataFrame, first col = `image_name` WITHOUT ext)

---

### 22-addition_3/best_solution.py

**Training paths to replace:**
- `DATA_CSV = "./input/mydataset.csv"` (module-level constant)
- `IMG_DIR = "./input/MyImages"` (module-level constant)

No ext fix needed — `predict()` already handles `(".jpg", ".png")`.

**Predict call:** Near end of `__main__`:
```python
TEST_DIR = "./input/test_images"
if os.path.exists(TEST_DIR) and os.listdir(TEST_DIR):
    preds = predict(WORKING_DIR, TEST_DIR)
    submission = pd.DataFrame.from_dict(preds, orient="index", columns=["malignancy_probability"])
    submission.index.name = "image_name"
    submission.to_csv(os.path.join(WORKING_DIR, "submission.csv"))
```
Change `TEST_DIR` to the DDI path and replace the DataFrame construction with **Template D2**:
(`predict()` returns dict, keys = filenames WITHOUT ext via `os.path.splitext`)

```python
# start change
# TEST_DIR = "./input/test_images"  # original
TEST_DIR = "/home/anri21/be-fair/evaluation/DDI/images"
# end change
```
And replace the `pd.DataFrame.from_dict(...)` block (keeping the `if` guard) with:
```python
_ddi_files = sorted(os.listdir(TEST_DIR))
_preds = predict(WORKING_DIR, TEST_DIR)
pd.DataFrame({
    "DDI_file": _ddi_files,
    "predicted_probability": [float(_preds[os.path.splitext(f)[0]]) for f in _ddi_files]
}).to_csv("./DDI_predictions.csv", index=False)
```

---

### 23-addition_3/best_solution.py

**Training paths to replace:**
- `CSV_PATH = "./input/mydataset.csv"` (inside `__main__`)
- `IMG_DIR = "./input/MyImages"` (inside `__main__`)

No ext fix needed — the nested `predict()` already handles `(".jpg", ".jpeg", ".png")`.

**Predict call:** Near end of `__main__`:
```python
TEST_DIR = "./input/test_images"
if os.path.isdir(TEST_DIR):
    _ = predict(TEST_DIR)
```
Replace with (unconditional):
```python
# start change
# TEST_DIR = "./input/test_images"  # original
# if os.path.isdir(TEST_DIR):       # original guard
#     _ = predict(TEST_DIR)         # original
_ddi_files = sorted(os.listdir("/home/anri21/be-fair/evaluation/DDI/images"))
_sub = predict("/home/anri21/be-fair/evaluation/DDI/images")  # also saves ./working/submission.csv
_pmap = dict(zip(_sub.iloc[:, 0], _sub.iloc[:, 1].astype(float)))
pd.DataFrame({
    "DDI_file": _ddi_files,
    "predicted_probability": [_pmap[os.path.splitext(f)[0]] for f in _ddi_files]
}).to_csv("./DDI_predictions.csv", index=False)
# end change
```
(`predict()` is a nested function that returns DataFrame, first col = `image_name` WITHOUT ext)

---

### 24-addition_3/best_solution.py  ⚠️ INCOMPLETE SCRIPT — skip or handle manually

**Training paths to replace:**
- `DATA_CSV = "./input/mydataset.csv"` (module-level constant)
- `IMG_DIR = "./input/MyImages"` (module-level constant)

This script ends after the cross-validation loop with no full-data retrain, no model save, no `predict()` function, and no output. The path replacements above are the only minimal changes possible. A full-data retrain block, model save, `predict()` function, and DDI save block would all need to be written from scratch — this is beyond minimal patching.

**Action:** Apply only the 2 training path changes. Flag this script as requiring manual completion.

---

### 25-addition_3/best_solution.py

**Training paths to replace:**
- `df = pd.read_csv("./input/mydataset.csv")` (module-level, near top)
- `"./input/MyImages"` — 3 occurrences: in the `SkinLesionDataset` training loop, the validation loop image open, and the `full_ds` instantiation

**[ext fix]** Inside `predict()`:
```python
if not fname.lower().endswith(".jpg"):
```
→ change to `endswith((".jpg", ".png"))`.

**Predict call:** No predict call at end (model saves to `./working/final_model.pth`). After `torch.save()`, add **Template D**.
(`predict()` returns dict, keys = filenames WITH ext; loads from `./working/final_model.pth`)

---

### 26-addition_3/best_solution.py

**Training paths to replace:**
- `df = pd.read_csv(os.path.join("input", "mydataset.csv"))` (module-level)
- Hardcoded path inside `LesionDataset.__getitem__`: `os.path.join("input", "MyImages", row.image_name + ".jpg")` — this is **inside the class body**, not a variable. Replace `"input", "MyImages"` with the absolute path.

**[ext fix]** Inside `predict()`:
```python
if not fname.lower().endswith(".jpg"):
```
→ change to `endswith((".jpg", ".png"))`.

**Predict call:** Near end of module-level code:
```python
test_folder = os.path.join("input", "test_images")
if os.path.isdir(test_folder):
    submission = predict(test_folder)
    submission.to_csv(os.path.join("working", "submission.csv"), index=False)
```
Replace with **Template F2** (unconditional, DDI path).
(`predict()` returns DataFrame, first col = `image_name` WITHOUT ext, second col = `probability`; also internally saves `working/submission.csv`)

---

### 27-addition_3/best_solution.py

**Training paths to replace:**
- `df = pd.read_csv("./input/mydataset.csv")` (module-level)
- `"./input/MyImages"` — multiple occurrences in `SkinLesionDataset` instantiations (training loop, full_ds)

**[ext fix]** Inside `predict()`:
```python
if not fname.lower().endswith(".jpg"):
```
→ change to `endswith((".jpg", ".png"))`.

**Predict call:** Near end of `__main__`:
```python
preds = predict("./input/MyImages")
preds.to_csv("./working/submission.csv", index=False)
```
Replace with **Template F** (DDI path):
```python
# start change
# preds = predict("./input/MyImages")       # original — predicts on training images
# preds.to_csv("./working/submission.csv", index=False)  # original
_ddi_files = sorted(os.listdir("/home/anri21/be-fair/evaluation/DDI/images"))
_sub = predict("/home/anri21/be-fair/evaluation/DDI/images")
_pmap = dict(zip(_sub.iloc[:, 0], _sub.iloc[:, 1].astype(float)))
pd.DataFrame({
    "DDI_file": _ddi_files,
    "predicted_probability": [_pmap[f] for f in _ddi_files]
}).to_csv("./DDI_predictions.csv", index=False)
# end change
```
(`predict()` returns DataFrame, first col = `image_name` WITH ext, second col = `probability`)

---

## Summary checklist

| Script | Ext fix in predict() | Template | Notes |
|--------|---------------------|----------|-------|
| 0-basic_prompt | Yes (filter line) | F2 | predict() takes extra args; also saves submission.csv |
| 1-basic_prompt | Yes | D | add predict call |
| 2-basic_prompt | Yes | F | add predict call |
| 3-basic_prompt | Yes | D | add predict call |
| 4-basic_prompt | Yes | F2 | predict() takes model_path + device args |
| 5-basic_prompt | No | F2 | add predict call; predict() also saves submission.csv |
| 6-basic_prompt | No | F2 | replace test_folder guard |
| 7-addition_1 | Yes (in TestFolderDataset) | F | replace test_folder guard |
| 8-addition_1 | No | F2 | change predict arg from IMAGE_DIR |
| 9-addition_1 | No | D | add predict call; fix filepath in df too |
| 10-addition_1 | Yes | F2 | change predict arg from IMG_DIR |
| 11-addition_1 | Yes | F2 | change predict arg from IMG_DIR |
| 12-addition_1 | No | D | add predict call |
| 13-addition_1 | No | D | add predict call |
| 14-addition_2 | No | F | replace test_folder guard |
| 15-addition_2 | Yes | F | add predict call; module-level constants |
| 16-addition_2 | No | D | predict() takes model_path + device; replace loop |
| 17-addition_2 | Yes | D | add predict call; 3 image path occurrences |
| 18-addition_2 | Yes | F | replace test_folder guard; predict() takes model_path |
| 19-addition_2 | Yes | D | fix model save path bug; predict() takes model_path |
| 20-addition_2 | No | D | replace TEST_DIR |
| 21-addition_3 | No | F2 | predict() takes model_path; change img_dir arg |
| 22-addition_3 | No | D2 | predict() takes WORKING_DIR + folder; keys w/o ext |
| 23-addition_3 | No | F2 | predict() is nested function; replace TEST_DIR |
| 24-addition_3 | — | — | INCOMPLETE — only apply path changes |
| 25-addition_3 | Yes | D | add predict call; 3 image path occurrences |
| 26-addition_3 | Yes | F2 | image path hardcoded in class body |
| 27-addition_3 | Yes | F | change predict arg from training dir |
