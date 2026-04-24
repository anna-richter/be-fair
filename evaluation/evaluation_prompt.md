# DDI Evaluation: Adapt an Evaluation Script for Inference (Evaluation Folder Only)

This document records the steps to convert an existing **evaluation script** (e.g. `evaluation/aide/<run_name>/<run_name>_best_solution_DDI.py`) into an **inference-only** script that evaluates a pretrained model on the DDI dataset **using only files inside the `evaluation/` folder**.

Constraints for this repo:

- All inputs needed for DDI evaluation live under `evaluation/` (including DDI images).
- The evaluation process must **not** read from, write to, or reference anything outside `evaluation/`.

---

## Commenting Convention

When disabling or modifying code, **never delete any line** — always comment it out so the original logic is preserved and recoverable.

- Use `#` to comment out **up to 3 lines**.
- Use `"""..."""` (triple-quoted string) to comment out **4 or more lines**.

**Never touch existing comments.** Do not edit, reword, remove, or "fix" any comment that is already present in the script — even if the comment appears factually outdated after your changes. Leave every original comment exactly as written.

**When modifying a line of code**, do not edit it in place. Instead:
1. Copy the original line and comment it out with `#` immediately above the new line.
2. Add the new (replacement) line directly below.
3. Wrap the pair with `# --- BEGIN CHANGE ---` before and `# --- END CHANGE ---` after so the modification is clearly bracketed.

```python
# --- BEGIN CHANGE ---
#img_files = sorted([f for f in os.listdir(image_dir) if f.lower().endswith(".jpg")])
img_files = sorted([f for f in os.listdir(image_dir) if f.lower().endswith(".png")])
# --- END CHANGE ---
```

The same bracket markers apply when inserting a block of **new** lines (no old line to comment out — just mark the region):

```python
# --- BEGIN CHANGE ---
ddi_image_dir = os.path.join("..", "..", "DDI", "images")
predictions = predict(ddi_image_dir, model_path=os.path.join("working", "pipeline.pkl"))
predictions.to_csv("run_DDI_predictions.csv", index=False)
# --- END CHANGE ---
```

**Summary of hard rules:**
- Zero lines deleted — only commented out.
- Zero existing comments altered.
- Every modified or newly inserted line is enclosed in `# --- BEGIN CHANGE ---` … `# --- END CHANGE ---` markers.

---

## Prerequisites

- A trained model artifact saved under `evaluation/aide/<run_name>/working/` (examples in this repo include `pipeline.pkl`, `*.pth`, `*.pt`).
- DDI images as `.png` files in `evaluation/DDI/images/`.
- An evaluation script located under `evaluation/aide/<run_name>/` (examples: `evaluation/aide/10-basic_prompt/10-basic_prompt_best_solution_DDI.py`, `evaluation/aide/9-basic_prompt/9-basic_prompt_best_solution_DDI.py`).

### Missing artifact policy

If a required model artifact does not exist under `working/`, **do not attempt to re-run a training script**. Instead, the DDI `__main__` entrypoint must check for the artifact's existence and emit a clear warning before exiting:

```python
# --- BEGIN CHANGE ---
model_path = os.path.join("working", "<artifact_name>")
if not os.path.exists(model_path):
    print(f"Warning: model artifact not found at {model_path}. Skipping DDI inference.")
    exit(1)
# --- END CHANGE ---
```

This keeps the script safe to run even when artifacts are missing, without silently producing empty or incorrect predictions.

---

## Step-by-Step Procedure

### 0. Rule: minimal diff, comment-only disabling

When adapting a script:

- **Never delete** any line — training code or otherwise. Disable it by commenting it out.
- **Never alter any existing comment**, even if it seems outdated or inaccurate after your changes. Leave every original comment exactly as written.
- **When modifying a line**, copy it, comment the copy out with `#`, place the new version directly below, and wrap the pair in `# --- BEGIN CHANGE ---` / `# --- END CHANGE ---` markers (see Commenting Convention above).
- **When inserting new lines**, wrap the entire insertion in `# --- BEGIN CHANGE ---` / `# --- END CHANGE ---` markers.
- Prefer the **smallest change** that makes DDI inference work.
- **Do a thorough check before changing anything**. If the script is already format-agnostic (e.g. it lists `*.png` already, or it uses `PIL.Image.open` on full paths, or it filters on a set like `{".jpg",".jpeg",".png"}`), do not add extra code.

### 0.1 Quick audit checklist (do this first, in order)

Open the target script under `evaluation/aide/<run_name>/` and verify:

- **Working directory assumptions**: does it rely on `./working`? If yes, you must run it from `evaluation/aide/<run_name>/`.
- **Where inference reads images from**: does `predict()` accept an image folder path, and does the `__main__` entrypoint point to DDI (`../../DDI/images`)?
- **Extension handling**: does it already list `.png` (or support multiple extensions)? If yes, make no changes here.
- **Filename expectations**: does it internally append `".jpg"`/`".png"` to an ID, or does it work with actual filenames? Only change this if it would break on DDI.
- **Output schema**: does it already produce `DDI_file` and `predicted_probability`? If yes, do not rename columns.
- **What model artifact it expects**: confirm the artifact path is under `evaluation/aide/<run_name>/working/` and matches what `predict()` loads (e.g. `pipeline.pkl`).

### 1. Comment out training data loading

Find the block that loads the training CSV and splits into train/validation sets. Wrap it in a `"""..."""` triple-quoted string so it is skipped without requiring the original dataset to be present.

**Pattern to find and wrap:**
```python
"""
# Paths and metadata
csv_path = "./input/<training_labels>.csv"
img_dir = "./input/<training_images_dir>"
df = pd.read_csv(csv_path)

# Train/validation split
train_df, val_df = train_test_split(
    df, test_size=0.2, stratify=df["three_partition_label"], random_state=42
)
"""
```

### 2. Comment out dataset and DataLoader setup

Find the block constructing `Dataset` objects, samplers, and `DataLoader`s for training and validation. Wrap it in `"""..."""`.

**Pattern to find and wrap:**
```python
"""
# Datasets & loaders
train_ds = SkinLesionDataset(train_df, img_dir, transform=train_tfms)
val_ds = SkinLesionDataset(val_df, img_dir, transform=val_tfms)
# ... class weights, WeightedRandomSampler, train_loader, val_loader ...
"""
```

### 3. Comment out training-time model/optimizer setup; keep inference essentials

Depending on the evaluation script, inference may require some outer-scope variables (e.g. `device`, `mean/std`, feature extractor instantiation). Keep only what inference needs; comment out anything that is only used for training (loss, optimizer, scheduler, training loop, metrics on training labels, etc.).

If the script’s `predict()` uses `num_ftrs` or similar architecture-specific constants, either:

- Keep the line that computes it (if it does not require training data), or
- Replace it with a correct constant for the architecture used by this script.

**Before:**
```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.densenet121(pretrained=True)
num_ftrs = model.classifier.in_features
model.classifier = nn.Linear(num_ftrs, 1)
model = model.to(device)
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.5)
```

**After:**
```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
"""
model = models.densenet121(pretrained=True)
num_ftrs = model.classifier.in_features
model.classifier = nn.Linear(num_ftrs, 1)
model = model.to(device)
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.5)
...training loop...
print(f"Best Validation AUROC: {best_auroc:.4f}")
"""
```

### 4. Comment out the entire training loop

Find the `for epoch in range(epochs):` block (including the preceding epoch/auroc setup and the final `print` of best AUROC) and include it in the same `"""..."""` block from step 3, or wrap it separately.

**Pattern:**
```python
"""
epochs = 5
best_auroc = 0.0
os.makedirs("./working", exist_ok=True)
for epoch in range(epochs):
    ...
print(f"Best Validation AUROC: {best_auroc:.4f}")
"""
```

### 5. Update the `predict()` function — input image source and file extension

For DDI evaluation in this repo, the images are `.png` files under `evaluation/DDI/images/`.

Only make changes here if the script would miss DDI files (e.g. it hard-filters `.jpg`, or it constructs filenames with `+ ".jpg"`).

If the script already supports `.png` (or supports multiple extensions), do not change it.

**Before:**
```python
files = [f for f in os.listdir(image_dir) if f.lower().endswith(".jpg")]
```

**After (do not delete the old line — comment it out and add the new line with change markers):**
```python
# --- BEGIN CHANGE ---
#files = [f for f in os.listdir(image_dir) if f.lower().endswith(".jpg")]
files = sorted([f for f in os.listdir(image_dir) if f.lower().endswith(".png")])
# --- END CHANGE ---
```

### 6. Update the `predict()` function — output column names

Rename the output DataFrame columns to match the required format.

Only do this if the script does not already output the required columns.

**Before:**
```python
return pd.DataFrame({"image": img_names, "malignant_prob": preds})
```

**After (do not delete the old line — comment it out and add the new line with change markers):**
```python
# --- BEGIN CHANGE ---
#return pd.DataFrame({"image": img_names, "malignant_prob": preds})
return pd.DataFrame({"DDI_file": img_names, "predicted_probability": preds})
# --- END CHANGE ---
```

### 7. Replace the final inference call (stay within `evaluation/`)

Replace the original submission generation block with a call that:
- Points to a model artifact under `evaluation/aide/<run_name>/working/`
- Points to the bundled DDI images directory `evaluation/DDI/images/`
- Saves the CSV under `evaluation/aide/<run_name>/` as `<run_name>_DDI_predictions.csv`

**Before:**
```python
# Generate submission.csv
submission = predict("./working/densenet121_best.pth", img_dir)
submission.to_csv("./working/submission.csv", index=False)
```

**After (do not delete the old lines — comment them out, then add new lines, all wrapped in change markers; do not touch the existing `# Generate submission.csv` comment):**
```python
# Generate submission.csv
# --- BEGIN CHANGE ---
#submission = predict("./working/densenet121_best.pth", img_dir)
#submission.to_csv("./working/submission.csv", index=False)
ddi_image_dir = os.path.join("..", "..", "DDI", "images")
predictions = predict(ddi_image_dir, model_path=os.path.join("working", "<artifact_name>"))
predictions.to_csv("<run_name>_DDI_predictions.csv", index=False)
print(f"Saved {len(predictions)} predictions to <run_name>_DDI_predictions.csv")
# --- END CHANGE ---
```

---

## Concrete recipe (recommended): `evaluation/aide/10-basic_prompt/10-basic_prompt_best_solution_DDI.py`

This repo already contains a script that is close to what you want:

- Script: `evaluation/aide/10-basic_prompt/10-basic_prompt_best_solution_DDI.py`
- Model artifact: `evaluation/aide/10-basic_prompt/working/pipeline.pkl`
- DDI images: `evaluation/DDI/images/*.png`

To adapt it with **minimal changes**:

### A) Comment out the training-only `__main__` block

In `evaluation/aide/10-basic_prompt/10-basic_prompt_best_solution_DDI.py`, wrap everything under:

```python
if __name__ == "__main__":
    ...
```

in a `"""..."""` block so none of the training data reads (`./input/...`) run.

### B) Inference: switch `.jpg` → `.png` (two small edits)

Only do these edits if the script currently hardcodes `.jpg`.

For each line that requires a change, comment out the original line and add the replacement directly below it, wrapped in `# --- BEGIN CHANGE ---` / `# --- END CHANGE ---` markers.

- If `ImageDataset.__getitem__` does `name + ".jpg"`:

```python
# --- BEGIN CHANGE ---
#        return self.transform(Image.open(os.path.join(self.folder, name + ".jpg")))
        return self.transform(Image.open(os.path.join(self.folder, name + ".png")))
# --- END CHANGE ---
```

- If `predict()` filters `.jpg`:

```python
# --- BEGIN CHANGE ---
#imgs = [os.path.splitext(f)[0] for f in os.listdir(image_folder) if f.lower().endswith(".jpg")]
imgs = sorted(
    os.path.splitext(f)[0]
    for f in os.listdir(image_folder)
    if f.lower().endswith(".png")
)
# --- END CHANGE ---
```

### C) Output: rename columns to the required DDI format

Do not delete the original return statement. Comment it out and add the replacement directly below, wrapped in change markers:

```python
# --- BEGIN CHANGE ---
#return pd.DataFrame({"image": [n + ".jpg" for n in names], "malignant_prob": probs})
return pd.DataFrame(
    {"DDI_file": [n + ".png" for n in names], "predicted_probability": probs}
)
# --- END CHANGE ---
```

This keeps the existing pipeline behavior, but formats output as required.

### D) Add a new DDI-only entrypoint (comment-only replacement)

Below the commented-out training `__main__` block, add a new `if __name__ == "__main__":` that:

- uses `ddi_image_dir = os.path.join("..", "..", "DDI", "images")`
- uses `model_path = os.path.join("working", "pipeline.pkl")`
- writes `10-basic_prompt_DDI_predictions.csv` into `evaluation/aide/10-basic_prompt/`

---

## Output

The script produces a CSV at `evaluation/aide/<run_name>/<run_name>_DDI_predictions.csv` with two columns:

| DDI_file   | predicted_probability |
|------------|-----------------------|
| 000001.png | 0.342...              |
| 000002.png | 0.817...              |
| ...        | ...                   |

`predicted_probability` is a float in [0, 1] representing the model's estimated probability that the lesion is malignant.

---

## Notes

- The DDI evaluation images are `.png` in `evaluation/DDI/images/`; do not assume `.jpg`.
- If the script uses test-time augmentation (TTA), keep it (it affects probabilities), but ensure it only depends on inference-time inputs.
- Any normalization constants (`mean/std`) used in transforms must remain defined wherever `predict()` expects them.
- Run the script from `evaluation/aide/<run_name>/` so `working/` resolves to `evaluation/aide/<run_name>/working/` and `../../DDI/images` resolves correctly.
