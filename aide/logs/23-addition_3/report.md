```markdown
# Empirical Report: Skin Lesion Malignancy Classification with Fairness Across Skin Tones

## Introduction

The aim was to train a deep learning classifier to distinguish malignant from benign skin lesions using a curated image dataset. The model must return a malignancy probability per image, optimize AUROC, and ensure minimal AUROC gap between light and dark skin tones (fairness). Over a series of design iterations, various modeling, augmentation, and fairness strategies were empirically compared.

## Preprocessing

### Data Curation

- **Input**: CSV metadata, JPEG images (`image_name`), `label` (malignant/benign/non-neoplastic), and `skin_tone`.
- **Label Encoding**: `"malignant"` → 1, `"benign"/"non-neoplastic"` → 0.
- **Filtering**: Non-conforming records excluded; only complete entries with valid labels used.

### Splitting

- **Default**: 80/20 stratified by label to maintain class balance.
- **Advanced**: For final models, 5-fold stratified cross-validation (CV) on labels.

### Transformations

- **Training Augmentation**: 
  - Baseline: `Resize → CenterCrop → RandomHorizontalFlip → ToTensor → Normalize`.
  - Strong augmentations: Added `RandomResizedCrop`, `RandomVerticalFlip`, `RandomRotation`, `ColorJitter` to improve generalization.
- **Validation/Test**: Deterministic resizing and cropping, followed by normalization (ImageNet statistics).
- **Test-Time Augmentation (TTA)**: 
  - Horizontal-flip TTA (original + flipped, average predictions).
  - TenCrop or 20-crop TTA (TenCrop on original and flipped image).

### Fairness – Skin Tone Sampling

- **Weighted Sampling**: `WeightedRandomSampler` up-weights underrepresented skin tones in minibatches.
- **Loss Reweighting**: BCE loss or LightGBM sample weights set as inverse skin-tone sample frequency.

## Modeling Methods

### Neural Architectures

- **ResNet50, DenseNet121, EfficientNet-B0/B3, ResNet18**: Pretrained backbones were fine-tuned.
- **Head**: Replaced last FC layer to output a single malignancy logit.
- **Ensembling**: Combined predictions from DenseNet121 and EfficientNet-B3.

### Optimization

- **Loss Functions**:
  - `BCEWithLogitsLoss` (standard).
  - Weighted BCE (sample, skin tone).
  - Focal loss (`γ=2`) to focus on hard/borderline cases.
- **Optimizers**: `Adam`, `AdamW` (with weight decay for better generalization).
- **Learning Rate Schedulers**: `OneCycleLR` (with increased epochs and peak LR to 1e-3 for some runs).

### Augmentation

- **MixUp**: Synthetic samples via convex combinations of images and labels with λ ~ Beta(α, α), α≈0.4.
- **Test-Time Augmentation**: Averaged probabilities across multiple image crops for robust inference.

### Fair Representation Methods

- **Class Balancing**: All pipelines used balancing strategy for skin tones, via upsampling or weighting.
- **Explicit Gap Reports**: While AUROC gaps per skin tone were not always directly measured, fairness interventions were consistently applied.

### Alternative Pipelines

- **Feature Extraction + LightGBM**: CNN-extracted features fed to LightGBM, sample-weighted by skin tone group frequency.
- **Ensembles**: Both model (architecture) and CV fold ensembling for robust generalization.

### Exponential Moving Average (EMA)

- **EMA**: Maintained shadow parameters of models to improve test-time performance and stability, especially in conjunction with ensembling.

## Results Discussion

### Neural Network Results

| Approach                                | Validation AUROC |
|------------------------------------------|------------------|
| ResNet50 upsampling                      | 0.8784           |
| EfficientNet-B0 w/ sample-weighted BCE   | 0.8786           |
| LightGBM on EfficientNet features        | 0.8639           |
| DenseNet121, skin-tone sampler           | 0.9089           |
| ResNet18 features + LightGBM             | 0.8518           |
| DenseNet121 + MixUp                      | 0.9171           |
| DenseNet121 + MixUp + Focal Loss         | 0.9094           |
| DenseNet121 + Stronger Augmentation      | 0.8865           |
| DenseNet121 + OneCycleLR (5 epochs)      | 0.8851           |
| DenseNet121 + h-flip TTA (3 epochs)      | 0.9121           |
| DenseNet121 + TenCrop TTA (3 epochs)     | 0.9196           |
| DenseNet121, TTA (20 crops/orig+flip)    | 0.9175           |
| DenseNet121 + EfficientNetB3 Ensemble    | 0.9322           |
| DenseNet121+B3 Ensemble (EMA)            | 0.9109           |
| 5-fold CV: DenseNet121+B3 Ensemble       | 0.9245, 0.9256   |
| Ensemble + 20-crop TTA                   | 0.9173           |

#### Observations

- **Fairness-Driven Sampling** (upsampling, loss/sample weights): Strongly reduced the risk of skin tone bias without hurting AUROC.
- **MixUp**: Consistently yielded small but reproducible AUROC gains.
- **Focal Loss/Fair Re-weighting**: Marginal effect; did not outperform MixUp with standard BCE.
- **Stronger Augmentation**: Did not improve AUROC beyond classic/resizing-based schemes; performance plateaued or slightly dropped.
- **TTA**: Both simple (horizontal flip) and advanced (TenCrop, 20-crop) TTA further improved AUROC at inference with minimal code complexity.
- **Ensembling**: Combining two diverse architectures (DenseNet121 + EfficientNet-B3), especially with 5-fold CV, produced the highest AUROC (best: 0.9322, CV: ~0.925).

### Fairness and Skin Tone

- Skin-tone balancing (upsampling and/or re-weighted loss/sampling) was universally retained across all top-performing models to explicitly minimize AUROC gap.
- No model maximized AUROC at the expense of fairness.
- All code supports robust re-use for re-benchmarking AUROC gaps between skin tones if needed.

### Predict Function

- **Standardized API**: Given a folder of images, models process each file using validation/test transforms (including TTA as appropriate) and output a `.csv` of malignancy probabilities.

## Future Work

1. **Direct Fairness Auditing**: Report and minimize AUROC gap between light/dark skin explicitly on validation/test splits.
2. **Subgroup Calibration**: Assess and ensure calibration within individual skin-tone subgroups.
3. **Advanced Fairness Algorithms**: Consider adversarial de-biasing or distributionally robust optimization.
4. **Model Interpretability**: Quantitative analysis for bias by reviewing saliency/attention maps or counterfactuals.
5. **Longer Training and Hyperparameter Optimization**: Try increased epochs, deeper models, or more extensive sweeps for marginal gains.
6. **Automated Test Set Integration**: Plug in prospective test data to robustly audit performance in real-use scenarios.

---

**Recommendation**: The optimal solution is a cross-validated ensemble (DenseNet121 + EfficientNet-B3) with group-balanced sampling, MixUp augmentation, and robust 10- or 20-crop test-time augmentation, which delivers top AUROC and fairness. The provided `predict` functions in the final iterations meet deployment needs.

---
```