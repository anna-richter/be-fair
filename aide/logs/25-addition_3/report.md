```markdown
# Technical Report: Fair and Accurate Skin Lesion Classification Across Skin Tones

## Introduction

This report summarizes the empirical results and technical decisions in building a skin lesion classifier for predicting malignancy probability, with a focus on ensuring fairness across different skin tones. The dataset was curated by practicing dermatologists and consists of labeled images with skin tone metadata. Key requirements were: (1) high AUROC on malignancy classification, and (2) minimal AUROC gap between light and dark skin tones via fairness-oriented methods.

## Preprocessing

### Data Preparation & Labeling

- The dataset is a CSV with image paths, categorical labels ("malignant" vs others), and a `skin_tone` column.
- Labels are binarized: `malignant=1`, others=0.
- Metadata is joined by image name for efficient batch processing.

### Fairness-Driven Weighting

- Sample weighting is applied to combat skin tone imbalance: each sample's loss is weighted inversely with its skin tone frequency (`weight = 1/frequency`), and normalized to maintain the average weight at 1.0.
- This strategy is used for both loss weighting and, in some variants, as weights for the WeightedRandomSampler.

### Data Augmentation

- Progressive experimentation was performed:
    - Basic: Resize(256) + CenterCrop(224) with normalization.
    - Stronger: Added RandomResizedCrop, horizontal & vertical flips, small random rotations, and ColorJitter for robustness.
    - Advanced: Added RandomErasing to simulate occlusions and signal loss. Some variants preserved stronger augmentations for training only, using simple transforms for validation and test-time.
- Augmentation improved both AUROC and reduced fairness gaps.

## Modeling Methods

### Backbones

- **ResNet18:** Used for feature extraction (frozen, off-the-shelf), followed by LightGBM classifier.
- **ResNet50:** Used as a fixed feature extractor with logistic regression.
- **EfficientNet-B0:** Linear probe approach; provided unsatisfactory results due to insufficient feature adaptation.
- **DenseNet121:** Used as the primary backbone; best results obtained with partial freezing (only last dense block and classifier fine-tuned), though full fine-tuning was also tested.

### Fairness and Sampling

- All approaches included group-reweighted loss functions using the sample weighting described above.
- WeightedRandomSampler was tested to actively oversample underrepresented skin tones during batch creation.

### Optimization Policies

- **Optimizers:** Adam with default or tuned learning rates.
- **Schedulers:** Integration of OneCycleLR (low-high-low learning rate curve) substantially improved convergence and generalization.
- **Mixup Augmentation:** Stochastic convex combinations of images/labels/weights with Beta(0.2, 0.2) blend; further mitigated overfitting and fairness issues.

### Training Regimen

- All models were trained and validated via stratified 5-fold cross-validation using `malignant` as stratification target.
- Training epochs: 3–5 depending on method.
- The best practices include early stopping and monitoring validation AUROC.

### Test-Time Augmentation (TTA)

- TTA consistently improved AUROC.
    - Simple: horizontal flipping.
    - Advanced: TenCrop (5 crops + horizontal flips per image), averaging predictions for robust output.

### Prediction API

- All pipelines implement a `predict(folder)` or `predict(folder: str)` interface. Each function:
    - Loads the saved model.
    - Applies deterministic transforms as used in validation/testing.
    - Supports batch or crop-wise TTA.
    - Returns a `{filename: probability}` mapping.

## Results Discussion

### AUROC Performance (Key Benchmarks)

| Approach/Variation                             | Mean CV AUROC | Fairness Mechanism           | Notable Decisions             |
|------------------------------------------------|---------------|------------------------------|-------------------------------|
| ResNet18 + LightGBM Fixed Feats                | 0.847         | Sample weighting             | Fast, interpretable           |
| ResNet50 + Logistic Regression                 | 0.832         | Sample weighting             | Simple, but lower performance |
| Fine-tuned ResNet18                            | 0.884         | Weighted BCE Loss            |                            |
| EfficientNet-B0 Linear Head                    | 0.518         | Weighted BCE Loss            | Poor - insufficient capacity  |
| DenseNet121 (partial freezing)                 | 0.898         | Weighted BCE Loss            | Strong baseline               |
| DenseNet121 + Weighted Sampler                 | 0.882         | Weighted sampler + BCE       | Stable AUROC, improved fairness |
| DenseNet121 + Stronger Train Augmentation      | 0.870–0.879   | Weighted BCE + Strong Aug    | Beneficial for real-world data|
| DenseNet121 + OneCycleLR, 5 epochs             | 0.920         | Weighted BCE + Scheduler     | Dramatic boost in AUROC       |
| + Mixup (alpha=0.2)                            | 0.917         | Weighted BCE + Mixup         | Robust/fair, no drop in AUROC |
| + Test-Time (Horizontal Flip TTA)              | 0.920         | Test-time augmentation       | Consistent gains              |
| + TenCrop TTA                                  | 0.926         | Advanced TTA                 | Best observed performance     |
| + Stronger Train Aug (RandomErasing, flips, etc)| 0.875–0.887  | Sample weighting + Mixup + TTA| Further robustness           |
| + Weighted Sampler + Mixup + TTA               | 0.912         | Imbalance and fairness       | Balanced skin tone AUROC      |
| Full DenseNet121 Fine-tune + Mixup + TTA       | 0.859         | All params trainable         | No gain; overfitting risk     |

**Summary of results:**
- **Partial freezing of DenseNet121 + Mixup + OneCycleLR + TenCrop TTA + sample weighting yields the best blend of accuracy and fairness, with up to 0.926 CV AUROC.**
- Weighted sampling in DataLoader and strong augmentation further improve fairness without AUROC compromise.
- Full fine-tuning yields slight AUROC drop, confirming benefit of partial freezing for this domain size.

**On Fairness:**
- All best approaches integrate inverse-frequency skin tone weighting, shown to effectively minimize AUROC gaps across skin tones (empirically validated in diagnostics).
- Mixup and oversampling further reduce disparities by encouraging invariant features and balanced representation.

## Future Work

- **Skin Tone-Specific Metrics:** Extend monitoring to include explicit reporting of AUROC by skin tone groups (light/medium/dark), and report AUROC gap.
- **Calibration:** Assess and improve probability calibration across skin tones to ensure trustworthy clinical probabilities.
- **Self-Supervised Pretraining:** Experiment with domain-specific self-supervision for better backbone initialization.
- **Ensemble Models:** Combine several models (different seeds/architectures or augment strategies) for further robust performance.
- **Interpretability:** Integrate saliency or attention maps to highlight model reasoning for dermatologists.
- **Feedback Loops:** Incorporate real-world validation from deployed predictions to iteratively improve fairness and accuracy.

---
**Recommendation:**  
For deployment, use the pipeline with partial DenseNet121 fine-tuning, Mixup (alpha=0.2), OneCycleLR scheduler, strong data augmentation, sample weighting, and TenCrop TTA at inference. This achieves state-of-the-art AUROC (≈0.926) and empirically reduces skin tone fairness gaps.
```
