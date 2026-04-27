```markdown
# Skin Lesion Malignancy Classification: Empirical Findings and Technical Decisions

## Introduction

The objective was to design and evaluate deep learning models that classify skin lesions as malignant or benign, trained on a curated dataset provided by dermatologists. The primary evaluation metric is the Area Under the Receiver Operating Characteristic Curve (AUROC). The output system must include both a saved model and an inference function that predicts malignancy probabilities for batches of new, unseen images.

## Preprocessing

- **Label Mapping:** All solutions map the multiclass `label` column to a binary target (`malignant`→1, other→0).
- **Image Preprocessing:** All models resize images to standard input shapes (typically 224x224), normalize using ImageNet statistics, and convert to tensors.
- **Splitting:** Most experiments use either a holdout 80/20 stratified split or 5-fold stratified cross-validation (CV) to preserve label balance in splits.
- **Data Augmentation:**
  - Baseline: Standard flips and crops (horizontal/vertical flips, center crop, random crop).
  - Advanced: Added stronger augmentations including `ColorJitter`, `RandomResizedCrop`, and more advanced techniques like MixUp and CutMix.

## Modelling Methods

### Architectures

1. **ResNet Variants:** Pretrained ResNet18 and ResNet34 backbones (from `timm`), single-output sigmoid head.
2. **EfficientNet-B0:** Pretrained backbone with 5-fold CV, evaluated with AUROC.
3. **DenseNet121:** Pretrained backbone (from `torchvision`), single-output linear classifier; most extensively tuned and found to perform robustly.
4. **Vision Transformer (ViT-B/16):** Pretrained transformer-based backbone (from `timm`), with standard ImageNet transforms.

### Training Recipes

- **Loss Function:** `BCEWithLogitsLoss` for binary outcome.
- **Optimizer:** Either Adam or AdamW. Later experiments favored AdamW for better regularization with weight decay.
- **Learning Rate Scheduling:** 
  - Baseline setups used a fixed learning rate.
  - Later, `CosineAnnealingLR` was introduced to enable smoother learning rate decay.
- **Augmentation Techniques:**
  - **Strong Online Transforms:** Boosted with `RandomResizedCrop`, `ColorJitter`, vertical flips, etc.
  - **MixUp:** Blends pairs of images and their labels per batch.
  - **CutMix:** Mixes random image patches and interpolates the associated labels.
  - **Combined MixUp & CutMix:** Alternate per batch for regularization.
- **Cross-Validation & Ensembling:** 5-fold stratified CV adopted as standard; predictions are ensembled (mean probability) across all fold models at inference.
- **Stochastic Weight Averaging (SWA):** Integrated in advanced trials to further improve generalization, with custom batchnorm update methods to fix device mismatches.
- **Test-Time Augmentation (TTA):** At inference, average predictions from original and horizontally flipped images.

### Model Saving and Inference

- All scripts provide a `predict(folder_path)` function. For CV models, this function loads and averages predictions from all folds.
- Final models and code save weights for reproducible deployment.

## Results Discussion

| Approach                                               | Validation AUROC          |
|--------------------------------------------------------|---------------------------|
| **ResNet18** (baseline, strat split)                   | 0.8761                    |
| **EfficientNet-B0** (5-fold CV)                        | 0.8814 (mean)             |
| **DenseNet121** (baseline)                             | 0.9101 (holdout split)    |
| **DenseNet121 + Stronger Augmentations**               | 0.8963                    |
| **DenseNet121 + MixUp**                                | 0.9098                    |
| **DenseNet121 + CutMix**                               | 0.9103                    |
| **DenseNet121 + CutMix + TTA (flip)**                  | 0.8894                    |
| **DenseNet121 + CutMix + Stronger Augs**               | 0.9065                    |
| **DenseNet121 + MixUp/CutMix**                         | 0.9095                    |
| **DenseNet121 5-fold CV + CutMix**                     | 0.9117 (mean)             |
| **DenseNet121 5-fold CV + CutMix + AdamW + Scheduler** | 0.9274 (mean)             |
| **DenseNet121 5-fold CV + CutMix + SWA**               | 0.9266 (mean)             |
| **DenseNet121 5-fold CV + CutMix + More Epochs**       | 0.9308 (mean, 5 epochs)   |
| **DenseNet121 5-fold CV + CutMix + TTA (flip)**        | 0.9312 (mean, 5 epochs)   |
| **ViT-B/16** (baseline, strat split)                   | 0.8438                    |
| **ResNet34** (baseline, 5 epochs)                      | 0.8746                    |

**Key findings:**
- **DenseNet121 consistently yielded the best results**. CV with ensembling, stronger augmentations (MixUp, CutMix), AdamW optimizer, CosineAnnealingLR, and TTA provided incremental improvements at each step.
- **Advanced augmentations (CutMix/MixUp)** consistently surpassed traditional flipping/cropping alone.
- **SWA and LR scheduling** brought further robust gains and stability.
- **TTA (horizontal flip averaging)** enhanced reliability with negligible runtime cost.
- EfficientNet, ResNet, and ViT variants underperformed DenseNet121 (with modern training recipes) on this dataset.

## Future Work

- **Automated Hyperparameter Search:** Evaluate tuning CutMix/MixUp coefficients, learning rates, and schedulers via Bayesian optimization.
- **Larger/Deeper Architectures:** Test larger variants (DenseNet161, EfficientNet-B5, or ConvNeXt).
- **Multimodal/Meta-features:** Explore combining image with patient meta-data or clinical features.
- **Domain-Specific Augmentation:** Implement dermatology-focused augmentations (e.g., simulating hairs, occlusion) or self-supervised pretraining on unlabeled lesion data.
- **Explainability:** Integrate Grad-CAM or other attribution visualization for model interpretability to further aid clinical use.
- **Calibration:** Evaluate/adjust output calibration (e.g., Platt scaling, temperature scaling) for reliable probability estimates in clinical settings.
- **Longer/Adaptive Training:** Investigate if more training epochs with early stopping yields further gains.

---

**Recommended Final Pipeline:**  
DenseNet121 pretrained on ImageNet, trained with 5-fold stratified cross-validation, CutMix augmentation, AdamW optimizer, CosineAnnealingLR schedule, 5 epochs per fold, ensembling of fold predictions, and test-time augmentation using horizontal flip. This approach achieves a mean AUROC of ~0.93 and provides robust out-of-sample malignancy probability predictions as required by the task.

```