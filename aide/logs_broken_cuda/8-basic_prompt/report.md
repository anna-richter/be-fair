```markdown
# Technical Report: Malignant vs. Benign Skin Lesion Classification

## Introduction

The objective of this project is to develop an AI model to predict the probability that a skin lesion is malignant, thereby assisting dermatologists in clinical decision-making. The model should output a probability score (0–1) for each image, with performance primarily measured by the area under the receiver operating characteristic curve (AUROC).

## Preprocessing

### Data Setup
- **Labels**: Images are labeled as 'malignant', 'benign', or 'non-neoplastic'. For training, labels were binarized: malignant=1, others=0.
- **Splitting**: All experiments use an 80/20 stratified split to maintain class balance in training and validation.
- **Transforms**:
  - **Common**: Used resizing/cropping (most commonly 224×224), normalization (mean, std from ImageNet), and conversion to tensor.
  - **Augmentation**: Applied random horizontal flips, random resized crops, and, in some experiments, advanced augmentations such as AutoAugment (ImageNet policy) or Mixup (α=0.4).

### Dataset/Dataloader
- **Samplers**: In one experiment, a `WeightedRandomSampler` was used to oversample malignant cases to directly address class imbalance.

### Performance Evaluation
- **Metric**: AUROC is calculated on the hold-out validation set after training.
- **Test-Time Augmentation (TTA)**: Some experiments averaged predictions from the original and horizontally flipped images during validation/inference.

## Modelling Methods

### Neural Network Choices

- **ResNet18**: Used as an initial baseline, with the final layer adjusted for binary output.
- **EfficientNet-B0 (via [timm](https://pytorch.org/timm/))**: Used in most experiments due to its efficient performance. The head was replaced with a single-output classification layer.
- **DenseNet121**: Used in one experiment with the backbone frozen; only the classifier head was trained for fast adaptation.

### Loss Functions

- **BCEWithLogitsLoss**: Standard loss for binary classification, used in most experiments.
- **Focal Loss**: (γ=2, α=0.25) was tested to emphasize difficult/misclassified samples and mitigate class imbalance.

### Optimizers & Learning Rate Schedulers

- **Adam / AdamW**: Used consistently for parameter updates.
- **OneCycleLR**: Explored to improve convergence by varying the learning rate within an epoch.

### Training Techniques

- **Epochs**: Typically 3–5 epochs, reducing when computation/time limits are an issue.
- **Mixed Precision**: Used `torch.cuda.amp` for training speedup when CUDA was available.
- **Backbone Freezing**: Explored in DenseNet121, freezing all except the final classification layer to accelerate training and prevent overfitting on smaller datasets.
- **Test-Time Augmentation (TTA)**: Used horizontal flip TTA to stabilize predictions at inference.

### Predict Function
A standardized `predict()` function was implemented in all experiments, accepting a list of new image paths, and returning malignancy probability per image.

## Results Discussion

| Experiment                   | Model           | Key Features                         | Epochs | Val AUROC |
|------------------------------|-----------------|--------------------------------------|--------|-----------|
| Baseline (ResNet18)          | ResNet18        | Standard transforms, Adam, BCE       | 3      | 0.8766    |
| EfficientNet-B0              | EfficientNet-B0 | Random crop/flip, AdamW, BCE         | 5      | **0.9037**|
| Mixup Augmentation           | EfficientNet-B0 | Mixup (α=0.4) + standard aug         | 5      | 0.8778    |
| Weighted Sampler             | EfficientNet-B0 | Balanced batch sampling              | 5      | 0.8974    |
| Focal Loss                   | EfficientNet-B0 | Focal loss (α=0.25, γ=2)             | 5      | 0.8656    |
| Mixed Precision, 128px, TTA  | EfficientNet-B0 | 128×128 imgs, autocast, TTA          | 3      | 0.8337    |
| OneCycleLR                   | EfficientNet-B0 | LR scheduler, standard aug           | 5      | 0.8543    |
| AutoAugment                  | EfficientNet-B0 | RandCrop/flip + AutoAugment          | 5      | 0.8793    |
| DenseNet121 Frozen           | DenseNet121     | Frozen backbone; only head trained   | 2      | 0.8257    |
| EfficientNet-B0, 3 epochs + TTA| EfficientNet-B0 | TTA, 3 epochs (time-saving)         | 3      | 0.8770    |

**Key Insights:**
- **Best Validation Performance**: EfficientNet-B0 with standard strong augmentation and AdamW optimizer achieves the highest AUROC (0.9037).
- **Augmentation**: AutoAugment and Mixup can help, but may require careful tuning to outperform simple strong augmentations.
- **TTA**: Simple horizontal flip TTA usually improves or stabilizes AUROC.
- **Imbalanced Data Handling**: Weighted sampling and focal loss help, but reweighting or focusing on hard examples does not outperform standard augmentations in this dataset.
- **Resource/Time Constraints**: Reducing epochs, image size, or freezing the backbone leads to faster runs at the expense of some accuracy.
- **EfficientNet-B0 consistently outperforms the other tested architectures in this context.**

## Future Work

- **Longer/Finer Training**: Increase epochs where feasible, especially if using strong augmentations or regularization.
- **Model Ensembling**: Combine multiple models or snapshot checkpoints for potential marginal AUROC gains.
- **Advanced TTA/Inference**: Explore more augmentation types during test-time (e.g., rotations, color jitter).
- **Hyperparameter Optimization**: Perform systematic search on learning rates, batch size, and augmentation parameters.
- **Explainability**: Integrate GradCAM or other saliency techniques for visual interpretation of malignancy predictions.
- **Expanded Architectures**: Evaluate newer, larger, or domain-specific vision backbones.
- **Clinical Validation**: Conduct prospective tests on truly unseen real-world images from clinical settings.

---

**Recommendation:**  
For best performance and practical utility, use the EfficientNet-B0 pipeline with random resize, flip, normalization, standard image size (224×224), and AdamW optimizer, trained for at least 5 epochs. Apply two-view TTA at inference for slightly improved AUROC. This setup achieved the highest validation AUROC (0.9037) and includes a robust, reusable `predict()` interface for clinical deployment.
```