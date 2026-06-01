import os
import pandas as pd
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import timm


class SkinLesionDataset(Dataset):
    def __init__(self, df, image_dir, transform):
        self.df = df.reset_index(drop=True)
        self.image_dir = image_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.image_dir, f"{row.image_name}.jpg")
        img = Image.open(img_path).convert("RGB")
        img = self.transform(img)
        label = torch.tensor(row.label, dtype=torch.float32)
        return img, label


def mixup_data(x, y, alpha=0.2):
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.0
    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(x.device)
    mixed_x = lam * x + (1 - lam) * x[index]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def predict(model, image_paths, base_transform, tta_transforms, device):
    """
    Predict malignancy probability for a list of image file paths using multiple TTA transforms.
    Args:
        model: trained PyTorch model
        image_paths: list of file paths to images
        base_transform: transform for resizing/normalizing without flips
        tta_transforms: list of transforms (each deterministic) to apply for TTA
        device: torch device
    Returns:
        probs: list of float probabilities between 0 and 1
    """
    model.eval()
    probs = []
    with torch.no_grad():
        for path in image_paths:
            img = Image.open(path).convert("RGB")
            # apply each TTA transform
            p_list = []
            for t in tta_transforms:
                inp = t(img).unsqueeze(0).to(device)
                logit = model(inp)
                p = torch.sigmoid(logit).item()
                p_list.append(p)
            probs.append(float(np.mean(p_list)))
    return probs


def main():
    # Settings
    DATA_CSV = "./input/mydataset.csv"
    IMAGE_DIR = "./input/MyImages"
    TEST_DIR = "./input/TestImages"
    SEED = 42
    BATCH_SIZE = 32
    NUM_EPOCHS = 5
    LR = 1e-4
    MIXUP_ALPHA = 0.2
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    # Load and filter data
    df = pd.read_csv(DATA_CSV)
    df = df[df.label.isin(["benign", "malignant"])].copy()
    df["label"] = df.label.map({"benign": 0, "malignant": 1})

    # Split train/validation
    train_df, val_df = train_test_split(
        df, test_size=0.2, random_state=SEED, stratify=df.label
    )

    # Transforms
    train_transform = T.Compose(
        [
            T.Resize((224, 224)),
            T.RandomHorizontalFlip(),
            T.RandomVerticalFlip(),
            T.RandomRotation(20),
            T.ToTensor(),
            T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )
    base_transform = T.Compose(
        [
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )
    tta_h = T.Compose(
        [
            T.Resize((224, 224)),
            T.RandomHorizontalFlip(p=1.0),
            T.ToTensor(),
            T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )
    tta_v = T.Compose(
        [
            T.Resize((224, 224)),
            T.RandomVerticalFlip(p=1.0),
            T.ToTensor(),
            T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )
    tta_hv = T.Compose(
        [
            T.Resize((224, 224)),
            T.RandomHorizontalFlip(p=1.0),
            T.RandomVerticalFlip(p=1.0),
            T.ToTensor(),
            T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )
    tta_transforms = [base_transform, tta_h, tta_v, tta_hv]

    # Datasets and loaders
    train_ds = SkinLesionDataset(train_df, IMAGE_DIR, train_transform)
    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True
    )

    # Model
    model = timm.create_model("efficientnet_b0", pretrained=True)
    in_f = model.classifier.in_features
    model.classifier = nn.Linear(in_f, 1)
    model = model.to(DEVICE)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    # Training loop with MixUp
    for epoch in range(NUM_EPOCHS):
        model.train()
        running_loss = 0.0
        for imgs, labels in train_loader:
            imgs = imgs.to(DEVICE)
            labels = labels.unsqueeze(1).to(DEVICE)
            imgs_mix, y_a, y_b, lam = mixup_data(imgs, labels, MIXUP_ALPHA)
            optimizer.zero_grad()
            logits = model(imgs_mix)
            loss = lam * criterion(logits, y_a) + (1 - lam) * criterion(logits, y_b)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * imgs.size(0)
        print(
            f"Epoch {epoch+1}/{NUM_EPOCHS}, Loss: {running_loss/len(train_loader.dataset):.4f}"
        )

    # Validation with multi-view TTA
    val_paths = [os.path.join(IMAGE_DIR, f"{name}.jpg") for name in val_df.image_name]
    val_labels = val_df.label.tolist()
    val_probs = predict(model, val_paths, base_transform, tta_transforms, DEVICE)
    val_roc = roc_auc_score(val_labels, val_probs)
    print(f"Validation AUROC with multi-view TTA: {val_roc:.4f}")

    # Test-time predictions and submission
    if os.path.isdir(TEST_DIR):
        test_paths = sorted(
            [
                os.path.join(TEST_DIR, f)
                for f in os.listdir(TEST_DIR)
                if f.endswith(".jpg")
            ]
        )
        test_probs = predict(model, test_paths, base_transform, tta_transforms, DEVICE)
        submission = pd.DataFrame(
            {
                "image_name": [
                    os.path.splitext(os.path.basename(p))[0] for p in test_paths
                ],
                "malignant_probability": test_probs,
            }
        )
        os.makedirs("./working", exist_ok=True)
        submission.to_csv("./working/submission.csv", index=False)


if __name__ == "__main__":
    main()
