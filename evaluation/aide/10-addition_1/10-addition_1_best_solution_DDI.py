import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image
from tqdm import tqdm


class SkinDataset(Dataset):
    def __init__(self, df, img_dir, transform):
        self.df = df.reset_index(drop=True)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.img_dir, row["image_name"] + ".jpg")
        image = Image.open(img_path).convert("RGB")
        img = self.transform(image)
        label = 1 if row["label"] == "malignant" else 0
        return img, label


def predict(folder_path):
    """
    Predict malignancy probability for all images in a folder using TenCrop TTA with EfficientNet-B3.
    Args:
        folder_path (str): Path to folder containing images (jpg/png).
    Returns:
        dict: mapping image filename to malignancy probability (0-1).
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = models.efficientnet_b3(pretrained=False)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, 1)
    model.load_state_dict(
        torch.load("./working/efficientnet_b3.pth", map_location=device)
    )
    model.to(device).eval()

    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    tta_transform = transforms.Compose(
        [
            transforms.Resize(320),
            transforms.TenCrop(300),
            transforms.Lambda(
                lambda crops: torch.stack(
                    [
                        transforms.Normalize(mean, std)(transforms.ToTensor()(crop))
                        for crop in crops
                    ]
                )
            ),
        ]
    )

    results = {}
    with torch.no_grad():
        for fname in os.listdir(folder_path):
            if not fname.lower().endswith((".jpg", ".png")):
                continue
            img = Image.open(os.path.join(folder_path, fname)).convert("RGB")
            crops = tta_transform(img)  # [10,3,300,300]
            bs, c, h, w = crops.size()
            inputs = crops.to(device)
            outputs = model(inputs)  # [10,1]
            probs = torch.sigmoid(outputs).cpu().view(bs)
            results[fname] = probs.mean().item()
    return results


def main():
    # Load and split data
    # start change
    # df = pd.read_csv("./input/mydataset.csv")  # original
    df = pd.read_csv("/home/anri21/be-fair/aide/MyData/mydataset.csv")
    # end change
    df["label"] = df["label"].astype(str)
    train_df, val_df = train_test_split(
        df, stratify=df["label"], test_size=0.2, random_state=42
    )

    # Transforms
    train_transform = transforms.Compose(
        [
            transforms.RandomResizedCrop(300),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(0.2, 0.2, 0.2),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    val_transform = transforms.Compose(
        [
            transforms.Resize(320),
            transforms.TenCrop(300),
            transforms.Lambda(
                lambda crops: torch.stack(
                    [
                        transforms.Normalize(mean, std)(transforms.ToTensor()(crop))
                        for crop in crops
                    ]
                )
            ),
        ]
    )

    # Datasets and loaders
    # start change
    # train_ds = SkinDataset(train_df, "./input/MyImages", train_transform)  # original
    # val_ds = SkinDataset(val_df, "./input/MyImages", val_transform)  # original
    train_ds = SkinDataset(train_df, "/home/anri21/be-fair/aide/MyData/MyImages", train_transform)
    val_ds = SkinDataset(val_df, "/home/anri21/be-fair/aide/MyData/MyImages", val_transform)
    # end change
    train_loader = DataLoader(
        train_ds, batch_size=32, shuffle=True, num_workers=4, pin_memory=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=16, shuffle=False, num_workers=4, pin_memory=True
    )

    # Model, loss, optimizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = models.efficientnet_b3(pretrained=True)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, 1)
    model.to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    # Training loop
    for epoch in range(5):
        model.train()
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/5")
        for images, labels in loop:
            images = images.to(device)
            labels = labels.float().unsqueeze(1).to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            loop.set_postfix(loss=loss.item())

    # Validation with TTA
    model.eval()
    all_labels, all_probs = [], []
    with torch.no_grad():
        for images, labels in val_loader:
            bs, ncrops, c, h, w = images.size()
            imgs = images.view(-1, c, h, w).to(device)
            outputs = model(imgs)
            probs = torch.sigmoid(outputs).cpu().view(bs, ncrops).mean(dim=1).numpy()
            all_probs.extend(probs.tolist())
            all_labels.extend(labels.numpy().tolist())
    auc = roc_auc_score(all_labels, all_probs)
    print(f"Validation AUROC (EfficientNet-B3 + TTA): {auc:.4f}")

    # Save model
    os.makedirs("./working", exist_ok=True)
    torch.save(model.state_dict(), "./working/efficientnet_b3.pth")


if __name__ == "__main__":
    main()
    # start change
    _ddi_files = sorted(os.listdir("/home/anri21/be-fair/evaluation/DDI/images"))
    _preds = predict("/home/anri21/be-fair/evaluation/DDI/images")
    pd.DataFrame({
        "DDI_file": _ddi_files,
        "predicted_probability": [float(_preds[f]) for f in _ddi_files]
    }).to_csv("./DDI_predictions.csv", index=False)
    # end change
