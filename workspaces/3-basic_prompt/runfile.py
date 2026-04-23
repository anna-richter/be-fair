import os
import pandas as pd
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models


class SkinLesionDataset(Dataset):
    def __init__(self, df, image_dir, transform=None):
        self.df = df.reset_index(drop=True)
        self.image_dir = image_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.loc[idx]
        img_path = os.path.join(self.image_dir, row["image_name"] + ".jpg")
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        label = torch.tensor(row["binary_label"], dtype=torch.long)
        return image, label


def predict(model_path: str, image_dir: str) -> pd.DataFrame:
    """
    Loads a trained model and returns a DataFrame with image names and
    the probability of being malignant.
    Args:
        model_path: path to the saved model .pth file
        image_dir: directory containing .jpg images
    Returns:
        DataFrame with columns ['image_name', 'malignant_prob']
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = models.resnet18(pretrained=False)
    model.fc = nn.Linear(model.fc.in_features, 2)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device).eval()

    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    results = []
    for fname in sorted(os.listdir(image_dir)):
        if not fname.lower().endswith(".jpg"):
            continue
        img = Image.open(os.path.join(image_dir, fname)).convert("RGB")
        x = transform(img).unsqueeze(0).to(device)
        with torch.no_grad():
            logits = model(x)
            prob = torch.softmax(logits, dim=1)[0, 1].item()
        results.append(
            {"image_name": os.path.splitext(fname)[0], "malignant_prob": prob}
        )
    return pd.DataFrame(results)


if __name__ == "__main__":
    # Paths
    DATA_CSV = "./input/mydataset.csv"
    IMAGE_DIR = "./input/MyImages"
    MODEL_PATH = "./working/resnet18_skin_224.pth"

    # Load and prepare data
    df = pd.read_csv(DATA_CSV)
    df["binary_label"] = (df["label"] == "malignant").astype(int)
    train_df, val_df = train_test_split(
        df, test_size=0.2, stratify=df["binary_label"], random_state=42
    )

    # Transforms and datasets with higher resolution
    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    train_ds = SkinLesionDataset(train_df, IMAGE_DIR, transform)
    val_ds = SkinLesionDataset(val_df, IMAGE_DIR, transform)
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, num_workers=4)

    # Model setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = models.resnet18(pretrained=True)
    model.fc = nn.Linear(model.fc.in_features, 2)
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    # Training loop
    epochs = 3
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * imgs.size(0)
        avg_loss = running_loss / len(train_ds)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")

    # Save model
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    torch.save(model.state_dict(), MODEL_PATH)

    # Validation and ROC AUC
    model.eval()
    all_probs, all_labels = [], []
    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs = imgs.to(device)
            logits = model(imgs)
            probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
            all_probs.extend(probs)
            all_labels.extend(labels.numpy())
    auc = roc_auc_score(all_labels, all_probs)
    print(f"Validation ROC AUC: {auc:.4f}")
