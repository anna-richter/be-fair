import os
import pandas as pd
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import torchvision.models as models


class SkinLesionDataset(Dataset):
    def __init__(self, df, img_dir, transform=None):
        self.df = df.reset_index(drop=True)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.img_dir, row.image_name + ".jpg")
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        label = float(row.label_bin)
        return image, label


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


def predict(model_path, image_folder, device="cpu"):
    """
    Load a trained model and return malignancy probabilities for all images in image_folder
    using Test-Time Augmentation (original + horizontal flip).
    Args:
        model_path (str): path to saved model .pth file
        image_folder (str): folder containing images (.jpg)
        device (str): 'cpu' or 'cuda'
    Returns:
        pandas.DataFrame with columns ['image_name','malignancy_prob']
    """
    device = torch.device(device)
    model = models.densenet121(pretrained=False)
    model.classifier = nn.Linear(model.classifier.in_features, 1)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device).eval()
    transform = T.Compose(
        [
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    results = []
    for fname in sorted(os.listdir(image_folder)):
        if not fname.lower().endswith(".jpg"):
            continue
        img = Image.open(os.path.join(image_folder, fname)).convert("RGB")
        x = transform(img).unsqueeze(0).to(device)
        x_flip = torch.flip(x, dims=[3])
        with torch.no_grad():
            p1 = torch.sigmoid(model(x))
            p2 = torch.sigmoid(model(x_flip))
            prob = ((p1 + p2) / 2).item()
        results.append({"image_name": fname.rsplit(".", 1)[0], "malignancy_prob": prob})
    return pd.DataFrame(results)


def main():
    seed = 42
    np.random.seed(seed)
    torch.manual_seed(seed)
    data_csv = "./input/mydataset.csv"
    img_dir = "./input/MyImages"
    df = pd.read_csv(data_csv)
    df["label_bin"] = (df["label"] == "malignant").astype(int)
    train_df, val_df = train_test_split(
        df, test_size=0.2, random_state=seed, stratify=df["label_bin"]
    )

    train_tf = T.Compose(
        [
            T.Resize((224, 224)),
            T.RandomHorizontalFlip(),
            T.RandomRotation(15),
            T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            T.RandomErasing(p=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3)),
        ]
    )
    val_tf = T.Compose(
        [
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    train_ds = SkinLesionDataset(train_df, img_dir, transform=train_tf)
    val_ds = SkinLesionDataset(val_df, img_dir, transform=val_tf)
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, num_workers=4)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = models.densenet121(pretrained=True)
    model.classifier = nn.Linear(model.classifier.in_features, 1)
    model.to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    epochs = 5
    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.unsqueeze(1).to(device)
            imgs_mix, y_a, y_b, lam = mixup_data(imgs, labels, alpha=0.2)
            optimizer.zero_grad()
            logits = model(imgs_mix)
            loss = lam * criterion(logits, y_a) + (1 - lam) * criterion(logits, y_b)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * imgs.size(0)
        print(f"Epoch {epoch}/{epochs} Train Loss: {running_loss/len(train_ds):.4f}")

    # Validation with TTA
    model.eval()
    all_probs, all_labels = [], []
    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs = imgs.to(device)
            imgs_flip = torch.flip(imgs, dims=[3])
            p1 = torch.sigmoid(model(imgs))
            p2 = torch.sigmoid(model(imgs_flip))
            probs = ((p1 + p2) / 2).cpu().numpy().reshape(-1)
            all_probs.extend(probs)
            all_labels.extend(labels.numpy())
    auc = roc_auc_score(all_labels, all_probs)
    print(f"Validation TTA AUROC with Mixup + RandomErasing: {auc:.4f}")

    os.makedirs("./working", exist_ok=True)
    model_path = "./working/densenet121_mixup_random_erasing.pth"
    torch.save(model.state_dict(), model_path)

    test_folder = "./input/test_images"
    if os.path.isdir(test_folder):
        df_sub = predict(model_path, test_folder, device=device)
        df_sub.to_csv("./working/submission.csv", index=False)


if __name__ == "__main__":
    main()
