import os
import pandas as pd
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import timm
import torchvision.transforms as T


class SkinLesionDataset(Dataset):
    def __init__(self, df, img_dir, transform=None):
        self.df = df.reset_index(drop=True)
        self.img_dir = img_dir
        self.transform = transform
        tones = sorted(self.df["skin_tone"].unique())
        self.tone2idx = {v: i for i, v in enumerate(tones)}

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.img_dir, row["image_name"] + ".jpg")
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        tone = torch.tensor(self.tone2idx[row["skin_tone"]], dtype=torch.long)
        label = torch.tensor(row["label_bin"], dtype=torch.float32)
        return img, tone, label


class EfficientNetSkinTone(nn.Module):
    def __init__(self, num_tones):
        super().__init__()
        self.backbone = timm.create_model(
            "efficientnet_b0", pretrained=True, num_classes=0, global_pool="avg"
        )
        feat_dim = self.backbone.num_features
        self.tone_emb = nn.Embedding(num_tones, 16)
        self.classifier = nn.Sequential(
            nn.Linear(feat_dim + 16, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, 1),
        )

    def forward(self, x, tone):
        x_feat = self.backbone(x)
        t = self.tone_emb(tone)
        return self.classifier(torch.cat([x_feat, t], dim=1)).squeeze(1)


def predict(model_path: str, folder_path: str, device: str = None):
    """
    Load a trained model checkpoint and return malignancy probabilities for all images
    in folder_path. Returns a dict mapping filenames to probability [0-1].
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(model_path, map_location=device)
    num_tones = checkpoint["num_tones"]
    model = EfficientNetSkinTone(num_tones).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    transform = T.Compose(
        [
            T.Resize(224),
            T.CenterCrop(224),
            T.ToTensor(),
            T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )
    results = {}
    with torch.no_grad():
        for fname in os.listdir(folder_path):
            if not fname.lower().endswith((".jpg", ".png")):
                continue
            img = Image.open(os.path.join(folder_path, fname)).convert("RGB")
            x = transform(img).unsqueeze(0).to(device)
            tone = torch.tensor([0], device=device, dtype=torch.long)
            prob = torch.sigmoid(model(x, tone)).item()
            results[fname] = prob
    return results


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    df = pd.read_csv("./input/mydataset.csv")
    df["label_bin"] = (df["label"] == "malignant").astype(int)
    train_df, val_df = train_test_split(
        df, test_size=0.2, stratify=df["label_bin"], random_state=42
    )
    train_tf = T.Compose(
        [
            T.Resize(256),
            T.RandomResizedCrop(224),
            T.RandomHorizontalFlip(),
            T.RandomRotation(15),
            T.ToTensor(),
            T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )
    val_tf = T.Compose(
        [
            T.Resize(224),
            T.CenterCrop(224),
            T.ToTensor(),
            T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )
    train_ds = SkinLesionDataset(train_df, "./input/MyImages", transform=train_tf)
    val_ds = SkinLesionDataset(val_df, "./input/MyImages", transform=val_tf)
    num_tones = len(train_ds.tone2idx)
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, num_workers=4)

    model = EfficientNetSkinTone(num_tones).to(device)
    pos = train_df["label_bin"].sum()
    neg = len(train_df) - pos
    pos_weight = torch.tensor(neg / pos, device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    # Cosine annealing scheduler over 10 epochs
    epochs = 10
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        for imgs, tones, labels in train_loader:
            imgs, tones, labels = imgs.to(device), tones.to(device), labels.to(device)
            optimizer.zero_grad()
            logits = model(imgs, tones)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * imgs.size(0)
        scheduler.step()
        print(f"Epoch {epoch} train loss: {running_loss/len(train_ds):.4f}")

    model.eval()
    all_labels, all_probs = [], []
    with torch.no_grad():
        for imgs, tones, labels in val_loader:
            imgs, tones = imgs.to(device), tones.to(device)
            logits = model(imgs, tones)
            probs = torch.sigmoid(logits).cpu().numpy()
            all_probs.extend(probs.tolist())
            all_labels.extend(labels.numpy().tolist())
    val_auc = roc_auc_score(all_labels, all_probs)
    print(f"Validation AUROC: {val_auc:.4f}")

    os.makedirs("./working", exist_ok=True)
    ckpt = {"model_state_dict": model.state_dict(), "num_tones": num_tones}
    model_path = "./working/efficientnet_skintone.pth"
    torch.save(ckpt, model_path)

    # If test folder exists, generate submission.csv
    for folder_name in ("test", "test_images", "test_imgs"):
        test_folder = f"./input/{folder_name}"
        if os.path.isdir(test_folder):
            preds = predict(model_path, test_folder, device=device)
            df_sub = pd.DataFrame(
                list(preds.items()), columns=["image_name", "probability"]
            )
            df_sub.to_csv("./working/submission.csv", index=False)
            print(f"Saved submission.csv with {len(df_sub)} entries.")
            break


if __name__ == "__main__":
    main()
