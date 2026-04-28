import os
import pandas as pd
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms, models
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score


class SkinLesionDataset(Dataset):
    def __init__(self, df, img_dir, transform):
        self.df = df.reset_index(drop=True)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = Image.open(os.path.join(self.img_dir, row.image_name + ".jpg")).convert(
            "RGB"
        )
        x = self.transform(img)
        y = torch.tensor(row.label, dtype=torch.float32)
        return x, y


def train_fold(model, loader, criterion, optimizer, device):
    model.train()
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        preds = model(imgs).squeeze(1)
        loss = criterion(preds, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


def eval_fold(model, loader, device):
    model.eval()
    ys, ps = [], []
    with torch.no_grad():
        for imgs, labels in loader:
            bs, nc, c, h, w = imgs.shape
            imgs = imgs.view(bs * nc, c, h, w).to(device)
            logits = model(imgs).squeeze(1)
            probs = torch.sigmoid(logits).view(bs, nc).mean(dim=1)
            ps.extend(probs.cpu().numpy())
            ys.extend(labels.numpy())
    return roc_auc_score(ys, ps)


def predict(model_path, img_folder):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    resize = transforms.Resize((224, 224))
    ten = transforms.TenCrop((224, 224))
    to_tensor = transforms.ToTensor()
    normalize = transforms.Normalize(mean, std)
    model = models.densenet121(pretrained=False)
    model.classifier = nn.Linear(model.classifier.in_features, 1)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device).eval()
    results = []
    for fname in sorted(os.listdir(img_folder)):
        if not fname.lower().endswith((".jpg", ".jpeg", ".png")):
            continue
        img = Image.open(os.path.join(img_folder, fname)).convert("RGB")
        img = resize(img)
        crops = ten(img)
        batch = torch.stack([normalize(to_tensor(c)) for c in crops]).to(device)
        with torch.no_grad():
            logits = model(batch).squeeze(1)
            prob = torch.sigmoid(logits).mean().item()
        results.append(
            {"image_name": os.path.splitext(fname)[0], "malignancy_probability": prob}
        )
    return pd.DataFrame(results)


def main():
    # start change
    # data_csv = "./input/mydataset.csv"  # original
    data_csv = "/home/anri21/be-fair/aide/MyData/mydataset.csv"
    # end change
    # start change
    # img_dir = "./input/MyImages"  # original
    img_dir = "/home/anri21/be-fair/aide/MyData/MyImages"
    # end change
    df = pd.read_csv(data_csv)
    df["label"] = (df.label == "malignant").astype(int)
    tone_counts = df.skin_tone.value_counts().to_dict()
    df["weight"] = df.skin_tone.map(lambda t: 1.0 / tone_counts.get(t, 1))
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]

    train_tf = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
            transforms.RandomErasing(p=0.5),
        ]
    )
    val_tf = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.TenCrop((224, 224)),
            transforms.Lambda(
                lambda crops: torch.stack(
                    [
                        transforms.Normalize(mean, std)(transforms.ToTensor()(c))
                        for c in crops
                    ]
                )
            ),
        ]
    )

    cv_scores = []
    num_epochs = 10  # increased from 5 to 10
    for fold, (tr_idx, val_idx) in enumerate(skf.split(df, df.label)):
        df_tr, df_val = df.iloc[tr_idx], df.iloc[val_idx]
        sampler = WeightedRandomSampler(
            df_tr.weight.values, len(df_tr), replacement=True
        )
        tr_loader = DataLoader(
            SkinLesionDataset(df_tr, img_dir, train_tf),
            batch_size=32,
            sampler=sampler,
            num_workers=4,
        )
        val_loader = DataLoader(
            SkinLesionDataset(df_val, img_dir, val_tf),
            batch_size=16,
            shuffle=False,
            num_workers=4,
        )

        model = models.densenet121(pretrained=True)
        model.classifier = nn.Linear(model.classifier.in_features, 1)
        model.to(device)
        criterion = nn.BCEWithLogitsLoss()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=num_epochs
        )

        for epoch in range(num_epochs):
            train_fold(model, tr_loader, criterion, optimizer, device)
            scheduler.step()

        score = eval_fold(model, val_loader, device)
        print(f"Fold {fold+1} AUROC: {score:.4f}")
        cv_scores.append(score)

    mean_score = np.mean(cv_scores)
    print(f"Mean CV AUROC: {mean_score:.4f}")

    # Retrain on full data
    full_sampler = WeightedRandomSampler(df.weight.values, len(df), replacement=True)
    full_loader = DataLoader(
        SkinLesionDataset(df, img_dir, train_tf),
        batch_size=32,
        sampler=full_sampler,
        num_workers=4,
    )
    model = models.densenet121(pretrained=True)
    model.classifier = nn.Linear(model.classifier.in_features, 1)
    model.to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    for epoch in range(num_epochs):
        train_fold(model, full_loader, criterion, optimizer, device)
        scheduler.step()

    os.makedirs("./working", exist_ok=True)
    model_path = "./working/model.pth"
    torch.save(model.state_dict(), model_path)
    print("Model saved.")

    # start change
    # sub = predict(model_path, img_dir)  # original — predicts on training images
    # sub.to_csv("./working/submission.csv", index=False)  # original
    _ddi_files = sorted(os.listdir("/home/anri21/be-fair/evaluation/DDI/images"))
    _sub = predict(model_path, "/home/anri21/be-fair/evaluation/DDI/images")
    _pmap = dict(zip(_sub.iloc[:, 0], _sub.iloc[:, 1].astype(float)))
    pd.DataFrame({
        "DDI_file": _ddi_files,
        "predicted_probability": [_pmap[os.path.splitext(f)[0]] for f in _ddi_files]
    }).to_csv("./DDI_predictions.csv", index=False)
    # end change


if __name__ == "__main__":
    main()
