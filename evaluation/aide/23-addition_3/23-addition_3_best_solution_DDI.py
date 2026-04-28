import os
import pandas as pd
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms, models
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score


class SkinLesionDataset(Dataset):
    def __init__(self, df, img_dir, transform=None):
        self.df = df.reset_index(drop=True)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        path = os.path.join(self.img_dir, row["image_name"] + ".jpg")
        img = Image.open(path).convert("RGB")
        data = self.transform(img) if self.transform else transforms.ToTensor()(img)
        label = torch.tensor(
            1.0 if row["label"] == "malignant" else 0.0, dtype=torch.float32
        )
        return data, label, row["skin_tone"], row["image_name"]


def mixup_data(x, y, alpha=0.4):
    lam = np.random.beta(alpha, alpha) if alpha > 0 else 1.0
    idx = torch.randperm(x.size(0), device=x.device)
    mixed = lam * x + (1 - lam) * x[idx]
    return mixed, y, y[idx], lam


def mixup_criterion(crit, pred, y_a, y_b, lam):
    return lam * crit(pred, y_a) + (1 - lam) * crit(pred, y_b)


def train_model(model, loader, criterion, optimizer, device, epochs, alpha):
    model.to(device)
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for imgs, labels, _, _ in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            inputs, y_a, y_b, lam = mixup_data(imgs, labels, alpha)
            optimizer.zero_grad()
            logits = model(inputs).squeeze(1)
            loss = mixup_criterion(criterion, logits, y_a, y_b, lam)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * imgs.size(0)
        print(
            f"Epoch {epoch+1}/{epochs} train loss: {running_loss/len(loader.dataset):.4f}"
        )
    return model


def evaluate_ensemble(models, loader, device):
    for m in models:
        m.eval()
    all_labels, all_probs = [], []
    with torch.no_grad():
        for imgs, labels, _, _ in loader:
            B, NC, C, H, W = imgs.size()
            x = imgs.view(B * NC, C, H, W).to(device)
            probs = []
            for m in models:
                logits = m(x).squeeze(1)
                p = torch.sigmoid(logits).view(B, NC).mean(dim=1)
                probs.append(p)
            avgp = torch.stack(probs, dim=0).mean(dim=0)
            all_labels.extend(labels.numpy())
            all_probs.extend(avgp.cpu().numpy())
    return roc_auc_score(all_labels, all_probs)


if __name__ == "__main__":
    # start change
    # IMG_DIR = "./input/MyImages"  # original
    IMG_DIR = "/home/anri21/be-fair/aide/MyData/MyImages"
    # end change
    # start change
    # CSV_PATH = "./input/mydataset.csv"  # original
    CSV_PATH = "/home/anri21/be-fair/aide/MyData/mydataset.csv"
    # end change
    # start change
    # TEST_DIR = "./input/test_images"  # original (replaced below in predict call section)
    # end change
    MODEL1_PATH = "./working/densenet121.pth"
    MODEL2_PATH = "./working/efficientnetb3.pth"
    BATCH_SIZE, LR, EPOCHS, ALPHA = 32, 1e-4, 3, 0.4
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    df = pd.read_csv(CSV_PATH)
    df = df[df["label"].isin(["malignant", "benign", "non-neoplastic"])].copy()
    train_df, val_df = train_test_split(
        df, test_size=0.2, stratify=df["label"], random_state=42
    )

    train_tf = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    to_tensor_norm = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize(mean, std)]
    )
    val_tf = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.TenCrop(224),
            transforms.Lambda(
                lambda crops: torch.stack([to_tensor_norm(c) for c in crops])
            ),
        ]
    )

    train_ds = SkinLesionDataset(train_df, IMG_DIR, train_tf)
    val_ds = SkinLesionDataset(val_df, IMG_DIR, val_tf)

    tones, counts = np.unique(train_df["skin_tone"], return_counts=True)
    freq = dict(zip(tones, counts))
    weights = [1.0 / freq[t] for t in train_df["skin_tone"].values]
    sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)

    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, sampler=sampler, num_workers=4
    )
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    # DenseNet121
    model1 = models.densenet121(pretrained=True)
    model1.classifier = nn.Linear(model1.classifier.in_features, 1)
    optimizer1 = torch.optim.Adam(model1.parameters(), lr=LR)
    criterion = nn.BCEWithLogitsLoss()
    model1 = train_model(
        model1, train_loader, criterion, optimizer1, device, EPOCHS, ALPHA
    )
    torch.save(model1.state_dict(), MODEL1_PATH)

    # EfficientNet-B3
    model2 = models.efficientnet_b3(pretrained=True)
    in_f = model2.classifier[1].in_features
    model2.classifier[1] = nn.Linear(in_f, 1)
    optimizer2 = torch.optim.Adam(model2.parameters(), lr=LR)
    model2 = train_model(
        model2, train_loader, criterion, optimizer2, device, EPOCHS, ALPHA
    )
    torch.save(model2.state_dict(), MODEL2_PATH)

    # Evaluate ensemble
    val_auc = evaluate_ensemble([model1, model2], val_loader, device)
    print(f"Validation AUROC (Ensemble): {val_auc:.4f}")

    # predict function
    def predict(image_folder):
        """Returns dataframe of malignancy probabilities by ensembling DenseNet121 and EfficientNetB3 with TenCrop TTA."""
        m1 = models.densenet121(pretrained=False)
        m1.classifier = nn.Linear(m1.classifier.in_features, 1)
        m1.load_state_dict(torch.load(MODEL1_PATH, map_location=device))
        m1.to(device).eval()
        m2 = models.efficientnet_b3(pretrained=False)
        in_f = m2.classifier[1].in_features
        m2.classifier[1] = nn.Linear(in_f, 1)
        m2.load_state_dict(torch.load(MODEL2_PATH, map_location=device))
        m2.to(device).eval()
        results = []
        for f in sorted(os.listdir(image_folder)):
            if f.lower().endswith((".jpg", ".jpeg", ".png")):
                img = Image.open(os.path.join(image_folder, f)).convert("RGB")
                crops = val_tf(img).view(-1, 3, 224, 224).to(device)
                with torch.no_grad():
                    p1 = torch.sigmoid(m1(crops).squeeze(1)).mean().item()
                    p2 = torch.sigmoid(m2(crops).squeeze(1)).mean().item()
                results.append(
                    {
                        "image_name": os.path.splitext(f)[0],
                        "malignancy_probability": (p1 + p2) / 2,
                    }
                )
        sub = pd.DataFrame(results)
        os.makedirs("./working", exist_ok=True)
        sub.to_csv("./working/submission.csv", index=False)
        return sub

    # start change
    # TEST_DIR = "./input/test_images"  # original
    # if os.path.isdir(TEST_DIR):       # original guard
    #     _ = predict(TEST_DIR)         # original
    _ddi_files = sorted(os.listdir("/home/anri21/be-fair/evaluation/DDI/images"))
    _sub = predict("/home/anri21/be-fair/evaluation/DDI/images")  # also saves ./working/submission.csv
    _pmap = dict(zip(_sub.iloc[:, 0], _sub.iloc[:, 1].astype(float)))
    pd.DataFrame({
        "DDI_file": _ddi_files,
        "predicted_probability": [_pmap[os.path.splitext(f)[0]] for f in _ddi_files]
    }).to_csv("./DDI_predictions.csv", index=False)
    # end change
