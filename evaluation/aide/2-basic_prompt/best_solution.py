import os
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR


class SkinDataset(Dataset):
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
        if self.transform:
            img = self.transform(img)
        label = torch.tensor(row["target"], dtype=torch.float32)
        return img, label


def rand_bbox(size, lam):
    _, _, H, W = size
    cut_rat = np.sqrt(1.0 - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)
    cx = np.random.randint(W)
    cy = np.random.randint(H)
    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)
    return bbx1, bby1, bbx2, bby2


def predict(folder_path):
    """
    Predict malignancy probabilities by averaging all fold models with horizontal-flip TTA.
    Returns a DataFrame with columns: image_name, probability.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tf = transforms.Compose(
        [
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    # load all fold models
    models_list = []
    for f in sorted(os.listdir("working")):
        if f.startswith("model_fold_") and f.endswith(".pth"):
            m = models.densenet121(pretrained=False)
            m.classifier = nn.Linear(m.classifier.in_features, 1)
            m.load_state_dict(
                torch.load(os.path.join("working", f), map_location=device)
            )
            m.to(device).eval()
            models_list.append(m)
    results = []
    for fname in sorted(os.listdir(folder_path)):
        if not fname.lower().endswith((".jpg", ".jpeg", ".png")):
            continue
        img = Image.open(os.path.join(folder_path, fname)).convert("RGB")
        t = tf(img).unsqueeze(0).to(device)  # shape [1,3,H,W]
        t_flip = torch.flip(t, dims=[3])
        probs_per_model = []
        with torch.no_grad():
            for m in models_list:
                logit1 = m(t)
                logit2 = m(t_flip)
                avg_logit = (logit1 + logit2) / 2.0
                probs_per_model.append(torch.sigmoid(avg_logit).item())
        avgp = float(np.mean(probs_per_model))
        results.append({"image_name": os.path.splitext(fname)[0], "probability": avgp})
    return pd.DataFrame(results)


def main():
    np.random.seed(42)
    torch.manual_seed(42)
    csv_path = os.path.join("input", "mydataset.csv")
    img_dir = os.path.join("input", "MyImages")
    df = pd.read_csv(csv_path)
    df["target"] = (df["label"] == "malignant").astype(int)

    # transforms
    train_tf = transforms.Compose(
        [
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cutmix_prob, cutmix_alpha = 0.5, 1.0
    batch_size = 32
    epochs = 5
    os.makedirs("working", exist_ok=True)
    fold_aucs = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(df, df["target"])):
        train_df, val_df = df.iloc[train_idx], df.iloc[val_idx]
        train_ds = SkinDataset(train_df, img_dir, train_tf)
        val_ds = SkinDataset(val_df, img_dir, train_tf)
        tl = DataLoader(
            train_ds,
            batch_size=batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True,
        )
        vl = DataLoader(
            val_ds, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True
        )

        model = models.densenet121(pretrained=True)
        model.classifier = nn.Linear(model.classifier.in_features, 1)
        model.to(device)

        opt = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
        scheduler = CosineAnnealingLR(opt, T_max=epochs, eta_min=1e-6)
        crit = nn.BCEWithLogitsLoss()

        # training
        for epoch in range(epochs):
            model.train()
            for imgs, targets in tl:
                imgs = imgs.to(device)
                targets = targets.to(device).unsqueeze(1)
                r = np.random.rand()
                opt.zero_grad()
                if r < cutmix_prob:
                    lam = np.random.beta(cutmix_alpha, cutmix_alpha)
                    idx2 = torch.randperm(imgs.size(0)).to(device)
                    ta, tb = targets, targets[idx2]
                    bbx1, bby1, bbx2, bby2 = rand_bbox(imgs.size(), lam)
                    imgs[:, :, bby1:bby2, bbx1:bbx2] = imgs[
                        idx2, :, bby1:bby2, bbx1:bbx2
                    ]
                    lam = 1 - (
                        (bbx2 - bbx1) * (bby2 - bby1) / (imgs.size(-1) * imgs.size(-2))
                    )
                    logits = model(imgs)
                    loss = crit(logits, ta) * lam + crit(logits, tb) * (1 - lam)
                else:
                    logits = model(imgs)
                    loss = crit(logits, targets)
                loss.backward()
                opt.step()
            scheduler.step()

        # validation with horizontal-flip TTA
        model.eval()
        all_p, all_t = [], []
        with torch.no_grad():
            for imgs, targets in vl:
                imgs = imgs.to(device)
                imgs_flip = torch.flip(imgs, dims=[3])
                logits1 = model(imgs)
                logits2 = model(imgs_flip)
                logits_avg = (logits1 + logits2) / 2.0
                probs = torch.sigmoid(logits_avg).cpu().numpy().flatten()
                all_p.extend(probs.tolist())
                all_t.extend(targets.numpy().flatten().tolist())
        auc = roc_auc_score(all_t, all_p)
        print(f"Fold {fold} ROC AUC: {auc:.4f}")
        fold_aucs.append(auc)
        torch.save(model.state_dict(), f"working/model_fold_{fold}.pth")

    mean_auc = np.mean(fold_aucs)
    print(f"Mean ROC AUC: {mean_auc:.4f}")

    # predict on test if exists
    test_folder = os.path.join("input", "test")
    if os.path.exists(test_folder):
        sub = predict(test_folder)
        sub.to_csv(os.path.join("working", "submission.csv"), index=False)


if __name__ == "__main__":
    main()
