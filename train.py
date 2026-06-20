import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import numpy as np
from PIL import Image, ImageFile

# Bozuk görüntüleri atla
ImageFile.LOAD_TRUNCATED_IMAGES = True
if __name__ == '__main__':

    # ── Config ──────────────────────────────────────────────
    DATA_DIR      = r"D:\Projeler\Python\Bone_Fracture_Detection\Bone_Fracture_Binary_Classification\Bone_Fracture_Binary_Classification"
    BATCH         = 32
    EPOCHS_FROZEN = 5
    EPOCHS_FT     = 8
    LR_FROZEN     = 1e-3
    LR_FT         = 1e-4
    IMG_SIZE      = 224
    DEVICE        = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {DEVICE}")

    # ── Transforms ──────────────────────────────────────────
    train_tf = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])
    val_tf = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])

    # ── Datasets ────────────────────────────────────────────
    train_ds = datasets.ImageFolder(os.path.join(DATA_DIR, "train"), transform=train_tf)
    val_ds   = datasets.ImageFolder(os.path.join(DATA_DIR, "val"),   transform=val_tf)
    test_ds  = datasets.ImageFolder(os.path.join(DATA_DIR, "test"),  transform=val_tf)

    train_loader = DataLoader(train_ds, batch_size=BATCH, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH, shuffle=False, num_workers=0)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH, shuffle=False, num_workers=0)

    print(f"Sınıflar: {train_ds.classes}")

    # ── Model ───────────────────────────────────────────────
    model = models.efficientnet_b0(weights="IMAGENET1K_V1")

    for param in model.parameters():
        param.requires_grad = False

    model.classifier = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(model.classifier[1].in_features, 2)
    )
    model = model.to(DEVICE)

    # ── Train / Eval fonksiyonları ───────────────────────────
    def train_one_epoch(loader, optimizer, criterion):
        model.train()
        total_loss, correct = 0, 0
        for imgs, labels in loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            out  = model(imgs)
            loss = criterion(out, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * imgs.size(0)
            correct    += (out.argmax(1) == labels).sum().item()
        return total_loss / len(loader.dataset), correct / len(loader.dataset)

    def evaluate(loader):
        model.eval()
        total_loss, correct = 0, 0
        crit = nn.CrossEntropyLoss()
        with torch.no_grad():
            for imgs, labels in loader:
                imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
                out  = model(imgs)
                loss = crit(out, labels)
                total_loss += loss.item() * imgs.size(0)
                correct    += (out.argmax(1) == labels).sum().item()
        return total_loss / len(loader.dataset), correct / len(loader.dataset)

    # ── Aşama 1: Frozen backbone ─────────────────────────────
    print("\n── Aşama 1: Frozen backbone eğitimi ──")
    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(model.classifier.parameters(), lr=LR_FROZEN)
    scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS_FROZEN)

    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

    for epoch in range(1, EPOCHS_FROZEN + 1):
        tl, ta = train_one_epoch(train_loader, optimizer, criterion)
        vl, va = evaluate(val_loader)
        scheduler.step()
        history["train_loss"].append(tl)
        history["val_loss"].append(vl)
        history["train_acc"].append(ta)
        history["val_acc"].append(va)
        print(f"Epoch {epoch}/{EPOCHS_FROZEN} — loss: {tl:.4f} | val_loss: {vl:.4f} | val_acc: {va:.4f}")

    # ── Aşama 2: Fine-tune ───────────────────────────────────
    print("\n── Aşama 2: Fine-tune ──")
    for param in model.parameters():
        param.requires_grad = True

    optimizer = AdamW(model.parameters(), lr=LR_FT, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS_FT)

    for epoch in range(1, EPOCHS_FT + 1):
        tl, ta = train_one_epoch(train_loader, optimizer, criterion)
        vl, va = evaluate(val_loader)
        scheduler.step()
        history["train_loss"].append(tl)
        history["val_loss"].append(vl)
        history["train_acc"].append(ta)
        history["val_acc"].append(va)
        print(f"Epoch {epoch}/{EPOCHS_FT} — loss: {tl:.4f} | val_loss: {vl:.4f} | val_acc: {va:.4f}")

    # ── Model kaydet ─────────────────────────────────────────
    torch.save(model.state_dict(), "fracture_model.pth")
    print("\nModel kaydedildi: fracture_model.pth")

    # ── Eğitim grafiği ───────────────────────────────────────
    epochs_range = range(1, EPOCHS_FROZEN + EPOCHS_FT + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(epochs_range, history["train_loss"], label="Train Loss")
    ax1.plot(epochs_range, history["val_loss"],   label="Val Loss")
    ax1.axvline(x=EPOCHS_FROZEN + 0.5, color="gray", linestyle="--", label="Fine-tune başlangıcı")
    ax1.set_title("Loss")
    ax1.legend()
    ax2.plot(epochs_range, history["train_acc"], label="Train Acc")
    ax2.plot(epochs_range, history["val_acc"],   label="Val Acc")
    ax2.axvline(x=EPOCHS_FROZEN + 0.5, color="gray", linestyle="--")
    ax2.set_title("Accuracy")
    ax2.legend()
    plt.tight_layout()
    plt.savefig("training_curves.png", dpi=150)
    plt.show()
    print("Grafik kaydedildi: training_curves.png")

    # ── Test seti değerlendirme ──────────────────────────────
    print("\n── Test seti sonuçları ──")
    model.eval()
    all_preds, all_labels, all_probs = [], [], []
    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs = imgs.to(DEVICE)
            out  = model(imgs)
            probs = torch.softmax(out, dim=1)[:, 1]
            all_preds.extend(out.argmax(1).cpu().numpy())
            all_labels.extend(labels.numpy())
            all_probs.extend(probs.cpu().numpy())

    print(classification_report(all_labels, all_preds, target_names=train_ds.classes))
    print(f"ROC-AUC: {roc_auc_score(all_labels, all_probs):.4f}")

    # ── Confusion matrix ─────────────────────────────────────
    cm = confusion_matrix(all_labels, all_preds)
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(train_ds.classes)
    ax.set_yticklabels(train_ds.classes)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, cm[i, j], ha="center", va="center", color="black", fontsize=14)
    ax.set_xlabel("Tahmin")
    ax.set_ylabel("Gerçek")
    ax.set_title("Confusion Matrix")
    plt.colorbar(im)
    plt.tight_layout()
    plt.savefig("confusion_matrix.png", dpi=150)
    plt.show()
    print("Confusion matrix kaydedildi: confusion_matrix.png")