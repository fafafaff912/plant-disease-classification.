"""
Обучение модели классификации состояния растений.
EfficientNet-B0 с transfer learning на PlantVillage.

Запуск:
    python train.py --data_dir data/PlantVillage --epochs 20 --batch_size 32
"""

import argparse
import json
import os
import random
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from tqdm import tqdm


# ========================== ВОСПРОИЗВОДИМОСТЬ ==========================

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device():
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Apple MPS")
    else:
        device = torch.device("cpu")
        print("CPU")
    return device


# ========================== ДАТАСЕТ ==========================

class PlantDataset(Dataset):
    """Загрузка изображений из папок (каждая папка = класс)."""

    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert("RGB")
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label


def load_dataset(data_dir):
    """Сканирование директории, возврат путей, меток и имён классов."""
    classes = sorted([
        d for d in os.listdir(data_dir)
        if os.path.isdir(os.path.join(data_dir, d))
    ])
    class_to_idx = {c: i for i, c in enumerate(classes)}

    paths, labels = [], []
    valid_ext = {".jpg", ".jpeg", ".png", ".bmp"}

    for cls in classes:
        cls_dir = os.path.join(data_dir, cls)
        for fname in os.listdir(cls_dir):
            if os.path.splitext(fname)[1].lower() in valid_ext:
                paths.append(os.path.join(cls_dir, fname))
                labels.append(class_to_idx[cls])

    print(f"Найдено {len(paths)} изображений, {len(classes)} классов")
    return paths, labels, classes


def get_transforms(mode="train", size=224):
    """Аугментации для train, простая предобработка для val/test."""
    mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]

    if mode == "train":
        return transforms.Compose([
            transforms.RandomResizedCrop(size, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(p=0.3),
            transforms.RandomRotation(25),
            transforms.ColorJitter(0.2, 0.2, 0.2, 0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
    else:
        return transforms.Compose([
            transforms.Resize(int(size * 1.14)),
            transforms.CenterCrop(size),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])


def create_loaders(data_dir, batch_size=32, seed=42):
    """Создание train/val/test загрузчиков (80/10/10 стратифицированно)."""
    paths, labels, classes = load_dataset(data_dir)

    # train 80% / temp 20%
    train_p, temp_p, train_l, temp_l = train_test_split(
        paths, labels, test_size=0.2, stratify=labels, random_state=seed
    )
    # val 50% temp / test 50% temp → по 10% от общего
    val_p, test_p, val_l, test_l = train_test_split(
        temp_p, temp_l, test_size=0.5, stratify=temp_l, random_state=seed
    )

    print(f"Train: {len(train_p)}, Val: {len(val_p)}, Test: {len(test_p)}")

    train_ds = PlantDataset(train_p, train_l, get_transforms("train"))
    val_ds = PlantDataset(val_p, val_l, get_transforms("val"))
    test_ds = PlantDataset(test_p, test_l, get_transforms("val"))

    kw = dict(num_workers=4, pin_memory=True)
    train_loader = DataLoader(train_ds, batch_size, shuffle=True, drop_last=True, **kw)
    val_loader = DataLoader(val_ds, batch_size, shuffle=False, **kw)
    test_loader = DataLoader(test_ds, batch_size, shuffle=False, **kw)

    return train_loader, val_loader, test_loader, classes


# ========================== МОДЕЛЬ ==========================

def create_model(num_classes, model_name="efficientnet_b0"):
    """
    Создание модели с pretrained backbone и новым классификатором.
    Поддерживает: efficientnet_b0, resnet50.
    """
    if model_name == "efficientnet_b0":
        model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
        in_features = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes),
        )
    elif model_name == "resnet50":
        model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        in_features = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes),
        )
    else:
        raise ValueError(f"Неизвестная модель: {model_name}")

    total = sum(p.numel() for p in model.parameters())
    print(f"Модель: {model_name}, параметров: {total / 1e6:.1f}M")
    return model


def freeze_backbone(model, model_name="efficientnet_b0"):
    """Заморозить backbone (только classifier обучается)."""
    if model_name == "efficientnet_b0":
        for param in model.features.parameters():
            param.requires_grad = False
    elif model_name == "resnet50":
        for name, param in model.named_parameters():
            if "fc" not in name:
                param.requires_grad = False


def unfreeze_backbone(model):
    """Разморозить все параметры."""
    for param in model.parameters():
        param.requires_grad = True


# ========================== ОБУЧЕНИЕ ==========================

def train_one_epoch(model, loader, criterion, optimizer, device, scaler=None):
    """Одна эпоха обучения."""
    model.train()
    total_loss, correct, total = 0, 0, 0

    for images, labels in tqdm(loader, desc="  Train", leave=False):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()

        if scaler:
            with torch.amp.autocast("cuda"):
                outputs = model(images)
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        total_loss += loss.item() * images.size(0)
        correct += (outputs.argmax(1) == labels).sum().item()
        total += labels.size(0)

    return total_loss / total, 100.0 * correct / total


@torch.no_grad()
def validate(model, loader, criterion, device):
    """Валидация модели."""
    model.eval()
    total_loss, correct, total = 0, 0, 0

    for images, labels in tqdm(loader, desc="  Val", leave=False):
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)

        total_loss += loss.item() * images.size(0)
        correct += (outputs.argmax(1) == labels).sum().item()
        total += labels.size(0)

    return total_loss / total, 100.0 * correct / total


def plot_curves(history, save_path):
    """Сохранение графиков loss и accuracy."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    epochs = range(1, len(history["train_loss"]) + 1)

    ax1.plot(epochs, history["train_loss"], "b-o", label="Train", ms=4)
    ax1.plot(epochs, history["val_loss"], "r-o", label="Val", ms=4)
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Loss", fontweight="bold")
    ax1.legend()
    ax1.grid(alpha=0.3)

    ax2.plot(epochs, history["train_acc"], "b-o", label="Train", ms=4)
    ax2.plot(epochs, history["val_acc"], "r-o", label="Val", ms=4)
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy (%)")
    ax2.set_title("Accuracy", fontweight="bold")
    ax2.legend()
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Графики сохранены: {save_path}")


def train(args):
    """Полный цикл обучения."""
    set_seed(args.seed)
    device = get_device()
    os.makedirs("results", exist_ok=True)

    # Данные
    train_loader, val_loader, test_loader, classes = create_loaders(
        args.data_dir, args.batch_size, args.seed
    )
    num_classes = len(classes)

    # Модель
    model = create_model(num_classes, args.model).to(device)

    # Заморозка backbone на первые 3 эпохи
    freeze_epochs = 3
    freeze_backbone(model, args.model)
    print(f"Backbone заморожен на {freeze_epochs} эпох")

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    # Оптимизатор с разными lr для backbone и classifier
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-7)

    # Mixed precision
    scaler = torch.amp.GradScaler("cuda") if device.type == "cuda" else None

    # Обучение
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_val_acc = 0
    patience, patience_counter = 5, 0

    print(f"\n{'='*60}")
    print(f"Обучение: {args.epochs} эпох, batch={args.batch_size}, lr={args.lr}")
    print(f"{'='*60}")

    for epoch in range(args.epochs):
        # Разморозка после freeze_epochs
        if epoch == freeze_epochs:
            unfreeze_backbone(model)
            # Обновляем lr: backbone медленнее
            optimizer = optim.AdamW([
                {"params": [p for n, p in model.named_parameters()
                            if "classifier" not in n and "fc" not in n],
                 "lr": args.lr * 0.1},
                {"params": [p for n, p in model.named_parameters()
                            if "classifier" in n or "fc" in n],
                 "lr": args.lr},
            ], weight_decay=1e-4)
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=args.epochs - freeze_epochs, eta_min=1e-7
            )
            print("  → Backbone разморожен, fine-tuning")

        t0 = time.time()
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device, scaler
        )
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        scheduler.step()
        dt = time.time() - t0

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        lr_now = optimizer.param_groups[-1]["lr"]
        print(
            f"Epoch {epoch+1:2d}/{args.epochs} ({dt:.0f}s) | "
            f"Train: loss={train_loss:.4f} acc={train_acc:.1f}% | "
            f"Val: loss={val_loss:.4f} acc={val_acc:.1f}% | lr={lr_now:.2e}"
        )

        # Сохранение лучшей модели
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            torch.save({
                "model_state_dict": model.state_dict(),
                "class_names": classes,
                "num_classes": num_classes,
                "model_name": args.model,
                "val_acc": val_acc,
                "epoch": epoch,
            }, "results/best_model.pth")
            print(f"  ✓ Лучшая модель сохранена (acc={val_acc:.1f}%)")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"  Early stopping (patience={patience})")
                break

    print(f"\nЛучшая Val Accuracy: {best_val_acc:.2f}%")

    # Графики
    plot_curves(history, "results/training_curves.png")

    # Сохранение истории
    with open("results/training_history.json", "w") as f:
        json.dump(history, f, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Обучение классификатора растений")
    parser.add_argument("--data_dir", type=str, default="data/PlantVillage")
    parser.add_argument("--model", type=str, default="efficientnet_b0",
                        choices=["efficientnet_b0", "resnet50"])
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    train(args)
