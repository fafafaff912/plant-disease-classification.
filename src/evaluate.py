"""
Оценка обученной модели: accuracy, F1, confusion matrix, classification report.

Запуск:
    python evaluate.py --data_dir data/PlantVillage --model_path results/best_model.pth
"""

import argparse
import json
import os
import random

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from tqdm import tqdm


# =================== Утилиты (повтор из train.py для автономности) ===================

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class PlantDataset(Dataset):
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
    return paths, labels, classes


def get_test_loader(data_dir, batch_size=32, seed=42):
    """Получить только test loader (те же сплиты что при обучении)."""
    paths, labels, classes = load_dataset(data_dir)
    _, temp_p, _, temp_l = train_test_split(
        paths, labels, test_size=0.2, stratify=labels, random_state=seed
    )
    _, test_p, _, test_l = train_test_split(
        temp_p, temp_l, test_size=0.5, stratify=temp_l, random_state=seed
    )

    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    test_ds = PlantDataset(test_p, test_l, transform)
    loader = DataLoader(test_ds, batch_size, shuffle=False, num_workers=4, pin_memory=True)
    print(f"Test: {len(test_ds)} изображений, {len(classes)} классов")
    return loader, classes


def create_model(num_classes, model_name="efficientnet_b0"):
    if model_name == "efficientnet_b0":
        model = models.efficientnet_b0(weights=None)
        in_features = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes),
        )
    elif model_name == "resnet50":
        model = models.resnet50(weights=None)
        in_features = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes),
        )
    return model


# =================== ОЦЕНКА ===================

@torch.no_grad()
def get_predictions(model, loader, device):
    """Получить все предсказания и метки."""
    model.eval()
    all_labels, all_preds = [], []

    for images, labels in tqdm(loader, desc="Оценка"):
        images = images.to(device)
        outputs = model(images)
        preds = outputs.argmax(1).cpu().numpy()
        all_labels.extend(labels.numpy())
        all_preds.extend(preds)

    return np.array(all_labels), np.array(all_preds)


def plot_confusion_matrix(labels, preds, class_names, save_path):
    """Построение и сохранение confusion matrix."""
    cm = confusion_matrix(labels, preds)
    # Нормализация
    cm_norm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

    fig_size = max(10, len(class_names) * 0.4)
    fig, ax = plt.subplots(figsize=(fig_size, fig_size))

    short_names = [n.replace("___", "\n")[:25] for n in class_names]

    sns.heatmap(
        cm_norm,
        annot=len(class_names) <= 20,
        fmt=".2f",
        cmap="Blues",
        xticklabels=short_names,
        yticklabels=short_names,
        ax=ax,
        square=True,
    )
    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("True", fontsize=12)
    ax.set_title("Normalized Confusion Matrix", fontsize=14, fontweight="bold")
    plt.xticks(rotation=45, ha="right", fontsize=6)
    plt.yticks(rotation=0, fontsize=6)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Confusion matrix: {save_path}")


def evaluate(args):
    """Полная оценка модели."""
    set_seed(42)
    device = get_device()
    os.makedirs("results", exist_ok=True)

    # Загрузка чекпоинта
    checkpoint = torch.load(args.model_path, map_location=device, weights_only=False)
    class_names = checkpoint["class_names"]
    model_name = checkpoint.get("model_name", "efficientnet_b0")
    num_classes = checkpoint["num_classes"]

    print(f"Модель: {model_name}, классов: {num_classes}")

    # Тестовые данные (с теми же сплитами)
    test_loader, _ = get_test_loader(args.data_dir, args.batch_size)

    # Модель
    model = create_model(num_classes, model_name).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    print("Веса загружены")

    # Предсказания
    labels, preds = get_predictions(model, test_loader, device)

    # Метрики
    metrics = {
        "accuracy": float(accuracy_score(labels, preds)),
        "f1_macro": float(f1_score(labels, preds, average="macro")),
        "f1_weighted": float(f1_score(labels, preds, average="weighted")),
        "precision_macro": float(precision_score(labels, preds, average="macro")),
        "recall_macro": float(recall_score(labels, preds, average="macro")),
        "num_test_samples": len(labels),
        "num_classes": num_classes,
    }

    print(f"\n{'='*50}")
    print("РЕЗУЛЬТАТЫ НА ТЕСТОВОМ НАБОРЕ")
    print(f"{'='*50}")
    print(f"  Accuracy:          {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
    print(f"  F1 (macro):        {metrics['f1_macro']:.4f}")
    print(f"  F1 (weighted):     {metrics['f1_weighted']:.4f}")
    print(f"  Precision (macro): {metrics['precision_macro']:.4f}")
    print(f"  Recall (macro):    {metrics['recall_macro']:.4f}")

    # Classification report
    report = classification_report(labels, preds, target_names=class_names, digits=4)
    print(f"\n{report}")

    with open("results/classification_report.txt", "w") as f:
        f.write("CLASSIFICATION REPORT\n")
        f.write("=" * 80 + "\n\n")
        f.write(report)

    # Confusion matrix
    plot_confusion_matrix(labels, preds, class_names, "results/confusion_matrix.png")

    # Per-class accuracy: worst/best
    cm = confusion_matrix(labels, preds)
    per_class_acc = cm.diagonal() / cm.sum(axis=1)
    sorted_idx = np.argsort(per_class_acc)

    print("\n5 худших классов:")
    for i in sorted_idx[:5]:
        print(f"  {class_names[i]}: {per_class_acc[i]:.4f}")
    print("\n5 лучших классов:")
    for i in sorted_idx[-5:]:
        print(f"  {class_names[i]}: {per_class_acc[i]:.4f}")

    # Сохранение метрик
    metrics["per_class_accuracy"] = {
        class_names[i]: float(per_class_acc[i]) for i in range(num_classes)
    }

    with open("results/metrics.json", "w") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    print(f"\nМетрики сохранены: results/metrics.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Оценка модели")
    parser.add_argument("--data_dir", type=str, default="data/PlantVillage")
    parser.add_argument("--model_path", type=str, default="results/best_model.pth")
    parser.add_argument("--batch_size", type=int, default=32)
    args = parser.parse_args()
    evaluate(args)
