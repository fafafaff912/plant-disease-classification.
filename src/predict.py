"""
Предсказание класса для одного изображения.

Запуск:
    python predict.py --image path/to/leaf.jpg --model results/best_model.pth
"""

import argparse
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


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


def predict(image_path, model_path, top_k=5):
    """Предсказание для одного изображения."""
    device = get_device()

    # Загрузка чекпоинта
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    class_names = checkpoint["class_names"]
    model_name = checkpoint.get("model_name", "efficientnet_b0")
    num_classes = checkpoint["num_classes"]

    # Модель
    model = create_model(num_classes, model_name).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # Предобработка
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    image = Image.open(image_path).convert("RGB")
    input_tensor = transform(image).unsqueeze(0).to(device)

    # Инференс
    with torch.no_grad():
        outputs = model(input_tensor)
        probs = torch.softmax(outputs, dim=1)

    top_probs, top_idx = torch.topk(probs, min(top_k, num_classes))
    top_probs = top_probs.squeeze().cpu().numpy()
    top_idx = top_idx.squeeze().cpu().numpy()

    # Вывод
    print(f"\n{'='*50}")
    print(f"Файл: {image_path}")
    print(f"Предсказание: {class_names[top_idx[0]]}")
    print(f"Уверенность: {top_probs[0]:.2%}")
    print(f"\nTop-{top_k}:")
    for i, (idx, prob) in enumerate(zip(top_idx, top_probs), 1):
        print(f"  {i}. {class_names[idx]}: {prob:.2%}")
    print(f"{'='*50}")

    # Визуализация
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    ax1.imshow(image)
    ax1.set_title(
        f"Класс: {class_names[top_idx[0]]}\nУверенность: {top_probs[0]:.1%}",
        fontsize=11, fontweight="bold",
    )
    ax1.axis("off")

    names = [class_names[i].replace("___", "\n").replace("_", " ") for i in top_idx]
    colors = ["#2ecc71"] + ["#3498db"] * (len(names) - 1)
    bars = ax2.barh(range(len(names)), top_probs, color=colors)
    ax2.set_yticks(range(len(names)))
    ax2.set_yticklabels(names, fontsize=9)
    ax2.set_xlabel("Вероятность")
    ax2.set_title("Top предсказания", fontweight="bold")
    ax2.set_xlim(0, 1)
    ax2.invert_yaxis()

    for bar, p in zip(bars, top_probs):
        ax2.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
                 f"{p:.1%}", va="center", fontsize=9)

    plt.tight_layout()
    save_path = "results/prediction.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Визуализация: {save_path}")
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Предсказание для изображения")
    parser.add_argument("--image", type=str, required=True, help="Путь к изображению")
    parser.add_argument("--model", type=str, default="results/best_model.pth")
    parser.add_argument("--top_k", type=int, default=5)
    args = parser.parse_args()

    if not os.path.exists(args.image):
        print(f"Файл не найден: {args.image}")
        sys.exit(1)
    if not os.path.exists(args.model):
        print(f"Модель не найдена: {args.model}")
        sys.exit(1)

    predict(args.image, args.model, args.top_k)
