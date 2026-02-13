import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms, models
from torchvision.datasets import ImageFolder

# Константы по умолчанию
DEFAULT_DATA_DIR = './data/plantvillage'  # путь к данным
DEFAULT_BATCH_SIZE = 32
DEFAULT_EPOCHS = 10
DEFAULT_LR = 0.001
DEFAULT_IMG_SIZE = 224
DEFAULT_VAL_SPLIT = 0.2

def get_transform(img_size, is_train=True):
    """
    Возвращает набор трансформаций для изображений.
    Для тренировочных данных применяется аугментация.
    """
    if is_train:
        transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    return transform

def load_data(data_dir, batch_size, img_size, val_split=DEFAULT_VAL_SPLIT):
    """
    Загружает данные из структуры папок (ImageFolder).
    Возвращает DataLoader для тренировки и валидации, а также список имён классов.
    """
    # Полный датасет с временной трансформацией (нужна только для получения списка файлов)
    full_dataset = ImageFolder(root=data_dir, transform=get_transform(img_size, is_train=False))
    class_names = full_dataset.classes
    num_samples = len(full_dataset)
    indices = list(range(num_samples))
    np.random.shuffle(indices)

    # Разделяем индексы на train/val
    val_size = int(val_split * num_samples)
    train_indices = indices[val_size:]
    val_indices = indices[:val_size]

    # Создаём датасеты с нужными трансформациями
    train_dataset = ImageFolder(root=data_dir, transform=get_transform(img_size, is_train=True))
    val_dataset = ImageFolder(root=data_dir, transform=get_transform(img_size, is_train=False))

    # Создаём сэмплеры
    train_sampler = torch.utils.data.SubsetRandomSampler(train_indices)
    val_sampler = torch.utils.data.SubsetRandomSampler(val_indices)

    # DataLoader'ы
    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, sampler=val_sampler, num_workers=4)

    return train_loader, val_loader, class_names

def create_model(num_classes, pretrained=True):
    """
    Создаёт модель ResNet18 с заменой последнего fully connected слоя.
    """
    model = models.resnet18(pretrained=pretrained)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model

def train_one_epoch(model, train_loader, criterion, optimizer, device):
    """
    Обучает модель одну эпоху.
    Возвращает средние потери и точность на тренировочных данных.
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc

def validate(model, val_loader, criterion, device):
    """
    Валидация модели.
    Возвращает средние потери, точность, а также списки предсказаний и истинных меток.
    """
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc, all_preds, all_labels

def plot_confusion_matrix(cm, class_names, save_path='confusion_matrix.png'):
    """
    Строит и сохраняет матрицу ошибок.
    """
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()

def main(args):
    # Определение устройства
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Загрузка данных
    train_loader, val_loader, class_names = load_data(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        img_size=args.img_size,
        val_split=args.val_split
    )
    print(f"Classes: {class_names}")
    print(f"Train samples: {len(train_loader.sampler)}, Val samples: {len(val_loader.sampler)}")

    # Создание модели
    model = create_model(num_classes=len(class_names), pretrained=True)
    model = model.to(device)

    # Функция потерь и оптимизатор
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

    best_val_acc = 0.0
    best_model_state = None

    # Цикл обучения
    for epoch in range(args.epochs):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, val_preds, val_labels = validate(model, val_loader, criterion, device)
        scheduler.step()

        print(f"\nEpoch {epoch+1}/{args.epochs}")
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

        # Сохраняем лучшую модель
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = model.state_dict().copy()
            torch.save(best_model_state, 'best_model.pth')
            print(f"Saved best model with val_acc={best_val_acc:.4f}")

    # Загрузка лучшей модели
    model.load_state_dict(best_model_state)

    # Финальная оценка на валидации
    val_loss, val_acc, val_preds, val_labels = validate(model, val_loader, criterion, device)

    # Расчёт метрик
    accuracy = accuracy_score(val_labels, val_preds)
    f1 = f1_score(val_labels, val_preds, average='weighted')
    cm = confusion_matrix(val_labels, val_preds)
    report = classification_report(val_labels, val_preds, target_names=class_names)

    print("\n===== Final Report =====")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"F1-score (weighted): {f1:.4f}")
    print("Confusion Matrix:")
    print(cm)
    print("\nClassification Report:")
    print(report)

    # Сохранение матрицы ошибок
    plot_confusion_matrix(cm, class_names, save_path='../reports/confusion_matrix.png')
    print("Confusion matrix saved as '../reports/confusion_matrix.png'")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Plant Disease Classification using CNN')
    parser.add_argument('--data_dir', type=str, default=DEFAULT_DATA_DIR,
                        help='Path to dataset folder')
    parser.add_argument('--batch_size', type=int, default=DEFAULT_BATCH_SIZE,
                        help='Batch size')
    parser.add_argument('--epochs', type=int, default=DEFAULT_EPOCHS,
                        help='Number of epochs')
    parser.add_argument('--lr', type=float, default=DEFAULT_LR,
                        help='Learning rate')
    parser.add_argument('--img_size', type=int, default=DEFAULT_IMG_SIZE,
                        help='Input image size')
    parser.add_argument('--val_split', type=float, default=DEFAULT_VAL_SPLIT,
                        help='Validation split ratio')
    args = parser.parse_args()

    main(args)
