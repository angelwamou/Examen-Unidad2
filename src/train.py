"""
Pregunta 4: Entrenamiento (CrossEntropyLoss + Adam), métricas y artefactos gráficos.

Ejecutar desde la raíz del repositorio:
    python src/train.py
    python src/train.py --epochs 30 --batch-size 128 --lr 0.001
"""
import argparse
import json
import os
import sys
import time

import matplotlib
matplotlib.use("Agg")  # guarda las figuras sin necesitar ventana
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from dataset import CLASSES, OUTPUT_DIR, SEED, get_dataloaders, set_seed  # noqa: E402
from model import CifarCNN, count_parameters  # noqa: E402


def run_epoch(model, loader, criterion, device, optimizer=None):
    """Una época de entrenamiento (si hay optimizer) o de evaluación."""
    training = optimizer is not None
    model.train() if training else model.eval()
    total_loss, correct, total = 0.0, 0, 0

    with torch.set_grad_enabled(training):
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)                 # (B, 10)
            loss = criterion(outputs, labels)

            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * images.size(0)
            correct += (outputs.argmax(1) == labels).sum().item()
            total += labels.size(0)

    return total_loss / total, correct / total


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    y_true, y_pred = [], []
    for images, labels in loader:
        outputs = model(images.to(device))
        y_pred.extend(outputs.argmax(1).cpu().numpy())
        y_true.extend(labels.numpy())
    return np.array(y_true), np.array(y_pred)


def plot_curves(history, path):
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.8))

    ax1.plot(epochs, history["train_loss"], "o-", label="Entrenamiento")
    ax1.plot(epochs, history["val_loss"], "s-", label="Validación")
    ax1.set_title("Curva de pérdida (CrossEntropyLoss)")
    ax1.set_xlabel("Época"); ax1.set_ylabel("Loss")
    ax1.grid(alpha=0.3); ax1.legend()

    ax2.plot(epochs, [a * 100 for a in history["train_acc"]], "o-", label="Entrenamiento")
    ax2.plot(epochs, [a * 100 for a in history["val_acc"]], "s-", label="Validación")
    ax2.set_title("Curva de exactitud (Accuracy)")
    ax2.set_xlabel("Época"); ax2.set_ylabel("Accuracy (%)")
    ax2.grid(alpha=0.3); ax2.legend()

    fig.suptitle(f"CNN CIFAR-10 — semilla {SEED}", fontsize=13)
    plt.tight_layout()
    plt.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"Curvas guardadas en: {path}")


def plot_confusion(y_true, y_pred, path):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(9, 7.5))
    im = ax.imshow(cm, cmap="Blues")
    fig.colorbar(im, ax=ax)
    ax.set_xticks(range(10)); ax.set_yticks(range(10))
    ax.set_xticklabels(CLASSES, rotation=45, ha="right")
    ax.set_yticklabels(CLASSES)
    ax.set_xlabel("Clase predicha"); ax.set_ylabel("Clase real")
    acc = (y_true == y_pred).mean() * 100
    ax.set_title(f"Matriz de confusión (test) — accuracy {acc:.2f}% — semilla {SEED}")
    thresh = cm.max() / 2
    for i in range(10):
        for j in range(10):
            ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=8,
                    color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"Matriz de confusión guardada en: {path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--workers", type=int, default=0)
    args = parser.parse_args()

    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"Dispositivo: {device} | Semilla: {SEED}")

    train_loader, val_loader, test_loader = get_dataloaders(
        batch_size=args.batch_size, num_workers=args.workers)

    # ---- 1. Configuración del entrenamiento ----
    model = CifarCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    print(f"Parámetros entrenables: {count_parameters(model):,}")

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_val_acc = 0.0
    best_path = os.path.join(OUTPUT_DIR, "best_model.pth")
    EPOCHS = args.epochs

    # ---- 2. Bucle de entrenamiento por épocas ----
    for epoch in range(EPOCHS):
        t0 = time.time()
        tr_loss, tr_acc = run_epoch(model, train_loader, criterion, device, optimizer)
        va_loss, va_acc = run_epoch(model, val_loader, criterion, device)

        history["train_loss"].append(tr_loss); history["train_acc"].append(tr_acc)
        history["val_loss"].append(va_loss); history["val_acc"].append(va_acc)

        if va_acc > best_val_acc:
            best_val_acc = va_acc
            torch.save(model.state_dict(), best_path)

        print(f"Época {epoch + 1:02d}/{EPOCHS} | "
              f"Train loss {tr_loss:.4f} acc {tr_acc * 100:6.2f}% | "
              f"Val loss {va_loss:.4f} acc {va_acc * 100:6.2f}% | "
              f"{time.time() - t0:.1f}s")

    # ---- Evaluación final sobre el conjunto de prueba independiente ----
    model.load_state_dict(torch.load(best_path, map_location=device))
    te_loss, te_acc = run_epoch(model, test_loader, criterion, device)
    y_true, y_pred = predict(model, test_loader, device)

    print("\n" + "=" * 60)
    print(f"Mejor accuracy de validación : {best_val_acc * 100:.2f}%")
    print(f"Test loss                    : {te_loss:.4f}")
    print(f"Test accuracy                : {te_acc * 100:.2f}%")
    print(f"Brecha final train-val (acc) : "
          f"{(history['train_acc'][-1] - history['val_acc'][-1]) * 100:+.2f} pp")
    print("=" * 60)
    report = classification_report(y_true, y_pred, target_names=CLASSES, digits=4)
    print(report)

    # ---- 3. Artefactos gráficos ----
    plot_curves(history, os.path.join(OUTPUT_DIR, "loss_accuracy.png"))
    plot_confusion(y_true, y_pred, os.path.join(OUTPUT_DIR, "confusion_matrix.png"))

    with open(os.path.join(OUTPUT_DIR, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump({"seed": SEED, "epochs": EPOCHS, "history": history,
                   "best_val_acc": best_val_acc, "test_loss": te_loss,
                   "test_acc": te_acc}, f, indent=2)
    with open(os.path.join(OUTPUT_DIR, "classification_report.txt"), "w",
              encoding="utf-8") as f:
        f.write(report)


if __name__ == "__main__":
    main()
