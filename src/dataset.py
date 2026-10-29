"""
Pregunta 1: Carga, visualización, transformaciones y partición estratificada de CIFAR-10.

Estudiante : Gonzales Ibañes Angel Erickson
Código     : 022100304G  ->  dos últimos dígitos: 04  ->  SEMILLA = 4
"""
import os
import random

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

# ----------------------------------------------------------------------------
# 1. SEMILLA INDIVIDUAL
# ----------------------------------------------------------------------------
SEED = 4  # dos últimos dígitos del código 022100304G

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")

CLASSES = ("avión", "automóvil", "pájaro", "gato", "ciervo",
           "perro", "rana", "caballo", "barco", "camión")

# Media y desviación estándar por canal RGB de CIFAR-10 (conjunto de entrenamiento)
CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR_STD = (0.2470, 0.2435, 0.2616)


def set_seed(seed: int = SEED) -> None:
    """Fija todas las fuentes de aleatoriedad para reproducibilidad."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _seed_worker(worker_id: int) -> None:
    """Hace reproducibles los workers del DataLoader."""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


# ----------------------------------------------------------------------------
# 3. PIPELINE DE TRANSFORMACIONES
# ----------------------------------------------------------------------------
def get_transforms():
    """Devuelve (transform_train, transform_eval).

    - Entrenamiento: Data Augmentation (RandomCrop + RandomHorizontalFlip)
      + ToTensor + Normalize.
    - Validación / prueba: solo ToTensor + Normalize (sin aumento, para medir
      el rendimiento sobre imágenes "reales").
    """
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),       # recorte aleatorio con relleno
        transforms.RandomHorizontalFlip(p=0.5),      # espejo horizontal
        transforms.ToTensor(),                       # [0,255] HWC -> [0,1] CHW
        transforms.Normalize(CIFAR_MEAN, CIFAR_STD)  # normalización por canal RGB
    ])
    transform_eval = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR_MEAN, CIFAR_STD)
    ])
    return transform_train, transform_eval


# ----------------------------------------------------------------------------
# 1 + 4. CARGA Y PARTICIÓN ESTRATIFICADA
# ----------------------------------------------------------------------------
def load_datasets(val_size: float = 0.2, seed: int = SEED, verbose: bool = True):
    """Descarga CIFAR-10 y divide el train oficial en train/val estratificado.

    Se crean dos instancias del train oficial (una con augmentation y otra sin)
    para que el subconjunto de validación NO reciba aumento de datos.
    """
    set_seed(seed)
    transform_train, transform_eval = get_transforms()

    full_train_aug = datasets.CIFAR10(DATA_DIR, train=True, download=True,
                                      transform=transform_train)
    full_train_eval = datasets.CIFAR10(DATA_DIR, train=True, download=True,
                                       transform=transform_eval)
    test_set = datasets.CIFAR10(DATA_DIR, train=False, download=True,
                                transform=transform_eval)

    targets = np.array(full_train_aug.targets)
    indices = np.arange(len(targets))

    # Partición estratificada: mantiene el 10 % de cada clase en ambos subconjuntos
    train_idx, val_idx = train_test_split(
        indices, test_size=val_size, stratify=targets,
        shuffle=True, random_state=seed)

    train_set = Subset(full_train_aug, train_idx)
    val_set = Subset(full_train_eval, val_idx)

    if verbose:
        print("=" * 60)
        print(f"PARTICIÓN ESTRATIFICADA (semilla = {seed})")
        print("=" * 60)
        print(f"Train oficial completo : {len(full_train_aug)} imágenes")
        print(f"  -> Entrenamiento     : {len(train_set)} imágenes")
        print(f"  -> Validación        : {len(val_set)} imágenes")
        print(f"Prueba (test)          : {len(test_set)} imágenes")
        x0, y0 = train_set[0]
        print(f"Shape de una imagen    : {tuple(x0.shape)}  (C, H, W)")
        print(f"Primeros 10 índices de train: {train_idx[:10].tolist()}")
        tr_counts = np.bincount(targets[train_idx], minlength=10)
        va_counts = np.bincount(targets[val_idx], minlength=10)
        print("\nDistribución por clase (train / val):")
        for i, c in enumerate(CLASSES):
            print(f"  {c:<10} {tr_counts[i]:>5} / {va_counts[i]:>4}")
        print("=" * 60)

    return train_set, val_set, test_set


def get_dataloaders(batch_size: int = 128, val_size: float = 0.2,
                    seed: int = SEED, num_workers: int = 0, verbose: bool = True):
    """Devuelve (train_loader, val_loader, test_loader)."""
    train_set, val_set, test_set = load_datasets(val_size, seed, verbose)

    g = torch.Generator()
    g.manual_seed(seed)  # controla el shuffle de cada época

    common = dict(batch_size=batch_size, num_workers=num_workers,
                  worker_init_fn=_seed_worker,
                  pin_memory=torch.cuda.is_available())
    train_loader = DataLoader(train_set, shuffle=True, generator=g, **common)
    val_loader = DataLoader(val_set, shuffle=False, **common)
    test_loader = DataLoader(test_set, shuffle=False, **common)
    return train_loader, val_loader, test_loader


# ----------------------------------------------------------------------------
# 2. VISUALIZACIÓN
# ----------------------------------------------------------------------------
def denormalize(img: torch.Tensor) -> np.ndarray:
    """Revierte la normalización para poder mostrar la imagen."""
    mean = torch.tensor(CIFAR_MEAN).view(3, 1, 1)
    std = torch.tensor(CIFAR_STD).view(3, 1, 1)
    img = img * std + mean
    return img.clamp(0, 1).permute(1, 2, 0).numpy()  # CHW -> HWC


def show_samples(dataset, n: int = 16, seed: int = SEED, save_path: str = None,
                 title: str = "Muestras de CIFAR-10"):
    """Grafica una cuadrícula de n imágenes con su etiqueta real."""
    rng = np.random.default_rng(seed)
    idxs = rng.choice(len(dataset), size=n, replace=False)
    cols = 4
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.6, rows * 2.8))
    for ax, i in zip(axes.flat, idxs):
        img, label = dataset[int(i)]
        ax.imshow(denormalize(img))
        ax.set_title(f"{CLASSES[label]} ({label})", fontsize=10)
        ax.axis("off")
    for ax in list(axes.flat)[n:]:
        ax.axis("off")
    fig.suptitle(f"{title} — semilla {seed}", fontsize=13)
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=120, bbox_inches="tight")
        print(f"Cuadrícula guardada en: {save_path}")
    plt.show()


if __name__ == "__main__":
    train_set, val_set, test_set = load_datasets()
    show_samples(train_set, n=16,
                 save_path=os.path.join(OUTPUT_DIR, "muestras_dataset.png"),
                 title="Muestras de entrenamiento (con augmentation)")
