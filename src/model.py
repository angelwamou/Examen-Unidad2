"""
Pregunta 3: Arquitectura CNN en PyTorch y validación de shapes 4D.

Flujo de tensores (B = batch size):
    Entrada            (B,   3, 32, 32)
    Bloque 1 -> pool   (B,  32, 16, 16)
    Bloque 2 -> pool   (B,  64,  8,  8)
    Bloque 3 -> pool   (B, 128,  4,  4)
    Flatten            (B, 2048)
    Linear + ReLU      (B, 256)
    Linear (salida)    (B, 10)
"""
import torch
import torch.nn as nn

SEED = 4


class ConvBlock(nn.Module):
    """Conv2d -> BatchNorm2d -> ReLU -> Conv2d -> BatchNorm2d -> ReLU -> MaxPool2d."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # reduce H y W a la mitad
        )

    def forward(self, x):
        return self.block(x)


class CifarCNN(nn.Module):
    def __init__(self, num_classes: int = 10, dropout: float = 0.5):
        super().__init__()
        self.block1 = ConvBlock(3, 32)    # 32x32 -> 16x16
        self.block2 = ConvBlock(32, 64)   # 16x16 -> 8x8
        self.block3 = ConvBlock(64, 128)  # 8x8   -> 4x4

        self.flatten = nn.Flatten()       # (B,128,4,4) -> (B,2048)
        # Cálculo dinámico del tamaño de entrada a la capa densa
        flat_features = self._get_flat_features()

        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(flat_features, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),  # logits (CrossEntropyLoss aplica softmax)
        )

    def _get_flat_features(self) -> int:
        """Pasa un tensor ficticio para calcular cuántas neuronas salen del Flatten."""
        with torch.no_grad():
            x = torch.zeros(1, 3, 32, 32)
            x = self.block3(self.block2(self.block1(x)))
            return x.numel()

    def forward(self, x, verbose: bool = False):
        if verbose:
            print(f"{'Entrada':<22}{tuple(x.shape)}")
        x = self.block1(x)
        if verbose:
            print(f"{'Bloque conv 1':<22}{tuple(x.shape)}")
        x = self.block2(x)
        if verbose:
            print(f"{'Bloque conv 2':<22}{tuple(x.shape)}")
        x = self.block3(x)
        if verbose:
            print(f"{'Bloque conv 3':<22}{tuple(x.shape)}")
        x = self.flatten(x)
        if verbose:
            print(f"{'Flatten':<22}{tuple(x.shape)}")
        x = self.classifier(x)
        if verbose:
            print(f"{'Salida (logits)':<22}{tuple(x.shape)}")
        return x


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def validate_shapes(batch_size: int = 8) -> None:
    """Prueba obligatoria: pasa un batch simulado 4D y verifica los shapes."""
    torch.manual_seed(SEED)
    model = CifarCNN()
    model.eval()
    dummy = torch.randn(batch_size, 3, 32, 32)  # (Batch, Channels, Height, Width)

    print("=" * 60)
    print("VALIDACIÓN DEL FORWARD PASS CON TENSORES 4D")
    print("=" * 60)
    with torch.no_grad():
        out = model(dummy, verbose=True)

    assert dummy.dim() == 4, "La entrada debe ser un tensor 4D (B, C, H, W)"
    assert out.shape == (batch_size, 10), f"Shape incorrecto: {out.shape}"
    print("-" * 60)
    print(f"Parámetros entrenables: {count_parameters(model):,}")
    print("✔ Aserción superada: la arquitectura procesa tensores 4D sin Shape Mismatch")
    print("=" * 60)


if __name__ == "__main__":
    validate_shapes()
