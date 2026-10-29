# Clasificación de Imágenes CIFAR-10 con CNN

**Universidad Andina del Cusco — Inteligencia Artificial (SIS070)**
Evaluación de la Segunda Unidad

| Campo | Valor |
|---|---|
| Estudiante | Gonzales Ibañes Angel Erickson |
| Código de matrícula | 022100304G |
| Semilla individual | **4** (`torch.manual_seed(4)`, `np.random.seed(4)`) |

## Estructura

```
├── src/
│   ├── dataset.py      # P1: carga, transformaciones, partición estratificada, visualización
│   ├── model.py        # P3: arquitectura CNN + validación de shapes 4D
│   └── train.py        # P4: entrenamiento (CrossEntropyLoss + Adam) y métricas
├── notebooks/
│   └── experiment.ipynb
├── outputs/
│   ├── loss_accuracy.png
│   └── confusion_matrix.png
├── requirements.txt
└── README.md
```

## Ejecución

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows  (Linux/Mac: source .venv/bin/activate)
pip install -r requirements.txt
python src/dataset.py           # P1: descarga, partición y cuadrícula de muestras
python src/model.py             # P3: validación de shapes
python src/train.py --epochs 20 # P4: entrenamiento y gráficas
```

---

## Pregunta 1 — Preparación de datos

- **Semilla 4** fijada en `random`, `numpy`, `torch` y en el `Generator` del `DataLoader`, por lo que el shuffle, la partición y los pesos iniciales son reproducibles.
- **Transformaciones de entrenamiento:** `RandomCrop(32, padding=4)` → `RandomHorizontalFlip(0.5)` → `ToTensor()` → `Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2470, 0.2435, 0.2616))`.
- **Validación y prueba:** solo `ToTensor()` + `Normalize()` (sin aumento, para medir sobre imágenes reales).
- **Partición estratificada** con `train_test_split(stratify=targets, random_state=4)`: 50 000 → **40 000 entrenamiento / 10 000 validación** (4 000 y 1 000 por clase). Prueba independiente: 10 000.

## Pregunta 2 — Justificación: CNN frente a MLP

### ¿Por qué un MLP colapsa o es ineficiente con CIFAR-10?

1. **Explosión de parámetros.** Una imagen de 32×32×3 se aplana en un vector de **3 072** valores. Una sola capa oculta de 1 024 neuronas ya requiere 3 072 × 1 024 ≈ **3,1 millones de pesos**. Con imágenes más grandes (224×224×3 = 150 528 entradas) la cifra se vuelve inmanejable. Muchos parámetros con solo 40 000 ejemplos favorecen el **sobreajuste**.
2. **Pérdida de la estructura espacial.** Al aplanar, el MLP destruye la relación de vecindad entre píxeles: un píxel y su vecino de la derecha son tan "distantes" como dos píxeles de esquinas opuestas. El modelo debe re-aprender desde cero que los píxeles cercanos están correlacionados.
3. **Sin invariancia a la traslación.** Si un gato aparece a la izquierda o a la derecha, para un MLP son patrones completamente distintos que activan pesos diferentes; necesitaría ver el objeto en todas las posiciones posibles.

### ¿Cómo lo resuelven las CNN?

| Propiedad | Efecto |
|---|---|
| **Conexiones locales (campo receptivo)** | Cada neurona mira solo una ventana de 3×3 píxeles, capturando bordes, texturas y esquinas. Las capas profundas combinan estos patrones en formas cada vez más complejas (jerarquía de características). |
| **Weight Sharing (pesos compartidos)** | El mismo filtro se desliza por toda la imagen. Una Conv2d de 3→32 canales con kernel 3×3 tiene solo 3·3·3·32 + 32 = **896 parámetros**, frente a millones en una capa densa. |
| **Equivariancia / invariancia espacial** | Como el filtro es el mismo en todas las posiciones, detecta un patrón donde sea que aparezca; el `MaxPool2d` añade además tolerancia a pequeños desplazamientos. |
| **Reducción progresiva de dimensión** | El pooling reduce H×W a la mitad en cada bloque, por lo que la capa densa final recibe un vector compacto de características (2 048) en vez de píxeles crudos. |

Resultado: la CNN usa **~0,82 M parámetros** y supera ampliamente a un MLP de tamaño comparable (que en CIFAR-10 suele quedarse alrededor de 50–55 % de accuracy).

## Arquitectura de la CNN

| # | Capa | Configuración | Shape de salida |
|---|---|---|---|
| 0 | Entrada | — | (B, 3, 32, 32) |
| 1 | **Bloque 1** | Conv2d(3→32, k3, p1) · BN · ReLU · Conv2d(32→32) · BN · ReLU · **MaxPool2d(2)** | (B, 32, 16, 16) |
| 2 | **Bloque 2** | Conv2d(32→64, k3, p1) · BN · ReLU · Conv2d(64→64) · BN · ReLU · **MaxPool2d(2)** | (B, 64, 8, 8) |
| 3 | **Bloque 3** | Conv2d(64→128, k3, p1) · BN · ReLU · Conv2d(128→128) · BN · ReLU · **MaxPool2d(2)** | (B, 128, 4, 4) |
| 4 | Flatten | 128 × 4 × 4 | (B, 2048) |
| 5 | Dropout(0.5) + **Linear** + ReLU | 2048 → 256 | (B, 256) |
| 6 | Dropout(0.5) + **Linear** | 256 → 10 | (B, 10) logits |

```
(B,3,32,32) ─[Bloque1]→ (B,32,16,16) ─[Bloque2]→ (B,64,8,8) ─[Bloque3]→ (B,128,4,4)
            ─[Flatten]→ (B,2048) ─[Linear]→ (B,256) ─[Linear]→ (B,10)
```

Con `padding=1` y kernel 3×3, la Conv2d conserva H×W: `H_out = (H + 2·1 − 3)/1 + 1 = H`. Solo el MaxPool2d(2) divide entre 2: 32 → 16 → 8 → 4. El tamaño de entrada a la primera capa `Linear` se calcula dinámicamente en `_get_flat_features()` para evitar errores de *Shape Mismatch*.

## Pregunta 3 — Validación del forward pass

`python src/model.py` pasa un batch simulado `torch.randn(8, 3, 32, 32)`, imprime los shapes intermedios y verifica con `assert` que la salida sea `(8, 10)`.

## Pregunta 4 — Entrenamiento y análisis crítico

**Configuración:** `CrossEntropyLoss`, `Adam(lr=1e-3, weight_decay=1e-4)`, batch 256, **3 épocas**, semilla 4, CPU. Se guarda el modelo con mejor accuracy de validación y se evalúa sobre el test.

```
python src/train.py --epochs 3 --batch-size 256
```

![Curvas](outputs/loss_accuracy.png)
![Matriz de confusión](outputs/confusion_matrix.png)

### Resultados con semilla 4

| Época | Train loss | Train acc | Val loss | Val acc |
|---|---|---|---|---|
| 1 | 1.6714 | 37.35 % | 1.3417 | 50.69 % |
| 2 | 1.2903 | 53.14 % | 1.2597 | 55.25 % |
| 3 | 1.0887 | 61.35 % | 1.0253 | 64.27 % |

| Métrica | Valor |
|---|---|
| Mejor accuracy de validación | 64.27 % |
| Test loss | 1.0185 |
| **Accuracy en test** | **64.33 %** |
| Brecha train − val (acc) | −2.92 pp |

| Clase | Precision | Recall | F1 |
|---|---|---|---|
| avión | 0.7967 | 0.5800 | 0.6713 |
| automóvil | 0.8938 | 0.7740 | 0.8296 |
| pájaro | 0.5924 | 0.3720 | 0.4570 |
| gato | 0.5217 | 0.1800 | 0.2677 |
| ciervo | 0.4799 | 0.7050 | 0.5711 |
| perro | 0.4805 | 0.7020 | 0.5705 |
| rana | 0.7766 | 0.7090 | 0.7412 |
| caballo | 0.4949 | 0.8330 | 0.6209 |
| barco | 0.8730 | 0.7770 | 0.8222 |
| camión | 0.7876 | 0.8010 | 0.7942 |
| **macro avg** | 0.6697 | 0.6433 | 0.6346 |

### Análisis crítico

El modelo alcanzó **64.33 % de accuracy en test**, prácticamente igual a la de validación (64.27 %), lo que indica que generaliza de forma consistente a datos no vistos.

**Overfitting: no.** La accuracy de validación (64.27 %) supera a la de entrenamiento (61.35 %), con una brecha de −2.92 pp, y la loss de validación (1.0253) es menor que la de entrenamiento (1.0887). Esto ocurre porque en entrenamiento actúan el data augmentation (RandomCrop, RandomHorizontalFlip) y el Dropout (0.5), mientras que la validación usa imágenes limpias con el modelo en modo `eval()`.

**Underfitting: sí, leve.** Con solo 3 épocas las curvas no convergen: la accuracy de validación subió de 50.69 % a 55.25 % y luego a 64.27 %, y la loss siguió bajando. El modelo aún estaba aprendiendo al terminar.

**Por clase:** los vehículos son los mejor clasificados (automóvil F1 = 0.83, barco 0.82, camión 0.79). La clase más difícil es **gato** (F1 = 0.27, recall 18 %), seguida de pájaro (0.46). Perro, ciervo y caballo tienen recall alto (0.70–0.83) pero precisión baja (≈0.48), es decir, el modelo les asigna muchas imágenes de otros animales. La confusión se concentra entre animales de apariencia similar a 32×32 píxeles.

**Mejoras:** entrenar 20–30 épocas, usar `ReduceLROnPlateau` o `CosineAnnealingLR`, ampliar el data augmentation (`ColorJitter`, `RandomRotation`) y probar arquitecturas más profundas (tipo ResNet).

## Conclusiones

1. La CNN de 3 bloques convolucionales (815 018 parámetros) procesó correctamente los tensores 4D y alcanzó 64.33 % en test con solo 3 épocas.
2. La semilla 4 y la partición estratificada garantizaron reproducibilidad y clases balanceadas (4 000 / 1 000).
3. El data augmentation y el Dropout evitaron el sobreajuste; el modelo presenta underfitting leve por el número limitado de épocas.
4. El modelo distingue mejor vehículos que animales; gato es la clase más difícil.
