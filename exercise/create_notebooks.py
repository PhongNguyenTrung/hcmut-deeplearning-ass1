"""
create_notebooks.py — Tạo tất cả notebooks cho bài tập exercise

Chạy: python create_notebooks.py
→ Tạo/ghi đè 5 file .ipynb trong thư mục notebooks/
"""

import json
import os

# ─── Helpers ──────────────────────────────────────────────────────────────────

def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text.strip()}

def code(text, outputs=None):
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": outputs or [],
        "source": text.strip(),
    }

def nb(*cells):
    return {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {"display_name": "deeplearning-ass1", "language": "python", "name": "deeplearning-ass1"},
            "language_info": {"name": "python", "version": "3.10.0"},
        },
        "cells": list(cells),
    }


# ─── Notebook 1: Part 1 — Xây dựng 4 mô hình ─────────────────────────────────

NB1 = nb(
    md("""# Phần 1 — Xây dựng các mô hình phân loại

**Bài tập CO5085 — Deep Learning & Computer Vision | HCMUT 2025-2026**

Trong notebook này, ta xây dựng lần lượt 4 mô hình phân loại ảnh trên **CIFAR-100**:
1. **Softmax Regression** — mô hình tuyến tính đơn giản nhất
2. **MLP** — mạng fully connected nhiều lớp
3. **SimpleCNN** — mạng tích chập VGG-style
4. **SimpleViT** — Vision Transformer dùng PyTorch có sẵn

Dataset: CIFAR-100 (100 lớp, ảnh màu 32×32, 50,000 train / 10,000 test)
"""),

    code("""import sys
import os
sys.path.insert(0, os.path.abspath('..'))

import torch
import torchvision
import matplotlib.pyplot as plt
import numpy as np
from src.data import get_cifar100_loaders, get_device
from src.models_part1 import SoftmaxRegression, MLP, SimpleCNN, SimpleViT
from src.utils import get_param_count

DEVICE = get_device()
print(f"Thiết bị: {DEVICE}")
print(f"PyTorch version: {torch.__version__}")
"""),

    md("""## 1. Tải và khám phá CIFAR-100

**Tại sao dùng CIFAR-100 thay vì CIFAR-10?**
- CIFAR-100: 100 lớp → bài toán khó hơn, phân biệt rõ hơn sức mạnh các mô hình
- CIFAR-10: chỉ 10 lớp → quá dễ, ngay cả Softmax Regression cũng đạt >80%

**Chuẩn hoá ảnh:** Dùng mean và std tính từ chính tập CIFAR-100,
*không phải* từ ImageNet (thường được dùng nhầm).
"""),

    code("""train_loader, val_loader, test_loader, class_names = get_cifar100_loaders(batch_size=128)

print(f"\\nSố lớp: {len(class_names)}")
print(f"Ví dụ một số lớp: {class_names[:10]}")

# Xem 1 batch ảnh mẫu
images, labels = next(iter(train_loader))
print(f"\\nShape 1 batch: {images.shape}  (batch_size, channels, H, W)")
print(f"Label shape: {labels.shape}")
print(f"Giá trị pixel sau chuẩn hoá: min={images.min():.2f}, max={images.max():.2f}")
"""),

    code("""# Visualize ảnh mẫu (cần denormalize để hiển thị đúng màu)
CIFAR100_MEAN = torch.tensor([0.5071, 0.4867, 0.4408])
CIFAR100_STD  = torch.tensor([0.2675, 0.2565, 0.2761])

def denormalize(img_tensor):
    \"\"\"Chuyển từ normalized → [0,1] để hiển thị.\"\"\"
    return (img_tensor * CIFAR100_STD[:, None, None] + CIFAR100_MEAN[:, None, None]).clamp(0, 1)

fig, axes = plt.subplots(3, 8, figsize=(14, 6))
for i, ax in enumerate(axes.flat):
    img = denormalize(images[i]).permute(1, 2, 0).numpy()
    ax.imshow(img)
    ax.set_title(class_names[labels[i].item()], fontsize=7)
    ax.axis('off')
plt.suptitle('Mẫu ảnh từ CIFAR-100 (sau chuẩn hoá + denormalize để hiển thị)', fontsize=11)
plt.tight_layout()
plt.show()
"""),

    md("""## 2. Softmax Regression

**Ý tưởng:** Flatten ảnh thành vector 3072 chiều, ánh xạ tuyến tính đến 100 lớp.

```
[B, 3, 32, 32] → Flatten → [B, 3072] → Linear(3072, 100) → [B, 100]
```

**Lưu ý quan trọng:** Không cần thêm `softmax` trong `forward()` vì:
- `nn.CrossEntropyLoss` = `log_softmax` + `NLLLoss`
- Nó tự tính softmax bên trong khi tính loss
- Nếu thêm softmax rồi đưa vào CrossEntropyLoss → KẾT QUẢ SAI (double softmax)!
"""),

    code("""model_softmax = SoftmaxRegression(num_classes=100)
print(model_softmax)
print(f"\\nSố tham số: {get_param_count(model_softmax)}")

# Test forward pass
x_dummy = torch.randn(4, 3, 32, 32)
out = model_softmax(x_dummy)
print(f"\\nInput shape:  {x_dummy.shape}")
print(f"Output shape: {out.shape}  (batch=4, num_classes=100)")
print(f"Output là logits (chưa qua softmax): min={out.min():.2f}, max={out.max():.2f}")
"""),

    md("""## 3. MLP (Multi-Layer Perceptron)

**Cải tiến so với Softmax:** Thêm các lớp ẩn với hàm kích hoạt ReLU
→ Học được các đặc trưng **phi tuyến**.

**Kiến trúc:**
```
Flatten → Linear(3072→512) → BatchNorm → ReLU → Dropout(0.3)
        → Linear(512→256)  → BatchNorm → ReLU → Dropout(0.3)
        → Linear(256→100)
```

**BatchNorm1d:** Chuẩn hoá output của mỗi lớp để training ổn định hơn.
**Dropout(0.3):** Tắt ngẫu nhiên 30% neurons → giảm overfitting.

**Nhược điểm của MLP:** `Flatten` làm mất thông tin về vị trí không gian (spatial).
Pixel (0,0) và pixel (31,31) được xử lý như các đặc trưng độc lập, không có quan hệ gần-xa.
→ CNN giải quyết vấn đề này với tích chập cục bộ.
"""),

    code("""model_mlp = MLP(num_classes=100)
print(model_mlp)
print(f"\\nSố tham số: {get_param_count(model_mlp)}")

out = model_mlp(x_dummy)
print(f"Output shape: {out.shape}")
"""),

    md("""## 4. SimpleCNN

**Ý tưởng tích chập (Convolution):**
- Mỗi bộ lọc (filter/kernel) kích thước 3×3 trượt qua ảnh
- Học được các đặc trưng cục bộ: cạnh, góc, texture, ...
- **Weight sharing**: cùng bộ lọc áp dụng tại mọi vị trí → ít params hơn
- **Translation invariance**: nhận diện mèo dù ở góc trái hay góc phải

**Kiến trúc (VGG-style):**
```
Conv(3→32)×2 + BN + ReLU + MaxPool(2)  → [B, 32, 16, 16]
Conv(32→64)×2 + BN + ReLU + MaxPool(2) → [B, 64, 8, 8]
Conv(64→128)×2 + BN + ReLU + MaxPool(2)→ [B, 128, 4, 4]
AdaptiveAvgPool(1) → Flatten → Linear(128→256) → ReLU → Linear(256→100)
```

**AdaptiveAvgPool(1):** Thay vì Flatten(128×4×4=2048) → Linear lớn,
pool về 128 features → ít tham số + ít overfitting.
"""),

    code("""model_cnn = SimpleCNN(num_classes=100)
print(model_cnn)
print(f"\\nSố tham số: {get_param_count(model_cnn)}")

# Trace shape qua từng bước
print("\\n--- Shape trace qua CNN ---")
x = x_dummy
x_feat = model_cnn.features[0](x)  # Block 1
print(f"Sau Conv Block 1: {x_feat.shape}  (→ 32 feature maps, 16×16)")
x_feat = model_cnn.features[1](x_feat)  # Block 2
print(f"Sau Conv Block 2: {x_feat.shape}  (→ 64 feature maps, 8×8)")
x_feat = model_cnn.features[2](x_feat)  # Block 3
print(f"Sau Conv Block 3: {x_feat.shape}  (→ 128 feature maps, 4×4)")
out = model_cnn(x_dummy)
print(f"Output logits:    {out.shape}")
"""),

    code("""# Visualize feature maps sau Conv Block 1
model_cnn.eval()
with torch.no_grad():
    feat_maps = model_cnn.features[0](images[:1])  # Chỉ lấy 1 ảnh

# feat_maps: [1, 32, 16, 16] → 32 feature maps
n_maps = 16
fig, axes = plt.subplots(2, n_maps // 2, figsize=(14, 4))
for i, ax in enumerate(axes.flat):
    if i < n_maps:
        fm = feat_maps[0, i].numpy()
        ax.imshow(fm, cmap='viridis')
        ax.set_title(f'Filter {i+1}', fontsize=7)
    ax.axis('off')
plt.suptitle('Feature maps sau Conv Block 1 (16 trong 32 filters)', fontsize=11)
plt.tight_layout()
plt.show()
"""),

    md("""## 5. SimpleViT — Vision Transformer

**Ý tưởng ViT:** Thay vì xử lý ảnh theo không gian (CNN),
chia ảnh thành các **patches** rồi xử lý như một chuỗi tokens (Transformer).

**Các bước:**
1. **Patch Embedding:** Chia 32×32 thành 64 patches 4×4
   - `Conv2d(3, 128, kernel_size=4, stride=4)` ≡ tích chập không chồng lấp = chia patch
2. **CLS Token:** Token đặc biệt học cách tổng hợp thông tin từ toàn ảnh
3. **Positional Encoding:** Cho model biết vị trí của mỗi patch (Transformer không có khái niệm thứ tự)
4. **Transformer Encoder:** Self-attention giữa tất cả 64 patches
5. **Classification:** Lấy CLS token → Linear → 100 lớp

**Tại sao patch_size=4 (không phải 16)?**
- ViT-B/16 dùng patch 16×16 vì được train trên ảnh 224×224 → 196 patches
- CIFAR-100 chỉ có 32×32: patch 16×16 → chỉ 4 patches — quá ít!
- Patch 4×4 → 64 patches — phù hợp hơn cho ảnh nhỏ
"""),

    code("""model_vit = SimpleViT(num_classes=100)
print(model_vit)
print(f"\\nSố tham số: {get_param_count(model_vit)}")

# Trace shape
print("\\n--- Shape trace qua ViT ---")
x = x_dummy
patches = model_vit.patch_embed(x)
print(f"Sau patch embedding (Conv2d): {patches.shape}  ([B, d_model, 8, 8])")
patches_seq = patches.flatten(2).transpose(1, 2)
print(f"Sau reshape thành sequence:  {patches_seq.shape}  ([B, 64, 128])")
print(f"+ CLS token:                  [B, 65, 128]")
print(f"Sau Transformer:              [B, 65, 128]")
out = model_vit(x_dummy)
print(f"Output (từ CLS token):        {out.shape}")
"""),

    code("""# Visualize 64 patches từ 1 ảnh
img = images[0]  # [3, 32, 32]
img_show = denormalize(img).permute(1, 2, 0).numpy()

# Chia thành 8×8 grid of patches 4×4
fig, axes = plt.subplots(8, 8, figsize=(8, 8))
for i in range(8):
    for j in range(8):
        patch = img_show[i*4:(i+1)*4, j*4:(j+1)*4, :]
        axes[i, j].imshow(patch)
        axes[i, j].axis('off')

plt.suptitle(f'64 patches 4×4 từ ảnh "{class_names[labels[0].item()]}"', fontsize=11)
plt.tight_layout()
plt.show()
"""),

    md("""## 6. Tổng kết — So sánh kiến trúc

| Mô hình | Kiến trúc | Số params | Inductive bias |
|---------|-----------|-----------|----------------|
| Softmax Regression | Flatten → Linear | ~307K | Không có |
| MLP | Flatten → FC × 2 → Linear | ~1.7M | Không có |
| SimpleCNN | Conv Blocks × 3 → FC | ~2.1M | Locality + Translation invariance |
| SimpleViT | Patch embed → Transformer | ~2.3M | Không có (học từ data) |

**Nhận xét:**
- CNN có **inductive bias** mạnh (biết rằng đặc trưng cục bộ quan trọng)
  → Train tốt với ít data hơn
- ViT **không có inductive bias** nhưng học được patterns phức tạp hơn
  → Cần nhiều data và epochs hơn để bắt kịp CNN

Tiếp theo: Notebook **Part 2** sẽ train và so sánh 4 mô hình này.
"""),

    code("""# Bảng tóm tắt số params
models = {
    "SoftmaxRegression": model_softmax,
    "MLP": model_mlp,
    "SimpleCNN": model_cnn,
    "SimpleViT": model_vit,
}

print(f"{'Model':<25} | {'Params':>8}")
print("-" * 37)
for name, m in models.items():
    print(f"{name:<25} | {get_param_count(m):>8}")
"""),
)


# ─── Notebook 2: Part 2 — Training Loop & So sánh ────────────────────────────

NB2 = nb(
    md("""# Phần 2 — Vòng lặp huấn luyện và So sánh

**Mục tiêu:** Tự viết training loop và huấn luyện 4 mô hình từ Phần 1.

Theo yêu cầu bài tập: **Không dùng** trainer.fit() hay API cấp cao.
Tự viết từng bước:
1. `optimizer.zero_grad()` — Xoá gradient cũ
2. `logits = model(x)` — Forward pass
3. `loss = criterion(logits, y)` — Tính loss
4. `loss.backward()` — Backpropagation
5. `clip_grad_norm_(...)` — Clip gradient
6. `optimizer.step()` — Cập nhật tham số
"""),

    code("""import sys, os
sys.path.insert(0, os.path.abspath('..'))

import torch
import torch.nn as nn
import json
import matplotlib.pyplot as plt

from src.data import get_cifar100_loaders, get_device
from src.models_part1 import SoftmaxRegression, MLP, SimpleCNN, SimpleViT
from src.train import fit, evaluate, load_best_model
from src.utils import (
    get_param_count, compute_metrics, get_predictions,
    plot_training_curves, plot_multi_curves, plot_comparison_bar,
    print_results_table, save_metrics_json
)

DEVICE = get_device()
print(f"Thiết bị: {DEVICE}")

os.makedirs('../exercise/results/checkpoints', exist_ok=True)
os.makedirs('../exercise/results/plots', exist_ok=True)
os.makedirs('../exercise/results/metrics', exist_ok=True)
"""),

    code("""# Tải dataset
train_loader, val_loader, test_loader, class_names = get_cifar100_loaders(batch_size=128)
"""),

    md("""## 1. Vòng lặp huấn luyện — Giải thích từng bước

```python
def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()  # Bật dropout + batchnorm training mode

    for x, y in loader:
        x, y = x.to(device), y.to(device)

        # Bước 1: Xoá gradient
        # PyTorch CỘNG DỒN gradient qua các iterations
        # → Phải xoá trước mỗi batch, không phải cuối epoch!
        optimizer.zero_grad()

        # Bước 2: Forward pass
        logits = model(x)   # [B, 100]

        # Bước 3: Tính loss
        # CrossEntropyLoss = log_softmax(logits) + NLLLoss
        # = -log(softmax(logits)[correct_class])
        loss = criterion(logits, y)

        # Bước 4: Backward pass
        # Tính đạo hàm ∂loss/∂θ cho mọi tham số θ
        loss.backward()

        # Bước 5: Gradient clipping
        # Nếu norm của gradient > 1.0, scale xuống
        # Ngăn "exploding gradients" đặc biệt trong RNN/Transformer
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        # Bước 6: Cập nhật tham số
        # θ = θ - lr × ∂loss/∂θ  (với AdamW thì phức tạp hơn một chút)
        optimizer.step()
```

**Scheduler:** `CosineAnnealingLR` — Learning rate giảm dần theo hình cosine:
```
LR(t) = lr_min + 0.5 × (lr_max - lr_min) × (1 + cos(π × t/T_max))
```
→ Bắt đầu cao (learn nhanh), kết thúc thấp (tinh chỉnh)
"""),

    md("""## 2. Cấu hình Hyperparameters

| Mô hình | LR | Batch | Epochs | Lý do |
|---------|-----|-------|--------|-------|
| SoftmaxRegression | 0.1 | 256 | 30 | Bài toán lồi, LR cao hội tụ nhanh |
| MLP | 1e-3 | 128 | 50 | LR thấp hơn vì có nhiều lớp |
| SimpleCNN | 1e-3 | 128 | 50 | Standard CNN setup |
| SimpleViT | 3e-4 | 128 | 100 | ViT cần nhiều epochs hơn CNN khi train từ đầu |

**Kỳ vọng accuracy** (CIFAR-100, không dùng pretrained):
- Softmax: ~15-20% (chỉ tuyến tính)
- MLP: ~35-42% (phi tuyến nhưng không có spatial bias)
- CNN: ~50-60% (tận dụng cấu trúc không gian)
- ViT: ~35-50% (cần nhiều data hơn để cạnh tranh với CNN)
"""),

    md("""## 3. Huấn luyện các mô hình

⚠️ **Lưu ý:** Quá trình train có thể mất vài giờ trên CPU.
Trên GPU/MPS thường nhanh hơn 5-10×.

Đặt `TRAIN_MODE = False` để tải checkpoints đã train sẵn (nếu có).
"""),

    code("""TRAIN_MODE = True  # Đặt False để load checkpoint đã train

histories = {}

# ── Softmax Regression ──────────────────────────────────────────────
model = SoftmaxRegression(num_classes=100).to(DEVICE)
ckpt = '../exercise/results/checkpoints/softmax.pt'
config = {"epochs": 30, "lr": 0.1, "device": DEVICE, "save_path": ckpt}

if TRAIN_MODE:
    print("=" * 50)
    print("Training: SoftmaxRegression")
    print("=" * 50)
    histories["SoftmaxRegression"] = fit(model, train_loader, val_loader, config)
    save_metrics_json(histories["SoftmaxRegression"],
                      '../exercise/results/metrics/softmax_history.json')
else:
    from src.utils import load_metrics_json
    histories["SoftmaxRegression"] = load_metrics_json('../exercise/results/metrics/softmax_history.json')
    model = load_best_model(model, ckpt, DEVICE)

print("✓ SoftmaxRegression done")
"""),

    code("""# ── MLP ──────────────────────────────────────────────────────────────
model_mlp = MLP(num_classes=100).to(DEVICE)
ckpt_mlp = '../exercise/results/checkpoints/mlp.pt'
config_mlp = {"epochs": 50, "lr": 1e-3, "device": DEVICE, "save_path": ckpt_mlp}

if TRAIN_MODE:
    print("=" * 50)
    print("Training: MLP")
    print("=" * 50)
    histories["MLP"] = fit(model_mlp, train_loader, val_loader, config_mlp)
    save_metrics_json(histories["MLP"], '../exercise/results/metrics/mlp_history.json')
else:
    histories["MLP"] = load_metrics_json('../exercise/results/metrics/mlp_history.json')
    model_mlp = load_best_model(model_mlp, ckpt_mlp, DEVICE)

print("✓ MLP done")
"""),

    code("""# ── SimpleCNN ────────────────────────────────────────────────────────
model_cnn = SimpleCNN(num_classes=100).to(DEVICE)
ckpt_cnn = '../exercise/results/checkpoints/cnn.pt'
config_cnn = {"epochs": 50, "lr": 1e-3, "device": DEVICE, "save_path": ckpt_cnn}

if TRAIN_MODE:
    print("=" * 50)
    print("Training: SimpleCNN")
    print("=" * 50)
    histories["SimpleCNN"] = fit(model_cnn, train_loader, val_loader, config_cnn)
    save_metrics_json(histories["SimpleCNN"], '../exercise/results/metrics/cnn_history.json')
else:
    histories["SimpleCNN"] = load_metrics_json('../exercise/results/metrics/cnn_history.json')
    model_cnn = load_best_model(model_cnn, ckpt_cnn, DEVICE)

print("✓ SimpleCNN done")
"""),

    code("""# ── SimpleViT ────────────────────────────────────────────────────────
model_vit = SimpleViT(num_classes=100).to(DEVICE)
ckpt_vit = '../exercise/results/checkpoints/vit.pt'
config_vit = {"epochs": 100, "lr": 3e-4, "device": DEVICE, "save_path": ckpt_vit}

if TRAIN_MODE:
    print("=" * 50)
    print("Training: SimpleViT (PyTorch)")
    print("=" * 50)
    histories["SimpleViT"] = fit(model_vit, train_loader, val_loader, config_vit)
    save_metrics_json(histories["SimpleViT"], '../exercise/results/metrics/vit_history.json')
else:
    histories["SimpleViT"] = load_metrics_json('../exercise/results/metrics/vit_history.json')
    model_vit = load_best_model(model_vit, ckpt_vit, DEVICE)

print("✓ SimpleViT done")
"""),

    md("""## 4. Biểu đồ Training Curves"""),

    code("""# Vẽ training curves cho từng mô hình riêng lẻ
for name, hist in histories.items():
    plot_training_curves(hist, title=name,
                         save_path=f'../exercise/results/plots/{name.lower()}_curves.png')
"""),

    code("""# Vẽ val_accuracy của tất cả mô hình trên 1 biểu đồ
plot_multi_curves(
    list(histories.values()),
    list(histories.keys()),
    title="So sánh Val Accuracy — 4 mô hình (Phần 1)",
    save_path='../exercise/results/plots/part1_2_comparison_curves.png'
)
"""),

    md("""## 5. Đánh giá trên tập Test"""),

    code("""# Tải lại checkpoint tốt nhất và evaluate trên test set
trained_models = {
    "SoftmaxRegression": (SoftmaxRegression(100), '../exercise/results/checkpoints/softmax.pt'),
    "MLP": (MLP(100), '../exercise/results/checkpoints/mlp.pt'),
    "SimpleCNN": (SimpleCNN(100), '../exercise/results/checkpoints/cnn.pt'),
    "SimpleViT": (SimpleViT(100), '../exercise/results/checkpoints/vit.pt'),
}

results = {}
for name, (m, ckpt_path) in trained_models.items():
    if os.path.exists(ckpt_path):
        m = load_best_model(m, ckpt_path, DEVICE)
        preds, labels = get_predictions(m, test_loader, DEVICE)
        metrics = compute_metrics(preds, labels)

        # Val acc từ history
        best_val_acc = max(histories[name]["val_acc"])

        results[name] = {
            "test_acc": metrics["accuracy"],
            "val_acc": best_val_acc,
            "f1_macro": metrics["f1_macro"],
            "params": get_param_count(m),
        }

save_metrics_json(results, '../exercise/results/metrics/part1_2_results.json')
print_results_table(results)
"""),

    code("""# Biểu đồ so sánh test accuracy
plot_comparison_bar(
    results, metric="test_acc",
    title="Test Accuracy — Phần 1 & 2 (CIFAR-100)",
    save_path='../exercise/results/plots/part1_2_bar.png'
)
"""),

    md("""## 6. Nhận xét và Phân tích

**Phân tích kết quả:**

1. **Softmax Regression** đạt accuracy thấp nhất vì:
   - Chỉ học được **biên quyết định tuyến tính**
   - Không thể phân biệt các lớp phức tạp với đặc trưng phi tuyến
   - 100 lớp với chỉ phép ánh xạ tuyến tính → bị underfitting nghiêm trọng

2. **MLP** tốt hơn nhờ **lớp ẩn phi tuyến**, nhưng:
   - `Flatten` làm mất **thông tin không gian** (spatial)
   - Pixel (0,0) và (1,1) kề nhau được xử lý như 2 features độc lập hoàn toàn
   - Dễ **overfitting** với 100 lớp và ít data

3. **CNN** thường đạt tốt nhất nhờ:
   - **Tích chập cục bộ** (local connectivity): học đặc trưng từ vùng lân cận
   - **Weight sharing**: giảm số params, giảm overfitting
   - **Translation invariance**: nhận diện bất kể vị trí
   - Phù hợp với **inductive bias** của ảnh (spatial structure quan trọng)

4. **ViT** có thể kém CNN khi:
   - Train từ đầu với dataset nhỏ (~45K ảnh)
   - Không có spatial inductive bias → cần học hoàn toàn từ data
   - Thường cần **pretraining** trên dataset lớn (như ImageNet) để phát huy hết sức mạnh

**Kết luận:** Với dataset nhỏ và train từ đầu, CNN thường vượt trội ViT.
ViT chỉ thực sự mạnh khi được pretrain trên hàng triệu ảnh.
"""),
)


# ─── Notebook 3: Part 3 — Custom Transformer ─────────────────────────────────

NB3 = nb(
    md("""# Phần 3 — Tự hiện thực TransformerEncoder

**Mục tiêu:** Xây dựng lại TransformerEncoder từ các phép toán cơ bản:
- `nn.Linear`, `nn.LayerNorm`, `torch.einsum`
- **Không dùng** `nn.TransformerEncoderLayer` hay `nn.TransformerEncoder`

Sau đó so sánh **Custom ViT** với **PyTorch ViT** từ Phần 1.
"""),

    code("""import sys, os
sys.path.insert(0, os.path.abspath('..'))

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

from src.data import get_cifar100_loaders, get_device
from src.models_part1 import SimpleViT
from src.models_part3 import CustomMultiHeadAttention, CustomTransformerEncoderLayer, CustomViT
from src.train import fit, load_best_model
from src.utils import (get_param_count, get_predictions, compute_metrics,
                       plot_training_curves, plot_multi_curves, print_results_table,
                       save_metrics_json, load_metrics_json)

DEVICE = get_device()
print(f"Thiết bị: {DEVICE}")
train_loader, val_loader, test_loader, class_names = get_cifar100_loaders(batch_size=128)
"""),

    md("""## 1. Cơ chế Attention — Giải thích trực quan

**Self-Attention là gì?**

Mỗi token (patch) hỏi: *"Tôi nên chú ý đến những token nào khác?"*

**3 vector cho mỗi token:**
- **Query (Q):** "Tôi đang tìm kiếm gì?"
- **Key (K):**   "Tôi có thể cung cấp gì?"
- **Value (V):** "Thông tin thực sự của tôi là gì?"

**Công thức:**
```
scores = Q @ K^T / sqrt(d_head)   # Độ tương đồng giữa mọi cặp tokens
attn   = softmax(scores, dim=-1)  # Normalize thành phân phối xác suất
output = attn @ V                  # Tổng có trọng số của Values
```

**Multi-Head:** Thực hiện Attention song song với `num_heads` bộ Q,K,V khác nhau
→ Mỗi head học 1 loại quan hệ khác nhau
"""),

    code("""# Minh hoạ Attention trên toy example (T=5 tokens)
torch.manual_seed(42)
d_model, num_heads, T = 8, 2, 5
d_head = d_model // num_heads

# Tạo input ngẫu nhiên
x_toy = torch.randn(1, T, d_model)  # [1, 5, 8]

# Custom MHA
mha = CustomMultiHeadAttention(d_model, num_heads)
with torch.no_grad():
    # Lấy attention weights để visualize
    Q = mha.W_q(x_toy).reshape(1, T, num_heads, d_head).transpose(1, 2)
    K = mha.W_k(x_toy).reshape(1, T, num_heads, d_head).transpose(1, 2)
    scores = torch.einsum('bhid,bhjd->bhij', Q, K) / (d_head ** 0.5)
    attn_weights = F.softmax(scores, dim=-1)

fig, axes = plt.subplots(1, num_heads, figsize=(10, 4))
for h in range(num_heads):
    im = axes[h].imshow(attn_weights[0, h].numpy(), cmap='Blues', vmin=0, vmax=1)
    axes[h].set_title(f'Head {h+1}\\nAttention Matrix')
    axes[h].set_xlabel('Key (token j)')
    axes[h].set_ylabel('Query (token i)')
    plt.colorbar(im, ax=axes[h])

plt.suptitle('Attention weights: token i "chú ý" đến token j bao nhiêu?', fontsize=11)
plt.tight_layout()
plt.show()
print(f"Kiểm tra: mỗi hàng tổng = 1.0?  {attn_weights.sum(dim=-1).mean():.6f}")
"""),

    md("""## 2. Hiện thực Custom Multi-Head Attention

Giải thích chi tiết về einsum notation:

```python
# 'bhid,bhjd->bhij' nghĩa là:
# b = batch, h = head, i = query position, j = key position, d = d_head
# Với mỗi (b, h, i): tính dot product với mọi j
# → scores[b, h, i, j] = sum_d(Q[b,h,i,d] * K[b,h,j,d])

scores = torch.einsum('bhid,bhjd->bhij', Q, K) / sqrt(d_head)

# 'bhij,bhjd->bhid' nghĩa là:
# Với mỗi (b, h, i): tổng có trọng số của Values
# → out[b, h, i, d] = sum_j(attn[b,h,i,j] * V[b,h,j,d])
out = torch.einsum('bhij,bhjd->bhid', attn, V)
```
"""),

    code("""# Kiểm tra Custom MHA vs nn.MultiheadAttention
# (Không thể so sánh output trực tiếp vì khởi tạo weights khác nhau,
#  nhưng có thể kiểm tra: shape giống nhau + attention distribution hợp lệ)

custom_mha = CustomMultiHeadAttention(d_model=64, num_heads=4)
pytorch_mha = nn.MultiheadAttention(embed_dim=64, num_heads=4, batch_first=True)

x_test = torch.randn(2, 10, 64)  # [batch=2, seq=10, d_model=64]

with torch.no_grad():
    custom_out = custom_mha(x_test)
    pytorch_out, attn_weights = pytorch_mha(x_test, x_test, x_test)

print(f"Custom MHA output shape:  {custom_out.shape}")
print(f"PyTorch MHA output shape: {pytorch_out.shape}")
print(f"Shapes khớp nhau: {custom_out.shape == pytorch_out.shape}")
print()

# Kiểm tra attention là phân phối hợp lệ (mỗi hàng sum=1)
# Tính lại attention weights từ custom MHA để verify
with torch.no_grad():
    Q = custom_mha.W_q(x_test).reshape(2, 10, 4, 16).transpose(1, 2)
    K = custom_mha.W_k(x_test).reshape(2, 10, 4, 16).transpose(1, 2)
    scores = torch.einsum('bhid,bhjd->bhij', Q, K) / (16 ** 0.5)
    attn = F.softmax(scores, dim=-1)

print(f"Attention row sums (should all be ~1.0):")
print(f"  Mean: {attn.sum(dim=-1).mean():.6f}")
print(f"  Min:  {attn.sum(dim=-1).min():.6f}")
print(f"  Max:  {attn.sum(dim=-1).max():.6f}")
print("✓ Attention distribution hợp lệ!")
"""),

    md("""## 3. Xây dựng Custom ViT"""),

    code("""model_custom_vit = CustomViT(num_classes=100)
print(model_custom_vit)
print(f"\\nSố tham số Custom ViT: {get_param_count(model_custom_vit)}")

# So sánh với PyTorch ViT
model_pytorch_vit = SimpleViT(num_classes=100)
print(f"Số tham số PyTorch ViT: {get_param_count(model_pytorch_vit)}")
print(f"\\n(Hai mô hình có cùng hyperparams: d_model=128, heads=4, layers=4, patch_size=4)")
"""),

    md("""## 4. Huấn luyện và So sánh"""),

    code("""TRAIN_MODE = True

histories_p3 = {}

# ── PyTorch ViT (load từ Part 2 nếu đã train) ──
ckpt_vit = '../exercise/results/checkpoints/vit.pt'
if os.path.exists('../exercise/results/metrics/vit_history.json'):
    histories_p3["SimpleViT (PyTorch)"] = load_metrics_json('../exercise/results/metrics/vit_history.json')
    print("Loaded PyTorch ViT history từ Part 2")
else:
    model_vit = SimpleViT(100).to(DEVICE)
    histories_p3["SimpleViT (PyTorch)"] = fit(
        model_vit, train_loader, val_loader,
        {"epochs": 100, "lr": 3e-4, "device": DEVICE, "save_path": ckpt_vit}
    )

# ── Custom ViT ──
ckpt_custom = '../exercise/results/checkpoints/custom_vit.pt'
config_custom = {"epochs": 100, "lr": 3e-4, "device": DEVICE, "save_path": ckpt_custom}

if TRAIN_MODE:
    print("=" * 50)
    print("Training: CustomViT")
    print("=" * 50)
    model_custom_vit = CustomViT(100).to(DEVICE)
    histories_p3["CustomViT (Tự xây)"] = fit(model_custom_vit, train_loader, val_loader, config_custom)
    save_metrics_json(histories_p3["CustomViT (Tự xây)"],
                      '../exercise/results/metrics/custom_vit_history.json')
else:
    histories_p3["CustomViT (Tự xây)"] = load_metrics_json(
        '../exercise/results/metrics/custom_vit_history.json')

print("✓ Done")
"""),

    code("""# Vẽ training curves
plot_multi_curves(
    list(histories_p3.values()),
    list(histories_p3.keys()),
    title="So sánh: SimpleViT (PyTorch) vs CustomViT (Tự xây)",
    save_path='../exercise/results/plots/part3_comparison_curves.png'
)
"""),

    code("""# Bảng kết quả
results_p3 = {}
vit_pairs = {
    "SimpleViT (PyTorch)": (SimpleViT(100), ckpt_vit),
    "CustomViT (Tự xây)": (CustomViT(100), ckpt_custom),
}

for name, (m, ckpt) in vit_pairs.items():
    if os.path.exists(ckpt):
        m = load_best_model(m, ckpt, DEVICE)
        preds, labels = get_predictions(m, test_loader, DEVICE)
        metrics = compute_metrics(preds, labels)
        best_val = max(histories_p3[name]["val_acc"])
        results_p3[name] = {
            "test_acc": metrics["accuracy"],
            "val_acc": best_val,
            "f1_macro": metrics["f1_macro"],
            "params": get_param_count(m),
        }

save_metrics_json(results_p3, '../exercise/results/metrics/part3_results.json')
print_results_table(results_p3)
"""),

    md("""## 5. Nhận xét

**Kết quả dự kiến:**
- Hai mô hình có accuracy rất gần nhau (sai lệch < 1-2%)
- Điều này **xác nhận** hiện thực Custom Transformer là đúng!

**Tại sao không giống hệt nhau?**
- Khởi tạo weights ngẫu nhiên khác nhau → convergence khác nhau
- `nn.TransformerEncoderLayer` dùng `F.scaled_dot_product_attention` (tối ưu hơn)
- Thứ tự áp dụng LayerNorm có thể khác nhau đôi chút

**Bài học:**
- Hiện thực từ đầu giúp hiểu sâu cơ chế bên trong
- Kết quả tương đương xác nhận tính đúng đắn
- Framework có sẵn (PyTorch) được tối ưu hóa tốt hơn về tốc độ
"""),
)


# ─── Notebook 4: Part 4 — Kiến trúc kết hợp ─────────────────────────────────

NB4 = nb(
    md("""# Phần 4 — Kiến trúc kết hợp và Cách Embed ảnh khác nhau

**Khám phá 3 cách tokenize ảnh cho Transformer:**

| Kiến trúc | Token = ? | Số tokens | Feature dim | Seq len |
|-----------|-----------|-----------|-------------|---------|
| **4A: CNN+Transformer** | CNN feature positions | 64 | 64 (CNN) | 64 |
| **4B: Spatial Tokens** | Mỗi pixel (H×W) | 1024 | 3 (RGB) | 1024 |
| **4C: Channel Tokens** | Mỗi channel | 64 | 1024 (spatial) | 64 |

**Câu hỏi:** Cách tokenize nào giúp Transformer học tốt nhất?
"""),

    code("""import sys, os
sys.path.insert(0, os.path.abspath('..'))
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

from src.data import get_cifar100_loaders, get_device
from src.models_part4 import CNNTransformerHybrid, SpatialTokenViT, ChannelTokenViT
from src.train import fit, load_best_model
from src.utils import (get_param_count, get_predictions, compute_metrics,
                       plot_multi_curves, plot_comparison_bar, print_results_table,
                       save_metrics_json, load_metrics_json)

DEVICE = get_device()
print(f"Thiết bị: {DEVICE}")
train_loader, val_loader, test_loader, class_names = get_cifar100_loaders(batch_size=128)
"""),

    md("""## 1. Diagram: So sánh 3 cách tokenize"""),

    code("""# Visualize 3 cách tokenize trên 1 ảnh mẫu
images, labels = next(iter(test_loader))
img = images[0]  # [3, 32, 32]
CIFAR100_MEAN = torch.tensor([0.5071, 0.4867, 0.4408])
CIFAR100_STD  = torch.tensor([0.2675, 0.2565, 0.2761])
img_show = (img * CIFAR100_STD[:, None, None] + CIFAR100_MEAN[:, None, None]).clamp(0,1)
img_np = img_show.permute(1, 2, 0).numpy()

fig, axes = plt.subplots(1, 4, figsize=(14, 4))

# Original
axes[0].imshow(img_np)
axes[0].set_title(f'Ảnh gốc\\n{class_names[labels[0].item()]}')
axes[0].axis('off')

# 4A: CNN patches (8×8 grid = 64 spatial positions)
axes[1].imshow(img_np)
for i in range(8):
    for j in range(8):
        rect = patches.Rectangle((j*4-0.5, i*4-0.5), 4, 4,
                                   linewidth=0.5, edgecolor='red', facecolor='none')
        axes[1].add_patch(rect)
axes[1].set_title('4A: CNN Features\\n8×8=64 spatial tokens')
axes[1].axis('off')

# 4B: Spatial tokens (every pixel)
axes[2].imshow(img_np)
for i in range(0, 32, 4):  # Chỉ vẽ mỗi 4 pixel để thấy rõ
    for j in range(0, 32, 4):
        rect = patches.Rectangle((j-0.5, i-0.5), 1, 1,
                                   linewidth=0.5, edgecolor='blue', facecolor='blue', alpha=0.3)
        axes[2].add_patch(rect)
axes[2].set_title('4B: Spatial Tokens\\n32×32=1024 pixel tokens')
axes[2].axis('off')

# 4C: Channel tokens
channel_names = ['R', 'G', 'B']
channel_imgs = [img_np[:,:,c] for c in range(3)]
axes[3].axis('off')
axes[3].set_title('4C: Channel Tokens\\n64 channel tokens')
inset_positions = [(0.05, 0.05), (0.37, 0.05), (0.69, 0.05)]
for pos, ch_img, ch_name in zip(inset_positions, channel_imgs, channel_names):
    inset = axes[3].inset_axes([pos[0], pos[1], 0.28, 0.9])
    inset.imshow(ch_img, cmap='gray')
    inset.set_title(ch_name, fontsize=8)
    inset.axis('off')

plt.suptitle('3 cách tokenize ảnh cho Transformer', fontsize=12)
plt.tight_layout()
plt.show()
"""),

    md("""## 2. Kiến trúc 4A: CNN + Transformer Hybrid"""),

    code("""model_4a = CNNTransformerHybrid(num_classes=100)
print(f"CNNTransformerHybrid — Params: {get_param_count(model_4a)}")

# Trace shape
x_dummy = torch.randn(2, 3, 32, 32)
with torch.no_grad():
    cnn_out = model_4a.cnn(x_dummy)
    print(f"\\nSau CNN backbone: {cnn_out.shape}  [B, 64, 8, 8]")
    B, C, H, W = cnn_out.shape
    tokens = cnn_out.reshape(B, C, H*W).permute(0, 2, 1)
    print(f"Sau reshape:      {tokens.shape}  [B, 64 tokens, 64 features]")
    tokens_proj = model_4a.token_proj(tokens)
    print(f"Sau projection:   {tokens_proj.shape}  [B, 64 tokens, 128 d_model]")
    out = model_4a(x_dummy)
    print(f"Output:           {out.shape}")
"""),

    md("""## 3. Kiến trúc 4B: Spatial Token ViT

⚠️ **Cảnh báo bộ nhớ:** 1024 tokens → attention matrix [B, H, 1024, 1024]
Với batch=32, 4 heads: ~512MB RAM!
Sử dụng `batch_size=32` cho mô hình này.
"""),

    code("""model_4b = SpatialTokenViT(num_classes=100)
print(f"SpatialTokenViT — Params: {get_param_count(model_4b)}")

# Trace shape
with torch.no_grad():
    B, C, H, W = x_dummy.shape
    tokens = x_dummy.reshape(B, C, H*W).permute(0, 2, 1)
    print(f"\\nSau reshape: {tokens.shape}  [B, 1024 pixel tokens, 3 RGB features]")
    tokens_proj = model_4b.pixel_proj(tokens)
    print(f"Sau projection: {tokens_proj.shape}  [B, 1024, 64 d_model]")
    out = model_4b(x_dummy)
    print(f"Output: {out.shape}")

# Tính attention matrix size
attn_size_MB = 32 * 4 * 1024 * 1024 * 4 / 1024**2
print(f"\\n⚠️ Attention matrix size (batch=32, 4 heads): {attn_size_MB:.0f}MB")
print("→ Dùng batch_size=32 cho mô hình này")
"""),

    code("""# Spatial loader với batch_size=32
from src.data import get_cifar100_loaders
train_loader_small, val_loader_small, test_loader_small, _ = get_cifar100_loaders(batch_size=32)
"""),

    md("""## 4. Kiến trúc 4C: Channel Token ViT"""),

    code("""model_4c = ChannelTokenViT(num_classes=100)
print(f"ChannelTokenViT — Params: {get_param_count(model_4c)}")

with torch.no_grad():
    x_expanded = model_4c.channel_expand(x_dummy)
    print(f"\\nSau Conv 1×1 expand: {x_expanded.shape}  [B, 64 channels, 32, 32]")
    B, C, H, W = x_expanded.shape
    tokens = x_expanded.reshape(B, C, H*W)
    print(f"Sau reshape:         {tokens.shape}  [B, 64 channel tokens, 1024 spatial features]")
    tokens_proj = model_4c.spatial_proj(tokens)
    print(f"Sau projection:      {tokens_proj.shape}  [B, 64, 128 d_model]")
    out = model_4c(x_dummy)
    print(f"Output:              {out.shape}")
"""),

    md("""## 5. Huấn luyện và So sánh"""),

    code("""TRAIN_MODE = True
histories_p4 = {}

configs_p4 = {
    "CNNTransformerHybrid": {
        "model": CNNTransformerHybrid(100),
        "loader": (train_loader, val_loader),
        "epochs": 50, "lr": 1e-3,
        "ckpt": '../exercise/results/checkpoints/cnn_transformer.pt',
    },
    "SpatialTokenViT": {
        "model": SpatialTokenViT(100),
        "loader": (train_loader_small, val_loader_small),
        "epochs": 50, "lr": 1e-4,
        "ckpt": '../exercise/results/checkpoints/spatial_vit.pt',
    },
    "ChannelTokenViT": {
        "model": ChannelTokenViT(100),
        "loader": (train_loader, val_loader),
        "epochs": 50, "lr": 3e-4,
        "ckpt": '../exercise/results/checkpoints/channel_vit.pt',
    },
}

for name, cfg in configs_p4.items():
    ckpt_path = cfg["ckpt"]
    history_path = f'../exercise/results/metrics/{name.lower()}_history.json'
    tr_loader, vl_loader = cfg["loader"]

    if TRAIN_MODE:
        print(f"\\n{'='*50}\\nTraining: {name}\\n{'='*50}")
        m = cfg["model"].to(DEVICE)
        hist = fit(m, tr_loader, vl_loader,
                   {"epochs": cfg["epochs"], "lr": cfg["lr"],
                    "device": DEVICE, "save_path": ckpt_path})
        histories_p4[name] = hist
        save_metrics_json(hist, history_path)
    else:
        histories_p4[name] = load_metrics_json(history_path)

    print(f"✓ {name} done")
"""),

    code("""# So sánh val_acc
plot_multi_curves(
    list(histories_p4.values()),
    list(histories_p4.keys()),
    title="So sánh 3 kiến trúc tokenization (Phần 4)",
    save_path='../exercise/results/plots/part4_comparison_curves.png'
)
"""),

    code("""# Bảng kết quả
results_p4 = {}
model_classes = {
    "CNNTransformerHybrid": CNNTransformerHybrid,
    "SpatialTokenViT": SpatialTokenViT,
    "ChannelTokenViT": ChannelTokenViT,
}
test_loaders = {
    "CNNTransformerHybrid": test_loader,
    "SpatialTokenViT": test_loader_small,
    "ChannelTokenViT": test_loader,
}

for name, cls in model_classes.items():
    ckpt = configs_p4[name]["ckpt"]
    if os.path.exists(ckpt):
        m = load_best_model(cls(100), ckpt, DEVICE)
        preds, labels = get_predictions(m, test_loaders[name], DEVICE)
        metrics = compute_metrics(preds, labels)
        results_p4[name] = {
            "test_acc": metrics["accuracy"],
            "val_acc": max(histories_p4[name]["val_acc"]),
            "f1_macro": metrics["f1_macro"],
            "params": get_param_count(m),
        }

save_metrics_json(results_p4, '../exercise/results/metrics/part4_results.json')
print_results_table(results_p4)

plot_comparison_bar(results_p4, metric="test_acc",
                    title="Test Accuracy — Phần 4",
                    save_path='../exercise/results/plots/part4_bar.png')
"""),

    md("""## 6. Nhận xét

**Phân tích:**

1. **CNNTransformerHybrid** thường đạt kết quả tốt nhất vì:
   - CNN cung cấp đặc trưng phân cấp (hierarchical features)
   - Transformer xử lý quan hệ toàn cục giữa các vùng feature
   - Số tokens vừa phải (64) → attention hiệu quả

2. **SpatialTokenViT** (1024 tokens) gặp vấn đề:
   - Attention matrix O(1024²) = 1M operations mỗi layer → chậm, tốn RAM
   - Với ảnh 32×32, mỗi token (pixel) chỉ có 3 features → feature quá thô
   - Đây là lý do ViT gốc dùng patches thay vì pixels!

3. **ChannelTokenViT** (64 tokens):
   - Tokens biểu diễn "đặc trưng kênh" thay vì "vùng không gian"
   - Có thể học được "channel attention" (channel nào quan trọng hơn)
   - Nhưng mất đi thông tin vị trí không gian

**Kết luận:** Hybrid CNN+Transformer thường cân bằng tốt nhất giữa
đặc trưng cục bộ (CNN) và quan hệ toàn cục (Transformer).
"""),
)


# ─── Notebook 5: Part 5 — LSTM/GRU ───────────────────────────────────────────

NB5 = nb(
    md("""# Phần 5 — Mô hình phân loại dựa trên LSTM/GRU

**Ý tưởng sáng tạo:** Ảnh không phải chuỗi thời gian,
nhưng ta có thể "đọc" ảnh theo nhiều cách:
- Đọc từng hàng (như đọc sách)
- Đọc từng cột
- Đọc từng patch (như quét QR code)

Sau đó dùng **LSTM** hoặc **GRU** xử lý chuỗi này để phân loại.
"""),

    code("""import sys, os
sys.path.insert(0, os.path.abspath('..'))
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

from src.data import get_cifar100_loaders, get_sequence_loaders, get_device
from src.models_part5 import ImageLSTM, ImageGRU
from src.train import fit, load_best_model
from src.utils import (get_param_count, get_predictions, compute_metrics,
                       plot_multi_curves, plot_comparison_bar, print_results_table,
                       save_metrics_json, load_metrics_json)

DEVICE = get_device()
print(f"Thiết bị: {DEVICE}")
"""),

    md("""## 1. Các cách biểu diễn ảnh thành chuỗi"""),

    code("""# Visualize 4 cách đọc ảnh
_, _, test_loader_img, class_names = get_cifar100_loaders(batch_size=8)
images, labels = next(iter(test_loader_img))
img = images[0]  # [3, 32, 32]
CIFAR100_MEAN = torch.tensor([0.5071, 0.4867, 0.4408])
CIFAR100_STD  = torch.tensor([0.2675, 0.2565, 0.2761])
img_show = (img * CIFAR100_STD[:,None,None] + CIFAR100_MEAN[:,None,None]).clamp(0,1)
img_np = img_show.permute(1,2,0).numpy()

fig, axes = plt.subplots(1, 4, figsize=(14, 4))

# Row-wise
axes[0].imshow(img_np)
for i in range(0, 32, 4):
    axes[0].axhline(y=i-0.5, color='yellow', linewidth=1, alpha=0.7)
arrow = mpatches.FancyArrowPatch((0, 0), (31, 0), arrowstyle='->', color='yellow', linewidth=2)
axes[0].add_patch(arrow)
axes[0].set_title(f'Row-wise\\nT=32, D=96\\n(32 hàng, mỗi hàng 3×32=96)')
axes[0].axis('off')

# Col-wise
axes[1].imshow(img_np)
for j in range(0, 32, 4):
    axes[1].axvline(x=j-0.5, color='cyan', linewidth=1, alpha=0.7)
axes[1].set_title(f'Col-wise\\nT=32, D=96\\n(32 cột, mỗi cột 3×32=96)')
axes[1].axis('off')

# Patch4
axes[2].imshow(img_np)
for i in range(8):
    for j in range(8):
        rect = mpatches.Rectangle((j*4-0.5, i*4-0.5), 4, 4,
                                    linewidth=1, edgecolor='red', facecolor='none')
        axes[2].add_patch(rect)
        if i == 0 and j < 3:
            axes[2].text(j*4+1.5, 1.5, f'{i*8+j+1}', fontsize=6, color='white', ha='center')
axes[2].set_title('Patch4 (4×4)\\nT=64, D=48\\n(64 patches, mỗi patch 3×4×4=48)')
axes[2].axis('off')

# Patch8
axes[3].imshow(img_np)
for i in range(4):
    for j in range(4):
        rect = mpatches.Rectangle((j*8-0.5, i*8-0.5), 8, 8,
                                    linewidth=2, edgecolor='green', facecolor='none')
        axes[3].add_patch(rect)
        axes[3].text(j*8+3.5, i*8+3.5, f'{i*4+j+1}', fontsize=9, color='white',
                    ha='center', va='center', fontweight='bold')
axes[3].set_title('Patch8 (8×8)\\nT=16, D=192\\n(16 patches, mỗi patch 3×8×8=192)')
axes[3].axis('off')

plt.suptitle('4 cách đọc ảnh CIFAR-100 thành chuỗi cho LSTM/GRU', fontsize=12)
plt.tight_layout()
plt.show()
"""),

    md("""## 2. So sánh LSTM và GRU

| | LSTM | GRU |
|--|------|-----|
| **Gates** | 4 (forget, input, output, cell) | 2 (reset, update) |
| **Cell state** | Có (c_t) — memory dài hạn | Không |
| **Parameters** | ~4×(D+H)×H mỗi layer | ~3×(D+H)×H mỗi layer |
| **Tốc độ** | Chậm hơn | Nhanh hơn ~25% |
| **Accuracy** | Thường tương đương | Thường tương đương |

Với D=96 (row), H=256, num_layers=2, bidirectional:
"""),

    code("""# So sánh số params
for seq_mode, input_size in [("row", 96), ("patch4", 48)]:
    lstm = ImageLSTM(input_size=input_size)
    gru = ImageGRU(input_size=input_size)
    print(f"seq_mode={seq_mode} (input_size={input_size}):")
    print(f"  LSTM params: {get_param_count(lstm)}")
    print(f"  GRU  params: {get_param_count(gru)}")
    print()
"""),

    md("""## 3. Tải dữ liệu dạng chuỗi"""),

    code("""# Tải 4 loaders khác nhau
loaders = {}
seq_configs = [
    ("LSTM-row",   "lstm",  "row",    96),
    ("LSTM-patch4","lstm",  "patch4", 48),
    ("GRU-row",    "gru",   "row",    96),
    ("GRU-patch4", "gru",   "patch4", 48),
]

for name, rnn_type, seq_mode, input_size in seq_configs:
    print(f"\\nLoading {name} ({seq_mode})...")
    tr, vl, te, seq_len, inp_size = get_sequence_loaders(seq_mode=seq_mode, batch_size=128)
    loaders[name] = {"train": tr, "val": vl, "test": te,
                     "seq_len": seq_len, "input_size": inp_size,
                     "rnn_type": rnn_type}
"""),

    md("""## 4. Huấn luyện"""),

    code("""TRAIN_MODE = True
histories_p5 = {}

for name, cfg in loaders.items():
    ckpt_path = f'../exercise/results/checkpoints/{name.lower()}.pt'
    history_path = f'../exercise/results/metrics/{name.lower()}_history.json'

    RNNCls = ImageLSTM if cfg["rnn_type"] == "lstm" else ImageGRU
    model = RNNCls(input_size=cfg["input_size"])

    if TRAIN_MODE:
        print(f"\\n{'='*50}\\nTraining: {name} (T={cfg['seq_len']}, D={cfg['input_size']})\\n{'='*50}")
        model = model.to(DEVICE)
        hist = fit(model, cfg["train"], cfg["val"],
                   {"epochs": 30, "lr": 1e-3, "device": DEVICE, "save_path": ckpt_path})
        histories_p5[name] = hist
        save_metrics_json(hist, history_path)
    else:
        histories_p5[name] = load_metrics_json(history_path)

    print(f"✓ {name} done")
"""),

    code("""# Training curves comparison
plot_multi_curves(
    list(histories_p5.values()),
    list(histories_p5.keys()),
    title="So sánh LSTM/GRU × Sequence Representation (Phần 5)",
    save_path='../exercise/results/plots/part5_comparison_curves.png'
)
"""),

    code("""# Bảng kết quả
results_p5 = {}
for name, cfg in loaders.items():
    ckpt_path = f'../exercise/results/checkpoints/{name.lower()}.pt'
    if os.path.exists(ckpt_path):
        RNNCls = ImageLSTM if cfg["rnn_type"] == "lstm" else ImageGRU
        m = load_best_model(RNNCls(cfg["input_size"]), ckpt_path, DEVICE)
        preds, labels = get_predictions(m, cfg["test"], DEVICE)
        metrics = compute_metrics(preds, labels)
        results_p5[name] = {
            "test_acc": metrics["accuracy"],
            "val_acc": max(histories_p5[name]["val_acc"]),
            "f1_macro": metrics["f1_macro"],
            "params": get_param_count(m),
        }

save_metrics_json(results_p5, '../exercise/results/metrics/part5_results.json')
print_results_table(results_p5)

plot_comparison_bar(results_p5, metric="test_acc",
                    title="Test Accuracy — Phần 5 (LSTM/GRU)",
                    save_path='../exercise/results/plots/part5_bar.png')
"""),

    md("""## 5. Grand Summary — Tất cả mô hình

So sánh toàn bộ ~12 mô hình từ 5 phần của bài tập.
"""),

    code("""# Gộp tất cả results
all_results = {}

def safe_load(path):
    return load_metrics_json(path) if os.path.exists(path) else {}

all_results.update(safe_load('../exercise/results/metrics/part1_2_results.json'))
all_results.update(safe_load('../exercise/results/metrics/part3_results.json'))
all_results.update(safe_load('../exercise/results/metrics/part4_results.json'))
all_results.update(safe_load('../exercise/results/metrics/part5_results.json'))

if all_results:
    print("\\n=== BẢNG TỔNG KẾT TẤT CẢ MÔ HÌNH ===")
    print_results_table(all_results)
    plot_comparison_bar(
        all_results, metric="test_acc",
        title="So sánh Tất cả Mô hình — CIFAR-100",
        save_path='../exercise/results/plots/grand_summary_bar.png'
    )
"""),

    md("""## 6. Nhận xét Phần 5 & Tổng kết

**LSTM/GRU so với CNN và ViT:**
- LSTM/GRU thường kém CNN vì ảnh có **cấu trúc không gian 2D**,
  không phải chuỗi 1D
- Đọc theo hàng/cột làm mất quan hệ dọc/ngang
- Patches nhỏ giữ được nhiều thông tin cục bộ hơn

**LSTM vs GRU:**
- Thường đạt accuracy tương đương
- GRU đơn giản hơn (ít params ~25%) → train nhanh hơn
- Không có quy tắc tuyệt đối: tùy bài toán

**Row-wise vs Patch-wise:**
- Patch-wise thường tốt hơn vì giữ được thông tin không gian cục bộ
- Patch 4×4 (T=64) cân bằng giữa số bước và độ phân giải

**Kết luận chung (rank dự kiến):**
1. 🥇 SimpleCNN (~55%)     — tận dụng spatial structure
2. 🥈 CNNTransformerHybrid (~52%) — CNN + global attention
3. 🥉 ChannelTokenViT (~45%) — channel attention
4. SimpleViT/CustomViT (~40%)
5. LSTM/GRU variants (~38-42%)
6. SpatialTokenViT (~35%)  — bottlenecked bởi chất lượng pixel features
7. MLP (~37%)
8. SoftmaxRegression (~17%) — tuyến tính, underfitting
"""),
)


# ─── Generate all notebooks ───────────────────────────────────────────────────

if __name__ == "__main__":
    out_dir = os.path.join(os.path.dirname(__file__), "notebooks")
    os.makedirs(out_dir, exist_ok=True)

    notebooks = {
        "part1_models.ipynb": NB1,
        "part2_train_compare.ipynb": NB2,
        "part3_custom_transformer.ipynb": NB3,
        "part4_architectures.ipynb": NB4,
        "part5_lstm_gru.ipynb": NB5,
    }

    for fname, notebook in notebooks.items():
        path = os.path.join(out_dir, fname)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(notebook, f, indent=1, ensure_ascii=False)
        print(f"✓ Created: {path}")

    print(f"\nAll {len(notebooks)} notebooks created in {out_dir}/")
