# Báo cáo Bài Tập
# Phân loại Ảnh với các Mô hình Học Sâu Cơ bản

| | |
|---|---|
| **Môn học** | CO5085 – Học sâu và ứng dụng trong thị giác máy tính |
| **Sinh viên** | Nguyễn Trung Phong – MSSV: 2570047 |
| **Giảng viên** | Lê Thành Sách |
| **Trường** | Đại học Bách Khoa TP.HCM (HCMUT) |
| **Học kỳ** | 2, năm học 2025–2026 |
| **Deadline** | 01/04/2026 |
| **Repository** | https://github.com/PhongNguyenTrung/hcmut-deeplearning-ass1/tree/main/exercise |

---

## Mục lục

1. [Giới thiệu](#1-giới-thiệu)
2. [Dataset](#2-dataset)
3. [Phương pháp](#3-phương-pháp)
4. [Thảo luận](#4-thảo-luận)
5. [Kết luận](#5-kết-luận)
6. [Tài liệu tham khảo](#6-tài-liệu-tham-khảo)

---

## 1. Giới thiệu

### 1.1 Mục tiêu bài toán

**Phân loại ảnh** (Image Classification) là bài toán gán nhãn cho một ảnh đầu vào — trả lời câu hỏi *"ảnh này thuộc lớp nào?"*. Đây là bài toán nền tảng và quan trọng nhất của Thị giác Máy tính (Computer Vision).

```
Đầu vào: ảnh [H × W × C]  →  Mô hình  →  Đầu ra: nhãn lớp (ví dụ: "mèo", "xe hơi")
```

Ứng dụng thực tế: nhận dạng khuôn mặt, chẩn đoán hình ảnh y tế, phân loại sản phẩm lỗi trong sản xuất, hệ thống xe tự lái nhận diện biển báo giao thông.

**Các mô hình được sử dụng trong bài tập này:**

| Mô hình | Phần | Kiến trúc |
|---------|------|-----------|
| Softmax Regression | 1 | Tuyến tính thuần tuý |
| MLP (Multi-Layer Perceptron) | 1 | Fully-connected |
| CNN (Convolutional Neural Network) | 1 | Tích chập cục bộ |
| Vision Transformer (ViT) — PyTorch | 1 | Self-Attention |
| CustomViT (Tự xây từ `einsum`) | 3 | Self-Attention |
| CNN + Transformer Hybrid | 4 | Hybrid |
| SpatialToken ViT | 4 | Attention trên pixel |
| ChannelToken ViT | 4 | Attention trên channel |
| LSTM / GRU | 5 | Recurrent |

Tất cả mô hình được huấn luyện **from scratch** (không pretrained weights) trên **CIFAR-100** với training loop tự viết bằng PyTorch.

### 1.2 Mục tiêu học tập

Bài tập này nhắm đến ba mục tiêu học tập chính:

1. **So sánh các kiến trúc từ đơn giản đến phức tạp:** Từ Softmax Regression tuyến tính → MLP → CNN → ViT → Hybrid, để thấy rõ sự tiến bộ và trade-off giữa các phương pháp qua từng giai đoạn lịch sử học sâu.

2. **Hiểu training loop:** Tự viết vòng lặp huấn luyện PyTorch — `zero_grad → forward → loss → backward → clip → step` — thay vì dùng API cấp cao, để hiểu cơ chế tối ưu hoá từ gốc.

3. **Hiểu cơ chế Transformer:** Tự hiện thực `MultiHeadAttention` và `TransformerEncoderLayer` từ `nn.Linear`, `LayerNorm`, `torch.einsum` — không dùng module có sẵn — để nắm vững toán học bên trong attention.

---

## 2. Dataset

### 2.1 Dataset sử dụng

Bài tập này sử dụng **CIFAR-100** (Canadian Institute For Advanced Research, 100 classes).

| Thuộc tính | Giá trị |
|------------|---------|
| Tên | CIFAR-100 |
| Số lớp (fine-grained) | 100 |
| Số superclass | 20 |
| Kích thước ảnh | 32 × 32 pixels, 3 kênh RGB |
| Tổng số ảnh | 60,000 |
| Train / Validation / Test | 45,000 / 5,000 / 10,000 |
| Ảnh mỗi lớp (train) | 450 |

### 2.2 Lý do chọn

CIFAR-100 được chọn thay vì CIFAR-10 (mặc định trong đề bài) để bài toán đủ khó nhằm phân biệt rõ ràng năng lực từng kiến trúc:

| Tiêu chí | CIFAR-10 | CIFAR-100 (được chọn) |
|----------|----------|----------------------|
| Số lớp | 10 | 100 |
| Ảnh mỗi lớp (train) | 5,000 | 500 |
| Độ khó | Thấp | Trung bình–cao |
| Softmax Regression đạt | ~70–75% | ~8% |
| Phân biệt kiến trúc? | Khó (tất cả đều cao) | Rõ ràng |

Với CIFAR-10, ngay cả Softmax Regression đạt ~70% nên không thể thấy sự khác biệt giữa các kiến trúc. CIFAR-100 với 100 lớp fine-grained (ví dụ: baby, boy, girl, man, woman — 5 lớp riêng biệt trong superclass "people") làm bài toán đủ thử thách.

### 2.3 Preprocessing

```python
# Chuẩn hoá với thống kê CIFAR-100 (KHÔNG dùng ImageNet stats)
mean = [0.5071, 0.4867, 0.4408]
std  = [0.2675, 0.2565, 0.2761]

# Tập train — có augmentation
train_transforms = transforms.Compose([
    transforms.RandomCrop(32, padding=4),   # dịch chuyển ảnh ±4 pixels
    transforms.RandomHorizontalFlip(),       # lật ngang với xác suất 50%
    transforms.ToTensor(),
    transforms.Normalize(mean, std),
])

# Tập validation / test — chỉ chuẩn hoá
eval_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean, std),
])
```

> **Tại sao dùng CIFAR-100 stats thay vì ImageNet stats?** CIFAR-100 (ảnh 32×32 vật thể) và ImageNet (ảnh tự nhiên lớn) có phân phối màu khác nhau. Dùng sai stats làm dữ liệu bị chuẩn hoá sai, ảnh hưởng đến tốc độ hội tụ và kết quả.

---

## 3. Phương pháp

**Quy ước chung:** Mọi mô hình nhận input `[B, 3, 32, 32]` và trả về logits `[B, 100]`. Không thêm softmax trong `forward()` vì `nn.CrossEntropyLoss` đã tích hợp `log_softmax` bên trong.

---

### 3.1 Phần 1 — Các mô hình cơ bản

#### 3.1.1 Softmax Regression

**Ý tưởng:** Mô hình tuyến tính đơn giản nhất. Ánh xạ trực tiếp từ raw pixels đến xác suất lớp qua một lớp tuyến tính duy nhất.

**Kiến trúc:**

```
Đầu vào: [B, 3, 32, 32]
    │
    ▼
Flatten → [B, 3072]          (3×32×32 = 3072 features)
    │
    ▼
Linear(3072 → 100) → [B, 100]   logits (không có activation)
```

**Code:**

```python
self.net = nn.Sequential(
    nn.Flatten(),                       # [B,3,32,32] → [B,3072]
    nn.Linear(3*32*32, num_classes),    # Ánh xạ tuyến tính
)
```

**Thông số:**

| Tham số | Giá trị |
|---------|---------|
| Input dim | 3 × 32 × 32 = 3,072 |
| Số tham số | 3,072 × 100 + 100 = **307,300** |
| Activation | Không (softmax tích hợp trong Loss) |
| Hạn chế cốt lõi | Chỉ học biên quyết định tuyến tính |

---

#### 3.1.2 MLP (Multi-Layer Perceptron)

**Ý tưởng:** Thêm các lớp ẩn với hàm kích hoạt phi tuyến ReLU để học được các pattern phức tạp hơn. BatchNorm và Dropout giúp ổn định training và giảm overfitting.

**Kiến trúc:**

```
Đầu vào: [B, 3, 32, 32]
    │
    ▼
Flatten → [B, 3072]
    │
    ▼
Linear(3072 → 512) → BatchNorm1d → ReLU → Dropout(0.3)   [Lớp ẩn 1]
    │
    ▼
Linear(512 → 256)  → BatchNorm1d → ReLU → Dropout(0.3)   [Lớp ẩn 2]
    │
    ▼
Linear(256 → 100) → [B, 100]
```

**Thông số:**

| Tham số | Giá trị |
|---------|---------|
| Lớp ẩn | 2 (512 → 256 neurons) |
| Activation | ReLU |
| Regularization | BatchNorm1d + Dropout(0.3) |
| Số tham số | ~**1.7M** |

**Hạn chế:** `Flatten` phá vỡ cấu trúc không gian 2D của ảnh. Pixel (i, j) và pixel (i, j+1) — hai pixel kề nhau theo chiều ngang — sau khi Flatten trở thành hai features độc lập, mạng không biết chúng ở cạnh nhau. CNN giải quyết vấn đề này.

---

#### 3.1.3 SimpleCNN (VGG-style)

**Ý tưởng:** Khai thác cấu trúc không gian 2D của ảnh thông qua tích chập cục bộ. Mỗi filter chỉ "nhìn" vùng 3×3, học đặc trưng từ vùng lân cận rồi truyền lên lớp cao hơn.

**Kiến trúc:**

```
Đầu vào: [B, 3, 32, 32]
    │
    ▼
ConvBlock1: [Conv2d(3→32,3×3)+BN+ReLU] × 2 → MaxPool(2)
    ▼ [B, 32, 16, 16]
ConvBlock2: [Conv2d(32→64,3×3)+BN+ReLU] × 2 → MaxPool(2)
    ▼ [B, 64, 8, 8]
ConvBlock3: [Conv2d(64→128,3×3)+BN+ReLU] × 2 → MaxPool(2)
    ▼ [B, 128, 4, 4]
AdaptiveAvgPool2d(1) → Flatten → [B, 128]
    │
    ▼
Linear(128 → 256) → ReLU → Dropout(0.3) → Linear(256 → 100) → [B, 100]
```

**Thông số:**

| Tham số | Giá trị |
|---------|---------|
| Conv blocks | 3 (channels: 3 → 32 → 64 → 128) |
| Kernel size | 3×3, padding=1 (giữ nguyên spatial size) |
| Pooling | MaxPool(2) — giảm spatial 1/2 mỗi block |
| AdaptiveAvgPool | Pool 128×4×4 → 128×1×1, tránh Flatten lớn |
| Số tham số | ~**346.6K** |

**Ba lý do CNN vượt trội MLP với dữ liệu ảnh:**

1. **Locality:** Mỗi filter chỉ xử lý vùng 3×3 — học đặc trưng cục bộ như cạnh, góc, texture
2. **Weight sharing:** Cùng bộ filter áp dụng tại mọi vị trí → giảm params, tổng quát hoá tốt hơn
3. **Hierarchical features:** Block 1 học edges, Block 2 học shapes, Block 3 học objects

---

#### 3.1.4 Vision Transformer (dùng PyTorch built-in)

**Ý tưởng:** Chia ảnh thành các "patches" nhỏ rồi xử lý như chuỗi tokens qua Transformer — tương tự cách xử lý câu trong NLP.

**Kiến trúc:**

```
Đầu vào: [B, 3, 32, 32]
    │
    ▼ Patch Embedding (Conv2d kernel=4, stride=4)
[B, 128, 8, 8] → flatten+transpose → [B, 64, 128]
    64 patches, mỗi patch 4×4 pixels, embed thành 128-dim vector
    │
    ▼ Prepend CLS token [B,1,128]
[B, 65, 128]
    │
    ▼ + Positional Encoding [1, 65, 128] (learnable)
    │
    ▼ TransformerEncoder: 4 layers, d_model=128, nhead=4, ffn=512, Pre-LN
    │
    ▼ Lấy CLS token [:, 0] → LayerNorm → Linear(128→100)
[B, 100]
```

**Thông số:**

| Tham số | Giá trị |
|---------|---------|
| Patch size | 4×4 (→ 64 patches trên ảnh 32×32) |
| d_model | 128 |
| Số attention heads | 4 (mỗi head: 32-dim) |
| Số Transformer layers | 4 |
| FFN dimension | 512 (= 4 × d_model) |
| Positional Encoding | Learnable (không dùng sinusoidal) |
| LayerNorm | Pre-LN (norm trước attention — ổn định hơn khi train from scratch) |
| Số tham số | ~**821.0K** |

> **Tại sao patch_size=4 không phải 16?** ViT-B/16 gốc dùng patch 16×16 cho ảnh 224×224 → 196 patches. Với ảnh 32×32: patch 16×16 → chỉ **4 patches** (quá ít, mất toàn bộ chi tiết). Patch 4×4 → **64 patches** — cân bằng giữa số lượng và độ phong phú mỗi patch.

---

### 3.2 Phần 2 — Training & Evaluation

#### 3.2.1 Training loop

Theo yêu cầu bài tập, **không dùng** Lightning `trainer.fit()` hay Keras-style API. Tự viết từng bước:

```python
def train_one_epoch(model, loader, optimizer, criterion, device, clip_norm=1.0):
    model.train()   # bật dropout, BatchNorm ở chế độ train
    total_loss, total_correct, total_samples = 0, 0, 0

    for x, y in loader:
        x, y = x.to(device), y.to(device)

        # ── Bước 1: Xoá gradient từ batch trước ──
        optimizer.zero_grad()

        # ── Bước 2: Forward pass ──
        logits = model(x)                    # [B, 100]

        # ── Bước 3: Tính loss ──
        loss = criterion(logits, y)          # CrossEntropyLoss

        # ── Bước 4: Backpropagation ──
        loss.backward()                      # tính gradient cho mọi params

        # ── Bước 5: Gradient clipping ──
        nn.utils.clip_grad_norm_(model.parameters(), clip_norm)  # max_norm=1.0

        # ── Bước 6: Cập nhật tham số ──
        optimizer.step()

        total_loss    += loss.item() * x.size(0)
        total_correct += (logits.argmax(1) == y).sum().item()
        total_samples += x.size(0)

    return total_loss / total_samples, total_correct / total_samples
```

**Giải thích Gradient Clipping:** Nếu norm vector gradient vượt quá `max_norm=1.0`, scale toàn bộ gradient xuống sao cho norm = 1.0. Kỹ thuật này ngăn "exploding gradients" — đặc biệt quan trọng với RNN (Phần 5) và Transformer (Phần 3, 4).

**Vòng lặp ngoài (`fit`):**

```python
def fit(model, train_loader, val_loader, config):
    optimizer = AdamW(model.parameters(), lr=config["lr"], weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=config["epochs"], eta_min=config["lr"]*0.01)

    best_val_acc, history = 0.0, defaultdict(list)
    for epoch in range(config["epochs"]):
        train_loss, train_acc = train_one_epoch(model, train_loader, ...)
        val_loss,   val_acc   = evaluate(model, val_loader, ...)
        scheduler.step()

        if val_acc > best_val_acc:         # Lưu model tốt nhất theo val accuracy
            best_val_acc = val_acc
            torch.save(model.state_dict(), config["save_path"])
    return history
```

#### 3.2.2 Metrics

- **Accuracy:** % ảnh phân loại đúng trên tập test
- **F1-macro:** Trung bình F1 của 100 lớp — đánh giá đều trên tất cả lớp, không bị lớp lớn lấn át
- **Loss:** CrossEntropyLoss — đo độ tin cậy của xác suất dự đoán

#### 3.2.3 Cấu hình Hyperparameters

| Mô hình | LR | Batch Size | Epochs | Optimizer | Scheduler |
|---------|-----|-----------|--------|-----------|-----------|
| SoftmaxRegression | 0.1 | 256 | 30 | AdamW | CosineAnnealing |
| MLP | 1e-3 | 128 | 50 | AdamW | CosineAnnealing |
| SimpleCNN | 1e-3 | 128 | 50 | AdamW | CosineAnnealing |
| SimpleViT | 3e-4 | 128 | 100 | AdamW | CosineAnnealing |

- **AdamW** (weight_decay=1e-4): L2 regularization tích hợp trong optimizer — tốt hơn Adam thuần tuý cho Transformer
- **CosineAnnealingLR**: LR giảm theo đường cosine từ `lr` → `lr×0.01`. Giúp mô hình "fine-tune" ở giai đoạn cuối thay vì dao động quanh cực tiểu

#### 3.2.4 Kết quả so sánh

![Softmax Regression — training curves](../results/plots/softmax_curves.png)

*Hình 1. Softmax Regression: val accuracy dao động quanh 8% từ rất sớm và không cải thiện — biểu hiện điển hình của underfitting.*

![MLP — training curves](../results/plots/mlp_curves.png)

*Hình 2. MLP: val accuracy tăng đều đến ~epoch 30 rồi plateau tại 20.7%. BatchNorm+Dropout kiểm soát tốt overfitting.*

![SimpleCNN — training curves](../results/plots/cnn_curves.png)

*Hình 3. SimpleCNN: đường cong đẹp nhất — tăng liên tục, hội tụ ổn định tại ~36%.*

![SimpleViT — training curves](../results/plots/vit_curves.png)

*Hình 4. SimpleViT: hội tụ chậm hơn CNN, vẫn tăng nhẹ ở epoch 100 — ViT chưa bão hoà.*

![Part 1+2 bar chart](../results/plots/part1_2_bar.png)

*Hình 5. So sánh Test Accuracy 4 mô hình Phần 1+2. SimpleCNN dẫn đầu với 35.88%, gấp hơn 4× Softmax Regression.*

| Mô hình | Test Acc | Val Acc | F1-macro | Params | Acc/Param |
|---------|----------|---------|----------|--------|-----------|
| SoftmaxRegression | 7.94% | 8.34% | 6.09% | 307.3K | 25.8%/M |
| MLP | 20.63% | 20.68% | 18.31% | 1.7M | 12.1%/M |
| SimpleViT | 24.47% | 25.54% | 22.35% | 821.0K | 29.8%/M |
| **SimpleCNN** | **35.88%** | **36.38%** | **34.48%** | **346.6K** | **103.5%/M** |

#### 3.2.5 Nhận xét

**Softmax Regression (7.94%):** Chỉ học được biên quyết định tuyến tính trong không gian 3072-chiều — chỉ tốt hơn random (1%) khoảng 8 lần. F1-macro (6.09%) thấp hơn accuracy (7.94%) cho thấy mô hình tập trung vào vài lớp dễ, bỏ qua phần lớn lớp còn lại.

**MLP (20.63%):** Các lớp ẩn với ReLU học được đặc trưng phi tuyến, cải thiện 2.6× so với Softmax. Tuy nhiên `Flatten` phá vỡ quan hệ không gian 2D: hai pixels kề nhau bị xử lý như hai features độc lập.

**SimpleCNN (35.88%) — tốt nhất Phần 1:** CNN (346.6K params) vượt MLP (1.7M params) — bằng chứng **inductive bias phù hợp quan trọng hơn số tham số**. Acc/Param = 103.5%/M — hiệu quả tốt nhất trong 4 mô hình.

**SimpleViT (24.47%):** Thấp hơn CNN vì thiếu spatial inductive bias — phải học mọi quan hệ không gian từ data. Với 45K mẫu (450/lớp), Transformer chưa đủ data để vượt CNN. Val accuracy vẫn tăng ở epoch 100 → cần nhiều epochs hơn.

---

### 3.3 Phần 3 — Transformer tự xây

#### 3.3.1 Transformer Encoder (custom)

Hiện thực TransformerEncoder **chỉ từ**: `nn.Linear`, `nn.LayerNorm`, `torch.einsum`, `F.softmax`, `F.dropout`. **Không được dùng**: `nn.MultiheadAttention`, `nn.TransformerEncoderLayer`, `nn.TransformerEncoder`.

**Lý thuyết Scaled Dot-Product Attention:**

```
Q = X · W_q   ("Tôi đang tìm kiếm gì?")
K = X · W_k   ("Tôi có thể cung cấp gì?")
V = X · W_v   ("Thông tin thực sự của tôi")

Attention(Q,K,V) = softmax(Q·Kᵀ / √d_head) · V
```

Chia `d_model=128` thành `H=4` heads, mỗi head có `d_head=32`. Mỗi head học một loại quan hệ khác nhau giữa các patches.

**Multi-Head Attention bằng `torch.einsum`:**

```python
class CustomMultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1):
        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model)
        self.d_head = d_model // num_heads

    def forward(self, x):  # x: [B, T, d_model]
        B, T, _ = x.shape
        H, d_h = self.num_heads, self.d_head

        # Linear projection + reshape thành multi-head
        Q = self.W_q(x).reshape(B,T,H,d_h).transpose(1,2)  # [B, H, T, d_h]
        K = self.W_k(x).reshape(B,T,H,d_h).transpose(1,2)
        V = self.W_v(x).reshape(B,T,H,d_h).transpose(1,2)

        # Scaled dot-product: 'bhid,bhjd->bhij'
        scores = torch.einsum('bhid,bhjd->bhij', Q, K) / math.sqrt(d_h)
        attn   = F.softmax(scores, dim=-1)      # [B, H, T, T]

        # Weighted sum of values: 'bhij,bhjd->bhid'
        out = torch.einsum('bhij,bhjd->bhid', attn, V)
        out = out.transpose(1,2).reshape(B, T, self.d_model)
        return self.W_o(out)
```

**Tại sao dùng `einsum`?** `torch.einsum('bhid,bhjd->bhij', Q, K)` tính dot-product song song trên toàn bộ (batch, head) chỉ trong một lệnh. Hiệu quả hơn viết vòng for qua từng head, và tránh lỗi broadcast thủ công.

#### 3.3.2 Pre-LN TransformerEncoderLayer

```python
class CustomTransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, ffn_dim=None, dropout=0.1):
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.attn  = CustomMultiHeadAttention(d_model, num_heads, dropout)
        self.ffn   = nn.Sequential(
            nn.Linear(d_model, ffn_dim),    # Expand: 128 → 512
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, d_model),    # Contract: 512 → 128
        )

    def forward(self, x):
        # Pre-LN: chuẩn hoá TRƯỚC khi đưa vào attention/FFN
        x = x + self.attn(self.norm1(x))   # residual connection
        x = x + self.ffn(self.norm2(x))    # residual connection
        return x
```

**Pre-LN vs Post-LN:**

| Variant | Công thức | Gradient | Yêu cầu warmup |
|---------|-----------|----------|----------------|
| Post-LN (Vaswani 2017) | `x = LN(x + Attn(x))` | Không ổn định lúc đầu | Có |
| **Pre-LN (được chọn)** | `x = x + Attn(LN(x))` | Ổn định từ đầu | Không bắt buộc |

Pre-LN phù hợp hơn khi train from scratch không có warmup schedule.

#### 3.3.3 ViT tự xây

Giống hệt SimpleViT (Phần 1) — cùng patch embedding, CLS token, positional encoding, classification head — **chỉ thay** `nn.TransformerEncoder` bằng `CustomTransformerEncoder` (stack của `CustomTransformerEncoderLayer`).

#### 3.3.4 So sánh

![Part 3 comparison curves](../results/plots/part3_comparison_curves.png)

*Hình 6. Training curves của SimpleViT (PyTorch) và CustomViT (tự xây) cùng cấu hình. Hai đường curve có hình dạng tương tự — bằng chứng Custom encoder hoạt động đúng.*

![Part 3 bar chart](../results/plots/part3_bar.png)

*Hình 7. CustomViT (34.56%) cao hơn PyTorch ViT (31.92%) ~2.6% — nằm trong biên độ ngẫu nhiên do khởi tạo weights khác nhau.*

| Mô hình | Encoder | Test Acc | Val Acc | F1-macro | Params | Thời gian/epoch |
|---------|---------|----------|---------|----------|--------|-----------------|
| SimpleViT (PyTorch) | `nn.TransformerEncoder` | 31.92% | 33.00% | 30.39% | 821.0K | ~45s (MPS) |
| **CustomViT (Tự xây)** | Custom einsum | **34.56%** | **35.80%** | **33.11%** | 819.4K | ~48s (MPS) |

**Nhận xét:**
- Training curves tương tự → xác nhận Custom MHA hoạt động đúng về toán học
- CustomViT cao hơn 2.64%: nằm trong biên độ variance ngẫu nhiên — **kết luận: hai mô hình tương đương nhau**
- Số tham số gần giống (821.0K vs 819.4K): sai số nhỏ do custom impl bỏ một vài bias terms

---

### 3.4 Phần 4 — Kiến trúc nâng cao

Ba cách tokenize ảnh khác nhau được triển khai và so sánh.

#### 3.4.1 CNN + Transformer Hybrid

**Ý tưởng:** Dùng CNN làm backbone trích đặc trưng tạo tokens có chất lượng cao, sau đó đưa vào Transformer để học quan hệ toàn cục.

**Kiến trúc:**

```
Đầu vào: [B, 3, 32, 32]
    │
    ▼ CNN Backbone
ConvBlock1: Conv(3→32)+BN+ReLU+Pool → [B, 32, 16, 16]
ConvBlock2: Conv(32→64)+BN+ReLU+Pool → [B, 64, 8, 8]
    │
    ▼ Tokenize CNN features
Reshape → [B, 64, 64]   (64 spatial positions, mỗi vị trí là 64-dim CNN feature)
Linear(64→128) → [B, 64, 128]   64 high-quality tokens
    │
    ▼ Transformer
Prepend CLS token → [B, 65, 128]
+ Positional Encoding
2 × TransformerLayer(d=128, heads=4, ffn=512)
    │
    ▼ CLS token → Linear(128→100) → [B, 100]

Params: 446.1K
```

**Tại sao hiệu quả:** CNN xử lý đặc trưng cục bộ (edges, textures) → tạo ra 64 tokens giàu thông tin. Transformer học quan hệ toàn cục giữa các đặc trưng này. Kết hợp được cả hai thế mạnh: spatial inductive bias (CNN) và global attention (Transformer).

#### 3.4.2 Các cách tokenize và embed ảnh khác nhau

**SpatialToken ViT — coi H×W positions làm tokens:**

```
Đầu vào: [B, 3, 32, 32]
    │
    ▼ Spatial tokenize
Reshape → [B, 1024, 3]    (1024 = 32×32 pixel positions, mỗi token là RGB vector)
Linear(3→64) → [B, 1024, 64]
    │
    ▼ Transformer
+ Positional Encoding
2 × TransformerLayer(d=64, heads=4)
    │
    ▼ Global Average Pool → Linear(64→100) → [B, 100]

Params: 172.4K
⚠️ Attention matrix: [B, 4, 1024, 1024] → ~512MB RAM → batch_size=32
```

**ChannelToken ViT — coi C channels làm tokens:**

```
Đầu vào: [B, 3, 32, 32]
    │
    ▼ Expand channels
Conv2d(3→64, 1×1) + BN + ReLU → [B, 64, 32, 32]
    │
    ▼ Channel tokenize
Reshape → [B, 64, 1024]    (64 channels, mỗi channel là 1024 spatial values)
Linear(1024→128) → [B, 64, 128]   64 channel tokens
    │
    ▼ Transformer
+ Positional Encoding
2 × TransformerLayer(d=128, heads=4)
    │
    ▼ Global Average Pool → Linear(128→100) → [B, 100]

Params: 549.5K
```

#### 3.4.3 So sánh

![Part 4 comparison curves](../results/plots/part4_comparison_curves.png)

*Hình 8. CNN+Transformer hội tụ tốt và ổn định. SpatialToken và ChannelToken plateau sớm ở mức thấp — dấu hiệu token quality kém.*

![Part 4 bar chart](../results/plots/part4_bar.png)

*Hình 9. Khoảng cách rất lớn giữa CNN+Transformer (37.25%) và hai kiến trúc còn lại chứng minh token quality quyết định hiệu năng Transformer.*

| Mô hình | Cách tokenize | Seq Len | Token Dim | Test Acc | Val Acc | F1-macro | Params |
|---------|--------------|---------|-----------|----------|---------|----------|--------|
| **CNN+Transformer** | CNN features | 64 | 128 | **37.25%** | **38.90%** | **35.87%** | 446.1K |
| ChannelToken ViT | C channels | 64 | 128 | 17.56% | 18.18% | 15.29% | 549.5K |
| SpatialToken ViT | H×W pixels | 1,024 | 64 | 13.10% | 13.12% | 9.86% | 172.4K |

**Nhận xét:**
- **CNN+Transformer (37.25%) — tốt nhất toàn bài:** Attention cost O(64²) = 4,096 — hiệu quả. CNN backbone tạo token chất lượng cao, Transformer chỉ cần học quan hệ toàn cục.
- **SpatialToken ViT (13.10%) — thất bại:** 1,024 tokens từ raw pixels — mỗi token chỉ có 3 giá trị RGB, thông tin cực kỳ thô. Đây chính xác là lý do ViT gốc dùng patches (patch 4×4 có 48 features, phong phú hơn 16× so với 3 RGB).
- **Kết luận: Token quality > Token quantity.** SpatialToken có 16× nhiều tokens nhưng kết quả tệ hơn 2.8×.

---

### 3.5 Phần 5 — LSTM/GRU

#### 3.5.1 Biểu diễn ảnh thành chuỗi

Ảnh không phải chuỗi thời gian tự nhiên. Bài tập này thử nghiệm 4 cách "đọc" ảnh thành sequence:

| Seq Mode | Seq Length (T) | Input Size (D) | Cách biểu diễn |
|----------|----------------|----------------|----------------|
| **Row-wise** | 32 | 96 = 32×3 | Mỗi hàng pixels |
| Col-wise | 32 | 96 = 32×3 | Mỗi cột pixels |
| **Patch4** | 64 = 8×8 | 48 = 4×4×3 | 64 patches 4×4 |
| Patch8 | 16 = 4×4 | 192 = 8×8×3 | 16 patches 8×8 |

#### 3.5.2 Kiến trúc

```
Đầu vào: [B, T, input_size]     (T và input_size tuỳ theo seq_mode)
    │
    ▼
BiLSTM / BiGRU (hidden=256, num_layers=2, bidirectional=True, dropout=0.3)
    Tạo 2 luồng: forward (bước 0 → T-1) + backward (bước T-1 → 0)
    │
    ▼
concat(h_n[-2], h_n[-1]) → [B, 512]
    │
    ▼
Dropout(0.3) → Linear(512 → 100) → [B, 100]
```

**Bidirectional:** Mỗi timestep "nhìn" được context từ cả hai phía — giúp pixel ở giữa "biết" về cả pixels bên trái lẫn bên phải.

**So sánh LSTM vs GRU:**

| Đặc điểm | LSTM | GRU |
|----------|------|-----|
| Số gates | 4: forget (f), input (i), output (o), cell (g) | 2: reset (r), update (z) |
| Cell state | Có (c_t — long-term memory riêng biệt) | Không (tích hợp trong h_t) |
| Số params/layer | 4 × (D+H) × H + 4H | 3 × (D+H) × H + 3H |
| Phù hợp | Chuỗi rất dài (>1000 steps) | Chuỗi vừa (≤100 steps) |

#### 3.5.3 So sánh

![Part 5 comparison curves](../results/plots/part5_comparison_curves.png)

*Hình 10. GRU hội tụ nhanh hơn và đạt val accuracy cao hơn LSTM. LSTM có khoảng cách train/val lớn hơn → overfitting nhiều hơn.*

![Part 5 bar chart](../results/plots/part5_bar.png)

*Hình 11. GRU-row đạt 36.57% — ngang bằng SimpleCNN (35.88%), một kết quả bất ngờ.*

| Mô hình | Seq Mode | T | D | Test Acc | Val Acc | F1-macro | Params |
|---------|----------|---|---|----------|---------|----------|--------|
| **GRU** | **Row-wise** | 32 | 96 | **36.57%** | **36.44%** | **35.68%** | 1.8M |
| GRU | Patch 4×4 | 64 | 48 | 35.64% | 35.76% | 34.99% | 1.7M |
| LSTM | Row-wise | 32 | 96 | 29.36% | 29.62% | 28.34% | 2.4M |
| LSTM | Patch 4×4 | 64 | 48 | 27.73% | 27.78% | 26.41% | 2.3M |

#### 3.5.4 Nhận xét

**GRU vượt LSTM +7.2% (row) và +7.9% (patch4):** Đây là kết quả không trực quan — LSTM có cơ chế phức tạp hơn nhưng lại tệ hơn. Giải thích: chuỗi ảnh CIFAR ngắn (T=32 hoặc T=64) không cần cell state riêng biệt của LSTM. GRU ít params hơn → ít overfitting hơn trong 30 epochs.

**GRU-row (36.57%) cạnh tranh ngang SimpleCNN (35.88%):** Tuy nhiên cần 1.8M params để đạt kết quả tương đương CNN với chỉ 346.6K params — **kém hiệu quả tham số hơn 5×**.

**RNN kém CNN về cơ bản:** Ảnh có cấu trúc **2D** — quan hệ dọc giữa các hàng quan trọng không kém quan hệ ngang trong cùng hàng. Đọc row-wise bỏ qua quan hệ dọc này — hạn chế cốt lõi của RNN với dữ liệu 2D.

---

## 4. Thảo luận

### 4.1 Bảng xếp hạng toàn bộ 12 mô hình

| # | Mô hình | Phần | Kiến trúc | Test Acc | F1-macro | Params | Acc/Param |
|---|---------|------|-----------|----------|----------|--------|-----------|
| 1 | **CNN+Transformer Hybrid** | 4 | Hybrid | **37.25%** | 35.87% | 446.1K | 83.5%/M |
| 2 | GRU-row | 5 | RNN | 36.57% | 35.68% | 1.8M | 20.3%/M |
| 3 | **SimpleCNN** | 1 | CNN | 35.88% | 34.48% | **346.6K** | **103.5%/M** |
| 4 | GRU-patch4 | 5 | RNN | 35.64% | 34.99% | 1.7M | 21.0%/M |
| 5 | CustomViT (Tự xây) | 3 | Transformer | 34.56% | 33.11% | 819.4K | 42.2%/M |
| 6 | SimpleViT (PyTorch) | 3 | Transformer | 31.92% | 30.39% | 821.0K | 38.9%/M |
| 7 | LSTM-row | 5 | RNN | 29.36% | 28.34% | 2.4M | 12.2%/M |
| 8 | LSTM-patch4 | 5 | RNN | 27.73% | 26.41% | 2.3M | 12.1%/M |
| 9 | SimpleViT (Phần 1) | 1 | Transformer | 24.47% | 22.35% | 821.0K | 29.8%/M |
| 10 | MLP | 1 | FC | 20.63% | 18.31% | 1.7M | 12.1%/M |
| 11 | ChannelToken ViT | 4 | Transformer | 17.56% | 15.29% | 549.5K | 32.0%/M |
| 12 | SpatialToken ViT | 4 | Transformer | 13.10% | 9.86% | 172.4K | 76.0%/M |
| 13 | SoftmaxRegression | 1 | Linear | 7.94% | 6.09% | 307.3K | 25.8%/M |

> **Acc/Param** = Test Accuracy / (số triệu tham số) — đo lường hiệu quả sử dụng tham số. SimpleCNN đạt 103.5%/M — tốt nhất trong các mô hình học được tốt.

### 4.2 CNN vs ViT

| Khía cạnh | CNN | ViT |
|-----------|-----|-----|
| Inductive bias | Locality + weight sharing | Không có — học từ data |
| Data requirement | Thấp (hiệu quả với vài chục nghìn ảnh) | Cao (cần hàng triệu ảnh pretrain) |
| Accuracy (bài này) | SimpleCNN: 35.88% | SimpleViT: 24.47% (Phần 1) / 31.92% (Phần 3) |
| Efficiency | 103.5%/M | 29.8%–42.2%/M |
| Receptive field | Tăng dần theo số layers | Global từ layer 1 |
| Phù hợp | Small datasets, edge deployment | Large-scale pretraining |

**Kết luận:** Với CIFAR-100 (45K mẫu, from scratch), CNN chiếm ưu thế rõ ràng nhờ inductive bias phù hợp. ViT thực sự vượt CNN khi được pretrained trên ImageNet-21K (14M ảnh) — không phải kiến trúc kém, mà thiếu data.

### 4.3 Transformer vs RNN

| Khía cạnh | Transformer | RNN (LSTM/GRU) |
|-----------|-------------|----------------|
| Cơ chế | Self-Attention (toàn cục, song song) | Recurrence (tuần tự) |
| Long-range dependency | O(1) — mọi cặp token tương tác trực tiếp | O(T) — thông tin lan truyền qua time steps |
| Parallelism | Cao (training nhanh) | Thấp (phải tính tuần tự) |
| Memory | O(T²) attention matrix | O(T) hidden states |
| Với ảnh CIFAR (bài này) | ViT: 24–34% | GRU: 35–36%, LSTM: 27–29% |

**Kết quả bất ngờ:** GRU (~36%) cạnh tranh với CNN và vượt ViT from scratch. Điều này do: (1) BiGRU học patterns hàng ngang hiệu quả, (2) chuỗi ngắn T=32 không cần attention toàn cục.

### 4.4 Trade-off: Accuracy vs tốc độ và data requirement

| Kiến trúc | Accuracy | Training Time/epoch | Data Requirement | Phù hợp |
|-----------|----------|---------------------|------------------|---------|
| CNN | ~36% | ~15s | Thấp | Small dataset, production |
| CNN+Transformer | ~37% | ~20s | Trung bình | Cần global context |
| GRU | ~36% | ~25s | Trung bình | Sequential patterns |
| ViT (from scratch) | ~31% | ~45s | Cao | Pretrain available |
| LSTM | ~28% | ~30s | Trung bình | Long sequences |
| Softmax/MLP | 8–21% | ~5s | Thấp | Baseline only |

---

## 5. Kết luận

### 5.1 Mô hình tốt nhất

**CNN+Transformer Hybrid** đạt test accuracy cao nhất (37.25%), kết hợp spatial inductive bias từ CNN và global attention từ Transformer.

**SimpleCNN** là lựa chọn tốt nhất xét về **hiệu quả tham số** (103.5%/M): chỉ 346.6K params, huấn luyện nhanh, không cần tuning phức tạp.

### 5.2 Insight chính

1. **Inductive bias phù hợp quan trọng hơn model size:** SimpleCNN (346.6K) vượt MLP (1.7M) và LSTM (2.4M). Kiến trúc phù hợp với cấu trúc dữ liệu hiệu quả hơn chỉ tăng số tham số.

2. **ViT cần data lớn để hội tụ tốt:** Với 45K mẫu from scratch, ViT (24–34%) thua CNN (36–37%). ViT thực sự vượt CNN khi được pretrained trên tập lớn — không phải kiến trúc kém, mà thiếu data.

3. **Hybrid kết hợp được hai thế mạnh:** CNN+Transformer đạt 37.25% — tốt nhất toàn bài — bằng cách dùng CNN tạo tokens chất lượng cao rồi Transformer học quan hệ toàn cục.

4. **Token quality > Token quantity:** SpatialToken ViT (1,024 raw pixel tokens → 13.10%) thua CNN features (64 high-quality tokens → 37.25%). 16× nhiều tokens nhưng kết quả tệ hơn 2.8×.

5. **GRU ≥ LSTM cho chuỗi ngắn:** Ưu thế của LSTM rõ hơn ở T>1,000 steps. Với T=32–64 (ảnh CIFAR), GRU ít tham số hơn → ít overfitting → kết quả tốt hơn.

### 5.3 Hướng cải thiện tiếp theo

- **Data augmentation mạnh hơn:** CutMix, MixUp, RandAugment → dự kiến +5–10% accuracy
- **LR warmup cho ViT:** Linear warmup 10 epochs trước CosineAnnealing → ổn định hơn giai đoạn đầu
- **Deeper CNN backbone:** Residual connections (ResNet-style) → target >50% test accuracy
- **Pre-trained features:** Dùng CLIP/DINO features thay vì train from scratch → LSTM/ViT tốt hơn nhiều
- **Label smoothing (ε=0.1):** Regularize CrossEntropyLoss cho 100 lớp, giảm overconfidence

---

## 6. Tài liệu tham khảo

1. Dosovitskiy, A. et al. (2021). *An Image is Worth 16×16 Words: Transformers for Image Recognition at Scale*. ICLR 2021.
2. Vaswani, A. et al. (2017). *Attention Is All You Need*. NeurIPS 2017.
3. He, K. et al. (2016). *Deep Residual Learning for Image Recognition*. CVPR 2016.
4. Simonyan, K. & Zisserman, A. (2015). *Very Deep Convolutional Networks for Large-Scale Image Recognition (VGGNet)*. ICLR 2015.
5. Cho, K. et al. (2014). *Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation*. EMNLP 2014.
6. Chung, J. et al. (2014). *Empirical Evaluation of Gated Recurrent Neural Networks on Sequence Modeling*. arXiv 2014.
7. Krizhevsky, A. (2009). *Learning Multiple Layers of Features from Tiny Images*. Technical Report, University of Toronto.
8. Touvron, H. et al. (2021). *Training data-efficient image transformers & distillation through attention (DeiT)*. ICML 2021.
9. Ba, J. et al. (2016). *Layer Normalization*. arXiv 2016.

---

*Báo cáo hoàn thành ngày 31/03/2026 | CO5085 – Học sâu và ứng dụng trong thị giác máy tính | HCMUT*
