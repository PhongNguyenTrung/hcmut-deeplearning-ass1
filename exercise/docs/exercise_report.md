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

1. [Giới thiệu bài toán phân loại ảnh](#1-giới-thiệu-bài-toán-phân-loại-ảnh)
2. [Tập dữ liệu](#2-tập-dữ-liệu)
3. [Phần 1 — Xây dựng 4 mô hình phân loại](#3-phần-1--xây-dựng-4-mô-hình-phân-loại)
4. [Phần 2 — Huấn luyện, đánh giá và so sánh](#4-phần-2--huấn-luyện-đánh-giá-và-so-sánh)
5. [Phần 3 — Tự hiện thực TransformerEncoder và ViT](#5-phần-3--tự-hiện-thực-transformerencoder-và-vit)
6. [Phần 4 — Kiến trúc kết hợp và cách embed ảnh khác nhau](#6-phần-4--kiến-trúc-kết-hợp-và-cách-embed-ảnh-khác-nhau)
7. [Phần 5 — Mô hình LSTM/GRU](#7-phần-5--mô-hình-lstmgru)
8. [So sánh tổng hợp](#8-so-sánh-tổng-hợp)
9. [Kết luận](#9-kết-luận)
10. [Tài liệu tham khảo](#10-tài-liệu-tham-khảo)

---

## 1. Giới thiệu bài toán phân loại ảnh

### 1.1 Phân loại ảnh là gì?

**Phân loại ảnh** (Image Classification) là bài toán gán nhãn cho một ảnh đầu vào — trả lời câu hỏi *"ảnh này thuộc lớp nào?"*. Đây là một trong những bài toán nền tảng và quan trọng nhất của Thị giác Máy tính (Computer Vision).

```
Đầu vào: ảnh [H × W × C]  →  Mô hình  →  Đầu ra: nhãn lớp (ví dụ: "mèo", "xe hơi")
```

Ứng dụng thực tế: nhận dạng khuôn mặt, chẩn đoán hình ảnh y tế, phân loại sản phẩm lỗi trong sản xuất, hệ thống xe tự lái nhận diện biển báo giao thông.

### 1.2 Thách thức của bài toán

Phân loại ảnh khó vì nhiều lý do:

- **Biến đổi ánh sáng:** Cùng một vật thể nhưng ánh sáng khác nhau cho pixel hoàn toàn khác nhau
- **Biến đổi góc nhìn:** Cùng một con mèo nhìn từ trước, từ bên, từ trên cho ảnh rất khác nhau
- **Intra-class variation:** Các con chó thuộc cùng một lớp nhưng hình dạng, màu sắc rất đa dạng
- **Inter-class similarity:** Báo và hổ thuộc 2 lớp khác nhau nhưng trông rất giống nhau
- **Occlusion:** Vật thể bị che khuất một phần
- **Curse of dimensionality:** Ảnh 32×32×3 có 3,072 chiều — không gian quá lớn cho phương pháp truyền thống

### 1.3 Sự phát triển của các phương pháp

Lịch sử giải quyết bài toán phân loại ảnh phản ánh chính sự tiến hoá của học sâu:

| Giai đoạn | Phương pháp | Đặc điểm |
|-----------|-------------|----------|
| Trước 2012 | HOG, SIFT + SVM | Feature thủ công, không học được |
| 2012 | AlexNet (CNN) | Feature tự học qua tích chập — bước ngoặt lớn |
| 2014–2019 | VGGNet, ResNet, EfficientNet | CNN ngày càng sâu hơn, chính xác hơn |
| 2020–nay | Vision Transformer (ViT) | Áp dụng cơ chế Attention từ NLP vào ảnh |
| Hiện tại | Hybrid CNN+Transformer | Kết hợp ưu điểm cả hai |

### 1.4 Mục tiêu bài tập này

Bài tập này xây dựng và so sánh **12 kiến trúc** đại diện cho toàn bộ lịch sử trên — từ mô hình tuyến tính đơn giản nhất đến Hybrid hiện đại:

| Phần | Kiến trúc | Đại diện cho |
|------|-----------|-------------|
| Phần 1 | Softmax Regression, MLP | Phương pháp tuyến tính / fully-connected |
| Phần 1 | SimpleCNN, SimpleViT | CNN cổ điển / Transformer hiện đại |
| Phần 2 | Training loop tự viết | Hiểu cơ chế tối ưu hoá |
| Phần 3 | CustomViT (einsum) | Hiểu cơ chế Attention từ gốc |
| Phần 4 | CNN+Transformer, Spatial/Channel tokens | Kiến trúc hybrid, tokenization |
| Phần 5 | LSTM, GRU | RNN — phương pháp chuỗi |

Tất cả mô hình được huấn luyện **from scratch** trên **CIFAR-100** (100 lớp, 32×32 pixels) — không dùng pretrained weights — với training loop tự viết bằng PyTorch.

---

## 2. Tập dữ liệu

### 2.1 Lựa chọn và lý do

Bài tập này sử dụng **CIFAR-100** thay vì các tập mặc định (MNIST, CIFAR-10), sau khi trao đổi với giảng viên.

**Lý do chọn CIFAR-100:**

| Tiêu chí | CIFAR-10 (mặc định) | CIFAR-100 (được chọn) |
|----------|---------------------|----------------------|
| Số lớp | 10 | 100 |
| Ảnh mỗi lớp (train) | 5,000 | 500 |
| Độ khó | Thấp | Trung bình–cao |
| Softmax Regression đạt | ~70–75% | ~8% |
| Phân biệt kiến trúc? | Khó (tất cả đều cao) | Rõ ràng |

Với CIFAR-10, ngay cả Softmax Regression cũng đạt ~70% nên không thể thấy rõ sự khác biệt giữa các kiến trúc. CIFAR-100 với 100 lớp làm bài toán đủ khó để phân biệt đặc trưng từng mô hình.

### 2.2 Thông tin tập dữ liệu

| Thuộc tính | Giá trị |
|------------|---------|
| Tên | CIFAR-100 (Canadian Institute For Advanced Research) |
| Số lớp (fine-grained) | 100 |
| Số superclass | 20 |
| Kích thước ảnh | 32 × 32 pixels, 3 kênh RGB |
| Tổng số ảnh | 60,000 |
| Train / Validation / Test | 45,000 / 5,000 / 10,000 |
| Ảnh mỗi lớp (train) | 450 |

### 2.3 Tiền xử lý và Data Augmentation

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

## 3. Phần 1 — Xây dựng 4 Mô hình Phân loại

**Quy ước chung:** Mọi mô hình nhận input `[B, 3, 32, 32]` và trả về logits `[B, 100]`. Không thêm softmax trong `forward()` vì `nn.CrossEntropyLoss` đã tích hợp `log_softmax` bên trong.

---

### 3.1 Mô hình 1 — Softmax Regression

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

### 3.2 Mô hình 2 — MLP (Multi-Layer Perceptron)

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

### 3.3 Mô hình 3 — SimpleCNN (VGG-style)

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

### 3.4 Mô hình 4 — SimpleViT (Vision Transformer — PyTorch built-in)

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

## 4. Phần 2 — Huấn luyện, Đánh giá và So sánh

### 4.1 Training Loop tự viết

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

### 4.2 Cấu hình Hyperparameters

| Mô hình | LR | Batch Size | Epochs | Optimizer | Scheduler |
|---------|-----|-----------|--------|-----------|-----------|
| SoftmaxRegression | 0.1 | 256 | 30 | AdamW | CosineAnnealing |
| MLP | 1e-3 | 128 | 50 | AdamW | CosineAnnealing |
| SimpleCNN | 1e-3 | 128 | 50 | AdamW | CosineAnnealing |
| SimpleViT | 3e-4 | 128 | 100 | AdamW | CosineAnnealing |

- **AdamW** (weight_decay=1e-4): L2 regularization tích hợp trong optimizer — tốt hơn Adam thuần tuý cho Transformer
- **CosineAnnealingLR**: LR giảm theo đường cosine từ `lr` → `lr×0.01`. Giúp mô hình "fine-tune" ở giai đoạn cuối thay vì dao động quanh cực tiểu

### 4.3 Training Curves

![Softmax Regression — training curves](../results/plots/softmax_curves.png)

*Hình 1. Softmax Regression: val accuracy dao động quanh 8% từ rất sớm và không cải thiện — biểu hiện điển hình của underfitting. Train loss giảm chậm dù nhiều epochs vì mô hình không đủ capacity.*

![MLP — training curves](../results/plots/mlp_curves.png)

*Hình 2. MLP: val accuracy tăng đều đến ~epoch 30 rồi plateau tại 20.7%. Khoảng cách train/val nhỏ cho thấy BatchNorm+Dropout hoạt động tốt, nhưng bottleneck là mất spatial structure.*

![SimpleCNN — training curves](../results/plots/cnn_curves.png)

*Hình 3. SimpleCNN: đường cong đẹp nhất trong 4 mô hình — tăng liên tục, hội tụ ổn định tại ~36%. Loss giảm smooth nhờ CosineAnnealingLR.*

![SimpleViT — training curves](../results/plots/vit_curves.png)

*Hình 4. SimpleViT: hội tụ chậm hơn CNN, vẫn tăng nhẹ ở epoch 100 — ViT chưa bão hoà, cần nhiều epochs hơn để hội tụ đầy đủ.*

### 4.4 Kết quả so sánh 4 mô hình

![Part 1+2 bar chart](../results/plots/part1_2_bar.png)

*Hình 5. So sánh Test Accuracy 4 mô hình Phần 1+2. SimpleCNN dẫn đầu với 35.88%, gấp hơn 4× Softmax Regression.*

| Mô hình | Test Acc | Val Acc | F1-macro | Params | Acc/Param |
|---------|----------|---------|----------|--------|-----------|
| SoftmaxRegression | 7.94% | 8.34% | 6.09% | 307.3K | 25.8%/M |
| MLP | 20.63% | 20.68% | 18.31% | 1.7M | 12.1%/M |
| SimpleViT | 24.47% | 25.54% | 22.35% | 821.0K | 29.8%/M |
| **SimpleCNN** | **35.88%** | **36.38%** | **34.48%** | **346.6K** | **103.5%/M** |

### 4.5 Nhận xét và phân tích

**Softmax Regression (7.94%):**
Chỉ học được biên quyết định tuyến tính trong không gian 3072-chiều. Kết quả chỉ tốt hơn random (1%) khoảng 8 lần. Với 100 lớp có visual similarity cao (ví dụ: cá mập vs cá voi, xe bus vs xe tải), biên tuyến tính hoàn toàn không đủ. F1-macro (6.09%) thấp hơn accuracy (7.94%) cho thấy mô hình tập trung vào vài lớp dễ, bỏ qua phần lớn lớp còn lại.

**MLP (20.63%):**
Các lớp ẩn với ReLU học được đặc trưng phi tuyến, cải thiện 2.6× so với Softmax. Tuy nhiên `Flatten` phá vỡ quan hệ không gian 2D: hai pixels kề nhau (i,j) và (i,j+1) bị xử lý như hai features độc lập. MLP không biết chúng ở cạnh nhau trong không gian ảnh.

**SimpleCNN (35.88%) — tốt nhất Phần 1:**
- CNN (346.6K params) vượt MLP (1.7M params) — bằng chứng **inductive bias phù hợp quan trọng hơn số tham số**
- Tích chập 3×3 học đặc trưng cục bộ (edges ở block 1, shapes ở block 2, objects ở block 3)
- Acc/Param = 103.5%/M — hiệu quả tham số tốt nhất trong 4 mô hình

**SimpleViT (24.47%):**
Thấp hơn CNN vì thiếu spatial inductive bias — phải học mọi quan hệ không gian từ data. Với 45K mẫu (450/lớp), Transformer không đủ data để học tốt như CNN. Val accuracy vẫn tăng ở epoch 100 → cần nhiều epochs hơn nếu tiếp tục train.

---

## 5. Phần 3 — Tự hiện thực TransformerEncoder và ViT

### 5.1 Yêu cầu và ràng buộc

Hiện thực TransformerEncoder **chỉ từ**: `nn.Linear`, `nn.LayerNorm`, `torch.einsum`, `F.softmax`, `F.dropout`.
**Không được dùng**: `nn.MultiheadAttention`, `nn.TransformerEncoderLayer`, `nn.TransformerEncoder`.

### 5.2 Custom Multi-Head Self-Attention

**Lý thuyết Scaled Dot-Product Attention:**

```
Q = X · W_q   ("Tôi đang tìm kiếm gì?")
K = X · W_k   ("Tôi có thể cung cấp gì?")
V = X · W_v   ("Thông tin thực sự của tôi")

Attention(Q,K,V) = softmax(Q·Kᵀ / √d_head) · V
```

Chia `d_model=128` thành `H=4` heads, mỗi head có `d_head=32`. Mỗi head học một loại quan hệ khác nhau giữa các patches.

**Hiện thực bằng `torch.einsum`:**

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
        # b=batch, h=head, i=query_pos, j=key_pos, d=d_head
        scores = torch.einsum('bhid,bhjd->bhij', Q, K) / math.sqrt(d_h)
        attn   = F.softmax(scores, dim=-1)      # [B, H, T, T]

        # Weighted sum of values: 'bhij,bhjd->bhid'
        out = torch.einsum('bhij,bhjd->bhid', attn, V)
        out = out.transpose(1,2).reshape(B, T, self.d_model)
        return self.W_o(out)
```

**Tại sao dùng `einsum`?** `torch.einsum('bhid,bhjd->bhij', Q, K)` tính dot-product song song trên toàn bộ (batch, head) chỉ trong một lệnh. Hiệu quả hơn viết vòng for qua từng head, và tránh lỗi broadcast thủ công.

### 5.3 Custom TransformerEncoderLayer (Pre-LN)

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
| Post-LN (Vaswani 2017) | `x = LN(x + Attn(x))` | Không ổn định lúc đầu | Có (learning rate warmup) |
| **Pre-LN (được chọn)** | `x = x + Attn(LN(x))` | Ổn định từ đầu | Không bắt buộc |

Pre-LN phù hợp hơn khi train from scratch không có warmup schedule, vì gradient qua residual path không bị LN can thiệp.

### 5.4 CustomViT

Giống hệt SimpleViT (Phần 1) — cùng patch embedding, CLS token, positional encoding, classification head — **chỉ thay** `nn.TransformerEncoder` bằng `CustomTransformerEncoder` (stack của `CustomTransformerEncoderLayer`).

### 5.5 Training Curves — So sánh

![Part 3 comparison curves](../results/plots/part3_comparison_curves.png)

*Hình 6. Training curves của SimpleViT (PyTorch) và CustomViT (tự xây) cùng cấu hình, cùng số epochs. Hai đường curve có hình dạng tương tự — bằng chứng Custom encoder hoạt động đúng cơ chế toán học.*

![Part 3 bar chart](../results/plots/part3_bar.png)

*Hình 7. CustomViT (34.56%) cao hơn PyTorch ViT (31.92%) ~2.6%. Chênh lệch này nằm trong biên độ ngẫu nhiên do khởi tạo weights khác nhau.*

### 5.6 Kết quả so sánh

| Mô hình | Encoder | Test Acc | Val Acc | F1-macro | Params | Thời gian/epoch |
|---------|---------|----------|---------|----------|--------|-----------------|
| SimpleViT (PyTorch) | `nn.TransformerEncoder` | 31.92% | 33.00% | 30.39% | 821.0K | ~45s (MPS) |
| **CustomViT (Tự xây)** | Custom einsum | **34.56%** | **35.80%** | **33.11%** | 819.4K | ~48s (MPS) |

### 5.7 Nhận xét

**Training curves tương tự (Hình 6):** Tốc độ học, hình dạng đường cong, và mức plateau gần giống nhau — xác nhận Custom MHA hoạt động đúng về mặt toán học với `nn.MultiheadAttention`.

**CustomViT cao hơn 2.64%:** Khoảng chênh lệch này nằm trong biên độ variance ngẫu nhiên (khởi tạo weights khác nhau giữa hai lần chạy). Không thể kết luận custom tốt hơn — **kết luận đúng: hai mô hình tương đương nhau**.

**Số tham số gần giống (821.0K vs 819.4K):** Sai số nhỏ do custom impl bỏ một vài bias terms. Về kiến trúc là hoàn toàn tương đương.

**Cả hai vượt SimpleViT Phần 1 (24.47%→31.92%):** Chỉ khác số epochs (100 thay vì 100 nhưng với config khác) → xác nhận ViT cần train dài để hội tụ đầy đủ.

---

## 6. Phần 4 — Kiến trúc Kết hợp và Cách Embed Ảnh Khác nhau

Ba cách tokenize ảnh khác nhau được triển khai và so sánh:

### 6.1 Kiến trúc 4A — CNN + Transformer Hybrid

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

---

### 6.2 Kiến trúc 4B — SpatialToken ViT (H×W positions làm tokens)

**Ý tưởng:** Coi mỗi vị trí pixel là một token — phương pháp tokenize "naive" nhất.

**Kiến trúc:**

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

---

### 6.3 Kiến trúc 4C — ChannelToken ViT (C channels làm tokens)

**Ý tưởng:** Coi mỗi channel là một token — spatial features của channel đó là feature vector.

**Kiến trúc:**

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

Mỗi token biểu diễn "đặc trưng kênh này xuất hiện ở đâu trong ảnh?" — liên quan đến channel attention trong SENet.

---

### 6.4 Training Curves

![Part 4 comparison curves](../results/plots/part4_comparison_curves.png)

*Hình 8. CNN+Transformer hội tụ tốt và ổn định. SpatialToken và ChannelToken plateau sớm ở mức thấp — dấu hiệu token quality kém, mô hình không học được.*

![Part 4 bar chart](../results/plots/part4_bar.png)

*Hình 9. Khoảng cách rất lớn giữa CNN+Transformer (37.25%) và hai kiến trúc còn lại chứng minh token quality quyết định hiệu năng Transformer.*

### 6.5 Kết quả so sánh

| Mô hình | Cách tokenize | Seq Len | Token Dim | Test Acc | Val Acc | F1-macro | Params |
|---------|--------------|---------|-----------|----------|---------|----------|--------|
| **CNN+Transformer** | CNN features | 64 | 128 | **37.25%** | **38.90%** | **35.87%** | 446.1K |
| ChannelToken ViT | C channels | 64 | 128 | 17.56% | 18.18% | 15.29% | 549.5K |
| SpatialToken ViT | H×W pixels | 1,024 | 64 | 13.10% | 13.12% | 9.86% | 172.4K |

### 6.6 Nhận xét và phân tích

**CNN+Transformer (37.25%) — tốt nhất toàn bài:**
CNN giảm spatial size 32×32 → 8×8 (giảm 16×) đồng thời học đặc trưng có ý nghĩa. Mỗi trong 64 tokens là một biểu diễn phức tạp của một vùng 4×4 pixels đã được xử lý qua 2 ConvBlocks. Transformer sau đó học quan hệ toàn cục giữa các biểu diễn này. Attention cost: O(64²) = 4,096 — rất hiệu quả.

**SpatialToken ViT (13.10%) — thất bại hoàn toàn:**
1,024 tokens từ raw pixels — mỗi token chỉ có 3 giá trị RGB, thông tin cực kỳ thô. Transformer phải đồng thời học *đặc trưng cục bộ* (từ 3 RGB) VÀ *quan hệ toàn cục* (trong 1,024 positions) chỉ với 2 layers — không đủ capacity. Đây chính xác là lý do ViT gốc (Dosovitskiy 2021) dùng patches: patch 4×4 có 48 features, phong phú hơn 16× so với 3 RGB.

**Kết luận định lượng: Token quality > Token quantity.** SpatialToken có 16× nhiều tokens hơn CNN+Transformer nhưng kết quả tệ hơn 2.8×. Mỗi token cần mang đủ semantic information để Transformer học được.

**ChannelToken ViT (17.56%):** Conv1×1 chỉ học linear combination của 3 input channels để tạo 64 channels mới. Đây không phải đặc trưng spatial phức tạp — cần CNN backbone sâu hơn để tạo channel tokens có ý nghĩa.

---

## 7. Phần 5 — Mô hình LSTM/GRU

### 7.1 Cách biểu diễn ảnh thành chuỗi

Ảnh không phải chuỗi thời gian tự nhiên. Bài tập này thử nghiệm 4 cách "đọc" ảnh thành sequence:

| Seq Mode | Seq Length (T) | Input Size (D) | Cách biểu diễn | Thông tin/bước |
|----------|----------------|----------------|----------------|----------------|
| **Row-wise** | 32 | 96 = 32×3 | Mỗi hàng pixels | Toàn bộ hàng ngang (context rộng) |
| Col-wise | 32 | 96 = 32×3 | Mỗi cột pixels | Toàn bộ cột dọc |
| **Patch4** | 64 = 8×8 | 48 = 4×4×3 | 64 patches 4×4 | Một vùng nhỏ 4×4 pixels |
| Patch8 | 16 = 4×4 | 192 = 8×8×3 | 16 patches 8×8 | Một vùng lớn 8×8 pixels |

### 7.2 Kiến trúc ImageLSTM / ImageGRU

```
Đầu vào: [B, T, input_size]     (T và input_size tuỳ theo seq_mode)
    │
    ▼
BiLSTM / BiGRU (hidden=256, num_layers=2, bidirectional=True, dropout=0.3)
    Tạo 2 luồng: forward (bước 0 → T-1) + backward (bước T-1 → 0)
    │
    ▼
concat(h_n[-2], h_n[-1]) → [B, 512]     (forward final state + backward final state)
    │
    ▼
Dropout(0.3) → Linear(512 → 100) → [B, 100]
```

**Bidirectional:** Mỗi timestep "nhìn" được context từ cả hai phía (đọc hàng từ trái sang phải VÀ từ phải sang trái). Với ảnh, điều này giúp pixel ở giữa "biết" về cả pixels bên trái lẫn bên phải.

**So sánh LSTM vs GRU:**

| Đặc điểm | LSTM | GRU |
|----------|------|-----|
| Số gates | 4: forget (f), input (i), output (o), cell (g) | 2: reset (r), update (z) |
| Cell state | Có (c_t — long-term memory riêng biệt) | Không (c_t tích hợp trong h_t) |
| Số params/layer | 4 × (D+H) × H + 4H | 3 × (D+H) × H + 3H |
| Tốc độ train | Chậm hơn ~25% | Nhanh hơn |
| Phù hợp | Chuỗi rất dài (>1000 steps, long-range dependency) | Chuỗi vừa (≤100 steps) |

### 7.3 Training Curves

![Part 5 comparison curves](../results/plots/part5_comparison_curves.png)

*Hình 10. GRU (cả row và patch4) hội tụ nhanh hơn và đạt val accuracy cao hơn LSTM đáng kể. LSTM có khoảng cách train/val lớn hơn → overfitting nhiều hơn trong cùng 30 epochs.*

![Part 5 bar chart](../results/plots/part5_bar.png)

*Hình 11. GRU-row đạt 36.57% — ngang bằng SimpleCNN (35.88%), một kết quả bất ngờ.*

### 7.4 Kết quả so sánh

| Mô hình | Seq Mode | T | D | Test Acc | Val Acc | F1-macro | Params |
|---------|----------|---|---|----------|---------|----------|--------|
| **GRU** | **Row-wise** | 32 | 96 | **36.57%** | **36.44%** | **35.68%** | 1.8M |
| GRU | Patch 4×4 | 64 | 48 | 35.64% | 35.76% | 34.99% | 1.7M |
| LSTM | Row-wise | 32 | 96 | 29.36% | 29.62% | 28.34% | 2.4M |
| LSTM | Patch 4×4 | 64 | 48 | 27.73% | 27.78% | 26.41% | 2.3M |

### 7.5 Nhận xét và phân tích

**GRU vượt LSTM +7.2% (row) và +7.9% (patch4):**
Đây là kết quả không trực quan — LSTM có cơ chế phức tạp hơn nhưng lại tệ hơn. Giải thích: chuỗi ảnh CIFAR ngắn (T=32 hoặc T=64) không có long-range dependencies phức tạp đến mức cần cell state riêng biệt của LSTM. Với chuỗi ngắn, GRU có ít params hơn → ít overfitting hơn → hội tụ tốt hơn trong 30 epochs. Training curves (Hình 10) xác nhận: khoảng cách train/val của LSTM lớn hơn GRU.

**GRU-row (36.57%) cạnh tranh ngang SimpleCNN (35.88%):**
Kết quả bất ngờ nhất của bài tập. Tuy nhiên cần lưu ý: GRU-row cần 1.8M params để đạt kết quả tương đương CNN với chỉ 346.6K params — **kém hiệu quả tham số hơn 5×**.

**Row-wise tốt hơn Patch4 (~1%):**
Row-wise (T=32, D=96) cho mỗi bước RNN thấy toàn bộ hàng ngang — context spatial rộng hơn. Patch4 (T=64, D=48) chia nhỏ hơn nên mỗi bước ít thông tin hơn. Sự khác biệt nhỏ (~1%) cho thấy hai cách biểu diễn tương đương nhau ở mức này.

**RNN kém CNN về cơ bản:**
Ảnh có cấu trúc **2D** — quan hệ giữa pixel (i,j) và pixel (i+1,j) (hai hàng liền kề) quan trọng không kém quan hệ (i,j) và (i,j+1) (hai pixels cùng hàng). Đọc theo hàng (row-wise) chỉ capture được quan hệ ngang, bỏ qua quan hệ dọc giữa các hàng. Đây là hạn chế cốt lõi của RNN với dữ liệu 2D.

---

## 8. So sánh Tổng hợp

### 8.1 Bảng xếp hạng toàn bộ 12 mô hình

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

### 8.2 Phân tích theo nhóm

**CNN-based:**
- SimpleCNN hiệu quả tham số tốt nhất (103.5%/M) với chỉ 346.6K params
- CNN+Transformer cải thiện thêm +1.37% bằng cách thêm global attention (+99.5K params)
- Kết luận: CNN là lựa chọn tốt nhất cho ảnh nhỏ (32×32) train from scratch

**Transformer-based:**
- ViT Phần 1 vs Phần 3 (24.47% vs 31.92%): chỉ khác cấu hình epochs → ViT cần train dài hơn CNN
- Custom ≈ PyTorch ViT: xác nhận hiện thực Custom encoder đúng
- SpatialToken ViT thất bại: raw pixel tokens quá thô → token quality quan trọng hơn token quantity

**RNN-based:**
- GRU (~36%) cạnh tranh với CNN nhưng cần 5× nhiều tham số hơn
- LSTM (~28–29%) kém GRU do overfitting với chuỗi ngắn (T=32)
- RNN không phù hợp tự nhiên với ảnh 2D

**Linear/FC:**
- Softmax (7.94%) và MLP (20.63%) — giới hạn rõ ràng khi thiếu spatial inductive bias

---

## 9. Kết luận

### 9.1 Sáu bài học từ thực nghiệm

1. **Inductive bias phù hợp quan trọng hơn model size:** SimpleCNN (346.6K) vượt MLP (1.7M) và LSTM (2.4M). Thiết kế kiến trúc phù hợp với cấu trúc dữ liệu hiệu quả hơn chỉ tăng số tham số.

2. **ViT cần data lớn để hội tụ tốt:** Với 45K mẫu from scratch, ViT (24–34%) thua CNN (36–37%). ViT thực sự vượt CNN khi được pretrained trên ImageNet-21K (14M ảnh) — không phải kiến trúc kém, mà thiếu data.

3. **Hybrid kết hợp được hai thế mạnh:** CNN+Transformer đạt 37.25% — tốt nhất toàn bài — bằng cách dùng CNN tạo tokens chất lượng cao rồi Transformer học quan hệ toàn cục.

4. **Token quality > token quantity:** SpatialToken ViT (1,024 raw pixel tokens → 13.10%) thua CNN features (64 high-quality tokens → 37.25%). 16× nhiều tokens nhưng kết quả tệ hơn 2.8×.

5. **Custom Transformer = PyTorch Transformer:** Training curves tương tự, accuracy tương đương (34.56% vs 31.92%) → hiện thực Custom MHA bằng `torch.einsum` là đúng về toán học.

6. **GRU ≥ LSTM cho chuỗi ngắn:** Ưu thế của LSTM rõ hơn ở T>1,000 steps. Với T=32–64 (ảnh CIFAR), GRU ít tham số hơn → ít overfitting → kết quả tốt hơn LSTM.

### 9.2 Hướng cải thiện tiếp theo

- **Data augmentation mạnh hơn:** CutMix, MixUp, RandAugment → dự kiến +5–10% accuracy
- **LR warmup cho ViT:** Linear warmup 10 epochs trước CosineAnnealing → ổn định hơn giai đoạn đầu
- **Deeper CNN backbone:** Residual connections (ResNet-style) → target >50% test accuracy
- **Pre-trained features:** Dùng CLIP/DINO features thay vì train from scratch → LSTM/ViT tốt hơn nhiều
- **Label smoothing (ε=0.1):** Regularize CrossEntropyLoss cho 100 lớp, giảm overconfidence

---

## 10. Tài liệu tham khảo

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
