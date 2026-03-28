# Báo cáo Bài Tập — Phân loại Ảnh với các Mô hình Học Sâu Cơ bản

| | |
|---|---|
| **Môn học** | CO5085 – Deep Learning & Ứng dụng trong Thị giác Máy tính |
| **Sinh viên** | Nguyễn Trung Phong – MSSV: 2570047 |
| **Giảng viên** | Lê Thành Sách |
| **Trường** | Đại học Bách Khoa TP.HCM (HCMUT) |
| **Học kỳ** | 2, năm học 2025–2026 |
| **Deadline** | 01/04/2026 |
| **Repository** | https://github.com/PhongNguyenTrung/hcmut-deeplearning-ass1/tree/main/exercise |

---

## Tóm tắt (Abstract)

Báo cáo trình bày quá trình xây dựng và so sánh **12 kiến trúc mạng nơ-ron sâu** cho bài toán phân loại ảnh trên tập dữ liệu **CIFAR-100** (100 lớp, 32×32 pixels, 50,000 mẫu train). Toàn bộ mô hình được huấn luyện **from scratch** — không dùng pretrained weights — với training loop tự viết bằng PyTorch, chạy trên Apple MPS (M-series).

**Kết quả nổi bật:**
- **CNN+Transformer Hybrid** đạt accuracy cao nhất: **37.25%**
- **SimpleCNN** hiệu quả nhất về tham số/accuracy: 35.88% với chỉ 346.6K params
- **GRU-row** bất ngờ cạnh tranh ngang CNN: 36.57%
- **SpatialToken ViT** thất bại hoàn toàn (13.10%) — minh chứng trực quan tại sao ViT dùng patches thay vì pixels
- **Custom TransformerEncoder** tự xây đạt kết quả tương đương PyTorch (34.56% vs 31.92%), xác nhận hiện thực đúng về mặt toán học

---

## 1. Tập dữ liệu & Tiền xử lý

### 1.1 CIFAR-100

| Thuộc tính | Giá trị |
|------------|---------|
| Số lớp (fine-grained) | 100 |
| Số lớp thô (superclass) | 20 |
| Kích thước ảnh | 32 × 32 pixels, 3 kênh màu RGB |
| Tập train / val / test | 45,000 / 5,000 / 10,000 |
| Ảnh mỗi lớp (train) | 450 |

**Lý do chọn CIFAR-100 thay vì CIFAR-10:** 100 lớp làm bài toán đủ khó để phân biệt rõ sức mạnh của từng kiến trúc. Với CIFAR-10, ngay cả Softmax Regression cũng đạt >80% — không có ý nghĩa so sánh.

### 1.2 Tiền xử lý và Data Augmentation

```python
# Normalize với CIFAR-100 stats (KHÔNG dùng ImageNet stats)
mean = [0.5071, 0.4867, 0.4408]
std  = [0.2675, 0.2565, 0.2761]

# Train: augmentation
transforms.RandomCrop(32, padding=4)    # dịch chuyển ảnh ±4px
transforms.RandomHorizontalFlip()       # lật ngang xác suất 50%
transforms.Normalize(mean, std)

# Val/Test: chỉ normalize, không augment
```

> **Tại sao không dùng ImageNet stats?** CIFAR-100 và ImageNet có phân phối màu rất khác nhau. Dùng sai stats làm dữ liệu không được chuẩn hoá đúng, ảnh hưởng đến tốc độ hội tụ và kết quả.

---

## 2. Phần 1 — Xây dựng 4 Mô hình Phân loại

Tất cả mô hình nhận input `[B, 3, 32, 32]` và trả về logits `[B, 100]`.

### 2.1 Softmax Regression

```
[B,3,32,32] → Flatten → [B, 3072] → Linear(3072, 100) → [B, 100]
```

| Tham số | Giá trị |
|---------|---------|
| Input features | 3 × 32 × 32 = 3,072 |
| Số tham số | 307,300 (307.3K) |
| Activation | Không — CrossEntropyLoss tự tính softmax bên trong |

> **Lưu ý:** Không thêm explicit `softmax` trong `forward()` vì `nn.CrossEntropyLoss = log_softmax + NLLLoss`. Thêm softmax trước sẽ cho kết quả sai.

### 2.2 MLP (Multi-Layer Perceptron)

```
Flatten → FC(3072→512)+BN+ReLU+Drop(0.3)
        → FC(512→256)+BN+ReLU+Drop(0.3)
        → FC(256→100)
Params: ~1.7M
```

**Hạn chế:** `Flatten` làm mất thông tin vị trí không gian — pixel (0,0) và pixel (31,31) xử lý như hai đặc trưng độc lập. CNN giải quyết vấn đề này.

### 2.3 SimpleCNN (VGG-style)

```
ConvBlock1: [Conv(3→32)+BN+ReLU]×2 + MaxPool(2)   → [B, 32, 16, 16]
ConvBlock2: [Conv(32→64)+BN+ReLU]×2 + MaxPool(2)  → [B, 64,  8,  8]
ConvBlock3: [Conv(64→128)+BN+ReLU]×2 + MaxPool(2) → [B,128,  4,  4]
AdaptiveAvgPool(1) → FC(128→256)+ReLU+Drop(0.3) → FC(256→100)
Params: ~346.6K
```

**Ưu điểm của CNN:**
- **Locality:** mỗi neuron chỉ "nhìn" vùng 3×3 — học đặc trưng cục bộ (cạnh, góc, texture)
- **Weight sharing:** cùng bộ lọc áp dụng tại mọi vị trí → giảm params đáng kể
- **Translation invariance:** nhận diện đặc trưng bất kể vị trí trong ảnh

### 2.4 SimpleViT (Vision Transformer — PyTorch)

```
1. Patch Embed: Conv2d(3,128,k=4,s=4) → [B,128,8,8] → flatten → [B,64,128]
2. CLS prepend → [B,65,128]  |  PosEmbed(learnable) [1,65,128]
3. TransformerEncoder: 4 layers, d=128, nhead=4, ffn=512, Pre-LN
4. Head: LN → CLS[:,0] → Linear(128,100)
Params: ~821.0K
```

> **Tại sao patch_size=4?** ViT-B/16 gốc dùng patch 16×16 cho ảnh 224×224 → 196 patches. Với 32×32: patch 16×16 → chỉ **4 patches** — quá ít. Patch 4×4 → **64 patches** — đủ thông tin.

---

## 3. Phần 2 — Training Loop & Kết quả Phần 1+2

### 3.1 Training Loop tự viết

```python
def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()                              # 1. Xoá gradient cũ
        logits = model(x)                                  # 2. Forward pass
        loss = criterion(logits, y)                        # 3. CrossEntropyLoss
        loss.backward()                                    # 4. Backpropagation
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # 5. Gradient clipping
        optimizer.step()                                   # 6. Cập nhật tham số
# Optimizer: AdamW(weight_decay=1e-4) + CosineAnnealingLR
```

**Gradient clipping (max_norm=1.0):** Nếu norm gradient vượt quá 1.0, scale xuống. Ngăn "exploding gradients", đặc biệt quan trọng với RNN và Transformer.

### 3.2 Hyperparameters

| Mô hình | LR | Batch | Epochs | Scheduler |
|---------|-----|-------|--------|-----------|
| SoftmaxRegression | 0.1 | 256 | 30 | CosineAnnealing |
| MLP | 1e-3 | 128 | 50 | CosineAnnealing |
| SimpleCNN | 1e-3 | 128 | 50 | CosineAnnealing |
| SimpleViT | 3e-4 | 128 | 100 | CosineAnnealing |

### 3.3 Training Curves

![Softmax training curves](../results/plots/softmax_curves.png)
*Hình 1a. Softmax Regression — val accuracy dao động quanh 8% và không cải thiện: dấu hiệu underfitting điển hình.*

![MLP training curves](../results/plots/mlp_curves.png)
*Hình 1b. MLP — val accuracy tăng đến ~epoch 30 rồi plateau tại 20.7%.*

![CNN training curves](../results/plots/cnn_curves.png)
*Hình 1c. SimpleCNN — đường cong học đẹp nhất: tăng liên tục, hội tụ ổn định tại ~36%.*

![ViT training curves](../results/plots/vit_curves.png)
*Hình 1d. SimpleViT — hội tụ chậm, vẫn tăng nhẹ ở cuối 100 epochs, chưa bão hoà.*

### 3.4 Kết quả Phần 1+2

![Part 1+2 bar chart](../results/plots/part1_2_bar.png)
*Hình 2. SimpleCNN dẫn đầu với 35.88%, gấp hơn 4× so với Softmax.*

| Mô hình | Test Acc | Val Acc | F1-macro | Params |
|---------|----------|---------|----------|--------|
| SoftmaxRegression | 7.94% | 8.34% | 6.09% | 307.3K |
| MLP | 20.63% | 20.68% | 18.31% | 1.7M |
| SimpleViT | 24.47% | 25.54% | 22.35% | 821.0K |
| **SimpleCNN** | **35.88%** | **36.38%** | **34.48%** | **346.6K** |

### 3.5 Phân tích Phần 1+2

**Softmax Regression (7.94%):** Chỉ học được biên quyết định tuyến tính. Kết quả chỉ tốt hơn random (1%) 8 lần — xác nhận giới hạn cứng của mô hình tuyến tính với 100 lớp phi tuyến.

**MLP (20.63%):** Học được đặc trưng phi tuyến nhờ ReLU, nhưng `Flatten` phá vỡ cấu trúc 2D. Cao hơn Softmax 2.6× nhưng vẫn kém CNN vì mất spatial structure.

**SimpleCNN (35.88%) — tốt nhất Phần 1:** Tích chập cục bộ học đặc trưng từ vùng lân cận. Weight sharing giảm params nhưng không giảm biểu diễn. SimpleCNN (346.6K params) vượt MLP (1.7M params) — bằng chứng rõ ràng **inductive bias phù hợp quan trọng hơn model size**.

**SimpleViT (24.47%):** Thấp hơn CNN vì: (1) không có spatial inductive bias; (2) 45K mẫu train quá ít; (3) cần nhiều epochs hơn. ViT thực sự mạnh khi pretrained trên ImageNet-21K (14M ảnh).

**Val ≈ Test accuracy** (khoảng cách <1%): không có data leakage, split strategy hợp lý.

---

## 4. Phần 3 — Custom TransformerEncoder từ đầu

### 4.1 Ràng buộc hiện thực

Chỉ được dùng: `nn.Linear`, `nn.LayerNorm`, `torch.einsum`, `F.softmax`.
Không được dùng: `nn.MultiheadAttention`, `nn.TransformerEncoderLayer`, `nn.TransformerEncoder`.

### 4.2 Custom Multi-Head Self-Attention

**Lý thuyết Scaled Dot-Product Attention:**

```
Q = X @ W_q   ("Tôi đang tìm kiếm gì?")
K = X @ W_k   ("Tôi có thể cung cấp gì?")
V = X @ W_v   ("Thông tin thực của tôi")

scores = Q @ K^T / sqrt(d_head)   ← scale để tránh gradient vanish
attn   = softmax(scores, dim=-1)
output = attn @ V
```

**Hiện thực với `torch.einsum`:**

```python
def forward(self, x):  # x: [B, T, d_model]
    B, T, _ = x.shape
    # Project và reshape thành multi-head
    Q = self.W_q(x).reshape(B, T, H, d_h).transpose(1, 2)  # [B, H, T, d_h]
    K = self.W_k(x).reshape(B, T, H, d_h).transpose(1, 2)
    V = self.W_v(x).reshape(B, T, H, d_h).transpose(1, 2)

    # 'bhid,bhjd->bhij': query[i] dot key[j] — song song trên tất cả (batch, head)
    scores = torch.einsum('bhid,bhjd->bhij', Q, K) / math.sqrt(d_h)
    attn   = F.softmax(scores, dim=-1)  # [B, H, T, T]

    # 'bhij,bhjd->bhid': tổng có trọng số của values
    out = torch.einsum('bhij,bhjd->bhid', attn, V)
    return self.W_o(out.transpose(1,2).reshape(B, T, d_model))
```

**Ký hiệu:** `b`=batch, `h`=head, `i`=query pos, `j`=key pos, `d`=d_head.

### 4.3 Pre-LN vs Post-LN

| Variant | Công thức | Gradient | Cần warmup? |
|---------|-----------|----------|-------------|
| Post-LN (paper gốc 2017) | `x = LN(x + Attn(x))` | Lớn lúc đầu | Có |
| **Pre-LN (được chọn)** | `x = x + Attn(LN(x))` | Ổn định hơn | Không bắt buộc |

Pre-LN phù hợp hơn khi train from scratch không có warmup schedule.

### 4.4 Kết quả Phần 3

![Part 3 comparison curves](../results/plots/part3_comparison_curves.png)
*Hình 3. Hai đường training curve có hình dạng tương tự nhau — bằng chứng Custom encoder hoạt động đúng cơ chế toán học.*

![Part 3 bar chart](../results/plots/part3_bar.png)
*Hình 4. CustomViT (34.56%) cao hơn PyTorch ViT (31.92%) ~2.6% do khởi tạo ngẫu nhiên, không phải lỗi hiện thực.*

| Mô hình | Encoder | Test Acc | Val Acc | F1-macro | Params |
|---------|---------|----------|---------|----------|--------|
| SimpleViT (PyTorch) | nn.TransformerEncoder | 31.92% | 33.00% | 30.39% | 821.0K |
| **CustomViT (Tự xây)** | Custom einsum | **34.56%** | **35.80%** | **33.11%** | 819.4K |

### 4.5 Phân tích Phần 3

**Hai training curves có hình dạng tương tự (Hình 3):** Tốc độ học, điểm bão hoà, và pattern train/val gần giống nhau — xác nhận Custom MHA hoạt động đúng như PyTorch MHA.

**CustomViT cao hơn 2.64%:** Với chỉ 1 lần chạy mỗi model, khoảng chênh lệch này hoàn toàn nằm trong biên độ ngẫu nhiên do khởi tạo weights. Kết luận duy nhất đúng: hai mô hình **tương đương nhau**.

**Cả hai vượt SimpleViT Phần 1 (24.47%→31.92%):** Chỉ khác số epochs → xác nhận ViT cần train dài hơn CNN mới hội tụ đầy đủ.

---

## 5. Phần 4 — Kiến trúc Tokenization Đa dạng

### 5.1 So sánh 3 cách tạo tokens

| Kiến trúc | Cách tạo tokens | Seq len | Token dim | Attention cost |
|-----------|-----------------|---------|-----------|----------------|
| CNN+Transformer | CNN features (8×8) | 64 | 128 | O(64²) = 4,096 |
| ChannelToken ViT | 64 channels sau Conv1×1 | 64 | 128 | O(64²) = 4,096 |
| SpatialToken ViT | 32×32 raw pixels | 1,024 | 64 | O(1024²) = 1,048,576 |

### 5.2 Kiến trúc chi tiết

**4A — CNN+Transformer Hybrid:**
```
[B,3,32,32] → Conv(3→32)+Pool → [B,32,16,16]
            → Conv(32→64)+Pool → [B,64,8,8]
            → reshape [B,64,64] → FC(64→128) → [B,64,128]  # 64 CNN feature tokens
            → CLS + PosEmbed → 2×TransformerLayer(d=128, h=4)
            → CLS[:,0] → FC(128,100)
Params: 446.1K
```

**4B — SpatialToken ViT:**
```
[B,3,32,32] → reshape [B,1024,3] → FC(3,64) → [B,1024,64]  # 1024 raw pixel tokens!
            → PosEmbed → 2×TransformerLayer(d=64, h=4)
            → GlobalAvgPool → FC(64,100)
Params: 172.4K  |  Vấn đề: attention [B,H,1024,1024] ~512MB với batch=32
```

**4C — ChannelToken ViT:**
```
[B,3,32,32] → Conv1×1(3→64)+BN+ReLU → [B,64,32,32]
            → reshape [B,64,1024] → FC(1024,128) → [B,64,128]  # 64 channel tokens
            → PosEmbed → 2×TransformerLayer(d=128, h=4)
            → GlobalAvgPool → FC(128,100)
Params: 549.5K
```

### 5.3 Kết quả Phần 4

![Part 4 comparison curves](../results/plots/part4_comparison_curves.png)
*Hình 5. CNNTransformer hội tụ tốt và ổn định. SpatialToken và ChannelToken bị plateau sớm ở mức thấp — dấu hiệu token quality kém.*

![Part 4 bar chart](../results/plots/part4_bar.png)
*Hình 6. Khoảng cách rất lớn giữa CNN+Transformer (37.25%) và hai mô hình còn lại — chứng minh chất lượng token quyết định hiệu năng Transformer.*

| Mô hình | Tokenization | Seq Len | Test Acc | Val Acc | F1-macro | Params |
|---------|-------------|---------|----------|---------|----------|--------|
| **CNN+Transformer** | CNN features | 64 | **37.25%** | **38.90%** | **35.87%** | 446.1K |
| ChannelToken ViT | C channels | 64 | 17.56% | 18.18% | 15.29% | 549.5K |
| SpatialToken ViT | H×W pixels | 1,024 | 13.10% | 13.12% | 9.86% | 172.4K |

### 5.4 Phân tích Phần 4

**CNN+Transformer (37.25%) — tốt nhất toàn bài:** CNN backbone học đặc trưng có ý nghĩa (edges, textures) trước khi đưa vào Transformer. Transformer sau đó học quan hệ toàn cục giữa các đặc trưng đã học. Kết hợp spatial inductive bias (CNN) với global attention (Transformer).

**SpatialToken ViT (13.10%) — thất bại hoàn toàn:** Mỗi pixel token chỉ có 3 giá trị RGB — thông tin quá thô. Transformer phải đồng thời học đặc trưng cục bộ VÀ quan hệ toàn cục từ 1,024 tokens — quá khó với chỉ 2 layers. Đây chính xác là lý do ViT gốc dùng **patches thay vì pixels**: patch 4×4 có 48 features, phong phú hơn 16× so với 3 RGB của mỗi pixel.

**ChannelToken ViT (17.56%):** Conv1×1 chỉ học linear combination của 3 input channels — không đủ để tạo channel tokens chất lượng cao. Cần CNN backbone sâu hơn.

**Kết luận: Token quality > Token quantity.** SpatialToken có 1,024 tokens (16× nhiều hơn) nhưng kém CNN features (64 tokens). Chất lượng thông tin mỗi token quan trọng hơn số lượng tokens.

---

## 6. Phần 5 — LSTM/GRU cho Phân loại Ảnh

### 6.1 Biểu diễn ảnh thành chuỗi

| Seq Mode | T (seq len) | D (input size) | Mô tả |
|----------|-------------|----------------|-------|
| **Row-wise** | 32 | 96 = 32×3 | Mỗi hàng pixels (context rộng/bước) |
| Col-wise | 32 | 96 = 32×3 | Mỗi cột pixels |
| **Patch4** | 64 = 8×8 | 48 = 4×4×3 | 64 patches 4×4 |
| Patch8 | 16 = 4×4 | 192 = 8×8×3 | 16 patches 8×8 |

### 6.2 Kiến trúc ImageLSTM / ImageGRU

```python
# input: [B, T, input_size]
BiLSTM/BiGRU(hidden=256, num_layers=2, bidirectional=True, dropout=0.3)
# Tạo 2 luồng: forward (0→T) + backward (T→0)
concat(h_n[-2], h_n[-1])  # [B, 512]
Dropout(0.3)
Linear(512, 100)
```

**Bidirectional:** Mỗi timestep nhìn được context từ cả 2 phía (trái→phải và phải→trái).

| | LSTM | GRU |
|--|------|-----|
| Gates | 4: forget, input, output, cell | 2: reset, update |
| Cell state | Có (c_t — long-term memory) | Không (chỉ h_t) |
| Params/layer | 4×(D+H)×H | 3×(D+H)×H |
| Phù hợp | Chuỗi rất dài (>1000 steps) | Chuỗi vừa (≤100 steps) |

### 6.3 Kết quả Phần 5

![Part 5 comparison curves](../results/plots/part5_comparison_curves.png)
*Hình 7. GRU (cả row và patch4) hội tụ nhanh hơn và đạt val accuracy cao hơn LSTM. LSTM có khoảng cách train/val lớn hơn → overfitting nhiều hơn.*

![Part 5 bar chart](../results/plots/part5_bar.png)
*Hình 8. GRU-row đạt 36.57% — gần bằng SimpleCNN (35.88%).*

| Mô hình | Seq Mode | T | D | Test Acc | Val Acc | F1-macro | Params |
|---------|----------|---|---|----------|---------|----------|--------|
| **GRU** | **Row-wise** | 32 | 96 | **36.57%** | **36.44%** | **35.68%** | 1.8M |
| GRU | Patch 4×4 | 64 | 48 | 35.64% | 35.76% | 34.99% | 1.7M |
| LSTM | Row-wise | 32 | 96 | 29.36% | 29.62% | 28.34% | 2.4M |
| LSTM | Patch 4×4 | 64 | 48 | 27.73% | 27.78% | 26.41% | 2.3M |

### 6.4 Phân tích Phần 5

**GRU vượt LSTM +7.2% (row) và +7.9% (patch4):** Chuỗi ảnh CIFAR ngắn (T=32–64) không cần long-term memory của LSTM. GRU ít params hơn → ít overfitting hơn → train hiệu quả hơn trong 30 epochs. Training curves (Hình 7) xác nhận: LSTM có khoảng cách train/val lớn hơn GRU.

**GRU-row (36.57%) gần bằng SimpleCNN (35.88%):** Bất ngờ lớn nhất của bài tập. Tuy nhiên GRU cần 1.8M params để đạt kết quả tương đương CNN với chỉ 346.6K params — kém hiệu quả hơn 5× về tham số.

**Row-wise tốt hơn Patch4 nhẹ (~1%):** Row-wise cho mỗi bước thấy toàn bộ hàng ngang (D=96) — context rộng hơn. Tuy nhiên sự khác biệt nhỏ cho thấy hai cách biểu diễn tương đương.

**LSTM/GRU kém CNN về cơ bản:** Ảnh có cấu trúc **2D**. Đọc theo hàng (1D) làm mất quan hệ chiều dọc giữa các hàng kề nhau — đây là hạn chế cốt lõi của RNN với dữ liệu ảnh.

---

## 7. Grand Summary — So sánh Tất cả Mô hình

### 7.1 Bảng xếp hạng

| # | Mô hình | Phần | Kiến trúc | Test Acc | F1-macro | Params | Acc/Param |
|---|---------|------|-----------|----------|----------|--------|-----------|
| 🥇 | **CNN+Transformer Hybrid** | 4 | Hybrid | **37.25%** | 35.87% | 446.1K | 83.5%/M |
| 🥈 | GRU-row | 5 | RNN | 36.57% | 35.68% | 1.8M | 20.3%/M |
| 🥉 | **SimpleCNN** | 1 | CNN | 35.88% | 34.48% | 346.6K | **103.5%/M** ★ |
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

> ★ **Acc/Param** = Test Accuracy / (Params in millions). SimpleCNN đạt hiệu quả tham số cao nhất trong các mô hình học được tốt (103.5%/M).

### 7.2 Phân tích theo nhóm

**CNN-based:**
- SimpleCNN (346.6K) — hiệu quả tham số tốt nhất (103.5%/M)
- CNN+Transformer cải thiện thêm 1.37% với +99.5K params
- *CNNs vẫn là lựa chọn tốt nhất cho ảnh nhỏ (32×32) train from scratch*

**Transformer-based:**
- ViT Phần 1 vs Phần 3 (24.47% vs 31.92%): chỉ khác số epochs → ViT cần train dài hơn CNN
- Custom ≈ PyTorch ViT: xác nhận hiện thực Custom encoder đúng
- SpatialToken ViT thất bại: raw pixel tokens quá thô
- *ViT cần pre-training hoặc rất nhiều data để vượt CNN from scratch*

**RNN-based:**
- GRU (~36%) cạnh tranh với CNN nhưng kém hiệu quả params (5× nhiều hơn)
- LSTM (~28–29%) kém GRU vì overfitting với chuỗi ngắn
- *RNN không phù hợp tự nhiên với ảnh — có thể học được nhưng kém hiệu quả*

**Linear/FC-based:**
- Softmax (7.94%) và MLP (20.63%) — giới hạn rõ ràng khi thiếu spatial bias

---

## 8. Kết luận

### 8.1 Bài học khoa học chính

1. **Inductive bias phù hợp > model size:** SimpleCNN (346.6K) vượt MLP (1.7M) và LSTM (2.4M). Thiết kế phù hợp với cấu trúc dữ liệu quan trọng hơn số tham số.

2. **ViT cần data lớn:** Với 45K mẫu, ViT from scratch (24–34%) thua CNN (36–37%). ViT thực sự vượt CNN khi pretrained trên ImageNet-21K (14M ảnh).

3. **Hybrid thắng tất cả:** CNN+Transformer kết hợp inductive bias cục bộ (CNN) với global attention (Transformer) — tốt nhất cả hai thế giới (37.25%).

4. **Token quality > token quantity:** SpatialToken ViT (1,024 raw pixel tokens, 13.10%) thua CNN+Transformer (64 CNN feature tokens, 37.25%). 16× nhiều tokens nhưng kết quả tệ hơn 2.8×.

5. **Custom Transformer = PyTorch Transformer:** Accuracy tương đương, training curves gần giống nhau — hiện thực Custom MHA bằng einsum là đúng về toán học.

6. **GRU ≥ LSTM cho chuỗi ngắn:** LSTM có ưu thế ở T>1000. Với T=32–64, GRU ít params hơn → ít overfitting hơn → kết quả tốt hơn.

### 8.2 Hướng cải thiện

- **Data augmentation mạnh hơn:** CutMix, MixUp, RandAugment → dự kiến +5–10%
- **LR warmup cho ViT:** Linear warmup 10 epochs trước CosineAnnealing → ổn định hơn
- **Deeper CNN backbone:** ResNet-like residual connections → target >50% test acc
- **Pre-trained token extractor:** CLIP/DINO features → LSTM/ViT có token chất lượng cao hơn
- **Label smoothing (ε=0.1):** Regularize CrossEntropyLoss cho 100 lớp

---

## Tài liệu tham khảo

1. Dosovitskiy, A. et al. (2021). *An Image is Worth 16×16 Words: Transformers for Image Recognition at Scale*. ICLR 2021.
2. Vaswani, A. et al. (2017). *Attention Is All You Need*. NeurIPS 2017.
3. He, K. et al. (2016). *Deep Residual Learning for Image Recognition*. CVPR 2016.
4. Cho, K. et al. (2014). *Learning Phrase Representations using RNN Encoder-Decoder*. EMNLP 2014.
5. Chung, J. et al. (2014). *Empirical Evaluation of Gated Recurrent Neural Networks*. arXiv 2014.
6. Krizhevsky, A. (2009). *Learning Multiple Layers of Features from Tiny Images*. Technical Report, Univ. of Toronto.
7. Touvron, H. et al. (2021). *Training data-efficient image transformers & distillation through attention*. ICML 2021.

---

*Báo cáo được tạo ngày 28/03/2026 | CO5085 – Deep Learning & Ứng dụng trong Thị giác Máy tính | HCMUT*
