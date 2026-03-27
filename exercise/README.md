# Exercise — Phân loại Ảnh từ Scratch trên CIFAR-100

**Môn:** CO5085 – Deep Learning & Ứng dụng trong Thị giác Máy tính
**Trường:** ĐH Bách Khoa TP.HCM (HCMUT)
**Học kỳ:** 2025–2026, HK2 | **Deadline:** 01/04/2026

---

## Nội dung bài tập

| Phần | Nội dung | Notebook |
|------|---------|---------|
| **Phần 1** | Xây dựng 4 mô hình: Softmax, MLP, CNN, ViT (PyTorch) | [part1_models.ipynb](notebooks/part1_models.ipynb) |
| **Phần 2** | Training loop tự viết + so sánh 4 mô hình | [part2_train_compare.ipynb](notebooks/part2_train_compare.ipynb) |
| **Phần 3** | Custom TransformerEncoder từ đầu + Custom ViT | [part3_custom_transformer.ipynb](notebooks/part3_custom_transformer.ipynb) |
| **Phần 4** | CNN+Transformer, Spatial tokens, Channel tokens | [part4_architectures.ipynb](notebooks/part4_architectures.ipynb) |
| **Phần 5** | LSTM/GRU với row/col/patch representations | [part5_lstm_gru.ipynb](notebooks/part5_lstm_gru.ipynb) |

---

## Cài đặt môi trường

```bash
# Từ thư mục gốc của project (hcmut-deeplearning-ass1/)
source .venv/bin/activate

# Cài thêm nếu cần
pip install scikit-learn seaborn tqdm
```

**Kernel Jupyter:** Chọn kernel `deeplearning-ass1` trong VS Code.

---

## Chạy Training

### Chạy từng phần bằng script

```bash
# Từ thư mục exercise/
cd exercise

# Phần 1 & 2: Train 4 mô hình cơ bản
python scripts/run_part1_2.py                # Train tất cả
python scripts/run_part1_2.py --model cnn   # Chỉ train CNN
python scripts/run_part1_2.py --epochs 5   # Test nhanh 5 epochs

# Phần 3: Custom TransformerEncoder
python scripts/run_part3.py
python scripts/run_part3.py --epochs 10   # Test nhanh

# Phần 4: Các kiến trúc tokenization
python scripts/run_part4.py

# Phần 5: LSTM/GRU
python scripts/run_part5.py
```

### Chạy qua Jupyter Notebook

```bash
# Từ thư mục exercise/
cd exercise
jupyter notebook
# Hoặc mở từng notebook trong VS Code
```

**Thứ tự chạy:** Part 1 → Part 2 → Part 3 → Part 4 → Part 5

---

## Cấu trúc thư mục

```
exercise/
├── src/                      # Source code (tách biệt với ass1)
│   ├── data.py              # CIFAR-100 data loaders
│   ├── train.py             # Custom training loop
│   ├── utils.py             # Metrics, visualization
│   ├── models_part1.py      # SoftmaxRegression, MLP, SimpleCNN, SimpleViT
│   ├── models_part3.py      # CustomMultiHeadAttention, CustomViT
│   ├── models_part4.py      # CNNTransformerHybrid, SpatialTokenViT, ChannelTokenViT
│   └── models_part5.py      # ImageLSTM, ImageGRU
├── notebooks/               # Jupyter notebooks (5 phần)
├── scripts/                 # Training scripts CLI
├── results/                 # Tự tạo khi chạy
│   ├── checkpoints/        # Model weights (.pt)
│   ├── plots/              # Biểu đồ PNG
│   └── metrics/            # Metrics JSON
└── create_notebooks.py      # Tái tạo notebooks từ đầu
```

---

## Dataset: CIFAR-100

- **100 lớp** (apple, bear, bicycle, ...)
- Ảnh màu **32×32 pixels**
- **50,000 train** / **10,000 test**
- Chia train: 45,000 train + 5,000 validation

Data được tải tự động vào `../data/image/` khi lần đầu chạy.

---

## Kết quả dự kiến

| Mô hình | Test Accuracy (dự kiến) |
|---------|------------------------|
| Softmax Regression | ~15–20% |
| MLP | ~35–42% |
| SimpleCNN | ~50–60% |
| SimpleViT / CustomViT | ~35–50% |
| CNN+Transformer Hybrid | ~50–58% |
| LSTM/GRU variants | ~35–45% |

> Không dùng pretrained weights — train từ đầu trên CIFAR-100.

---

## Thiết bị

Script tự chọn: **CUDA** → **MPS** (Apple Silicon) → **CPU**

Thời gian ước tính (MPS M2):
- Phần 1+2: ~2–4 giờ (4 mô hình × 30–100 epochs)
- Phần 3: ~2 giờ (100 epochs)
- Phần 4: ~1.5 giờ (3 mô hình × 50 epochs)
- Phần 5: ~1 giờ (4 configs × 30 epochs)
