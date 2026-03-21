# CO5085 – Deep Learning & Ứng Dụng trong Thị Giác Máy Tính
## Bài Tập Lớn 1: Phân loại trên Ảnh, Văn bản và Đa phương thức

**Giảng viên:** Lê Thành Sách
**Môn học:** CO5085 – Năm học 2025–2026, HK2
**Hạn nộp:** 23h59, 25/03/2026

---

## 📋 Mục Tiêu

So sánh các mô hình Deep Learning pretrained trên ba loại dữ liệu:

| Loại dữ liệu | Dataset | Mô hình so sánh |
|---|---|---|
| 🖼️ **Ảnh** | CIFAR-100 | CNN (ResNet-50, EfficientNet-B0) vs. ViT (ViT-B/16, DeiT-Small) |
| 📝 **Văn bản** | 20 Newsgroups | RNN (BiLSTM, GRU) vs. Transformer (DistilBERT, BERT-base) |
| 🔀 **Đa phương thức** | CIFAR-100 Superclasses | Zero-shot (CLIP) vs. Few-shot (CLIP + linear probe) |

---

## 🚀 Cài Đặt

```bash
python3 -m venv .venv
source .venv/bin/activate

pip install torch torchvision
pip install -r requirements.txt
pip install ipykernel
python -m ipykernel install --user --name=deeplearning-ass1
```

> macOS Apple Silicon: PyTorch tự động dùng MPS. Không cần `--index-url`.

---

## ▶️ Chạy Training

```bash
# Image models (ResNet-50, EfficientNet-B0, ViT-B/16, DeiT-Small)
python scripts/train_image.py
python scripts/train_image.py --model resnet50  # chạy từng model

# Text models (BiLSTM, GRU, DistilBERT, BERT)
python scripts/train_text.py
python scripts/train_text.py --model distilbert

# Multimodal (CLIP zero-shot + few-shot)
python scripts/train_multimodal.py
```

Kết quả (checkpoints, metrics, biểu đồ) được lưu vào `results/`.

---

## 📓 Notebooks

Tất cả notebooks được sinh tự động bởi `create_notebooks.py`:

| Notebook | Nội dung |
|----------|---------|
| `01_eda_image.ipynb` | EDA – CIFAR-100 |
| `02_eda_text.ipynb` | EDA – 20 Newsgroups |
| `03_eda_multimodal.ipynb` | EDA – Multimodal (CIFAR-100 Superclasses + CLIP) |
| `04_image_cnn_vit.ipynb` | So sánh CNN vs. ViT |
| `05_text_rnn_transformer.ipynb` | So sánh RNN vs. Transformer |
| `06_multimodal_zeroshot_fewshot.ipynb` | CLIP Zero-shot vs. Few-shot |
| `07_extensions.ipynb` | Grad-CAM, Error Analysis, Fine-tune Strategy, Gradio Demo |

```bash
# Tái tạo tất cả notebooks
python create_notebooks.py
```

> **Không chỉnh sửa file `.ipynb` trực tiếp** — mọi thay đổi phải qua `create_notebooks.py`.

---

## 📊 Kết Quả

*(Sẽ được cập nhật sau khi hoàn thành thực nghiệm)*

---

## 🔗 Links

- **GitHub Pages:** [Link GitHub Pages]
- **Video Demo:** [Link video demo]
- **Video Trình Bày:** [Link YouTube]
