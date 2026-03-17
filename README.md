# CO5085 – Deep Learning & Ứng Dụng trong Thị Giác Máy Tính
## Bài Tập Lớn 1: Phân loại trên Ảnh, Văn bản và Đa phương thức

**Giảng viên:** Lê Thành Sách  
**Môn học:** CO5085 – Năm học 2025–2026, HK2  
**Hạn nộp:** 23h59, 25/03/2026

---

## 📋 Mục Tiêu

So sánh các mô hình Deep Learning pretrained trên ba loại dữ liệu:

| Loại dữ liệu | Mô hình so sánh |
|---|---|
| 🖼️ **Ảnh** | CNN (ResNet-50, EfficientNet-B0) vs. ViT (ViT-B/16, DeiT-S) |
| 📝 **Văn bản** | RNN/BiLSTM vs. Transformer (DistilBERT, BERT) |
| 🔀 **Đa phương thức** | Zero-shot (CLIP) vs. Few-shot (CLIP + linear probe) |

---

## 📁 Cấu Trúc Dự Án

```
hcmut-deeplearning-ass1/
├── data/
│   ├── image/        # CIFAR-100 / Food-101
│   ├── text/         # 20 Newsgroups / DBpedia
│   └── multimodal/   # Flickr30k / MS-COCO subset
├── notebooks/
│   ├── 01_eda_image.ipynb
│   ├── 02_eda_text.ipynb
│   ├── 03_eda_multimodal.ipynb
│   ├── 04_image_cnn_vit.ipynb
│   ├── 05_text_rnn_transformer.ipynb
│   ├── 06_multimodal_zeroshot_fewshot.ipynb
│   └── 07_extensions.ipynb
├── src/
│   ├── datasets.py   # Dataset & DataLoader
│   ├── models.py     # Model architectures
│   ├── train.py      # Training utilities
│   └── evaluate.py   # Evaluation & metrics
├── results/          # Figures, logs, metrics
├── docs/             # GitHub Pages source
└── requirements.txt
```

---

## 🚀 Cài Đặt

```bash
pip install -r requirements.txt
```

---

## 📊 Kết Quả

*(Sẽ được cập nhật sau khi hoàn thành thực nghiệm)*

---

## 🔗 Links

- **GitHub Pages:** [Link GitHub Pages]
- **Video Demo:** [Link video demo]
- **Video Trình Bày:** [Link YouTube]
