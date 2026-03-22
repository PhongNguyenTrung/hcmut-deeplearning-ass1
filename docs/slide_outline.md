# Slide Outline – CO5085 Assignment 1
**Sinh viên:** Nguyễn Trung Phong – MSSV: 2570047
**Môn:** CO5085 – Deep Learning & Computer Vision Applications
**HCMUT, 2025–2026**

---

## Slide 1 – Trang bìa

**Tiêu đề:** So sánh Kiến trúc Deep Learning
**Phụ đề:** CNN vs ViT · RNN vs Transformer · Zero-shot vs Few-shot

- Môn: CO5085 – Deep Learning & Computer Vision Applications
- Sinh viên: Nguyễn Trung Phong – MSSV: 2570047
- Giảng viên: Lê Thành Sách
- HCMUT, Học kỳ 2, 2025–2026

---

## Slide 2 – Tổng quan bài toán

**3 nhiệm vụ phân loại:**

| # | Nhiệm vụ | Dataset | Classes |
|---|---|---|---|
| 1 | Image Classification | CIFAR-100 | 100 |
| 2 | Text Classification | 20 Newsgroups | 20 |
| 3 | Multimodal (CLIP) | CIFAR-100 Superclasses | 20 |

**Câu hỏi nghiên cứu:**
*Kiến trúc Transformer có thực sự vượt trội CNN/RNN trong các bài toán phân loại không?*

---

## Slide 3 – Kiến trúc & Phương pháp

| Domain | Model | Loại | Params |
|---|---|---|---|
| Image | ResNet-50 | CNN | 25.6M |
| Image | EfficientNet-B0 | CNN | 5.3M |
| Image | **ViT-B/16** | Transformer | 86M |
| Text | BiLSTM | RNN | ~5M |
| Text | GRU | RNN | ~4M |
| Text | **DistilBERT** | Transformer | 66M |
| Multimodal | **CLIP ViT-B/32** | Vision-Language | 151M |

**Cấu hình training chung:** AdamW · CosineAnnealingLR · Gradient Clipping (max_norm=1.0)

---

## Slide 4 – EDA: Dữ liệu

**CIFAR-100:** 50K train / 10K test · 100 classes · 32×32 px
*(Dùng ảnh: cifar100_class_dist.png, cifar100_samples.png)*

**20 Newsgroups:** ~18K bài viết · 20 chủ đề · sau khi xóa headers/footers
*(Dùng ảnh: newsgroups_class_dist.png, newsgroups_length_dist.png)*

**Augmentation (Image):** RandomCrop, RandomHorizontalFlip, ColorJitter
*(Dùng ảnh: augmentation_preview.png)*

---

## Slide 5 – Kết quả: Image Classification (CIFAR-100)

**5 epochs · Pretrained backbone · Fine-tuned toàn bộ**

| Model | Type | Test Accuracy | F1-Macro |
|---|---|---|---|
| ResNet-50 | CNN | 44.11% | 0.434 |
| EfficientNet-B0 | CNN | 45.55% | 0.448 |
| **ViT-B/16** | **ViT** | **89.60%** | **0.896** |

*(Dùng ảnh: image_comparison_acc.png, resnet50_curves.png, vit_b16_curves.png)*

**Nhận xét:**
- ViT vượt CNN ~2x nhờ self-attention toàn cục trên patches 16×16
- EfficientNet hiệu quả hơn ResNet: cùng accuracy nhưng ít params hơn 5x
- CNN hội tụ nhanh hơn nhưng bị giới hạn bởi inductive bias cục bộ

---

## Slide 6 – Kết quả: Text Classification (20 Newsgroups)

**5 epochs (RNN) · 3 epochs (DistilBERT)**

| Model | Type | Test Accuracy | F1-Macro |
|---|---|---|---|
| BiLSTM | RNN | 32.22% | 0.306 |
| GRU | RNN | 37.85% | 0.361 |
| **DistilBERT** | **Transformer** | **69.04%** | **0.668** |

*(Dùng ảnh: text_comparison_acc.png, distilbert_curves.png)*

**Nhận xét:**
- DistilBERT vượt GRU **+31.2%** nhờ pre-training trên 66M tham số
- GRU > BiLSTM: gating mechanism đơn giản hơn phù hợp tập dữ liệu nhỏ
- RNN bị giới hạn bởi sequential processing, không tận dụng được ngữ cảnh xa

---

## Slide 7 – Kết quả: Multimodal CLIP (CIFAR-100 Superclasses)

**CLIP ViT-B/32 · 20 superclasses · Test set 500 ảnh**

| Approach | Shots | Accuracy | F1-Macro |
|---|---|---|---|
| Zero-shot | 0 | 54.60% | 0.517 |
| Few-shot | 1 | 32.80% | 0.338 |
| Few-shot | 5 | 61.20% | 0.622 |
| Few-shot | 10 | 76.40% | 0.766 |
| **Few-shot** | **20** | **93.00%** | **0.932** |

*(Dùng ảnh: multimodal_comparison_acc.png)*

**Nhận xét:**
- 1-shot thấp hơn zero-shot: linear head chưa đủ data để học distribution
- Từ 5-shot: vượt zero-shot và tăng mạnh theo số shots
- 20-shot đạt 93% — CLIP rất hiệu quả khi có ít nhãn

---

## Slide 8 – Extension: Grad-CAM & Error Analysis

*(Dùng ảnh: gradcam_resnet50.png, confusion_matrix_resnet50.png)*

**Grad-CAM – ResNet-50:**
- Model tập trung đúng vào vùng object chính
- Các lớp cuối học đặc trưng ngữ nghĩa cao cấp

**Error Analysis:**
- ResNet-50 hay nhầm các class có hình thái tương đồng (ví dụ: động vật cùng họ)
- Confusion matrix cho thấy lỗi tập trung trong cùng superclass

---

## Slide 9 – Extension: Fine-tune Strategy

**ResNet-50 trên CIFAR-100**

| Strategy | Mô tả |
|---|---|
| Freeze Backbone | Chỉ train classification head |
| Full Fine-tune | Train toàn bộ mạng |

**Kết luận:** Full Fine-tune cho accuracy cao hơn khi có đủ dữ liệu training, Freeze Backbone hội tụ nhanh hơn với ít epochs.

---

## Slide 10 – Demo App

**Gradio CIFAR-100 Classifier:**
- Input: Ảnh bất kỳ
- Output: Top-5 predictions với confidence score
- Model: ResNet-50 (fine-tuned, 44.11% test accuracy)

*(Screenshot hoặc live demo)*

---

## Slide 11 – Kết luận

**Transformer > CNN/RNN ở cả 3 domain:**

| Domain | Winner | Accuracy |
|---|---|---|
| Image | ViT-B/16 | **89.60%** |
| Text | DistilBERT | **69.04%** |
| Multimodal | CLIP 20-shot | **93.00%** |

**Trade-off:**
- Transformer yêu cầu nhiều tài nguyên hơn (params, VRAM)
- Nhưng kết quả vượt trội rõ rệt, đặc biệt khi có pretrained weights
- CLIP few-shot là hướng tiếp cận mạnh khi thiếu dữ liệu nhãn

**Hướng phát triển:**
- Tăng epochs + stronger augmentation cho CNN
- Thử DeiT-S (nhẹ hơn ViT, hiệu năng tương đương)
- Fine-tune CLIP end-to-end thay vì chỉ linear probe

---

## Slide 12 – Tài liệu tham khảo

1. He et al. (2016). Deep Residual Learning for Image Recognition. *CVPR*.
2. Tan & Le (2019). EfficientNet: Rethinking Model Scaling. *ICML*.
3. Dosovitskiy et al. (2021). An Image is Worth 16x16 Words. *ICLR*.
4. Vaswani et al. (2017). Attention Is All You Need. *NeurIPS*.
5. Sanh et al. (2019). DistilBERT: a distilled version of BERT. *arXiv*.
6. Radford et al. (2021). Learning Transferable Visual Models From Natural Language Supervision (CLIP). *ICML*.
