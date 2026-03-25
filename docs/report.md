# Báo cáo Bài Tập Lớn – CO5085

**Đề tài:** So sánh Kiến trúc Deep Learning: CNN vs. ViT · RNN vs. Transformer · Zero-shot vs. Few-shot

| | |
|---|---|
| **Môn học** | CO5085 – Deep Learning & Computer Vision Applications |
| **Sinh viên** | Nguyễn Trung Phong – MSSV: 2570047 |
| **Giảng viên** | Lê Thành Sách |
| **Trường** | Đại học Bách Khoa TP.HCM (HCMUT) |
| **Học kỳ** | 2, năm học 2025–2026 |
| **Repository** | https://github.com/PhongNguyenTrung/hcmut-deeplearning-ass1 |

---

## Tóm tắt (Abstract)

Bài tập lớn này trình bày một nghiên cứu thực nghiệm so sánh các kiến trúc mạng nơ-ron sâu trên ba nhiệm vụ: (1) phân loại ảnh trên CIFAR-100 với CNN (ResNet-50) và Vision Transformer (ViT-B/16); (2) phân loại văn bản trên 20 Newsgroups với RNN (GRU) và DistilBERT; (3) học đa phương thức (multimodal) với CLIP ViT-B/32 thực hiện zero-shot image–text retrieval trên **Flickr30k** (1,000 ảnh test, 5,000 captions). Kết quả thực nghiệm cho thấy kiến trúc Transformer vượt trội rõ rệt ở cả ba domain: ViT-B/16 đạt 89.60% (so với 44.11% của ResNet-50), DistilBERT đạt 69.04% (so với 37.85% của GRU), và CLIP zero-shot đạt Recall@1 = **78.9%** (Image→Text) và **58.8%** (Text→Image) trên Flickr30k. Những kết quả này khẳng định vai trò then chốt của cơ chế self-attention và pre-training quy mô lớn trong học sâu hiện đại.

---

## 1. Giới thiệu (Introduction)

Trong thập kỷ vừa qua, các mạng tích chập (Convolutional Neural Network – CNN) và mạng hồi tiếp (Recurrent Neural Network – RNN) đã là trụ cột của học sâu trong thị giác máy tính và xử lý ngôn ngữ tự nhiên. Tuy nhiên, kể từ khi kiến trúc Transformer được đề xuất bởi Vaswani et al. (2017) và sau đó được áp dụng thành công vào thị giác máy tính (Dosovitskiy et al., 2021) cũng như học đa phương thức (Radford et al., 2021), ranh giới về hiệu năng giữa các kiến trúc này đã dần thay đổi.

Bài tập lớn này đặt ra ba câu hỏi nghiên cứu:

1. **CNN vs. ViT:** Kiến trúc Vision Transformer có vượt trội CNN trong bài toán phân loại ảnh fine-grained (CIFAR-100) hay không?
2. **RNN vs. Transformer:** DistilBERT có thực sự mạnh hơn GRU trong phân loại văn bản ngắn không?
3. **Zero-shot Retrieval:** CLIP có thể tìm kiếm ảnh–văn bản hiệu quả trên dataset thật (Flickr30k) mà không cần training không?

Báo cáo được tổ chức như sau: Mục 2 trình bày tổng quan lý thuyết; Mục 3 mô tả dữ liệu và tiền xử lý; Mục 4 trình bày kiến trúc và phương pháp thực nghiệm; Mục 5 phân tích kết quả; Mục 6 trình bày các thí nghiệm mở rộng; Mục 7 thảo luận và Mục 8 kết luận.

---

## 2. Tổng quan lý thuyết (Related Work)

### 2.1 CNN cho Thị giác Máy tính

**ResNet** (He et al., 2016) giải quyết vấn đề vanishing gradient trong mạng sâu thông qua kết nối tắt (skip connection), cho phép huấn luyện các mạng lên đến hàng trăm lớp. ResNet-50 với 25.6M tham số là baseline CNN phổ biến nhất trong các bài toán phân loại ảnh và được chọn làm đại diện cho kiến trúc CNN trong bài tập này.

### 2.2 Vision Transformer (ViT)

**ViT** (Dosovitskiy et al., 2021) chia ảnh thành các patch 16×16, biến mỗi patch thành một token, và áp dụng kiến trúc Transformer encoder chuẩn. Nhờ cơ chế self-attention toàn cục, ViT có thể học được quan hệ không gian ở phạm vi rộng mà CNN khó đạt được. Tuy nhiên, ViT đòi hỏi lượng dữ liệu pre-training rất lớn (ImageNet-21K) để cho kết quả tốt.

### 2.3 RNN cho Xử lý Ngôn ngữ

**GRU** (Cho et al., 2014) sử dụng cơ chế cổng (gating) gồm reset gate và update gate để kiểm soát luồng thông tin theo thời gian, giải quyết phần nào vấn đề vanishing gradient trong RNN truyền thống. So với LSTM, GRU có kiến trúc đơn giản hơn (2 cổng thay vì 3) và thường hội tụ nhanh hơn trên dataset vừa. GRU được chọn làm đại diện RNN trong bài tập này. Hạn chế cơ bản của RNN là xử lý tuần tự (sequential processing) không thể song song hóa và khó nắm bắt phụ thuộc xa (long-range dependency).

### 2.4 Transformer cho NLP

**"Attention Is All You Need"** (Vaswani et al., 2017) thay thế hoàn toàn RNN bằng cơ chế multi-head self-attention, cho phép mô hình truy cập đồng thời mọi vị trí trong chuỗi. **BERT** (Devlin et al., 2019) mở rộng tư tưởng này với pre-training hai chiều trên corpus lớn, tạo ra representations giàu ngữ nghĩa. **DistilBERT** (Sanh et al., 2019) là phiên bản nén của BERT (66M tham số, bằng 60% BERT) sử dụng knowledge distillation, giữ lại 97% hiệu năng.

### 2.5 Vision-Language Models (CLIP)

**CLIP** (Radford et al., 2021) học đồng thời representations của ảnh và văn bản thông qua contrastive learning trên 400 triệu cặp ảnh-văn bản. Kết quả là một không gian embedding chung cho phép so sánh trực tiếp ảnh và text mà không cần fine-tuning — đây là nền tảng của zero-shot image–text retrieval.

---

## 3. Dữ liệu và Tiền xử lý

### 3.1 CIFAR-100 (Image Classification)

CIFAR-100 (Krizhevsky & Hinton, 2009) gồm 60,000 ảnh màu 32×32 thuộc 100 class được chia thành 20 superclass. Tập huấn luyện có 50,000 ảnh (500 ảnh/class), tập kiểm tra có 10,000 ảnh (100 ảnh/class).

**Tiền xử lý và augmentation:**
- **Train:** `RandomCrop(32, padding=4)` → `RandomHorizontalFlip(p=0.5)` → `ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2)` → `ToTensor()` → `Normalize(mean=[0.507, 0.487, 0.441], std=[0.267, 0.256, 0.276])`
- **Val/Test:** `ToTensor()` → `Normalize()`
- **ViT-B/16:** thêm `Resize(224)` do mô hình yêu cầu đầu vào 224×224

### 3.2 20 Newsgroups (Text Classification)

20 Newsgroups (Lang, 1995) gồm khoảng 18,000 bài đăng thuộc 20 nhóm tin tức khác nhau (chính trị, thể thao, khoa học, tôn giáo, v.v.). Tập train: ~11,314 bài; tập test: ~7,532 bài (sau khi lọc `remove=('headers', 'footers', 'quotes')`).

**Tiền xử lý:**
- **DistilBERT:** sử dụng `DistilBertTokenizer` (vocab 30,522 từ), `max_length=256`, padding + truncation, batch size 32
- **GRU:** sử dụng cùng tokenizer nhưng dùng embeddings thô (vocab_size=30,522, embed_dim=300), max_length=256, batch size 64

### 3.3 Flickr30k (Multimodal Retrieval)

**Flickr30k** (Young et al., 2014) là benchmark chuẩn cho bài toán image–text retrieval. Dataset gồm 31,783 ảnh, mỗi ảnh đi kèm 5 captions mô tả nội dung thật do con người viết. Chúng tôi sử dụng **test split chính thức** (1,000 ảnh × 5 captions = 5,000 cặp ảnh–văn bản) được tải từ HuggingFace (`AnyModal/flickr30k`).

**Đặc điểm dataset:**
- Ảnh màu độ phân giải cao (trung bình ~500×400 px), nội dung đa dạng: người, động vật, phong cảnh, hoạt động ngoài trời
- Mỗi ảnh có đúng 5 captions viết độc lập → ground truth rõ ràng cho đánh giá retrieval
- Không cần gán nhãn thủ công — dùng trực tiếp cặp (ảnh, caption) làm ground truth

---

## 4. Kiến trúc và Phương pháp

### 4.1 Image Classification

| Model | Kiến trúc | Tham số | Classification Head |
|---|---|---|---|
| ResNet-50 | CNN, residual blocks | 25.6M | Dropout(0.3) + Linear(2048→100) |
| ViT-B/16 | Transformer, patches 16×16 | 86M | Linear(768→100) |

Cả hai mô hình sử dụng pretrained weights (ImageNet-1K cho ResNet-50, ImageNet-21K cho ViT) và fine-tune toàn bộ mạng trên CIFAR-100.

### 4.2 Text Classification

| Model | Kiến trúc | Tham số | Config |
|---|---|---|---|
| GRU | RNN bidirectional | ~4M | 2 lớp, hidden_dim=256, dropout=0.3 |
| DistilBERT | Transformer 6 layers | 66M | distilbert-base-uncased + Linear(768→20) |

GRU được khởi tạo ngẫu nhiên (không dùng pretrained embeddings), trong khi DistilBERT sử dụng toàn bộ pretrained weights từ HuggingFace.

### 4.3 Multimodal Learning (CLIP – Image–Text Retrieval)

**CLIP ViT-B/32** (151M tham số) với cả image encoder và text encoder được frozen hoàn toàn — không fine-tune, không cần dữ liệu có nhãn.

**Task: Zero-shot Image–Text Retrieval** trên Flickr30k test set (1,000 ảnh, 5,000 captions).

**Phương pháp:**
1. Encode toàn bộ 1,000 ảnh → ma trận $\mathbf{V} \in \mathbb{R}^{1000 \times 512}$ (L2-normalized)
2. Encode toàn bộ 5,000 captions → ma trận $\mathbf{T} \in \mathbb{R}^{5000 \times 512}$ (L2-normalized)
3. Tính similarity matrix $\mathbf{S} = \mathbf{V} \cdot \mathbf{T}^\top \in \mathbb{R}^{1000 \times 5000}$

**Đánh giá – Recall@K:**
- **Image→Text (I→T):** Với mỗi ảnh, rank 5,000 captions theo similarity → kiểm tra có caption đúng trong top-K không
- **Text→Image (T→I):** Với mỗi caption, rank 1,000 ảnh theo similarity → kiểm tra có ảnh đúng trong top-K không
- K ∈ {1, 5, 10}

### 4.4 Cấu hình Huấn luyện

| Hyperparameter | ResNet-50 | ViT-B/16 | GRU | DistilBERT |
|---|---|---|---|---|
| Optimizer | AdamW | AdamW | AdamW | AdamW |
| Learning rate | 1×10⁻³ | 5×10⁻⁵ | 1×10⁻³ | 2×10⁻⁵ |
| Epochs | 5 | 5 | 5 | 3 |
| Batch size | 128 | 64 | 64 | 32 |
| Scheduler | CosineAnnealingLR | CosineAnnealingLR | CosineAnnealingLR | CosineAnnealingLR |
| Gradient clipping | max_norm=1.0 | max_norm=1.0 | max_norm=1.0 | max_norm=1.0 |
| Weight decay | 1×10⁻⁴ | 1×10⁻⁴ | 1×10⁻⁴ | 1×10⁻⁴ |

Tất cả thí nghiệm được thực hiện trên Apple M-series (MPS backend). Mixed-precision AMP chỉ được kích hoạt trên CUDA; trên MPS/CPU sử dụng full-precision float32.

---

## 5. Kết quả và Phân tích

### 5.1 Image Classification (CIFAR-100)

| Model | Loại | Test Accuracy | F1-Macro |
|---|---|---|---|
| ResNet-50 | CNN | 44.11% | 0.434 |
| **ViT-B/16** | **Transformer** | **89.60%** | **0.896** |

**Phân tích:** ViT-B/16 vượt ResNet-50 gần **2 lần** (89.60% vs 44.11%). Kết quả này phản ánh sức mạnh của cơ chế self-attention toàn cục: ViT có thể học được mối quan hệ giữa tất cả các patch trong ảnh ngay từ layer đầu tiên, trong khi CNN chỉ tổng hợp thông tin cục bộ qua từng lớp tích chập. Tuy nhiên, cần lưu ý rằng ViT sử dụng pretrained weights từ ImageNet-21K (14M ảnh) — một lợi thế đáng kể so với ResNet-50 chỉ dùng ImageNet-1K.

![Image Classification Comparison](../results/image_comparison_acc.png)

### 5.2 Text Classification (20 Newsgroups)

| Model | Loại | Test Accuracy | F1-Macro |
|---|---|---|---|
| GRU | RNN | 37.85% | 0.361 |
| **DistilBERT** | **Transformer** | **69.04%** | **0.668** |

**Phân tích:** DistilBERT vượt GRU **+31.2 điểm phần trăm** — khoảng cách lớn nhất trong ba domain. Lợi thế đến từ hai yếu tố chính: (1) pre-training bidirectional trên Wikipedia + BookCorpus (~16GB text) giúp mô hình có representations ngữ nghĩa sâu sắc; (2) cơ chế self-attention xử lý song song toàn bộ chuỗi, tránh được bottleneck của sequential processing trong RNN.

![Text Classification Comparison](../results/text_comparison_acc.png)

### 5.3 Multimodal Learning (CLIP – Flickr30k Retrieval)

**CLIP ViT-B/32 Zero-shot Retrieval** trên Flickr30k test set (1,000 ảnh, 5,000 captions):

| Hướng Retrieval | R@1 | R@5 | R@10 |
|---|---|---|---|
| **Image→Text (I→T)** | **78.90%** | **94.90%** | **98.20%** |
| **Text→Image (T→I)** | **58.78%** | **83.48%** | **90.02%** |

**Phân tích:**

- **Image→Text mạnh hơn Text→Image**: Một ảnh có thể được mô tả bởi nhiều captions khác nhau (5 captions/ảnh) → CLIP dễ tìm được ít nhất 1 caption đúng trong top-K. Ngược lại, từ 1 caption cụ thể cần tìm đúng 1 ảnh trong pool 1,000 ảnh → bài toán khó hơn.

- **R@1 = 78.9% (I→T)** là kết quả ấn tượng cho zero-shot — không training, không labeled data. Điều này cho thấy CLIP đã học được correspondence giữa visual concepts và ngôn ngữ tự nhiên đủ tốt để retrieve trực tiếp.

- **R@10 = 98.2% (I→T)**: Trong 10 captions được retrieve đầu tiên, 98.2% trường hợp có caption đúng — cho thấy embedding space của CLIP phân ly rất tốt giữa các ảnh khác nhau.

- So với **SOTA có supervision** trên Flickr30k (R@1 I→T ≈ 90%+), CLIP zero-shot (78.9%) vẫn kém hơn ~10 điểm, nhưng **không cần training gì cả** — đây là trade-off hợp lý.

![Multimodal Retrieval](../results/multimodal_retrieval.png)

---

## 6. Phân tích Mở rộng (Extensions)

### 6.1 Grad-CAM – Phân tích Khả năng Diễn giải

Gradient-weighted Class Activation Mapping (Grad-CAM; Selvaraju et al., 2017) được áp dụng cho ResNet-50 để trực quan hóa vùng ảnh mà mô hình chú ý khi đưa ra dự đoán. Kết quả cho thấy các lớp tích chập cuối (layer4) tập trung vào đặc trưng ngữ nghĩa cấp cao (hình dạng tổng thể của đối tượng) thay vì các đặc trưng cục bộ như màu sắc hay texture. Heatmap của Grad-CAM nhìn chung bao phủ đúng vùng đối tượng chính trong ảnh.

![Grad-CAM ResNet-50](../results/gradcam_resnet50.png)

### 6.2 Error Analysis – Phân tích Lỗi

Confusion matrix của ResNet-50 trên CIFAR-100 cho thấy xu hướng rõ ràng: phần lớn lỗi xảy ra giữa các class thuộc cùng superclass (ví dụ: nhầm lẫn giữa các loài gặm nhấm trong superclass "small mammals", hoặc giữa các phương tiện trong superclass "vehicles 1"). Điều này phù hợp với đặc điểm của CIFAR-100: các class trong cùng superclass có visual similarity cao ở độ phân giải 32×32.

![Confusion Matrix ResNet-50](../results/confusion_matrix_resnet50.png)

### 6.3 Chiến lược Fine-tune

Hai chiến lược fine-tuning được so sánh trên ResNet-50:

| Chiến lược | Mô tả | Đặc điểm |
|---|---|---|
| Freeze Backbone | Chỉ train classification head | Hội tụ nhanh, ít params được cập nhật |
| Full Fine-tune | Train toàn bộ mạng | Accuracy cao hơn khi có đủ dữ liệu |

**Kết luận:** Full Fine-tune cho accuracy cao hơn khi có đủ dữ liệu (50K ảnh train), vì toàn bộ feature extractor được tối ưu hóa cho domain đích. Freeze Backbone hội tụ nhanh hơn và phù hợp hơn trong điều kiện tài nguyên tính toán hạn chế hoặc dataset nhỏ.

### 6.4 Demo App (Gradio)

Một ứng dụng demo tương tác được xây dựng bằng Gradio, cho phép người dùng tải lên bất kỳ ảnh nào và nhận kết quả phân loại top-5 cùng confidence score từ ResNet-50 (đã fine-tune, 44.11% test accuracy). Ứng dụng sử dụng pipeline đồng nhất với quá trình đánh giá: `Resize(40)` → `CenterCrop(32)` → `Normalize`.

---

## 7. Thảo luận

### 7.1 Transformer vs. Kiến trúc Truyền thống

Kết quả thực nghiệm trên cả ba domain đều xác nhận xu hướng trong literature: Transformer vượt trội CNN và RNN khi có pretrained weights phù hợp. Tuy nhiên, sự so sánh này không hoàn toàn "công bằng" vì:

- ViT được pre-train trên ImageNet-21K (14M ảnh), trong khi ResNet-50 chỉ dùng ImageNet-1K (1.2M ảnh)
- DistilBERT được pre-train trên corpus văn bản khổng lồ, trong khi GRU khởi tạo ngẫu nhiên

Điều này gợi ý rằng lợi thế của Transformer một phần đến từ **quy mô pre-training** hơn là chỉ từ kiến trúc self-attention.

### 7.2 Hiệu quả Tham số

GRU (~4M params) và DistilBERT (66M params) chênh nhau ~16× về số tham số, nhưng DistilBERT vượt trội về accuracy (+31.2 điểm). Điều này cho thấy với bài toán NLP, **chất lượng pre-training** quan trọng hơn nhiều so với kích thước mô hình đơn thuần — GRU khởi tạo ngẫu nhiên không thể cạnh tranh với một Transformer đã được pre-train trên hàng chục GB văn bản, dù nhỏ hơn gấp nhiều lần.

### 7.3 CLIP và Zero-shot Retrieval

Kết quả R@1 = 78.9% (I→T) trên Flickr30k mà không cần training cho thấy CLIP đã học được một không gian embedding chung rất mạnh từ 400M cặp ảnh–văn bản. Điều này mở ra hướng ứng dụng quan trọng: trong các hệ thống tìm kiếm ảnh bằng văn bản (visual search), CLIP có thể được dùng trực tiếp như một backbone mạnh mà không cần fine-tune trên domain cụ thể.

### 7.4 Hạn chế

- Số epochs huấn luyện ít (5 epochs cho ResNet-50/GRU, 3 cho DistilBERT) do giới hạn thời gian tính toán trên MPS
- CIFAR-100 có độ phân giải 32×32 — quá nhỏ so với điều kiện pre-training của ViT (224×224), khiến CNN có thể bị đánh giá thấp hơn khả năng thực
- Số lượng thí nghiệm lặp (random seeds) chưa đủ để báo cáo mean ± std

---

## 8. Kết luận (Conclusion)

Nghiên cứu này đã so sánh toàn diện các kiến trúc deep learning trên ba bài toán phân loại với kết quả rõ ràng:

| Domain | Winner | Metric |
|---|---|---|
| Image Classification | ViT-B/16 | Accuracy = **89.60%** |
| Text Classification | DistilBERT | Accuracy = **69.04%** |
| Multimodal Retrieval | CLIP Zero-shot | R@1 (I→T) = **78.9%** |

**Kết luận chính:**
1. Transformer vượt trội CNN và RNN ở cả ba domain, chủ yếu nhờ cơ chế self-attention toàn cục và pre-training quy mô lớn
2. Pre-trained weights là yếu tố then chốt: fine-tuning mô hình pretrained hiệu quả hơn nhiều so với train từ đầu
3. CLIP zero-shot đạt R@1 = 78.9% (I→T) trên Flickr30k mà không cần bất kỳ dữ liệu training nào — minh chứng cho sức mạnh của contrastive pre-training quy mô lớn

**Hướng phát triển tiếp theo:**
- Tăng số epochs và sử dụng stronger augmentation (MixUp, CutMix) cho ResNet-50 để đánh giá công bằng hơn
- Thử nghiệm GRU với pretrained word embeddings (GloVe, FastText) để cải thiện baseline RNN
- Fine-tune CLIP end-to-end thay vì chỉ linear probe
- Báo cáo kết quả trung bình nhiều lần chạy (mean ± std) để đảm bảo tính tin cậy thống kê

---

## Tài liệu Tham khảo (References)

1. He, K., Zhang, X., Ren, S., & Sun, J. (2016). Deep residual learning for image recognition. *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)*, 770–778.

2. Dosovitskiy, A., Beyer, L., Kolesnikov, A., Weissenborn, D., Zhai, X., Unterthiner, T., ... & Houlsby, N. (2021). An image is worth 16×16 words: Transformers for image recognition at scale. *International Conference on Learning Representations (ICLR)*.

3. Cho, K., van Merrienboer, B., Gulcehre, C., Bahdanau, D., Bougares, F., Schwenk, H., & Bengio, Y. (2014). Learning phrase representations using RNN encoder-decoder for statistical machine translation. *Proceedings of the 2014 Conference on Empirical Methods in Natural Language Processing (EMNLP)*, 1724–1734.

4. Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., ... & Polosukhin, I. (2017). Attention is all you need. *Advances in Neural Information Processing Systems (NeurIPS)*, 30, 5998–6008.

5. Sanh, V., Debut, L., Chaumond, J., & Wolf, T. (2019). DistilBERT, a distilled version of BERT: Smaller, faster, cheaper and lighter. *arXiv preprint arXiv:1910.01108*.

6. Radford, A., Kim, J. W., Hallacy, C., Ramesh, A., Goh, G., Agarwal, S., ... & Sutskever, I. (2021). Learning transferable visual models from natural language supervision. *Proceedings of the 38th International Conference on Machine Learning (ICML)*, 8748–8763.

7. Selvaraju, R. R., Cogswell, M., Das, A., Vedantam, R., Parikh, D., & Batra, D. (2017). Grad-CAM: Visual explanations from deep networks via gradient-based localization. *Proceedings of the IEEE International Conference on Computer Vision (ICCV)*, 618–626.

8. Young, P., Lai, A., Hodosh, M., & Hockenmaier, J. (2014). From image descriptions to visual denotations: New similarity metrics for semantic inference over event descriptions. *Transactions of the Association for Computational Linguistics*, 2, 67–78.
