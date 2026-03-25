# Script Trình bày – CO5085 Assignment 1
**Nguyễn Trung Phong – MSSV: 2570047**
**Ước tính: ~9–10 phút**

---

## Slide 1 – Trang bìa *(~15s)*

Xin chào thầy. Em là Nguyễn Trung Phong, MSSV 2570047, môn CO5085 – Deep Learning và Computer Vision Applications, học kỳ 2 năm học 2025–2026.

Bài báo cáo hôm nay trình bày kết quả so sánh các kiến trúc deep learning cho bài toán phân loại ảnh, văn bản và đa phương thức.

---

## Slide 2 – Nội dung trình bày *(~20s)*

Bài báo cáo gồm 8 phần chính — từ cơ sở lý thuyết, phân tích dữ liệu EDA, chuẩn bị DataLoader và Augmentation, đến kết quả thực nghiệm của 3 bài toán và kết luận tổng hợp. Em sẽ đi lần lượt theo thứ tự.

---

## Slide 3 – Bối cảnh & Câu hỏi nghiên cứu *(~40s)*

CNN và RNN từng là kiến trúc chủ đạo trong Computer Vision và NLP suốt hơn một thập kỷ. Nhưng từ 2017 đến 2021, kiến trúc Transformer lần lượt chinh phục NLP với BERT, Computer Vision với ViT, và multimodal với CLIP.

Câu hỏi thực tiễn đặt ra là: khi nào nên dùng Transformer thay CNN hay RNN, và chi phí tính toán là bao nhiêu?

Bài assignment này trả lời ba câu hỏi cụ thể:

- **Q1** — ResNet-50 hay ViT-B/16 tốt hơn trên CIFAR-100?
- **Q2** — GRU hay DistilBERT tốt hơn trên 20 Newsgroups?
- **Q3** — CLIP có thể phân loại ảnh hiệu quả với 0 hoặc rất ít mẫu có nhãn trên Flickr30k không?

---

## Slide 4 – Section header: Cơ sở lý thuyết *(~5s)*

*(chuyển slide)*

---

## Slide 5 – Kiến trúc Phân loại Ảnh: CNN vs. ViT *(~50s)*

ResNet-50 là CNN kinh điển sử dụng tích chập cục bộ kernel 3×3 — nó học đặc trưng theo từng vùng ảnh, và receptive field tăng dần qua các lớp từ texture đến semantic. Điểm đặc biệt là skip connection, giúp giải quyết vanishing gradient và cho phép train mạng rất sâu. ResNet-50 có 25.6 triệu tham số và được pre-train trên ImageNet-1K với 1.2 triệu ảnh.

ViT-B/16 hoạt động hoàn toàn khác — nó chia ảnh 224×224 thành 196 patch 16×16, mỗi patch trở thành một token embedding, rồi đưa vào 12 lớp Transformer encoder với multi-head self-attention. Điểm mạnh then chốt là mọi patch có thể "nhìn thấy" mọi patch khác ngay từ layer đầu tiên — không bị giới hạn bởi receptive field như CNN. ViT-B/16 có 86 triệu tham số và được pre-train trên ImageNet-21K với 14 triệu ảnh.

---

## Slide 6 – Kiến trúc Text: GRU vs. DistilBERT *(~50s)*

GRU là mạng hồi tiếp xử lý chuỗi tuần tự token-by-token, mang hidden state qua từng bước. Nó dùng reset gate và update gate để kiểm soát thông tin được nhớ hoặc quên. Phiên bản em dùng là Bidirectional — đọc chuỗi theo cả hai chiều để nắm ngữ cảnh đầy đủ hơn. Hạn chế lớn nhất là sequential bottleneck — không song song hóa được và khó học long-range dependency. Quan trọng là GRU được khởi tạo ngẫu nhiên, không có pre-training.

DistilBERT là phiên bản nén của BERT — 6 lớp Transformer thay vì 12 nhưng giữ lại 97% hiệu năng nhờ knowledge distillation. Nó xử lý toàn bộ chuỗi song song bằng self-attention toàn cục, và đã được pre-train trên Wikipedia cộng BookCorpus — khoảng 16GB văn bản.

---

## Slide 7 – CLIP *(~45s)*

CLIP sử dụng hai encoder song song: image encoder dùng ViT-B/32 và text encoder dùng Transformer. Cả hai được train cùng nhau với mục tiêu contrastive — các cặp ảnh-văn bản khớp nhau phải có cosine similarity cao, còn các cặp không khớp thì similarity thấp. CLIP được pre-train trên 400 triệu cặp ảnh-văn bản từ Internet, tạo ra một không gian embedding chung — ảnh và văn bản mô tả cùng khái niệm nằm gần nhau trong không gian này.

Em ứng dụng CLIP theo hai hướng cho bài toán phân loại đa phương thức: **zero-shot** — dùng prompt "a photo of a {class}" rồi chọn class có cosine similarity cao nhất; và **few-shot** — frozen CLIP encoder cộng với linear head được train trên K ảnh/class. Không có bước fine-tune encoder.

---

## Slide 8 – Section header: Dữ liệu & Tiền xử lý *(~5s)*

*(chuyển slide)*

---

## Slide 9 – EDA: CIFAR-100 *(~25s)*

CIFAR-100 có 60 nghìn ảnh 32×32, 100 class, phân bố hoàn toàn cân bằng — 500 ảnh mỗi class khi train.

Pipeline tiền xử lý cho tập train gồm: RandomCrop với padding 4, RandomHorizontalFlip, và ColorJitter để augment. Tập val và test chỉ normalize. Riêng ViT cần thêm bước resize lên 256 rồi CenterCrop về 224.

Thách thức ở đây là ảnh 32×32 rất nhỏ — các class trong cùng superclass rất dễ bị nhầm lẫn.

---

## Slide 10 – EDA: Văn bản & Multimodal *(~30s)*

20 Newsgroups có khoảng 18 nghìn bài đăng về 20 chủ đề từ chính trị, thể thao, khoa học đến tôn giáo. Độ dài văn bản dao động rất lớn — từ 50 đến hơn 5,000 từ — nên em truncate tại 256 token. Phân phối class tương đối cân bằng.

Flickr30k là dataset thực tế với 31 nghìn ảnh thật, mỗi ảnh có 5 captions do con người viết. Em dùng test split gồm 1,000 ảnh và gán nhãn pseudo-label bằng keyword matching từ captions — ra 10 semantic class: people, dog, water, sports, outdoor, horse, bicycle, food, nature, indoor. Mỗi ảnh đều có cặp ảnh-caption thật, đáp ứng yêu cầu multimodal.

---

## Slide 11 – Section header: Dataset, DataLoader & Augmentation *(~5s)*

*(chuyển slide)*

---

## Slide 12 – Dataset, DataLoader & Augmentation *(~30s)*

Đây là tổng hợp cấu hình DataLoader và Augmentation cho cả 3 bài toán.

Với ảnh — CIFAR-100: split 40K/10K/10K, batch size 128 cho ResNet và 32 cho ViT; augmentation trên tập train gồm RandomCrop, HorizontalFlip, ColorJitter.

Với văn bản — 20 Newsgroups: dùng chung DistilBertTokenizer, max_length 256, batch size 32 cho DistilBERT và 64 cho GRU.

Với multimodal — Flickr30k: test set 1,000 ảnh, keyword labeling ra 10 class, CLIP preprocessing chuẩn: Resize 224 → CenterCrop 224 → Normalize.

---

## Slide 13 – Section header: Bài toán 1 *(~5s)*

*(chuyển slide)*

---

## Slide 14 – Bài toán 1: Cài đặt thực nghiệm *(~25s)*

Cả hai model được fine-tune toàn bộ trên CIFAR-100 trong 5 epochs. ResNet-50 dùng learning rate 0.001 và batch size 128. ViT-B/16 dùng learning rate nhỏ hơn 20 lần — 5×10⁻⁵ — để tránh phá vỡ pretrained features, batch size 32.

Cả hai đều dùng CosineAnnealingLR và gradient clipping max_norm=1.0 để ổn định training.

---

## Slide 15 – Bài toán 1: Kết quả *(~45s)*

Kết quả rất rõ ràng: ResNet-50 đạt 44.11%, còn ViT-B/16 đạt **89.60%** — hơn **45.5 điểm phần trăm**, gần như gấp đôi.

Lý do chính: ViT dùng self-attention toàn cục ngay từ layer đầu, mọi patch trong ảnh tương tác trực tiếp với nhau. ResNet thì chỉ nhìn vùng 3×3 ở mỗi bước — inductive bias cục bộ này không phù hợp với ảnh nhỏ 32×32.

Cần lưu ý: ViT được pre-train trên ImageNet-21K với 14 triệu ảnh, trong khi ResNet-50 chỉ pre-train trên 1.2 triệu ảnh — lợi thế dữ liệu tới 11 lần. Vì vậy lợi thế của ViT đến từ cả kiến trúc lẫn quy mô pre-training — không hoàn toàn thuần túy về kiến trúc.

---

## Slide 16 – Bài toán 1: Training Curves *(~20s)*

Nhìn vào training curves, ResNet-50 hội tụ nhanh và ổn định, có dấu hiệu nhẹ overfitting ở cuối.

ViT-B/16 khởi đầu chậm hơn ở epoch 1–2 — đây là đặc trưng của Transformer, cần thời gian warm-up để thích nghi với task mới. Sau đó tăng rất mạnh và val loss không tăng — không có overfitting trong 5 epochs.

---

## Slide 17 – Section header: Bài toán 2 *(~5s)*

*(chuyển slide)*

---

## Slide 18 – Bài toán 2: Cài đặt thực nghiệm *(~25s)*

Điểm đáng chú ý trong setup Bài toán 2: cả GRU và DistilBERT đều dùng cùng DistilBertTokenizer và max_length 256 — tức là embedding space đầu vào đồng nhất. Sự khác biệt chỉ đến từ kiến trúc xử lý và pretrained weights.

GRU train 5 epochs với lr=0.001, không có pretrained embeddings. DistilBERT chỉ cần 3 epochs với lr=2×10⁻⁵.

---

## Slide 19 – Bài toán 2: Kết quả *(~40s)*

GRU đạt 37.85%, DistilBERT đạt **69.04%** — khoảng cách **31.2 điểm** — đây là gap lớn nhất trong cả 3 domain.

GRU bị hạn chế bởi xử lý tuần tự — không song song, khó học long-range dependency với chuỗi 256 token. Quan trọng hơn, GRU phải học từ đầu trên chỉ 11 nghìn bài — quá ít để khái quát hóa tốt.

Kết luận rút ra: **với NLP, pre-training là yếu tố quyết định hơn kiến trúc đơn thuần**. DistilBERT hội tụ nhanh vì representations đã giàu ngữ nghĩa từ trước.

---

## Slide 20 – Bài toán 2: Training Curves *(~20s)*

GRU đạt khoảng 35–38% rồi bão hòa sớm — training loss tiếp tục giảm nhưng val loss tăng nhẹ, dấu hiệu overfitting rõ.

DistilBERT chỉ cần 3 epochs để đạt 69%, val loss giảm đều, không overfitting. Tốc độ hội tụ vượt trội hoàn toàn.

---

## Slide 21 – Section header: Bài toán 3 *(~5s)*

*(chuyển slide)*

---

## Slide 22 – Bài toán 3: Phương pháp & Cài đặt *(~35s)*

Bài toán 3 so sánh hai cách phân loại ảnh với CLIP trên Flickr30k — zero-shot và few-shot.

Zero-shot: không cần bất kỳ ảnh train nào — chỉ dùng text prompt "a photo of a {class}" rồi tính cosine similarity với image embedding.

Few-shot K=1/5/10/20: freeze hoàn toàn CLIP encoder, chỉ train thêm một linear head kích thước 512×10 trên K ảnh mỗi class. Với 20-shot, tổng số ảnh train chỉ là 200 ảnh.

---

## Slide 23 – Bài toán 3: Kết quả Zero-shot vs. Few-shot *(~50s)*

Kết quả rất ấn tượng. Zero-shot đạt **54.6%** accuracy với 0 ảnh train — chỉ nhờ text prompt.

1-shot thụt lùi xuống 32.8% — 1 ảnh/class không đủ để linear head ước lượng phân phối, dẫn đến overfit nặng.

Từ 5-shot trở đi thì vượt zero-shot: 5-shot đạt 61.2%, 10-shot đạt 76.4%, và **20-shot đạt 93%** với chỉ 200 ảnh train.

Điều này cho thấy CLIP features cực kỳ phân ly — chỉ cần một linear head nhỏ là đủ tách 10 class một cách chính xác. Zero-shot rất mạnh, nhưng few-shot với đủ mẫu vượt trội rõ rệt.

---

## Slide 24 – Section header: Kết quả tổng hợp & Kết luận *(~5s)*

*(chuyển slide)*

---

## Slide 25 – Kết quả tổng hợp & Kết luận *(~40s)*

Tóm lại kết quả ba bài toán:

Phân loại ảnh — ResNet-50: 44.11% so với ViT-B/16: 89.60% — hơn 45.5 điểm.
Phân loại văn bản — GRU: 37.85% so với DistilBERT: 69.04% — hơn 31.2 điểm.
Đa phương thức — Zero-shot: 54.6% so với CLIP 20-shot: 93.0%.

Ba kết luận chính:

**Thứ nhất**, Transformer vượt CNN và RNN trên cả 3 domain nhờ self-attention toàn cục và pre-training quy mô lớn.

**Thứ hai**, pre-trained weights là yếu tố then chốt — fine-tuning từ pretrained vượt xa train từ đầu, đặc biệt khi data ít.

**Thứ ba**, CLIP few-shot đạt 93% accuracy với chỉ 200 ảnh train; zero-shot đạt 54.6% mà không cần bất kỳ dữ liệu training nào — minh chứng rõ nhất cho sức mạnh của contrastive pre-training trên 400 triệu cặp ảnh-văn bản.

---

## Slide 26 – Cảm ơn *(~10s)*

Cảm ơn thầy đã lắng nghe. Toàn bộ code, notebooks và kết quả thực nghiệm được lưu trên repository của em. Em xin hết.

---

## Ghi chú khi quay

- **Tốc độ:** ~130 từ/phút — đọc rõ, không quá nhanh
- **Section header slides** (4, 8, 11, 13, 17, 21, 24): chỉ dừng 2–3 giây, không cần đọc
- **Slide có công thức** (slide 5, 7): đọc bằng lời thay vì ký hiệu toán — ví dụ "công thức self-attention dùng softmax của QK transpose chia căn dk nhân V"
- **Dừng 2–3 giây** khi chuyển slide để tránh bị cắt hình
- **Nhấn mạnh** các con số quan trọng: 89.60%, 69.04%, 93.00%, +45.5 pp, +31.2 pp
