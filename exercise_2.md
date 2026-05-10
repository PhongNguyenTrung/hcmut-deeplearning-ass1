ĐẠI HỌC BÁCH KHOA
Khoa Khoa học và Kỹ thuật Máy tính
Bài tập lớn số 2
Học sâu và ứng dụng trong thị giác máy tính
Mã môn học: CO5085
Năm học & Học kỳ: 2025–2026, Học kỳ 2
Giảng viên: Lê Thành Sách
Đề bài giao cho sinh viên thực hiện
Contents
1 Chủ đề và hình thức thực hiện 2
1.1 Chủ đề . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 2
1.2 Hình thức thực hiện . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 2
2 Mục tiêu 2
3 Yêu cầu về tập dữ liệu 3
3.1 Ràng buộc chung . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 3
3.2 Gợi ý theo chủ đề . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 3
4 Yêu cầu về kỹ thuật 4
4.1 Phân đoạn ảnh . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 4
4.2 Phát hiện đối tượng . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 4
4.3 Định danh đối tượng . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 4
4.4 Tạo sinh hình ảnh . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 5
4.5 Metric đánh giá . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 5
5 Tiêu chí chấm điểm 5
6 Yêu cầu về báo cáo và sản phẩm nộp 6
6.1 Landing page (GitHub Pages) . . . . . . . . . . . . . . . . . . . . . . . . . 6
6.1.1 Trang chung (trang chủ) . . . . . . . . . . . . . . . . . . . . . . . . 6
6.1.2 Trang cho bài tập lớn số 2 . . . . . . . . . . . . . . . . . . . . . . . 6
7 Nộp báo cáo và hạn nộp 7
7.1 Yêu cầu nộp báo cáo . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 7
7.2 Hạn nộp . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 7
1
1 Chủ đề và hình thức thực hiện
1.1 Chủ đề
Mỗi nhóm chọn một trong các chủ đề sau để thực hiện bài tập lớn số 2:
1. Phân đoạn ảnh (Image segmentation): phân đoạn ngữ nghĩa (semantic segmentation), phân đoạn theo instance (instance segmentation), hoặc phân đoạn theo panoptic.
2. Phát hiện đối tượng từ ảnh (Object detection): phát hiện và định vị đối tượng
trong ảnh (bounding box), có thể kèm theo phân loại.
3. Định danh đối tượng (Object recognition / Re-identification): nhận diện hoặc định
danh đối tượng (ví dụ: nhận diện khuôn mặt, định danh phương tiện, ReID).
4. Tạo sinh hình ảnh (Image generation): tạo ảnh mới từ mô tả, từ ảnh khác (translation, super-resolution, inpainting) hoặc từ nhiễu (generative models).
5. Chủ đề khác: nhóm có thể đề xuất chủ đề khác trong phạm vi thị giác máy tính và
học sâu, nhưng bắt buộc phải thảo luận và được sự đồng ý của giảng viên
trên lớp trước khi triển khai.
Nhóm cần nêu rõ chủ đề đã chọn trong báo cáo và trên landing page. Một khi đã đăng
ký chủ đề (qua LMS hoặc theo hướng dẫn của giảng viên), nhóm không nên đổi chủ đề
nếu không có lý do đặc biệt và sự đồng ý của giảng viên.
1.2 Hình thức thực hiện
• Thực hiện theo nhóm 2–3 sinh viên.
• Sinh viên đăng ký nhóm theo link công bố trên LMS (có thể dùng chung nhóm với bài
tập lớn số 1).
• Lưu ý: Nhóm đăng ký sẽ thực hiện chung cho tất cả bài tập và bài tập lớn trong môn
học.
2 Mục tiêu
Sau khi hoàn thành bài tập lớn này, sinh viên có khả năng:
• Vận dụng kiến thức học sâu phù hợp với chủ đề đã chọn: mô hình phân đoạn (U-Net,
FCN, DeepLab, Mask R-CNN, v.v.), mô hình phát hiện đối tượng (YOLO, Faster RCNN, DETR, v.v.), mô hình định danh (metric learning, ReID, face recognition), hoặc
mô hình tạo sinh (GAN, VAE, Diffusion, Stable Diffusion, v.v.).
2
• Chuẩn bị dữ liệu (Dataset, DataLoader), áp dụng augmentation phù hợp và thiết lập
quy trình huấn luyện/đánh giá theo bài toán (metric: IoU, mAP, FID, v.v.).
• So sánh và phân tích kết quả giữa ít nhất hai phương pháp hoặc hai biến thể (ví dụ:
kiến trúc khác nhau, backbone khác nhau, có/không pretrained) qua bảng số liệu và
biểu đồ.
• Trình bày và bảo vệ kết quả qua báo cáo, video demo và video trình bày; tổ chức mã
nguồn và tài liệu trên GitHub Pages (cập nhật landing page đã tạo ở bài tập lớn số 1).
• (Tùy chọn) Mở rộng bằng kỹ thuật giải thích mô hình, tối ưu hiệu năng, hoặc hướng
tiếp cận khác để nâng cao điểm số.
3 Yêu cầu về tập dữ liệu
Nhóm cần chọn tập dữ liệu phù hợp với chủ đề đã chọn và đáp ứng các yêu cầu tối
thiểu sau:
3.1 Ràng buộc chung
• Phạm vi: Tập dữ liệu công khai, có bản quyền sử dụng cho mục đích học tập (ví dụ:
COCO, Pascal VOC, ADE20K, Cityscapes, CelebA, CIFAR, MNIST, các bộ dữ liệu
trên Kaggle/Hugging Face phù hợp).
• Kích thước: Đủ để huấn luyện và đánh giá có ý nghĩa (ít nhất vài nghìn mẫu ảnh
hoặc tương đương tùy bài toán). Dataset quá nhỏ có thể bị đánh giá thấp.
• Độ khó: Ưu tiên tập có độ phức tạp vừa phải trở lên; tránh tập quá đơn giản (ví dụ
chỉ một vài lớp, ảnh rất đồng nhất).
3.2 Gợi ý theo chủ đề
• Phân đoạn ảnh: Pascal VOC, COCO-Stuff, ADE20K, Cityscapes, CamVid (semantic/instance/panoptic tùy bài toán).
• Phát hiện đối tượng: COCO, Pascal VOC, Open Images (subset), KITTI (nếu làm
phương tiện).
• Định danh đối tượng: LFW, CelebA (khuôn mặt); Market-1501, MSMT17 (ReID);
dataset phương tiện/đối tượng tùy đề tài.
• Tạo sinh hình ảnh: MNIST/CIFAR (khởi đầu); CelebA, LSUN; text-to-image:
COCO captions, Conceptual Captions; hoặc dataset chuyên cho style transfer, superresolution, inpainting.
3
Nhóm nên nêu rõ trong báo cáo lý do chọn dataset và cách đáp ứng các ràng buộc
trên. Nếu có thắc mắc về tính hợp lệ của dataset, nên trao đổi với giảng viên trước khi
triển khai.
4 Yêu cầu về kỹ thuật
Nhóm cần triển khai và so sánh ít nhất hai phương pháp hoặc hai biến thể phù hợp
với chủ đề. Gợi ý theo từng chủ đề:
4.1 Phân đoạn ảnh
So sánh ít nhất hai trong các hướng (có thể dùng mô hình pretrained):
• Kiến trúc: FCN, U-Net, DeepLab (v2/v3/v3+), PSPNet, SegFormer, Mask R-CNN
(nếu làm instance), v.v.
• Backbone khác nhau: ResNet vs ViT vs ConvNeXt.
• So sánh semantic vs instance (nếu dữ liệu cho phép).
4.2 Phát hiện đối tượng
So sánh ít nhất hai trong các hướng:
• Họ one-stage vs two-stage: ví dụ YOLO (v5/v8/v11) vs Faster R-CNN, DETR.
• Backbone hoặc kiến trúc khác nhau (CNN vs Transformer).
• So sánh tốc độ (FPS) và độ chính xác (mAP).
4.3 Định danh đối tượng
So sánh ít nhất hai trong các hướng:
• Mô hình nhận diện khuôn mặt: ArcFace, CosFace, FaceNet; hoặc mô hình ReID (metric
learning, part-based, transformer).
• So sánh loss: Cross-Entropy vs Contrastive vs Triplet vs ArcFace.
• So sánh backbone hoặc cách trích đặc trưng.
4
4.4 Tạo sinh hình ảnh
So sánh ít nhất hai trong các hướng:
• Họ mô hình: GAN (DCGAN, StyleGAN, Conditional GAN) vs VAE vs Diffusion
(DDPM, Stable Diffusion, LDM).
• Text-to-image vs image-to-image (translation, super-resolution, inpainting) tùy đề tài.
• So sánh chất lượng (FID, IS, LPIPS) và/hoặc tốc độ sinh ảnh.
4.5 Metric đánh giá
Nhóm cần báo cáo metric chuẩn theo bài toán:
• Phân đoạn: Ít nhất IoU (mean IoU), có thể thêm Dice, pixel accuracy.
• Phát hiện đối tượng: mAP (ví dụ mAP@0.5, mAP@0.5:0.95); có thể báo cáo FPS.
• Định danh: Accuracy, Recall@k, mAP (tùy bài toán).
• Tạo sinh: FID, Inception Score (IS); với ảnh có điều kiện có thể dùng thêm accuracy
của classifier trên ảnh sinh.
5 Tiêu chí chấm điểm
• 60% điểm: Từ kết quả sử dụng mô hình (pretrained hoặc train từ đầu) phù hợp chủ
đề, huấn luyện/fine-tune và đánh giá theo đúng yêu cầu so sánh ít nhất hai phương
pháp/biến thể ở Mục 4, cùng với báo cáo số liệu và phân tích.
• 40% điểm: Từ các phần mở rộng do nhóm tự chọn. Gợi ý theo chủ đề:
– Phân đoạn: Giải thích vùng quan trọng (Grad-CAM, attention map); so sánh
backbone/decoder; augmentation (CutMix, copy-paste); đánh giá trên tập khác
domain.
– Phát hiện: Phân tích lỗi (confusion, FN/FP); so sánh tốc độ–độ chính xác; thử
trên ảnh video hoặc real-time demo.
– Định danh: Phân tích embedding (t-SNE, PCA); so sánh loss; đánh giá trên tập
khác (cross-dataset).
– Tạo sinh: So sánh GAN vs Diffusion (chất lượng, ổn định); điều khiển thuộc
tính (attribute, text); super-resolution/inpainting kèm metric; demo ứng dụng
(Gradio/Streamlit).
– Chung: Demo ứng dụng (giao diện đơn giản); tối ưu mô hình (quantization,
pruning); báo cáo reproducibility (seed, môi trường). Nhóm có thể đề xuất hình
thức mở rộng khác (nên nêu rõ trong báo cáo và có thể trao đổi với giảng viên).
5
6 Yêu cầu về báo cáo và sản phẩm nộp
6.1 Landing page (GitHub Pages)
Các nhóm cập nhật landing page trên GitHub Pages (đã tạo ở bài tập lớn số 1) để bổ
sung bài tập lớn số 2.
6.1.1 Trang chung (trang chủ)
Trang chung cần chứa:
• Tên nhóm
• Tên các thành viên
• Tên giảng viên môn học
• Link đến từng bài tập lớn của môn học (bài tập lớn số 1 và bài tập lớn số 2)
6.1.2 Trang cho bài tập lớn số 2
Cần có một trang riêng cho bài tập lớn số 2 chứa:
• Tên bài tập lớn số 2 và chủ đề đã chọn (phân đoạn / phát hiện đối tượng / định danh
/ tạo sinh / khác).
• (Ngắn gọn) Tên thành viên + giảng viên
• Link đến video demo của bài tập này
• Link đến video (YouTube) trình bày báo cáo của bài tập này. Lưu ý: giảng viên
sẽ xem video trong quá trình chấm nếu các nhóm chưa kịp báo cáo trực tiếp.
• Link đến code của bài tập này
• Link đến nội dung trình bày của bài tập này, gồm ít nhất:
1. Báo cáo tìm hiểu về bài toán và tập dữ liệu (EDA)
2. Báo cáo về chuẩn bị Dataset, DataLoader và Augmentation
3. Báo cáo về xây dựng, huấn luyện, đánh giá và so sánh (ít nhất hai) phương pháp
4. Báo cáo kết quả thực nghiệm: bảng số liệu; biểu đồ; phân tích và thảo luận
5. Các báo cáo mở rộng khác (nếu có)
6
7 Nộp báo cáo và hạn nộp
7.1 Yêu cầu nộp báo cáo
Ngoài nội dung đã đăng trên GitHub Pages, mỗi nhóm cần nộp slide trình bày tổng
hợp các nội dung đã thực hiện cho bài tập lớn số 2.
• Định dạng file báo cáo: PDF. Sinh viên chuẩn bị slide ở bất kỳ phần mềm nào (ví
dụ: PowerPoint, Google Slides, Keynote, v.v.) nhưng phải đảm bảo định dạng PDF để
tránh lỗi khi nộp.
• Nộp trực tiếp file báo cáo vào LMS.
7.2 Hạn nộp
• Ngày nộp: (Giảng viên cập nhật; ví dụ: 23h59, ngày 08/04/2026.)
• Nộp trễ: Mỗi tuần nộp trễ so với hạn tương ứng bị trừ 20% điểm.
7