ĐẠI HỌC BÁCH KHOA
Khoa Khoa học và Kỹ thuật Máy tính
Bài tập — Phân loại ảnh với các mô hình
học sâu cơ bản
Học sâu và ứng dụng trong thị giác máy tính
Mã môn học: CO5085
Năm học & Học kỳ: 2025–2026, Học kỳ 2
Giảng viên: Lê Thành Sách
Đề bài giao cho sinh viên thực hiện
Contents
1 Mục tiêu 2
2 Tập dữ liệu 2
2.1 Gợi ý mở rộng (tùy chọn) . . . . . . . . . . . . . . . . . . . . . . . . . . . 2
3 Nội dung bài tập 2
3.1 Phần 1 — Xây dựng các mô hình phân loại . . . . . . . . . . . . . . . . . 2
3.2 Phần 2 — Huấn luyện, đánh giá và so sánh . . . . . . . . . . . . . . . . . 3
3.3 Phần 3 — Tự hiện thực TransformerEncoder và ViT . . . . . . . . . . . . 3
3.4 Phần 4 — Các kiến trúc kết hợp và cách embed ảnh khác nhau . . . . . . 4
3.5 Phần 5 — Mô hình phân loại dựa trên LSTM/GRU . . . . . . . . . . . . . 4
4 Tiêu chí chấm điểm 4
5 Yêu cầu nộp và hạn nộp 5
5.1 Sản phẩm nộp . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 5
5.2 Hình thức thực hiện . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 5
5.3 Hạn nộp . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 5
1
1 Mục tiêu
Bài tập giúp sinh viên nắm vững cơ bản về học sâu và ứng dụng trong thị giác
máy tính, cụ thể:
• Xây dựng và so sánh các mô hình phân loại ảnh từ đơn giản đến phức tạp: softmax
regression, MLP, CNN, Vision Transformer (ViT), và mô hình dựa trên LSTM/GRU.
• Tự viết vòng lặp huấn luyện (training loop) bằng PyTorch, sử dụng backward có sẵn
để tính đạo hàm, thay vì chỉ gọi API cấp cao.
• Hiểu cấu trúc bên trong Transformer qua việc tự hiện thực khối TransformerEncoder
từ các phép toán cơ bản.
• Khám phá các cách thiết kế kiến trúc khác nhau: kết hợp CNN–Transformer, các cách
tokenize/embed ảnh khác nhau, và biểu diễn ảnh thành chuỗi cho RNN.
2 Tập dữ liệu
Sinh viên chọn một tập dữ liệu nhỏ trong số các gợi ý sau (hoặc tập tương đương, nếu
có ý kiến giảng viên):
• MNIST: chữ số viết tay 0–9, ảnh xám 28 × 28, 10 lớp.
• Fashion-MNIST: ảnh thời trang (áo, giày, túi, v.v.), 28 × 28, 10 lớp.
• CIFAR-10: ảnh màu 32 × 32, 10 lớp (máy bay, ô tô, chim, mèo, hươu, chó, ếch, ngựa,
tàu, xe tải).
2.1 Gợi ý mở rộng (tùy chọn)
Nếu muốn thử tập khác cùng mức độ: KMNIST (chữ cái tiếng Nhật), SVHN (số nhà),
EMNIST (chữ và số). Các tập quá nhỏ (ít lớp, ít mẫu) hoặc quá lớn so với phạm vi bài
tập có thể không phù hợp; nên trao đổi với giảng viên nếu chọn ngoài danh sách trên.
Sinh viên cần nêu rõ trong báo cáo tập dữ liệu đã chọn và (ngắn gọn) lý do chọn.
3 Nội dung bài tập
3.1 Phần 1 — Xây dựng các mô hình phân loại
Xây dựng lần lượt các mô hình sau cho bài toán phân loại ảnh trên tập dữ liệu đã chọn:
1. Softmax regression: mô hình tuyến tính (flatten ảnh thành vector, ánh xạ tuyến
tính + softmax) để phân loại.
2
2. MLP (Multi-Layer Perceptron): mạng fully connected nhiều tầng (ẩn + kích hoạt),
đầu ra softmax.
3. Mô hình CNN-based: mạng tích chập (convolutional layers, pooling, có thể batch
norm) + classifier; kiến trúc đơn giản phù hợp với kích thước ảnh nhỏ (ví dụ MNIST/FashionMNIST/CIFAR-10).
4. Mô hình ViT đơn giản: sử dụng các khối có sẵn trong PyTorch (TransformerEncoderLayer,
TransformerEncoder, v.v.) để xây dựng Vision Transformer phân loại — gồm bước
patch embedding (ảnh → chuỗi patch), positional encoding, encoder Transformer, và
head phân loại.
3.2 Phần 2 — Huấn luyện, đánh giá và so sánh
• Tự viết giải thuật huấn luyện (training loop) dựa trên PyTorch: vòng lặp theo epoch/-
batch, gọi loss.backward() và cập nhật tham số bằng optimizer (SGD/Adam, v.v.).
Không dùng API cấp cao như trainer.fit() của Lightning nếu mục đích là chỉ gọi
một lần; bài tập yêu cầu sinh viên hiểu rõ bước forward → loss → backward → step.
• Sử dụng vòng lặp này để huấn luyện, đánh giá (trên tập validation/test) và so sánh
bốn mô hình ở Phần 1 (softmax, MLP, CNN, ViT).
• Lưu ý: Trình bày kết quả khoa học và rõ ràng (bảng số liệu, biểu đồ loss/accuracy,
nhận xét ngắn) — đây cũng là một tiêu chí chấm điểm.
3.3 Phần 3 — Tự hiện thực TransformerEncoder và ViT
1. Tự hiện thực khối TransformerEncoder (hoặc tương đương) từ các phép toán cơ bản
của PyTorch: như Linear, LayerNorm, và einsum. Không dùng nn.TransformerEncoderLayer
hay nn.TransformerEncoder có sẵn cho phần này.
2. Xây dựng mô hình ViT phân loại dựa trên khối tự hiện thực ở trên (patch embedding
và positional encoding có thể dùng cách đơn giản đã dùng ở Phần 1).
3. Huấn luyện, đánh giá và so sánh:
• ViT dùng khối tự xây (Phần 3)
• ViT dùng khối có sẵn trong PyTorch (Phần 1)
Trình bày kết quả (accuracy, loss, thời gian huấn luyện nếu có) và nhận xét ngắn.
3
3.4 Phần 4 — Các kiến trúc kết hợp và cách embed ảnh khác
nhau
1. Xây dựng một số kiến trúc phân loại cho cùng tập dữ liệu theo các hướng khác nhau,
ví dụ:
• Kết hợp CNN và Transformer: ví dụ dùng CNN làm backbone trích đặc
trưng, sau đó chuỗi đặc trưng đưa vào Transformer; hoặc ngược lại.
• Các cách tokenizer và embed ảnh khác nhau: patch theo ô lưới; patch có
overlap; patch kích thước khác nhau.
• Transformer với chiều đặc trưng khác nhau:
– Coi chiều không gian H ×W như các vector (mỗi vị trí spatial là một token).
– Coi chiều kênh C như các vector (mỗi “token” tương ứng với kênh).
– Kết hợp hoặc biến thể tương tự (ví dụ: flatten theo hàng/cột, v.v.).
Sinh viên chọn ít nhất hai hoặc ba cách khác nhau để triển khai và so sánh.
2. Huấn luyện, đánh giá và so sánh các kiểu xây dựng mô hình (bảng số liệu, biểu đồ,
nhận xét).
3.5 Phần 5 — Mô hình phân loại dựa trên LSTM/GRU
1. Xây dựng mô hình phân loại dựa trên LSTM hoặc GRU cho cùng tập dữ liệu ảnh.
Cần tự chọn cách biểu diễn ảnh thành chuỗi (sequence), ví dụ:
• Mỗi hàng của ảnh là một bước thời gian (sequence length = số hàng).
• Mỗi cột của ảnh là một bước thời gian.
• Dãy các block/patch (chia ảnh thành các ô, mỗi ô flatten thành vector, thứ tự
theo hàng hoặc theo cột).
• Cách khác (cần nêu rõ trong báo cáo).
2. Huấn luyện, đánh giá và so sánh ít nhất hai cách biểu diễn chuỗi (hoặc so sánh
LSTM vs GRU với cùng cách biểu diễn). Trình bày kết quả khoa học và nhận xét.
4 Tiêu chí chấm điểm
• Phần 1 (mô hình): Đủ bốn mô hình (softmax, MLP, CNN, ViT dùng khối có sẵn),
code rõ ràng, có thể chạy được.
4
• Phần 2 (huấn luyện và so sánh): Training loop tự viết đúng (forward, loss, backward, optimizer step); có đánh giá và so sánh bốn mô hình; trình bày kết quả khoa
học và rõ ràng (bảng, biểu đồ, nhận xét).
• Phần 3 (TransformerEncoder tự xây): Hiện thực đúng khối encoder từ phép toán
cơ bản; ViT dùng khối tự xây chạy được; so sánh với ViT dùng khối có sẵn.
• Phần 4 (kiến trúc đa dạng): Có ít nhất hai (hoặc ba) cách xây dựng khác nhau
(CNN+Transformer, tokenizer/embed khác nhau, chiều H*W vs C); huấn luyện, đánh
giá và so sánh; trình bày rõ ràng.
• Phần 5 (LSTM/GRU): Mô hình LSTM/GRU với ít nhất hai cách biểu diễn chuỗi
(hoặc LSTM vs GRU); huấn luyện, đánh giá và so sánh; trình bày rõ ràng.
• Chất lượng báo cáo: Cấu trúc logic, bảng số liệu và biểu đồ đẹp, dễ đọc; nhận xét
và phân tích ngắn gọn nhưng có nội dung. Trình bày khoa học và đẹp được tính
vào điểm.
Phân bổ điểm chi tiết có thể do giảng viên quy định thêm (ví dụ theo tỉ lệ phần 1–5
và chất lượng báo cáo).
5 Yêu cầu nộp và hạn nộp
5.1 Sản phẩm nộp
• Báo cáo (PDF): mô tả ngắn tập dữ liệu, kiến trúc từng mô hình (có thể kèm sơ đồ),
quy trình huấn luyện, bảng và biểu đồ kết quả, so sánh và nhận xét cho từng phần.
Trình bày khoa học, rõ ràng.
• Mã nguồn (Python/PyTorch): tổ chức rõ ràng (theo phần hoặc theo mô hình), có
README hướng dẫn chạy (môi trường, lệnh train/eval). Nộp qua LMS (file nén) hoặc
link repository (GitHub, v.v.) theo hướng dẫn của giảng viên.
5.2 Hình thức thực hiện
Thực hiện theo nhóm đã chọn cho bài tập lớn số 1.
5.3 Hạn nộp
• Ngày nộp: 01/04/2026)
• Nộp trễ: Theo quy định của môn học (ví dụ: mỗi tuần trễ trừ 20% điểm).
5