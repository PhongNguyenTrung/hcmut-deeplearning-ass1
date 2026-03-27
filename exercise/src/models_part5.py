"""
models_part5.py — Mô hình phân loại ảnh dựa trên LSTM/GRU (Phần 5)

Ý tưởng: Ảnh không phải chuỗi thời gian, NHƯNG ta có thể "đọc" ảnh
như đọc văn bản — theo hàng, theo cột, hoặc theo từng patch.

Các mô hình:
1. ImageLSTM — LSTM bidirectional
2. ImageGRU  — GRU bidirectional (đơn giản hơn LSTM, thường tương đương)

Input: Tensor hình [B, T, input_size] (sequence đã được tạo bởi SequenceTransform)
Output: Logits [B, num_classes]

So sánh LSTM vs GRU:
┌──────────┬───────────────────────────────────┬──────────────┐
│          │ Cơ chế                            │ Params/layer │
├──────────┼───────────────────────────────────┼──────────────┤
│ LSTM     │ 4 gates: forget, input, output,   │ 4×(D+H)×H    │
│          │ cell gate                         │              │
├──────────┼───────────────────────────────────┼──────────────┤
│ GRU      │ 2 gates: reset, update            │ 3×(D+H)×H    │
│          │ (đơn giản hơn 33%)                │              │
└──────────┴───────────────────────────────────┴──────────────┘
D = input_size, H = hidden_size
"""

import torch
import torch.nn as nn


# ─── Chung cho cả LSTM và GRU ─────────────────────────────────────────────────

class _RNNBase(nn.Module):
    """
    Base class chung cho ImageLSTM và ImageGRU.
    Không instantiate trực tiếp — dùng ImageLSTM hoặc ImageGRU.
    """

    def __init__(self, rnn_type, input_size, hidden_size=256, num_layers=2,
                 num_classes=100, dropout=0.3):
        super().__init__()
        assert rnn_type in ("LSTM", "GRU"), "rnn_type phải là 'LSTM' hoặc 'GRU'"
        self.rnn_type = rnn_type
        self.hidden_size = hidden_size

        # Tạo RNN layer
        # batch_first=True: input shape [B, T, D] (không phải [T, B, D])
        # bidirectional=True: đọc chuỗi theo cả 2 chiều
        #   → output hidden state có kích thước gấp đôi
        RNN = nn.LSTM if rnn_type == "LSTM" else nn.GRU
        self.rnn = RNN(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0,  # dropout không áp dụng cho single-layer
        )

        # Sau bidirectional RNN: hidden state có size hidden_size * 2
        # (forward pass + backward pass)
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size * 2, num_classes),
        )

    def _get_last_hidden(self, hidden):
        """
        Lấy hidden state của bước thời gian cuối cùng từ bidirectional RNN.

        Với bidirectional RNN, hidden state có shape:
        - LSTM: hidden = (h_n, c_n) với h_n shape [num_layers*2, B, hidden_size]
        - GRU:  hidden = h_n với shape [num_layers*2, B, hidden_size]

        Ta lấy layer cuối cùng (forward + backward):
        - h_n[-2]: forward direction, last layer
        - h_n[-1]: backward direction, last layer
        - Ghép lại: [B, hidden_size * 2]
        """
        if self.rnn_type == "LSTM":
            h_n, c_n = hidden   # LSTM trả về tuple (h_n, c_n)
        else:
            h_n = hidden        # GRU chỉ trả về h_n

        # h_n shape: [num_layers * num_directions, B, hidden_size]
        # Lấy forward (index -2) và backward (index -1) của layer cuối
        last_forward = h_n[-2]   # [B, hidden_size]
        last_backward = h_n[-1]  # [B, hidden_size]

        # Ghép: [B, hidden_size * 2]
        return torch.cat([last_forward, last_backward], dim=1)

    def forward(self, x):
        """
        Args:
            x: [B, T, input_size] — chuỗi đã được tạo bởi SequenceTransform
        Returns:
            logits: [B, num_classes]
        """
        # RNN xử lý toàn bộ chuỗi
        output, hidden = self.rnn(x)
        # output: [B, T, hidden_size*2] — output tại mỗi bước
        # hidden: h_n (và c_n nếu LSTM)

        # Lấy hidden state cuối cùng
        last_hidden = self._get_last_hidden(hidden)  # [B, hidden_size*2]

        return self.head(last_hidden)  # [B, num_classes]


class ImageLSTM(_RNNBase):
    """
    Bidirectional LSTM cho phân loại ảnh.

    LSTM (Long Short-Term Memory) có 4 gates:
    - Forget gate: "Quên" thông tin cũ không liên quan
    - Input gate: Thông tin mới nào được lưu vào memory
    - Output gate: Thông tin nào được đưa ra từ memory
    - Cell gate: Giá trị mới cho memory cell

    Ưu điểm: Xử lý tốt các phụ thuộc dài (long-range dependencies)
    Nhược điểm: Nhiều tham số hơn GRU (~33%)

    Args:
        input_size: Kích thước feature tại mỗi bước (ví dụ: 96 cho "row" mode)
        hidden_size: Số hidden units mỗi direction
        num_layers: Số lớp LSTM xếp chồng
        num_classes: Số lớp phân loại (100 cho CIFAR-100)
    """

    def __init__(self, input_size, hidden_size=256, num_layers=2, num_classes=100, dropout=0.3):
        super().__init__("LSTM", input_size, hidden_size, num_layers, num_classes, dropout)


class ImageGRU(_RNNBase):
    """
    Bidirectional GRU cho phân loại ảnh.

    GRU (Gated Recurrent Unit) có 2 gates:
    - Reset gate: "Quên" bao nhiêu hidden state cũ
    - Update gate: Tỉ lệ kết hợp giữa hidden state cũ và mới

    Ưu điểm: Ít tham số hơn LSTM (~33%), train nhanh hơn
    Thường đạt accuracy tương đương LSTM trên nhiều tasks

    Args: Tương tự ImageLSTM
    """

    def __init__(self, input_size, hidden_size=256, num_layers=2, num_classes=100, dropout=0.3):
        super().__init__("GRU", input_size, hidden_size, num_layers, num_classes, dropout)
