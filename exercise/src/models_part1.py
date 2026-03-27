"""
models_part1.py — 4 mô hình phân loại ảnh cơ bản (Phần 1)

Tất cả mô hình:
- Nhận input: [B, 3, 32, 32] (batch, channels, height, width)
- Trả về: logits [B, num_classes]
- KHÔNG có explicit softmax trong forward() vì CrossEntropyLoss tự tính softmax

Các mô hình được xây dựng từ đơn giản đến phức tạp:
1. SoftmaxRegression — mô hình tuyến tính đơn giản nhất
2. MLP             — mạng fully connected với hidden layers
3. SimpleCNN       — mạng tích chập VGG-style
4. SimpleViT       — Vision Transformer dùng khối có sẵn của PyTorch
"""

import torch
import torch.nn as nn
import math


# ─── 1. Softmax Regression ────────────────────────────────────────────────────

class SoftmaxRegression(nn.Module):
    """
    Mô hình tuyến tính đơn giản nhất.

    Công thức: logits = W * flatten(x) + b
    Trong đó: W có shape [num_classes, 3*32*32] = [100, 3072]

    Ưu điểm: Đơn giản, train nhanh
    Nhược điểm: Không học được đặc trưng phi tuyến, accuracy thấp (~15-20% trên CIFAR-100)
    """

    def __init__(self, num_classes=100):
        super().__init__()
        input_dim = 3 * 32 * 32  # = 3072

        self.net = nn.Sequential(
            nn.Flatten(),                      # [B, 3, 32, 32] → [B, 3072]
            nn.Linear(input_dim, num_classes), # [B, 3072] → [B, 100]
        )

    def forward(self, x):
        return self.net(x)


# ─── 2. MLP (Multi-Layer Perceptron) ─────────────────────────────────────────

class MLP(nn.Module):
    """
    Mạng fully connected nhiều lớp ẩn.

    Kiến trúc:
        Flatten → Linear(3072→512) → BN → ReLU → Dropout
                → Linear(512→256) → BN → ReLU → Dropout
                → Linear(256→100)

    Ưu điểm hơn Softmax Regression: Học được đặc trưng phi tuyến
    Nhược điểm: Flatten làm mất thông tin không gian (spatial)
    → CNN sẽ giải quyết vấn đề này
    """

    def __init__(self, num_classes=100, dropout=0.3):
        super().__init__()
        input_dim = 3 * 32 * 32  # = 3072

        self.net = nn.Sequential(
            nn.Flatten(),

            # Lớp ẩn 1: 3072 → 512
            nn.Linear(input_dim, 512),
            nn.BatchNorm1d(512),   # Chuẩn hoá để training ổn định hơn
            nn.ReLU(),
            nn.Dropout(dropout),   # Tắt ngẫu nhiên 30% neurons để tránh overfitting

            # Lớp ẩn 2: 512 → 256
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout),

            # Lớp output: 256 → 100
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        return self.net(x)


# ─── 3. SimpleCNN ─────────────────────────────────────────────────────────────

class SimpleCNN(nn.Module):
    """
    Mạng tích chập (VGG-style) với 3 ConvBlocks.

    Kiến trúc:
        ConvBlock1: Conv(3→32) × 2 + BN + ReLU + MaxPool → [B, 32, 16, 16]
        ConvBlock2: Conv(32→64) × 2 + BN + ReLU + MaxPool → [B, 64, 8, 8]
        ConvBlock3: Conv(64→128) × 2 + BN + ReLU + MaxPool → [B, 128, 4, 4]
        Classifier: AdaptiveAvgPool(1) → Flatten → Linear(128→256) → ReLU → Linear(256→100)

    Tại sao CNN tốt hơn MLP?
    - Tích chập học đặc trưng cục bộ (local features)
    - Weight sharing: cùng bộ lọc áp dụng cho mọi vị trí
    - Translation invariance: nhận diện được đặc trưng bất kể vị trí

    AdaptiveAvgPool(1): Thay vì Flatten(128*4*4=2048) rồi Linear lớn,
    ta pool về [B, 128, 1, 1] → giảm params và ít overfitting hơn.
    """

    def __init__(self, num_classes=100, dropout=0.3):
        super().__init__()

        def conv_block(in_c, out_c):
            return nn.Sequential(
                nn.Conv2d(in_c, out_c, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_c),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_c, out_c, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_c),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),  # Giảm spatial size xuống 1/2
            )

        self.features = nn.Sequential(
            conv_block(3, 32),    # [B, 3, 32, 32] → [B, 32, 16, 16]
            conv_block(32, 64),   # → [B, 64, 8, 8]
            conv_block(64, 128),  # → [B, 128, 4, 4]
        )

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),  # [B, 128, 4, 4] → [B, 128, 1, 1]
            nn.Flatten(),             # → [B, 128]
            nn.Linear(128, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)


# ─── 4. SimpleViT (Vision Transformer dùng PyTorch có sẵn) ───────────────────

class SimpleViT(nn.Module):
    """
    Vision Transformer (ViT) đơn giản dùng nn.TransformerEncoder của PyTorch.

    Ý tưởng ViT: Chia ảnh thành các "patches" (mảnh nhỏ),
    sau đó xử lý như một chuỗi tokens, giống như Transformer trong NLP.

    Kiến trúc:
    1. Patch Embedding: chia 32×32 thành các patch 4×4
       → Conv2d(3, d_model, 4, 4) tương đương chia patch + linear projection
       → Output: [B, 64, d_model] (64 patches)

    2. CLS Token: Thêm 1 token đặc biệt [CLS] ở đầu chuỗi
       → [B, 65, d_model]
       Token này sẽ "tổng hợp" thông tin từ toàn bộ ảnh

    3. Positional Encoding: Thêm thông tin về vị trí của mỗi patch
       → Vì Transformer không biết thứ tự, cần encoding này

    4. Transformer Encoder: Xử lý chuỗi với self-attention
       → Mỗi patch "chú ý" đến tất cả patches khác

    5. Classification Head: Lấy [CLS] token → Linear → predictions

    Tại sao patch_size=4 cho CIFAR-100 32×32?
    - patch_size=16 (như ViT-B/16) → chỉ 4 patches — quá ít thông tin
    - patch_size=4 → 64 patches — cân bằng tốt giữa số lượng và kích thước
    - patch_size=2 → 256 patches — quá nhiều, khó train từ đầu
    """

    def __init__(self, num_classes=100, d_model=128, nhead=4, num_layers=4,
                 patch_size=4, dropout=0.1):
        super().__init__()
        assert 32 % patch_size == 0, "Image size phải chia hết cho patch_size"

        self.patch_size = patch_size
        num_patches = (32 // patch_size) ** 2  # = 64 cho patch_size=4
        self.d_model = d_model

        # ── Patch Embedding ──
        # Conv2d với kernel_size=patch_size và stride=patch_size
        # → mỗi "pixel" trong output tương ứng 1 patch trong input
        self.patch_embed = nn.Conv2d(3, d_model, kernel_size=patch_size, stride=patch_size)

        # ── CLS Token ──
        # nn.Parameter: tham số được học (không phải hằng số)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))

        # ── Positional Encoding ──
        # +1 vì có CLS token
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, d_model))

        # ── Transformer Encoder (dùng PyTorch có sẵn — Part 3 sẽ tự xây) ──
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,  # FFN dimension thường = 4 × d_model
            dropout=dropout,
            batch_first=True,   # Input format: [batch, seq, features] (không phải [seq, batch, features])
            norm_first=True,    # Pre-LN (ổn định hơn Post-LN khi train từ đầu)
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # ── Classification Head ──
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, num_classes)

        # Khởi tạo weights tốt hơn
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, x):
        B = x.shape[0]

        # ── Bước 1: Patch Embedding ──
        # [B, 3, 32, 32] → [B, d_model, 8, 8]
        x = self.patch_embed(x)
        # [B, d_model, 8, 8] → [B, 64, d_model]
        x = x.flatten(2).transpose(1, 2)  # flatten spatial dims, swap axes

        # ── Bước 2: Thêm CLS token ──
        # cls_token: [1, 1, d_model] → expand → [B, 1, d_model]
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)  # [B, 65, d_model]

        # ── Bước 3: Thêm Positional Encoding ──
        x = x + self.pos_embed

        # ── Bước 4: Transformer Encoder ──
        x = self.transformer(x)  # [B, 65, d_model]

        # ── Bước 5: Classification ──
        # Lấy CLS token (index 0) → đại diện cho toàn bộ ảnh
        cls_out = self.norm(x[:, 0, :])  # [B, d_model]
        return self.head(cls_out)         # [B, num_classes]
