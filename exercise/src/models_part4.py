"""
models_part4.py — Các kiến trúc kết hợp và embed ảnh khác nhau (Phần 4)

Ba cách tokenize ảnh khác nhau cho Transformer:

A. CNNTransformerHybrid:
   - CNN backbone trích xuất feature maps
   - Các feature maps → chuỗi tokens cho Transformer
   - Lợi thế: CNN học đặc trưng cục bộ, Transformer học phụ thuộc xa

B. SpatialTokenViT (H×W positions làm tokens):
   - Mỗi vị trí pixel (hoặc nhóm pixels) là 1 token
   - Feature vector của token = giá trị pixel tại vị trí đó
   - Seq length = 1024 (32×32) → attention O(1024²) — ĐẮT!
   - Quan trọng để hiểu tại sao ViT dùng patches thay vì pixels

C. ChannelTokenViT (C channels làm tokens):
   - Mỗi feature channel là 1 token
   - Feature vector = giá trị của channel đó tại mọi vị trí spatial
   - Liên quan đến ý tưởng "channel attention" trong SENet
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ─── A. CNN + Transformer Hybrid ──────────────────────────────────────────────

class CNNTransformerHybrid(nn.Module):
    """
    Kết hợp CNN (backbone) + Transformer (encoder).

    Luồng dữ liệu:
        [B, 3, 32, 32]
        → CNN Block 1: Conv+BN+ReLU + MaxPool → [B, 32, 16, 16]
        → CNN Block 2: Conv+BN+ReLU + MaxPool → [B, 64, 8, 8]
        → Reshape: [B, 64, 64] (64 spatial positions, each 64-dim)
        → Linear projection: [B, 64, 128]
        → CLS token: [B, 65, 128]
        → Positional Encoding: [B, 65, 128]
        → Transformer Encoder (2 layers): [B, 65, 128]
        → CLS output → Linear → [B, 100]

    Tại sao hybrid tốt?
    - CNN xử lý đặc trưng cục bộ hiệu quả (translation invariance)
    - Transformer xử lý quan hệ toàn cục giữa các vùng ảnh
    - Tốt hơn CNN thuần ở việc nắm bắt context toàn cảnh
    - Ít tốn bộ nhớ hơn ViT thuần (CNN giảm spatial size trước)
    """

    def __init__(self, num_classes=100, d_model=128, nhead=4, num_transformer_layers=2, dropout=0.1):
        super().__init__()

        # ── CNN Backbone ──
        def cnn_block(in_c, out_c):
            return nn.Sequential(
                nn.Conv2d(in_c, out_c, 3, padding=1),
                nn.BatchNorm2d(out_c),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
            )

        self.cnn = nn.Sequential(
            cnn_block(3, 32),   # [B, 3, 32, 32] → [B, 32, 16, 16]
            cnn_block(32, 64),  # → [B, 64, 8, 8]
        )

        # ── Token Projection ──
        # Mỗi spatial position (8×8=64 vị trí) với 64 features → project lên d_model
        self.token_proj = nn.Linear(64, d_model)  # 64 features từ CNN → d_model

        # Số tokens = 8×8 = 64 spatial positions
        num_tokens = 8 * 8  # = 64

        # ── CLS token & Positional Encoding ──
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_tokens + 1, d_model))
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        # ── Transformer Encoder ──
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=d_model * 4, dropout=dropout,
            batch_first=True, norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_transformer_layers)

        # ── Classification Head ──
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, num_classes)

    def forward(self, x):
        B = x.shape[0]

        # CNN: [B, 3, 32, 32] → [B, 64, 8, 8]
        x = self.cnn(x)

        # Reshape: [B, 64, 8, 8] → [B, 64, 64] (64 tokens, mỗi token 64-dim)
        B, C, H, W = x.shape
        x = x.reshape(B, C, H * W).permute(0, 2, 1)  # [B, H*W, C]

        # Project: [B, 64, 64] → [B, 64, d_model]
        x = self.token_proj(x)

        # Thêm CLS token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)  # [B, 65, d_model]

        # Positional encoding
        x = x + self.pos_embed

        # Transformer
        x = self.transformer(x)

        # Classification
        return self.head(self.norm(x[:, 0, :]))


# ─── B. Spatial Token ViT (H×W positions làm tokens) ─────────────────────────

class SpatialTokenViT(nn.Module):
    """
    Mỗi trong 32×32 = 1024 vị trí spatial là 1 token.

    Đây là cách "tự nhiên" nhất để tokenize ảnh:
    "Mỗi pixel là 1 từ trong câu"

    NHƯNG: 1024 tokens → attention matrix [B, H, 1024, 1024]
    → Với B=32, H=4: 32 × 4 × 1024 × 1024 × 4 bytes ≈ 512MB!
    → Phải dùng batch_size nhỏ (32 thay vì 128)

    Bài học: Đây là lý do ViT dùng PATCHES thay vì pixels!

    Kiến trúc:
        [B, 3, 32, 32] → reshape [B, 1024, 3]
        → Linear(3, 64) → [B, 1024, 64]
        → Positional Encoding
        → 2× TransformerEncoderLayer (d=64, heads=4)
        → Global Average Pool → Linear(64, 100)
    """

    def __init__(self, num_classes=100, d_model=64, nhead=4, num_layers=2, dropout=0.1):
        super().__init__()

        num_tokens = 32 * 32  # = 1024

        # Project từng pixel: 3 kênh màu → d_model features
        self.pixel_proj = nn.Linear(3, d_model)

        # Positional encoding cho 1024 vị trí
        self.pos_embed = nn.Parameter(torch.zeros(1, num_tokens, d_model))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        # Transformer (số layers ít để giảm thời gian train)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=d_model * 4, dropout=dropout,
            batch_first=True, norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Global average pooling + head (không dùng CLS token — tiết kiệm 1 vị trí)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, num_classes)

    def forward(self, x):
        B, C, H, W = x.shape  # [B, 3, 32, 32]

        # Reshape: [B, 3, 32, 32] → [B, 1024, 3]
        x = x.reshape(B, C, H * W).permute(0, 2, 1)

        # Project: [B, 1024, 3] → [B, 1024, d_model]
        x = self.pixel_proj(x)

        # Positional encoding
        x = x + self.pos_embed

        # Transformer
        x = self.transformer(x)  # [B, 1024, d_model]

        # Global average pool: lấy trung bình qua tất cả tokens
        x = x.mean(dim=1)  # [B, d_model]

        return self.head(self.norm(x))


# ─── C. Channel Token ViT (C channels làm tokens) ────────────────────────────

class ChannelTokenViT(nn.Module):
    """
    Mỗi channel là 1 token, spatial features là feature vector của token đó.

    Ý tưởng: "Thay vì hỏi 'vị trí này trông như thế nào?',
              ta hỏi 'kênh đặc trưng này có mặt ở đâu?'"

    Liên quan đến:
    - SENet (Squeeze-and-Excitation Networks): channel attention
    - MLP-Mixer: dùng cả spatial mixing và channel mixing

    Kiến trúc:
        [B, 3, 32, 32]
        → Conv2d(3→64, 1×1): expand channels → [B, 64, 32, 32]
        → reshape [B, 64, 1024]: 64 channel tokens, mỗi token 1024-dim (spatial)
        → Linear(1024→128): project xuống d_model
        → Positional Encoding cho 64 tokens
        → 2× TransformerEncoderLayer (d=128, heads=4)
        → Global avg pool → Linear(128, 100)

    Tại sao expand channels với Conv 1×1?
    - CIFAR-100 chỉ có 3 channels → quá ít tokens (chỉ 3!)
    - Conv 1×1 học các "feature channels" có ý nghĩa từ 3 kênh màu gốc
    """

    def __init__(self, num_classes=100, num_channels=64, d_model=128,
                 nhead=4, num_layers=2, dropout=0.1):
        super().__init__()

        spatial_dim = 32 * 32  # = 1024

        # Expand từ 3 channels lên num_channels với Conv 1×1
        self.channel_expand = nn.Sequential(
            nn.Conv2d(3, num_channels, kernel_size=1),
            nn.BatchNorm2d(num_channels),
            nn.ReLU(inplace=True),
        )

        # Project spatial features xuống d_model
        self.spatial_proj = nn.Linear(spatial_dim, d_model)

        # Positional encoding cho num_channels tokens
        self.pos_embed = nn.Parameter(torch.zeros(1, num_channels, d_model))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        # Transformer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=d_model * 4, dropout=dropout,
            batch_first=True, norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Head
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, num_classes)

    def forward(self, x):
        B = x.shape[0]

        # Expand channels: [B, 3, 32, 32] → [B, 64, 32, 32]
        x = self.channel_expand(x)

        # Reshape: [B, 64, 32, 32] → [B, 64, 1024]
        # Mỗi trong 64 channels trở thành 1 token với 1024-dim feature vector
        B, C, H, W = x.shape
        x = x.reshape(B, C, H * W)  # [B, 64, 1024]

        # Project: [B, 64, 1024] → [B, 64, d_model]
        x = self.spatial_proj(x)

        # Positional encoding
        x = x + self.pos_embed

        # Transformer
        x = self.transformer(x)  # [B, 64, d_model]

        # Global average pool qua channels
        x = x.mean(dim=1)  # [B, d_model]

        return self.head(self.norm(x))
