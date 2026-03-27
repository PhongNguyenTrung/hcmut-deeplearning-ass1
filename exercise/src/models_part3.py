"""
models_part3.py — Tự hiện thực TransformerEncoder từ đầu (Phần 3)

YÊU CẦU: Chỉ được dùng nn.Linear, nn.LayerNorm, và torch.einsum.
KHÔNG được dùng nn.TransformerEncoderLayer hay nn.TransformerEncoder.

Các class được xây dựng:
1. CustomMultiHeadAttention   — Self-attention từ đầu với einsum
2. CustomTransformerEncoderLayer — Encoder block với Pre-LN
3. CustomTransformerEncoder   — Stack nhiều encoder layers
4. CustomViT                  — ViT dùng encoder tự xây (so sánh với SimpleViT)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ─── 1. Custom Multi-Head Self-Attention ──────────────────────────────────────

class CustomMultiHeadAttention(nn.Module):
    """
    Multi-Head Self-Attention tự hiện thực.

    Công thức cơ bản (1 head):
        Q = X @ W_q    (Queries)
        K = X @ W_k    (Keys)
        V = X @ W_v    (Values)
        scores = Q @ K^T / sqrt(d_head)
        attn = softmax(scores, dim=-1)
        output = attn @ V

    Multi-head: chia d_model thành num_heads heads, mỗi head tính attention riêng,
    sau đó ghép lại và project qua W_o.

    Tại sao dùng einsum?
    - Rõ ràng về chiều dữ liệu
    - 'bhid,bhjd->bhij' nghĩa là: với mỗi batch (b) và head (h),
      tính dot product giữa query tại vị trí i và key tại vị trí j
    """

    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        assert d_model % num_heads == 0, "d_model phải chia hết cho num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_head = d_model // num_heads  # chiều mỗi head

        # 4 linear projections (chỉ dùng nn.Linear như yêu cầu)
        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.d_head)  # hệ số chia để tránh gradient vanishing

    def forward(self, x):
        """
        Args:
            x: [B, T, d_model]  (B=batch, T=sequence length)
        Returns:
            out: [B, T, d_model]
        """
        B, T, _ = x.shape

        # ── Bước 1: Project thành Q, K, V ──
        # Mỗi tensor: [B, T, d_model]
        Q = self.W_q(x)
        K = self.W_k(x)
        V = self.W_v(x)

        # ── Bước 2: Chia thành nhiều heads ──
        # [B, T, d_model] → [B, T, num_heads, d_head] → [B, num_heads, T, d_head]
        Q = Q.reshape(B, T, self.num_heads, self.d_head).transpose(1, 2)
        K = K.reshape(B, T, self.num_heads, self.d_head).transpose(1, 2)
        V = V.reshape(B, T, self.num_heads, self.d_head).transpose(1, 2)

        # ── Bước 3: Tính attention scores bằng einsum ──
        # 'b h i d, b h j d -> b h i j' nghĩa là:
        # Với batch b, head h: tính dot product giữa query[i] và key[j]
        # Kết quả: [B, num_heads, T, T] — ma trận attention
        scores = torch.einsum('bhid,bhjd->bhij', Q, K) / self.scale

        # ── Bước 4: Softmax để được attention weights ──
        # Mỗi hàng của attention matrix tổng = 1.0
        attn = F.softmax(scores, dim=-1)  # [B, num_heads, T, T]
        attn = self.dropout(attn)

        # ── Bước 5: Weighted sum của Values bằng einsum ──
        # 'b h i j, b h j d -> b h i d' nghĩa là:
        # Token i nhận được: tổng có trọng số (attention weights) của tất cả values
        out = torch.einsum('bhij,bhjd->bhid', attn, V)  # [B, num_heads, T, d_head]

        # ── Bước 6: Ghép các heads lại ──
        # [B, num_heads, T, d_head] → [B, T, num_heads, d_head] → [B, T, d_model]
        out = out.transpose(1, 2).contiguous().reshape(B, T, self.d_model)

        # ── Bước 7: Output projection ──
        return self.W_o(out)


# ─── 2. Custom Transformer Encoder Layer ──────────────────────────────────────

class CustomTransformerEncoderLayer(nn.Module):
    """
    Một block của Transformer Encoder tự xây.

    Dùng kiến trúc Pre-LN (Layer Norm TRƯỚC attention/FFN):
        x = x + Attention(LayerNorm(x))
        x = x + FFN(LayerNorm(x))

    Tại sao Pre-LN tốt hơn Post-LN khi train từ đầu?
    - Post-LN (paper gốc "Attention is All You Need"): LN sau residual
      → gradient thường lớn hơn → cần learning rate warmup cẩn thận
    - Pre-LN: LN trước attention → gradient ổn định hơn → dễ train hơn
    """

    def __init__(self, d_model, num_heads, ffn_dim=None, dropout=0.1):
        super().__init__()
        if ffn_dim is None:
            ffn_dim = d_model * 4  # FFN thường rộng hơn 4× so với d_model

        # Pre-LN norms
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        # Self-attention (tự xây)
        self.attn = CustomMultiHeadAttention(d_model, num_heads, dropout)

        # Feed-Forward Network (FFN)
        # Chỉ dùng nn.Linear như yêu cầu bài tập
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ffn_dim),
            nn.GELU(),             # Activation mượt mà hơn ReLU, phổ biến trong Transformer
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        """
        Args/Returns: [B, T, d_model]

        Luồng dữ liệu:
            x → LayerNorm → Attention → (+x) → LayerNorm → FFN → (+x)
        """
        # Residual connection 1: Self-Attention với Pre-LN
        x = x + self.attn(self.norm1(x))

        # Residual connection 2: FFN với Pre-LN
        x = x + self.ffn(self.norm2(x))

        return x


# ─── 3. Custom Transformer Encoder (Stack) ────────────────────────────────────

class CustomTransformerEncoder(nn.Module):
    """
    Stack nhiều CustomTransformerEncoderLayer.
    Tương đương với nn.TransformerEncoder nhưng tự xây.
    """

    def __init__(self, d_model, num_heads, num_layers, ffn_dim=None, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            CustomTransformerEncoderLayer(d_model, num_heads, ffn_dim, dropout)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(d_model)  # Final norm sau tất cả layers

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return self.norm(x)


# ─── 4. Custom ViT ────────────────────────────────────────────────────────────

class CustomViT(nn.Module):
    """
    Vision Transformer dùng CustomTransformerEncoder (tự xây từ đầu).

    Kiến trúc GIỐNG HỆT SimpleViT (Part 1) nhưng thay:
    - nn.TransformerEncoder → CustomTransformerEncoder
    - nn.TransformerEncoderLayer → CustomTransformerEncoderLayer

    Mục đích: So sánh kết quả với SimpleViT để kiểm tra hiện thực đúng.
    Accuracy hai mô hình phải gần nhau (sai lệch nhỏ do khởi tạo khác nhau).
    """

    def __init__(self, num_classes=100, d_model=128, nhead=4, num_layers=4,
                 patch_size=4, dropout=0.1):
        super().__init__()
        assert 32 % patch_size == 0

        self.patch_size = patch_size
        num_patches = (32 // patch_size) ** 2
        self.d_model = d_model

        # Patch embedding (giống SimpleViT)
        self.patch_embed = nn.Conv2d(3, d_model, kernel_size=patch_size, stride=patch_size)

        # CLS token và Positional encoding (giống SimpleViT)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, d_model))

        # ── Phần khác biệt: dùng CustomTransformerEncoder ──
        self.transformer = CustomTransformerEncoder(
            d_model=d_model,
            num_heads=nhead,
            num_layers=num_layers,
            dropout=dropout,
        )

        # Head (giống SimpleViT)
        self.head = nn.Linear(d_model, num_classes)

        # Khởi tạo
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, x):
        B = x.shape[0]

        # Patch embedding: [B, 3, 32, 32] → [B, 64, d_model]
        x = self.patch_embed(x).flatten(2).transpose(1, 2)

        # Thêm CLS token: [B, 65, d_model]
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)

        # Positional encoding
        x = x + self.pos_embed

        # Transformer (tự xây): [B, 65, d_model]
        x = self.transformer(x)

        # Classification từ CLS token
        # Lưu ý: CustomTransformerEncoder đã có LayerNorm ở cuối,
        # nên không cần thêm norm ở đây
        return self.head(x[:, 0, :])
