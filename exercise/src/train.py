"""
train.py — Vòng lặp huấn luyện tự viết (Custom Training Loop)

Đây là phần QUAN TRỌNG nhất của bài tập (Phần 2).
Thay vì dùng API cấp cao như trainer.fit() của Lightning,
ta tự viết từng bước:

    for epoch in range(epochs):
        for x, y in train_loader:
            optimizer.zero_grad()   # 1. Xoá gradient cũ
            logits = model(x)       # 2. Forward pass
            loss = criterion(logits, y)  # 3. Tính loss
            loss.backward()         # 4. Backpropagation (tính gradient)
            clip_grad_norm_(...)    # 5. Clip gradient (tránh exploding gradients)
            optimizer.step()        # 6. Cập nhật tham số
"""

import os
import time
import json
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR


def train_one_epoch(model, loader, optimizer, criterion, device, clip_norm=1.0):
    """
    Huấn luyện model qua 1 epoch đầy đủ.

    Args:
        model: nn.Module cần huấn luyện
        loader: DataLoader của tập train
        optimizer: SGD, Adam, AdamW, ...
        criterion: Hàm loss (thường là nn.CrossEntropyLoss)
        device: "cuda", "mps", hoặc "cpu"
        clip_norm: Giới hạn norm của gradient (tránh gradient exploding)

    Returns:
        avg_loss: Loss trung bình trên epoch
        accuracy: Độ chính xác trên epoch (0.0 → 1.0)
    """
    model.train()  # Bật dropout, batch norm ở mode train

    total_loss = 0.0
    correct = 0
    total = 0

    for x, y in loader:
        x, y = x.to(device), y.to(device)

        # ── Bước 1: Xoá gradient từ iteration trước ──
        # Quan trọng! PyTorch CỘNG DỒN gradient, nên phải xoá trước mỗi batch.
        optimizer.zero_grad()

        # ── Bước 2: Forward pass ──
        logits = model(x)  # shape: [batch, num_classes]

        # ── Bước 3: Tính loss ──
        # CrossEntropyLoss = log_softmax + NLLLoss
        # Nhận vào logits (chưa qua softmax), tự tính softmax bên trong.
        loss = criterion(logits, y)

        # ── Bước 4: Backward pass (tính đạo hàm) ──
        loss.backward()

        # ── Bước 5: Gradient clipping ──
        # Nếu gradient quá lớn, norm của vector gradient sẽ được scale xuống clip_norm.
        # Điều này giúp ổn định huấn luyện, đặc biệt với RNN và Transformer.
        nn.utils.clip_grad_norm_(model.parameters(), clip_norm)

        # ── Bước 6: Cập nhật tham số ──
        optimizer.step()

        # Theo dõi metrics
        total_loss += loss.item() * x.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == y).sum().item()
        total += x.size(0)

    avg_loss = total_loss / total
    accuracy = correct / total
    return avg_loss, accuracy


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    """
    Đánh giá model trên tập val hoặc test (không tính gradient).

    @torch.no_grad() tắt việc tính gradient → nhanh hơn và tiết kiệm RAM.

    Returns:
        avg_loss: Loss trung bình
        accuracy: Độ chính xác (0.0 → 1.0)
    """
    model.eval()  # Tắt dropout, batch norm dùng running stats

    total_loss = 0.0
    correct = 0
    total = 0

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = criterion(logits, y)

        total_loss += loss.item() * x.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == y).sum().item()
        total += x.size(0)

    return total_loss / total, correct / total


def fit(model, train_loader, val_loader, config):
    """
    Vòng lặp huấn luyện đầy đủ qua nhiều epochs.

    Args:
        model: nn.Module
        train_loader, val_loader: DataLoader
        config (dict): Cấu hình huấn luyện với các keys:
            - epochs (int): Số epochs
            - lr (float): Learning rate
            - weight_decay (float): L2 regularization (mặc định 1e-4)
            - device (str): "cuda"/"mps"/"cpu"
            - save_path (str): Đường dẫn lưu checkpoint tốt nhất
            - verbose (bool): In kết quả mỗi epoch (mặc định True)

    Returns:
        history (dict): {
            "train_loss": [...], "val_loss": [...],
            "train_acc": [...],  "val_acc": [...]
        }
    """
    epochs = config["epochs"]
    lr = config["lr"]
    weight_decay = config.get("weight_decay", 1e-4)
    device = config["device"]
    save_path = config.get("save_path", None)
    verbose = config.get("verbose", True)

    model = model.to(device)

    # AdamW = Adam với weight decay đúng cách (không apply decay cho bias/LayerNorm)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    # CosineAnnealingLR: LR giảm dần theo hình cosine từ lr → ~0 trong T_max epochs
    # Điều này giúp model hội tụ tốt hơn ở giai đoạn cuối.
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr * 0.01)

    # Loss function: CrossEntropyLoss cho bài toán phân loại nhiều lớp
    criterion = nn.CrossEntropyLoss()

    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    best_val_acc = 0.0

    for epoch in range(1, epochs + 1):
        t0 = time.time()

        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        scheduler.step()

        # Lưu checkpoint nếu val_acc tốt hơn trước
        if val_acc > best_val_acc and save_path is not None:
            best_val_acc = val_acc
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            torch.save(model.state_dict(), save_path)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        if verbose:
            elapsed = time.time() - t0
            print(
                f"Epoch {epoch:03d}/{epochs} | "
                f"loss: {train_loss:.4f} | acc: {train_acc:.4f} | "
                f"val_loss: {val_loss:.4f} | val_acc: {val_acc:.4f} | "
                f"{elapsed:.1f}s"
            )

    if save_path is not None:
        print(f"Best val acc: {best_val_acc:.4f} → saved to {save_path}")

    return history


def load_best_model(model, save_path, device):
    """Tải lại checkpoint tốt nhất vào model."""
    model.load_state_dict(torch.load(save_path, map_location=device))
    model = model.to(device)
    model.eval()
    return model
