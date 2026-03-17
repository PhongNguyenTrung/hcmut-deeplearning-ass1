"""
train.py – Training utilities
CO5085 – Assignment 1
"""

import os
import time
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
from tqdm.auto import tqdm


def train_one_epoch(model, loader, optimizer, criterion, device, scaler=None):
    """Run one training epoch. Returns (avg_loss, accuracy)."""
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    for batch in tqdm(loader, desc="Train", leave=False):
        # Support both image/text and plain (images, labels) batches
        if isinstance(batch, dict):
            labels = batch["label"].to(device)
            inputs = {k: v.to(device) for k, v in batch.items() if k != "label"}
        else:
            images, labels = batch
            images, labels = images.to(device), labels.to(device)
            inputs = images

        optimizer.zero_grad()

        if scaler is not None:  # Mixed precision
            with torch.cuda.amp.autocast():
                outputs = model(**inputs) if isinstance(inputs, dict) else model(inputs)
                logits = outputs.logits if hasattr(outputs, "logits") else outputs
                loss = criterion(logits, labels)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(**inputs) if isinstance(inputs, dict) else model(inputs)
            logits = outputs.logits if hasattr(outputs, "logits") else outputs
            loss = criterion(logits, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        total_loss += loss.item() * labels.size(0)
        preds = logits.argmax(dim=-1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    """Evaluate model on a data loader. Returns (avg_loss, accuracy)."""
    model.eval()
    total_loss, correct, total = 0.0, 0, 0

    for batch in tqdm(loader, desc="Eval", leave=False):
        if isinstance(batch, dict):
            labels = batch["label"].to(device)
            inputs = {k: v.to(device) for k, v in batch.items() if k != "label"}
        else:
            images, labels = batch
            images, labels = images.to(device), labels.to(device)
            inputs = images

        outputs = model(**inputs) if isinstance(inputs, dict) else model(inputs)
        logits = outputs.logits if hasattr(outputs, "logits") else outputs
        loss = criterion(logits, labels)

        total_loss += loss.item() * labels.size(0)
        preds = logits.argmax(dim=-1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    return total_loss / total, correct / total


def train(
    model,
    train_loader,
    val_loader,
    num_epochs: int = 10,
    lr: float = 2e-4,
    weight_decay: float = 1e-4,
    device: str = "cuda",
    save_path: str = "./results/best_model.pt",
    scheduler_type: str = "cosine",       # "cosine" | "plateau"
    use_amp: bool = True,                  # Mixed-precision (GPU only)
    patience: int = 5,                     # Early stopping
):
    """
    Full training loop with:
      - AdamW optimizer
      - Cosine / ReduceLROnPlateau scheduler
      - Mixed-precision (AMP)
      - Early stopping
      - Best-model checkpointing
    
    Returns history dict: {train_loss, val_loss, train_acc, val_acc}
    """
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr, weight_decay=weight_decay
    )

    if scheduler_type == "cosine":
        scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs)
    else:
        scheduler = ReduceLROnPlateau(optimizer, mode="max", patience=2, factor=0.5)

    scaler = torch.cuda.amp.GradScaler() if (use_amp and device == "cuda") else None

    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    best_val_acc = 0.0
    no_improve = 0
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)

    for epoch in range(1, num_epochs + 1):
        t0 = time.time()
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device, scaler)
        val_loss,   val_acc   = evaluate(model, val_loader, criterion, device)

        if scheduler_type == "cosine":
            scheduler.step()
        else:
            scheduler.step(val_acc)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        elapsed = time.time() - t0
        print(f"Epoch {epoch:03d}/{num_epochs} | "
              f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f} | "
              f"LR: {optimizer.param_groups[0]['lr']:.2e} | {elapsed:.1f}s")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), save_path)
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"Early stopping at epoch {epoch} (no improvement for {patience} epochs)")
                break

    print(f"\n✅ Best Val Accuracy: {best_val_acc:.4f} | Saved to {save_path}")
    return history
