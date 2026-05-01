"""
train.py — Vòng lặp huấn luyện cho Object Detection

Faster R-CNN: SGD + momentum + StepLR (chuẩn theo paper gốc)
YOLOv8: thin wrapper around ultralytics API (ultralytics tự xử lý toàn bộ)
"""

import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from .data import get_device


# ─── Faster R-CNN Training ────────────────────────────────────────────────────

def train_one_epoch_frcnn(
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: str,
    scaler=None,
    print_freq: int = 50,
    max_batches: int = None,
) -> dict:
    """
    Một epoch training cho Faster R-CNN.

    Khác với classification, Faster R-CNN trả về dict losses (không phải logits):
        loss_classifier:   phân loại vùng ROI
        loss_box_reg:      tinh chỉnh bounding box (ROI head)
        loss_objectness:   RPN có object hay không
        loss_rpn_box_reg:  tinh chỉnh bbox từ RPN

    Tổng loss = tổng cộng 4 thành phần trên.

    Args:
        model: Faster R-CNN model (đang ở training mode)
        loader: DataLoader trả về (list_images, list_targets)
        optimizer: SGD optimizer
        device: "cuda", "mps", hoặc "cpu"
        scaler: GradScaler cho AMP (chỉ dùng được trên CUDA)
        print_freq: in log sau bao nhiêu batch

    Returns:
        dict với tổng loss và từng thành phần loss trung bình trên epoch
    """
    model.train()
    total_loss = 0.0
    loss_components = {
        "loss_classifier": 0.0,
        "loss_box_reg": 0.0,
        "loss_objectness": 0.0,
        "loss_rpn_box_reg": 0.0,
    }
    n_batches = len(loader)

    for i, (images, targets) in enumerate(loader):
        if max_batches is not None and i >= max_batches:
            break
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        optimizer.zero_grad()

        if scaler is not None:
            with torch.cuda.amp.autocast():
                loss_dict = model(images, targets)
                losses = sum(loss for loss in loss_dict.values())
            scaler.scale(losses).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            losses.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
            optimizer.step()

        batch_loss = losses.item()
        total_loss += batch_loss
        for k in loss_components:
            if k in loss_dict:
                loss_components[k] += loss_dict[k].item()

        if (i + 1) % print_freq == 0 or (i + 1) == n_batches:
            print(f"  Batch [{i + 1}/{n_batches}] loss={batch_loss:.4f}")

    n = max(n_batches, 1)
    return {
        "total_loss": total_loss / n,
        **{k: v / n for k, v in loss_components.items()},
    }


@torch.no_grad()
def evaluate_frcnn(
    model: torch.nn.Module,
    loader: DataLoader,
    device: str,
    max_batches: int = None,
) -> float:
    """
    Tính validation loss cho Faster R-CNN.

    Lưu ý: Faster R-CNN chỉ trả về losses khi ở chế độ training,
    nên ta tạm thời set model.train() để tính val loss.

    Returns:
        val_loss trung bình trên toàn bộ validation set
    """
    model.train()
    total_loss = 0.0
    n_batches = len(loader)

    for i, (images, targets) in enumerate(loader):
        if max_batches is not None and i >= max_batches:
            break
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        with torch.no_grad():
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
        total_loss += losses.item()

    model.eval()
    return total_loss / max(n_batches, 1)


def fit_frcnn(
    model: torch.nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    config: dict = None,
) -> dict:
    """
    Vòng lặp fine-tune hoàn chỉnh cho Faster R-CNN.

    Optimizer: SGD + momentum (chuẩn theo Faster R-CNN paper)
    Scheduler: StepLR — giảm LR 10x ở epoch 7/10

    Config keys:
        epochs (int):          số epoch (default 10)
        lr (float):            learning rate (default 0.005)
        momentum (float):      SGD momentum (default 0.9)
        weight_decay (float):  L2 regularization (default 0.0005)
        step_size (int):       giảm LR sau bao nhiêu epoch (default 7)
        gamma (float):         hệ số giảm LR (default 0.1)
        device (str):          thiết bị (default auto-detect)
        save_path (str):       đường dẫn lưu checkpoint (default results/checkpoints/frcnn_voc.pth)
        verbose (bool):        in log chi tiết (default True)

    Returns:
        history dict: {"train_loss", "val_loss", "epoch_times", "best_epoch"}
    """
    if config is None:
        config = {}

    epochs = config.get("epochs", 10)
    lr = config.get("lr", 0.005)
    momentum = config.get("momentum", 0.9)
    weight_decay = config.get("weight_decay", 0.0005)
    step_size = config.get("step_size", 7)
    gamma = config.get("gamma", 0.1)
    device = config.get("device", get_device())
    save_path = config.get("save_path", "results/checkpoints/frcnn_voc.pth")
    verbose = config.get("verbose", True)
    max_batches = config.get("max_batches", None)  # None = full epoch; int = limit for dry-run

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    model = model.to(device)

    # AMP chỉ dùng được trên CUDA
    scaler = torch.cuda.amp.GradScaler() if device == "cuda" else None

    optimizer = torch.optim.SGD(
        [p for p in model.parameters() if p.requires_grad],
        lr=lr,
        momentum=momentum,
        weight_decay=weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=step_size, gamma=gamma
    )

    history = {"train_loss": [], "val_loss": [], "epoch_times": []}
    best_val_loss = float("inf")

    if verbose:
        print(f"[Train] Fine-tune Faster R-CNN trên {device} | {epochs} epochs")
        print(f"        LR={lr}, momentum={momentum}, batch={train_loader.batch_size}")

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        if verbose:
            print(f"\n── Epoch {epoch}/{epochs} ──")

        train_metrics = train_one_epoch_frcnn(
            model, train_loader, optimizer, device, scaler, max_batches=max_batches)
        val_loss = evaluate_frcnn(model, val_loader, device, max_batches=max_batches)
        scheduler.step()

        elapsed = time.time() - t0
        history["train_loss"].append(train_metrics["total_loss"])
        history["val_loss"].append(val_loss)
        history["epoch_times"].append(elapsed)

        if verbose:
            print(f"  train_loss={train_metrics['total_loss']:.4f} | "
                  f"val_loss={val_loss:.4f} | "
                  f"lr={scheduler.get_last_lr()[0]:.6f} | "
                  f"time={elapsed:.1f}s")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            history["best_epoch"] = epoch
            torch.save(model.state_dict(), save_path)
            if verbose:
                print(f"  ✓ Saved checkpoint → {save_path}")

    if verbose:
        print(f"\n[Train] Done! Best epoch={history.get('best_epoch', '?')} | "
              f"best_val_loss={best_val_loss:.4f}")

    return history


# ─── YOLOv8 Training ──────────────────────────────────────────────────────────

def train_yolov8(
    data_yaml: str,
    model_size: str = "n",
    epochs: int = 20,
    imgsz: int = 640,
    batch: int = 16,
    project: str = "results",
    name: str = "yolo_voc",
    device: str = "auto",
    exist_ok: bool = True,
) -> str:
    # ultralytics chỉ chấp nhận "cpu", "mps", hoặc CUDA device index ("0")
    # "auto" không hợp lệ trên non-CUDA machines
    if device == "auto":
        device = get_device()
        if device == "cuda":
            device = "0"
    """
    Fine-tune YOLOv8 trên Pascal VOC bằng ultralytics API.

    ultralytics xử lý toàn bộ: mosaic augmentation, cosine LR decay,
    multi-scale training, EMA, mixed precision — không cần viết thêm.

    Args:
        data_yaml:  đường dẫn đến voc.yaml (từ prepare_yolo_dataset)
        model_size: "n" (nano), "s" (small), "m" (medium)
        epochs:     số epoch huấn luyện
        imgsz:      kích thước ảnh đầu vào (640 là chuẩn cho YOLOv8)
        batch:      batch size (-1 để auto-detect theo VRAM)
        project:    thư mục gốc lưu kết quả
        name:       tên subfolder trong project
        device:     "auto" để tự chọn, hoặc "0", "cpu", "mps"

    Returns:
        str: đường dẫn đến file best.pt
    """
    try:
        from ultralytics import YOLO
    except ImportError:
        raise ImportError("Cần cài: pip install ultralytics>=8.0.0")

    print(f"[Train] Fine-tune YOLOv8{model_size} | {epochs} epochs | imgsz={imgsz}")
    model = YOLO(f"yolov8{model_size}.pt")

    results = model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        project=project,
        name=name,
        device=device,
        exist_ok=exist_ok,
        verbose=True,
        save=True,
        plots=True,
    )

    # ultralytics saves to runs/detect/{project}/{name}/ — use save_dir from results
    try:
        actual_save_dir = Path(results.save_dir)
        best_pt = actual_save_dir / "weights" / "best.pt"
        if best_pt.exists():
            print(f"[Train] YOLOv8 best checkpoint: {best_pt}")
            return str(best_pt)
    except Exception:
        pass

    # fallback: search common locations
    for search_root in [Path(project), Path("runs")]:
        for candidate in search_root.rglob("best.pt"):
            print(f"[Train] YOLOv8 best checkpoint (found): {candidate}")
            return str(candidate)

    return str(Path(project) / name / "weights" / "best.pt")


def load_frcnn_checkpoint(model: torch.nn.Module, path: str, device: str) -> torch.nn.Module:
    """Load saved state_dict vào Faster R-CNN model."""
    state = torch.load(path, map_location=device)
    model.load_state_dict(state)
    model.to(device)
    print(f"[Checkpoint] Loaded {path}")
    return model
