"""
data.py — Tải và xử lý dữ liệu CIFAR-100

CIFAR-100: 100 lớp, 32×32 pixels màu (RGB), 50,000 train / 10,000 test
Mỗi lớp có 500 ảnh train và 100 ảnh test.
"""

import os
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms


# ─── Thống kê chuẩn hoá CIFAR-100 (tính từ tập train) ───────────────────────
# Lưu ý: Đây là giá trị đặc thù cho CIFAR-100, KHÔNG phải ImageNet!
# ImageNet dùng mean=[0.485,0.456,0.406] — không chính xác cho CIFAR-100.
CIFAR100_MEAN = (0.5071, 0.4867, 0.4408)
CIFAR100_STD = (0.2675, 0.2565, 0.2761)


def get_device():
    """Tự động chọn thiết bị: CUDA → MPS (Apple Silicon) → CPU."""
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def get_cifar100_loaders(
    data_dir=None,
    batch_size=128,
    num_workers=2,
    val_split=5000,
    seed=42,
):
    """
    Tải CIFAR-100 và trả về DataLoader cho train/val/test.

    Args:
        data_dir: Thư mục chứa dataset. Mặc định tự tìm ../data/image.
        batch_size: Số ảnh mỗi batch.
        num_workers: Số luồng tải dữ liệu.
        val_split: Số ảnh tách từ train làm validation (mặc định 5000).
        seed: Random seed để tái lập kết quả.

    Returns:
        train_loader, val_loader, test_loader, class_names
    """
    if data_dir is None:
        # Tự tìm thư mục data/ từ vị trí file này
        data_dir = os.path.join(os.path.dirname(__file__), "..", "..", "data", "image")
        data_dir = os.path.abspath(data_dir)

    # ── Transform cho tập TRAIN (có augmentation để tránh overfitting) ──
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),       # Cắt ngẫu nhiên sau khi đệm 4 pixel
        transforms.RandomHorizontalFlip(),           # Lật ngang ngẫu nhiên (50%)
        transforms.ToTensor(),                       # Chuyển [0,255] → [0.0,1.0]
        transforms.Normalize(CIFAR100_MEAN, CIFAR100_STD),  # Chuẩn hoá
    ])

    # ── Transform cho tập VAL/TEST (chỉ chuẩn hoá, không augment) ──
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR100_MEAN, CIFAR100_STD),
    ])

    # Tải dataset gốc
    full_train = datasets.CIFAR100(data_dir, train=True, download=True, transform=train_transform)
    val_dataset = datasets.CIFAR100(data_dir, train=True, download=False, transform=test_transform)
    test_dataset = datasets.CIFAR100(data_dir, train=False, download=True, transform=test_transform)

    # Chia train/val một cách cố định (không random mỗi lần chạy)
    torch.manual_seed(seed)
    indices = torch.randperm(len(full_train)).tolist()
    val_indices = indices[:val_split]
    train_indices = indices[val_split:]

    train_subset = Subset(full_train, train_indices)
    val_subset = Subset(val_dataset, val_indices)

    pin_memory = (get_device() == "cuda")

    train_loader = DataLoader(
        train_subset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_subset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory,
    )

    class_names = full_train.classes  # 100 tên lớp
    print(f"Train: {len(train_subset):,} | Val: {len(val_subset):,} | Test: {len(test_dataset):,}")
    print(f"Classes: {len(class_names)} | Batch size: {batch_size}")

    return train_loader, val_loader, test_loader, class_names


# ─── Sequence Transform cho Part 5 (LSTM/GRU) ────────────────────────────────

class SequenceTransform:
    """
    Chuyển ảnh CIFAR-100 thành chuỗi (sequence) cho LSTM/GRU.

    Có 4 cách đọc ảnh như một chuỗi:
    - "row"   : Mỗi hàng pixels là 1 bước thời gian → shape [32, 96]
                (T=32 bước, mỗi bước = 3 kênh × 32 pixels = 96 features)
    - "col"   : Mỗi cột pixels là 1 bước thời gian → shape [32, 96]
    - "patch4": Ảnh chia thành patches 4×4 theo thứ tự hàng → shape [64, 48]
                (64 patches, mỗi patch = 4×4×3 = 48 features)
    - "patch8": Patches 8×8 → shape [16, 192]
                (16 patches, mỗi patch = 8×8×3 = 192 features)
    """

    def __init__(self, seq_mode="row"):
        assert seq_mode in ("row", "col", "patch4", "patch8"), \
            f"seq_mode phải là 'row', 'col', 'patch4', hoặc 'patch8', không phải '{seq_mode}'"
        self.seq_mode = seq_mode

    def __call__(self, img_tensor):
        """
        Args:
            img_tensor: Tensor hình [3, 32, 32] (đã qua ToTensor + Normalize)
        Returns:
            Tensor hình [T, input_size]
        """
        C, H, W = img_tensor.shape  # C=3, H=W=32

        if self.seq_mode == "row":
            # [3, 32, 32] → [32, 3, 32] → [32, 96]
            # Mỗi hàng: lấy tất cả 3 kênh của hàng đó, flatten
            return img_tensor.permute(1, 0, 2).reshape(H, C * W)

        elif self.seq_mode == "col":
            # [3, 32, 32] → [32, 3, 32] → transpose → [32, 96]
            return img_tensor.permute(2, 0, 1).reshape(W, C * H)

        elif self.seq_mode == "patch4":
            patch_size = 4
            return self._make_patches(img_tensor, patch_size)

        else:  # patch8
            patch_size = 8
            return self._make_patches(img_tensor, patch_size)

    def _make_patches(self, img_tensor, patch_size):
        """Chia ảnh thành các patches theo thứ tự từ trái sang phải, trên xuống dưới."""
        C, H, W = img_tensor.shape
        n_h = H // patch_size  # số patch theo chiều dọc
        n_w = W // patch_size  # số patch theo chiều ngang
        num_patches = n_h * n_w
        patch_dim = patch_size * patch_size * C

        # Reshape: [C, H, W] → [C, n_h, ps, n_w, ps] → [n_h, n_w, C, ps, ps] → [n_h*n_w, C*ps*ps]
        x = img_tensor.reshape(C, n_h, patch_size, n_w, patch_size)
        x = x.permute(1, 3, 0, 2, 4).contiguous()  # [n_h, n_w, C, ps, ps]
        x = x.reshape(num_patches, patch_dim)
        return x


class SequenceDataset(torch.utils.data.Dataset):
    """Wrap CIFAR-100 với SequenceTransform."""

    def __init__(self, cifar_dataset, seq_mode):
        self.dataset = cifar_dataset
        self.seq_transform = SequenceTransform(seq_mode)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img, label = self.dataset[idx]
        return self.seq_transform(img), label


def get_sequence_loaders(seq_mode="row", data_dir=None, batch_size=128, num_workers=2, seed=42):
    """
    Tải CIFAR-100 dưới dạng chuỗi cho LSTM/GRU.

    Args:
        seq_mode: "row", "col", "patch4", hoặc "patch8"
        Các tham số còn lại giống get_cifar100_loaders

    Returns:
        train_loader, val_loader, test_loader, seq_len, input_size
    """
    if data_dir is None:
        data_dir = os.path.join(os.path.dirname(__file__), "..", "..", "data", "image")
        data_dir = os.path.abspath(data_dir)

    # Transform cơ bản (không augment, vì SequenceTransform xử lý sau)
    base_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR100_MEAN, CIFAR100_STD),
    ])
    train_transform_aug = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR100_MEAN, CIFAR100_STD),
    ])

    full_train_aug = datasets.CIFAR100(data_dir, train=True, download=True, transform=train_transform_aug)
    full_train_base = datasets.CIFAR100(data_dir, train=True, download=False, transform=base_transform)
    test_dataset = datasets.CIFAR100(data_dir, train=False, download=True, transform=base_transform)

    torch.manual_seed(seed)
    indices = torch.randperm(len(full_train_aug)).tolist()
    val_indices = indices[:5000]
    train_indices = indices[5000:]

    train_subset = SequenceDataset(Subset(full_train_aug, train_indices), seq_mode)
    val_subset = SequenceDataset(Subset(full_train_base, val_indices), seq_mode)
    test_seq = SequenceDataset(test_dataset, seq_mode)

    # Xác định shape đầu ra
    sample_seq, _ = train_subset[0]
    seq_len, input_size = sample_seq.shape

    pin_memory = (get_device() == "cuda")

    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=pin_memory)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=pin_memory)
    test_loader = DataLoader(test_seq, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, pin_memory=pin_memory)

    print(f"[seq_mode={seq_mode}] seq_len={seq_len}, input_size={input_size}")
    print(f"Train: {len(train_subset):,} | Val: {len(val_subset):,} | Test: {len(test_seq):,}")

    return train_loader, val_loader, test_loader, seq_len, input_size
