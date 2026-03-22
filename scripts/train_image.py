"""
train_image.py – Train all 4 image models on CIFAR-100.
CO5085 – Assignment 1

Run:
    python scripts/train_image.py              # all 4 models
    python scripts/train_image.py --model resnet50
"""

import argparse
import json
import os
import sys

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from datasets import get_cifar100_loaders, get_image_transforms
from evaluate import compute_metrics, get_predictions, plot_training_curves
from models import get_deit_small, get_efficientnet_b0, get_resnet50, get_vit_b16
from train import train

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
NUM_WORKERS = 0 if DEVICE == "mps" else 2  # macOS multiprocessing workaround
NUM_CLASSES = 100

# Model configs: (factory_fn, lr, epochs, batch_size, img_size)
MODEL_CONFIGS = {
    "resnet50": (
        lambda: get_resnet50(NUM_CLASSES, pretrained=True),
        1e-3, 5, 128, 32,
    ),
    "efficientnet": (
        lambda: get_efficientnet_b0(NUM_CLASSES, pretrained=True),
        1e-3, 5, 128, 32,
    ),
    "vit_b16": (
        lambda: get_vit_b16(NUM_CLASSES, pretrained=True),
        2e-5, 5, 32, 224,
    ),
    "deit_small": (
        lambda: get_deit_small(NUM_CLASSES, pretrained=True),
        2e-5, 5, 32, 224,
    ),
}


def get_loaders(batch_size, img_size):
    """Return CIFAR-100 loaders with the given image size."""
    if img_size == 32:
        return get_cifar100_loaders(
            data_dir=os.path.join(RESULTS_DIR, "..", "data", "image"),
            batch_size=batch_size,
            num_workers=NUM_WORKERS,
        )
    # For ViT / DeiT — resize CIFAR-100 to 224×224
    normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    train_tf = transforms.Compose([
        transforms.Resize(256), transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.4, 0.4, 0.4),
        transforms.ToTensor(), normalize,
    ])
    val_tf = transforms.Compose([
        transforms.Resize(256), transforms.CenterCrop(224),
        transforms.ToTensor(), normalize,
    ])
    data_dir = os.path.join(RESULTS_DIR, "..", "data", "image")
    train_ds = datasets.CIFAR100(data_dir, train=True,  download=True, transform=train_tf)
    test_ds  = datasets.CIFAR100(data_dir, train=False, download=True, transform=val_tf)

    val_size = int(0.2 * len(train_ds))
    train_ds, val_ds = torch.utils.data.random_split(
        train_ds, [len(train_ds) - val_size, val_size],
        generator=torch.Generator().manual_seed(42),
    )
    val_ds.dataset.transform = val_tf  # apply val transforms on val split

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=NUM_WORKERS, pin_memory=(DEVICE == "cuda"))
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=NUM_WORKERS, pin_memory=(DEVICE == "cuda"))
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, num_workers=NUM_WORKERS, pin_memory=(DEVICE == "cuda"))
    return train_loader, val_loader, test_loader


def run_model(name):
    print(f"\n{'='*60}")
    print(f"  Training: {name}  |  Device: {DEVICE}")
    print(f"{'='*60}")

    model_fn, lr, epochs, batch_size, img_size = MODEL_CONFIGS[name]
    train_loader, val_loader, test_loader = get_loaders(batch_size, img_size)

    model = model_fn()
    save_path = os.path.join(RESULTS_DIR, f"{name}_best.pt")

    history = train(
        model, train_loader, val_loader,
        num_epochs=epochs, lr=lr,
        device=DEVICE, save_path=save_path,
        scheduler_type="cosine",
    )

    # Save history
    with open(os.path.join(RESULTS_DIR, f"{name}_history.json"), "w") as f:
        json.dump(history, f, indent=2)

    # Plot training curves
    plot_training_curves(
        history, model_name=name,
        save_path=os.path.join(RESULTS_DIR, f"{name}_curves.png"),
    )

    # Evaluate on test set
    model.load_state_dict(torch.load(save_path, map_location=DEVICE))
    preds, labels, _ = get_predictions(model, test_loader, DEVICE)
    metrics = compute_metrics(preds, labels, verbose=True)

    with open(os.path.join(RESULTS_DIR, f"{name}_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\n{name} → Acc: {metrics['accuracy']:.4f} | F1: {metrics['f1_macro']:.4f}")
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=list(MODEL_CONFIGS.keys()), default=None,
                        help="Run a single model. Omit to run all.")
    parser.add_argument("--epochs", type=int, default=None,
                        help="Override number of epochs for all models.")
    args = parser.parse_args()

    if args.epochs:
        for k, (fn, lr, _, bs, img) in MODEL_CONFIGS.items():
            MODEL_CONFIGS[k] = (fn, lr, args.epochs, bs, img)

    models_to_run = [args.model] if args.model else list(MODEL_CONFIGS.keys())
    all_results = {}

    for name in models_to_run:
        all_results[name] = run_model(name)

    # Summary
    print("\n\n=== SUMMARY ===")
    for name, m in all_results.items():
        print(f"  {name:20s}  Acc: {m['accuracy']:.4f}  F1: {m['f1_macro']:.4f}")

    with open(os.path.join(RESULTS_DIR, "image_all_metrics.json"), "w") as f:
        json.dump(all_results, f, indent=2)


if __name__ == "__main__":
    main()
