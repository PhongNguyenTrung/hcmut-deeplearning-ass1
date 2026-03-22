"""
eval_image.py – Evaluate a trained image model from checkpoint (no re-training).
CO5085 – Assignment 1

Run:
    python scripts/eval_image.py --model vit_b16
    python scripts/eval_image.py --model resnet50
"""

import argparse
import json
import os
import sys

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from datasets import get_cifar100_loaders
from evaluate import compute_metrics, get_predictions
from models import get_deit_small, get_efficientnet_b0, get_resnet50, get_vit_b16

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
NUM_WORKERS = 0 if DEVICE == "mps" else 2
NUM_CLASSES = 100

MODEL_CONFIGS = {
    "resnet50":    (lambda: get_resnet50(NUM_CLASSES, pretrained=False),    128, 32),
    "efficientnet": (lambda: get_efficientnet_b0(NUM_CLASSES, pretrained=False), 128, 32),
    "vit_b16":     (lambda: get_vit_b16(NUM_CLASSES, pretrained=False),     32, 224),
    "deit_small":  (lambda: get_deit_small(NUM_CLASSES, pretrained=False),  32, 224),
}


def get_test_loader(batch_size, img_size):
    normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    if img_size == 32:
        tf = transforms.Compose([transforms.ToTensor(), normalize])
    else:
        tf = transforms.Compose([
            transforms.Resize(256), transforms.CenterCrop(224),
            transforms.ToTensor(), normalize,
        ])
    data_dir = os.path.join(RESULTS_DIR, "..", "data", "image")
    test_ds = datasets.CIFAR100(data_dir, train=False, download=True, transform=tf)
    return DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                      num_workers=NUM_WORKERS, pin_memory=(DEVICE == "cuda"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=list(MODEL_CONFIGS.keys()), required=True)
    args = parser.parse_args()

    name = args.model
    model_fn, batch_size, img_size = MODEL_CONFIGS[name]

    ckpt_path = os.path.join(RESULTS_DIR, f"{name}_best.pt")
    if not os.path.exists(ckpt_path):
        print(f"Checkpoint not found: {ckpt_path}")
        return

    print(f"Loading checkpoint: {ckpt_path}")
    model = model_fn()
    model.load_state_dict(torch.load(ckpt_path, map_location=DEVICE))
    model.to(DEVICE)

    print(f"Evaluating {name} on CIFAR-100 test set (device={DEVICE})...")
    test_loader = get_test_loader(batch_size, img_size)
    preds, labels, _ = get_predictions(model, test_loader, DEVICE)
    metrics = compute_metrics(preds, labels, verbose=True)

    out_path = os.path.join(RESULTS_DIR, f"{name}_metrics.json")
    with open(out_path, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\n{name} → Acc: {metrics['accuracy']:.4f} | F1: {metrics['f1_macro']:.4f}")
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
