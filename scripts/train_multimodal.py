"""
train_multimodal.py – CLIP zero-shot and few-shot classification.
CO5085 – Assignment 1

Uses CIFAR-100 superclasses (20 classes) as proxy multimodal dataset.

Run:
    python scripts/train_multimodal.py
    python scripts/train_multimodal.py --n_test 1000 --shots 1 5 10 20
"""

import argparse
import json
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as T
from collections import defaultdict
from sklearn.metrics import accuracy_score, f1_score
from torch.optim import Adam
from torchvision import datasets

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from evaluate import compare_models
from models import CLIPFewShotClassifier, CLIPZeroShotClassifier

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "image")
os.makedirs(RESULTS_DIR, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

SUPERCLASSES = [
    "aquatic mammals", "fish", "flowers", "food containers", "fruit and vegetables",
    "household electrical devices", "household furniture", "insects", "large carnivores",
    "large man-made outdoor things", "large natural outdoor scenes",
    "large omnivores and herbivores", "medium-sized mammals", "non-insect invertebrates",
    "people", "reptiles", "small mammals", "trees", "vehicles 1", "vehicles 2",
]

# CIFAR-100 fine-class → superclass index mapping
COARSE_LABELS = [
    4, 1, 14, 8, 0, 6, 7, 7, 18, 3, 3, 14, 9, 18, 7, 11, 3, 9, 7, 11,
    6, 11, 5, 10, 7, 6, 13, 15, 3, 15, 0, 11, 1, 10, 12, 14, 16, 9, 11, 5,
    5, 19, 8, 8, 15, 13, 14, 17, 18, 10, 16, 4, 17, 4, 2, 0, 17, 4, 18, 17,
    10, 3, 2, 12, 12, 16, 12, 1, 9, 19, 2, 10, 0, 1, 16, 12, 9, 13, 15, 13,
    16, 19, 2, 4, 6, 19, 5, 5, 8, 19, 18, 1, 2, 15, 6, 0, 17, 8, 14, 13,
]


def load_cifar100_test(n_test: int = 500):
    """Load n_test CIFAR-100 test images as PIL Images with superclass labels."""
    ds = datasets.CIFAR100(
        root=DATA_DIR, train=False, download=True,
        transform=T.Compose([T.Resize(224), T.ToTensor()]),
    )
    to_pil = T.ToPILImage()
    images = [to_pil(ds[i][0]) for i in range(min(n_test, len(ds)))]
    labels = [COARSE_LABELS[ds[i][1]] for i in range(min(n_test, len(ds)))]
    return images, labels, ds


def zero_shot(test_images, test_labels):
    print("\n--- Zero-shot ---")
    clf = CLIPZeroShotClassifier(class_names=SUPERCLASSES, device=DEVICE)
    batch_size = 50
    all_preds = []
    for i in range(0, len(test_images), batch_size):
        _, preds = clf.predict(test_images[i:i + batch_size])
        all_preds.extend(preds.tolist())

    acc = accuracy_score(test_labels, all_preds)
    f1  = f1_score(test_labels, all_preds, average="macro", zero_division=0)
    print(f"  Zero-shot → Acc: {acc:.4f} | F1: {f1:.4f}")
    return {"accuracy": acc, "f1_macro": f1}


def few_shot(k: int, test_images, test_labels, full_ds):
    """Train linear probe with k shots per class on CIFAR-100 test set (train split)."""
    print(f"  {k}-shot", end="  ")
    clf = CLIPFewShotClassifier(num_classes=len(SUPERCLASSES), device=DEVICE).to(DEVICE)

    # Gather k examples per superclass from the full test dataset
    class_examples = defaultdict(list)
    for idx in range(len(full_ds)):
        img_t, fine_label = full_ds[idx]
        coarse = COARSE_LABELS[fine_label]
        if len(class_examples[coarse]) < k:
            class_examples[coarse].append(T.ToPILImage()(img_t))

    support_imgs, support_labels = [], []
    for c in range(len(SUPERCLASSES)):
        imgs = class_examples[c]
        support_imgs.extend(imgs)
        support_labels.extend([c] * len(imgs))

    X_support = clf.encode_images(support_imgs)
    y_support = torch.tensor(support_labels, dtype=torch.long).to(DEVICE)

    # Train linear head
    optimizer = Adam(clf.classifier.parameters(), lr=1e-2)
    criterion = nn.CrossEntropyLoss()
    for _ in range(200):
        optimizer.zero_grad()
        loss = criterion(clf.classifier(X_support), y_support)
        loss.backward()
        optimizer.step()

    # Evaluate
    clf.eval()
    with torch.no_grad():
        test_feats = clf.encode_images(test_images)
        preds = clf.classifier(test_feats).argmax(-1).cpu().numpy()

    acc = accuracy_score(test_labels, preds)
    f1  = f1_score(test_labels, preds, average="macro", zero_division=0)
    print(f"→ Acc: {acc:.4f} | F1: {f1:.4f}")
    torch.save(clf.state_dict(), os.path.join(RESULTS_DIR, f"clip_fewshot_{k}shot_best.pt"))
    return {"accuracy": acc, "f1_macro": f1}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_test", type=int, default=500,
                        help="Number of test images to evaluate on")
    parser.add_argument("--shots", nargs="+", type=int, default=[1, 5, 10, 20],
                        help="K-shot values to evaluate")
    args = parser.parse_args()

    print(f"Device: {DEVICE}")
    test_images, test_labels, full_ds = load_cifar100_test(args.n_test)
    print(f"Test images: {len(test_images)} | Superclasses: {len(SUPERCLASSES)}")

    results = {}

    # Zero-shot
    results["Zero-shot"] = zero_shot(test_images, test_labels)

    # Few-shot
    print("\n--- Few-shot ---")
    for k in args.shots:
        results[f"{k}-shot"] = few_shot(k, test_images, test_labels, full_ds)

    # Save results
    with open(os.path.join(RESULTS_DIR, "multimodal_all_metrics.json"), "w") as f:
        json.dump(results, f, indent=2)

    # Comparison chart
    compare_models(
        results, metric="accuracy",
        save_path=os.path.join(RESULTS_DIR, "multimodal_comparison_acc.png"),
    )
    compare_models(
        results, metric="f1_macro",
        save_path=os.path.join(RESULTS_DIR, "multimodal_comparison_f1.png"),
    )

    print("\n\n=== SUMMARY ===")
    for name, m in results.items():
        print(f"  {name:15s}  Acc: {m['accuracy']:.4f}  F1: {m['f1_macro']:.4f}")


if __name__ == "__main__":
    main()
