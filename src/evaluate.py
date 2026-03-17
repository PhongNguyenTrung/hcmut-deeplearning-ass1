"""
evaluate.py – Metrics & Visualization
CO5085 – Assignment 1
"""

import os
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report, confusion_matrix
)
from tqdm.auto import tqdm


# ─────────────────────────────────────────────
# 1. Collect Predictions
# ─────────────────────────────────────────────

@torch.no_grad()
def get_predictions(model, loader, device):
    """Run model on loader, return (all_preds, all_labels, all_probs)."""
    model.eval()
    all_preds, all_labels, all_probs = [], [], []

    for batch in tqdm(loader, desc="Predicting", leave=False):
        if isinstance(batch, dict):
            labels = batch["label"].to(device)
            inputs = {k: v.to(device) for k, v in batch.items() if k != "label"}
            outputs = model(**inputs)
        else:
            images, labels = batch
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)

        logits = outputs.logits if hasattr(outputs, "logits") else outputs
        probs = torch.softmax(logits, dim=-1)
        preds = probs.argmax(dim=-1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        all_probs.extend(probs.cpu().numpy())

    return np.array(all_preds), np.array(all_labels), np.array(all_probs)


# ─────────────────────────────────────────────
# 2. Metrics
# ─────────────────────────────────────────────

def compute_metrics(preds, labels, class_names=None, verbose=True):
    """Compute accuracy, F1-macro, and per-class metrics."""
    acc = accuracy_score(labels, preds)
    f1  = f1_score(labels, preds, average="macro", zero_division=0)

    results = {"accuracy": acc, "f1_macro": f1}

    if verbose:
        print(f"Accuracy : {acc:.4f}")
        print(f"F1-Macro : {f1:.4f}")
        print("\nClassification Report:")
        print(classification_report(labels, preds,
                                     target_names=class_names,
                                     zero_division=0))
    return results


# ─────────────────────────────────────────────
# 3. Plots
# ─────────────────────────────────────────────

def plot_confusion_matrix(preds, labels, class_names, save_path: str = None,
                           figsize=(12, 10), normalize: bool = True):
    """Plot and optionally save confusion matrix."""
    cm = confusion_matrix(labels, preds)
    if normalize:
        cm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
        fmt = ".2f"
    else:
        fmt = "d"

    plt.figure(figsize=figsize)
    sns.heatmap(cm, annot=True, fmt=fmt, cmap="Blues",
                xticklabels=class_names, yticklabels=class_names,
                linewidths=0.5, annot_kws={"size": 7})
    plt.title("Confusion Matrix", fontsize=14, fontweight="bold")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.xticks(rotation=45, ha="right", fontsize=7)
    plt.yticks(rotation=0, fontsize=7)
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Confusion matrix saved → {save_path}")
    plt.show()


def plot_training_curves(history: dict, model_name: str = "Model",
                          save_path: str = None):
    """Plot loss and accuracy curves from training history."""
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Loss
    axes[0].plot(epochs, history["train_loss"], label="Train", marker="o", markersize=3)
    axes[0].plot(epochs, history["val_loss"],   label="Val",   marker="s", markersize=3)
    axes[0].set_title(f"{model_name} – Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Accuracy
    axes[1].plot(epochs, history["train_acc"], label="Train", marker="o", markersize=3)
    axes[1].plot(epochs, history["val_acc"],   label="Val",   marker="s", markersize=3)
    axes[1].set_title(f"{model_name} – Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.suptitle(model_name, fontsize=14, fontweight="bold")
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Training curves saved → {save_path}")
    plt.show()


def compare_models(results: dict, metric: str = "accuracy", save_path: str = None):
    """
    Bar chart comparing multiple models.
    results: {"ModelName": {"accuracy": float, "f1_macro": float}, ...}
    """
    names = list(results.keys())
    values = [results[n][metric] for n in names]
    colors = plt.cm.Set2(np.linspace(0, 1, len(names)))

    plt.figure(figsize=(max(6, len(names) * 1.2), 5))
    bars = plt.bar(names, values, color=colors, edgecolor="black", linewidth=0.8)
    for bar, val in zip(bars, values):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                 f"{val:.4f}", ha="center", va="bottom", fontsize=9, fontweight="bold")
    plt.title(f"Model Comparison – {metric.replace('_', ' ').title()}", fontsize=13)
    plt.ylabel(metric.replace("_", " ").title())
    plt.ylim(0, min(1.05, max(values) + 0.1))
    plt.xticks(rotation=20, ha="right")
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Comparison chart saved → {save_path}")
    plt.show()
