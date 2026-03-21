"""
train_text.py – Train all 4 text models on 20 Newsgroups.
CO5085 – Assignment 1

Run:
    python scripts/train_text.py              # all 4 models
    python scripts/train_text.py --model distilbert
"""

import argparse
import json
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from datasets import get_20newsgroups_loaders
from evaluate import compute_metrics, get_predictions, plot_training_curves
from models import BiLSTMClassifier, GRUClassifier, get_bert, get_distilbert
from train import train

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
NUM_WORKERS = 0 if DEVICE == "mps" else 2
NUM_CLASSES = 20

# (lr, epochs, batch_size)
MODEL_CONFIGS = {
    "bilstm":      (1e-3,  10, 64),
    "gru":         (1e-3,  10, 64),
    "distilbert":  (2e-5,  5,  32),
    "bert":        (2e-5,  5,  16),
}


def build_model(name, vocab_size):
    if name == "bilstm":
        return BiLSTMClassifier(vocab_size=vocab_size, num_classes=NUM_CLASSES)
    if name == "gru":
        return GRUClassifier(vocab_size=vocab_size, num_classes=NUM_CLASSES)
    if name == "distilbert":
        return get_distilbert(NUM_CLASSES)
    if name == "bert":
        return get_bert(NUM_CLASSES)
    raise ValueError(f"Unknown model: {name}")


def run_model(name, train_loader, val_loader, test_loader, vocab_size):
    print(f"\n{'='*60}")
    print(f"  Training: {name}  |  Device: {DEVICE}")
    print(f"{'='*60}")

    lr, epochs, _ = MODEL_CONFIGS[name]
    model = build_model(name, vocab_size)
    save_path = os.path.join(RESULTS_DIR, f"{name}_best.pt")

    history = train(
        model, train_loader, val_loader,
        num_epochs=epochs, lr=lr,
        device=DEVICE, save_path=save_path,
        scheduler_type="cosine",
    )

    with open(os.path.join(RESULTS_DIR, f"{name}_history.json"), "w") as f:
        json.dump(history, f, indent=2)

    plot_training_curves(
        history, model_name=name,
        save_path=os.path.join(RESULTS_DIR, f"{name}_curves.png"),
    )

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
    args = parser.parse_args()

    models_to_run = [args.model] if args.model else list(MODEL_CONFIGS.keys())

    # Build dataloaders — reuse across all models for the same batch size
    # RNN models and transformer models may use different batch sizes
    all_results = {}

    for name in models_to_run:
        _, epochs, batch_size = MODEL_CONFIGS[name]
        tokenizer_name = "bert-base-uncased" if name == "bert" else "distilbert-base-uncased"
        train_loader, val_loader, test_loader, tokenizer, _ = get_20newsgroups_loaders(
            tokenizer_name=tokenizer_name,
            batch_size=batch_size,
            num_workers=NUM_WORKERS,
        )
        vocab_size = tokenizer.vocab_size
        all_results[name] = run_model(name, train_loader, val_loader, test_loader, vocab_size)

    print("\n\n=== SUMMARY ===")
    for name, m in all_results.items():
        print(f"  {name:20s}  Acc: {m['accuracy']:.4f}  F1: {m['f1_macro']:.4f}")

    with open(os.path.join(RESULTS_DIR, "text_all_metrics.json"), "w") as f:
        json.dump(all_results, f, indent=2)


if __name__ == "__main__":
    main()
