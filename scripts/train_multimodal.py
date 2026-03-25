"""
train_multimodal.py – CLIP zero-shot image–text retrieval on Flickr30k.
CO5085 – Assignment 1

Evaluates CLIP ViT-B/32 on Flickr30k test split (1,000 images × 5 captions)
using Recall@K (R@1, R@5, R@10) for both Image→Text and Text→Image retrieval.

Run:
    python scripts/train_multimodal.py
    python scripts/train_multimodal.py --batch_size 32
"""

import argparse
import json
import os
import sys

import clip
import matplotlib.pyplot as plt
import torch

# Import HuggingFace datasets before inserting src into path
# (avoids shadow by src/datasets.py)
from datasets import load_dataset as hf_load_dataset

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"


def encode_images(clip_model, preprocess, images, batch_size=64):
    feats = []
    for i in range(0, len(images), batch_size):
        batch = torch.stack([preprocess(img) for img in images[i:i + batch_size]]).to(DEVICE)
        with torch.no_grad():
            f = clip_model.encode_image(batch).float()
            f /= f.norm(dim=-1, keepdim=True)
        feats.append(f.cpu())
    return torch.cat(feats)  # (N, 512)


def encode_texts(clip_model, texts, batch_size=64):
    feats = []
    for i in range(0, len(texts), batch_size):
        tokens = clip.tokenize(texts[i:i + batch_size], truncate=True).to(DEVICE)
        with torch.no_grad():
            f = clip_model.encode_text(tokens).float()
            f /= f.norm(dim=-1, keepdim=True)
        feats.append(f.cpu())
    return torch.cat(feats)  # (M, 512)


def recall_at_k(image_feats, caption_feats, k_list, captions_per_image=5):
    """Compute Recall@K for Image→Text and Text→Image retrieval."""
    results = {}

    # Image → Text
    sim_i2t = image_feats @ caption_feats.T  # (1000, 5000)
    i2t = {}
    for k in k_list:
        hits = sum(
            bool(
                set(range(i * captions_per_image, i * captions_per_image + captions_per_image))
                & set(sim_i2t[i].topk(k).indices.tolist())
            )
            for i in range(len(image_feats))
        )
        i2t[f"R@{k}"] = hits / len(image_feats)
    results["Image→Text"] = i2t

    # Text → Image
    sim_t2i = caption_feats @ image_feats.T  # (5000, 1000)
    t2i = {}
    for k in k_list:
        hits = sum(
            (cap_idx // captions_per_image) in sim_t2i[cap_idx].topk(k).indices.tolist()
            for cap_idx in range(len(caption_feats))
        )
        t2i[f"R@{k}"] = hits / len(caption_feats)
    results["Text→Image"] = t2i

    return results


def plot_retrieval(results, k_list, save_path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    colors = ["#4C72B0", "#DD8452", "#55A868"]
    for ax, (direction, recalls) in zip(axes, results.items()):
        vals = [recalls[f"R@{k}"] for k in k_list]
        bars = ax.bar([f"R@{k}" for k in k_list], vals, color=colors[:len(k_list)], width=0.5)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=11, fontweight="bold")
        ax.set_title(f"{direction} Retrieval", fontsize=13, fontweight="bold")
        ax.set_ylabel("Recall@K")
        ax.set_ylim(0, 1.05)
        ax.grid(axis="y", alpha=0.3)
    plt.suptitle("CLIP Zero-shot Image–Text Retrieval – Flickr30k", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved: {save_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=64)
    args = parser.parse_args()

    print(f"Device: {DEVICE}")
    print("Loading CLIP ViT-B/32...")
    clip_model, preprocess = clip.load("ViT-B/32", device=DEVICE)
    clip_model.eval()

    print("Loading Flickr30k test split from HuggingFace...")
    ds = hf_load_dataset("AnyModal/flickr30k", split="test")
    print(f"Loaded: {len(ds)} images, {len(ds) * 5} captions")

    images = [row["image"].convert("RGB") for row in ds]
    captions_nested = [row["original_alt_text"] for row in ds]
    flat_captions = [cap for caps in captions_nested for cap in caps]

    print(f"Encoding {len(images)} images...")
    image_feats = encode_images(clip_model, preprocess, images, args.batch_size)

    print(f"Encoding {len(flat_captions)} captions...")
    caption_feats = encode_texts(clip_model, flat_captions, args.batch_size)

    print(f"Image feats: {image_feats.shape} | Caption feats: {caption_feats.shape}")

    K_LIST = [1, 5, 10]
    results = recall_at_k(image_feats, caption_feats, K_LIST)

    print("\n=== CLIP Zero-shot Retrieval – Flickr30k Test Set ===")
    header = "               " + "  ".join(f"R@{k:>2}" for k in K_LIST)
    print(header)
    print("-" * len(header))
    for direction, recalls in results.items():
        vals = "  ".join(f"{recalls[f'R@{k}']:.4f}" for k in K_LIST)
        print(f"{direction:<15} {vals}")

    # Save results JSON
    out_path = os.path.join(RESULTS_DIR, "multimodal_all_metrics.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved metrics: {out_path}")

    # Save chart
    plot_retrieval(results, K_LIST, os.path.join(RESULTS_DIR, "multimodal_retrieval.png"))


if __name__ == "__main__":
    main()
