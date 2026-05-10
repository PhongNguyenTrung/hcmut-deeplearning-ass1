"""Generate docs/banner.png — clean technical banner for the README hero.

Run: python docs/banner.py
"""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def main() -> None:
    repo_root = Path(__file__).resolve().parent.parent

    yolo = json.loads((repo_root / "exercise_2/results/metrics/yolo_results.json").read_text())
    frcnn = json.loads((repo_root / "exercise_2/results/metrics/frcnn_results.json").read_text())

    fig, ax = plt.subplots(figsize=(12, 3.2), dpi=160)
    ax.set_facecolor("#0d1117")
    fig.patch.set_facecolor("#0d1117")

    ax.text(0.02, 0.78, "Deep Learning Portfolio",
            transform=ax.transAxes, fontsize=28, fontweight="bold",
            color="#f0f6fc", family="DejaVu Sans")
    ax.text(0.02, 0.52, "HCMUT CO5085  ·  Vision  ·  Language  ·  Multimodal",
            transform=ax.transAxes, fontsize=13,
            color="#8b949e", family="DejaVu Sans")

    bullets = [
        "9 models  ·  4 datasets  ·  ~25 notebooks",
        f"YOLOv8n  {yolo['mAP_50']*100:.1f}% mAP@0.5  ·  {yolo['fps']:.0f} FPS",
        f"Faster R-CNN  {frcnn['mAP_50']*100:.1f}% mAP@0.5  ·  {frcnn['fps']:.1f} FPS",
    ]
    for i, line in enumerate(bullets):
        ax.text(0.02, 0.28 - i * 0.16, line,
                transform=ax.transAxes, fontsize=10.5,
                color="#7ee787" if i == 0 else "#79c0ff",
                family="DejaVu Sans Mono")

    rng = np.random.default_rng(7)
    x = np.linspace(0.55, 0.98, 80)
    base = 1 - 0.6 * np.exp(-(x - 0.55) * 6)
    for k, alpha in enumerate([0.25, 0.45, 0.7, 1.0]):
        y_line = base + 0.04 * np.sin(x * (12 + k * 4)) + rng.normal(0, 0.012, x.size)
        ax.plot(x, y_line, transform=ax.transAxes,
                color="#58a6ff", alpha=alpha, linewidth=1.4)

    ax.text(0.55, 0.92, "training curves",
            transform=ax.transAxes, fontsize=8.5,
            color="#6e7681", family="DejaVu Sans Mono")

    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    out = repo_root / "docs/banner.png"
    fig.savefig(out, bbox_inches="tight", facecolor="#0d1117", pad_inches=0.15)
    print(f"Wrote {out} ({out.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
