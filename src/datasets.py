"""
datasets.py – Dataset & DataLoader utilities
CO5085 – Assignment 1
"""

import os
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import datasets, transforms
from PIL import Image
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer


# ─────────────────────────────────────────────
# 1. Image Dataset (CIFAR-100 / Food-101)
# ─────────────────────────────────────────────

def get_image_transforms(train: bool = True, img_size: int = 224):
    """Returns torchvision transforms for image datasets."""
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
    if train:
        return transforms.Compose([
            transforms.RandomResizedCrop(img_size),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
            transforms.ToTensor(),
            normalize,
        ])
    else:
        return transforms.Compose([
            transforms.Resize(img_size + 32),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            normalize,
        ])


def get_cifar100_loaders(data_dir: str = "./data/image", batch_size: int = 64,
                          num_workers: int = 2):
    """Download & return CIFAR-100 DataLoaders."""
    train_ds = datasets.CIFAR100(
        root=data_dir, train=True, download=True,
        transform=get_image_transforms(train=True, img_size=32)
    )
    test_ds = datasets.CIFAR100(
        root=data_dir, train=False, download=True,
        transform=get_image_transforms(train=False, img_size=32)
    )
    # Split train → train + val (80/20)
    val_size = int(0.2 * len(train_ds))
    train_size = len(train_ds) - val_size
    train_ds, val_ds = random_split(train_ds, [train_size, val_size],
                                     generator=torch.Generator().manual_seed(42))
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                               num_workers=num_workers, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                               num_workers=num_workers, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False,
                               num_workers=num_workers, pin_memory=True)
    return train_loader, val_loader, test_loader


def get_food101_loaders(data_dir: str = "./data/image", batch_size: int = 32,
                         num_workers: int = 2, img_size: int = 224):
    """Download & return Food-101 DataLoaders."""
    train_ds = datasets.Food101(
        root=data_dir, split="train", download=True,
        transform=get_image_transforms(train=True, img_size=img_size)
    )
    test_ds = datasets.Food101(
        root=data_dir, split="test", download=True,
        transform=get_image_transforms(train=False, img_size=img_size)
    )
    val_size = int(0.15 * len(train_ds))
    train_size = len(train_ds) - val_size
    train_ds, val_ds = random_split(train_ds, [train_size, val_size],
                                     generator=torch.Generator().manual_seed(42))
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                               num_workers=num_workers, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                               num_workers=num_workers, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False,
                               num_workers=num_workers, pin_memory=True)
    return train_loader, val_loader, test_loader


# ─────────────────────────────────────────────
# 2. Text Dataset (20 Newsgroups / DBpedia)
# ─────────────────────────────────────────────

class TextDataset(Dataset):
    """Generic text classification dataset for HuggingFace tokenizers."""

    def __init__(self, texts, labels, tokenizer, max_length: int = 256):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {
            "input_ids":      encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "label":          torch.tensor(self.labels[idx], dtype=torch.long),
        }


def get_20newsgroups_loaders(tokenizer_name: str = "distilbert-base-uncased",
                              batch_size: int = 32, max_length: int = 256,
                              num_workers: int = 2):
    """Return 20 Newsgroups DataLoaders using sklearn + HuggingFace tokenizer."""
    from sklearn.datasets import fetch_20newsgroups

    train_raw = fetch_20newsgroups(subset="train", remove=("headers", "footers", "quotes"))
    test_raw  = fetch_20newsgroups(subset="test",  remove=("headers", "footers", "quotes"))

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    # train/val split
    tr_texts, val_texts, tr_labels, val_labels = train_test_split(
        train_raw.data, train_raw.target, test_size=0.15, random_state=42
    )

    train_ds = TextDataset(tr_texts,         tr_labels,         tokenizer, max_length)
    val_ds   = TextDataset(val_texts,         val_labels,         tokenizer, max_length)
    test_ds  = TextDataset(test_raw.data, list(test_raw.target), tokenizer, max_length)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=num_workers)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, test_loader, tokenizer, 20


# ─────────────────────────────────────────────
# 3. Multimodal Dataset (Flickr30k)
# ─────────────────────────────────────────────

class Flickr30kDataset(Dataset):
    """
    Simple Flickr30k-style dataset for image-text pairs.
    Expects a CSV with columns: [image_path, caption, label]
    """

    def __init__(self, df: pd.DataFrame, image_transform=None,
                 tokenizer=None, max_length: int = 77):
        self.df = df.reset_index(drop=True)
        self.image_transform = image_transform or get_image_transforms(train=False)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # Image
        image = Image.open(row["image_path"]).convert("RGB")
        if self.image_transform:
            image = self.image_transform(image)

        # Text
        caption = str(row["caption"])

        result = {"image": image, "caption": caption}

        if "label" in row:
            result["label"] = torch.tensor(int(row["label"]), dtype=torch.long)

        if self.tokenizer is not None:
            enc = self.tokenizer(caption, padding="max_length", truncation=True,
                                  max_length=self.max_length, return_tensors="pt")
            result["input_ids"]      = enc["input_ids"].squeeze(0)
            result["attention_mask"] = enc["attention_mask"].squeeze(0)

        return result
