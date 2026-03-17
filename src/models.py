"""
models.py – Model architectures
CO5085 – Assignment 1
"""

import torch
import torch.nn as nn
from torchvision import models
from transformers import (
    AutoModelForSequenceClassification,
    ViTForImageClassification,
    ViTModel,
)
import clip


# ─────────────────────────────────────────────
# 1. CNN Models (ResNet-50, EfficientNet-B0)
# ─────────────────────────────────────────────

def get_resnet50(num_classes: int, pretrained: bool = True, freeze_backbone: bool = False):
    """ResNet-50 with custom classification head."""
    weights = models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
    model = models.resnet50(weights=weights)

    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False

    # Replace the final FC layer
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(in_features, num_classes)
    )
    return model


def get_efficientnet_b0(num_classes: int, pretrained: bool = True, freeze_backbone: bool = False):
    """EfficientNet-B0 with custom classification head."""
    weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
    model = models.efficientnet_b0(weights=weights)

    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False

    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(in_features, num_classes)
    )
    return model


# ─────────────────────────────────────────────
# 2. ViT Models
# ─────────────────────────────────────────────

def get_vit_b16(num_classes: int, pretrained: bool = True, freeze_backbone: bool = False):
    """Vision Transformer ViT-B/16 fine-tuned for classification."""
    model_name = "google/vit-base-patch16-224-in21k"
    model = ViTForImageClassification.from_pretrained(
        model_name,
        num_labels=num_classes,
        ignore_mismatched_sizes=True,
    )
    if freeze_backbone:
        for name, param in model.named_parameters():
            if "classifier" not in name:
                param.requires_grad = False
    return model


def get_deit_small(num_classes: int, pretrained: bool = True, freeze_backbone: bool = False):
    """DeiT-Small for classification."""
    model_name = "facebook/deit-small-patch16-224"
    model = ViTForImageClassification.from_pretrained(
        model_name,
        num_labels=num_classes,
        ignore_mismatched_sizes=True,
    )
    if freeze_backbone:
        for name, param in model.named_parameters():
            if "classifier" not in name:
                param.requires_grad = False
    return model


# ─────────────────────────────────────────────
# 3. RNN / BiLSTM (Text)
# ─────────────────────────────────────────────

class BiLSTMClassifier(nn.Module):
    """Bidirectional LSTM text classifier."""

    def __init__(self, vocab_size: int, embed_dim: int = 300, hidden_dim: int = 256,
                 num_layers: int = 2, num_classes: int = 20, dropout: float = 0.3,
                 pad_idx: int = 0, pretrained_embeddings=None):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        if pretrained_embeddings is not None:
            self.embedding.weight = nn.Parameter(torch.tensor(pretrained_embeddings, dtype=torch.float))

        self.lstm = nn.LSTM(
            embed_dim, hidden_dim, num_layers=num_layers,
            bidirectional=True, dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, input_ids, attention_mask=None, **kwargs):
        # input_ids: (B, L)
        embedded = self.dropout(self.embedding(input_ids))  # (B, L, E)
        output, (hidden, _) = self.lstm(embedded)
        # Concatenate forward and backward hidden states
        hidden_cat = torch.cat([hidden[-2], hidden[-1]], dim=1)  # (B, H*2)
        return self.fc(self.dropout(hidden_cat))


class GRUClassifier(nn.Module):
    """Bidirectional GRU text classifier."""

    def __init__(self, vocab_size: int, embed_dim: int = 300, hidden_dim: int = 256,
                 num_layers: int = 2, num_classes: int = 20, dropout: float = 0.3,
                 pad_idx: int = 0):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.gru = nn.GRU(
            embed_dim, hidden_dim, num_layers=num_layers,
            bidirectional=True, dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, input_ids, attention_mask=None, **kwargs):
        embedded = self.dropout(self.embedding(input_ids))
        output, hidden = self.gru(embedded)
        hidden_cat = torch.cat([hidden[-2], hidden[-1]], dim=1)
        return self.fc(self.dropout(hidden_cat))


# ─────────────────────────────────────────────
# 4. Transformer (HuggingFace) for Text
# ─────────────────────────────────────────────

def get_distilbert(num_classes: int, model_name: str = "distilbert-base-uncased",
                   freeze_backbone: bool = False):
    """DistilBERT for sequence classification."""
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=num_classes
    )
    if freeze_backbone:
        for name, param in model.named_parameters():
            if "classifier" not in name and "pre_classifier" not in name:
                param.requires_grad = False
    return model


def get_bert(num_classes: int, model_name: str = "bert-base-uncased",
             freeze_backbone: bool = False):
    """BERT for sequence classification."""
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=num_classes
    )
    if freeze_backbone:
        for name, param in model.named_parameters():
            if "classifier" not in name and "pooler" not in name:
                param.requires_grad = False
    return model


# ─────────────────────────────────────────────
# 5. Multimodal – CLIP-based Classifiers
# ─────────────────────────────────────────────

class CLIPZeroShotClassifier:
    """
    Zero-shot classifier using CLIP.
    Usage:
        clf = CLIPZeroShotClassifier(class_names)
        probs = clf.predict(images)   # images: list of PIL Images
    """

    def __init__(self, class_names: list, model_name: str = "ViT-B/32",
                 device: str = "cuda" if torch.cuda.is_available() else "cpu",
                 prompt_template: str = "a photo of a {}"):
        self.device = device
        self.class_names = class_names
        self.model, self.preprocess = clip.load(model_name, device=device)

        # Encode text prompts
        prompts = [prompt_template.format(c) for c in class_names]
        tokens = clip.tokenize(prompts).to(device)
        with torch.no_grad():
            self.text_features = self.model.encode_text(tokens)
            self.text_features /= self.text_features.norm(dim=-1, keepdim=True)

    @torch.no_grad()
    def predict(self, images):
        """images: list of PIL Images. Returns (probs, preds)."""
        processed = torch.stack([self.preprocess(img) for img in images]).to(self.device)
        image_features = self.model.encode_image(processed)
        image_features /= image_features.norm(dim=-1, keepdim=True)

        logits = (100.0 * image_features @ self.text_features.T)
        probs = logits.softmax(dim=-1)
        preds = probs.argmax(dim=-1)
        return probs.cpu().numpy(), preds.cpu().numpy()


class CLIPFewShotClassifier(nn.Module):
    """
    Few-shot classifier: CLIP image encoder + trainable linear head.
    Train only the linear head on K examples per class.
    """

    def __init__(self, num_classes: int, clip_model_name: str = "ViT-B/32",
                 device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        super().__init__()
        self.device = device
        self.clip_model, self.preprocess = clip.load(clip_model_name, device=device)

        # Freeze CLIP
        for param in self.clip_model.parameters():
            param.requires_grad = False

        # Trainable linear head
        embed_dim = self.clip_model.visual.output_dim
        self.classifier = nn.Linear(embed_dim, num_classes)

    def encode_images(self, images):
        processed = torch.stack([self.preprocess(img) for img in images]).to(self.device)
        with torch.no_grad():
            features = self.clip_model.encode_image(processed).float()
            features /= features.norm(dim=-1, keepdim=True)
        return features

    def forward(self, pixel_values):
        with torch.no_grad():
            features = self.clip_model.encode_image(pixel_values).float()
            features /= features.norm(dim=-1, keepdim=True)
        return self.classifier(features)
