"""
__init__.py – Export main utilities from src/
"""
from .datasets import get_cifar100_loaders, get_food101_loaders, get_20newsgroups_loaders, TextDataset, Flickr30kDataset, get_image_transforms  # noqa: F401
from .models import (get_resnet50, get_vit_b16,  # noqa: F401
                     GRUClassifier, get_distilbert,
                     CLIPZeroShotClassifier, CLIPFewShotClassifier)
from .train import train, train_one_epoch, evaluate  # noqa: F401
from .evaluate import get_predictions, compute_metrics, plot_confusion_matrix, plot_training_curves, compare_models  # noqa: F401
