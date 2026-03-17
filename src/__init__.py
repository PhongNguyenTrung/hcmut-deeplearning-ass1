"""
__init__.py – Export main utilities from src/
"""
from .datasets import get_cifar100_loaders, get_food101_loaders, get_20newsgroups_loaders, TextDataset, Flickr30kDataset, get_image_transforms
from .models import (get_resnet50, get_efficientnet_b0, get_vit_b16, get_deit_small,
                     BiLSTMClassifier, GRUClassifier, get_distilbert, get_bert,
                     CLIPZeroShotClassifier, CLIPFewShotClassifier)
from .train import train, train_one_epoch, evaluate
from .evaluate import get_predictions, compute_metrics, plot_confusion_matrix, plot_training_curves, compare_models
