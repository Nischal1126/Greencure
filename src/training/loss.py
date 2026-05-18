import torch
import torch.nn as nn


def get_loss_function(label_smoothing: float = 0.1):
    return nn.CrossEntropyLoss(
        label_smoothing=label_smoothing
    )