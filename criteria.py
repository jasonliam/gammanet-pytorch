import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# batch-wise dice score
# source: https://discuss.pytorch.org/t/calculating-dice-coefficient/44154
def dice_coeff(pred, target, smooth=1.):
    m1 = pred.reshape(-1).float()  # Flatten
    m2 = target.reshape(-1).float()  # Flatten
    intersection = (m1 * m2).sum().float()
    return (2. * intersection + smooth) / (m1.sum() + m2.sum() + smooth)

# multi-label dice score
def dice_coeff_mc(pred, target, drop_neg_label=False, smooth=1.):
    num_labels = pred.size(1)
    dice = torch.zeros(num_labels)
    for i in range(num_labels):
        dice[i] = dice_coeff(pred[:, i], (target == i))
    if drop_neg_label:
        return torch.mean(dice[1:])
    else:
        return torch.mean(dice)

# cross entropy
def cross_entropy(predictions, targets, epsilon=1e-12):
    predictions = torch.clamp(predictions, epsilon, 1. - epsilon)
    N = predictions.shape[0]
    ce = -torch.sum(targets * torch.log(predictions+1e-9)) / N
    return ce
