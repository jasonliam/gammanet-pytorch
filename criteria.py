import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# batch-wise dice score
# source: https://discuss.pytorch.org/t/calculating-dice-coefficient/44154
def dice_coeff(pred, target, smooth = 1.):
    pred = nn.Sigmoid()(pred)
    num = pred.size(0)
    m1 = pred.view(num, -1).float()  # Flatten
    m2 = target.view(num, -1).float()  # Flatten
    intersection = (m1 * m2).sum().float()
    return (2. * intersection + smooth) / (m1.sum() + m2.sum() + smooth)

# multi-label dice score
def dice_coeff_mc(pred, target, drop_neg_label=False, smooth = 1.):
    num_labels = pred.size(1)
    dice = []
    for i in range(num_labels):
        dice += [dice_coeff(pred[:,i], (target==i))]
    dice = torch.stack(dice)
    if drop_neg_label:
        return torch.mean(dice[1:])
    else:
        return torch.mean(dice)

