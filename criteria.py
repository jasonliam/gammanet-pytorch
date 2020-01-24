import numpy as np
import torch
import torch.nn.functional as F


# source: https://discuss.pytorch.org/t/calculating-dice-coefficient/44154
def dice_coeff(pred, target):
    smooth = 1.
    num = pred.size(0)
    m1 = pred.view(num, -1).float()  # Flatten
    m2 = target.view(num, -1).float()  # Flatten
    intersection = (m1 * m2).sum().float()
    return (2. * intersection + smooth) / (m1.sum() + m2.sum() + smooth)

# multi-label dice score
def dice_coeff_mc(pred, target):
    smooth = 1.
    num = pred.size(0)
    num_labels = pred.size(1)
    dice = []
    for i in range(num_labels):
        m1 = pred[:,i].view(num, -1).float()  # Flatten
        m2 = (target.view(num, -1) == i).float()  # Flatten
        intersection = (m1 * m2).sum().float()
        dice += [((2. * intersection + smooth) / (m1.sum() + m2.sum() + smooth)).item()]
    return dice

