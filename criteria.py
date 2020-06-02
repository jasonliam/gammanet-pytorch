import torch

'''
Weights Evaluated on the training dataset. For now, We're just supporting LV (Class 3)
(1236051, 1328344, 1298596, 98189937) - Counts for (C1, C2, C3, C0)
(100816877, 100724584, 100754332, 3862991) - (D1, D2, D3, D0) Negative Counts [ Total - (C1, C2, C3, C0)]
(81.56368709705343, 75.82718331998338, 77.58712640420886) Weights for BCE (C1, C2, C3)
'''


def get_criterion(config_data):
    if config_data['model']['criterion'] == 'bce':
        return torch.nn.BCEWithLogitsLoss()
    elif config_data['model']['criterion'] == 'weighted':
        return torch.nn.BCEWithLogitsLoss(pos_weight=torch.Tensor([77.58712640420886]).cuda())
    elif config_data['model']['criterion'] == 'dice':
        return soft_dice_loss
    else:
        raise Exception('Invalid Criterion received')


def soft_dice_loss(pred, target, smooth=1.):
    pred = torch.sigmoid(pred)
    m1 = pred.reshape(-1).float()  # Flatten
    m2 = target.reshape(-1).float()  # Flatten
    intersection = (m1 * m2).sum().float()
    return 1 - (2. * intersection + smooth) / (m1.sum() + m2.sum() + smooth)


# batch-wise dice score
# source: https://discuss.pytorch.org/t/calculating-dice-coefficient/44154
def dice_coeff(pred, target, smooth=1.):
    pred = (torch.sigmoid(pred) > 0.5).int()
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
    ce = -torch.sum(targets * torch.log(predictions + 1e-9)) / N
    return ce
