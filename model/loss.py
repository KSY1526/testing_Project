import torch.nn.functional as F
import torch.nn as nn

def nll_loss(output, target):
    return F.nll_loss(output, target)

def movie_loss(output, target):
    _loss = nn.BCELoss()
    return _loss(output, target)