import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class DiceLoss(nn.Module):
    """Dice Loss to handle class imbalance"""
    def __init__(self, smooth: float = 1e-5, ignore_index: int = -100):
        super().__init__()
        self.smooth = smooth
        self.ignore_index = ignore_index

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute Dice loss between prediction and target"""
        if self.ignore_index != -100:
            mask = (target != self.ignore_index)
            pred = pred * mask.unsqueeze(1).float()
            target = target * mask
        
        num_classes = pred.shape[1]
        pred_softmax = F.softmax(pred, dim=1)
        
        total_loss = 0.0
        for cls in range(num_classes):
            pred_flat = pred_softmax[:, cls].contiguous().view(-1)
            target_flat = (target == cls).float().contiguous().view(-1)
            
            intersection = (pred_flat * target_flat).sum()
            union = pred_flat.sum() + target_flat.sum()
            
            dice = (2. * intersection + self.smooth) / (union + self.smooth)
            total_loss += (1. - dice)
        
        return total_loss / num_classes
