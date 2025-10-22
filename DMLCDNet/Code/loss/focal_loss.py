import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class FocalLoss(nn.Module):
    """
    Adjust Focal Loss parameters to reduce over-focus on positive class.
    Lower alpha (positive class weight) to increase attention to negative class (background)
    and reduce false positives.
    """
    def __init__(self, alpha: float = 0.5,  
                 gamma: float = 2.0, 
                 ignore_index: int = -100):
        super().__init__()
        self.alpha = alpha 
        self.gamma = gamma
        self.ignore_index = ignore_index

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute Focal Loss between prediction and target"""
        ce_loss = F.cross_entropy(
            pred, target, ignore_index=self.ignore_index, reduction='none'
        )
        
        p_t = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - p_t) ** self.gamma * ce_loss
        
        if self.ignore_index != -100:
            mask = (target != self.ignore_index)
            focal_loss = focal_loss * mask.float()
        
        return focal_loss.mean()
