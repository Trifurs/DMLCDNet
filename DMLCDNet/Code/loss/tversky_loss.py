import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class TverskyLoss(nn.Module):
    """
    Adjust Tversky Loss parameters to increase penalty on false positives.
    Increase beta (false positive weight) and decrease alpha (false negative weight).
    """
    def __init__(self, alpha: float = 0.4, 
                 beta: float = 0.6,        
                 smooth: float = 1e-5):
        super().__init__()
        self.alpha = alpha  # False negative weight
        self.beta = beta    # False positive weight
        self.smooth = smooth

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute Tversky loss between prediction and target"""
        num_classes = pred.shape[1]
        pred_softmax = F.softmax(pred, dim=1)
        
        total_loss = 0.0
        for cls in range(num_classes):
            pred_flat = pred_softmax[:, cls].contiguous().view(-1)
            target_flat = (target == cls).float().contiguous().view(-1)
            
            tp = (pred_flat * target_flat).sum()
            fp = ((1 - target_flat) * pred_flat).sum()
            fn = (target_flat * (1 - pred_flat)).sum()
            
            tversky = (tp + self.smooth) / (tp + self.alpha * fn + self.beta * fp + self.smooth)
            total_loss += (1 - tversky)
        
        return total_loss / num_classes
