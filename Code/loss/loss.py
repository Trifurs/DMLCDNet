import torch
import torch.nn as nn
from loss.dice_loss import DiceLoss
from loss.boundary_loss import BoundaryLoss
from loss.focal_loss import FocalLoss
from loss.tversky_loss import TverskyLoss


class LandslideLoss(nn.Module):
    """Hybrid loss function for landslide change detection"""
    def __init__(
        self, 
        dice_weight: float = 1.0,
        boundary_weight: float = 0.5,
        focal_weight: float = 1.0,
        tversky_weight: float = 1.0,  
        ignore_index: int = -100
    ):
        super().__init__()
        self.dice_loss = DiceLoss(smooth=1e-5, ignore_index=ignore_index)
        self.boundary_loss = BoundaryLoss()
        self.focal_loss = FocalLoss(alpha=0.5, gamma=2.0, ignore_index=ignore_index)  
        self.tversky_loss = TverskyLoss(alpha=0.4, beta=0.6)  
        
        self.dice_weight = dice_weight
        self.boundary_weight = boundary_weight
        self.focal_weight = focal_weight
        self.tversky_weight = tversky_weight  

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> dict:
        """Compute combined loss components and total loss"""
        dice = self.dice_loss(pred, target)
        boundary = self.boundary_loss(pred, target)
        focal = self.focal_loss(pred, target)
        tversky = self.tversky_loss(pred, target)
        
        total_loss = (
            self.dice_weight * dice +
            self.boundary_weight * boundary +
            self.focal_weight * focal +
            self.tversky_weight * tversky  
        )
        
        return {
            'total': total_loss,
            'dice': dice,
            'boundary': boundary,
            'focal': focal,
            'tversky': tversky
        }
