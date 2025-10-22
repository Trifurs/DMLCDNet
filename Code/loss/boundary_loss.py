import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class BoundaryLoss(nn.Module):
    """Boundary loss to enhance attention on edges"""
    def __init__(self, theta0: float = 3.0, theta1: float = 5.0):
        super().__init__()
        self.theta0 = theta0
        self.theta1 = theta1
        
        # Initialize edge detection kernels (Sobel operator)
        self.kernel_x = torch.tensor([
            [-1., 0., 1.],
            [-2., 0., 2.],
            [-1., 0., 1.]
        ]).view(1, 1, 3, 3)
        
        self.kernel_y = torch.tensor([
            [-1., -2., -1.],
            [0., 0., 0.],
            [1., 2., 1.]
        ]).view(1, 1, 3, 3)

    def get_boundary(self, x: torch.Tensor) -> torch.Tensor:
        """Get the boundary of the input mask"""
        b, c, h, w = x.shape
        
        if c > 1:
            x = torch.argmax(x, dim=1, keepdim=True).float()
            c = x.shape[1]
        
        kernel_x = self.kernel_x.to(x.device).repeat(c, 1, 1, 1)
        kernel_y = self.kernel_y.to(x.device).repeat(c, 1, 1, 1)
        
        grad_x = F.conv2d(x, kernel_x, padding=1, groups=c)
        grad_y = F.conv2d(x, kernel_y, padding=1, groups=c)
        
        boundary = torch.sqrt(grad_x ** 2 + grad_y ** 2)
        boundary = (boundary > 0.1).float()
        return boundary

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute boundary loss between prediction and target"""
        pred_softmax = F.softmax(pred, dim=1)
        pred_boundary = self.get_boundary(pred_softmax)
        
        target_onehot = F.one_hot(target, num_classes=pred.shape[1]).permute(0, 3, 1, 2).float()
        target_boundary = self.get_boundary(target_onehot)
        
        bce_loss = F.binary_cross_entropy_with_logits(
            pred_boundary, target_boundary, reduction='none'
        )
        
        distance = torch.abs(pred_boundary - target_boundary)
        weight = torch.where(
            distance < self.theta0, 
            torch.exp(-(distance ** 2) / (2 * self.theta1 ** 2)),
            torch.zeros_like(distance)
        )
        
        return (weight * bce_loss).mean()
