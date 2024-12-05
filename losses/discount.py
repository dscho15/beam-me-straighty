import torch
import torch.nn as nn
from typing import Tuple, Callable


class DiscountedLoss(nn.Module):
    def __init__(self, error_function: Callable, discount: float = 0.99, scale: float = 1.0, invert_discount: bool = False):
        super().__init__()
        self.discount = discount
        self.error_function = error_function
        self.scale = scale
        self.invert_discount = invert_discount

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Computes the discounted L1 loss between the input and target tensors

        Args:
            input (torch.Tensor): Model prediction (B, n_preds, N)
            target (torch.Tensor): Target tensor (B, N)

        Returns:
            torch.Tensor: Discounted L1 loss
        """
        assert input.size(-1) == target.size(-1), "input and target must have the same size"

        n_preds = input.size(1)

        target = target.unsqueeze(1)
        
        if self.invert_discount is True:
            discount = self.discount ** torch.arange(n_preds, device=input.device).reshape(1, -1, 1)
        else:
            discount = self.discount ** (n_preds - torch.arange(n_preds, device=input.device).reshape(1, -1, 1) - 1)
                
        error = self.error_function(input, target)
        
        error = error * discount
        
        error = error.sum(dim=1)
        
        return error.mean() * self.scale

class DiscountedL1Loss(DiscountedLoss):
    def __init__(self, discount: float = 0.99, scale: float = 1.0, inverted_discount: bool = False):
        l1 = lambda x, y: torch.abs(x - y)
        super().__init__(l1, discount, scale, inverted_discount)

class DiscountedMSELoss(DiscountedLoss):
    def __init__(self, discount: float = 0.99, scale: float = 1.0, inverted_discount: bool = False):
        mse = lambda x, y: (x - y) ** 2
        super().__init__(mse, discount, scale, inverted_discount)

class DiscountedHuberLoss(DiscountedLoss):
    def __init__(self, discount: float = 0.99, scale: float = 1.0, inverted_discount: bool = False):
        huber = nn.SmoothL1Loss()
        super().__init__(huber, discount, scale, inverted_discount)

if __name__ == '__main__':
    loss = DiscountedMSELoss()
    input = torch.randn(5, 4, 8)
    target = torch.ones(5, 8)
    output = loss(input, target)
