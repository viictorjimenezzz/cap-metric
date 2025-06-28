from typing import Optional, Callable
import torch
from torch import Tensor, nn


class ApproximationCapacityKernel(nn.Module):
    """Computes the approximation capacity for a binary classification problem."""

    def __init__(
            self,
            beta0: float,
            energy_fn: Callable,
        ):  
        super().__init__()
        self.energy_fn = energy_fn
        
        # Learnable parameters
        self.beta = torch.nn.Parameter(
            torch.tensor([beta0], dtype=torch.float), requires_grad=True
        )
        self.cap = torch.tensor([0.0], dtype=torch.float)

    def compute_mutual_information(
        self,
        prob1: Tensor,
        prob2: Tensor,
        beta: Optional[float] = None
    ) -> Tensor:
        """
        Compute mutual information for relaxed hypothesis space (factorized form).
        This is the original implementation but with different energy functions.
        """
        beta = self.beta if beta is None else beta

        # Compute energy:
        n = len(prob1)
        e1_0 = self.energy_fn(prob1, torch.zeros(n, dtype=torch.int8))
        e1_1 = self.energy_fn(prob1, torch.ones(n, dtype=torch.int8))
        e2_0 = self.energy_fn(prob2, torch.zeros(n, dtype=torch.int8))
        e2_1 = self.energy_fn(prob2, torch.ones(n, dtype=torch.int8))

        # Factorized mutual information computation
        num = torch.log(
            torch.exp(-beta * (e1_0 + e2_0)) + torch.exp(-beta * (e1_1 + e2_1))
        )
        den = torch.log(
            (torch.exp(-beta * e1_0) + torch.exp(-beta * e1_1))
            * (torch.exp(-beta * e2_0) + torch.exp(-beta * e2_1))
        )
        mi = torch.sum(num - den, dim=0)
        return mi

    def forward(self, loss1: Tensor, loss2: Tensor):
        """Forward pass with gradient computation."""
        self.beta.data.clamp_(min=0.0)
        with torch.set_grad_enabled(True):
            cap = self.compute_mutual_information(loss1, loss2)
            self.cap = self.cap + cap.detach().to(self.cap.device)
            return cap

    def evaluate(self, loss1: Tensor, loss2: Tensor, beta: float):
        """Evaluation without gradient computation."""
        with torch.set_grad_enabled(False):
            cap = self.compute_mutual_information(loss1, loss2, beta=beta)
            self.cap = self.cap + cap
            return cap

    def reset(self):
        """Reset accumulated state for a new computation sequence."""
        self.cap = torch.tensor([0.0], dtype=torch.float)

    @property
    def module(self):
        """Returns the kernel itself. It helps the kernel be accessed in both DDP and non-DDP mode."""
        return self