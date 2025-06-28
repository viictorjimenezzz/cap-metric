from typing import Optional, Any, Callable
from tqdm import tqdm
import gc

import torch
from torch import Tensor, optim
from torch.utils.data import TensorDataset, DataLoader
from torchmetrics import Metric

from capmetric.kernel import ApproximationCapacityKernel
from capmetric.binary import (
    get_normalizer_fn as get_normalizer_fn_binary,
    get_energy_fn as get_energy_fn_binary,
    get_regularizer_fn as get_regularizer_fn_binary
)


class CAP(Metric):
    """
    Approximation Capacity (CAP) Metric implemented with torchmetrics.

    Args:
        kernel: The ApproximationCapacityKernel instance
        dataset: The ReconstructionLoss dataset
        batch_size: Batch size for processing the dataset
        n_epochs: Number of optimization epochs
        learning_rate: Learning rate for beta optimization
        compute_on_step: Whether to compute the metric on each step
        dist_sync_on_step: Synchronize metric state across processes at each step
        process_group: DDP process group
        dist_sync_fn: Function to synchronize metrics across processes
    """

    is_differentiable = False
    higher_is_better = True
    full_state_update = False

    def __init__(
        self,
        beta0: Optional[float] = 1.0,
        normalization_type: Optional[str] = "minmax",
        normalization_params: Optional[dict] = None,
        energy_type: Optional[str] = "baseline",
        energy_params: Optional[dict] = None,
        regularization_type: Optional[str] = "none",
        regularization_params: Optional[dict] = None,
        binary: Optional[bool] = True,
        lr: Optional[float] = 0.01,
        n_epochs: Optional[int] = 5,
        batch_size: Optional[int] = 64,
        process_group: Optional[Any] = None,
        dist_sync_fn: Optional[Callable] = None,
    ):  
        self.beta0 = beta0
        if self.beta0 < 0.0:
            raise ValueError("'beta' must be non-negative.")
        
        super().__init__(
            dist_sync_on_step=False,
            process_group=process_group,
            dist_sync_fn=dist_sync_fn,
        )

        # Parameters for energy and regularization functions:
        self.binary = binary
        self.normalization_type = normalization_type
        self.normalization_params = normalization_params or {}
        self.energy_type = energy_type
        self.energy_params = energy_params or {}
        self.regularization_type = regularization_type
        self.regularization_params = regularization_params or {}

        # Learning parameters:
        self.n_epochs = n_epochs
        self.learning_rate = lr
        self.batch_size = batch_size

        # Initialize CAP:
        self.add_state("cap", default=torch.tensor(0.0), dist_reduce_fx="sum")

        # Normalizer of the loss:
        self.normalizer_fn = self._get_normalizer_fn()

    def _get_normalizer_fn(self):
        if self.binary:
            normalizer_fn = get_normalizer_fn_binary(
                self.normalization_type,
                self.normalization_params
            )
        else:
            raise NotImplementedError("Non-binary support is not implemented yet.")
        
        return normalizer_fn
    
    def _get_energy_fn(self):
        if self.binary:
            energy_fn = get_energy_fn_binary(
                self.energy_type,
                self.energy_params
            )
        else:
            raise NotImplementedError("Non-binary support is not implemented yet.")
        
        return energy_fn
    
    def _get_regularizer_fn(self):
        if self.binary:
            regularizer_fn = get_regularizer_fn_binary(
                self.regularization_type,
                self.regularization_params
            )
        else:
            raise NotImplementedError("Non-binary support is not implemented yet.")

        return regularizer_fn

    def update(self, loss1: Tensor, loss2: Tensor, **kwargs):
        """Optimize beta parameter over multiple epochs."""
        self.reset()

        # Normalize the losses:
        loss1, loss2 = self.normalizer_fn(loss1, loss2)

        # Add loss mean and std to the energy parameters:
        combined = torch.cat([loss1, loss2], dim=0)
        self.energy_params.update({
            "mean": combined.mean().item(),
            "std": combined.std().item(),
        })
        del combined; gc.collect()

        # Initialize kernel with updated energy function:
        kernel = ApproximationCapacityKernel(
            beta0=self.beta0,
            energy_fn=self._get_energy_fn(),
        )
        dev = kernel.beta.device
        optimizer = optim.Adam([kernel.beta], lr=self.learning_rate)

        dataloader = DataLoader(
            TensorDataset(loss1, loss2),
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=0,
        )
        # Optimization loop:
        best_beta = 0.0
        best_cap = -float("inf")
        for _ in tqdm(
            range(self.n_epochs), desc="CAP Optimization Progress", unit="epoch"
        ):
            kernel.reset()

            with torch.set_grad_enabled(True):
                for batch in dataloader:
                    loss1, loss2 = batch[0].to(dev), batch[1].to(dev)

                    # Maximize capacity
                    optimizer.zero_grad()
                    loss = -kernel(loss1, loss2)
                    loss.backward()
                    optimizer.step()

            # Update best beta if this epoch is better
            if kernel.cap.item() > best_cap:
                best_cap = kernel.cap.item()
                best_beta = kernel.beta.clone().detach()

        # Perform a final evaluation with the best beta
        with torch.no_grad():
            kernel.reset()
            for batch in dataloader:
                loss1, loss2 = batch[0].to(dev), batch[1].to(dev)
                _ = kernel.evaluate(loss1, loss2, beta=best_beta)

            # Set a value to the metric state
            self.cap.fill_(kernel.cap.item())
        return

    def compute(self) -> Tensor:
        """Return the computed CAP value."""
        return self.cap.item()
