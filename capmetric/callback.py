from typing import Optional
from collections import defaultdict

import torch
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.utilities.memory import garbage_collection_cuda

from capmetric.metric import ApproximationCapacity


class ApproximationCapacityCallback(Callback):
    """
    Callback to compute the CAP (Conditional Average Precision) metric during training.

    Args:
        beta0 (float): Initial value for the beta parameter.
    """

    def __init__(
        self,
        capmetric: ApproximationCapacity,
        output_name: str,
        log_every_n_epochs: Optional[int] = 1,
    ):
        super().__init__()
        self.capmetric = capmetric
        self.output_name = output_name
        self.log_every_n_epochs = log_every_n_epochs

        # Cache the outputs
        self.cache = defaultdict(list)

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):  
        if (pl_module.current_epoch + 1) % self.log_every_n_epochs != 0:
            return

        dset_key = list(getattr(trainer, f"val_dataloaders").keys())[dataloader_idx]
        self.cache[dset_key].append(outputs[self.output_name].detach().cpu())

    def on_validation_epoch_end(self, trainer, pl_module):
        """Compute CAP for the two first datasets"""

        if (pl_module.current_epoch + 1) % self.log_every_n_epochs != 0:
            return

        # Obtain the loss from the cache:
        list_dset_key = list(self.cache.keys())
        loss1 = torch.cat(self.cache[list_dset_key[0]])
        loss2 = torch.cat(self.cache[list_dset_key[1]])
        self.cache = defaultdict(list) # reset
        garbage_collection_cuda()

        # Compute the metric:
        self.capmetric.update(loss1[: min(len(loss1), len(loss2))], loss2[: min(len(loss1), len(loss2))])
        pl_module.log_dict(
            {
                "cap": self.capmetric.compute(),
            },
            prog_bar=False,
            on_step=False,
            on_epoch=True,
            logger=True,
            sync_dist=True,
        )
