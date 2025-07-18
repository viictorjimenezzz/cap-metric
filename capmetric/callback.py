from typing import Optional
from collections import defaultdict

import torch
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.utilities.memory import garbage_collection_cuda

from capmetric.metric import ApproximationCapacity
from capmetric.binary import get_pairing_fn


class ApproximationCapacityCallback(Callback):
    """
    Callback to compute the CAP (Conditional Average Precision) metric during validation.

    Args:
        beta0 (float): Initial value for the beta parameter.
    """

    def __init__(
        self,
        capmetric: ApproximationCapacity,
        output_name: str,
        data_pairs: dict,
        pairing_type: str,
        log_every_n_epochs: Optional[int] = 1,
    ):
        super().__init__()
        self.capmetric = capmetric
        self.output_name = output_name
        self.data_pairs = data_pairs
        self.pairing_fn = get_pairing_fn(pairing_type)
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
        for pair_name, (ds_name1, ds_name2) in self.data_pairs.items():
            self.capmetric.reset()
            if ds_name1 not in list_dset_key or ds_name2 not in list_dset_key:
                print(f"CAP: {pair_name} datasets not found in cache")
                continue

            # Concatenate losses and pair:
            loss1 = torch.cat(self.cache[ds_name1])
            loss2 = torch.cat(self.cache[ds_name2])
            indices1, indices2 = self.pairing_fn(loss1, loss2)
            loss1, loss2 = loss1[indices1], loss2[indices2]

            # Compute the metric:
            self.capmetric.update(
                loss1[: min(len(loss1), len(loss2))],
                loss2[: min(len(loss1), len(loss2))]
            )
            pl_module.log_dict(
                {
                    f"cap/{pair_name}": self.capmetric.compute(),
                },
                prog_bar=False,
                on_step=False,
                on_epoch=True,
                logger=True,
                sync_dist=True,
            )
            
        self.cache = defaultdict(list)
        del loss1, loss2, indices1, indices2; garbage_collection_cuda()
