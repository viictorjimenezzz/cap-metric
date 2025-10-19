from typing import Optional
from collections import defaultdict

import torch
from pytorch_lightning.callbacks import Callback

from capmetric.metric import ApproximationCapacity
from capmetric.binary import get_pairing_fn


class ApproximationCapacityCallback(Callback):
    """
    Compute the CAP metric.

    Args:
        capmetric: ApproximationCapacity instance (handles beta optimization internally).
        output_name: key in `outputs` dict returned by *_step with the logits/losses to use.
        data_pairs: mapping like {"pairA": ("val_ds1","val_ds2"), ...}
        pairing_type: name understood by `get_pairing_fn`.
        log_every_n_epochs: only affects validation; test runs once per test loop.
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
        self.cache = defaultdict(list)

    def _get_loader_key(self, trainer, stage: str, dataloader_idx: int):
        attr = f"{stage}_dataloaders"  
        loaders = getattr(trainer, attr)
        if isinstance(loaders, dict):
            return list(loaders.keys())[dataloader_idx]
        return f"{stage}_{dataloader_idx}"

    def _accumulate(self, stage: str, trainer, outputs, dataloader_idx: int):
        key = self._get_loader_key(trainer, stage, dataloader_idx)
        if outputs is None:
            return
        tensor = outputs[self.output_name]
        self.cache[key].append(tensor.detach().cpu())

    def _compute_and_log(self, pl_module, stage: str):
        list_dset_key = list(self.cache.keys())

        for pair_name, (ds_name1, ds_name2) in self.data_pairs.items():
            self.capmetric.reset()

            if ds_name1 not in list_dset_key or ds_name2 not in list_dset_key:
                print(f"CAP [{stage}]: {pair_name} datasets not found in cache")
                continue

            loss1 = torch.cat(self.cache[ds_name1])
            loss2 = torch.cat(self.cache[ds_name2])

            indices1, indices2 = self.pairing_fn(loss1, loss2)
            loss1, loss2 = loss1[indices1], loss2[indices2]
            n = min(len(loss1), len(loss2))

            with torch.inference_mode(False):
                with torch.enable_grad():
                    self.capmetric.update(loss1[:n], loss2[:n])

            pl_module.log_dict(
                {f"cap/{stage}/{pair_name}": self.capmetric.compute()},
                prog_bar=False,
                on_step=False,
                on_epoch=True,
                logger=True,
                sync_dist=True,
            )

        self.cache = defaultdict(list)
        del loss1, loss2, indices1, indices2

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        if (pl_module.current_epoch + 1) % self.log_every_n_epochs != 0:
            return
        self._accumulate("val", trainer, outputs, dataloader_idx)

    def on_validation_epoch_end(self, trainer, pl_module):
        if (pl_module.current_epoch + 1) % self.log_every_n_epochs != 0:
            return
        self._compute_and_log(pl_module, stage="val")

    def on_test_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        self._accumulate("test", trainer, outputs, dataloader_idx)

    def on_test_epoch_end(self, trainer, pl_module):
        self._compute_and_log(pl_module, stage="test")
