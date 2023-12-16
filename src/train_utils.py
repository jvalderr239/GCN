import dataclasses
import logging
import logging.config
from typing import Any, Optional

import torch
from torch import nn
from torch.optim import Optimizer, lr_scheduler
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from .datasets import TrajectoryDataset
from .metrics import criterion
from .utils import get_project_root

# setup logger
ROOT_DIR = get_project_root()
log_file_path = ROOT_DIR / "logging.conf"
logging.config.fileConfig(str(log_file_path))
log = logging.getLogger(__name__)


@dataclasses.dataclass
class Trainer:
    # Model parameters
    num_events: int = 2
    num_spatial: int = 1
    num_temporal: int = 5
    output_feat: int = 5
    seq_len: int = 25
    pred_seq_len: int = 30
    kernel_size: int = 3
    cnn_name: Optional[str] = None
    pretrained: bool = True
    cnn_dropout: float = 0.3
    clip: Optional[float] = None

    # Dataset parameters
    num_features = 21
    num_spatial_features = 7
    num_nodes = 22

    # Train parameters
    lr: float = 0.01
    root_dir: str = str(ROOT_DIR / "resources") + "/"

    def update(self, arg, value):
        setattr(self, arg, value)

    def get_scheduler(self, optimizer):
        return lr_scheduler.ExponentialLR(optimizer, gamma=0.2)


def generate_dataloader(
    datatype: str,
    batch_size: int,
    shuffle=False,
    num_workers: int = 0,
    root_dir: Optional[str] = None,
    **kwargs,
) -> DataLoader:
    dataset = generate_trajectory_dataset(
        datatype=datatype, root_dir=root_dir, **kwargs
    )
    return DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers
    )


def generate_trajectory_dataset(
    datatype: str, obs_len: int, pred_len: int, root_dir: Optional[str] = None
) -> Dataset:
    if datatype not in ("train", "validation", "test"):
        raise ValueError("Invalid datatype. Expected one of(train, validation, test)")

    if root_dir is None:
        root_dir = str(ROOT_DIR / "resources") + "/"
        log.debug(f"Defaulting to root: {root_dir}")

    return TrajectoryDataset(
        datatype=datatype, root_dir=root_dir, obs_len=obs_len, pred_len=pred_len
    )


def checkpoint(model, filename):
    torch.save(model.state_dict(), filename)


def resume(model, filename):
    model.load_state_dict(torch.load(filename))


def train_one_epoch(
    epoch_index: int,
    model: nn.Module,
    training_loader: DataLoader,
    optimizer: Optimizer,
    tb_writer: SummaryWriter,
    clip: Optional[float],
    device: Any,
):
    running_loss = 0.0
    last_loss = 0.0
    n_total_steps = len(training_loader)

    # Here, we use enumerate(training_loader) instead of
    # iter(training_loader) so that we can track the batch
    # index and do some intra-epoch reporting
    for i, batch_data in enumerate((pbar := tqdm(training_loader))):
        # Every data instance is an input + label pair
        pbar.set_description(f"{i+1}/{n_total_steps} batches")
        V_obs, A_obs = batch_data["data"]
        truth_labels = batch_data["labels"]

        # Zero your gradients for every batch!
        optimizer.zero_grad()

        # V_obs = batch,seq,node,feat
        # V_obs_tmp = batch,feat,seq,node
        V_obs_tmp = V_obs.permute(0, 3, 1, 2)

        # Make predictions for this batch
        V_pred, _, simo = model(V_obs_tmp, A_obs.squeeze())

        # Compute the loss and its gradients
        loss = criterion((V_pred, simo), truth_labels.copy(), device)
        loss.backward()
        if clip is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip)
        # Adjust learning weights
        optimizer.step()

        # Gather data and report
        running_loss += loss.item()

        if (i + 1) % (int(n_total_steps)) == 0:
            last_loss = running_loss / 1000  # loss per batch
            log.debug(f"batch {i + 1} loss: {last_loss}")
            tb_x = epoch_index * len(training_loader) + i + 1
            tb_writer.add_scalar("Loss/train", last_loss, tb_x)
            running_loss = 0.0

    return last_loss