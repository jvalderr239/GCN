import dataclasses
import logging
import logging.config
from datetime import datetime
from typing import Any, Dict, Optional, Tuple

import torch
import torch.optim.lr_scheduler as lr_scheduler
from torch import nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter

from .datasets import TrajectoryDataset
from .metrics import bivariate_graph_loss
from .utils import get_project_root

# setup logger
ROOT_DIR = get_project_root()
log_file_path = ROOT_DIR / "logging.conf"
logging.config.fileConfig(str(log_file_path))
log = logging.getLogger(__name__)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


@dataclasses.dataclass
class Trainer:
    # Model parameters
    num_events: int = 3
    num_nodes: int = 22
    num_spatial: int = 1
    num_temporal: int = 1
    num_features: int = 2
    output_feat: int = 5
    seq_len: int = 8
    pred_seq_len: int = 12
    kernel_size: int = 3
    cnn_name: Optional[str] = None
    pretrained: bool = True
    cnn_dropout: float = 0.3

    # Train parameters
    lr: float = 0.001
    checkpoint_dir = f"{str(ROOT_DIR)}/checkpoint/{datetime.now()}/"
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
) -> DataLoader:
    dataset = generate_trajectory_dataset(datatype=datatype, root_dir=root_dir)
    return DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers
    )


def generate_trajectory_dataset(
    datatype: str, root_dir: Optional[str] = None
) -> Dataset:
    if datatype not in ("train", "validation", "test"):
        raise ValueError("Invalid datatype. Expected one of(train, validation, test)")

    if root_dir is None:
        root_dir = str(ROOT_DIR / "resources") + "/"
        log.debug(f"Defaulting to root: {root_dir}")

    return TrajectoryDataset(datatype=datatype, root_dir=root_dir)


def criterion(outputs: Tuple[torch.Tensor, torch.Tensor], truth_labels: Dict[str, Any]):
    loss_map: Dict[str, nn.NLLLoss] = {
        "node_index": nn.BCELoss(),
        "event_type": nn.BCELoss(),
        "time_of_event": nn.BCELoss(),
    }

    losses = 0

    # Trajectory loss
    V_pred, simo = outputs
    V_truth, _ = truth_labels.pop("trajectory"), truth_labels.pop("graph")
    V_truth = V_truth.squeeze()
    V_pred = V_pred.squeeze()

    losses += bivariate_graph_loss(V_pred, V_truth)

    # CNN Prediction loss
    for _, (key, pred) in enumerate(simo.items()):
        losses += loss_map[key](pred, truth_labels[key].to(device))

    return losses


def train_one_epoch(
    epoch_index: int,
    model: nn.Module,
    training_loader: DataLoader,
    optimizer: Optimizer,
    tb_writer: SummaryWriter,
):
    running_loss = 0.0
    last_loss = 0.0

    # Here, we use enumerate(training_loader) instead of
    # iter(training_loader) so that we can track the batch
    # index and do some intra-epoch reporting
    for i, batch_data in enumerate(training_loader):
        # Every data instance is an input + label pair

        V_obs, A_obs = batch_data["data"]
        truth_labels = batch_data["labels"]

        # Zero your gradients for every batch!
        optimizer.zero_grad()

        # V_obs = batch,seq,node,feat
        # V_obs_tmp = batch,feat,seq,node
        V_obs_tmp = V_obs.permute(0, 3, 1, 2)

        # Make predictions for this batch
        V_pred, _, simo = model(V_obs_tmp, A_obs.squeeze())

        V_pred = V_pred.permute(0, 2, 3, 1)

        # Compute the loss and its gradients
        loss = criterion((V_pred, simo), truth_labels.copy())
        loss.backward()

        # Adjust learning weights
        optimizer.step()

        # Gather data and report
        running_loss += loss.item()
        if i % 1000 == 999:
            last_loss = running_loss / 1000  # loss per batch
            log.debug(f"batch {i + 1} loss: {last_loss}")
            tb_x = epoch_index * len(training_loader) + i + 1
            tb_writer.add_scalar("Loss/train", last_loss, tb_x)
            running_loss = 0.0

    return last_loss
