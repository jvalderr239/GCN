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
from .metrics import calc_accuracy, criterion
from .utils import get_project_root

# setup logger
ROOT_DIR = get_project_root()
log_file_path = ROOT_DIR / "logging.conf"
logging.config.fileConfig(str(log_file_path))
log = logging.getLogger("trainer")


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
    cnn_name: Optional[str] = "inceptionv3"
    pretrained: bool = True
    cnn_dropout: float = 0.3
    clip: Optional[float] = None

    # Dataset parameters
    num_features = 21
    num_spatial_features = 7
    num_nodes = 22

    # Train parameters
    lr: float = 0.01

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
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0 if datatype == "train" else 2,
        pin_memory=True,
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
    running_time_acc = 0.0
    running_node_acc = 0.0
    running_event_acc = 0.0

    # Make sure gradient tracking is on, and do a pass over the data
    model.train(True)

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
        V_pred, _, simo = model(V_obs_tmp.to(device), A_obs.squeeze().to(device))

        # Compute the loss and its gradients
        loss = criterion((V_pred, simo), truth_labels.copy(), device)
        loss.backward()

        if clip is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip)
        # Adjust learning weights
        optimizer.step()

        # Gather data and report
        running_loss += loss.item()

        # Compute accuracy
        event_acc, node_acc, time_acc = calc_accuracy(simo, truth_labels)
        running_event_acc += event_acc
        running_node_acc += node_acc
        running_time_acc += time_acc
        if (i + 1) % (len(training_loader)) == 0:
            last_loss = running_loss / 1000  # loss per batch
            log.debug(f"batch {i + 1} loss: {last_loss}")
            tb_x = epoch_index * len(training_loader) + i + 1
            tb_writer.add_scalar("Loss/train", last_loss, tb_x)
            running_loss = 0.0

    avg_ve_acc = 100 * running_event_acc / len(training_loader)
    avg_vn_acc = 100 * running_node_acc / len(training_loader)
    avg_vt_acc = 100 * running_time_acc / len(training_loader)

    return last_loss, avg_ve_acc, avg_vn_acc, avg_vt_acc


def validate(
    model: nn.Module,
    validation_loader: DataLoader,
    device: Any,
):
    running_vloss = 0.0
    running_time_acc = 0.0
    running_node_acc = 0.0
    running_event_acc = 0.0
    # Set the model to evaluation mode, disabling dropout and using population
    # statistics for batch normalization.
    model.eval()

    # Disable gradient computation and reduce memory consumption.
    with torch.no_grad():
        for vbatchdata in validation_loader:
            V_obs, A_obs = vbatchdata["data"]
            truth_labels = vbatchdata["labels"]

            # V_obs = batch,seq,node,feat
            # V_obs_tmp = batch,feat,seq,node
            V_obs_tmp = V_obs.permute(0, 3, 1, 2)

            # Make predictions for this batch
            V_pred, _, simo = model(  # pylint: disable=not-callable
                V_obs_tmp.to(device), A_obs.squeeze().to(device)
            )
            running_vloss += criterion((V_pred, simo), truth_labels.copy(), device)

            # Compute accuracy
            event_acc, node_acc, time_acc = calc_accuracy(simo, truth_labels)
            running_event_acc += event_acc
            running_node_acc += node_acc
            running_time_acc += time_acc

    avg_vloss = running_vloss / len(validation_loader)
    avg_ve_acc = 100 * running_event_acc / len(validation_loader)
    avg_vn_acc = 100 * running_node_acc / len(validation_loader)
    avg_vt_acc = 100 * running_time_acc / len(validation_loader)

    return avg_vloss, avg_ve_acc, avg_vn_acc, avg_vt_acc
