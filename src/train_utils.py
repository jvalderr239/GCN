import dataclasses
import logging
import logging.config
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
from torch import nn
from torch.optim import Optimizer, lr_scheduler
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter

from .datasets import TrajectoryDataset
from .metrics import (compute_accuracy, compute_ade, compute_bivariate_params,
                      compute_fde, criterion)
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
        num_workers=0 if datatype == "train" else num_workers,
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


def infer(
    model: nn.Module, V_obs: torch.Tensor, A_obs: torch.Tensor, device: Any
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    # V_obs = batch,seq,node,feat
    # V_obs_tmp = batch,feat,seq,node
    V_obs_tmp = V_obs.permute(0, 3, 1, 2)

    # Make predictions for this batch
    V_pred, _, simo = model(V_obs_tmp.to(device), A_obs.squeeze().to(device))

    # Convert back to (batch,seq,node,feat)
    V_pred = V_pred.permute(0, 2, 3, 1)

    return V_pred, simo


def train_one_epoch(
    epoch_index: int,
    model: nn.Module,
    training_loader: DataLoader,
    optimizer: Optimizer,
    tb_writer: SummaryWriter,
    clip: Optional[float],
    device: Any,
) -> Tuple[float, float, float, float]:
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
        V_obs, A_obs, *_ = batch_data["data"]
        truth_labels = batch_data["labels"]
        # Zero your gradients for every batch!
        optimizer.zero_grad()

        V_pred, simo = infer(model, V_obs=V_obs, A_obs=A_obs, device=device)
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
        event_acc, node_acc, time_acc = compute_accuracy(
            simo, truth_labels.copy(), device
        )
        running_event_acc += event_acc
        running_node_acc += node_acc
        running_time_acc += time_acc
        if (i + 1) % 10 == 0:
            last_loss = running_loss / 10  # loss per 10 batches
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
) -> Tuple[float, ...]:
    running_vloss = 0.0
    running_time_acc = 0.0
    running_node_acc = 0.0
    running_event_acc = 0.0
    avg_vloss = 0.0
    # Set the model to evaluation mode, disabling dropout and using population
    # statistics for batch normalization.
    model.eval()

    # Disable gradient computation and reduce memory consumption.
    with torch.no_grad():
        for iv, vbatchdata in enumerate(validation_loader):
            V_obs, A_obs, *_ = vbatchdata["data"]
            truth_labels = vbatchdata["labels"]

            V_pred, simo = infer(model, V_obs=V_obs, A_obs=A_obs, device=device)

            running_vloss += criterion(
                (V_pred, simo), truth_labels.copy(), device
            ).item()
            if (iv + 1) % 10 == 0:
                avg_vloss += running_vloss / 10  # per 10 batches
                running_vloss = 0
            # Compute accuracy
            event_acc, node_acc, time_acc = compute_accuracy(
                simo, truth_labels.copy(), device
            )
            running_event_acc += event_acc
            running_node_acc += node_acc
            running_time_acc += time_acc

    avg_vloss = running_vloss / len(validation_loader)
    avg_ve_acc = 100 * running_event_acc / len(validation_loader)
    avg_vn_acc = 100 * running_node_acc / len(validation_loader)
    avg_vt_acc = 100 * running_time_acc / len(validation_loader)

    return avg_vloss, avg_ve_acc, avg_vn_acc, avg_vt_acc


def test(
    model: nn.Module, test_loader: DataLoader, device: Any, sample_steps: int = 20
) -> Tuple[float, ...]:
    # Set the model to evaluation mode, disabling dropout and using population
    # statistics for batch normalization.
    model.eval()
    running_tloss = 0.0
    running_time_acc = 0.0
    running_node_acc = 0.0
    running_event_acc = 0.0
    avg_tloss = 0.0

    ade, fde = [], []
    # Disable gradient computation and reduce memory consumption.
    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            V_obs, A_obs, obs_traj = batch["data"]
            truth_labels = batch["labels"]

            # Size: Batch, seq, node, feat
            V_truth, V_truth_rel = (
                truth_labels["trajectory"],
                truth_labels["trajectory_relative"],
            )
            num_obs_sequences, num_pred_sequences, num_nodes = (
                V_obs.size(1),
                V_truth.size(1),
                V_obs.size(2),
            )
            # Get the absolute path for each observation and truth
            V_obs_x_abs = torch.cumsum(V_obs[..., 0], dim=1) + obs_traj[
                :, 0, :, 0
            ].unsqueeze(1).repeat(1, num_obs_sequences, 1)
            V_obs_y_abs = torch.cumsum(V_obs[..., 1], dim=1) + obs_traj[
                :, 0, :, 1
            ].unsqueeze(1).repeat(1, num_obs_sequences, 1)
            V_obs_abs = torch.concat(
                (V_obs_x_abs.unsqueeze(-1), V_obs_y_abs.unsqueeze(-1)), dim=-1
            )

            V_tr_x_abs = torch.cumsum(V_truth_rel[..., 0], dim=1) + V_truth[
                :, 0, :, 0
            ].unsqueeze(1).repeat(1, num_pred_sequences, 1)
            V_tr_y_abs = torch.cumsum(V_truth_rel[..., 1], dim=1) + V_truth[
                :, 0, :, 1
            ].unsqueeze(1).repeat(1, num_pred_sequences, 1)
            V_tr_abs = torch.concat(
                (V_tr_x_abs.unsqueeze(-1), V_tr_y_abs.unsqueeze(-1)), dim=-1
            )

            # Generate deviation data from predictions
            # V_pred includes x, y, sx, sy, corr deviations
            V_pred, simo = infer(model, V_obs=V_obs, A_obs=A_obs, device=device)

            running_tloss += criterion(
                (V_pred, simo), truth_labels.copy(), device
            ).item()
            if (i + 1) % 10 == 0:
                avg_tloss += running_tloss / 10  # per 10 batches
                running_tloss = 0
            # Compute accuracy
            event_acc, node_acc, time_acc = compute_accuracy(
                simo, truth_labels.copy(), device
            )
            running_event_acc += event_acc
            running_node_acc += node_acc
            running_time_acc += time_acc

            # Get multivariate variables from generated deviation data
            *_, cov, mean = compute_bivariate_params(V_pred=V_pred)
            mvnormal = torch.distributions.multivariate_normal.MultivariateNormal(
                mean, cov
            )

            sample_ade, sample_fde = np.empty((num_nodes, 1)), np.empty((num_nodes, 1))
            for _ in range(sample_steps):
                V_sample = mvnormal.sample()

                V_sample_x_abs = torch.cumsum(V_sample[..., 0], dim=1) + V_truth[
                    :, 0, :, 0
                ].unsqueeze(1).repeat(1, num_pred_sequences, 1)
                V_sample_y_abs = torch.cumsum(V_sample[..., 1], dim=1) + V_truth[
                    :, 0, :, 1
                ].unsqueeze(1).repeat(1, num_pred_sequences, 1)
                V_sample_abs = torch.concat(
                    (V_sample_x_abs.unsqueeze(-1), V_sample_y_abs.unsqueeze(-1)), dim=-1
                )
                ade_per_node = compute_ade(
                    V_sample_abs.contiguous(), V_tr_abs.contiguous()
                )
                fde_per_node = compute_fde(
                    V_sample_abs.contiguous(), V_tr_abs.contiguous()
                )

                sample_ade = (
                    np.minimum(ade_per_node, sample_ade)
                    if sample_ade.size
                    else ade_per_node
                )
                sample_fde = (
                    np.minimum(fde_per_node, sample_fde)
                    if sample_fde.size
                    else fde_per_node
                )
            ade.append(sample_ade)
            fde.append(sample_fde)

    avg_tloss = running_tloss / len(test_loader)
    avg_te_acc = 100 * running_event_acc / len(test_loader)
    avg_tn_acc = 100 * running_node_acc / len(test_loader)
    avg_tt_acc = 100 * running_time_acc / len(test_loader)
    res_ade = float(np.sum(np.array(ade)) / len(ade))
    res_fde = float(np.sum(np.array(fde)) / len(fde))

    return res_ade, res_fde, avg_tloss, avg_te_acc, avg_tn_acc, avg_tt_acc
