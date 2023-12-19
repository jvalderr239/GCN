from typing import Any, Dict, Tuple

import numpy as np
import torch
from torcheval.metrics import MeanSquaredError, MultilabelAccuracy

multilabel_acc = MultilabelAccuracy()
mse = MeanSquaredError()


def compute_ade(pred: torch.Tensor, truth: torch.Tensor):
    """
    Calculate the average displacement error across sequences and batches.
    Generate metric for each node

    Arguments:
        pred -- _description_
        truth -- _description_
    """
    assert pred.size() == truth.size()
    N, T, *_ = pred.size()
    # Calculate average error for each node
    # (batch, seq, node, feat) -> (node, batch, seq, feat)
    truth = truth.permute(2, 0, 1, 3)
    pred = pred.permute(2, 0, 1, 3)
    pred_np, truth_np = pred.numpy(), truth.numpy()
    x_sq_err = np.square(pred_np[..., 0] - truth_np[..., 0])
    y_sq_err = np.square(pred_np[..., 1] - truth_np[..., 1])

    ade = np.sum(np.sum(x_sq_err + y_sq_err, axis=-1), axis=-1) / (N * T)

    return ade[..., np.newaxis]


def compute_fde(pred: torch.Tensor, truth: torch.Tensor):
    """
    Calculate the final displacement error across batches
    Generate metric for each node

    Arguments:
        pred -- Prediction
        truth -- Truth
    """
    assert pred.size() == truth.size()
    N, *_ = pred.size()
    # Calculate average error for each node
    # (batch, seq, node, feat) -> (node, batch, seq, feat)
    truth = truth.permute(2, 0, 1, 3)
    pred = pred.permute(2, 0, 1, 3)
    pred_np, truth_np = pred.numpy(), truth.numpy()
    x_sq_err = np.square(pred_np[..., -1, 0] - truth_np[..., -1, 0])
    y_sq_err = np.square(pred_np[..., -1, 1] - truth_np[..., -1, 1])
    num = np.sqrt(x_sq_err + y_sq_err)
    fde = np.sum(num, axis=-1) / N
    return fde[..., np.newaxis]


def compute_accuracy(
    outputs: Dict[str, Any], truth: Dict[str, Any], device: Any
) -> Tuple[float, float, float]:
    """
        Compute accuracies for multilabel outputs

    Arguments:
        outputs -- Predictions
        truth -- Truth labels

    Returns:
        Accuracy for each cnn output
    """
    time_acc = (
        mse.update(
            outputs["time_of_event"].squeeze().to(device),
            truth["time_of_event"].squeeze().to(device),
        )
        .compute()
        .float()
        .item()
    )
    event_acc = (
        multilabel_acc.update(
            outputs["event_type"].squeeze().to(device),
            truth["event_type"].squeeze().to(device),
        )
        .compute()
        .float()
        .item()
    )
    node_acc = (
        multilabel_acc.update(
            outputs["node_index"].squeeze().to(device),
            truth["node_index"].squeeze().to(device),
        )
        .compute()
        .float()
        .item()
    )

    return event_acc, node_acc, time_acc


def criterion(
    outputs: Tuple[torch.Tensor, Dict[str, Any]], truth_labels: Dict[str, Any], device
):
    loss_map: Dict[str, Any] = {
        "node_index": torch.nn.BCELoss(),
        "event_type": torch.nn.BCELoss(),
        "time_of_event": torch.nn.MSELoss(),
    }

    losses = 0

    # Trajectory loss
    V_pred, simo = outputs
    V_truth, _ = truth_labels["trajectory_relative"], truth_labels["graph"]
    # Convert output features (batch,seq,node,feat)
    V_truth = V_truth.squeeze().to(device)
    V_pred = V_pred.squeeze().to(device)

    losses += bivariate_graph_loss(V_pred, V_truth)
    # CNN Prediction loss
    for _, (key, pred) in enumerate(simo.items()):  # pylint-ignore: attr-defined
        losses += loss_map[key](pred.to(device), truth_labels[key].to(device))

    return losses


def compute_bivariate_params(V_pred: torch.Tensor):
    """_summary_

    Arguments:
        V_pred -- Predictions
    """
    sx = torch.exp(
        V_pred[..., 2],
    )  # sx
    sy = torch.exp(V_pred[..., 3])  # sy
    corr = torch.tanh(V_pred[..., 4])  # corr

    cov = torch.zeros(V_pred.shape[0], V_pred.shape[1], V_pred.shape[2], 2, 2)
    cov[..., 0, 0] = sx * sx
    cov[..., 0, 1] = corr * sx * sy
    cov[..., 1, 0] = corr * sx * sy
    cov[..., 1, 1] = sy * sy
    mean = V_pred[..., 0:2]

    return sx, sy, corr, cov, mean


def bivariate_graph_loss(V_pred: torch.Tensor, V_trgt: torch.Tensor):
    # mux, muy, sx, sy, corr
    normx = V_trgt[..., 0] - V_pred[..., 0]
    normy = V_trgt[..., 1] - V_pred[..., 1]

    sx, sy, corr, *_ = compute_bivariate_params(V_pred=V_pred)
    sxsy = sx * sy
    z = (normx / sx) ** 2 + (normy / sy) ** 2 - 2 * ((corr * normx * normy) / sxsy)
    negRho = 1 - corr**2

    # Numerator
    result = torch.exp(-z / (2 * negRho))
    # Normalization factor
    denom = 2 * np.pi * (sxsy * torch.sqrt(negRho))

    # Final PDF calculation
    result = result / denom

    # Numerical stability
    epsilon = 1e-20

    result = -torch.log(torch.clamp(result, min=epsilon))
    result = torch.mean(result)

    return result
