from typing import Any, Dict, Tuple

import numpy as np
import torch
from torcheval.metrics import MeanSquaredError, MultilabelAccuracy

multilabel_acc = MultilabelAccuracy()
mse = MeanSquaredError()


def calc_accuracy(
    outputs: Dict[str, Any],
    truth: Dict[str, Any],
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
        mse.update(outputs["time_of_event"].squeeze(), truth["time_of_event"].squeeze())
        .compute()
        .float()
        .item()
    )
    event_acc = (
        multilabel_acc.update(
            outputs["event_type"].squeeze(), truth["event_type"].squeeze()
        )
        .compute()
        .float()
        .item()
    )
    node_acc = (
        multilabel_acc.update(
            outputs["node_index"].squeeze(), truth["node_index"].squeeze()
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
    V_truth, _ = truth_labels["trajectory"], truth_labels["graph"]
    # Convert output features (batch,seq,node,feat)
    V_truth = V_truth.squeeze().to(device)
    V_pred = V_pred.squeeze().permute(0, 2, 3, 1).to(device)

    losses += bivariate_graph_loss(V_pred, V_truth)
    # CNN Prediction loss
    for _, (key, pred) in enumerate(simo.items()):  # pylint-ignore: attr-defined
        losses += loss_map[key](pred.to(device), truth_labels[key].to(device))

    return losses


def bivariate_graph_loss(V_pred: torch.Tensor, V_trgt: torch.Tensor):
    # mux, muy, sx, sy, corr
    normx = V_trgt[..., 0] - V_pred[..., 0]
    normy = V_trgt[..., 1] - V_pred[..., 1]

    sx = torch.exp(V_pred[..., 2])  # sx
    sy = torch.exp(V_pred[..., 3])  # sy
    corr = torch.tanh(V_pred[..., 4])  # corr
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
