import numpy as np
import torch


def checkpoint(model, filename):
    torch.save(model.state_dict(), filename)


def resume(model, filename):
    model.load_state_dict(torch.load(filename))


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
