from typing import Any, Optional

import numpy as np


class GCNLayer:
    """
    Graphical Neural Network layer
    """

    def __init__(
        self,
        n_inputs: int,
        n_outputs: int,
        activation: Optional[Any] = None,
        name: str = "",
    ) -> None:
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        self.activation = activation
        self.name = name

        # LeCun initialization
        limit = np.sqrt(3 / float(self.n_inputs))
        self.w = np.random.uniform(
            low=-limit, high=limit, size=(self.n_inputs, self.n_outputs)
        )
        self.h: np.ndarray
        self.w: np.ndarray
        self.x: np.ndarray

    def __repr__(self) -> str:
        return f"GCN: W{'_' + self.name if self.name else ''} ({self.n_inputs}, {self.n_outputs})"

    def forward(self, *, A: np.ndarray, X: np.ndarray, W: Optional[np.ndarray] = None):
        """_summary_

        Arguments:
            A -- _description_
            X -- _description_

        Keyword Arguments:
            W -- _description_ (default: {None})

        Returns:
            _description_
        """
        if W is None:
            W = self.w

        self.x = (A @ X).T

        H = W @ self.x
        if self.activation is not None:
            H = self.activation(H)

        self.h = H
        return self.h.T
