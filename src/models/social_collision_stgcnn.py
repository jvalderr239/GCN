from typing import Optional

from torch import nn
from torchvision import models

from .cnn import EVENT_PREDICTOR_CNN
from .st_gcn import ST_GCN


class SOCIAL_COLLISION_STGCNN(nn.Module):
    """
    The social collision spatio-temporal graphical neural network is composed of:
        Spatio-Temporal GCN that operates on graphical representation of athlete
            trajectories on a field for feature extraction
        Time-Extrapolator CNN takes the trajectory history from the STGCNN as inputs
            to predict the future trajectories of all players on the field through convolution
        Transfer Learning Based Collision Predictor that takes as inputs the trajectory history
            from STGCNN to predict the time of attack (TOA) of a player
            (e.g. tackle, fumble, first_contact)

    Sources:
        https://openaccess.thecvf.com/content_CVPR_2020/papers/Mohamed_Social-STGCNN_A_Social_Spatio-Temporal_Graph_Convolutional_Neural_Network_for_Human_CVPR_2020_paper.pdf
        https://www.hindawi.com/journals/wcmc/2022/2545958/
    """

    def __init__(
        self,
        num_classes: int,
        num_events: int,
        n_stgcnn: int = 1,
        n_txpcnn: int = 1,
        input_feat: int = 2,
        output_feat: int = 5,
        seq_len: int = 8,
        pred_seq_len: int = 12,
        kernel_size: int = 3,
        *,
        cnn: Optional[str] = None,
        pretrained: bool = True,
        cnn_dropout: float = 0.3,
        **kwargs,
    ):
        super().__init__()

        self.n_stgcnn = n_stgcnn
        self.n_txpcnn = n_txpcnn

        self.st_gcns = nn.ModuleList()
        self.st_gcns.append(ST_GCN(input_feat, output_feat, (kernel_size, seq_len)))
        for _ in range(1, self.n_stgcnn):
            self.st_gcns.append(
                ST_GCN(output_feat, output_feat, (kernel_size, seq_len), **kwargs)
            )

        self.tpcnns = nn.ModuleList()
        self.tpcnns.append(nn.Conv2d(seq_len, pred_seq_len, 3, padding=1))
        for _ in range(1, self.n_txpcnn):
            self.tpcnns.append(nn.Conv2d(pred_seq_len, pred_seq_len, 3, padding=1))
        self.tpcnn_ouput = nn.Conv2d(pred_seq_len, pred_seq_len, 3, padding=1)

        self.prelus = nn.ModuleList()
        for _ in range(self.n_txpcnn):
            self.prelus.append(nn.PReLU())

        if cnn is not None:
            if cnn.lower() not in dir(models):
                raise ValueError(
                    f"Invalid model name: {cnn}."
                    f"Expected one of the following: {dir(models)}"
                )
            selected_model: Optional[nn.Module] = getattr(models, cnn.lower())
            self.cnn = selected_model(pretrained=pretrained)
            self.cnn.fc = nn.Sequential(
                nn.Linear(self.cnn.in_features, 64),
                nn.ReLU(),
                nn.Linear(64, num_classes),
                nn.Sigmoid(),
            )
        else:
            self.cnn = EVENT_PREDICTOR_CNN(
                in_channels=output_feat,
                num_classes=num_classes,
                num_events=num_events,
                dropout=cnn_dropout,
            )

    def forward(self, v, a):
        """
        Feed through model architecture to yield predicted trajectory,
            adjacency matrix and predicted time of attack (TOA)
        """
        for k in range(self.n_stgcnn):
            v, a = self.st_gcns[k](v, a)

        v = v.view(v.shape[0], v.shape[2], v.shape[1], v.shape[3])

        # Use feature extractor to predict tackler and time of attack
        event, player, toa = self.cnn(v)

        v = self.prelus[0](self.tpcnns[0](v))

        for k in range(1, self.n_txpcnn - 1):
            v = self.prelus[k](self.tpcnns[k](v)) + v

        v = self.tpcnn_ouput(v)
        v = v.view(v.shape[0], v.shape[2], v.shape[1], v.shape[3])

        # Return multi-modal output for trajectory generation and time of attack
        return v, a, (event, player, toa)
