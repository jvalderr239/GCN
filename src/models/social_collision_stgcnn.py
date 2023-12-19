import logging
import logging.config
from typing import Optional

import torch
from torch import nn

from src.utils import get_project_root

from .cnn import EVENT_PREDICTOR_CNN, PRETRAINED_EVENT_PREDICTOR_CNN
from .st_gcn import ST_GCN

# setup logger
log_file_path = get_project_root() / "logging.conf"
logging.config.fileConfig(str(log_file_path))
log = logging.getLogger("models")


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
        num_events: int,
        num_nodes: int,
        num_spatial_nodes: int,
        n_stgcnn: int = 1,
        n_txpcnn: int = 1,
        input_feat: int = 2,
        input_cnn_feat: int = 15,
        output_feat: int = 5,
        seq_len: int = 8,
        pred_seq_len: int = 12,
        kernel_size: int = 3,
        *,
        cnn: Optional[str] = None,
        pretrained: bool = True,
        blocks_to_retrain: int = 15,
        cnn_dropout: float = 0.3,
        **kwargs,
    ):
        super().__init__()

        self.num_spatial_nodes = num_spatial_nodes
        self.n_stgcnn = n_stgcnn
        self.n_txpcnn = n_txpcnn

        log.info(f"Constructing {self.n_stgcnn} spatial layers")
        self.st_gcns = nn.ModuleList()
        self.st_gcns.append(ST_GCN(input_feat, output_feat, (kernel_size, seq_len)))
        for _ in range(1, self.n_stgcnn):
            self.st_gcns.append(
                ST_GCN(output_feat, output_feat, (kernel_size, seq_len), **kwargs)
            )
        log.info(f"Constructing {self.n_txpcnn} temporal layers")
        self.tpcnns = nn.ModuleList()
        self.tpcnns.append(nn.Conv2d(seq_len, pred_seq_len, 3, padding=1))
        for _ in range(1, self.n_txpcnn):
            self.tpcnns.append(nn.Conv2d(pred_seq_len, pred_seq_len, 3, padding=1))
        self.tpcnn_ouput = nn.Conv2d(pred_seq_len, pred_seq_len, 3, padding=1)

        self.prelus = nn.ModuleList()
        for _ in range(self.n_txpcnn):
            self.prelus.append(nn.PReLU())

        self.cnn = (
            PRETRAINED_EVENT_PREDICTOR_CNN(
                in_channels=input_cnn_feat,
                name=cnn,
                pretrained=pretrained,
                num_events=num_events,
                num_nodes=num_nodes,
                dropout=cnn_dropout,
                blocks_to_retrain=blocks_to_retrain,
            )
            if cnn is not None
            else EVENT_PREDICTOR_CNN(
                in_channels=input_cnn_feat,
                num_events=num_events,
                num_nodes=num_nodes,
                dropout=cnn_dropout,
            )
        )
        log.info(
            f"Built {self.cnn.name.lower()} model for prediction"  # pylint: disable=union-attr, operator
        )
        self.name = f"SOCIAL_COLLISON_STGCNN_{self.cnn.name}"  # pylint: disable=union-attr, operator

    def forward(self, v, a):
        """
        Feed through model architecture to yield predicted trajectory,
            adjacency matrix and predicted time of attack (TOA)
        """
        v_original = v.clone()
        for k in range(self.n_stgcnn):
            v, a = self.st_gcns[k](v, a)

        # Use feature extractor to predict tackle, tackler and time of attack
        # Use the spatial node output along with categorical features
        cnn_input = torch.concat((v, v_original[:, self.num_spatial_nodes :]), dim=1)
        simo = self.cnn(cnn_input)
        v = v.view(v.shape[0], v.shape[2], v.shape[1], v.shape[3])

        v = self.prelus[0](self.tpcnns[0](v))

        for k in range(1, self.n_txpcnn - 1):
            v = self.prelus[k](self.tpcnns[k](v)) + v

        v = self.tpcnn_ouput(v)
        v = v.view(v.shape[0], v.shape[2], v.shape[1], v.shape[3])

        # Return multi-modal output for trajectory generation and time of attack
        return v, a, simo
