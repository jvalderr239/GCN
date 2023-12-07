from torch import nn
from torchvision import models

from .st_gcn import ST_GCN

MODEL_MAP = {
    "resnet": models.resnet18,
    "inception": models.inception_v3,
    "densenet": models.densenet161,
    "wide_resnet": models.wide_resnet50_2,
    "mobilenet_small": models.mobilenet_v3_small,
    "mobilenet_large": models.mobilenet_v3_large,
}


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
        n_stgcnn: int = 1,
        n_txpcnn: int = 1,
        input_feat: int = 2,
        output_feat: int = 5,
        seq_len: int = 8,
        pred_seq_len: int = 12,
        kernel_size: int = 3,
        *,
        cnn: str = "mobilenet_small",
        pretrained: bool = True,
        **kwargs,
    ):
        super().__init__()

        assert cnn.lower() in MODEL_MAP.keys()

        self.cnn = MODEL_MAP[cnn.lower()](pretrained=pretrained)
        self.cnn.fc = nn.Sequential(
            nn.Linear(self.cnn.in_features, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

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

    def forward(self, v, a):
        """
        Feed through model architecture to yield predicted trajectory,
            adjacency matrix and predicted time of attack (TOA)
        """
        for k in range(self.n_stgcnn):
            v, a = self.st_gcns[k](v, a)

        v = v.view(v.shape[0], v.shape[2], v.shape[1], v.shape[3])

        conv_output = self.cnn(v)

        v = self.prelus[0](self.tpcnns[0](v))

        for k in range(1, self.n_txpcnn - 1):
            v = self.prelus[k](self.tpcnns[k](v)) + v

        v = self.tpcnn_ouput(v)
        v = v.view(v.shape[0], v.shape[2], v.shape[1], v.shape[3])

        # Return multi-modal output for trajectory generation and
        return v, a, conv_output
