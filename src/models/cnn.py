from typing import Iterable, Union

from torch import nn


class EVENT_PREDICTOR_CNN(nn.Module):
    def __init__(
        self,
        in_channels: int,
        num_classes: int = 22,
        num_events: int = 1,
        dropout: float = 0.3,
        hidden_fc_dim: int = 128,
        leaky: bool = True,
    ):
        super(EVENT_PREDICTOR_CNN, self).__init__()

        self.conv_layers: nn.ModuleList = [
            EVENT_PREDICTOR_CNN.build_conv_layer(in_c=in_channels, out_c=32),
            EVENT_PREDICTOR_CNN.build_conv_layer(32, 64),
        ]

        self.fc1 = nn.Linear(2**in_channels * 64, hidden_fc_dim)
        self.time_of_event_fc2 = nn.Linear(hidden_fc_dim, num_events)
        self.involved_classes_fc2 = nn.Linear(hidden_fc_dim, num_classes)
        self.event_fc2 = nn.Linear(hidden_fc_dim, num_events)

        self.batch = nn.BatchNorm1d(hidden_fc_dim)
        self.drop = nn.Dropout(p=dropout)
        self.relu = nn.LeakyReLU()
        if not leaky:
            self.relu = nn.ReLU()

    @staticmethod
    def build_conv_layer(
        in_c,
        out_c,
        kernel_size: Union[Iterable[int], int] = (3, 3, 3),
        padding: int = 0,
        output_filt_size: Iterable[int] = (2, 2, 2),
    ):
        return nn.Sequential(
            nn.Conv3d(in_c, out_c, kernel_size=kernel_size, padding=padding),
            nn.LeakyReLU(),
            nn.MaxPool3d(output_filt_size),
        )

    def forward(self, x):
        out = x
        for m in self.conv_layers:
            out = m(out)
        out = out.view(out.size(0), -1)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.batch(out)
        out = self.drop(out)

        return {
            "event_type": nn.Sigmoid()(self.event_fc2(out)),
            "node_index": nn.Sigmoid()(self.involved_classes_fc2(out)),
            "time_of_attack": nn.ReLU()(self.time_of_event_fc2(out)),
        }
