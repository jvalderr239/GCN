from typing import Tuple, Union

from torch import flatten, nn
from torchvision import models


class EVENT_PREDICTOR_CNN(nn.Module):
    def __init__(
        self,
        in_channels: int,
        num_events: int = 3,
        dropout: float = 0.3,
        kernel_size: int = 3,
        hidden_fc_dim: int = 128,
        leaky: bool = True,
    ):
        super(EVENT_PREDICTOR_CNN, self).__init__()

        self.conv_layers = [
            EVENT_PREDICTOR_CNN.build_conv_layer(
                in_c=in_channels, out_c=hidden_fc_dim, kernel_size=kernel_size
            ),
            EVENT_PREDICTOR_CNN.build_conv_layer(
                hidden_fc_dim, hidden_fc_dim * 4, kernel_size=1
            ),
        ]

        self.fc1 = nn.Linear(kernel_size * hidden_fc_dim * 4, hidden_fc_dim // 2)
        self.event_type_fc = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_features=(hidden_fc_dim // 2), out_features=num_events),
            nn.Sigmoid(),
        )
        self.node_index_fc = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_features=(hidden_fc_dim // 2), out_features=1),
        )
        self.time_of_event_fc = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_features=(hidden_fc_dim // 2), out_features=num_events),
            nn.ReLU(),
        )

        self.batch = nn.BatchNorm1d(hidden_fc_dim // 2)
        self.drop = nn.Dropout(p=dropout)
        self.relu = nn.ReLU() if not leaky else nn.LeakyReLU()

    @staticmethod
    def build_conv_layer(
        in_c,
        out_c,
        kernel_size: Union[Tuple[int, int], int] = (3, 3),
        padding: int = 0,
        output_filt_size: Union[int, Tuple[int, ...]] = (2, 2),
    ):
        return nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=kernel_size, padding=padding, stride=2),
            nn.LeakyReLU(),
            nn.MaxPool2d(output_filt_size),
        )

    def forward(self, x):
        out = x
        for m in self.conv_layers:
            out = m(out)
        out = out.view(out.shape[0], out.shape[2] * out.shape[1])
        out = self.fc1(out.squeeze())
        out = self.relu(out)
        out = self.batch(out)
        out = self.drop(out)

        return {
            "event_type": self.event_type_fc(out),
            "node_index": self.node_index_fc(out),
            "time_of_attack": self.time_of_event_fc(out),
        }


class PRETRAINED_EVENT_PREDICTOR_CNN(nn.Module):
    def __init__(
        self,
        in_channels: int,
        name: str,
        pretrained: bool,
        num_events: int = 1,
        dropout: float = 0.3,
    ):
        super(PRETRAINED_EVENT_PREDICTOR_CNN, self).__init__()
        if name.lower() not in dir(models):
            raise ValueError(
                f"Invalid model name: {name}."
                f"Expected one of the following: {dir(models)}"
            )
        selected_model: nn.Module = getattr(models, name.lower())(pretrained=pretrained)
        # self.cnn = nn.Sequential(*(list(selected_model.children())[:-1]))
        first_conv_layer = [
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=3,
                kernel_size=3,
                stride=1,
                padding=0,
                dilation=1,
                groups=1,
                bias=True,
            )
        ]
        first_conv_layer.extend(
            list(selected_model.children())[:-1]
        )  # pylint-ignore: arg-type
        self.cnn = nn.Sequential(*first_conv_layer)
        self.event_type_fc = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_features=512, out_features=num_events),
            nn.Sigmoid(),
        )
        self.node_index_fc = nn.Sequential(
            nn.Dropout(p=dropout), nn.Linear(in_features=512, out_features=1)
        )
        self.time_of_event_fc = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_features=512, out_features=num_events),
            nn.ReLU(),
        )

    def forward(self, x):
        x = self.cnn(x)
        x = flatten(x, 1)

        return {
            "event_type": self.event_type_fc(x),
            "node_index": self.node_index_fc(x),
            "time_of_attack": self.time_of_event_fc(x),
        }
