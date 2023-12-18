from typing import Tuple, Union

from torch import flatten, nn, randn
from torchvision import models


class EVENT_PREDICTOR_CNN(nn.Module):
    def __init__(
        self,
        in_channels: int,
        num_events: int = 3,
        num_nodes: int = 22,
        dropout: float = 0.3,
        kernel_size: int = 3,
        hidden_fc_dim: int = 128,
        leaky: bool = True,
    ):
        super().__init__()
        self.name = self.__class__

        self.conv_layers = [
            self.build_conv_layer(
                in_c=in_channels, out_c=hidden_fc_dim, kernel_size=kernel_size
            ),
            self.build_conv_layer(hidden_fc_dim, hidden_fc_dim * 4, kernel_size=1),
        ]

        self.fc1 = nn.Linear(hidden_fc_dim * 4, hidden_fc_dim // 2)
        self.event_type_fc = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_features=(hidden_fc_dim // 2), out_features=num_events),
            nn.Sigmoid(),
        )
        self.node_index_fc = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_features=(hidden_fc_dim // 2), out_features=num_nodes),
            nn.Sigmoid(),
        )
        self.time_of_event_fc = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_features=(hidden_fc_dim // 2), out_features=num_events),
            nn.ReLU(),
        )

        self.batch = nn.BatchNorm1d(hidden_fc_dim // 2)
        self.drop = nn.Dropout(p=dropout)
        self.relu = nn.ReLU() if not leaky else nn.LeakyReLU()

    def build_conv_layer(
        self,
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
        out = out.unsqueeze(1)
        return {
            "event_type": self.event_type_fc(out),
            "node_index": self.node_index_fc(out),
            "time_of_event": self.time_of_event_fc(out),
        }


class PRETRAINED_EVENT_PREDICTOR_CNN(nn.Module):
    def __init__(
        self,
        in_channels: int,
        name: str,
        pretrained: bool = True,
        num_events: int = 1,
        num_nodes: int = 22,
        kernel_size: int = 3,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.name = self.__class__
        self.cnn, cnn_output_dim = self._get_base_model(
            name=name.lower(),
            pretrained=pretrained,
            in_channels=in_channels,
            kernel_size=kernel_size,
        )
        self.event_type_fc = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_features=cnn_output_dim, out_features=num_events),
            nn.Sigmoid(),
        )
        self.node_index_fc = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_features=cnn_output_dim, out_features=num_nodes),
            nn.Sigmoid(),
        )
        self.time_of_event_fc = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_features=cnn_output_dim, out_features=num_events),
            nn.ReLU(),
        )

    def _get_base_model(
        self,
        name: str,
        pretrained: bool,
        in_channels: int,
        kernel_size: int,
    ):
        """
        Generate base model and reformat with convolutional layer to reshape as
        expected input size

        Raises:
            ValueError: If model name is invalid and cannot be pulled from torch

        Returns:
            Tuple containing model and output size
        """
        if name.lower() not in dir(models):
            raise ValueError(
                f"Invalid model name: {name}."
                f"Expected one of the following: {dir(models)}"
            )
        if "resnet" not in name.lower():
            raise ValueError("Currently, there is only support for ResNet backbones...")

        selected_model: nn.Module = getattr(models, name.lower())(pretrained=pretrained)

        # Fine-tune pretrained model
        if pretrained:
            for param in selected_model.parameters():
                param.requires_grad = False
            for i in range(-1, -5, -1):
                for param in selected_model.features[i].parameters():
                    param.requires_grad = True

        first_conv_layer = [
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=3,
                kernel_size=kernel_size,
                stride=1,
                padding=0,
                dilation=1,
                groups=1,
                bias=True,
            ),
            nn.MaxPool2d(kernel_size=2, stride=2),
        ]
        first_conv_layer.extend(
            list(selected_model.children())[:-1]
        )  # pylint-ignore: arg-type
        base_model = nn.Sequential(*first_conv_layer)

        # get output shape
        x = randn(5, in_channels, 56, 22)
        output_dim = base_model(x).size(1)
        return base_model, output_dim

    def forward(self, x):
        x = self.cnn(x)
        x = flatten(x, 1)
        x = x.unsqueeze(1)
        return {
            "event_type": self.event_type_fc(x),
            "node_index": self.node_index_fc(x),
            "time_of_event": self.time_of_event_fc(x),
        }
