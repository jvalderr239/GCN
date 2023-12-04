from torch import nn

from .gcn import GCN


class ST_GCN(nn.Module):
    r"""
    Applies a spatial temporal graph convolution over an input graph sequence.

    Args:
        in_channels (int): Number of features in the input sequence data
        out_channels (int): Number of features produced by the convolution
        kernel_size (tuple): Size of the temporal convolving kernel and graph convolving kernel
        stride (int, optional): Stride of the temporal convolution. Default: 1
        dropout (int, optional): Dropout rate of the final output. Default: 0
        residual (bool, optional): If ``True``, applies a residual mechanism. Default: ``True``
    Shape:
        - Input[0]:
            Input graph sequence format: (batch_size, in_channels, len_input_seq, num_nodes)

        - Input[1]:
            Input graph adjacency matrix format: (kernel_size[1], num_nodes, num_nodes)

        - Output[0]:
            Output graph sequence format: (batch_size, out_channels, len_output_seq, num_nodes, num_nodes)

        - Output[1]:
            Graph adjacency matrix for output data format: (kernel_size[1], num_nodes, num_nodes)
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        use_mdn=False,
        stride=1,
        dropout=0,
        residual=True,
    ):
        super().__init__()

        assert len(kernel_size) == 2
        assert kernel_size[0] % 2 == 1

        padding = ((kernel_size[0] - 1) // 2, 0)
        self.use_mdn = use_mdn

        self.gcn = GCN(in_channels, out_channels, kernel_size[1])

        self.tcn = nn.Sequential(
            nn.BatchNorm2d(out_channels),
            nn.PReLU(),
            nn.Conv2d(
                out_channels,
                out_channels,
                (kernel_size[0], 1),
                (stride, 1),
                padding,
            ),
            nn.BatchNorm2d(out_channels),
            nn.Dropout(dropout, inplace=True),
        )

        if not residual:
            self.residual = lambda x: 0

        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x

        else:
            self.residual = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=(stride, 1)),
                nn.BatchNorm2d(out_channels),
            )

        self.prelu = nn.PReLU()

    def forward(self, x, A):
        res = self.residual(x)
        x, A = self.gcn(x, A)

        x = self.tcn(x) + res

        if not self.use_mdn:
            x = self.prelu(x)

        return x, A
