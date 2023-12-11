from torch import einsum, nn


class GCN(nn.Module):
    # Source : https://github.com/yysijie/st-gcn/blob/master/net/st_gcn.py
    """
    Graph convolutional network describing graph network format

    Args:
        in_channels (int): Number of channels in the input sequence data
        out_channels (int): Number of channels produced by the convolution
        kernel_size (int): Size of the graph convolving kernel
        t_kernel_size (int): Size of the temporal convolving kernel
        t_stride (int, optional): Stride of the temporal convolution. Default: 1
        t_padding (int, optional): Temporal zero-padding added to both sides of
            the input. Default: 0
        t_dilation (int, optional): Spacing between temporal kernel elements.
            Default: 1
        bias (bool, optional): If ``True``, adds a learnable bias to the output.
            Default: ``True``
    Shape:
        - Input[0]:
            Input graph sequence format: (batch_size, in_channels, len_input_seq, num_nodes)

        - Input[1]:
            Input graph adjacency matrix format:
                (kernel_size[1], num_nodes, num_nodes)

        - Output[0]:
            Output graph sequence format:
                (batch_size, out_channels, len_output_seq, num_nodes, num_nodes)

        - Output[1]:
            Graph adjacency matrix for output data format: (kernel_size[1], num_nodes, num_nodes)
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        t_kernel_size=1,
        t_stride=1,
        t_padding=0,
        t_dilation=1,
        bias=True,
    ):
        super().__init__()

        # Convolutional kernel size
        self.kernel_size = kernel_size

        # Convolutional layer dimensions
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=(t_kernel_size, 1),
            padding=(t_padding, 0),
            stride=(t_stride, 1),
            dilation=(t_dilation, 1),
            bias=bias,
        )

    def forward(self, x, A):
        print(A.size())
        print(self.kernel_size)
        assert A.size(0) == self.kernel_size
        x = self.conv(x)
        x = einsum("nctv,tvw->nctw", (x, A))
        return x.contiguous(), A
