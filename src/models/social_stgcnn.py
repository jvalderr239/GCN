from torch import nn

from .st_gcn import ST_GCN


class SOCIAL_STGCNN(nn.Module):
    def __init__(
        self,
        n_stgcnn=1,
        n_txpcnn=1,
        input_feat=2,
        output_feat=5,
        seq_len=8,
        pred_seq_len=12,
        kernel_size=3,
    ):
        super().__init__()
        self.n_stgcnn = n_stgcnn
        self.n_txpcnn = n_txpcnn

        self.st_gcns = nn.ModuleList()
        self.st_gcns.append(ST_GCN(input_feat, output_feat, (kernel_size, seq_len)))
        for _ in range(1, self.n_stgcnn):
            self.st_gcns.append(
                ST_GCN(output_feat, output_feat, (kernel_size, seq_len))
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
        for k in range(self.n_stgcnn):
            v, a = self.st_gcns[k](v, a)

        v = v.view(v.shape[0], v.shape[2], v.shape[1], v.shape[3])

        v = self.prelus[0](self.tpcnns[0](v))

        for k in range(1, self.n_txpcnn - 1):
            v = self.prelus[k](self.tpcnns[k](v)) + v

        v = self.tpcnn_ouput(v)
        v = v.view(v.shape[0], v.shape[2], v.shape[1], v.shape[3])

        return v, a
