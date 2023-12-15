import dataclasses
from datetime import datetime
from typing import Optional

import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data import DataLoader

from .datasets import TrajectoryDataset


@dataclasses.dataclass
class Trainer:
    # Model parameters
    num_events: int = 3
    num_spatial: int = 1
    num_temporal: int = 1
    num_features: int = 2
    output_feat: int = 5
    seq_len: int = 8
    pred_seq_len: int = 12
    kernel_size: int = 3
    cnn_name: Optional[str] = None
    pretrained: bool = True
    cnn_dropout: float = 0.3

    # Train parameters
    lr: float = 0.001
    checkpoint_dir = f"../checkpoint/{datetime.now()}/"
    root_dir: str = "../resources/"
    batch_size: int = 1

    def update(self, arg, value):
        setattr(self, arg, value)

    def get_scheduler(self, optimizer):
        return lr_scheduler.ExponentialLR(optimizer, gamma=0.2)

    def get_dataloaders(self):
        train_dataset = self.get_trajectory_dataset("train")

        validation_dataset = self.get_trajectory_dataset("validation")

        loader_train = DataLoader(
            train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=0
        )

        loader_val = DataLoader(
            validation_dataset, batch_size=self.batch_size, shuffle=False, num_workers=1
        )

        return loader_train, loader_val

    def get_trajectory_dataset(self, datatype: str):
        return TrajectoryDataset(datatype=datatype, root_dir=self.root_dir)
