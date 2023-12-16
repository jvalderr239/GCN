import logging
import logging.config
import os
from datetime import datetime

import torch
from fire import Fire
from torch.utils.tensorboard import SummaryWriter

from src import train_utils
from src.models.social_collision_stgcnn import \
    SOCIAL_COLLISION_STGCNN as social_c_stgcnn
from src.utils import get_project_root

# setup logger
log_file_path = get_project_root() / "logging.conf"
logging.config.fileConfig(str(log_file_path))
log = logging.getLogger(__name__)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def train(epochs: int, **kwargs):
    # Defining the model
    trainer = train_utils.Trainer(**kwargs)
    model = social_c_stgcnn(
        num_events=trainer.num_events,
        n_stgcnn=trainer.num_spatial,
        n_txpcnn=trainer.num_temporal,
        input_feat=trainer.num_features,
        output_feat=trainer.output_feat,
        seq_len=trainer.seq_len,
        pred_seq_len=trainer.pred_seq_len,
        kernel_size=trainer.kernel_size,
        num_nodes=trainer.num_nodes,
        cnn=trainer.cnn_name,
        pretrained=trainer.pretrained,
        cnn_dropout=trainer.cnn_dropout,
    ).cuda()

    # Training settings

    optimizer = torch.optim.SGD(model.parameters(), lr=trainer.lr)
    scheduler = trainer.get_scheduler(optimizer=optimizer)

    loader_train = train_utils.generate_dataloader(
        "train", batch_size=8, shuffle=True, num_workers=0, root_dir=trainer.root_dir
    )
    loader_val = train_utils.generate_dataloader(
        "validation",
        batch_size=8,
        shuffle=False,
        num_workers=1,
        root_dir=trainer.root_dir,
    )

    if not os.path.exists(trainer.checkpoint_dir):
        os.makedirs(trainer.checkpoint_dir)

    checkpoint = torch.load(trainer.checkpoint_dir)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    # Initializing in a separate cell so we can easily add more epochs to the same run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    writer = SummaryWriter(
        f"{str(get_project_root())}/train_results/runs/stgcnn_{timestamp}"
    )
    epoch_number = 0

    best_vloss = 1_000_000.0

    for epoch_number in range(epochs):
        log.info(f"EPOCH {epoch_number + 1}:")

        # Make sure gradient tracking is on, and do a pass over the data
        model.train(True)
        avg_loss = train_utils.train_one_epoch(
            epoch_index=epoch_number,
            model=model,
            training_loader=loader_train,
            optimizer=optimizer,
            tb_writer=writer,
        )

        running_vloss = 0.0
        # Set the model to evaluation mode, disabling dropout and using population
        # statistics for batch normalization.
        model.eval()

        # Disable gradient computation and reduce memory consumption.
        with torch.no_grad():
            for iv, vbatchdata in enumerate(loader_val):
                V_obs, A_obs = vbatchdata["data"]
                truth_labels = vbatchdata["labels"]

                # V_obs = batch,seq,node,feat
                # V_obs_tmp = batch,feat,seq,node
                V_obs_tmp = V_obs.permute(0, 3, 1, 2)

                # Make predictions for this batch
                V_pred, _, simo = model(V_obs_tmp, A_obs.squeeze())

                V_pred = V_pred.permute(0, 2, 3, 1)
                vloss = train_utils.criterion((V_pred, simo), truth_labels.copy())
                running_vloss += vloss

        avg_vloss = running_vloss / (iv + 1)
        log.info(f"LOSS train {avg_loss} valid {avg_vloss}".format(avg_loss, avg_vloss))

        # Log the running loss averaged per batch
        # for both training and validation
        writer.add_scalars(
            "Training vs. Validation Loss",
            {"Training": avg_loss, "Validation": avg_vloss},
            epoch_number + 1,
        )
        writer.flush()

        # Track best performance, and save the model's state
        if avg_vloss < best_vloss:
            best_vloss = avg_vloss
            model_path = f"{str(get_project_root())}/train_results/model_{timestamp}_{epoch_number}"
            torch.save(model.state_dict(), model_path)

        epoch_number += 1


if __name__ == "main":
    Fire(train)
