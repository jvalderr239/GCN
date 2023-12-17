import logging
import logging.config
from datetime import datetime
from typing import Optional

import torch
from fire import Fire
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from src import train_utils
from src.models.social_collision_stgcnn import SOCIAL_COLLISION_STGCNN as mimo
from src.utils import get_project_root

# setup logger
log_file_path = get_project_root() / "logging.conf"
logging.config.fileConfig(str(log_file_path))
log = logging.getLogger(__name__)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
log.info(f"Working with {device} for training")


def train(
    epochs: int,
    dest_dir: str,
    root_dir: Optional[str] = None,
    batch_size: int = 1,
    **kwargs,
):
    # Defining the model
    if root_dir is None:
        root_dir = str(get_project_root() / "resources") + "/"
    trainer = train_utils.Trainer(**kwargs)
    log.info("Loading training set")
    loader_train = train_utils.generate_dataloader(
        "train",
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        root_dir=root_dir,
        obs_len=trainer.seq_len,
        pred_len=trainer.pred_seq_len,
    )
    log.info("Loading validation set")
    loader_val = train_utils.generate_dataloader(
        "validation",
        batch_size=batch_size,
        shuffle=False,
        num_workers=1,
        root_dir=root_dir,
        obs_len=trainer.seq_len,
        pred_len=trainer.pred_seq_len,
    )
    log.info("Loading test set")
    loader_val = train_utils.generate_dataloader(
        "test",
        batch_size=batch_size,
        shuffle=False,
        num_workers=1,
        root_dir=root_dir,
        obs_len=trainer.seq_len,
        pred_len=trainer.pred_seq_len,
    )

    stgcnn_model = mimo(
        num_events=trainer.num_events,
        num_spatial_nodes=trainer.num_spatial_features,
        n_stgcnn=trainer.num_spatial,
        n_txpcnn=trainer.num_temporal,
        input_feat=trainer.num_features,
        input_cnn_feat=trainer.num_features
        - trainer.num_spatial_features
        + trainer.output_feat,
        output_feat=trainer.output_feat,
        seq_len=trainer.seq_len,
        pred_seq_len=trainer.pred_seq_len,
        kernel_size=trainer.kernel_size,
        num_nodes=trainer.num_nodes,
        cnn=trainer.cnn_name,
        pretrained=trainer.pretrained,
        cnn_dropout=trainer.cnn_dropout,
    ).to(device=device)

    # Training settings
    optimizer = torch.optim.SGD(stgcnn_model.parameters(), lr=trainer.lr)
    scheduler = trainer.get_scheduler(optimizer=optimizer)

    # Initializing in a separate cell so we can easily add more epochs to the same run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    writer = SummaryWriter(f"{dest_dir}/runs/social_collision_stgcnn_{timestamp}")

    best_vloss = 1_000_000.0
    early_stop_thresh = 10
    log.info(f"Training for {epochs} epochs")
    for epoch_number in (pbar := tqdm(range(epochs))):
        pbar.set_description(f"EPOCH {epoch_number + 1}:")

        # Make sure gradient tracking is on, and do a pass over the data
        stgcnn_model.train(True)
        avg_loss = train_utils.train_one_epoch(
            epoch_index=epoch_number,
            model=stgcnn_model,
            training_loader=loader_train,
            optimizer=optimizer,
            tb_writer=writer,
            clip=trainer.clip,
            device=device,
        )

        running_vloss = 0.0
        # Set the model to evaluation mode, disabling dropout and using population
        # statistics for batch normalization.
        stgcnn_model.eval()

        # Disable gradient computation and reduce memory consumption.
        with torch.no_grad():
            for iv, vbatchdata in enumerate(loader_val):
                V_obs, A_obs = vbatchdata["data"]
                truth_labels = vbatchdata["labels"]

                # V_obs = batch,seq,node,feat
                # V_obs_tmp = batch,feat,seq,node
                V_obs_tmp = V_obs.permute(0, 3, 1, 2)

                # Make predictions for this batch
                V_pred, _, simo = stgcnn_model(  # pylint: disable=not-callable
                    V_obs_tmp.to(device), A_obs.squeeze().to(device)
                )

                V_pred = V_pred.permute(0, 2, 3, 1)
                vloss = train_utils.criterion(
                    (V_pred, simo), truth_labels.copy(), device
                )
                running_vloss += vloss

        avg_vloss = running_vloss / (iv + 1)  # pylint: disable=undefined-loop-variable
        log.info(f"LOSS train {avg_loss} valid {avg_vloss}".format(avg_loss, avg_vloss))

        before_lr = optimizer.param_groups[0]["lr"]
        scheduler.step()
        after_lr = optimizer.param_groups[0]["lr"]
        log.info(f"Epoch {epoch_number}: SGD lr {before_lr:.4} -> {after_lr:.4}")
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
            best_epoch = epoch_number
            model_path = (
                f"{dest_dir}/social_collision_stgcnn_{timestamp}_{epoch_number}.pt"
            )
            train_utils.checkpoint(stgcnn_model, model_path)
        elif epoch_number - best_epoch > early_stop_thresh:
            log.warning(f"Early stopped training at epoch {epoch_number}")
            break  # terminate the training loop
        epoch_number += 1


if __name__ == "__main__":
    Fire(train)
