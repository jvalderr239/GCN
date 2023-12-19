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
log = logging.getLogger()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def train(
    epochs: int,
    dest_dir: str,
    root_dir: Optional[str] = None,
    batch_size: int = 1,
    early_stop_thresh: int = 100,
    **kwargs,
):
    # Defining the model
    if root_dir is None:
        root_dir = str(get_project_root() / "resources") + "/"
    trainer = train_utils.Trainer(**kwargs)
    log.info(f"Working with {device} for training")
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
    loader_test = train_utils.generate_dataloader(
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
        blocks_to_retrain=trainer.blocks_to_retrain,
    ).to(device)

    # Training settings
    optimizer = torch.optim.SGD(stgcnn_model.parameters(), lr=trainer.lr)
    scheduler = trainer.get_scheduler(optimizer=optimizer)

    # Initializing in a separate cell so we can easily add more epochs to the same run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    writer = SummaryWriter(f"{dest_dir}/runs/social_collision_stgcnn_{timestamp}")

    best_vloss = 1_000_000.0

    log.info(f"Training for {epochs} epochs")
    for epoch_number in (pbar := tqdm(range(epochs))):
        pbar.set_description(f"EPOCH {epoch_number + 1}")

        avg_loss, avg_e_acc, avg_n_acc, avg_t_acc = train_utils.train_one_epoch(
            epoch_index=epoch_number,
            model=stgcnn_model,
            training_loader=loader_train,
            optimizer=optimizer,
            tb_writer=writer,
            clip=trainer.clip,
            device=device,
        )

        log.info(f"Time Prediction Training MSE: {avg_t_acc}")
        log.info(f"Event Type Training Accuracy: {avg_e_acc}")
        log.info(f"Node Index Training Accuracy: {avg_n_acc}")

        avg_vloss, avg_ve_acc, avg_vn_acc, avg_vt_acc = train_utils.validate(
            model=stgcnn_model,
            validation_loader=loader_val,
            device=device,
        )

        log.info(f"Time Prediction Validation MSE: {avg_vt_acc}")
        log.info(f"Event Type Validation Accuracy: {avg_ve_acc}")
        log.info(f"Node Index Validation Accuracy: {avg_vn_acc}")

        log.info(f"Training Loss: {avg_loss} Validation Loss: {avg_vloss}")

        before_lr = optimizer.param_groups[0]["lr"]
        scheduler.step()
        after_lr = optimizer.param_groups[0]["lr"]
        log.info(f"Epoch {epoch_number + 1}: SGD lr {before_lr:.4} -> {after_lr:.4}")

        # Log the running loss averaged per batch
        # for both training and validation
        writer.add_scalars(
            "Training vs. Validation Loss",
            {"Training": avg_loss, "Validation": avg_vloss},
            epoch_number + 1,
        )
        writer.add_scalars(
            "Training vs. Validation Time Prediction MSE",
            {"Training": avg_t_acc, "Validation": avg_vt_acc},
            epoch_number + 1,
        )
        writer.add_scalars(
            "Training vs. Validation Event Type Accuracy",
            {"Training": avg_e_acc, "Validation": avg_ve_acc},
            epoch_number + 1,
        )
        writer.add_scalars(
            "Training vs. Validation Node Index Accuracy",
            {"Training": avg_n_acc, "Validation": avg_vn_acc},
            epoch_number + 1,
        )
        writer.flush()

        # Track best performance, and save the model's state
        if avg_vloss < best_vloss:
            best_vloss = avg_vloss
            best_epoch = epoch_number + 1
            model_path = f"{dest_dir}/{stgcnn_model.name.lower()}_{timestamp}_{epoch_number + 1}.pt"
            train_utils.checkpoint(stgcnn_model, model_path)
        elif (epoch_number + 1) - best_epoch > early_stop_thresh:
            log.warning(f"Early stopped training at epoch {epoch_number + 1}")
            break  # terminate the training loop

    log.info("Validating against test set")
    ade, fde, avg_tloss, avg_te_acc, avg_tn_acc, avg_tt_acc = train_utils.test(
        model=stgcnn_model,
        test_loader=loader_test,
        device=device,
    )

    log.info(f"Time Prediction Test MSE: {avg_tt_acc}")
    log.info(f"Event Type Test Accuracy: {avg_te_acc}")
    log.info(f"Node Index Test Accuracy: {avg_tn_acc}")

    log.info(f"Test Loss: {avg_tloss}")
    log.info(f"ADE: {ade}, FDE: {fde}")
    log.info("Finished Training...")


if __name__ == "__main__":
    Fire(train)
