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
    epoch = checkpoint["epoch"]
    loss = checkpoint["loss"]

    # Initializing in a separate cell so we can easily add more epochs to the same run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    writer = SummaryWriter(
        "{}/runs/fashion_trainer_{}".format(str(get_project_root()), timestamp)
    )
    epoch_number = 0

    best_vloss = 1_000_000.0

    for epoch in range(epochs):
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
            for i, vdata in enumerate(loader_val):
                vinputs, vlabels = vdata
                voutputs = model(vinputs)
                vloss = loss_fn(voutputs, vlabels)
                running_vloss += vloss

        avg_vloss = running_vloss / (i + 1)
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
            model_path = "model_{}_{}".format(timestamp, epoch_number)
            torch.save(model.state_dict(), model_path)

        epoch_number += 1


if __name__ == "main":
    Fire(train)
