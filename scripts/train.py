import os

import torch.optim as optim
from fire import Fire

from src.models.social_collision_stgcnn import \
    SOCIAL_COLLISION_STGCNN as social_c_stgcnn
from src.train_utils import Trainer

# Defining the model
trainer = Trainer()
model = social_c_stgcnn(
    num_events=trainer.num_events,
    n_stgcnn=trainer.num_spatial,
    n_txpcnn=trainer.num_temporal,
    input_feat=trainer.num_features,
    output_feat=trainer.output_feat,
    seq_len=trainer.seq_len,
    pred_seq_len=trainer.pred_seq_len,
    kernel_size=trainer.kernel_size,
    cnn=trainer.cnn_name,
    pretrained=trainer.pretrained,
    cnn_dropout=trainer.cnn_dropout,
).cuda()

# Training settings

optimizer = optim.SGD(model.parameters(), lr=trainer.lr)
scheduler = trainer.get_scheduler(optimizer=optimizer)

loader_train, loader_val = trainer.get_dataloaders()

if not os.path.exists(trainer.checkpoint_dir):
    os.makedirs(trainer.checkpoint_dir)

# Training
metrics = {"train_loss": [], "val_loss": []}
constant_metrics = {"min_val_epoch": -1, "min_val_loss": 9999999999999999}


def train(epoch):
    model.train()
    loss_batch = 0
    batch_count = 0
    is_fst_loss = True
    loader_len = len(trainer.loader_train)
    turn_point = (
        int(loader_len / trainer.batch_size) * trainer.batch_size
        + loader_len % trainer.batch_size
        - 1
    )

    for cnt, batch in enumerate(trainer.loader_train):
        batch_count += 1
        # Get data
        batch = [tensor.cuda() for tensor in batch]
        (
            obs_traj,
            pred_traj_gt,
            obs_traj_rel,
            pred_traj_gt_rel,
            non_linear_ped,
            loss_mask,
            V_obs,
            A_obs,
            V_tr,
            A_tr,
        ) = batch

        optimizer.zero_grad()
        # Forward
        # V_obs = batch,seq,node,feat
        # V_obs_tmp = batch,feat,seq,node
        V_obs_tmp = V_obs.permute(0, 3, 1, 2)

        V_pred, _ = model(V_obs_tmp, A_obs.squeeze())

        V_pred = V_pred.permute(0, 2, 3, 1)

        V_tr = V_tr.squeeze()
        A_tr = A_tr.squeeze()
        V_pred = V_pred.squeeze()

        if batch_count % args.batch_size != 0 and cnt != turn_point:
            l = graph_loss(V_pred, V_tr)
            if is_fst_loss:
                loss = l
                is_fst_loss = False
            else:
                loss += l

        else:
            loss = loss / args.batch_size
            is_fst_loss = True
            loss.backward()

            if args.clip_grad is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)

            optimizer.step()
            # Metrics
            loss_batch += loss.item()
            print("TRAIN:", "\t Epoch:", epoch, "\t Loss:", loss_batch / batch_count)

    metrics["train_loss"].append(loss_batch / batch_count)


if __name__ == "main":
    Fire(train)
