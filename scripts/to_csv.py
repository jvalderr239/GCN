import random

import pandas as pd
from fire import Fire
from tqdm import tqdm

from src.datasets import TrajectoryDataset as dataset


def generate_csv(
    split: float,
    csv_name: str = "tracking*",
    *,
    root_dir: str = "../resources/nfl-big-data-bowl-2024/",
    min_frames: int = 60,
    max_frames: int = 70,
    seed: int = 239,
):
    assert 0 < split < 1, f"Split must be a float. Got {split}"
    frame_data, fmax, fmin = dataset.read_tracking_data(
        root_dir=root_dir,
        datatype=csv_name,
        min_frames=min_frames,
        max_frames=max_frames,
    )
    random.shuffle(frame_data)
    train_size = int(len(frame_data) * split)
    val_size = int((len(frame_data) - train_size) * 0.5)

    train, val, test = pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    pbar = tqdm(total=train_size)
    for frames in frame_data[:train_size]:
        pbar.update(1)
        inter = pd.concat(frames, axis=0)
        train = pd.concat((train, inter), axis=0)
    pbar.close()
    train.to_csv(root_dir + "train.csv")

    pbar = tqdm(total=val_size)
    for frames in frame_data[train_size : train_size + val_size]:
        pbar.update(1)
        inter = pd.concat(frames, axis=0)
        val = pd.concat((val, inter), axis=0)
    pbar.close()
    val.to_csv(root_dir + "validation.csv")

    pbar = tqdm(total=len(frame_data) - len(val))
    for frames in frame_data[train_size + val_size :]:
        pbar.update(1)
        inter = pd.concat(frames, axis=0)
        test = pd.concat((test, inter), axis=0)
    pbar.close()
    test.to_csv(root_dir + "test.csv")
    print("Finished generating csvs...")


if __name__ == "main":
    Fire(generate_csv)
