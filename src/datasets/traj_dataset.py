import math

import pandas as pd
from torch.utils.data import Dataset


class TrajectoryDataset(Dataset):
    """
    Dataloder for the Trajectory of NFL players
    """

    def __init__(
        self,
        data: pd.DataFrame,
        obs_len=50,
        pred_len=50,
        skip=1,
        norm_lap_matr=True,
    ):
        """
        Args:
        - csv_dir: Directory containing dataset files in the format
        <frame_id> <ped_id> <x> <y>
        - obs_len: Number of time-steps in input trajectories
        - pred_len: Number of time-steps in output trajectories
        - skip: Number of frames to skip while making the dataset
        - threshold: Minimum error to be considered for non linear traj
        when using a linear predictor
        - delim: Delimiter in the dataset files
        """
        super(TrajectoryDataset, self).__init__()

        self.max_num_players = 22
        self.df = data
        self.obs_len = obs_len
        self.pred_len = pred_len
        self.skip = skip
        self.seq_len = self.obs_len + self.pred_len
        self.norm_lap_matr = norm_lap_matr

        for _, grouped_game in self.df(["gameId", "playId"]):
            frame_data = []
            frames = grouped_game["frameId"].unique()
            for frame in frames:
                frame_data.append(
                    grouped_game[grouped_game.frameId == frame].values.tolist()
                )
        num_sequences = int(math.ceil((len(frames) - self.seq_len + 1) / skip))
        pass
