# pylint: disable=arg-type
import logging
import logging.config
import math
import os
from enum import IntEnum
from glob import glob

import networkx as nx
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

# setup logger
# log_file_path = "../../logging.conf"
# logging.config.fileConfig(log_file_path)
# log = logging.getLogger(__name__)


class DATA_COLUMNS(IntEnum):
    X = 0
    Y = 1
    SPEED = 2
    ACC = 3
    DISTANCE_TRAVELED = 4
    ORIENTATION = 5
    DIRECTION = 6
    WEIGHT = 7
    COLLECT_TIME = 8
    IS_IN_POSSESSION = 9
    IS_BALL_CARRIER = 10
    IS_HOME = 11
    IS_FIRST_CONTACT = 12
    IS_TACKLE = 13
    IS_FUMBLE = 14
    IS_INVOLVED = 15
    TIME_TO_ATTACK = 16


class TrajectoryDataset(Dataset):
    """
    Dataloder for the Trajectory of NFL players
    """

    def __init__(
        self,
        datatype: str,
        obs_len=25,
        pred_len=30,
        *,
        min_frames: int = 60,
        max_frames: int = 70,
        batch_size=1,
        norm_lap_matr=True,
        shuffle=True,
        root_dir: str = "../../resources/",
    ):
        """
        Args:
        - obs_len: Number of time-steps in input trajectories
        - pred_len: Number of time-steps in output trajectories
        - min_frames: Minimum number of frames per play
        - max_frames: Maximum number of frames per play
        - batch_size: Number of elements to extract
        - norm_lap_matrix: Whether to compute laplacian norm for graph
        - shuffle: Whether to shuffle indices at each epoch
        - root_dir: Location of NFL Big Data csvs
        """
        super(TrajectoryDataset, self).__init__()

        assert datatype in ("train", "validation", "test"), (
            "Datatype needs to be one " "of the following: (train, validation, test)"
        )
        assert max_frames >= min_frames, "min_frames must be greater than max_frames"

        # Extract appropriate data by frame range
        self.min_frames, self.max_frames = min_frames, max_frames
        tracking_df, self.data, fmax, fmin = TrajectoryDataset.read_tracking_data(
            root_dir=root_dir,
            datatype=datatype,
            min_frames=self.min_frames,
            max_frames=self.max_frames,
        )
        print(
            f"Processing {len(self.data)} plays ranging from  {self.min_frames} to {self.max_frames} lengths..."
        )
        print(f"Data includes frame lengths starting from {fmin} to {fmax}")

        # Keep track of relevant data from csv
        data_dir = root_dir + "nfl-big-data-bowl-2024/"
        self.games_df = pd.read_csv(data_dir + "games.csv")
        self.players_df = pd.read_csv(data_dir + "players.csv")
        self.plays_df = pd.read_csv(data_dir + "plays.csv")
        self.tackles_df = pd.read_csv(data_dir + "tackles.csv")

        # Get player relevant attributes
        self.player_attributes = self.players_df.set_index("nflId")
        self.max_weight = self.player_attributes["weight"].values.max()
        self.min_weight = self.player_attributes["weight"].values.min()
        self.mean_speed, self.std_speed = (
            tracking_df[["s"]].mean(),
            tracking_df[["s"]].std(),
        )
        self.mean_acc, self.std_acc = (
            tracking_df[["s"]].mean(),
            tracking_df[["s"]].std(),
        )

        # Attributes for input sequence lengths
        self.obs_len = obs_len
        self.pred_len = pred_len
        self.seq_len = self.obs_len + self.pred_len
        self.norm_lap_matr = norm_lap_matr

        # Dataset attributes
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        "Denotes the number of batches per epoch"
        return int(np.floor(len(self.data) / self.batch_size))

    def __getitem__(self, index):
        random_index = self.indexes[index]
        return self.__process_data(random_index)

    def on_epoch_end(self):
        "Shuffles indexes after each epoch"
        self.indexes = np.arange(len(self.data))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    @staticmethod
    def read_tracking_data(
        root_dir: str,
        datatype: str,
        min_frames: int,
        max_frames: int,
    ):
        """
        Generate dataset from csvs in root_dir.
            Data to be generated:
                games dataframe
                players dataframe
                plays dataframe
                tackles dataframe
                tracking dataframe

        Arguments:
            root_dir -- Location of csvs
        """

        # Concatenate all tracking data csvs into single dataframe
        tracking_files = []
        for tracking_filename in glob(root_dir + f"{datatype}.csv"):
            file = pd.read_csv(tracking_filename)
        tracking_files.append(file)
        tracking_df = pd.concat(tracking_files, ignore_index=True)

        # Exclude football data for now
        tracking_df.drop(
            tracking_df[tracking_df["displayName"] == "football"].index, inplace=True
        )

        # Collect data by gameId and playId
        frame_data = {}
        frame_max, frame_min = -1, 100000
        index = 0
        for _, grouped_game in tracking_df.groupby(["gameId", "playId"]):
            current_frames = []
            # Filter by tackle sequences
            if "tackle" not in grouped_game["event"].values:
                continue
            # Filter by sequence length
            num_sequences = len(grouped_game["frameId"].unique())
            if not (min_frames <= num_sequences <= max_frames):
                continue

            frame_max = max(frame_max, num_sequences)
            frame_min = min(frame_min, num_sequences)
            frames = grouped_game["frameId"].unique()
            for frame in frames:
                current_frames.append(grouped_game[grouped_game.frameId == frame])

            frame_data[index] = current_frames
            index += 1

        # Add augmentations as reversed forms of original sequence
        for _, frames in frame_data.copy().items():
            frame_data[index] = frames[::-1]
            index += 1

        return tracking_df, frame_data, frame_max, frame_min

    def __process_data(self, index: int):
        """
        Retrieve element from tracking dataset as a multi-output dict

        Arguments:
            index -- Index to fetch from in

        Returns:
            Dictionary containing data and labels
        """
        # (batch_size, frames, num_nodes, num_features)
        data = None
        graph = []
        default_tackle_time = -1
        seq_list = None
        toa = default_tackle_time

        # Time related attributes for entire play
        for frames in self.data[index]:
            event, *_ = frames["event"].unique()
            if str(event) == "tackle":
                toa = pd.to_datetime(
                    frames["time"], format="%Y-%m-%d %H:%M:%S.%f"
                ).unique()

        first_frame, last_frame = self.data[index][0], self.data[index][-1]
        start_time = pd.to_datetime(
            first_frame[0]["time"], format="%Y-%m-%d %H:%M:%S.%f"
        ).unique()
        total_time_of_play = (
            pd.to_datetime(last_frame["time"], format="%Y-%m-%d %H:%M:%S.%f").unique()
            - start_time
        ).total_seconds()

        # Parse entire play and collect relevant attributes for each player
        for frames in self.data[index]:
            frame = frames.sort_values(by=["nflId"])
            merged_games = pd.merge(
                self.games_df, frame, left_on="gameId", right_on="gameId"
            )
            merged_plays = pd.merge(
                self.plays_df,
                frame,
                left_on=["gameId", "playId"],
                right_on=["gameId", "playId"],
            )
            merged_tackles = pd.merge(
                self.tackles_df,
                frame,
                left_on=["gameId", "playId"],
                right_on=["gameId", "playId"],
                suffixes=["_tackler", None],
            )
            # Temporal data takes into account position, velocity, acceleration
            # distance traveled, orientation, angle of player motion,
            # whether player on home team, has possession, is ball carrier
            x_vals, y_vals = frame[["x"]].values / 120, frame[["y"]].values / 53.3
            s_vals = (frame[["s"]] - self.mean_speed) / self.std_speed
            a_vals = (frame[["a"]] - self.mean_acc) / self.std_acc
            dis_vals = frame[["dis"]].values
            o_vals, dir_vals = frame[["o"]].values / 360, frame[["dir"]].values / 360
            weight_vals = (
                (
                    self.player_attributes.loc[frame["nflId"]]["weight"].values
                    - self.min_weight
                )
                / (self.max_weight - self.min_weight)
            )[..., np.newaxis]
            collect_time = (
                pd.to_datetime(frame["time"], format="%Y-%m-%d %H:%M:%S.%f")
                .apply(
                    lambda x: float(
                        ((x - start_time) / total_time_of_play).total_seconds()[0]
                    )
                )
                .values[..., np.newaxis]
            )

            is_home = np.where(
                merged_games["homeTeamAbbr"].values == merged_games["club"].values, 1, 0
            )[..., np.newaxis]
            is_in_possesion = np.where(
                merged_plays["possessionTeam"].values == merged_plays["club"].values,
                1,
                0,
            )[..., np.newaxis]
            is_ball_carrier = np.where(
                merged_plays["nflId"].values == merged_plays["ballCarrierId"].values,
                1,
                0,
            )[..., np.newaxis]

            # Label related attributes
            is_tackle_play = frame[["event"]].isin(["tackle"])
            is_fumble_play = frame[["event"]].isin(["fumble"])
            is_first_contact_play = frame[["event"]].isin(["first_contact"])
            is_involved = np.isin(
                frame[["nflId"]].values, merged_tackles["nflId_tackler"].unique()
            )

            time_to_attack = (
                pd.to_datetime(frame["time"], format="%Y-%m-%d %H:%M:%S.%f")
                .apply(lambda x: float((toa - x).total_seconds()[0]))
                .values[..., np.newaxis]
            )
            current = np.concatenate(
                [
                    x_vals,
                    y_vals,
                    s_vals,
                    a_vals,
                    dis_vals,
                    o_vals,
                    dir_vals,
                    weight_vals,
                    collect_time,  # numerical
                    is_in_possesion,
                    is_ball_carrier,
                    is_home,  # categorical
                    is_first_contact_play,
                    is_tackle_play,
                    is_fumble_play,
                    is_involved,
                    time_to_attack,
                ],  # label-related
                axis=1,
            )[np.newaxis, ...]
            # Generate sequences of frames (len_input_seq)
            seq_list = (
                np.concatenate((seq_list, current), axis=0)
                if seq_list is not None
                else current
            )

            # Generate graph from relevant attributes (each player connected to all of the other team)
            graph.append(TrajectoryDataset.get_graph(current))

        # Batch sequences of frames (batch_size)
        data = (
            np.concatenate((data, seq_list), axis=0)
            if data is not None
            else seq_list[np.newaxis, ...]
        )
        toa_label: bool = toa == default_tackle_time
        toa_time_label = (
            (toa - start_time).total_seconds() if toa != default_tackle_time else [-1]
        )

        # convert from (batch_size, frames, num_nodes, num_features)
        # to the expected format (batch_size, num_features, frames, num_nodes)
        obs_traj, obs_truth = (
            data[:, : self.obs_len + 1],
            data[:, self.obs_len + 1 : self.pred_len],
        )
        graph_traj, graph_truth = (
            graph[: self.obs_len + 1],
            graph[self.obs_len + 1 : self.pred_len],
        )

        return {
            "data": [
                torch.from_numpy(obs_traj).type(torch.float),
                torch.from_numpy(graph_traj).type(torch.float),
            ],
            "labels": {
                "trajectory": torch.from_numpy(obs_truth).type(torch.float),
                "graph": torch.from_numpy(graph_truth).type(torch.float),
                "event_type": torch.Tensor([toa_label]).type(torch.float),
                "node_index": torch.from_numpy(is_involved.T).type(torch.float),
                "time_of_attack": torch.Tensor([toa_time_label]).type(torch.float),
            },
        }

    @staticmethod
    def get_graph(single_frame):
        adj = []
        for elements in single_frame:
            current_element = []
            for other_elements in single_frame:
                is_not_same_team = (
                    elements[DATA_COLUMNS.IS_HOME]
                    != other_elements[DATA_COLUMNS.IS_HOME]
                )
                if is_not_same_team:
                    # Closer players influence current player more
                    current_element.append(
                        TrajectoryDataset.influence(elements, other_elements)
                    )
                    # Team members don't influence current player directly
                else:
                    current_element.append(0)
            adj.append(current_element)

        # Current attributes influence directly
        A = np.matrix(adj) + np.identity(len(adj))
        G = nx.Graph(np.array(A))
        A = nx.normalized_laplacian_matrix(G).toarray()
        return A

    @staticmethod
    def influence(player1: np.ndarray, player2: np.ndarray):
        """
        Compute the influence from one player1 to player2.
        The influence is computed by the l2 norm of
            position, velocity and acceleration
        """
        NORM = math.sqrt(
            (player1[DATA_COLUMNS.X] - player2[DATA_COLUMNS.X]) ** 2
            + (player1[DATA_COLUMNS.Y] - player2[DATA_COLUMNS.Y]) ** 2
        )
        if NORM == 0:
            return 0
        return 1 / (NORM)
