import logging
import logging.config
import math
from enum import IntEnum
from glob import glob

import networkx as nx
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from src.utils import get_project_root

# setup logger
log_file_path = get_project_root() / "logging.conf"
logging.config.fileConfig(str(log_file_path))
log = logging.getLogger("datasets")


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
        shuffle=False,
        field_width: float = 53.3,  # yards
        field_length: float = 120,  # yards
        max_player_angle: float = 360.0,  # degrees
        num_downs: int = 4,
        num_first_down_yards: int = 10,
        root_dir: str = "../../resources/",
        seed=239,
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
        super().__init__()

        assert datatype in ("train", "validation", "test"), (
            "Datatype needs to be one " "of the following: (train, validation, test)"
        )
        assert max_frames >= min_frames, "min_frames must be greater than max_frames"
        np.random.seed(seed)
        torch.manual_seed(seed)

        # Extract appropriate data by frame range
        self.field_width, self.field_length = field_width, field_length
        self._frame_features = ["x", "y", "dis", "o", "dir", "time"]
        self._plays_features = [
            "yardsToGo",
            "quarter",
            "passProbability",
            "defendersInTheBox",
            "defendersInTheBox",
            "absoluteYardlineNumber",
            "club",
            "ballCarrierId",
        ]
        self._games_features = ["homeTeamAbbr", "club", "possessionTeam"]

        # Keep track of relevant data from csv
        data_dir = root_dir + "nfl-big-data-bowl-2024/"
        self.games_df = pd.read_csv(data_dir + "games.csv")
        self.players_df = pd.read_csv(data_dir + "players.csv")
        self.plays_df = pd.read_csv(data_dir + "plays.csv")
        self.tackles_df = pd.read_csv(data_dir + "tackles.csv")

        self.min_frames, self.max_frames = min_frames, max_frames
        tracking_df, frame_data, fmax, fmin = self.read_tracking_data(
            root_dir=root_dir,
            datatype=datatype,
            min_frames=self.min_frames,
            max_frames=self.max_frames,
        )
        log.info(
            f"Processing {len(frame_data)} plays ranging from {fmin} to {fmax} lengths..."
        )

        # Get player relevant attributes
        self.player_attributes = self.players_df.set_index("nflId")
        self.max_weight = self.player_attributes["weight"].values.max()
        self.min_weight = self.player_attributes["weight"].values.min()
        self.mean_speed, self.std_speed = (
            tracking_df[["s"]].mean(),
            tracking_df[["s"]].std(),
        )
        self.mean_acc, self.std_acc = (
            tracking_df[["a"]].mean(),
            tracking_df[["a"]].std(),
        )
        self.mean_distance_traveled, self.std_distance_traveled = (
            tracking_df[["dis"]].mean(),
            tracking_df[["dis"]].std(),
        )
        self.max_angle = max_player_angle
        self.num_downs = num_downs
        self.num_first_down_yards = num_first_down_yards

        # Attributes for input sequence lengths
        self.obs_len = obs_len
        self.pred_len = pred_len
        self.seq_len = self.obs_len + self.pred_len
        self.norm_lap_matr = norm_lap_matr

        # Format data and labels

        self.data = frame_data  # self.__format_data(frame_data)

        # Dataset attributes
        self.num_features = 21
        self.num_spatial_features = 7
        self.num_nodes = 22
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

    def read_tracking_data(
        self,
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
        tracking_df = tracking_df[tracking_df["displayName"] != "football"]

        # Collect data by gameId and playId
        frame_data = {}
        frame_max, frame_min = -1, 100000
        index = 0

        for _, grouped_game in (
            pbar := tqdm(tracking_df.groupby(["gameId", "playId"]))
        ):
            pbar.set_description()
            current_frames = []
            # Filter by tackle sequences
            if "tackle" not in grouped_game["event"].values:
                continue
            # Filter by sequence length
            num_sequences = len(grouped_game["frameId"].unique())
            if not min_frames <= num_sequences <= max_frames:
                continue

            # Filter any nan values
            merged_plays_df = pd.merge(
                self.plays_df,
                grouped_game,
                left_on=["gameId", "playId"],
                right_on=["gameId", "playId"],
            )
            if merged_plays_df[self._plays_features].isnull().values.any():
                continue

            frame_max = max(frame_max, num_sequences)
            frame_min = min(frame_min, num_sequences)
            frames = grouped_game["frameId"].unique()
            for frame in frames:
                current_frames.append(grouped_game[grouped_game.frameId == frame])

            frame_data[index] = current_frames
            index += 1

        # Add augmentations as reversed forms of original sequence
        if datatype == "train":
            for _, frames in frame_data.copy().items():
                frame_data[index] = frames[::-1]
                index += 1
            log.info(
                f"Finished grouping {len(frame_data)} sequences including augmented instances"
            )
        else:
            log.info(f"Finished grouping {len(frame_data)} sequences")

        return tracking_df, frame_data, frame_max, frame_min

    def __process_data(self, index: int):
        """
        Retrieve element from tracking dataset as a multi-output dict

        Arguments:
            index -- Index to fetch from (gameId, playId) sequences

        Returns:
            Dictionary containing formatted data and labels
        """
        # (batch_size, frames, num_nodes, num_features)
        data, seq_list = np.array([]), np.array([])
        toa, tofc = None, None
        graph = []
        lst_of_frames = self.data[index]
        # Time related attributes for entire play
        for frames in lst_of_frames:
            event, *_ = frames["event"].unique()
            if str(event) == "tackle":
                toa = pd.to_datetime(
                    frames["time"], format="%Y-%m-%d %H:%M:%S.%f"
                ).unique()
            if str(event) == "first_contact":
                tofc = pd.to_datetime(
                    frames["time"], format="%Y-%m-%d %H:%M:%S.%f"
                ).unique()

        first_frame, last_frame = lst_of_frames[0], lst_of_frames[-1]
        start_time = pd.to_datetime(
            first_frame["time"], format="%Y-%m-%d %H:%M:%S.%f"
        ).unique()
        total_time_of_play = (
            pd.to_datetime(last_frame["time"], format="%Y-%m-%d %H:%M:%S.%f").unique()
            - start_time
        ).total_seconds()

        # Parse entire play and collect relevant attributes for each player
        for frames in lst_of_frames:
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
            x_vals, y_vals = (
                frame[["x"]].values / self.field_length,
                frame[["y"]].values / self.field_width,
            )
            s_vals = (frame[["s"]] - self.mean_speed) / self.std_speed
            a_vals = (frame[["a"]] - self.mean_acc) / self.std_acc
            dis_vals = (
                frame[["dis"]] - self.mean_distance_traveled
            ) / self.std_distance_traveled
            o_vals, dir_vals = (
                frame[["o"]].values / self.max_angle,
                frame[["dir"]].values / self.max_angle,
            )
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

            # Game related features
            yards_to_go = (
                merged_plays["yardsToGo"].values / self.num_first_down_yards
            )[..., np.newaxis]
            current_down = (merged_plays["quarter"].values / self.num_downs)[
                ..., np.newaxis
            ]
            pass_probability = (merged_plays["passProbability"].values)[..., np.newaxis]
            num_active_defenders = (merged_plays["defendersInTheBox"].values)[
                ..., np.newaxis
            ]
            yards_to_touchdown = (
                merged_plays["absoluteYardlineNumber"].values / self.field_length
            )[..., np.newaxis]
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

            current = np.concatenate(
                [
                    x_vals,
                    y_vals,
                    s_vals,
                    a_vals,
                    dis_vals,
                    o_vals,
                    dir_vals,
                    collect_time,  # spatial features
                    weight_vals,
                    is_in_possesion,
                    is_ball_carrier,
                    is_home,
                    num_active_defenders,
                    yards_to_go,
                    current_down,
                    pass_probability,
                    yards_to_touchdown,
                    is_first_contact_play,
                    is_tackle_play,
                    is_fumble_play,
                    is_involved,  # categorical
                ],
                axis=1,
            )[np.newaxis, ...]
            # Generate sequences of frames (len_input_seq)
            seq_list = (
                np.concatenate((seq_list, current), axis=0)
                if seq_list.size
                else current
            )
            # Generate graph from relevant attributes
            # (each player connected to all of the other team)
            graph.append(TrajectoryDataset.get_graph(current))

        # Batch sequences of frames (batch_size)
        data = (
            np.concatenate((data, seq_list), axis=0)
            if data.size
            else seq_list[np.newaxis, ...]
        )
        toa_label: bool = toa is not None
        toa_time_label = (
            (toa - start_time).total_seconds()
            if toa is not None
            else -total_time_of_play
        )[0]
        tofc_label: bool = tofc is not None
        tofc_time_label = (
            (tofc - start_time).total_seconds()
            if tofc is not None
            else -total_time_of_play
        )[0]
        # convert from (batch_size, frames, num_nodes, num_features)
        # to the expected format (batch_size, num_features, frames, num_nodes)
        obs_traj, obs_truth = (
            data[:, : self.obs_len],
            data[:, self.obs_len : self.obs_len + self.pred_len],
        )
        graph_traj, graph_truth = (
            graph[: self.obs_len],
            graph[self.obs_len : self.obs_len + self.pred_len],
        )
        return {
            "data": [
                torch.from_numpy(obs_traj.squeeze()).type(torch.float),
                torch.from_numpy(np.array(graph_traj).squeeze()).type(torch.float),
            ],
            "labels": {
                "trajectory": torch.from_numpy(obs_truth.squeeze()).type(torch.float),
                "graph": torch.from_numpy(np.array(graph_truth).squeeze()).type(
                    torch.float
                ),
                "event_type": torch.from_numpy(
                    np.array([[toa_label, tofc_label]])
                ).type(torch.float),
                "node_index": torch.from_numpy(is_involved.T).type(torch.float),
                "time_of_event": torch.from_numpy(
                    np.array([[toa_time_label, tofc_time_label]])
                ).type(torch.float),
            },
        }

    @staticmethod
    def get_graph(single_frame: np.ndarray):
        """
        Generate the adjacency matrix per single frame instance

        Arguments:
            single_frame -- Single frame formatted as (1, num_nodes, num_features)

        Returns:
            Adjacency matrix for given frame
        """
        adj = []
        for elements in single_frame.squeeze():
            current_element = []
            for other_elements in single_frame.squeeze():
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
