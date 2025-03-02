# from ctypes import Union
from dataclasses import dataclass
import datetime
import os
import pickle
from typing import List, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import tqdm


class Russia(Dataset):
    BANDS = [
        "B02",
        "B03",
        "B04",
        "B05",
        "B06",
        "B07",
        "B08",
        "B8A",
        "B11",
        "B12",
    ] # Sentinel-2A bands to use
    CACHE_DIR = "cache"
    NORMALIZATION_FACTOR = 1e-4 # scale bands values from [0, 10000] to [0, 1]
    PARTITIONS = ("train", "test")
    TIME_INTERVAL_START = (4, 1)
    TIME_INTERVALS_END = {
        1: (4, 30),
        2: (5, 31),
        3: (6, 30),
        4: (7, 31),
        5: (8, 31),
        6: (9, 30),
    }
    YEARS = range(2018, 2023)

    def __init__(
        self,
        root,
        partition="train",
        sequencelength=60,
        year=2018,
        return_id=False,
        use_cache=True,
        broadcast_y=True,
        n_months=6,
    ):
        # checks
        assert year in self.YEARS
        assert partition in self.PARTITIONS

        # paths
        features_filepath = os.path.join(root,
                                         f"russia-{year}",
                                         f"{partition}_features.csv.zip")
        classmapping_path = os.path.join(root, "classmapping.csv")
        fieldsmapping_path = os.path.join(root,
                                          f"russia-{year}",
                                          "parcelsmapping.csv")

        # set object attributes
        self.year = year
        self.sequencelength = sequencelength
        self.partition = partition
        self.return_id = return_id
        self.broadcast_y = broadcast_y
        self.n_months = n_months
        self.use_cache = use_cache

        # create cache dir if it does not exist
        if not os.path.exists(self.CACHE_DIR):
            os.mkdir(self.CACHE_DIR)

        self.cache_file = os.path.join(
            self.CACHE_DIR,
            f"russia-{self.year}-{self.partition}-{self.n_months}m.dump"
        )

        print(f"Data: {self.partition}, year: {self.year}, months: {self.n_months}")

        # load and preprocess data
        self.classmapping_df = pd.read_csv(classmapping_path)
        self.fieldsmapping_df = pd.read_csv(fieldsmapping_path)
        self.fieldid2classid = {
            row.field_id: row.class_id
            for _, row in self.fieldsmapping_df.iterrows()
        }
        self.cropid2cropname = {
            row["class_id"]: row["class_name"]
            for _, row in self.classmapping_df.iterrows()
        }

        if self.use_cache and os.path.exists(self.cache_file):
            print("Loading from cache")
            self.load_from_cache()
        else:
            print(f"Reading from disk")
            self.features_df = pd.read_csv(features_filepath)
            self.preprocess_features()
            self.get_xy()
            self.save_to_cache()

        self.items_count = len(self.X_list)
        print(f"# of fields: {self.items_count}")

    def __getitem__(self, index):
        # Special case for string indices X and y
        # used to get all X or y values as ndarrays
        # for Random Forest / boosting methods
        if type(index) is str:
            if index == "X":
                X_mapped = map(
                    lambda x: self.adjust_to_sequencelength(x).flatten(),
                    self.X_list
                )
                X = np.stack(list(X_mapped))
                return X
            elif index == "y":
                y = np.array(self.y_list)
                return y
            else:
                raise IndexError("String index must be either X or y")

        # General case for torch dataloader
        X = self.X_list[index]

        y = self.y_list[index]
        if self.broadcast_y:
            y = np.full(X.shape[0], fill_value=y)

        X, y = self.adjust_to_sequencelength(X, y)

        X = torch.from_numpy(X).type(torch.float)
        y = torch.from_numpy(y).type(torch.long)

        if self.return_id:
            return X, y, self.field_ids_list[index]
        else:
            return X, y

    def __len__(self):
        return self.items_count

    def load_from_cache(self):
        try:
            with open(self.cache_file, "rb") as f:
                self.X_list, self.y_list, self.field_ids_list = pickle.load(f)
        except Exception as e:
            raise RuntimeError("Cannot load data from cache") from e

    def preprocess_features(self):
        print("Preprocessing features")
        self.features_df["timestamp"] = self.features_df["timestamp"].apply(
            lambda x: x.split(" ")[0]
        )
        self.features_df["timestamp"] = pd.to_datetime(self.features_df["timestamp"])
        self.features_df.set_index("timestamp", inplace=True)
        start_date = '-'.join(map(str, [self.year, *self.TIME_INTERVAL_START]))
        end_date = '-'.join(map(str, [self.year, *self.TIME_INTERVALS_END[self.n_months]]))
        self.features_df = self.features_df.loc[start_date:end_date]

    def interpolate_transform(self, input_timeseries):
        data = input_timeseries[self.BANDS]
        data = data.reindex(
            pd.date_range(
                start=datetime.datetime(data.index[0].year, *self.TIME_INTERVAL_START),
                end=datetime.datetime(data.index[0].year, *self.TIME_INTERVALS_END[self.n_months]),
                freq="1D",
            )
        )
        data = data.interpolate(method="linear")
        data = data.fillna(method="ffill").fillna(method="bfill")
        # normalize only bands features
        b_cols = [
            feature
            for feature in self.features_df[self.BANDS].columns
            if feature[0] == "B"
        ]
        data[b_cols] = data[b_cols] * self.NORMALIZATION_FACTOR
        return data

    def get_xy(self):
        print("Preparing X and y")
        self.X_list, self.y_list, self.field_ids_list = [], [], []
        for field_id, sub_df in tqdm.tqdm(self.features_df.groupby("field_id")):
            bands_data = self.interpolate_transform(sub_df[self.BANDS]).to_numpy()
            self.X_list.append(bands_data)
            self.y_list.append(np.array(sub_df["class_id"].values[0]))
            self.field_ids_list.append(field_id)

    def adjust_to_sequencelength(self, X, y=None):
        sample_length = X.shape[0]

        if sample_length < self.sequencelength:
            # time series shorter than "sequencelength" will be zero-padded
            npad = self.sequencelength - sample_length
            X = np.pad(X, [(0, npad), (0, 0)], 'constant', constant_values=0)
            if y is not None and self.broadcast_y:
                y = np.pad(y, (0, npad), 'constant', constant_values=0)
        elif sample_length > self.sequencelength:
            # time series longer than "sequencelength" will be sub-sampled
            idxs = np.random.choice(sample_length, self.sequencelength, replace=False)
            idxs.sort()
            X = X[idxs]
            if y is not None and self.broadcast_y:
                y = y[idxs]

        if y is not None:
            return X, y
        else:
            return X

    def save_to_cache(self):
        try:
            with open(self.cache_file, "wb") as f:
                pickle.dump([self.X_list, self.y_list, self.field_ids_list], f)
        except Exception as e:
            print(f"Cannot save data to cache: {e}")

    @property
    def crops(self):
        return self.cropid2cropname


