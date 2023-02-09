# from ctypes import Union
from dataclasses import dataclass
import datetime
import os
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
    TIME_INTERVAL = [(4, 1), (9, 30)] # (month, day_of_month)
    NORMALIZATION_FACTOR = 1e-4 # scale bands values from [0, 10000] to [0, 1]

    def __init__(
        self,
        root,
        partition="train",
        sequencelength=182,
        year=2018,
        return_id=False
    ):
        # paths
        # we use only selected year
        features_filepath = os.path.join(root,
                                         f"russia-{year}",
                                         f"{partition}_features.csv.zip")
        classmapping_path = os.path.join(root, "classmapping.csv")
        fieldsmapping_path = os.path.join(root,
                                          f"russia-{year}",
                                          "parcelsmapping.csv")

        # set object attributes
        self.sequencelength = sequencelength
        assert partition in ("train", "test")
        self.partition = partition
        #if self.partition == "train":
        #    self.transform = default_transform
        #else:
        #   self.transform = shift_transform
        self.classmapping_df = pd.read_csv(classmapping_path)
        self.fieldsmapping_df = pd.read_csv(fieldsmapping_path)
        self.fieldid2classid = {
            row.field_id: row.class_id
            for _, row in self.fieldsmapping_df.iterrows()
        }
        self.return_id = return_id

        # load and preprocess data
        print(f"Reading {self.partition} features")
        self.features_df = pd.read_csv(features_filepath)
        self.items_count = self.features_df.field_id.nunique()
        self.preprocess_features()
        self.get_xy()

        print(
            f"Russia dataset for {year} year ({self.partition} part) is loaded.",
            f"It contains {self.items_count} fields"
        )

    def __getitem__(self, index):
        # dataset[index]
        X = self.X_list[index]

        # get length of this sample
        t = X.shape[0]

        y = np.array([self.y_list[index]] * t)

        if t < self.sequencelength:
            # time series shorter than "sequencelength" will be zero-padded
            npad = self.sequencelength - t
            X = np.pad(X, [(0, npad), (0, 0)], 'constant', constant_values=0)
            y = np.pad(y, (0, npad), 'constant', constant_values=0)
        elif t > self.sequencelength:
            # time series longer than "sequencelength" will be sub-sampled
            idxs = np.random.choice(t, self.sequencelength, replace=False)
            idxs.sort()
            X = X[idxs]
            y = y[idxs]

        X = torch.from_numpy(X).type(torch.float)
        y = torch.from_numpy(y).type(torch.long)

        if self.return_id:
            return X, y, self.field_ids_list[index]
        else:
            return X, y

    def __len__(self):
        # len(dataset)
        return self.items_count

    def preprocess_features(self):
        print("Preprocessing features")
        self.features_df["timestamp"] = self.features_df["timestamp"].apply(
            lambda x: x.split(" ")[0]
        )
        self.features_df["timestamp"] = pd.to_datetime(self.features_df["timestamp"])
        self.features_df.set_index("timestamp", inplace=True)

    def interpolate_transform(self, input_timeseries):
        data = input_timeseries[self.BANDS]
        data = data.reindex(
            pd.date_range(
                start=datetime.datetime(data.index[0].year, *self.TIME_INTERVAL[0]),
                end=datetime.datetime(data.index[0].year, *self.TIME_INTERVAL[1]),
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
            self.X_list.append(
                self.interpolate_transform(sub_df[self.BANDS]).to_numpy()
            )
            self.y_list.append(np.array(sub_df["class_id"].values[0]))
            self.field_ids_list.append(field_id)
