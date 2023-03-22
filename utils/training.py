from dataclasses import dataclass
import os
import torch


@dataclass
class TrainConfig:
    dataset: str = "russia"
    alpha: float = 0.5
    epsilon: float = 10
    learning_rate: float = 10e-3
    weight_decay: float = 0
    patience: int = 30
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    epochs: int = 50
    sequencelength: int = 183
    batchsize: int = 128
    dataroot: str = os.path.join(os.environ["HOME"],"elects_data")
    snapshot: str = "snapshots/model.pth"
    resume: bool = False
    year: int = 2021
    use_cache: bool = True
    model: str = "earlyrnn"
    n_months: int = 6
    visualize: bool = False
    hyperparameters: dict = None
    geo: bool = False


def make_stats_df(stats_list):
    dfs = []
    for i, stat in enumerate(stats_list):
        df = pd.DataFrame(stat)
        df["n_months"] = i + 1
        dfs.append(df)
    result = pd.concat(dfs)
    result.reset_index(drop=True, inplace=True)

    return result
