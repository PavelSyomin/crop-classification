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
    sequencelength: int = 182
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
    n_iter: int = 50


def make_stats_dataframe(train_results):
    dfs = []
    for n_months, result in train_results.items():
        df = pd.DataFrame(result["stats"])
        df["n_months"] = n_months
        dfs.append(df)

    dataframe = pd.concat(dfs)
    dataframe.reset_index(drop=True, inplace=True)

    return dataframe
