from data import BavarianCrops, BreizhCrops, ModisCDL, SustainbenchCrops, Russia
from torch.utils.data import DataLoader
from earlyrnn import EarlyRNN
import torch
from tqdm import tqdm
from loss import EarlyRewardLoss
import numpy as np
from utils import VisdomLogger
import sklearn.metrics
import pandas as pd
import argparse
import os
import random
from temp_cnn_model import TempCNN
from transformer_model import TransformerModel


seed = 42
torch.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)


def parse_args():
    parser = argparse.ArgumentParser(description='Run ELECTS Early Classification training on the BavarianCrops dataset.')
    parser.add_argument('--dataset', type=str, default="bavariancrops", choices=["bavariancrops","breizhcrops", "ghana", "southsudan", "unitedstates", "russia"], help="dataset")
    parser.add_argument('--alpha', type=float, default=0.5, help="trade-off parameter of earliness and accuracy (eq 6): "
                                                                 "1=full weight on accuracy; 0=full weight on earliness")
    parser.add_argument('--epsilon', type=float, default=10, help="additive smoothing parameter that helps the "
                                                                  "model recover from too early classificaitons (eq 7)")
    parser.add_argument('--learning-rate', type=float, default=1e-3, help="Optimizer learning rate")
    parser.add_argument('--weight-decay', type=float, default=0, help="weight_decay")
    parser.add_argument('--patience', type=int, default=30, help="Early stopping patience")
    parser.add_argument('--device', type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        choices=["cuda", "cpu"], help="'cuda' (GPU) or 'cpu' device to run the code. "
                                                     "defaults to 'cuda' if GPU is available, otherwise 'cpu'")
    parser.add_argument('--epochs', type=int, default=100, help="number of training epochs")
    parser.add_argument('--sequencelength', type=int, default=180, help="sequencelength of the time series. If samples are shorter, "
                                                                "they are zero-padded until this length; "
                                                                "if samples are longer, they will be undersampled")
    parser.add_argument('--batchsize', type=int, default=128, help="number of samples per batch")
    parser.add_argument('--dataroot', type=str, default=os.path.join(os.environ["HOME"],"elects_data"), help="directory to download the "
                                                                                 "BavarianCrops dataset (400MB)."
                                                                                 "Defaults to home directory.")
    parser.add_argument('--snapshot', type=str, default="snapshots/model.pth",
                        help="pytorch state dict snapshot file")
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--year', type=int, default=2021, help="year for Russia dataset (2018–2022) or None for all years")
    parser.add_argument('--use_cache', action="store_true", help="save preprocessed data in cache files")
    parser.add_argument('--model', type=str, default="earlyrnn", choices=["earlyrnn", "transformer", "tempcnn"], help="model to use")
    parser.add_argument('--n_months', type=int, default=6, help="use selected number of months from April 1. E.g. if this argument is set to 3, then the model will use data from April 1 to June 30")

    args = parser.parse_args()

    if args.patience < 0:
        args.patience = None

    return args

def main(args):

    if args.model not in ("earlyrnn", "transformer", "tempcnn"):
        raise ValueError(f"Unrecognized model {args.model}")

    if args.dataset == "bavariancrops":
        dataroot = os.path.join(args.dataroot,"bavariancrops")
        nclasses = 7
        input_dim = 13
        class_weights = None
        train_ds = BavarianCrops(root=dataroot,partition="train", sequencelength=args.sequencelength)
        test_ds = BavarianCrops(root=dataroot,partition="valid", sequencelength=args.sequencelength)
    elif args.dataset == "unitedstates":
        args.dataroot = "/data/modiscdl/"
        args.sequencelength = 24
        dataroot = args.dataroot
        nclasses = 8
        input_dim = 1
        train_ds = ModisCDL(root=dataroot,partition="train", sequencelength=args.sequencelength)
        test_ds = ModisCDL(root=dataroot,partition="valid", sequencelength=args.sequencelength)
    elif args.dataset == "breizhcrops":
        dataroot = os.path.join(args.dataroot,"breizhcrops")
        nclasses = 9
        input_dim = 13
        train_ds = BreizhCrops(root=dataroot,partition="train", sequencelength=args.sequencelength)
        test_ds = BreizhCrops(root=dataroot,partition="valid", sequencelength=args.sequencelength)
    elif args.dataset in ["ghana"]:
        use_s2_only = False
        average_pixel = False
        max_n_pixels = 50
        dataroot = args.dataroot
        nclasses = 4
        input_dim = 12 if use_s2_only else 19  # 12 sentinel 2 + 3 x sentinel 1 + 4 * planet
        args.epochs = 500
        args.sequencelength = 365
        train_ds = SustainbenchCrops(root=dataroot,partition="train", sequencelength=args.sequencelength,
                                     country="ghana",
                                     use_s2_only=use_s2_only, average_pixel=average_pixel,
                                     max_n_pixels=max_n_pixels)
        val_ds = SustainbenchCrops(root=dataroot,partition="val", sequencelength=args.sequencelength,
                                    country="ghana", use_s2_only=use_s2_only, average_pixel=average_pixel,
                                    max_n_pixels=max_n_pixels)

        train_ds = torch.utils.data.ConcatDataset([train_ds, val_ds])

        test_ds = SustainbenchCrops(root=dataroot,partition="test", sequencelength=args.sequencelength,
                                    country="ghana", use_s2_only=use_s2_only, average_pixel=average_pixel,
                                    max_n_pixels=max_n_pixels)
    elif args.dataset in ["southsudan"]:
        use_s2_only = False
        dataroot = args.dataroot
        nclasses = 4
        args.sequencelength = 365
        input_dim = 12 if use_s2_only else 19 # 12 sentinel 2 + 3 x sentinel 1 + 4 * planet
        args.epochs = 500
        train_ds = SustainbenchCrops(root=dataroot,partition="train", sequencelength=args.sequencelength, country="southsudan", use_s2_only=use_s2_only)
        val_ds = SustainbenchCrops(root=dataroot,partition="val", sequencelength=args.sequencelength, country="southsudan", use_s2_only=use_s2_only)

        train_ds = torch.utils.data.ConcatDataset([train_ds, val_ds])
        test_ds = SustainbenchCrops(root=dataroot, partition="val", sequencelength=args.sequencelength,
                                   country="southsudan", use_s2_only=use_s2_only)
    elif args.dataset in ["russia"]:
        dataroot = os.path.join(args.dataroot, "russia")
        nclasses = 13
        input_dim = 10
        class_weights = None
        if args.year is None:
            years_range = range(2018, 2023)
        else:
            years_range = (args.year,)
        broadcast_y = True if args.model == "earlyrnn" else False
        train_datasets = [
            Russia(root=dataroot,
                   partition="train",
                   sequencelength=args.sequencelength,
                   year=current_year,
                   use_cache=args.use_cache,
                   return_id=False,
                   broadcast_y=broadcast_y,
                   n_months=args.n_months)
            for current_year in years_range
        ]
        test_datasets = [
            Russia(root=dataroot,
                   partition="test",
                   sequencelength=args.sequencelength,
                   year=current_year,
                   use_cache=args.use_cache,
                   return_id=False,
                   broadcast_y=broadcast_y,
                   n_months=args.n_months)
            for current_year in years_range
        ]
        train_ds = torch.utils.data.ConcatDataset(train_datasets)
        test_ds = torch.utils.data.ConcatDataset(test_datasets)

    else:
        raise ValueError(f"dataset {args.dataset} not recognized")

    print(f"Total length of data: train={len(train_ds)}, test={len(test_ds)}")

    traindataloader = DataLoader(
        train_ds,
        batch_size=args.batchsize, shuffle=True)
    testdataloader = DataLoader(
        test_ds,
        batch_size=args.batchsize, shuffle=True)

    print("X shape:", train_ds[0][0].shape, "y shape:", train_ds[0][1].shape)

    if args.model == "earlyrnn":
        model = EarlyRNN(nclasses=nclasses, input_dim=input_dim).to(args.device)
    elif args.model == "transformer":
        model = TransformerModel(
            input_dim=input_dim,
            num_classes=nclasses,
            d_model=args.sequencelength,
            n_head=1,
            n_layers=3,
            d_inner=512,
            activation="relu",
            dropout=0.4,
        ).to(args.device)
    elif args.model == "tempcnn":
        model = TempCNN(
            input_dim=input_dim,
            num_classes=nclasses,
            sequencelength=args.sequencelength,
        ).to(args.device)

    #optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    # exclude decision head linear bias from weight decay
    decay, no_decay = list(), list()
    for name, param in model.named_parameters():
        if name == "stopping_decision_head.projection.0.bias":
            no_decay.append(param)
        else:
            decay.append(param)
    if args.model == "earlyrnn":
        optimizer = torch.optim.AdamW([{'params': no_decay, 'weight_decay': 0, "lr": args.learning_rate}, {'params': decay}],
                                      lr=args.learning_rate, weight_decay=args.weight_decay)
        criterion = EarlyRewardLoss(alpha=args.alpha, epsilon=args.epsilon)
    elif args.model == "transformer":
        optimizer = torch.optim.Adam(
            filter(lambda x: x.requires_grad, model.parameters()),
            betas=(0.9, 0.98),
            eps=1e-09,
            #     ν = 1.31 · 10−3 and λ = 5.52 · 10−8
            weight_decay=5.52e-8,
            lr=1.31e-3,
        )
        criterion = torch.nn.NLLLoss()
    elif args.model == "tempcnn":
        optimizer = torch.optim.Adam(
            filter(lambda x: x.requires_grad, model.parameters()),
            weight_decay=5e-8,
            lr=args.learning_rate
        )
        criterion = torch.nn.NLLLoss()

    if args.resume and os.path.exists(args.snapshot):
        model.load_state_dict(torch.load(args.snapshot, map_location=args.device))
        optimizer_snapshot = os.path.join(os.path.dirname(args.snapshot),
                                          os.path.basename(args.snapshot).replace(".pth", "_optimizer.pth")
                                          )
        optimizer.load_state_dict(torch.load(optimizer_snapshot, map_location=args.device))
        df = pd.read_csv(args.snapshot + ".csv")
        train_stats = df.to_dict("records")
        start_epoch = train_stats[-1]["epoch"]
        print(f"resuming from {args.snapshot} epoch {start_epoch}")
    else:
        train_stats = []
        start_epoch = 1
    visdom_logger = VisdomLogger()

    not_improved = 0
    with tqdm(range(start_epoch, args.epochs + 1)) as pbar:
        for epoch in pbar:
            if args.model == "earlyrnn":
                trainloss = train_epoch(model, traindataloader, optimizer, criterion, device=args.device)
                testloss, stats = test_epoch(model, testdataloader, criterion, args.device)
            elif args.model in ("transformer", "tempcnn"):
                trainloss = train_epoch_transformer(model, traindataloader, optimizer, criterion, device=args.device)
                testloss, stats = test_epoch_transformer(model, testdataloader, criterion, args.device)

            # statistic logging and visualization...
            precision, recall, fscore, support = sklearn.metrics.precision_recall_fscore_support(
                y_pred=stats["predictions_at_t_stop"][:, 0], y_true=stats["targets"][:, 0], average="macro",
                zero_division=0)
            accuracy = sklearn.metrics.accuracy_score(
                y_pred=stats["predictions_at_t_stop"][:, 0], y_true=stats["targets"][:, 0])
            kappa = sklearn.metrics.cohen_kappa_score(
                stats["predictions_at_t_stop"][:, 0], stats["targets"][:, 0])

            classification_loss = stats["classification_loss"].mean()
            earliness_reward = stats["earliness_reward"].mean()
            earliness = 1 - (stats["t_stop"].mean() / (args.sequencelength - 1))

            stats["confusion_matrix"] = sklearn.metrics.confusion_matrix(y_pred=stats["predictions_at_t_stop"][:, 0],
                                                                         y_true=stats["targets"][:, 0])

            train_stats.append(
                dict(
                    epoch=epoch,
                    trainloss=trainloss,
                    testloss=testloss,
                    accuracy=accuracy,
                    precision=precision,
                    recall=recall,
                    fscore=fscore,
                    kappa=kappa,
                    earliness=earliness,
                    classification_loss=classification_loss,
                    earliness_reward=earliness_reward
                )
            )

            visdom_logger(stats)
            visdom_logger.plot_boxplot(stats["targets"][:, 0], stats["t_stop"][:, 0], tmin=0, tmax=args.sequencelength)
            df = pd.DataFrame(train_stats).set_index("epoch")
            visdom_logger.plot_epochs(df[["precision", "recall", "fscore", "kappa"]], name="accuracy metrics")
            visdom_logger.plot_epochs(df[["trainloss", "testloss"]], name="losses")
            visdom_logger.plot_epochs(df[["accuracy", "earliness"]], name="accuracy, earliness")
            visdom_logger.plot_epochs(df[["classification_loss", "earliness_reward"]], name="loss components")

            savemsg = ""
            if len(df) > 2:
                if testloss < df.testloss.iloc[:-1].values.min():
                    savemsg = f"saving model to {args.snapshot}"
                    os.makedirs(os.path.dirname(args.snapshot), exist_ok=True)
                    torch.save(model.state_dict(), args.snapshot)

                    optimizer_snapshot = os.path.join(os.path.dirname(args.snapshot),
                                                      os.path.basename(args.snapshot).replace(".pth", "_optimizer.pth")
                                                      )
                    torch.save(optimizer.state_dict(), optimizer_snapshot)

                    df.to_csv(args.snapshot + ".csv")
                    not_improved = 0 # reset early stopping counter
                else:
                    not_improved += 1 # increment early stopping counter
                    if args.patience is not None:
                        savemsg = f"early stopping in {args.patience - not_improved} epochs."
                    else:
                        savemsg = ""

            pbar.set_description(f"epoch {epoch}: trainloss {trainloss:.2f}, testloss {testloss:.2f}, "
                     f"accuracy {accuracy:.2f}, earliness {earliness:.2f}. "
                     f"classification loss {classification_loss:.2f}, earliness reward {earliness_reward:.2f}. {savemsg}")

            if args.patience is not None:
                if not_improved > args.patience:
                    print(f"stopping training. testloss {testloss:.2f} did not improve in {args.patience} epochs.")
                    break


def train_epoch(model, dataloader, optimizer, criterion, device):
    losses = []
    model.train()
    for batch in dataloader:
        optimizer.zero_grad()
        X, y_true = batch
        X, y_true = X.to(device), y_true.to(device)

        log_class_probabilities, probability_stopping = model(X)

        loss = criterion(log_class_probabilities, probability_stopping, y_true)

        #assert not loss.isnan().any()
        if not torch.isnan(loss).any():
            loss.backward()
            optimizer.step()

            losses.append(loss.cpu().detach().numpy())

    return np.stack(losses).mean()

def test_epoch(model, dataloader, criterion, device):
    model.eval()

    stats = []
    losses = []
    for batch in dataloader:
        X, y_true = batch
        X, y_true = X.to(device), y_true.to(device)

        log_class_probabilities, probability_stopping, predictions_at_t_stop, t_stop = model.predict(X)
        loss, stat = criterion(log_class_probabilities, probability_stopping, y_true, return_stats=True)

        stat["loss"] = loss.cpu().detach().numpy()
        stat["probability_stopping"] = probability_stopping.cpu().detach().numpy()
        stat["class_probabilities"] = log_class_probabilities.exp().cpu().detach().numpy()
        stat["predictions_at_t_stop"] = predictions_at_t_stop.unsqueeze(-1).cpu().detach().numpy()
        stat["t_stop"] = t_stop.unsqueeze(-1).cpu().detach().numpy()
        stat["targets"] = y_true.cpu().detach().numpy()

        stats.append(stat)

        losses.append(loss.cpu().detach().numpy())

    # list of dicts to dict of lists
    stats = {k: np.vstack([dic[k] for dic in stats]) for k in stats[0]}
    #print(stats["predictions_at_t_stop"], stats["targets"])

    return np.stack(losses).mean(), stats

def train_epoch_transformer(model, dataloader, optimizer, criterion, device):
    losses = []
    model.train()

    for batch in dataloader:
        optimizer.zero_grad()
        X, y_true, *_ = batch
        X, y_true = X.to(device), y_true.to(device)

        #print(X[0, :5, :])
        #break
        logprobabilities = model(X)
        loss = criterion(logprobabilities, y_true)
        #print(loss)

        if not torch.isnan(loss).any():
            loss.backward()
            optimizer.step()

            losses.append(loss.cpu().detach().numpy())
        else:
            print("Loss contain NaN")
    #print(np.stack(losses))
    return np.stack(losses).mean()


def test_epoch_transformer(model, dataloader, criterion, device):
    model.eval()

    stats = []
    losses = []
    with torch.no_grad():
        for batch in dataloader:
            X, y_true, *_ = batch
            X, y_true = X.to(device), y_true.to(device)
            batch_size, sequencelength, _ = X.shape

            log_class_probabilities = model(X)
            loss = criterion(log_class_probabilities, y_true)
            loss = loss.cpu().detach().numpy()

            predictions_at_t_stop = log_class_probabilities.argmax(1)
            t_stop = np.full((batch_size, 1), fill_value=sequencelength - 1)
            probability_stopping = np.zeros((batch_size, sequencelength))
            probability_stopping[:, -1] = 1

            stat = {
                "classification_loss": loss,
                "earliness_reward": np.array([0.0]),
                "probability_making_decision": np.ones((batch_size, sequencelength)),
            }

            stat["loss"] = loss

            stat["probability_stopping"] = probability_stopping
            stat["class_probabilities"] = log_class_probabilities.unsqueeze(1).repeat(1, sequencelength, 1).exp().cpu().detach().numpy()
            stat["predictions_at_t_stop"] = predictions_at_t_stop.unsqueeze(-1).cpu().detach().numpy()
            stat["t_stop"] = t_stop
            stat["targets"] = y_true.unsqueeze(1).repeat(1, sequencelength).cpu().detach().numpy()

            stats.append(stat)

            losses.append(loss)

    # list of dicts to dict of lists
    stats = {k: np.vstack([dic[k] for dic in stats]) for k in stats[0]}
    #print(stats["predictions_at_t_stop"], stats["targets"])

    return np.stack(losses).mean(), stats


def default_transform(x: np.ndarray, sequencelength: int) -> np.ndarray:
    # choose with replacement if sequencelength smaller als choose_t
    replace = False if x.shape[0] >= sequencelength else True
    idxs = np.random.choice(x.shape[0], sequencelength, replace=replace)
    idxs.sort()
    x = x[idxs]
    return x


def shift_transform(x: np.ndarray, sequencelength: int) -> np.ndarray:
    max_shift = x.shape[0] - sequencelength
    if max_shift <= 0:
        return x
    shift = np.random.randint(0, max_shift - 1)
    x = x[shift : shift + sequencelength]
    return x


if __name__ == '__main__':
    args = parse_args()
    main(args)
