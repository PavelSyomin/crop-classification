import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
import os
#from models.EarlyClassificationModel import EarlyClassificationModel
from torch.nn.modules.transformer import TransformerEncoder, TransformerEncoderLayer
from torch.nn.modules import LayerNorm, Linear, Sequential, ReLU


class EarlyTransformer(nn.Module):
    def __init__(self, input_dim=13, nclasses=7, d_inner=128, d_model=182, nhead=2, nlayers=2, activation="relu", dropout=0.017998950510888446):
        super(EarlyTransformer, self).__init__()

        # input transformations
        self.intransforms = nn.Sequential(
            nn.LayerNorm(input_dim), # normalization over D-dimension. T-dimension is untouched
        )

        self.backbone = TransformerBackbone(
                input_dim=input_dim,
                num_classes=nclasses,
                d_model=d_model,
                n_head=nhead,
                n_layers=nlayers,
                d_inner=d_inner,
                activation=activation,
                dropout=dropout
        )

        # Heads
        self.classification_head = ClassificationHead(d_model, nclasses)
        self.stopping_decision_head = DecisionHead(d_model)

    def forward(self, x):
        x = self.intransforms(x)

        outputs = self.backbone(x)
        log_class_probabilities = self.classification_head(outputs)
        probabilitiy_stopping = self.stopping_decision_head(outputs)

        return log_class_probabilities, probabilitiy_stopping

    @torch.no_grad()
    def predict(self, x):
        logprobabilities, deltas = self.forward(x)

        def sample_stop_decision(delta):
            dist = torch.stack([1 - delta, delta], dim=1)
            return torch.distributions.Categorical(dist).sample().bool()

        batchsize, sequencelength, nclasses = logprobabilities.shape

        stop = list()
        for t in range(sequencelength):
            # stop if sampled true and not stopped previously
            if t < sequencelength - 1:
                stop_now = sample_stop_decision(deltas[:, t])
                stop.append(stop_now)
            else:
                # make sure to stop last
                last_stop = torch.ones(tuple(stop_now.shape)).bool()
                if torch.cuda.is_available():
                    last_stop = last_stop.cuda()
                stop.append(last_stop)

        # stack over the time dimension (multiple stops possible)
        stopped = torch.stack(stop, dim=1).bool()

        # is only true if stopped for the first time
        first_stops = (stopped.cumsum(1) == 1) & stopped

        # time of stopping
        t_stop = first_stops.long().argmax(1)

        # all predictions
        predictions = logprobabilities.argmax(-1)

        # predictions at time of stopping
        predictions_at_t_stop = torch.masked_select(predictions, first_stops)

        return logprobabilities, deltas, predictions_at_t_stop, t_stop


class ClassificationHead(torch.nn.Module):

    def __init__(self, hidden_dims, nclasses):
        super(ClassificationHead, self).__init__()
        self.projection = nn.Sequential(
            nn.Linear(hidden_dims, nclasses, bias=True),
            nn.LogSoftmax(dim=2))

    def forward(self, x):
        return self.projection(x)


class DecisionHead(torch.nn.Module):

    def __init__(self, hidden_dims):
        super(DecisionHead, self).__init__()
        self.projection = nn.Sequential(
            nn.Linear(hidden_dims, 1, bias=True),
            nn.Sigmoid()
        )

        # initialize bias to predict late in first epochs
        torch.nn.init.normal_(self.projection[0].bias, mean=-2e1, std=1e-1)


    def forward(self, x):
        return self.projection(x).squeeze(2)


class TransformerBackbone(nn.Module):
    def __init__(
        self,
        input_dim=4,
        num_classes=25,
        d_model=182,
        n_head=2,
        n_layers=5,
        d_inner=128,
        activation="relu",
        dropout=0.017998950510888446,
    ):

        super(TransformerBackbone, self).__init__()
        self.modelname = (
            f"TransformerBackbone_input-dim={input_dim}_num-classes={num_classes}_"
            f"d-model={d_model}_d-inner={d_inner}_n-layers={n_layers}_n-head={n_head}_"
            f"dropout={dropout}"
        )

        encoder_layer = TransformerEncoderLayer(
            d_model, n_head, d_inner, dropout, activation
        )
        encoder_norm = LayerNorm(d_model)

        self.inlinear = Linear(input_dim, d_model)
        self.relu = ReLU()
        self.transformerencoder = TransformerEncoder(
            encoder_layer, n_layers, encoder_norm
        )

    def forward(self, x):
        x = self.inlinear(x)
        x = self.relu(x)
        x = x.transpose(0, 1)  # N x T x D -> T x N x D
        x = self.transformerencoder(x)
        x = x.transpose(0, 1)  # T x N x D -> N x T x D

        return x


if __name__ == "__main__":
    model = EarlyRNN()
