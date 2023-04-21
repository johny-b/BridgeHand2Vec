import torch
import pandas as pd
import numpy as np
from bh2vec.training import inputs_to_numpy

def process_files(file):
    df = pd.read_csv(file, sep=';', index_col=0)
    X, Y = inputs_to_numpy(df)
    return X, Y

def eval_max_errors(net, X, Y):
    X_tensor = torch.tensor(X, dtype=torch.float32)
    Y_tensor = torch.tensor(Y, dtype=torch.float32)

    out = net(X_tensor)

    mae = (out - Y_tensor).abs()
    max05 = (mae <= 0.5).type(torch.float32).mean(dim=0)
    max1 = (mae <= 1).type(torch.float32).mean(dim=0)
    max2 = (mae <= 2).type(torch.float32).mean(dim=0)
    return max05, max1, max2


def eval_max_round_errors(net, X, Y):
    X_tensor = torch.tensor(X, dtype=torch.float32)

    out = net(X_tensor).detach().numpy()
    out_round = np.around(out)
    Y_round = np.around(Y)

    mae_round = np.abs((out_round - Y_round))
    max0 = (mae_round == 0).mean(axis=0)
    max1 = (mae_round <= 1).mean(axis=0)
    max2 = (mae_round <= 2).mean(axis=0)

    return max0, max1, max2