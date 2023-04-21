import numpy as np
import pandas as pd
import torch
from torch.utils.tensorboard import SummaryWriter

torch.manual_seed(0)
torch.use_deterministic_algorithms(True)

import random
random.seed(0)

np.random.seed(0)

from . import bridge_helpers

def train(net, train_fname, val_fname, epochs):
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(net.parameters(), 5e-3, eps=1e-4, weight_decay=0)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.999)
    writer = SummaryWriter('runs/vectors')

    df = pd.read_csv(train_fname, sep=';', index_col=0)
    X, Y = inputs_to_numpy(df)
    X_tensor = torch.tensor(X, dtype=torch.float32)
    Y_tensor = torch.tensor(Y, dtype=torch.float32)

    df_val = pd.read_csv(val_fname, sep=';', index_col=0)
    X_val, Y_val = inputs_to_numpy(df_val)
    X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
    Y_val_tensor = torch.tensor(Y_val, dtype=torch.float32)

    train = torch.utils.data.TensorDataset(X_tensor, Y_tensor)

    def seed_worker(worker_id):
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    g = torch.Generator()
    g.manual_seed(0)

    train_loader = torch.utils.data.DataLoader(train, batch_size=10000, shuffle=True,
                                               worker_init_fn=seed_worker,
                                               generator=g,)

    n = 0
    for i in range(epochs):
        for batch_x, batch_y in train_loader:
            out = net(batch_x)
            loss = criterion(out, batch_y)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            out_val = net(X_val_tensor)
            mse_val = ((out_val - Y_val_tensor) ** 2).mean()
            writer.add_scalars('mse', {'train': loss.item(), 'val': mse_val}, n)

            mae = (out - batch_y).abs().mean()
            mae_val = (out_val - Y_val_tensor).abs().mean()
            writer.add_scalars('mae', {'train': mae, 'val': mae_val}, n)

            n += 1
        scheduler.step()

    return mse_val

def inputs_to_numpy(data, swap=True):
    rowsX = []
    rowsY = []

    for _, row in data.iterrows():
        binaryN = bridge_helpers.pbn_to_binary(row['N']).reshape(1, -1)
        binaryS = bridge_helpers.pbn_to_binary(row['S']).reshape(1, -1)
        rowsX.append(np.concatenate((binaryN, binaryS), axis=1))
        if swap:
            rowsX.append(np.concatenate((binaryS, binaryN), axis=1))
        resN = ['N' + t for t in bridge_helpers.TRUMPS]
        resS = ['S' + t for t in bridge_helpers.TRUMPS]
        rowsY.append(np.array(row[resN]).reshape(1, -1))
        if swap:
            rowsY.append(np.array(row[resS]).reshape(1, -1))

    X = np.concatenate(rowsX, axis=0)
    Y = np.concatenate(rowsY, axis=0).astype('float32')
    return X, Y
