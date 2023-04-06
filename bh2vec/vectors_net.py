import torch
import torch.nn as nn
from . import bridge_helpers
import numpy as np
import pandas as pd
from torch.utils.tensorboard import SummaryWriter


class VectorsNet(nn.Module):
    def __init__(self, n_hid=8):

        super().__init__()
        # embedding
        self.emb1 = nn.Linear(52, 32)
        self.emb2 = nn.Linear(32, 32)
        self.emb3 = nn.Linear(32, n_hid)
        self.act = nn.ELU()

        # prediction
        self.hid1 = nn.Linear(2 * n_hid, 128)
        self.hid2 = nn.Linear(128, 128)
        self.out = nn.Linear(128, 5)

    def forward(self, x):
        x = x.view(-1, 104)
        x1 = x[:, :52]
        x2 = x[:, 52:]
        x1 = self.get_vector(x1)
        x2 = self.get_vector(x2)
        x = self.get_prediction_from_vectors(x1, x2)

        return x

    def get_prediction_from_vectors(self, x1, x2):
        x = torch.cat((x1, x2), 1)
        x = self.act(self.hid1(x))
        x = self.act(self.hid2(x))
        x = self.out(x)

        return x

    def get_vector(self, hand):
        x = self.act(self.emb1(hand))
        x = self.act(self.emb2(x))
        x = self.emb3(x)
        return x


class VectorsNetNorm(nn.Module):
    def __init__(self, n_hid=8):

        super().__init__()
        # embedding
        self.emb1 = nn.Linear(52, 32)
        self.emb2 = nn.Linear(32, 32)
        self.emb3 = nn.Linear(32, n_hid)
        self.act = nn.ELU()
        self.batch_norm = torch.nn.BatchNorm1d(n_hid, affine=False)

        # prediction
        self.hid1 = nn.Linear(2 * n_hid, 128)
        self.hid2 = nn.Linear(128, 128)
        self.out = nn.Linear(128, 5)

    def forward(self, x):
        x = x.view(-1, 104)
        x1 = x[:, :52]
        x2 = x[:, 52:]
        x1 = self.get_vector(x1)
        x2 = self.get_vector(x2)
        x = self.get_prediction_from_vectors(x1, x2)

        return x

    def get_prediction_from_vectors(self, x1, x2):
        x = torch.cat((x1, x2), 1)
        x = self.act(self.hid1(x))
        x = self.act(self.hid2(x))
        x = self.out(x)

        return x

    def get_vector(self, hand):
        x = self.act(self.emb1(hand))
        x = self.act(self.emb2(x))
        x = self.emb3(x)
        x = self.batch_norm(x)
        return x

class VectorsNet2(nn.Module):
    def __init__(self, n_hid=8):

        super().__init__()
        # embedding
        self.emb1 = nn.Linear(52, 32)
        self.emb2 = nn.Linear(32, 32)
        self.emb3 = nn.Linear(32, 32)
        self.emb4 = nn.Linear(32, n_hid)
        self.act = nn.ELU()

        # prediction
        self.hid1 = nn.Linear(2 * n_hid, 128)
        self.out = nn.Linear(128, 5)

    def forward(self, x):
        x = x.view(-1, 104)
        x1 = x[:, :52]
        x2 = x[:, 52:]
        x1 = self.get_vector(x1)
        x2 = self.get_vector(x2)

        x = torch.cat((x1, x2), 1)
        x = self.act(self.hid1(x))
        x = self.out(x)

        return x

    def get_vector(self, hand):
        x = self.act(self.emb1(hand))
        x = self.act(self.emb2(x))
        x = self.act(self.emb3(x))
        x = self.emb4(x)
        return x



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


def eval_max_errors(net, file):
    df = pd.read_csv(file, sep=';', index_col=0)
    X, Y = inputs_to_numpy(df)
    X_tensor = torch.tensor(X, dtype=torch.float32)
    Y_tensor = torch.tensor(Y, dtype=torch.float32)

    out = net(X_tensor)

    mae = (out - Y_tensor).abs()
    max05 = (mae <= 0.5).type(torch.float32).mean(dim=0)
    max1 = (mae <= 1).type(torch.float32).mean(dim=0)
    max2 = (mae <= 2).type(torch.float32).mean(dim=0)
    return max05, max1, max2


def eval_max_round_errors(net, file):
    df = pd.read_csv(file, sep=';', index_col=0)
    X, Y = inputs_to_numpy(df)
    X_tensor = torch.tensor(X, dtype=torch.float32)

    out = net(X_tensor).detach().numpy()
    out_round = np.around(out)
    Y_round = np.around(Y)

    mae_round = np.abs((out_round - Y_round))
    max0 = (mae_round == 0).mean(axis=0)
    max1 = (mae_round <= 1).mean(axis=0)
    max2 = (mae_round <= 2).mean(axis=0)

    return max0, max1, max2


def load_vectors(net, file):
    df = pd.read_csv(file, sep=';', index_col=0)
    X, Y = inputs_to_numpy(df)
    X = X[:, :52]
    X_tensor = torch.tensor(X, dtype=torch.float32)

    return X, net.get_vector(X_tensor).detach().numpy()


def load_vectors_for_hands(net, df, hand_col='hand'):
    rowsX = []
    for _, row in df.iterrows():
        binary = bridge_helpers.pbn_to_binary(row[hand_col]).reshape(1, -1)
        rowsX.append(binary)
    X = np.concatenate(rowsX, axis=0)
    X_tensor = torch.tensor(X, dtype=torch.float32)
    return net.get_vector(X_tensor).detach().numpy()


def euclidean_dist(x1, x2):
    return np.sqrt(((x1 - x2) ** 2).mean())


def find_nearest_hands_from_vec(hand_vec, net, vectors, n=10):
    hands, vec = vectors
    dist = np.apply_along_axis(lambda x: euclidean_dist(x, hand_vec), axis=1, arr=vec)
    indx = np.argsort(dist)[:n]

    return hands[indx], dist[indx]


def find_nearest_hands(hand, net, vectors, n=10):
    hand_binary = bridge_helpers.pbn_to_binary(hand).reshape(1, -1)
    hand_tensor = torch.tensor(hand_binary, dtype=torch.float32)
    hand_vec = net.get_vector(hand_tensor).detach().numpy()

    return find_nearest_hands_from_vec(hand_vec, net, vectors, n)


def binary_hand_to_vec(hand_binary, net):
    hand_tensor = torch.tensor(hand_binary, dtype=torch.float32)
    hand_vec = net.get_vector(hand_tensor).detach().numpy()
    return hand_vec


def hand_to_vec(hand, net):
    hand_binary = bridge_helpers.pbn_to_binary(hand).reshape(1, -1)
    hand_tensor = torch.tensor(hand_binary, dtype=torch.float32)
    hand_vec = net.get_vector(hand_tensor).detach().numpy()
    return hand_vec

def predict_tricks(net, handN, handS):
    binaryN = bridge_helpers.pbn_to_binary(handN).reshape(1, -1)
    binaryS = bridge_helpers.pbn_to_binary(handS).reshape(1, -1)
    X = np.concatenate((binaryN, binaryS), axis=1)
    X_tensor = torch.tensor(X, dtype=torch.float32)
    out = net(X_tensor).detach().numpy()
    return out


def train(file, val_file, n_hid=8, epochs=100):
    net = VectorsNetNorm(n_hid)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(net.parameters(), 5e-3, weight_decay=0)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.999)
    writer = SummaryWriter('runs/vectors')

    df = pd.read_csv(file, sep=';', index_col=0)
    X, Y = inputs_to_numpy(df)
    X_tensor = torch.tensor(X, dtype=torch.float32)
    Y_tensor = torch.tensor(Y, dtype=torch.float32)

    df_val = pd.read_csv(val_file, sep=';', index_col=0)
    X_val, Y_val = inputs_to_numpy(df_val)
    X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
    Y_val_tensor = torch.tensor(Y_val, dtype=torch.float32)

    train = torch.utils.data.TensorDataset(X_tensor, Y_tensor)
    train_loader = torch.utils.data.DataLoader(train, batch_size=10000, shuffle=True)

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

            # mae
            mae = (out - batch_y).abs().mean()
            mae_val = (out_val - Y_val_tensor).abs().mean()
            writer.add_scalars('mae', {'train': mae, 'val': mae_val}, n)

            n += 1
        scheduler.step()

    return net, mse_val
