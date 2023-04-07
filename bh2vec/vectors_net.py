import torch
import torch.nn as nn

class VectorsNetNorm(nn.Module):
    def __init__(self, n_hid=8):
        super().__init__()

        # embedding
        self.emb1 = nn.Linear(52, 32)
        self.emb2 = nn.Linear(32, 32)
        self.emb3 = nn.Linear(32, n_hid)
        self.act = nn.ELU()
        self.batch_norm = nn.BatchNorm1d(n_hid, affine=False)

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
