import torch

from bh2vec.training import train
from bh2vec.vectors_net import VectorsNetNorm

net = VectorsNetNorm(n_hid=8)
train(net, 'data/vectorsnet_train.csv', 'data/vectorsnet_val.csv', epochs=200)
torch.save(net.state_dict(), "model2.pth")
