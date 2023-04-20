import torch
from bh2vec.training import train
from bh2vec.vectors_net import VectorsNetNorm

torch.manual_seed(0)
torch.use_deterministic_algorithms(True)

net = VectorsNetNorm(n_hid=8)
train(net, 'data/vectorsnet_train_small.csv', 'data/vectorsnet_val_small.csv', epochs=1)
torch.save(net.state_dict(), "model1.pth")
