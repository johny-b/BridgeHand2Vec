import torch

from bh2vec.training import train
from bh2vec.vectors_net import VectorsNetNorm

device = 'cuda' if torch.cuda.is_available() else 'cpu'
net = VectorsNetNorm(n_hid=8).to(device)

train(net, 'data/vectorsnet_train_small.csv', 'data/vectorsnet_val_small.csv', epochs=200)
torch.save(net.state_dict(), "new_model.pth")
