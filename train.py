from bh2vec.vectors_net import train
import torch

net, _ = train('data/vectorsnet_train.csv', 'data/vectorsnet_val.csv', epochs=200)
torch.save(net.state_dict(), "new_model.pth")
