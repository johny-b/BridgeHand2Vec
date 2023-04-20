import torch

from bh2vec.vectors_net import VectorsNetNorm
from bh2vec.tools import hand_to_vec

net1 = VectorsNetNorm()
net1.load_state_dict(torch.load("model1.pth"))
net1.eval()

hand = 'KQT875.KJ5.9.AQJ'
hand_vec1 = hand_to_vec(net1, hand)
print(hand_vec1)

