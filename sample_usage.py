from bh2vec.vectors_net import VectorsNetNorm
from bh2vec.tools import hand_to_vec, predict_tricks  #, vec_to_hand
import torch

PATH = 'model.pth'

#   Load the model
net = VectorsNetNorm()
net.load_state_dict(torch.load(PATH))
net.eval()

#   Predict tricks for (clubs, diamonds, heards, spades, nt), dealer is hand_n
hand_n = 'A432.A432.432.32'
hand_s = 'KQJ5.KQJ5.76.765'
print(predict_tricks(net, hand_n, hand_s))

#   Calculate the embedding for the given hand
print(hand_to_vec(net, hand_n))
print(hand_to_vec(net, hand_s))

#   Find nearest hand to a given embedding
# embedding = [1, 1, 1, 1, 0, 0, 0, 0]
# print(vec_to_hand(net, embedding))
