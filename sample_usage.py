from bh2vec.vectors_net import VectorsNetNorm, predict_tricks
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
