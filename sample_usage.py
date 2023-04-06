from bh2vec.vectors_net import VectorsNetNorm
from bh2vec.tools import hand_to_vec, predict_tricks, vec_to_hand
import torch

PATH = 'model.pth'

#   Load the model
net = VectorsNetNorm()
net.load_state_dict(torch.load(PATH))
net.eval()

#   Predict tricks for (clubs, diamonds, heards, spades, nt), dealer is hand_n
hand_n = 'AQ72.KT632.AJ3.9'
hand_s = 'K943.A.Q9876.KQ2'
tricks = predict_tricks(net, hand_n, hand_s).tolist()
print(f"Expected number of tricks for hands {hand_n} and {hand_s}:")
print("\n".join(list(f" {suit}: {num_tricks}" for suit, num_tricks in (zip('cdhsn', tricks)))))
print()

# #   Calculate the embedding for the given hand
print(f"Hand {hand_n} embedding: {hand_to_vec(net, hand_n)}")
print(f"Hand {hand_s} embedding: {hand_to_vec(net, hand_s)}")
print()

#   Find hand that is close to a given embedding.
#   NOTE: this is a random process that will yield different results in
#   separate runs.
embedding = [1, 1, 1, 1, 0, 0, 0, 0]
closest_hand = vec_to_hand(net, embedding)
print(f"Hand closest to {embedding}: {closest_hand}")
print()

#   Find the "opposite" hand to hand_n.
hand_n_embedding = hand_to_vec(net, hand_n)
opposite_embedding = hand_n_embedding * -1
opposite_hand = vec_to_hand(net, opposite_embedding)
print(f"Hand opposite to {hand_n}: {opposite_hand}")
print()

#   Find hands with similar features as hand_n, but more/less intensive
stronger_embedding = hand_n_embedding * 1.5
weaker_embedding = hand_n_embedding * 0.5
stronger_hand = vec_to_hand(net, stronger_embedding)
weaker_hand = vec_to_hand(net, weaker_embedding)
print(f"Hand {hand_n} is between {stronger_hand} and {weaker_hand}")
print()
