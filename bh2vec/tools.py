import torch
import numpy as np

from . import bridge_helpers

def hand_to_vec(net, hand_pbn):
    """Calculate the embedding for a given hand."""
    hand_binary = bridge_helpers.pbn_to_binary(hand_pbn).reshape(1, -1)
    hand_tensor = torch.tensor(hand_binary, dtype=torch.float32)
    hand_vec = net.get_vector(hand_tensor).detach().numpy()
    return hand_vec

def predict_tricks(net, hand_n_pbn, hand_s_pbn):
    """Predict the number of tricks for two given hands.

    First hand (n) is assumed to be the dealer.

    Returns a five-element tuple with number of tricks for
    (clubs, diamonds, hearts, spades, nontrump)."""

    binaryN = bridge_helpers.pbn_to_binary(hand_n_pbn).reshape(1, -1)
    binaryS = bridge_helpers.pbn_to_binary(hand_s_pbn).reshape(1, -1)
    X = np.concatenate((binaryN, binaryS), axis=1)
    X_tensor = torch.tensor(X, dtype=torch.float32)
    out = net(X_tensor).detach().numpy()
    return out
