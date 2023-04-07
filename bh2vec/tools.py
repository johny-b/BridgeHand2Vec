import torch
import numpy as np

from . import bridge_helpers as bh

def predict_tricks(net, hand_n_pbn, hand_s_pbn):
    """Predict the number of tricks for two given hands.

    First hand (n) is assumed to be the dealer.

    Returns a five-element tuple with number of tricks for
    (clubs, diamonds, hearts, spades, nontrump)."""

    binaryN = bh.pbn_to_binary(hand_n_pbn).reshape(1, -1)
    binaryS = bh.pbn_to_binary(hand_s_pbn).reshape(1, -1)
    X = np.concatenate((binaryN, binaryS), axis=1)
    X_tensor = torch.tensor(X, dtype=torch.float32)
    out = net(X_tensor).detach().numpy()
    return out[0]

def hand_to_vec(net, hand_pbn):
    """Calculate the embedding for a given hand."""
    hand_binary = bh.pbn_to_binary(hand_pbn).reshape(1, -1)
    hand_tensor = torch.tensor(hand_binary, dtype=torch.float32)
    hand_vec = net.get_vector(hand_tensor).detach().numpy()
    return hand_vec[0]

def vec_to_hand(net, target_vec, *, num_attempts=1):
    best_hands = {}
    
    for i in range(num_attempts):
        current_hand = bh.get_random_hand()
        while True:
            current_hand_vec = hand_to_vec(net, current_hand)
            dist = np.linalg.norm(target_vec - current_hand_vec)
            new_hand = _single_step_closer(net, current_hand, target_vec, dist)
            if new_hand is None:
                break
            else:
                current_hand = new_hand
        
        best_hands[current_hand] = dist

    best_hand = max(best_hands, key=lambda x: best_hands[x])
    return best_hand

def _single_step_closer(net, hand, target_vec, dist):
    hand_bin = bh.pbn_to_binary(hand)
    card_ixs = np.where(hand_bin)
    non_card_ixs = np.where(np.invert(hand_bin))

    best_new_hand = None
    best_new_dist = None
    for card_ix in card_ixs[0]:
        for non_card_ix in non_card_ixs[0]:
            new_hand_bin = hand_bin.copy()
            new_hand_bin[card_ix] = False
            new_hand_bin[non_card_ix] = True
            new_hand_pbn = bh.binary_to_pbn(new_hand_bin)
            new_hand_vec = hand_to_vec(net, new_hand_pbn)
            new_dist = np.linalg.norm(new_hand_vec - target_vec)
            if new_dist < dist:
                if best_new_dist is None or new_dist < best_new_dist:
                    best_new_dist = new_dist
                    best_new_hand = new_hand_pbn

    return best_new_hand
