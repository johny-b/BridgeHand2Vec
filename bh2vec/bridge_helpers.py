import itertools
import numpy as np

SUITS = ['C', 'D', 'H', 'S']
TRUMPS = SUITS + ['NT']
CARDS = ['2', '3', '4', '5', '6', '7', '8', '9', 'T', 'J', 'Q', 'K', 'A']
DECK = [c + s for s, c in itertools.product(SUITS, CARDS)]
CARD_TO_IDX = {c: i for i, c in enumerate(DECK)}

BIDS = ['PASS'] + [str(h) + t for h, t in itertools.product(range(1, 8), TRUMPS)]


def pbn_to_binary(hand):
    i = 0
    binary = np.zeros(52, dtype=np.bool8)
    for s in SUITS[::-1]:
        while i < 16 and hand[i] != '.':
            card = hand[i] + s
            binary[CARD_TO_IDX[card]] = 1
            i += 1
        i += 1
    return binary


def binary_to_hand(binary):
    return list(np.array(DECK)[binary])


def _select_suit(hand, suit):
    return ''.join(sorted([c[0] for c in hand if c[1] == suit],
                   key=lambda c: CARDS.index(c), reverse=True))


def deal_to_pbn(hands):
    pbn = 'N:' + hand_to_pbn(hands[0]) + ' ' + hand_to_pbn(hands[1]) + ' '
    pbn += hand_to_pbn(hands[2]) + ' ' + hand_to_pbn(hands[3])
    return pbn


def hand_to_pbn(hand):
    c = _select_suit(hand, 'C')
    d = _select_suit(hand, 'D')
    h = _select_suit(hand, 'H')
    s = _select_suit(hand, 'S')

    return s + '.' + h + '.' + d + '.' + c


def hand_to_binary(hand):
    return pbn_to_binary(hand_to_pbn(hand))


def binary_to_pbn(binary):
    return hand_to_pbn(binary_to_hand(binary))

def get_random_hand():
    deck = DECK.copy()
    np.random.shuffle(deck)
    return hand_to_pbn(deck[:13])
