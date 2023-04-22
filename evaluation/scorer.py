TRICKS_FIRST = {'C': 20, 'D': 20, 'H': 30, 'S': 30, 'NT': 40}
TRICKS_NEXT = {'C': 20, 'D': 20, 'H': 30, 'S': 30, 'NT': 30}
BONUSES = {'contract': 50, 'game': 300, 'slam': 500, 'grand_slam': 1000}
UNDER_TRICKS = 50


def calculate_score(contract, n_tricks):
    '''
    contract: 4S, 3NT
    '''
    contract_tricks = int(contract[0])

    if contract_tricks + 6 > n_tricks:
        # contract not made
        return -UNDER_TRICKS * (contract_tricks + 6 - n_tricks)

    trump = contract[1:]
    tricks_score = TRICKS_FIRST[trump] + (contract_tricks - 1) * TRICKS_NEXT[trump]

    score = tricks_score
    # overtricks
    score += (n_tricks - contract_tricks - 6) * TRICKS_NEXT[trump]

    if tricks_score >= 100:
        score += BONUSES['game']
    else:
        score += BONUSES['contract']

    if contract_tricks == 6:
        score += BONUSES['slam']

    if contract_tricks == 7:
        score += BONUSES['grand_slam']

    return score


# print(calculate_score('7NT', 13))
