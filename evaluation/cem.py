'''
cross entrophy method
inspired by: https://github.com/yandexdataschool/Practical_RL/blob/master/week01_intro/deep_crossentropy_method.ipynb
'''

import torch
import torch.nn as nn
from bh2vec import bridge_helpers
from evaluation import scorer
import numpy as np
from torch.utils.tensorboard import SummaryWriter

N_ACTIONS = len(bridge_helpers.BIDS)


class Boards():
    def __init__(self, df):
        self.df = df
        self.next = 0
        self.size = len(self.df)

    def get_hand(self, index, player='N'):
        self.next = index % len(self.df) + 1
        hand_pbn = self.df.iloc[index % self.size][player]
        return bridge_helpers.pbn_to_binary(hand_pbn)

    def get_reward(self, contract, index, player='N'):
        row = self.df.iloc[index % self.size]
        minimax = row['best_score_' + player]

        if contract == 'PASS':
            contract_score = 0
        else:
            tricks = [value for key, value in row[2:].to_dict().items() if key[1] == player and key[2:] == contract[1:]]
            score = [scorer.calculate_score(contract, t) for t in tricks]
            contract_score = sum(score)/len(score)

        return contract_score - minimax, contract_score

    def __len__(self):
        return self.size


class OneStepPolicyAgent(nn.Module):
    def __init__(self, inp_size=52, hid1_size=128, hid2_size=128, hid3_size=128):

        super().__init__()
        self.linear1 = nn.Linear(inp_size, hid1_size)
        self.linear2 = nn.Linear(hid1_size, hid2_size)
        self.linear3 = nn.Linear(hid2_size, hid3_size)
        self.relu = nn.ReLU()
        self.out = nn.Linear(hid3_size, N_ACTIONS)

    def forward(self, state):
        x = self.relu(self.linear1(state))
        x = self.relu(self.linear2(x))
        x = self.relu(self.linear3(x))
        x = self.out(x)
        return x


def predict_prob(state, agent, emb_net=None):
    state = torch.tensor(state, dtype=torch.float32)

    with torch.no_grad():
        if emb_net is not None:
            state = state.unsqueeze(0)
            state = emb_net.get_vector(state)
            state = torch.flatten(state)
        logits = agent(state)
        sm = nn.Softmax(dim=0)
        probs = sm(logits).numpy()

    return probs


def generate_sessions(agent, boards, n=100, player='N', next=True, deterministic=False, emb_net=None):

    states = []
    actions = []
    rewards = []
    raw_scores = []
    if next:
        offset = boards.next
    else:
        offset = 0
    for i in range(n):
        s = boards.get_hand(i + offset, player=player)
        probs = predict_prob(s, agent, emb_net=emb_net)
        if deterministic:
            a = np.argmax(probs)
        else:
            a = np.random.choice(N_ACTIONS, p=probs)
        r, raw_score = boards.get_reward(bridge_helpers.BIDS[a], i + offset, player=player)

        states.append(s)
        actions.append(a)
        rewards.append(r)
        raw_scores.append(raw_score)

    return states, actions, rewards, raw_scores


def select_elites(states, actions, rewards, percentile=50):
    thr = np.percentile(np.array(rewards), percentile)
    idx = np.array(rewards) > thr

    elite_states = np.array(states)[idx]
    elite_actions = np.array(actions)[idx]

    return elite_states, elite_actions


def train(agent, boards, boards_val=None, n_sessions=10, n_steps=10, percentile=70, emb_net=None, log_dir='runs/cem'):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(agent.parameters(), 1e-3, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
    writer = SummaryWriter(log_dir)
    player = 'N'
    for i in range(1, n_steps + 1):
        states, actions, rewards, raw_scores = generate_sessions(agent, boards, n=n_sessions, player=player,
                                                                 emb_net=emb_net)
        mean_reward = sum(rewards)/len(rewards)
        mean_raw_score = sum(raw_scores)/len(raw_scores)
        elite_states, elite_actions = select_elites(states, actions, rewards, percentile=70)
        writer.add_scalar('reward', mean_reward, i)
        writer.add_scalar('raw score', mean_raw_score, i)

        states = torch.tensor(elite_states, dtype=torch.float32)
        actions = torch.tensor(elite_actions, dtype=torch.long)

        optimizer.zero_grad()

        if emb_net is not None:
            states = emb_net.get_vector(states)
        logits = agent(states)

        probs = nn.functional.softmax(logits, -1)
        log_probs = nn.functional.log_softmax(logits, -1)
        entropy = - torch.sum(probs * log_probs)
        loss_cross = criterion(logits, actions)
        loss = loss_cross - 0.01 * entropy
        loss.backward()
        optimizer.step()

        if i % 100 == 0:
            scheduler.step()

        writer.add_scalar('entropy', entropy.item(), i)
        writer.add_scalar('CrossEntropyLoss', loss_cross.item(), i)

        if boards_val is not None and i % 100 == 0:
            states, actions, rewards, raw_scores = generate_sessions(agent, boards_val, n=10, player=player,
                                                                     next=False, deterministic=True, emb_net=emb_net)
            text = ''
            for j in range(len(states)):
                text += bridge_helpers.binary_to_pbn(states[j]) + '   '
                text += bridge_helpers.BIDS[actions[j]] + '   '
                text += 'minimax diff: ' + str(rewards[j])
                text += ' score: ' + str(raw_scores[j]) + '  \n'
            writer.add_text('bids', text, i)

        if (i * n_sessions) % len(boards) == 0:
            # boards.i = 0
            if player == 'N':
                player = 'S'
            else:
                player = 'N'
