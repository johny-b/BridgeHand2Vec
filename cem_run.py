import torch
import pandas as pd
from evaluation import cem
from bh2vec.vectors_net import VectorsNetNorm


# with vectors
EMB_PATH = 'models/model.pth'
emb_net = VectorsNetNorm()
emb_net.load_state_dict(torch.load(EMB_PATH))
emb_net.eval()
print('-------------------------------')
df = pd.read_csv('evaluation/data/cem/cem_train.csv', sep=';', index_col=0)
df_val = pd.read_csv('evaluation/data/cem/cem_val.csv', sep=';', index_col=0)
boards = cem.Boards(df)
boards_val = cem.Boards(df_val)
agent_vec = cem.OneStepPolicyAgent(inp_size=8)
cem.train(agent_vec, boards, boards_val, n_sessions=100, n_steps=1000, percentile=70, emb_net=emb_net,
          log_dir='runs/cem_vectors')
print('----------agent vec performance ona a training set---------------------')
boards = cem.Boards(df)
states, actions, rewards, raw_scores = cem.generate_sessions(agent_vec, boards, n=len(boards), deterministic=True,
                                                             emb_net=emb_net)
print('rewards: ', sum(rewards)/len(rewards))
print('raw scores: ', sum(raw_scores)/len(raw_scores))
print('----------agent vec performance ona a validation set--------------------')
boards = cem.Boards(df_val)
states, actions, rewards, raw_scores = cem.generate_sessions(agent_vec, boards, n=len(boards), deterministic=True,
                                                             emb_net=emb_net)
print('rewards: ', sum(rewards)/len(rewards))
print('raw scores: ', sum(raw_scores)/len(raw_scores))

# baseline without vectors
print('-------------------------------')
boards = cem.Boards(df)
boards_val = cem.Boards(df_val)
agent = cem.OneStepPolicyAgent()
cem.train(agent, boards, boards_val, n_sessions=100, n_steps=1000, percentile=70)
print('----------agent binary performance ona a training set---------------------')
boards = cem.Boards(df)
states, actions, rewards, raw_scores = cem.generate_sessions(agent, boards, n=len(boards), deterministic=True)
print('rewards: ', sum(rewards)/len(rewards))
print('raw scores: ', sum(raw_scores)/len(raw_scores))
print('----------agent binary performance ona a validation set---------------------')
boards = cem.Boards(df_val)
states, actions, rewards, raw_scores = cem.generate_sessions(agent, boards, n=len(boards), deterministic=True)
print('rewards: ', sum(rewards)/len(rewards))
print('raw scores: ', sum(raw_scores)/len(raw_scores))
