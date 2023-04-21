import torch
import os
from bh2vec.vectors_net import VectorsNetNorm
from evaluation.eval import eval_max_errors, process_files
'''
Variance of models accuracy
'''

PATH = 'models'
models = ['vec5_batch_norm.pth', 'model.pth']

X, Y = process_files('data/vectorsnet_test.csv')

for m in models:
    net = VectorsNetNorm()
    net.load_state_dict(torch.load(os.path.join(PATH, m)))
    net.eval()

    max05, max1, max2 = eval_max_errors(net, X, Y)
    print('suit;', max05[:4].mean().item(), ';', max1[:4].mean().item(), ';', max2[:4].mean().item())
    print('nt;', max05[4].mean().item(), ';', max1[4].mean().item(), ';', max2[4].mean().item())