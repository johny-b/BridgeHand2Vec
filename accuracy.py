"""Calculate the accuracy of a model"""
import torch
from bh2vec.vectors_net import VectorsNetNorm
from evaluation import eval_max_errors, eval_max_round_errors, process_files

PATH = 'model.pth'

net = VectorsNetNorm()
net.load_state_dict(torch.load(PATH))
net.eval()

X, Y = process_files('data/vectorsnet_test.csv')

max05, max1, max2 = eval_max_errors(net, X, Y)
print('MAE erros---------------------------------------')
print('error <= 0.5: ', max05, ', avg suit: ', max05[:4].mean(), ', avg: ', max05.mean())
print('error <= 1: ', max1, ', avg suit: ', max1[:4].mean(), 'avg: ', max1.mean())
print('error <= 2: ', max2, ', avg suit: ', max2[:4].mean(), 'avg: ', max2.mean())
print('-----------------------------------------------')
print('rounded errors---------------------------------')
max0, max1, max2 = eval_max_round_errors(net, X, Y)
print('error == 0: ', max0, ', avg suit: ', max0[:4].mean(), ', avg: ', max0.mean())
print('error <= 1: ', max1, ', avg suit: ', max1[:4].mean(), 'avg: ', max1.mean())
print('error <= 2: ', max2, ', avg suit: ', max2[:4].mean(), 'avg: ', max2.mean())
print('-----------------------------------------------')
