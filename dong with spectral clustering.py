# -*- coding: utf-8 -*-
"""
Created on Mon Apr  5 00:50:31 2021

@author: guanyunfeng
"""


from sklearn.cluster import SpectralClustering as SC
import numpy as np
import matplotlib.pyplot as plt

import torch
from torch import nn

import timeit

X = np.load('data_algo.npy') #given data

#X = np.random.rand(1000, 10)

np.random.shuffle(X)

n, dim = X.shape # dataset size, dimensionality of given data
n_clusters = 8 # number of bins
n_neighbors = 10 # k
epochs = 80 # number of training epochs

sc = SC(
    n_clusters=n_clusters,
    affinity='nearest_neighbors',
    n_neighbors=n_neighbors,
    )
labels = sc.fit_predict(X)

# visualization of partitioning on given dataset
# plt.scatter(X[:, 0], X[:, 1], c=labels)
# u, indices = np.unique(labels, return_index=True)
# plt.bar(u, indices)

class MLP(nn.Module):
    def __init__(self, dim, n_class):
        super(MLP, self).__init__()
        self.twolayer = nn.Sequential(
            nn.Linear(dim, 20*dim),
            nn.ReLU(),
            nn.Linear(20*dim, 20*dim),
            nn.ReLU(),
            nn.Linear(20*dim, 5*dim),
            nn.ReLU(),
            nn.Linear(5*dim, n_class),
            nn.ReLU()
        )
    
    def forward(self, x):
        return self.twolayer(x)

model = MLP(dim, n_clusters)

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(
    model.parameters(), lr=1e-2)

def train(X, labels, model, loss_fn, optimizer):
    size, dim = X.shape
    for i in range(size):
        x, y = torch.unsqueeze(X[i], 0), torch.unsqueeze(labels[i], 0)
        pred = model(x)
        loss = loss_fn(pred, y)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # if i % 100 == 0:
        #     print("sample {}".format(i))
    
def test(X, labels, model):
    match = 0
    for i in range(n):
        match += (1 if model(X[i]).argmax()==labels[i] else 0)
    print("Accuracy: {}".format(match/n))

t_X = torch.tensor(X, dtype=torch.float)
t_labels = torch.tensor(labels, dtype=torch.long)

for t in range(epochs):
    print("Epoch {}".format(t))
    train(t_X, t_labels, model, loss_fn, optimizer)
    test(t_X, t_labels, model)

partition = {}
for i in range(n_clusters):
    partition[i] = []
for i in range(n):
    partition[int(model(t_X[i]).argmax())].append(i)

# %%

# Choose random data points 1% to be queries.
number_of_queries = int(0.01*n)
print('Generating queries')
np.random.seed(4057218)
queries = t_X[len(t_X) - number_of_queries:]
print('Done')
    
test_match = 0
for query in queries:
    neighborhood = partition[int(model(query).argmax())]
    test_label = np.linalg.norm(query - t_X[neighborhood], axis = 1).argsort()[1]
    test_neighbor = neighborhood[test_label]
    
    global_neighbor = np.linalg.norm(query - t_X, axis = 1).argsort()[1]
    test_match += (1 if test_neighbor == global_neighbor else 0)
    # print(int(model(t_X[test_neighbor]).argmax()), int(model(t_X[global_neighbor]).argmax()))
print('Recall=', test_match/number_of_queries)
    


