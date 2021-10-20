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

n = 1000
dim = 10
n_clusters = 8
X = np.random.rand(n, dim) #given data

sc = SC(
    n_clusters=n_clusters,
    affinity='nearest_neighbors',
    n_neighbors=10,
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
            nn.Linear(dim, 10*dim),
            nn.ReLU(),
            nn.Linear(10*dim, 5*dim),
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

epochs = 40
for t in range(epochs):
    print("Epoch {}".format(t))
    train(t_X, t_labels, model, loss_fn, optimizer)
    test(t_X, t_labels, model)
        
        
    
