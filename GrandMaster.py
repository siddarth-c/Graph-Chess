"""
Name: GrandMaster.py
Aim: To predict the winner of a chess match using Graph Neural Networks
Author: C Siddarth
Date: June, 2021
"""

# Install dependies (to run in Google Colab)
# !pip install -q torch-scatter -f https://pytorch-geometric.com/whl/torch-1.9.0+cu102.html
# !pip install -q torch-sparse -f https://pytorch-geometric.com/whl/torch-1.9.0+cu102.html
# !pip install -q torch-geometric

# Import dependies
import torch
from torch_geometric.nn import GCNConv # Feel free to use any Graph Layer
from torch.nn import Linear
from torch_geometric.data import Data
import torch.nn.functional as F

import numpy as np

# Used for visualiztion of the graph
# import networkx as nx
# import matplotlib.pyplot as plt
# from torch_geometric.utils.convert import to_networkx

import pandas as pd

# Section 1 - Loading and Preprocessing

path = 'out.txt' # Path to data

file1 = open(path, 'r')
lines = file1.readlines()
file1.close()

white = []
black = []
result = []

lines = lines[1:]

for line in lines:
    split1 = line.strip().split(' ')
    white.append(int(split1[0]) - 1) # White player's index
    black.append(int(split1[1]) - 1) # Black player's index
    result.append(int(split1[-1].split('\t')[0])) # Game's result

data = {'White' : white,
        'Black' : black,
        'Result' : result}

df = pd.DataFrame(data)
number_of_nodes = 7301 # Total number of unique players/chess IDs

x = torch.zeros((number_of_nodes, 7)).float()

# [Win as White %, Loss as White %, Draw as White %, Win as Black %, Loss as Black %, Draw as Black %]
for player_ind in range(number_of_nodes): # All players

    i = 0
    num_total_matches = len(df.loc[(df['White'] == player_ind) | (df['Black'] == player_ind)])
    
    for player_color in ['White', 'Black']: # Two possible colours 
        for game_result in [1, -1, 0]: # Win, Lose, Draw

            num_specific_matches = len(df.loc[(df[player_color] == player_ind) & (df.Result == game_result)])
            
            multiplier = 1
            if i == 1 or i == 4: # Features related to 'lossing' are multipled by -1
                multiplier = -1
            
            x[player_ind][i] = multiplier * num_specific_matches/num_total_matches

            i = i + 1

    x[player_ind][i] =  num_total_matches


# Section 2 - Graph Construction

di = {-1:-1, 0:0.5, 1:1}
edge_weigth = df.replace({'Result': di})['Result']
y = torch.FloatTensor(df.Result)

w2b_edge = torch.FloatTensor([df.White, df.Black]).type(torch.int64)
b2w_edge = torch.FloatTensor([df.Black, df.White]).type(torch.int64)

edge_index = torch.cat((w2b_edge, b2w_edge), dim = -1).type(torch.int64)

graph = Data(x = x, edge_index = edge_index)
graph = graph.cuda()

total_edges = len(w2b_edge[0])

train_mask = torch.tensor([True for i in range(int(.80 * total_edges))] + [False for i in range(total_edges - int(.80 * total_edges))])

valid_mask = torch.logical_not(train_mask)

# Uncomment the following to see the detailed statistics of the graph

# print('Total edges present:', total_edges)
# print('Total edges covered:', sum(train_mask | valid_mask | test_mask).item())
# print('Average node degree:', graph.num_edges / graph.num_nodes)
# print()
# print('Total number of matches:', len(y))
# print('Percentage of white wins:', (100 * sum(y == 1)/len(y)).item(), '%')
# print('Percentage of black wins:', (100 * sum(y == -1)/len(y)).item(), '%')
# print('Percentage of draws:', (100 * sum(y == 0)/len(y)).item(), '%')


# Uncomment the following to visualize the constructed graph

# color = ['r', 'g', 'b']
# edge_color = [color[int(i)+1] for i in y]
# chess_graph = to_networkx(graph)
# plt.figure(1,figsize=(50,50)) 
# nx.draw(chess_graph, cmap=plt.get_cmap('Set1'), edge_color = edge_color, node_size = 30, linewidths = 6)
# plt.show()

# Section 3 - Model definition and training

class GCN(torch.nn.Module):

    def __init__(self, inp):

        super(GCN, self).__init__()
      
        h1 = 8
        h2 = 16
        h3 = 32
        h4 = 4

        # Edge weights are also a parameter which will be optimized along with the weights (more on this in repo)
        self.edge_weight = torch.nn.Parameter(torch.ones(graph.num_edges))

        # 3 Graph Layers
        self.conv1 = GCNConv(inp, h1)
        self.conv2 = GCNConv(h1, h2)
        self.conv3 = GCNConv(h2, h3)

        # 2 Hidden Layers
        self.lin1 = Linear(2 * h3, h4)
        self.lin2 = Linear(h4, 3)

    def forward(self, x, edge_index, w2b):

      R = torch.nn.ReLU()

      x = R(self.conv1(x, edge_index, edge_weight = self.edge_weight.sigmoid()))
      x = R(self.conv2(x, edge_index, edge_weight = self.edge_weight.sigmoid()))
      x = R(self.conv3(x, edge_index, edge_weight = self.edge_weight.sigmoid()))

      concat = torch.zeros((len(w2b[0]), 2 * len(x[0]))).cuda()
      index = 0

      for source, target in zip(w2b[0], w2b[1]):

          concat[index] = torch.cat((x[source], x[target]))
          index = index + 1

      hidden = self.lin1(concat) 
      edge_pred = self.lin2(hidden) 

      return edge_pred


di = {1:2, 0:1, -1:0}
edge_y = torch.LongTensor(df.replace({'Result': di})['Result'])


model = GCN(inp = 7).cuda() # Model
criterion = torch.nn.CrossEntropyLoss()  # Loss criterion.
optimizer = torch.optim.Adam(model.parameters(), lr = 0.01)  # Optimizer.


for epoch in range(201):

    # Training
    model.train()
    
    optimizer.zero_grad()  # Clear gradients.
    out = model(graph.x, edge_index.cuda(), torch.stack((w2b_edge[0][train_mask].cuda(), w2b_edge[1][train_mask].cuda())))  # Perform a single forward pass.
    tloss = criterion(out,  edge_y[train_mask].cuda())  # Compute the loss solely based on the training edges.
    
    tloss.backward()  # Derive gradients.
    optimizer.step()  # Update parameters based on gradients.

    pred = torch.tensor([torch.argmax(o).item() for o in out])
    true = edge_y[train_mask]

    tacc = (sum(pred == true) / len(pred)).item()

    # Validation
    model.eval()

    out = model(graph.x, edge_index.cuda(), torch.stack((w2b_edge[0][valid_mask].cuda(), w2b_edge[1][valid_mask].cuda())))
    vloss = criterion(out,  edge_y[valid_mask].cuda())  
    pred = torch.tensor([torch.argmax(o).item() for o in out])
    true = edge_y[valid_mask]
    vacc = (sum(pred == true) / len(pred)).item()

    if epoch % 1 == 0:
      print(f'Epoch: {epoch + 1:03d}, Train Loss: {tloss.item():.4f}, Train Acc: {tacc:.4f}, Valid Loss: {vloss.item():.4f},  Valid Acc: {vacc:.4f}')

    print('-' * 5)
