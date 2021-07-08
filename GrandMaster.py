"""
Name: GrandMaster.py
Aim: To predict the winner of a chess match using Graph Neural Networks
Author: C Siddarth
Date: July, 2021
"""

# Install dependies (to run in Google Colab)
# !pip install -q torch-scatter -f https://pytorch-geometric.com/whl/torch-1.9.0+cu102.html
# !pip install -q torch-sparse -f https://pytorch-geometric.com/whl/torch-1.9.0+cu102.html
# !pip install -q torch-geometric

# Import dependies
import torch
from torch_geometric.data import Data
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch.nn import Linear
import numpy as np
import glob
import pandas as pd
# Used for visualiztion of the graph
# import networkx as nx
# import matplotlib.pyplot as plt
# from torch_geometric.utils.convert import to_networkx


# Section 1 - Loading and Preprocessing

# Path to the CSV files
df1 = pd.read_csv('training_data.csv')
df2 = pd.read_csv('cross_validation_dataset.csv')
df3 = pd.read_csv('test_data.csv')

df4 = pd.concat([df1, df2, df3])

unique_id = list(set(list(df4['White Player #'].unique()) + list(df4['Black Player #'].unique())))

unique_id_dict = {}

for i in range(len(unique_id)):
    unique_id_dict[unique_id[i]] = i

df4['White Player #'] = df4['White Player #'].map(unique_id_dict)
df4['Black Player #'] = df4['Black Player #'].map(unique_id_dict)

unique_id = list(set(list(df4['White Player #'].unique()) + list(df4['Black Player #'].unique())))
number_of_nodes = len(unique_id) # Total number of unique players/chess IDs


# Section 2 - Graph Construction

x = torch.zeros((number_of_nodes, 7)).float()

# [Win as White %, Loss as White %, Draw as White %, Win as Black %, Loss as Black %, Draw as Black %, Total Matches Played]
for player_ind in range(number_of_nodes): # All players

    i = 0
    num_total_matches = len(df4.loc[(df4['White Player #'] == player_ind) | (df4['Black Player #'] == player_ind)])
    
    if num_total_matches == 0:
        x[player_ind] = 0
        continue
    
    for player_color in ['White Player #', 'Black Player #']: # Two possible colours 
        for game_result in [1, 0, 0.5]: # Win, Lose, Draw

            num_specific_matches = len(df4.loc[(df4[player_color] == player_ind) & (df4.Score == game_result)])
            
            multiplier = 1
            if i == 1 or i == 4: # Features related to 'lossing' are multipled by -1
                multiplier = -1
    
            x[player_ind][i] = multiplier * num_specific_matches/num_total_matches

            i = i + 1

    x[player_ind][i] =  num_total_matches

results_dict = {0:0, 0.5:1, 1:2}

w_results = torch.FloatTensor(list(df4['Score'].map(results_dict)))
b_results = 3 - w_results

edge_result = torch.cat((w_results, b_results), dim = -1).type(torch.int64)

w2b_edge = torch.FloatTensor([list(df4['White Player #']), list(df4['Black Player #'])]).type(torch.int64)
b2w_edge = torch.FloatTensor([list(df4['Black Player #']), list(df4['White Player #'])]).type(torch.int64)

edge_index = torch.cat((w2b_edge, b2w_edge), dim = -1).type(torch.int64)

graph = Data(x = x, edge_index = edge_index).cuda()

train_valid_mask = torch.logical_or(torch.logical_or(w_results == 0, w_results == 1), w_results == 2)
test_mask = torch.logical_not(train_valid_mask)

total_pre_test = sum(train_valid_mask)
total_test = sum(test_mask)
total = len(train_valid_mask)

train_mask = torch.tensor([True for i in range(int(.85 * total_pre_test))] + [False for i in range(total - int(.85 * total_pre_test))])

valid_mask = torch.tensor([False for i in range(int(.85 * total_pre_test))] + [True for i in range(total_pre_test - int(.85 * total_pre_test))] + [False for i in range(total - total_pre_test)])

test_mask = torch.tensor([False for i in range(total_pre_test)] + [True for i in range(total_test)])

# # To plot the graph
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

        self.conv1 = GCNConv(inp, h1)
        self.conv2 = GCNConv(h1, h2)
        self.conv3 = GCNConv(h2, h3)

        self.lin1 = Linear(2 * h3, h4)
        self.lin2 = Linear(h4, 3)

    def forward(self, x, edge_index, w2b):

      R = torch.nn.ReLU()

      x = R(self.conv1(x, edge_index))
      x = R(self.conv2(x, edge_index))
      x = R(self.conv3(x, edge_index))   

      concat = torch.zeros((len(w2b[0]), 2 * len(x[0]))).cuda()
      index = 0

      for source, target in zip(w2b[0], w2b[1]):

          concat[index] = torch.cat((x[source], x[target]))
          index = index + 1

      hidden = self.lin1(concat) 
      edge_pred = self.lin2(hidden) 

      return edge_pred

edge_y = w_results.long()

model = GCN(inp = 7).cuda()

criterion = torch.nn.CrossEntropyLoss()  # Define loss criterion.
optimizer = torch.optim.Adam(model.parameters(), lr = 0.01)  # Define optimizer.


for epoch in range(201):

    model.train()
    
    optimizer.zero_grad()  # Clear gradients.
    out = model(graph.x, edge_index.cuda(), torch.stack((w2b_edge[0][train_mask].cuda(), w2b_edge[1][train_mask].cuda())))  # Perform a single forward pass.
    tloss = criterion(out,  edge_y[train_mask].cuda())  # Compute the loss solely based on the training nodes.
    
    tloss.backward()  # Derive gradients.
    optimizer.step()  # Update parameters based on gradients.

    pred = torch.tensor([torch.argmax(o).item() for o in out])
    true = edge_y[train_mask]

    tacc = (sum(pred == true) / len(pred)).item()

    model.eval()

    out = model(graph.x, edge_index.cuda(), torch.stack((w2b_edge[0][valid_mask].cuda(), w2b_edge[1][valid_mask].cuda())))
    vloss = criterion(out,  edge_y[valid_mask].cuda())  # Compute the loss solely based on the validation nodes.
    
    pred = torch.tensor([torch.argmax(o).item() for o in out])
    true = edge_y[valid_mask]

    vacc = (sum(pred == true) / len(pred)).item()


    # # Uncomment to display result
    # if epoch % 5 == 0:
        
    #     print(f'Epoch: {epoch + 1:03d}, Train Loss: {tloss.item():.4f}, Train Acc: {tacc:.4f}, Valid Loss: {vloss.item():.4f},  Valid Acc: {vacc:.4f}')
    #     print('-' * 5)


# Testing on validation data due to unavailabilty of submit option
mse = torch.nn.MSELoss()

model.eval()
out = model(graph.x, edge_index.cuda(), torch.stack((w2b_edge[0][valid_mask].cuda(), w2b_edge[1][valid_mask].cuda())))
pred = torch.tensor([torch.argmax(o).item() for o in out])
vloss = mse(pred,  w_results[valid_mask])  # Compute the loss solely based on the validation nodes