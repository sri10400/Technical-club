import networkx as nx
import pandas as pd
import numpy as np
import torch
from torch_geometric.data import Data
import zipfile
import os

# Path to the ZIP file
zip_path = r'C:\Users\amaln\Downloads\ml-1m.zip'

# Extract the ZIP file
extract_path = r'C:\Users\amaln\Downloads\ml-1m'
if not os.path.exists(extract_path):
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_path)

# Path to the ratings file
file_path = os.path.join(extract_path, 'ml-1m', 'ratings.dat')

# Read MovieLens data
columns = ['user_id', 'item_id', 'rating', 'timestamp']
data = pd.read_csv(file_path, sep="::", names=columns, engine='python', usecols=['user_id', 'item_id', 'rating'])

# Encode user and movie IDs as numerical indices
unique_users = data['user_id'].unique()
unique_movies = data['item_id'].unique()

user_mapping = {id_: i for i, id_ in enumerate(unique_users)}
movie_mapping = {id_: i + len(unique_users) for i, id_ in enumerate(unique_movies)}

# Convert to graph format
edges = []
edge_weights = []

for _, row in data.iterrows():
    user_idx = user_mapping[row['user_id']]
    movie_idx = movie_mapping[row['item_id']]
    
    edges.append([user_idx, movie_idx])
    edges.append([movie_idx, user_idx])  # Bidirectional edges
    edge_weights.append(row['rating'])
    edge_weights.append(row['rating'])

edges = torch.tensor(edges, dtype=torch.long).t()  # Shape: (2, num_edges)
edge_weights = torch.tensor(edge_weights, dtype=torch.float)

# Create node features (optional: dummy features)
num_nodes = len(unique_users) + len(unique_movies)
node_features = torch.eye(num_nodes)  # Identity matrix as features

# Create PyTorch Geometric Data object
graph_data = Data(x=node_features, edge_index=edges, edge_attr=edge_weights)

# Print Graph Info
print(graph_data)
