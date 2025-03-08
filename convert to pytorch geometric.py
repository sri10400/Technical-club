import torch
import pandas as pd
from torch_geometric.data import Data
import numpy as np

# Load prepared MovieLens data
file_path = r'C:\Users\amaln\Downloads\ml-1m\ml-1m\ratings.dat'

columns = ['user_id', 'movie_id', 'rating', 'timestamp']
data = pd.read_csv(file_path, sep="::", names=columns, engine='python', usecols=['user_id', 'movie_id', 'rating'])

# Convert user and movie IDs to zero-based index
data['user_id'] -= 1
data['movie_id'] -= 1

# Load trained embeddings
user_embeddings = torch.load("user_embeddings.pt")  # Shape: (num_users, embedding_dim)
movie_embeddings = torch.load("movie_embeddings.pt")  # Shape: (num_movies, embedding_dim)

# Combine embeddings into a single node feature matrix
node_features = torch.cat([user_embeddings, movie_embeddings], dim=0)  # Shape: (num_nodes, embedding_dim)

# Create edge index (user-movie interactions)
edge_index = torch.tensor(np.array([data['user_id'].values, data['movie_id'].values + len(user_embeddings)]), dtype=torch.long)

# Create edge attributes (ratings as weights)
edge_attr = torch.tensor(data['rating'].values, dtype=torch.float)

# Create PyTorch Geometric Data object
graph_data = Data(x=node_features, edge_index=edge_index, edge_attr=edge_attr)

print(graph_data)  # Verify the output


