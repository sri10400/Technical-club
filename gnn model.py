import torch

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Define GNN Model
class GNNModel(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GNNModel, self).__init__()
        self.conv1 = torch.nn.Linear(input_dim, hidden_dim)
        self.conv2 = torch.nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = self.conv2(x)
        return x

# Initialize model BEFORE calling `.to(device)`
input_dim, hidden_dim, output_dim = 64, 128, 64
model = GNNModel(input_dim, hidden_dim, output_dim)

# Move model to GPU (if available)
model.to(device)
print("✅ Model moved to device successfully!")

'''
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)


# ======================= 1. Define GNN Model =======================

class GNNModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GNNModel, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, output_dim)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        return x

# Initialize the model
input_dim = 64
hidden_dim = 128
output_dim = 64
model = GNNModel(input_dim, hidden_dim, output_dim)
optimizer = optim.Adam(model.parameters(), lr=0.01)
loss_fn = nn.MSELoss()
print("✅ GNN Model Defined!")

# ======================= 2. Load and Process Dataset =======================

file_path = r"C:\Users\amaln\Downloads\ml-1m\ml-1m\ratings.dat"
columns = ['user_id', 'movie_id', 'rating']
data = pd.read_csv(file_path, sep="::", names=columns, engine='python', usecols=['user_id', 'movie_id', 'rating'])

data['user_id'] -= 1  # Convert user IDs to zero-based index
data['movie_id'] -= 1  # Convert movie IDs to zero-based index

# ======================= 3. Load and Align Embeddings =======================
user_embeddings = torch.load("user_embeddings.pt")
movie_embeddings = torch.load("movie_embeddings.pt")

expected_num_movies = 3952
if movie_embeddings.shape[0] < expected_num_movies:
    padding = torch.zeros((expected_num_movies - movie_embeddings.shape[0], 64))
    movie_embeddings = torch.cat([movie_embeddings, padding], dim=0)
    torch.save(movie_embeddings, "fixed_movie_embeddings.pt")
    print("✅ Movie embeddings padded and saved!")

# Combine user and movie embeddings
x = torch.cat([user_embeddings, movie_embeddings], dim=0)

# ======================= 4. Define Graph Structure =======================
num_users = user_embeddings.shape[0]
num_movies = expected_num_movies

data = data[data["movie_id"] < num_movies]  # Ensure valid movie IDs
edge_index_np = np.array([
    data['user_id'].values, 
    data['movie_id'].values + num_users
])
edge_index = torch.tensor(edge_index_np, dtype=torch.long)

print(f"x shape: {x.shape}")
print(f"edge_index shape: {edge_index.shape}")

# ======================= 5. Debugging Missing Movie IDs =======================
actual_movie_ids = set(data["movie_id"].unique())
expected_movie_ids = set(range(num_movies))
missing_movies = expected_movie_ids - actual_movie_ids
extra_movies = actual_movie_ids - expected_movie_ids
print(f"Missing Movie IDs: {missing_movies}")
print(f"Extra Movie IDs: {extra_movies}")

# Ensure all movies have embeddings
aligned_movie_embeddings = np.zeros((num_movies, movie_embeddings.shape[1]))
for i in range(num_movies):
    if i in actual_movie_ids:
        aligned_movie_embeddings[i] = movie_embeddings[i].detach().cpu().numpy()
movie_embeddings = torch.tensor(aligned_movie_embeddings, dtype=torch.float32)

print("New movie_embeddings shape:", movie_embeddings.shape)

# ======================= 6. Model Training =======================
epochs = 10
for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    user_movie_embeddings = model(x, edge_index)
    loss = loss_fn(user_movie_embeddings, x)  # Dummy loss for now
    loss.backward()
    optimizer.step()
    print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item()}")
print("🎉 Training Completed!")

# ======================= 7. Evaluation Metrics =======================
predicted_scores = torch.matmul(user_embeddings, movie_embeddings.T).detach().numpy()
ratings = pd.read_csv(file_path, sep="::", engine="python", names=["user_id", "movie_id", "rating", "timestamp"])
ground_truth = ratings[ratings["rating"] >= 4].groupby("user_id")["movie_id"].apply(list).to_dict()

def recall_at_k(predictions, ground_truth, k=10):
    hit_count = 0
    total_users = len(predictions)
    for user_id, true_movies in ground_truth.items():
        top_k_movies = np.argsort(predictions[user_id])[-k:]
        hit_count += len(set(top_k_movies) & set(true_movies))
    return hit_count / total_users

def ndcg_at_k(predictions, ground_truth, k=10):
    ndcg_total = 0
    total_users = len(predictions)
    for user_id, true_movies in ground_truth.items():
        top_k_movies = np.argsort(predictions[user_id])[-k:]
        dcg = sum([1 / np.log2(i + 2) if movie in true_movies else 0 for i, movie in enumerate(top_k_movies)])
        idcg = sum([1 / np.log2(i + 2) for i in range(min(len(true_movies), k))])
        ndcg_total += dcg / idcg if idcg > 0 else 0
    return ndcg_total / total_users

recall_k = recall_at_k(predicted_scores, ground_truth, k=10)
ndcg_k = ndcg_at_k(predicted_scores, ground_truth, k=10)
print(f"✅ Recall@10: {recall_k:.4f}")
print(f"✅ NDCG@10: {ndcg_k:.4f}")

# ======================= 8. Graph Visualization =======================

torch.save(edge_index, "edge_index.pth")
edge_list = edge_index.numpy().T.tolist()
G = nx.Graph()
G.add_edges_from(edge_list)
plt.figure(figsize=(10, 7))
nx.draw(G, pos=nx.spring_layout(G, seed=42), node_size=10, edge_color="gray", alpha=0.7,
        node_color=["blue" if node < num_users else "red" for node in G.nodes])
plt.title("Graph with Users (Blue) and Movies (Red)")
plt.show()

print(f"Number of nodes: {G.number_of_nodes()}")
print(f"Number of edges: {G.number_of_edges()}")
'''
