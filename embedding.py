import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import zipfile
import os

# Load and extract dataset
zip_path = r'C:\Users\amaln\Downloads\ml-1m.zip'
extract_path = r'C:\Users\amaln\Downloads\ml-1m'
if not os.path.exists(extract_path):
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_path)

# Load MovieLens ratings
file_path = os.path.join(extract_path, 'ml-1m', 'ratings.dat')
columns = ['user_id', 'movie_id', 'rating', 'timestamp']
data = pd.read_csv(file_path, sep="::", names=columns, engine='python', usecols=['user_id', 'movie_id', 'rating'])

# Convert to zero-based indexing
data['user_id'] -= 1
data['movie_id'] -= 1

# Get number of unique users and movies
num_users = data['user_id'].nunique()
num_movies = data['movie_id'].nunique()

print("Number of unique users:", num_users)
print("Number of unique movies:", num_movies)
print("User IDs range:", data['user_id'].min(), "-", data['user_id'].max())
print("Movie IDs range:", data['movie_id'].min(), "-", data['movie_id'].max())

# Define embedding model
class EmbeddingModel(nn.Module):
    def __init__(self, num_users, num_movies, embedding_dim):
        super(EmbeddingModel, self).__init__()
        self.user_embedding = nn.Embedding(num_users + 1, embedding_dim)  # +1 for safety
        self.movie_embedding = nn.Embedding(num_movies + 1, embedding_dim)  # +1 for safety
        self.fc = nn.Linear(embedding_dim * 2, 1)

    def forward(self, user_ids, movie_ids):
        user_ids = user_ids.clamp(0, num_users)  # Ensure valid range
        movie_ids = movie_ids.clamp(0, num_movies)  # Ensure valid range

        user_embeds = self.user_embedding(user_ids)
        movie_embeds = self.movie_embedding(movie_ids)
        x = torch.cat([user_embeds, movie_embeds], dim=1)
        return self.fc(x).squeeze(1)

# Initialize model
embedding_dim = 64
model = EmbeddingModel(num_users, num_movies, embedding_dim)


# Loss and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Convert data to tensors
user_ids = torch.tensor(data['user_id'].values, dtype=torch.long)
movie_ids = torch.tensor(data['movie_id'].values, dtype=torch.long)
ratings = torch.tensor(data['rating'].values, dtype=torch.float)

# Training loop
epochs = 10
for epoch in range(epochs):
    optimizer.zero_grad()
    predictions = model(user_ids, movie_ids)
    loss = criterion(predictions, ratings)
    loss.backward()
    optimizer.step()
    
    print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item()}")

print("Embedding training completed! 🎉")

# Save embeddings
torch.save(model.user_embedding.weight, "user_embeddings.pt")
torch.save(model.movie_embedding.weight, "movie_embeddings.pt")


