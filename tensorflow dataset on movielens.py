#dataset of movielens from chrome.
import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt
import zipfile
import os

# Path to the ZIP file
zip_path = r'C:\Users\amaln\Downloads\ml-1m.zip'  # Use raw string (r''), or double backslashes

# Extract the ZIP file
extract_path = r'C:\Users\amaln\Downloads\ml-1m'
if not os.path.exists(extract_path):  # Extract only if not already extracted
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_path)

# Path to the ratings file
file_path = os.path.join(extract_path, 'ml-1m', 'ratings.dat')

# Read data (separator "::" is used in older MovieLens datasets)
columns = ['user_id', 'item_id', 'rating', 'timestamp']
data = pd.read_csv(file_path, sep="::", names=columns, engine='python', usecols=['user_id', 'item_id', 'rating'])

# Create a graph
G = nx.Graph()

# Add edges (User-Movie interactions)
for _, row in data.iterrows():
    G.add_edge(f"user_{row['user_id']}", f"movie_{row['item_id']}", weight=row['rating'])

# Visualize a small subset of the graph for clarity
plt.figure(figsize=(10, 10))
small_G = nx.subgraph(G, list(G.nodes)[:20])  # Show only 20 nodes for readability
nx.draw(small_G, with_labels=True, node_color='lightblue', edge_color='gray')
plt.show()
