# Copyright (c) 2025 [Samarvir Singh Vasale/samarvir1]
# Licensed under the MIT License.
# See LICENSE file in the project root for full license information.

import torch
import torch.nn as nn
from torch_geometric.nn import GINConv, global_mean_pool
from torch_geometric.nn import Sequential as PyGSequential 
from torch_geometric.data import Data, Batch 
from torch.utils.data import Dataset, DataLoader

import networkx as nx
import numpy as np
import os
import random
from tqdm import tqdm
import time

# --- CONFIGURATION ---
NUM_GRAPHS = 2000
MIN_NODES = 10
MAX_NODES = 20
EDGE_PROB = 0.3
OUTPUT_DIR = "graph_data"

# Model Hyperparameters
NUM_NODE_FEATURES = 1
HIDDEN_CHANNELS = 64
EMBEDDING_DIM = 32
LEARNING_RATE = 0.001
EPOCHS = 50
BATCH_SIZE = 32
# ---------------------


# A. DATA GENERATION FUNCTIONS

def to_pyg_data(G):
    """Converts a networkx graph to a PyTorch Geometric Data object."""
    num_nodes = G.number_of_nodes()
    
    degrees = np.array([G.degree(i) for i in range(num_nodes)])
    x = torch.tensor(degrees, dtype=torch.float).unsqueeze(1)
    
    edge_index = torch.tensor(list(G.edges)).t().contiguous()
    row, col = edge_index
    edge_index = torch.cat([edge_index, torch.stack([col, row], dim=0)], dim=1)
    
    return Data(x=x, edge_index=edge_index, num_nodes=num_nodes)

def create_isomorphic_pair(num_nodes, p):
    G1 = nx.fast_gnp_random_graph(num_nodes, p, seed=random.randint(0, 10000))
    while G1.number_of_edges() == 0 and num_nodes > 1:
        G1 = nx.fast_gnp_random_graph(num_nodes, p, seed=random.randint(0, 10000))

    mapping = {i: v for i, v in enumerate(np.random.permutation(num_nodes))}
    G2 = nx.relabel_nodes(G1, mapping, copy=True)
    
    return to_pyg_data(G1), to_pyg_data(G2), torch.tensor([1.0], dtype=torch.float)

def create_non_isomorphic_pair(num_nodes_range, p):
    while True:
        N = random.randint(*num_nodes_range)
        
        G1 = nx.fast_gnp_random_graph(N, p, seed=random.randint(0, 10000))
        G2 = nx.fast_gnp_random_graph(N, p, seed=random.randint(0, 10000))
        
        if G1.number_of_edges() != G2.number_of_edges() or G1.number_of_nodes() != G2.number_of_nodes():
             continue
        
        if not nx.is_isomorphic(G1, G2):
            return to_pyg_data(G1), to_pyg_data(G2), torch.tensor([0.0], dtype=torch.float)

def generate_dataset(num_pairs):
    print(f"Generating {num_pairs} graph pairs...")
    data_list = []
    
    for i in tqdm(range(num_pairs)):
        num_nodes = random.randint(MIN_NODES, MAX_NODES)
        
        if i % 2 == 0:
            g1, g2, label = create_isomorphic_pair(num_nodes, EDGE_PROB)
        else:
            g1, g2, label = create_non_isomorphic_pair((MIN_NODES, MAX_NODES), EDGE_PROB)
            
        data_list.append((g1, g2, label))
        
    random.shuffle(data_list)
    
    if not os.path.exists(OUTPUT_DIR): os.makedirs(OUTPUT_DIR)
    output_path = os.path.join(OUTPUT_DIR, f"isomorphism_data_{num_pairs}.pt")
    torch.save(data_list, output_path)
    print(f"\n✅ Data generation complete. Saved to: {output_path}")
    return data_list


# B. MODEL ARCHITECTURE

class GNNEncoder(nn.Module):
    """GIN module to create a permutation-invariant graph embedding."""
    def __init__(self, num_node_features, hidden_channels, embedding_dim):
        super().__init__()
        
        self.conv1 = GINConv(self._build_mlp(num_node_features, hidden_channels))
        self.conv2 = GINConv(self._build_mlp(hidden_channels, hidden_channels))
        self.conv3 = GINConv(self._build_mlp(hidden_channels, hidden_channels))
        
        self.final_lin = nn.Linear(hidden_channels, embedding_dim)

    def _build_mlp(self, in_channels, out_channels):
        return PyGSequential('x', [
            (nn.Linear(in_channels, out_channels), 'x -> x'),
            (nn.BatchNorm1d(out_channels), 'x -> x'),
            nn.ReLU(inplace=True)
        ])
        
    def forward(self, x, edge_index, batch):
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index).relu()
        x = self.conv3(x, edge_index).relu()
        
        x = global_mean_pool(x, batch) 
        
        return self.final_lin(x)

class SiameseGIN(nn.Module):
    """Siamese network for graph comparison."""
    def __init__(self, num_node_features, hidden_channels, embedding_dim):
        super().__init__()
        
        self.encoder = GNNEncoder(num_node_features, hidden_channels, embedding_dim)
        
        self.classifier = nn.Sequential(
            nn.Linear(embedding_dim, hidden_channels // 2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_channels // 2, 1),
            nn.Sigmoid()
        )

    def forward_one(self, data):
        return self.encoder(data.x, data.edge_index, data.batch)

    def forward(self, data_g1, data_g2):
        v1 = self.forward_one(data_g1)
        v2 = self.forward_one(data_g2)
        
        diff = torch.abs(v1 - v2)
        
        prediction = self.classifier(diff)
        
        return prediction


# C. TRAINING SETUP

class SiameseGraphDataset(Dataset):
    """Custom dataset for (G1, G2, label) tuples."""
    def __init__(self, data_list):
        self.data_list = data_list
    
    def __len__(self):
        return len(self.data_list)
    
    def __getitem__(self, idx):
        return self.data_list[idx]

def collate_fn(batch):
    """Custom collate function to batch graph pairs."""
    g1_list, g2_list, y_list = zip(*batch)
    
    g1_batch = Batch.from_data_list(g1_list)
    g2_batch = Batch.from_data_list(g2_list)
    
    y_batch = torch.stack(y_list).to(g1_batch.x.device)
    
    return g1_batch, g2_batch, y_batch


def train_model(model, train_loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    correct_predictions = 0
    
    for data_g1, data_g2, labels in tqdm(train_loader, desc="Training"):
        data_g1, data_g2, labels = data_g1.to(device), data_g2.to(device), labels.to(device)
        
        optimizer.zero_grad()
        output = model(data_g1, data_g2)
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        preds = torch.round(output)
        correct_predictions += (preds == labels).sum().item()

    avg_loss = total_loss / len(train_loader)
    accuracy = correct_predictions / (len(train_loader.dataset))
    return avg_loss, accuracy


def main():
    data_path = os.path.join(OUTPUT_DIR, f"isomorphism_data_{NUM_GRAPHS}.pt")
    if os.path.exists(data_path):
        print(f"Loading existing data from {data_path}...")
        data_list = torch.load(data_path)
    else:
        data_list = generate_dataset(NUM_GRAPHS)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    train_size = int(0.8 * len(data_list))
    train_data = data_list[:train_size]
    
    train_dataset = SiameseGraphDataset(train_data)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)

    model = SiameseGIN(NUM_NODE_FEATURES, HIDDEN_CHANNELS, EMBEDDING_DIM).to(device)
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    print(f"\nModel initialized. Starting training for {EPOCHS} epochs...")

    for epoch in range(1, EPOCHS + 1):
        train_loss, train_acc = train_model(model, train_loader, optimizer, criterion, device)
        
        print(f"Epoch {epoch:02d} | Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")

    model_save_path = f'siamese_gin_iso_detector_{int(time.time())}.pt'
    torch.save(model.state_dict(), model_save_path)
    print(f"\n✅ Training complete. Model saved to: {model_save_path}")


if __name__ == '__main__':

    main()
