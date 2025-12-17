import torch
import torch.nn as nn
from torch_geometric.data import Batch
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
import os
import glob
import random
from tqdm import tqdm

from siamese_isomorphism_solver import (
    SiameseGIN, GNNEncoder, NUM_NODE_FEATURES, HIDDEN_CHANNELS, 
    EMBEDDING_DIM, to_pyg_data, collate_fn, SiameseGraphDataset
) 


# --- CONFIGURATION ---
DATA_PATH = os.path.join("graph_data", "isomorphism_data_2000.pt")
NUM_EMBEDDINGS_TO_PLOT = 500
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_model(path, device):
    """Loads the trained model state."""
    model = SiameseGIN(NUM_NODE_FEATURES, HIDDEN_CHANNELS, EMBEDDING_DIM).to(device)
    model.load_state_dict(torch.load(path, map_location=device, weights_only=True)) 
    model.eval()
    print(f"Model successfully loaded from {path}")
    return model

def generate_wl_hard_pair():
    """Generates a non-isomorphic pair indistinguishable by the 1-WL (GIN) test."""
    
    # Graph 1: The Prism Graph (C4 x K2)
    edges_g1 = [(1, 2), (2, 3), (3, 4), (4, 1), (5, 6), (6, 7), (7, 8), (8, 5), 
                (1, 5), (2, 6), (3, 7), (4, 8)]
    G1 = nx.Graph()
    G1.add_edges_from(edges_g1)

    # Graph 2: A known WL-equivalent graph
    edges_g2 = [(1, 2), (2, 3), (3, 4), (4, 1), (1, 3), (2, 4), (5, 6), (6, 7), 
                (7, 8), (8, 5), (5, 7), (6, 8)]
    G2 = nx.Graph()
    G2.add_edges_from(edges_g2)

    # CRITICAL FIX: Relabel nodes from 1-N to 0-(N-1)
    num_nodes = G1.number_of_nodes()
    mapping = {i: i-1 for i in range(1, num_nodes + 1)} 
    
    G1 = nx.relabel_nodes(G1, mapping, copy=True)
    G2 = nx.relabel_nodes(G2, mapping, copy=True)

    if not nx.is_isomorphic(G1, G2):
        print("WL-hard graphs generated successfully (and are non-isomorphic).")
        return to_pyg_data(G1), to_pyg_data(G2)
    else:
        raise ValueError("Error in generating WL-hard pair: graphs unexpectedly isomorphic.")

def get_embeddings(model, data_loader, device):
    """Generates and collects the graph embeddings."""
    embeddings = []
    labels = []

    with torch.no_grad():
        for data_g1, data_g2, y in tqdm(data_loader, desc="Generating Embeddings"):
            data_g1, data_g2 = data_g1.to(device), data_g2.to(device)
            
            v1 = model.forward_one(data_g1)
            v2 = model.forward_one(data_g2)

            embeddings.extend(v1.cpu().numpy())
            embeddings.extend(v2.cpu().numpy())
            
            labels.extend(y.cpu().squeeze().numpy())
            labels.extend(y.cpu().squeeze().numpy())
            
    return np.array(embeddings), np.array(labels)

def evaluate_metrics(model, data_list, device):
    """Calculates Accuracy, Precision, Recall, and F1-score on a validation set."""
    model.eval()
    
    train_size = int(0.8 * len(data_list))
    val_data = data_list[train_size:]
    
    val_dataset = SiameseGraphDataset(val_data)
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=64, shuffle=False, collate_fn=collate_fn)
    
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for data_g1, data_g2, labels in tqdm(val_loader, desc="Evaluating Metrics"):
            data_g1, data_g2, labels = data_g1.to(device), data_g2.to(device), labels.to(device)
            
            output = model(data_g1, data_g2)
            preds = torch.round(output).cpu().numpy()
            
            all_preds.extend(preds.flatten())
            all_labels.extend(labels.cpu().numpy().flatten())

    accuracy = accuracy_score(all_labels, all_preds)
    precision, recall, f1, support = precision_recall_fscore_support(
        all_labels, all_preds, average=None, labels=[0, 1]
    )

    print("\n--- Quantitative Metrics (Validation Set) ---")
    print(f"Overall Accuracy: {accuracy:.4f}")
    print(f"{'Metric':<10} | {'Non-Isomorphic (0)':<20} | {'Isomorphic (1)':<20}")
    print(f"{'-'*10} | {'-'*20} | {'-'*20}")
    print(f"{'Precision':<10} | {precision[0]:<20.4f} | {precision[1]:<20.4f}")
    print(f"{'Recall':<10} | {recall[0]:<20.4f} | {recall[1]:<20.4f}")
    print(f"{'F1-Score':<10} | {f1[0]:<20.4f} | {f1[1]:<20.4f}")


def plot_embeddings_tsne(embeddings, labels, title="t-SNE Visualization of Graph Embeddings"):
    """Performs t-SNE with a randomly generated seed and plots the resulting 2D embeddings."""
    
    # Generate a random seed for the t-SNE algorithm
    random_seed = random.randint(1, 100000)
    print(f"\nStarting t-SNE dimensionality reduction with random_state={random_seed}...")
    
    tsne = TSNE(n_components=2, random_state=random_seed, perplexity=30)
    embeddings_2d = tsne.fit_transform(embeddings)

    plt.figure(figsize=(10, 8))
    
    iso_idx = np.where(labels == 1.0)
    plt.scatter(embeddings_2d[iso_idx, 0], embeddings_2d[iso_idx, 1], 
                c='blue', label='Isomorphic Pair', alpha=0.6, marker='o')

    non_iso_idx = np.where(labels == 0.0)
    plt.scatter(embeddings_2d[non_iso_idx, 0], embeddings_2d[non_iso_idx, 1], 
                c='red', label='Non-Isomorphic Pair', alpha=0.6, marker='x')

    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()
    print("t-SNE plot displayed.")


def run_analysis():
    # --- 1. Model Path Detection ---
    model_files = glob.glob('siamese_gin_iso_detector_*.pt')
    
    if not model_files:
        print("Error: No trained model file found matching 'siamese_gin_iso_detector_*.pt'. Please run training first.")
        return
    
    # Select the model file with the most recent creation time
    model_path = max(model_files, key=os.path.getctime)
    
    # --- 2. Load Model ---
    model = load_model(model_path, DEVICE)
    
    # --- 3. Test on WL-Hard Pairs ---
    g1_wl_data, g2_wl_data = generate_wl_hard_pair()
    
    wl_batch_g1 = Batch.from_data_list([g1_wl_data]).to(DEVICE)
    wl_batch_g2 = Batch.from_data_list([g2_wl_data]).to(DEVICE)
    
    wl_prediction = model(wl_batch_g1, wl_batch_g2).item()
    
    print("\n--- WL-Hard Pair Test ---")
    print("These graphs are NON-ISOMORPHIC, but GIN is known to struggle with them.")
    print(f"Model's Prediction P(Isomorphic): {wl_prediction:.4f}")
    
    if wl_prediction > 0.9:
        print("✅ GNN confirms the expected WL failure: Predicts HIGH probability of isomorphism.")
    else:
        print("⚠️ GNN appears to distinguish them, suggesting it learned a subtle feature.")

    # --- 4. Quantitative Metrics ---
    if not os.path.exists(DATA_PATH):
        print(f"\nError: Data file not found at {DATA_PATH}. Cannot calculate metrics.")
        return 
        
    full_data = torch.load(DATA_PATH)
    evaluate_metrics(model, full_data, DEVICE)

    # --- 5. t-SNE Visualization ---
    plot_data = full_data[:NUM_EMBEDDINGS_TO_PLOT] 
    plot_dataset = SiameseGraphDataset(plot_data)
    plot_loader = torch.utils.data.DataLoader(
        plot_dataset, batch_size=64, shuffle=False, collate_fn=collate_fn)

    embeddings, labels = get_embeddings(model, plot_loader, DEVICE)
    plot_embeddings_tsne(embeddings, labels, 
                         title="GNN Embedding Space: Isomorphic (Blue) vs. Non-Isomorphic (Red)")


if __name__ == '__main__':
    run_analysis()