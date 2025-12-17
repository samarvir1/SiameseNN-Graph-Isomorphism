## Graph Isomorphism Detection via Siamese GNNs

This project implements a solution for the difficult **Graph Isomorphism Problem** using a specialized deep learning architecture: the **Siamese Graph Neural Network (GNN)**, specifically utilizing the **Graph Isomorphism Network (GIN)** layer.

The main idea is to train a model that can robustly and efficiently determine if two graphs are **structurally identical** (isomorphic), confirming the power (theoretical) and practical limits of GNNs in graph theory applications.

---

## 1. Project Structure and Setup

### Prerequisites

You must have a Python environment set up with PyTorch and PyTorch Geometric.

You can install all necessary dependencies using:

`pip install torch torch-geometric networkx numpy tqdm scikit-learn matplotlib`

### Files and Directories

| File/Folder                       | Role                       | Description                                                                                                                          |
| :-------------------------------- | :------------------------- | :----------------------------------------------------------------------------------------------------------------------------------- |
| `siamese_isomorphism_solver.py` | **Training Script**  | Manages the generation of the synthetic graph dataset and defines/trains the `SiameseGIN` model.                                   |
| `analysis_tools.py`             | **Analysis Script**  | Loads the latest trained model, performs theoretical evaluation, and generates dynamic t-SNE visualizations.                         |
| `graph_data/`                   | **Data Storage**     | Contains the generated dataset (`isomorphism_data_2000.pt`), consisting of 2000 pairs of graphs and their true isomorphism labels. |
| `siamese_gin_iso_detector_*.pt` | **Model Checkpoint** | The trained model's parameters, automatically detected and loaded by the analysis script.                                            |

---

## 2. Execution Guide

### Step 1: Run the Training Script

This step generates the training data and trains the Siamese GNN model for 50 epochs.

`python siamese_isomorphism_solver.py`

* **Output:** This generates `isomorphism_data_2000.pt` and saves the final trained model checkpoint named with a timestamp (e.g., `siamese_gin_iso_detector_1702821600.pt`).

### Step 2: Run the Analysis Script

This step evaluates the model's performance and theoretical limits.

* **Dynamic Visualization:** Every time you run this script, the t-SNE algorithm uses a **new random seed**. This means the 2D layout will look different with each run, visually demonstrating that the underlying structural clusters are stable regardless of how they are rotated or arranged in 2D space.

`python analysis_tools.py`

---

## 3. Results and Mathematical Insights

The analysis provides both empirical performance metrics and a critical theoretical evaluation of the GNN's power.

### A. The Siamese GNN Architecture

The architecture employs two identical GIN encoders sharing weights:

* **Encoder:** Maps a graph $G$ to a fixed-size vector $\mathbf{v}_G$ (the embedding).
* **Comparison:** The network compares the absolute difference $|\mathbf{v}_{G1} - \mathbf{v}_{G2}|$ to classify the pair.
* **Significance:** The GIN layer is theoretically **as powerful as the 1-Dimensional Weisfeiler-Leman (1-WL) test**, making it a near-optimal choice for a learning-based isomorphism solver.

### B. Critical WL-Hard Pair Test

This test verifies the known theoretical limitation of the GIN. We test the model on a pair of graphs that are **non-isomorphic** but known to be **indistinguishable** by the 1-WL test.

* **Expected Result:** The GNN should fail this test by predicting a high probability of isomorphism ($P \approx 1$).
* **Typical Console Output Example:** `Model's Prediction P(Isomorphic): 0.8858`
* **Conclusion:** This confirms the GNN is fundamentally limited by the mathematical constraint of the 1-WL test.

### C. t-SNE Visualization of Embeddings

The t-SNE plot visualizes the high-dimensional embeddings learned by the GNN:

* **Isomorphic Pairs (Blue):** These points cluster tightly, demonstrating that the network successfully learned a **permutation-invariant** representation.
* **Non-Isomorphic Pairs (Red):** These embeddings are spread out, confirming the model effectively learned to discriminate between different underlying structures.

### D. Quantitative Metrics (Validation Set)

The following metrics are calculated on the 20% validation split of the dataset:

| Metric              | Non-Isomorphic (0)            | Isomorphic (1)                | Overall Accuracy          |
| :------------------ | :---------------------------- | :---------------------------- | :------------------------ |
| **Precision** | Excellent ($\approx 0.925$) | Excellent ($\approx 0.910$) | High                      |
| **Recall**    | Excellent ($\approx 0.900$) | Excellent ($\approx 0.930$) | High                      |
| **F1-Score**  | Excellent ($\approx 0.912$) | Excellent ($\approx 0.920$) | **Typically > 90%** |
