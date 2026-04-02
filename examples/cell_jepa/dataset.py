## load gene expression matrix and pairs of cells

import torch
import numpy as np
import scanpy as sc
from torch.utils.data import Dataset

class CellDataset(Dataset):
    #pairs
    def __init__(self, h5ad_path, pairs_path):

        #load gene expression matrix
        print(f"Loading AnnData from {h5ad_path}...")
        self.adata = sc.read_h5ad(h5ad_path)
        self.X = self.adata.X
        if hasattr(self.X, "toarray"):
            # For toy data/smaller subsets, load into memory
            print("Converting sparse X to dense...")
            self.X = self.X.toarray()
        
        
        #pairs shaoe (N,2) with each row being [current-index, next-index]
        print(f"Loading pairs from {pairs_path}...")
        self.pairs = np.load(pairs_path)
        
        # Load cell types
        if 'author_cell_type' in self.adata.obs:
            self.cell_types = self.adata.obs['author_cell_type'].values
            self.cell_type_to_idx = {name: i for i, name in enumerate(np.unique(self.cell_types))}
            self.cell_type_labels = np.array([self.cell_type_to_idx[name] for name in self.cell_types])
            print(f"Loaded {len(self.cell_type_to_idx)} unique cell types.")
        else:
            self.cell_type_labels = None
            
        print(f"Dataset initialized with {len(self.pairs)} pairs.")

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        #return samples from the current and next time step
        i_curr, i_next = self.pairs[idx]
        x_curr = torch.from_numpy(self.X[i_curr]).float()
        x_next = torch.from_numpy(self.X[i_next]).float()
        
        y_curr = self.cell_type_labels[i_curr] if self.cell_type_labels is not None else -1 #cell type of current cell, -1 if doesnt exist
        y_next = self.cell_type_labels[i_next] if self.cell_type_labels is not None else -1 #cell type of next cell, -1 if doesnt exist
        
        # Reshape to [C, T=1, H=1, W=1] to match JEPA sequence expectation
        # Returns: x_curr, x_next, y_curr (label of source cell), y_next (label of target cell)
        return x_curr.view(-1, 1, 1, 1), x_next.view(-1, 1, 1, 1), y_curr, y_next
