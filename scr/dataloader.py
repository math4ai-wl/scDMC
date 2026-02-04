import scanpy as sc
import torch 
from torch.utils.data import DataLoader,Dataset
import numpy as np


class scDataset(Dataset):
    def __init__(self,adata_path,raw_layer_name='counts',use_cuda_cache=False):
        super().__init__()
        print('Loading adata from',adata_path)
        self.adata = sc.read_h5ad(adata_path)

        X_matrix = self.adata.X
        if hasattr(X_matrix, 'toarray'):
            X_matrix = X_matrix.toarray()
        X_matrix = np.asarray(X_matrix, dtype=np.float32)
        self.data = X_matrix.copy()

        # Raw counts source priority:
        # 1) specified layer (e.g., "counts")
        # 2) adata.raw.X
        # 3) fallback to adata.X
        if raw_layer_name is not None and raw_layer_name in self.adata.layers:
            raw_matrix = self.adata.layers[raw_layer_name]
        elif self.adata.raw is not None and self.adata.raw.X is not None:
            raw_matrix = self.adata.raw.X
        else:
            print(
                "Warning: raw counts not found; using adata.X as counts and applying log1p(normalized counts) as model input."
            )
            raw_matrix = X_matrix

            # When only counts are available, keep raw_data as counts and set data to log1p(library-size normalized counts).
            counts = np.asarray(raw_matrix, dtype=np.float32)
            libsize = counts.sum(axis=1, keepdims=True)
            libsize[libsize <= 0] = 1.0
            norm = counts / libsize * 1e4
            self.data = np.log1p(norm).astype(np.float32)

        if hasattr(raw_matrix, 'toarray'):
            raw_matrix = raw_matrix.toarray()
        self.raw_data = np.asarray(raw_matrix, dtype=np.float32)
        
        # Size factor (library size).
        if 'size_factors' in self.adata.obs.columns:
            self.size_factor = np.asarray(self.adata.obs['size_factors'].values, dtype=np.float32)
        else:
            self.size_factor = np.asarray(self.raw_data.sum(axis=1), dtype=np.float32)
        self.size_factor[self.size_factor <= 0] = 1.0
        
        self.data = torch.tensor(self.data).float()
        self.raw_data = torch.tensor(self.raw_data).float()
        self.size_factor = torch.tensor(self.size_factor).float()
        
        print(f"Data loaded: {self.data.shape}")
        print(f"Raw data loaded: {self.raw_data.shape}")
        
    def __len__(self):
        return self.data.shape[0]
    
    def __getitem__(self,idx):
        return self.data[idx],self.raw_data[idx],self.size_factor[idx]
        
    
