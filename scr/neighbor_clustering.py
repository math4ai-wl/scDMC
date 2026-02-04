import torch
import torch.nn.functional as F
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components 

class NeighborClusteringEuclidean:
    def __init__(self, device='cuda', merge_lambda: float = 0.5, disable_merge: bool = False):
        self.device = device
        self.merge_lambda = float(merge_lambda)
        self.disable_merge = bool(disable_merge)
        
    def step(self, Z, labels, distance_threshold=None):
        if self.disable_merge:
            # No merging: keep labels unchanged.
            if isinstance(labels, torch.Tensor):
                labels_np = labels.detach().cpu().numpy()
            else:
                labels_np = np.asarray(labels)
            return int(len(np.unique(labels_np))), labels

        labels_tensor = torch.tensor(labels, device=self.device)
        unique_labels = torch.unique(labels_tensor)
        num_clusters = len(unique_labels)
        
        if num_clusters <= 1:
            return num_clusters, labels
        
        dim = Z.shape[1]
        centers = torch.zeros((num_clusters, dim), device=self.device)
        
        for i, lbl in enumerate(unique_labels):
            mask = (labels_tensor == lbl)
            cluster_points = Z[mask]
            center = torch.mean(cluster_points, dim=0)
            centers[i] = center
        
        dist_matrix = torch.cdist(centers, centers, p=2)
        
        dist_matrix.fill_diagonal_(float('inf'))
        
        neighbor_dist, neighbor_indices = torch.min(dist_matrix, dim=1)

        # Merge policy: use an absolute threshold if provided; otherwise use an adaptive threshold.
        # Adaptive form:
        #   threshold = mean(nearest_centroid_dist) - merge_lambda * std(nearest_centroid_dist)
        # Larger merge_lambda => stricter (fewer merges). Smaller merge_lambda => more aggressive merging.
        if distance_threshold is not None:
            selected_mask = neighbor_dist <= distance_threshold
        else:
            mean_dist = torch.mean(neighbor_dist)
            std_dist = torch.std(neighbor_dist)
            threshold = mean_dist - float(self.merge_lambda) * std_dist
            if threshold < neighbor_dist.min():
                threshold = neighbor_dist.min() * 1.1
            
            selected_mask = neighbor_dist <= threshold

        selected_src = torch.nonzero(selected_mask, as_tuple=False).squeeze(1)

        if selected_src.numel() == 0:
            return num_clusters, labels

        selected_dst = neighbor_indices[selected_src]
        
        src_np = selected_src.cpu().numpy()
        dst_np = selected_dst.cpu().numpy()
        
        connections = np.concatenate([np.stack([src_np, dst_np], axis=1),
                                      np.stack([dst_np, src_np], axis=1)], axis=0)
        data = np.ones(connections.shape[0])
        adj_matrix = coo_matrix(
            (data, (connections[:, 0], connections[:, 1])),
            shape=(num_clusters, num_clusters)
        )
        
        n_components, new_cluster_ids = connected_components(adj_matrix, directed=False, return_labels=True)
        
        unique_labels_np = unique_labels.cpu().numpy()
        label_mapping = {old: new for old, new in zip(unique_labels_np, new_cluster_ids)}
        new_labels = np.array([label_mapping[lbl] for lbl in labels])
        
        return n_components, new_labels
