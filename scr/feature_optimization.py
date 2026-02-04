import torch
import torch.nn as nn
import torch.nn.functional as F

class BackFeature(nn.Module):
    def __init__(self,batch_size,num_classes,device):
        """
        Feature regularizer used during self-training.
        """
        super(BackFeature,self).__init__()
        self.batch_size = batch_size
        self.num_classes = num_classes
        self.device = device

    def forward(self,features,labels):
        """
        Args:
            features: feature vectors (N, D)
            labels: cluster labels (N,)
        """
        features = F.normalize(features,p=2,dim=1)
        sim_features = torch.mm(features,features.t())
        dist = 1. - sim_features
        
        if not isinstance(labels,torch.Tensor):
            labels = torch.tensor(labels,dtype = torch.long)
        labels = labels.to(self.device).view(-1,1)
        
        same_mask = torch.eq(labels,labels.t()).float().to(self.device)
        same_mask.fill_diagonal_(0)
        
        diff_mask = 1. - same_mask
        diff_mask.fill_diagonal_(0)
        
        # Pull same-cluster samples closer while penalizing high similarity across different clusters.
        same_loss = (dist*same_mask).sum()/(same_mask.sum()+1e-8)
        diff_sim_penalty = F.relu(sim_features)*diff_mask
        diff_loss = diff_sim_penalty.sum()/(diff_mask.sum()+1e-8)
        return same_loss+diff_loss       
