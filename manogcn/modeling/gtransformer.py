import torch
from torch import nn

def orthogonal_procrustes(M):
    u, s, v = M.svd()
    return u.matmul(v.permute(0,2,1)), s.sum(dim=1)

def align(A, M):
    R, s = orthogonal_procrustes(M)
    return A.matmul(R) * s.view(-1,1,1)

class GlobalTransformer(nn.Module):
    def __init__(self, in_features):
        super(GlobalTransformer, self).__init__()
        self.refeature = nn.Linear(in_features, in_features)
        num_verts = 778 * 3
        self.transformer = nn.Sequential(
            nn.Linear(in_features + num_verts, in_features),
            nn.ReLU(),
            nn.Linear(in_features, 3*3+2+1)
        )
    
    def forward(self, batched_verts, features):
        features = self.refeature(features).relu()
        features = torch.cat([
            batched_verts.view(len(batched_verts),-1), 
            features
        ], dim=1)
        
        transforms = self.transformer(features)
        svdM, shift, scale = transforms[:,:9].view(-1,3,3), \
            transforms[:,9:11].view(-1,1,2), transforms[:,-1].view(-1,1,1)
        batched_uvds = align(batched_verts, svdM) * scale
        batched_uvds[:,:,:2] = batched_uvds[:,:,:2] + shift
        return batched_uvds
        
def build_gtransformer(cfg, in_features):
    return GlobalTransformer(in_features)
