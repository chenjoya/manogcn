import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter

from .mano import MANO

class GraphConv(nn.Module):
    def __init__(self, in_features, out_features):
        super(GraphConv, self).__init__()
        self.fc = nn.Linear(in_features=in_features, out_features=out_features)
    
    def laplacian(self, A):
        D = A.sum(dim=0) ** (-0.5)
        return D * A * D

    def forward(self, X, A):
        return self.fc(self.laplacian(A).matmul(X)).relu()

class MANOGCN(nn.Module):
    def __init__(self, cfg, in_features, num_layers, num_pose=16, num_shape=10, num_vert=778):
        super(MANOGCN, self).__init__()
        self.mano = MANO(device='cuda')
        
        self.refeature = nn.Linear(in_features, in_features)

        pose_features = in_features // num_pose 
        self.pose_f_fc = nn.Linear(in_features, num_pose*pose_features)
        self.shape_f_fc = nn.Linear(in_features, num_shape*num_vert)
        
        self.pose_gcn = nn.ModuleList([GraphConv(pose_features+3, pose_features) for _ in range(num_layers)])
        self.shape_gcn = nn.ModuleList([GraphConv(num_shape+3, num_shape) for _ in range(num_layers)])
        
        pose_graph = torch.eye(num_pose, dtype=torch.float, device="cuda")
        for k, v in self.mano.parent.items():
            pose_graph[k][v] = pose_graph[v][k] = 1 
        mesh_graph = torch.eye(num_vert, dtype=torch.float, device="cuda")
        for i, j, k in self.mano.F:
            mesh_graph[i][j] = mesh_graph[j][k] = mesh_graph[i][k] = 1 
            mesh_graph[j][i] = mesh_graph[k][j] = mesh_graph[k][i] = 1 
        
        self.pose_graph = pose_graph
        self.mesh_graph = mesh_graph

        self.pose_fc = nn.Linear(in_features=pose_features, out_features=3)
        self.shape_fc = nn.Linear(in_features=num_vert, out_features=1)
        
        self.pose_gfcs = nn.ModuleList([nn.Linear(in_features=pose_features, out_features=3) for _ in range(num_layers)])
        self.shape_gfcs = nn.ModuleList([nn.Linear(in_features=num_vert, out_features=1) for _ in range(num_layers)])

        self.num_pose = num_pose
        self.num_shape = num_shape
        self.num_layers = num_layers
    
    def normalize(self, batched_points):
        mean = batched_points.mean(dim=1)
        shift = batched_points - mean.view(-1, 1, 3)
        scale = shift.view(len(shift), -1).norm(dim=1) + 1e-8
        return shift / scale.view(-1,1,1)
    
    def manor(self, pose_fs, shape_fs, pose_fc, shape_fc):
        # diff mano mapping
        batch = len(pose_fs)
        poses = pose_fc(pose_fs).view(batch, -1) 
        shapes = shape_fc(shape_fs).view(batch, -1) 
        verts = self.mano(poses, shapes)
        verts = self.normalize(verts)
        joints16 = self.mano.verts2joints16(verts)
        return verts, joints16
    
    def forward(self, x): 
        x = self.refeature(x).relu()
        batch = len(x)

        pose_fs = self.pose_f_fc(x).relu().view(batch, 16, -1)
        shape_fs = self.shape_f_fc(x).relu().view(batch, 10, -1) 
        
        verts, joints16 = self.manor(pose_fs, shape_fs, self.pose_fc, self.shape_fc)
        
        for i in range(self.num_layers):
            # feature aggregation 
            pose_gfs = torch.cat([pose_fs, joints16], dim=2)
            pose_gfs = self.pose_gcn[i](pose_gfs, self.pose_graph)
            pose_gfc = self.pose_gfcs[i]
            
            shape_gfs = torch.cat([shape_fs.permute(0,2,1), verts], dim=2)    
            shape_gfs = self.shape_gcn[i](shape_gfs, self.mesh_graph).permute(0,2,1)
            shape_gfc = self.shape_gfcs[i]
            verts, joints16 = self.manor(pose_gfs, shape_gfs, pose_gfc, shape_gfc)
        
        return verts

def build_manogcn(cfg, in_features):
    num_layers = cfg.MODEL.MANOGCN.NUM_LAYERS
    return MANOGCN(cfg, in_features, num_layers)
