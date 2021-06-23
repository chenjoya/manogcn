import numpy as np
import os
import _pickle as pickle

import torch 
from torch import nn

def rodrigues(r):
    B, N, _  = r.shape
    theta = r.norm(dim=2).unsqueeze(2)
    w = r / theta
    alpha, beta = theta.cos(), theta.sin()
    gamma = 1 - alpha
    omegav_matrix = torch.tensor([
        [[0, 0, 0],[0, 0, 1],[0, -1, 0]],
        [[0, 0, -1],[0, 0, 0],[1, 0, 0]],
        [[0, 1, 0],[-1, 0, 0],[0, 0, 0]],
        ], device=r.device, dtype=torch.float
    )
    omegav = w.view(-1,3).matmul(omegav_matrix).permute(1,0,2).view(B,N,3,3)
    A = w.unsqueeze(3).matmul(w.unsqueeze(2))
    eye = torch.eye(3, device=A.device).view(1,1,3,3).repeat(B,N,1,1)
    # A: B x N x 3 x 3, gamma: B x N x 1 x 1
    R = A * gamma.unsqueeze(-1) + omegav * beta.unsqueeze(-1) + eye * alpha.unsqueeze(-1)
    return R

def lrotmin(p): 
    batch = p.shape[0]
    eye = torch.eye(3, device=p.device)
    return (rodrigues(p[:, 3:].view(batch, -1, 3)) - eye).view(batch, -1)

def channel_matmul(a, b):
    return (a[:,0,:] * b[0] + a[:,1,:] * b[1] + a[:,2,:] * b[2] + a[:,3,:] * b[3]).T

class MANO(object):
    def __init__(self, device):
        super(MANO, self).__init__()
        cwd = os.path.dirname(__file__)
        model_path = os.path.join(cwd, 'MANO_RIGHT.pkl')
        smpl_data = pickle.load(open(model_path, 'rb'), encoding='latin1')
        self.hands_mean = torch.from_numpy(smpl_data['hands_mean']).float().to(device)
        self.posedirs = torch.from_numpy(smpl_data['posedirs']).float().to(device)
        self.v_template = torch.from_numpy(smpl_data['v_template']).float().to(device)
        self.shapedirs = torch.from_numpy(np.array(smpl_data['shapedirs'])).float().to(device)
        self.J_regressor = torch.from_numpy(smpl_data['J_regressor'].toarray()).float().to(device)
        self.weights = torch.from_numpy(smpl_data['weights']).float().to(device)
        
        self.F = smpl_data['f']

        kintree_table = smpl_data['kintree_table']
        id_to_col = {int(kintree_table[1,i]) : i for i in range(kintree_table.shape[1])}
        # {1: 0, 2: 1, 3: 2, 4: 0, 5: 4, 6: 5, 7: 0, 8: 7, 9: 8, 10: 0, 11: 10, 12: 11, 13: 0, 14: 13, 15: 14}
        self.parent = {i : id_to_col[int(kintree_table[0,i])] for i in range(1, kintree_table.shape[1])}
        
        self.joint2vert = {
            4: 744,  #ThumbT
            8: 320,  #IndexT
            12: 443,  #MiddleT
            16: 555,  #RingT
            20: 672  #PinkT
        }
        self.mano2joint = {
            0: 0, #Wrist
            1: 5, 2: 6, 3: 7, #Index
            4: 9, 5: 10, 6: 11, #Middle
            7: 17, 8: 18, 9: 19, # Pinky
            10: 13, 11: 14, 12: 15, # Ring
            13: 1, 14: 2, 15: 3, # Thumb
        } 

    def merge(self, r, j):
        a = torch.cat([r, j[:,:,None]], dim=2)
        b = torch.tensor([[[0., 0., 0., 1.]]]*a.shape[0], device=r.device)
        return torch.cat([a, b], dim=1)
    
    def pack(self, r):
        # r: 4
        zero = torch.zeros(4, 3, device=r.device)
        return torch.cat([zero, r[:, None]], dim=1)
    
    def estimate(self, poses, v, J):
        B, N, _ = J.shape
        poses = poses.view(B,-1,3)
        rs = rodrigues(poses)
        # rs: bxNx3x3
        A = [self.merge(rs[:, 0], J[:, 0])] 
        for i in range(1, 16):
            parent = self.parent[i]
            a = A[parent].matmul(self.merge(rs[:, i], J[:, i] - J[:, parent]))
            A.append(a)
        A = torch.stack(A, dim=1)
        J_cat = torch.cat([J, torch.zeros(B,N,1, device=J.device)], dim=2).unsqueeze(-1)
        A = A - torch.cat([
            torch.zeros(B,N,4,3, device=A.device), 
            A.matmul(J_cat)
        ], dim=3)
        T = A.permute(0,2,3,1).matmul(self.weights.T)
        rest_shape_h = torch.cat([v.permute(0,2,1), torch.ones(B,1,v.shape[1], device=v.device)], dim=1)
        verts = torch.stack([channel_matmul(T[i], rest_shape_h[i] )for i in range(B)])
        return verts[:,:,:3] 
    
    def verts2joints21(self, verts):
        joints16 = self.verts2joints16(verts)
        joints21 = [None for _ in range(21)]
        for mano_id, joint_id in self.mano2joint.items():
            joints21[joint_id] = joints16[:, mano_id]
        for joint_id, vert_id in self.joint2vert.items():
            joints21[joint_id] = verts[:, vert_id]
        joints21 = torch.stack(joints21, dim=1)
        return joints21
    
    def verts2joints16(self, verts):
        joints16 = self.J_regressor.matmul(verts)
        return joints16

    def __call__(self, poses, shapes):
        """ Poses the MANO model according to the root keypoint given. """
        fullposes = torch.cat([poses[:, :3], poses[:, 3:] + self.hands_mean], dim=1)
        v_shaped = self.shapedirs.matmul(shapes.T).permute(2,0,1) + self.v_template
        J = self.J_regressor.matmul(v_shaped)
        v_posed = v_shaped + self.posedirs.matmul(lrotmin(fullposes).T).permute(2,0,1)
        verts = self.estimate(fullposes, v_posed, J)
        return verts

if __name__ == "__main__":
    mano = MANO(device="cpu")
    pose = torch.zeros(2, 48)
    shape = torch.zeros(2, 10)
    verts, joints16 = mano(pose, shape)
    print(verts.mean(dim=1))
