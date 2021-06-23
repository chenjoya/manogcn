import torch

class AlignLoss(object):

    def orthogonal_procrustes(self, A, B):
        u, s, vt = torch.linalg.svd(A.permute(0,2,1).bmm(B))
        return u.matmul(vt), s.sum(dim=1)

    def align(self, A, B):
        R, s = self.orthogonal_procrustes(A, B)
        return A.matmul(R) * s.view(-1,1,1)
    
    def __call__(self, batched_verts, batched_verts_gt, batched_uvds, batched_uvds_gt): 
        
        aligned_verts_loss = torch.norm(
            self.align(batched_verts, batched_verts_gt) - batched_verts_gt, dim=2
        ).sum() / (len(batched_verts) * 10)
        
        uvds_loss = torch.norm(
            batched_uvds[:,:,:2] - batched_uvds_gt[:,:,:2], dim=2
        ).mean() * 3

        return dict(
            aligned_verts_loss=aligned_verts_loss, 
            uvds_loss=uvds_loss
        )

def build_loss(cfg):
    return AlignLoss() 
