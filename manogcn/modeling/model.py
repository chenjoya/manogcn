import torch
from torch import nn

from .hrnet import build_hrnet
from .resnetbody import build_resnetbody
from .mano import MANO
from .manogcn import build_manogcn 
from .loss import build_loss
from .gtransformer import build_gtransformer

class Model(nn.Module):
    def __init__(self, cfg):
        super(Model, self).__init__()
        self.manonet = build_resnetbody(cfg.MODEL.MANONET.ARCHITECTURE)
        self.gtransnet = build_resnetbody(cfg.MODEL.GTRANSNET.ARCHITECTURE)
        self.hrnet = build_hrnet(cfg, self.manonet.in_channels, self.gtransnet.in_channels)
        
        self.manogcn = build_manogcn(cfg, self.manonet.out_channels)
        self.gtransformer = build_gtransformer(cfg, self.gtransnet.out_channels)
        
        self.loss = build_loss(cfg)
        self.device = torch.device(cfg.MODEL.DEVICE)
    
    def forward(self, hand_images):
        images = torch.stack([hi.image for hi in hand_images]).to(self.device)
        mano_features, gtrans_features = self.hrnet(images)
        mano_features = self.manonet(mano_features)
        gtrans_features = self.gtransnet(gtrans_features)
        
        batched_verts = self.manogcn(mano_features)
        batched_uvds = self.gtransformer(batched_verts, gtrans_features)
        
        if self.training:
            batched_verts_gt = torch.stack(
                [hi.verts for hi in hand_images]).to(images.device)
            batched_verts_gt = self.manogcn.normalize(batched_verts_gt)

            batched_uvds_gt = torch.stack(
                [hi.uvds for hi in hand_images]).to(images.device)
            return self.loss(batched_verts, batched_verts_gt, 
                batched_uvds, batched_uvds_gt)
        else:
            batched_joints = self.manogcn.mano.verts2joints21(batched_verts)
            batched_uvd_joints = self.manogcn.mano.verts2joints21(batched_uvds)
            for hand_image, verts, joints, uvds, uvd_joints in zip(
                hand_images, batched_verts, batched_joints, batched_uvds, batched_uvd_joints
            ):
                hand_image = hand_image.delete('image')
                hand_image.verts = verts
                hand_image.joints = joints
                hand_image.uvds = uvds
                hand_image.uvd_joints = uvd_joints
            return hand_images
