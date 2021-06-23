import os, zipfile, json, cv2
import numpy as np

import torch
from torchvision import transforms

from manogcn.modeling.mano import MANO
from manogcn.structures import HandImage
from manogcn.utils.comm import is_main_process, all_gather

def xyzs2uvds(xyzs, K):
    uvds = xyzs.mm(K.T)
    uvds[:,:2] /= uvds[:,2:]
    return uvds

class HO3D(torch.utils.data.Dataset):
    def __init__(self, root_dir, image_file, K_file, vert_file=None, xyz_file=None):
        image_files = json.load(open(image_file))
        image_files = [os.path.join(root_dir, r) for r in image_files]
        Ks = torch.tensor(json.load(open(K_file)))

        self.hand_images = [HandImage(camK=K, path=image_file, idx=idx) \
            for idx, (image_file, K) in enumerate(zip(image_files, Ks))]

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            ),
        ])

        C = torch.tensor([[1., 0., 0.], [0, -1., 0.], [0., 0., -1.]])
        try:
            verts = torch.tensor(json.load(open(vert_file))).matmul(C.T)
            for hand_image in self.hand_images:
                hand_image.verts = verts[hand_image.idx] 
                hand_image.uvds = xyzs2uvds(hand_image.verts, hand_image.camK)
                hand_image.uvds[:, 0] /= 640
                hand_image.uvds[:, 1] /= 480
        except:
            pass
    
    def __getitem__(self, idx):
        hand_image = self.hand_images[idx].fresh()
        hand_image.image = self.transform(
            cv2.cvtColor(
                cv2.imread(hand_image.path), 
                cv2.COLOR_BGR2RGB
            )
        )
        return hand_image
    
    def __len__(self):
        return len(self.hand_images) 
    
    def evaluate(self, hand_images, logger, save_json_file="", visualize_dir=""):
        batched_verts = torch.stack([hi.verts for hi in hand_images])
        batched_joints = torch.stack([hi.joints for hi in hand_images])
        idxs = torch.tensor([hi.idx for hi in hand_images], 
            device=batched_verts.device)

        batched_verts = all_gather(batched_verts)
        batched_joints = all_gather(batched_joints)
        idxs = all_gather(idxs)
        
        #batched_uvds = torch.stack([hi.uvds for hi in hand_images])
        #batched_uvds = all_gather(batched_uvds)

        if not is_main_process():
            return
        assert idxs.numel() == idxs.max() + 1

        _, ranks = idxs.sort()
        batched_verts = batched_verts[ranks].cpu()
        batched_joints = batched_joints[ranks].cpu()
        #batched_uvds = batched_uvds[ranks].cpu()
        
        if save_json_file:
            logger.info(f"Generating json results for 21 joints and 778 verts to {save_json_file}.")
            with open(save_json_file, 'w') as f:
                json.dump([batched_joints.tolist(), batched_verts.tolist()], f)
            save_zip_file = save_json_file.replace('json', 'zip')
            with zipfile.ZipFile(save_zip_file, mode='w', compression=zipfile.ZIP_DEFLATED) as zf:
                _, save_json_name = os.path.split(save_json_file)
                zf.write(save_json_file, arcname=save_json_name)
            logger.info("Json results for 21 joints and 778 verts saved!")
        
        if visualize_dir:
            logger.info("Generating visualization results for 3D/2D mesh and pose.")
            faces = torch.from_numpy(MANO(device="cpu").F.astype(np.int64))
            results = [{'path': path, 'vertices': uvds, 'faces': faces} for 
                path, uvds in zip(self.image_files, batched_uvds)]
            torch.save(results, "test.pth")
