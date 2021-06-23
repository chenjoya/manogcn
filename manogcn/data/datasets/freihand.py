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

# 0: Wrist, 1~4: Thumb, 5~8: Index, 9~12: Middle, 13~16: Ring, 17~20: Pink
class FreiHAND(torch.utils.data.Dataset):
    def __init__(self, image_dir, K_file, 
    mano_file=None, xyz_file=None, vert_file=None):
        image_files = [
            os.path.join(image_dir, r) for r in sorted(os.listdir(image_dir))
        ]
        Ks = json.load(open(K_file))
        repeat = len(image_files) // len(Ks)
        Ks = torch.tensor(Ks * repeat).squeeze()
        
        self.hand_images = [HandImage(camK=K, path=image_file, idx=idx) \
            for idx, (image_file, K) in enumerate(zip(image_files, Ks))]
        self.image_files = image_files
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            ),
        ])
        
        try:
            verts = json.load(open(vert_file))
            verts = torch.tensor(verts * repeat).squeeze()
            for hand_image in self.hand_images:
                hand_image.verts = verts[hand_image.idx] 
                hand_image.uvds = xyzs2uvds(hand_image.verts, hand_image.camK)
                hand_image.uvds[:, :2] /= 224
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
    
    def visualize(self, hi):
        verts = xyzs2uvds(hi.verts, hi.camK)
        joints = xyzs2uvds(hi.joints, hi.camK) 
        faces = torch.from_numpy(MANO(device="cpu").F.astype(np.int64))
        hi = {'path': hi.path, 'vertices': verts, 'faces': faces, 'joints': joints}
        torch.save(hi, "hi.pth")
    
    def evaluate(self, hand_images, logger, save_json_file="", visualize_dir=""):
        batched_verts = all_gather(torch.stack([hi.verts for hi in hand_images]))
        batched_joints = all_gather(torch.stack([hi.joints for hi in hand_images]))
        batched_uvds = all_gather(torch.stack([hi.uvds for hi in hand_images]))
        batched_uvd_joints = all_gather(torch.stack([hi.uvd_joints for hi in hand_images]))
        idxs = all_gather(
            torch.tensor([hi.idx for hi in hand_images], 
                device=batched_verts.device)
        )
        
        assert idxs.numel() == idxs.max() + 1

        if not is_main_process():
            return

        _, ranks = idxs.sort()
        batched_verts = batched_verts[ranks].cpu()
        batched_joints = batched_joints[ranks].cpu()
        batched_uvds = batched_uvds[ranks].cpu()
        batched_uvd_joints = batched_uvd_joints[ranks].cpu()
        
        # save json results
        if save_json_file:
            logger.info(f"Generating json results for 21 joints and 778 verts to {save_json_file}.")
            dirname = os.path.dirname(save_json_file)
            if not os.path.isdir(dirname):
                os.makedirs(dirname)
            with open(save_json_file, 'w') as f:
                json.dump([batched_joints.tolist(), batched_verts.tolist()], f)
            save_zip_file = save_json_file.replace('json', 'zip')
            with zipfile.ZipFile(save_zip_file, mode='w', compression=zipfile.ZIP_DEFLATED) as zf:
                _, save_json_name = os.path.split(save_json_file)
                zf.write(save_json_file, arcname=save_json_name)
            logger.info("Json results for 21 joints and 778 verts saved!")
        
        if visualize_dir:
            if not os.path.isdir(visualize_dir):
                os.makedirs(visualize_dir)
            vis_file = os.path.join(visualize_dir, "visualize.pth")
            logger.info("Generating visualization results for 3D/2D mesh and pose to {vis_file}.")
            faces = torch.from_numpy(MANO(device="cpu").F.astype(np.int64))
            results = [{'path': path, 'verts': verts, 'uvds': uvds, 'faces': faces, 'joints': uvd_joints} for 
                path, verts, uvds, uvd_joints in zip(self.image_files, batched_verts, batched_uvds, batched_uvd_joints)]
            torch.save(results, vis_file)
            
