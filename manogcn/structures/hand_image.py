import torch
from dataclasses import dataclass

@dataclass
class HandImage:
    image: torch.tensor = None
    verts: torch.tensor = None
    joints: torch.tensor = None
    camK: torch.tensor = None
    uvds: torch.tensor = None
    idx: int = -1
    path: str = ''

    def fresh(self, ):
        hand_image = HandImage()
        hand_image.__dict__.update(self.__dict__)
        return hand_image

    def delete(self, attr):
        self.__dict__.pop(attr)
        return self