import os
from yacs.config import CfgNode as CN

# -----------------------------------------------------------------------------
# Config definition
# -----------------------------------------------------------------------------

_C = CN()

_C.MODEL = CN()
_C.MODEL.WEIGHT = ""

_C.MODEL.HRNET = CN()
_C.MODEL.HRNET.PRETRAINED = ""
#_C.MODEL.HRNET.EXTRA = CN(new_allowed=True)
_C.MODEL.HRNET.EXTRA = CN()
_C.MODEL.HRNET.EXTRA.PRETRAINED_LAYERS = ('conv1','bn1','conv2','bn2','layer1','transition1','stage2','transition2','stage3','transition3','stage4')
_C.MODEL.HRNET.EXTRA.FINAL_CONV_KERNEL = 1
_C.MODEL.HRNET.EXTRA.STAGE2 = CN()
_C.MODEL.HRNET.EXTRA.STAGE2.NUM_MODULES = 1
_C.MODEL.HRNET.EXTRA.STAGE2.NUM_BRANCHES = 2
_C.MODEL.HRNET.EXTRA.STAGE2.BLOCK = "BASIC"
_C.MODEL.HRNET.EXTRA.STAGE2.NUM_BLOCKS = (4, 4)
_C.MODEL.HRNET.EXTRA.STAGE2.NUM_CHANNELS = (32, 64)
_C.MODEL.HRNET.EXTRA.STAGE2.FUSE_METHOD = "SUM"
_C.MODEL.HRNET.EXTRA.STAGE3 = CN()
_C.MODEL.HRNET.EXTRA.STAGE3.NUM_MODULES = 4
_C.MODEL.HRNET.EXTRA.STAGE3.NUM_BRANCHES = 3
_C.MODEL.HRNET.EXTRA.STAGE3.BLOCK = "BASIC"
_C.MODEL.HRNET.EXTRA.STAGE3.NUM_BLOCKS = (4, 4, 4)
_C.MODEL.HRNET.EXTRA.STAGE3.NUM_CHANNELS = (32, 64, 128)    
_C.MODEL.HRNET.EXTRA.STAGE3.FUSE_METHOD = "SUM"
_C.MODEL.HRNET.EXTRA.STAGE4 = CN()
_C.MODEL.HRNET.EXTRA.STAGE4.NUM_MODULES = 3
_C.MODEL.HRNET.EXTRA.STAGE4.NUM_BRANCHES = 4
_C.MODEL.HRNET.EXTRA.STAGE4.BLOCK = "BASIC"
_C.MODEL.HRNET.EXTRA.STAGE4.NUM_BLOCKS = (4, 4, 4, 4)
_C.MODEL.HRNET.EXTRA.STAGE4.NUM_CHANNELS = (32, 64, 128, 256)
_C.MODEL.HRNET.EXTRA.STAGE4.FUSE_METHOD = "SUM"

_C.MODEL.MANONET = CN()
_C.MODEL.MANONET.ARCHITECTURE = "resnet50"

_C.MODEL.GTRANSNET = CN()
_C.MODEL.GTRANSNET.ARCHITECTURE = "resnet18"

_C.MODEL.MANOGCN = CN()
_C.MODEL.MANOGCN.NUM_LAYERS = 3

_C.DATASETS = CN()
_C.DATASETS.TRAIN =()
_C.DATASETS.TEST = ()

_C.SOLVER = CN()
_C.SOLVER.BATCH_SIZE = 64
_C.SOLVER.MILESTONES = (6, 7)
_C.SOLVER.EPOCHS = 8
_C.SOLVER.LR = 0.1
_C.SOLVER.MOMENTUM = 0.9
_C.SOLVER.WEIGHT_DECAY = 1e-4
_C.SOLVER.GAMMA = 0.1
_C.SOLVER.TEST_PERIOD = 2
_C.SOLVER.CHECKPOINT_PERIOD = 2

_C.DATALOADER = CN()
_C.DATALOADER.NUM_WORKERS = 2

_C.TEST = CN()
_C.TEST.BATCH_SIZE = 64
_C.TEST.SAVE = True
_C.TEST.VISUALIZE = False

_C.PATHS_CATALOG = os.path.join(os.path.dirname(__file__), "paths_catalog.py")
