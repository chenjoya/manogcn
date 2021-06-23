model=manogcnx3_1x_freihand

gpus=1
gpun=1
master_port=29502

CUDA_VISIBLE_DEVICES=$gpus python -m torch.distributed.launch --nproc_per_node=$gpun --master_port $master_port \
    test_net.py --config-file configs/$model\.yaml TEST.SAVE False TEST.VISUALIZE False MODEL.MANOGCN.NUM_LAYERS 3 
