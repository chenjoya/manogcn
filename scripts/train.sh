config_file="configs/manogcn_1x_freihand.yaml"
gpus=4,5
gpun=2

# ------------------------ need not change -----------------------------------
CUDA_VISIBLE_DEVICES=$gpus python -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node=$gpun \
    train_net.py --config-file $config_file
