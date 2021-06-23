config_file="configs/manogcnx3_1x_freihand.yaml"
gpus=1,2
gpun=2

# ------------------------ need not change -----------------------------------
CUDA_VISIBLE_DEVICES=$gpus python -m torch.distributed.launch --nproc_per_node=$gpun --master_port=$((RANDOM + 10000)) \
    train_net.py --config-file $config_file
