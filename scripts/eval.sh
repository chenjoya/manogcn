model=manogcnx3_1x_freihand

gpus=0
gpun=1
master_port=29502

for epoch in "8e"
do
    weight=outputs/$model/model_$epoch\.pth
    CUDA_VISIBLE_DEVICES=$gpus python -m torch.distributed.launch --nproc_per_node=$gpun --master_port $master_port \
    test_net.py --config-file configs/$model\.yaml MODEL.WEIGHT $weight TEST.SAVE False TEST.VISUALIZE True
done 
