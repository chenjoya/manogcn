model=manogcn_1x_freihand

gpus=0
gpun=1

for epoch in "8e"
do
    weight=outputs/$model/model_$epoch\.pth
    CUDA_VISIBLE_DEVICES=$gpus python -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node=$gpun \
    test_net.py --config-file configs/$model\.yaml TEST.SAVE False TEST.VISUALIZE True
done 
