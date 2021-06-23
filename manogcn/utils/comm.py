"""
This file contains primitives for multi-gpu communication.
This is useful when doing distributed training.
"""

import pickle
import time

import torch
import torch.distributed as dist

def get_world_size():
    if not dist.is_available():
        return 1
    if not dist.is_initialized():
        return 1
    return dist.get_world_size()

def get_rank():
    if not dist.is_available():
        return 0
    if not dist.is_initialized():
        return 0
    return dist.get_rank()

def is_main_process():
    return get_rank() == 0

def synchronize():
    """
    Helper function to synchronize (barrier) among all processes when
    using distributed training
    """
    if not dist.is_available():
        return
    if not dist.is_initialized():
        return
    world_size = dist.get_world_size()
    if world_size == 1:
        return
    dist.barrier()

def all_gather(data):
    world_size = get_world_size()
    if world_size == 1:
        return data
    batch = torch.tensor(data.shape[0], dtype=torch.long, device=data.device)
    batches = [torch.tensor(0, dtype=torch.long, device=data.device) for _ in range(world_size)]
    dist.all_gather(batches, batch)
    max_batch = max(batches).item()
    
    max_shape = list(data.shape)
    max_shape[0] = max_batch

    datas = [torch.zeros(max_shape, dtype=data.dtype, device=data.device) for _ in range(world_size)]
    if batch != max_batch:
        pad_shape = max_shape
        pad_shape[0] = max_batch - batch 
        data = torch.cat([data, torch.zeros(pad_shape, dtype=data.dtype, device=data.device)])
    dist.all_gather(datas, data)

    datas = [data[:batch] for batch, data in zip(batches, datas)]
    return torch.cat(datas)

def reduce_dict(input_dict, average=True):
    """
    Args:
        input_dict (dict): all the values will be reduced
        average (bool): whether to do average or sum
    Reduce the values in the dictionary from all processes so that process with rank
    0 has the averaged results. Returns a dict with the same fields as
    input_dict, after reduction.
    """
    world_size = get_world_size()
    if world_size < 2:
        return input_dict
    with torch.no_grad():
        names = []
        values = []
        # sort the keys so that they are consistent across processes
        for k in sorted(input_dict.keys()):
            names.append(k)
            values.append(input_dict[k])
        values = torch.stack(values, dim=0)
        dist.reduce(values, dst=0)
        if dist.get_rank() == 0 and average:
            # only main process gets accumulated, so only divide by
            # world_size in this case
            values /= world_size
        reduced_dict = {k: v for k, v in zip(names, values)}
    return reduced_dict
