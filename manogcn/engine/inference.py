import logging
import time
import os
from tqdm import tqdm

import torch

from ..utils.comm import is_main_process, get_world_size, synchronize
from ..utils.timer import Timer, get_time_str

def compute_on_dataset(model, data_loader, timer):
    model.eval()
    outputs = []
    for batches in tqdm(data_loader):
        with torch.no_grad():
            timer.tic()
            output = model(batches)
            torch.cuda.synchronize()
            outputs.append(output)
            timer.toc()
    outputs = sum(outputs, [])
    return outputs

"""def _accumulate_predictions_from_multiple_gpus(predictions_per_gpu, idxs_per_gpu):
    all_predictions = [all_gather(p) for p in predictions_per_gpu]
    all_idxs = all_gather(idxs_per_gpu)
    if not is_main_process():
        return
    if all_idxs.numel() != all_idxs.max() + 1:
        logger = logging.getLogger("manogcn.inference")
        logger.warning(
            "Number of images that were gathered from multiple processes is not "
            "a contiguous set. Some images might be missing from the evaluation"
        )
    _, idxs = all_idxs.sort()
    all_predictions = [p[idxs].cpu() for p in all_predictions]
    return all_predictions
"""

def inference(
        model,
        data_loader,
        save_json_file,
        visualize_dir,
        device="cuda",
    ):
    # convert to a torch.device for efficiency
    device = torch.device(device)
    num_devices = get_world_size()
    logger = logging.getLogger("manogcn.inference")
    dataset = data_loader.dataset
    logger.info("Start evaluation on {} dataset (Size: {}).".format(dataset.__class__.__name__, len(dataset)))
    total_timer = Timer()
    inference_timer = Timer()
    total_timer.tic()
    predictions = compute_on_dataset(model, data_loader, inference_timer)
    # wait for all processes to complete before measuring the time
    synchronize()
    total_time = total_timer.toc()
    total_time_str = get_time_str(total_time)
    logger.info(
        "Total run time: {} ({} s / inference per device, on {} devices)".format(
            total_time_str, total_time * num_devices / len(dataset), num_devices
        )
    )
    total_infer_time = get_time_str(inference_timer.total_time)
    logger.info(
        "Model inference time: {} ({} s / inference per device, on {} devices)".format(
            total_infer_time,
            inference_timer.total_time * num_devices / len(dataset),
            num_devices,
        )
    )
    
    dataset.evaluate(
        predictions, logger,
        save_json_file=save_json_file, 
        visualize_dir=visualize_dir
    )
