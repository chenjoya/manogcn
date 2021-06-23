import logging
from tqdm import tqdm

import torch

from ..utils.comm import get_world_size, synchronize
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

def inference(
        model,
        data_loader,
        save_json_file,
        visualize_dir,
    ):
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
