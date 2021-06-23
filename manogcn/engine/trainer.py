import datetime
import logging
import os
import time
from tqdm import tqdm

import torch
import torch.distributed as dist

from manogcn.data import make_data_loader
from manogcn.utils.comm import get_world_size, synchronize, reduce_dict, is_main_process
from manogcn.utils.metric_logger import MetricLogger
from manogcn.engine.inference import inference
from manogcn.utils.miscellaneous import mkdir

def do_train(
    cfg,
    model,
    data_loader,
    data_loaders_val,
    optimizer,
    scheduler,
    checkpointer,
    device,
    checkpoint_period,
    test_period,
    arguments,
    output_dir,
):
    logger = logging.getLogger("manogcn.trainer")
    logger.info("Start training")
    meters = MetricLogger(delimiter="  ")
    epochs = cfg.SOLVER.EPOCHS

    model.train()
    start_training_time = time.time()

    for epoch in range(arguments["epoch"] + 1, epochs + 1):
        max_iteration = len(data_loader)
        last_epoch_iteration = (epochs - epoch) * max_iteration
        end = time.time()

        for _iteration, batches in enumerate(data_loader):
            data_time = time.time() - end
            iteration = _iteration + 1
            
            optimizer.zero_grad()
            loss_dict = model(batches)
            losses = sum(loss for loss in loss_dict.values())
            losses.backward()
            optimizer.step()
            batch_time = time.time() - end

            if iteration % 100 == 0 or iteration == max_iteration:
                  meters.update(time=batch_time, data=data_time)
                  loss_dict_reduced = reduce_dict(loss_dict)
                  losses_reduced = sum(loss for loss in loss_dict_reduced.values())
                  meters.update(loss=losses_reduced, **loss_dict_reduced)
                  eta_seconds = meters.time.global_avg * (max_iteration - iteration + last_epoch_iteration)
                  eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                  logger.info(
                      meters.delimiter.join(
                          [
                              "eta: {eta}",
                              "epoch: {epoch}/{max_epoch}",
                              "iteration: {iteration}/{max_iteration}",
                              "{meters}",
                              "lr: {lr:.6f}",
                              "max mem: {memory:.0f}",
                          ]
                      ).format(
                          eta=eta_string,
                          epoch=epoch,
                          max_epoch=epochs,
                          iteration=iteration,
                          max_iteration=max_iteration,
                          meters=str(meters),
                          lr=optimizer.param_groups[0]["lr"],
                          memory=torch.cuda.max_memory_allocated() / 1024.0 / 1024.0,
                      )
                  )
            end = time.time()
            
        scheduler.step()
        arguments["epoch"] = epoch
        
        if epoch % checkpoint_period == 0:
            checkpointer.save(f"model_{epoch}e", **arguments)
        
        if data_loaders_val is not None and test_period > 0 and epoch % test_period == 0:
            dataset_names = cfg.DATASETS.TEST
            output_folders = [None for _ in dataset_names]

            for idx, dataset_name in enumerate(dataset_names):
                output_folder = os.path.join(output_dir, "inference", dataset_name)
                mkdir(output_folder)
                output_folders[idx] = output_folder

            for output_folder, data_loader_val in zip(output_folders, data_loaders_val):
                save_json_basename = f"model_{epoch}e.json"
                save_json_file = os.path.join(output_folder, save_json_basename)
                synchronize()
                inference(
                    model,
                    data_loader_val,
                    save_json_file=save_json_file,
                    visualize_dir="", # do not visualize here
                    device=cfg.MODEL.DEVICE,
                )
                synchronize()
                model.train()

    total_training_time = time.time() - start_training_time
    total_time_str = str(datetime.timedelta(seconds=total_training_time))
    logger.info(
        "Total training time: {} ({:.4f} s / it)".format(
            total_time_str, total_training_time / (arguments["epoch"])
        )
    )
