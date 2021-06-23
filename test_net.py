import argparse
import os

import torch

from manogcn.config import cfg
from manogcn.data import make_data_loader
from manogcn.engine.inference import inference
from manogcn.modeling import build_model
from manogcn.utils.checkpoint import Checkpointer
from manogcn.utils.comm import synchronize, get_rank
from manogcn.utils.logger import setup_logger
from manogcn.utils.miscellaneous import mkdir

def main():
    parser = argparse.ArgumentParser(description="manogcn")
    parser.add_argument(
        "--config-file",
        default="configs/freihand.yaml",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )
    args = parser.parse_args()

    num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    distributed = num_gpus > 1

    if distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend="nccl")
        synchronize()

    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    save_dir = ""
    logger = setup_logger("manogcn", save_dir, get_rank())
    logger.info("Using {} GPUs".format(num_gpus))
    logger.info(cfg)

    logger.info("Collecting env info (might take some time)")
    from torch.utils.collect_env import get_pretty_env_info
    logger.info("\n" + get_pretty_env_info())

    model = build_model(cfg)
    model.to(cfg.MODEL.DEVICE)

    if distributed:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.local_rank]
        )

    weight_file =  cfg.MODEL.WEIGHT
    
    basename = os.path.basename(args.config_file)
    basename = os.path.splitext(basename)[0]
    output_dir = os.path.join("outputs/", basename)
    checkpointer = Checkpointer(cfg, model, save_dir=output_dir)
    _ = checkpointer.load(f=weight_file, use_latest=False)
    
    output_folders = [None] * len(cfg.DATASETS.TEST)
    dataset_names = cfg.DATASETS.TEST
    if output_dir:
        for idx, dataset_name in enumerate(dataset_names):
            output_folder = os.path.join(output_dir, "inference", dataset_name)
            mkdir(output_folder)
            output_folders[idx] = output_folder
    data_loaders_val = make_data_loader(cfg, is_train=False, is_distributed=distributed)

    for output_folder, dataset_name, data_loader_val in zip(output_folders, dataset_names, data_loaders_val):
        js_basename = os.path.basename(weight_file).replace('pth', 'json') if weight_file else "test.json"
        save_json_file = os.path.join(output_folder, js_basename) if cfg.TEST.SAVE else ""
        visualize_dir = output_folder if cfg.TEST.VISUALIZE else ""
        inference(
            model,
            data_loader_val,
            save_json_file=save_json_file,
            visualize_dir=visualize_dir,
            device=cfg.MODEL.DEVICE,
        )
        synchronize()

if __name__ == "__main__":
    main()
