import argparse
import os

import torch
from torch import optim
from torch import nn

from manogcn.config import cfg
from manogcn.data import make_data_loader
from manogcn.engine.inference import inference
from manogcn.engine.trainer import do_train
from manogcn.modeling import build_model
from manogcn.utils.checkpoint import Checkpointer
from manogcn.utils.comm import synchronize, get_rank
from manogcn.utils.logger import setup_logger
from manogcn.utils.miscellaneous import mkdir, save_config

def train(cfg, local_rank, distributed, output_dir):
    model = build_model(cfg)
    device = torch.device(cfg.MODEL.DEVICE)
    torch.backends.cudnn.benchmark = True
    model.to(device)
    
    optimizer = optim.SGD(
        model.parameters(),
        lr=cfg.SOLVER.LR,
        momentum=cfg.SOLVER.MOMENTUM,
        weight_decay=cfg.SOLVER.WEIGHT_DECAY
    )
    scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=cfg.SOLVER.MILESTONES,
        gamma=cfg.SOLVER.GAMMA
    )

    if distributed:
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[local_rank], output_device=local_rank
        )

    arguments = {"epoch": 0}
    save_to_disk = get_rank() == 0
    checkpointer = Checkpointer(
        cfg, model, optimizer, scheduler, output_dir, save_to_disk
    )
    if cfg.MODEL.WEIGHT:
        extra_checkpoint_data = checkpointer.load(f=cfg.MODEL.WEIGHT, use_latest=False)
    else:
        extra_checkpoint_data = checkpointer.load(f=None, use_latest=True)
    
    arguments.update(extra_checkpoint_data)

    data_loader = make_data_loader(
        cfg,
        is_train=True,
        is_distributed=distributed,
    )

    test_period = cfg.SOLVER.TEST_PERIOD
    if test_period > 0:
        data_loader_val = make_data_loader(cfg, is_train=False, is_distributed=distributed)
    else:
        data_loader_val = None

    checkpoint_period = cfg.SOLVER.CHECKPOINT_PERIOD

    
    do_train(
        cfg,
        model,
        data_loader,
        data_loader_val,
        optimizer,
        scheduler,
        checkpointer,
        device,
        checkpoint_period,
        test_period,
        arguments,
        output_dir,
    )

    return model

def main():
    parser = argparse.ArgumentParser(description="manogcn")
    parser.add_argument(
        "--config-file",
        default="",
        metavar="FILE",
        help="path to config file",
        type=str,
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
    args.distributed = num_gpus > 1

    if args.distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend="nccl")
        synchronize()

    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    model_name = os.path.splitext(os.path.basename(args.config_file))[0]
    output_dir = os.path.join("outputs", model_name)
    if output_dir:
        mkdir(output_dir)

    logger = setup_logger("manogcn", output_dir, get_rank())
    logger.info("Using {} GPUs".format(num_gpus))
    logger.info(args)

    logger.info("Collecting env info (might take some time)")
    from torch.utils.collect_env import get_pretty_env_info
    logger.info("\n" + get_pretty_env_info())

    logger.info("Loaded configuration file {}".format(args.config_file))
    with open(args.config_file, "r") as cf:
        config_str = "\n" + cf.read()
        logger.info(config_str)
    logger.info("Running with config:\n{}".format(cfg))

    output_config_path = os.path.join(output_dir, 'config.yml')
    logger.info("Saving config into: {}".format(output_config_path))
    # save overloaded model config in the output directory
    save_config(cfg, output_config_path)

    model = train(cfg, args.local_rank, args.distributed, output_dir)

if __name__ == "__main__":
    main()
