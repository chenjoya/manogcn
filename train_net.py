import argparse, os

import torch
from torch import optim, nn
import torch.distributed as dist

from manogcn.config import cfg
from manogcn.data import make_data_loader
from manogcn.engine.trainer import do_train
from manogcn.modeling import build_model
from manogcn.utils.checkpoint import Checkpointer
from manogcn.utils.logger import setup_logger
from manogcn.utils.miscellaneous import mkdir, save_config

def train(cfg, device_id, num_gpus, output_dir, logger):
    model = build_model(cfg).cuda(device_id)
    torch.backends.cudnn.benchmark = True
    
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
    
    arguments = {"epoch": 0}
    checkpointer = Checkpointer(
        cfg, model, optimizer, scheduler, output_dir
    )
    if cfg.MODEL.WEIGHT:
        extra_checkpoint_data = checkpointer.load(f=cfg.MODEL.WEIGHT, use_latest=False)
    else:
        extra_checkpoint_data = checkpointer.load(f=None, use_latest=True)
    arguments.update(extra_checkpoint_data)

    if num_gpus > 1:
        logger.info("Use PyTorch DDP training")
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[device_id]
        )
    
    logger.info("Prepare Datasets")
    data_loader = make_data_loader(
        cfg,
        is_train=True,
        is_distributed=num_gpus > 1,
    )
    test_period = cfg.SOLVER.TEST_PERIOD
    if test_period > 0:
        data_loader_val = make_data_loader(cfg, is_train=False, is_distributed=num_gpus > 1)
    else:
        data_loader_val = None
    logger.info("Finish Datasets Preparation")

    checkpoint_period = cfg.SOLVER.CHECKPOINT_PERIOD
    
    do_train(
        cfg,
        model,
        data_loader,
        data_loader_val,
        optimizer,
        scheduler,
        checkpointer,
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
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )
    args = parser.parse_args()
    args.num_gpus = int(os.environ["WORLD_SIZE"])
    args.device_id = int(os.environ["LOCAL_RANK"])
    
    if args.num_gpus > 1:
        torch.cuda.set_device(args.device_id)
        torch.distributed.init_process_group(backend="nccl")
    
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    model_name = os.path.splitext(os.path.basename(args.config_file))[0]
    output_dir = os.path.join("outputs", model_name)
    if output_dir:
        mkdir(output_dir)
    
    logger = setup_logger("manogcn", output_dir, args.device_id)
    logger.info("Using {} GPUs".format(args.num_gpus))
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

    model = train(cfg, args.device_id, args.num_gpus, output_dir, logger)

if __name__ == "__main__":
    main()
