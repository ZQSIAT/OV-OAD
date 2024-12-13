# ------------------------------------------------------------------------------
# Copyright (c) 2021-2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# NVIDIA CORPORATION & AFFILIATES and its licensors retain all intellectual
# property and proprietary rights in and to this software, related
# documentation and any modifications thereto.  Any use, reproduction,
# disclosure or distribution of this software and related documentation
# without an express license agreement from NVIDIA CORPORATION is strictly
# prohibited.
#
# Written by Jiarui Xu
# ------------------------------------------------------------------------------
# Modified by Jilan Xu
# Modified by Qingsong Zhao
# -------------------------------------------------------------------------

import argparse
import os
import os.path as osp
import subprocess

import mmcv
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
from datasets import build_text_transform
from main_pretrain import validate_seg, validate_oad, single_frame_validate_oad
from mmcv.image import tensor2imgs
from mmcv.parallel import MMDistributedDataParallel
from mmcv.runner import set_random_seed
from models import build_model
from omegaconf import OmegaConf, read_write
import numpy as np
# from segmentation.evaluation import build_seg_dataloader, build_seg_dataset, build_seg_inference
from oad.evaluation import build_oad_dataloaders, build_oad_dataset, build_oad_inference
from utils import get_config, get_logger, load_checkpoint
from transformers import AutoTokenizer, RobertaTokenizer
from ipdb import set_trace
from IPython import embed
os.environ["TOKENIZERS_PARALLELISM"] = "false"
from oad.datasets import frame_level_map_n_cap
import clip
import pickle

tokenizer_dict = {
    # 'Bert': AutoTokenizer.from_pretrained('distilbert-base-uncased', TOKENIZERS_PARALLELISM=False,),
    'Bert': AutoTokenizer.from_pretrained('/mnt/petrelfs/xxx/xxxx/code/ovoad/models/pretrained_models/bert-base-uncased', TOKENIZERS_PARALLELISM=False,),  # load tokenizer from local
    # 'Roberta': RobertaTokenizer.from_pretrained('roberta-base'),
    'TextTransformer': None,
    'CLIPTransformer': clip.tokenize,
}



def parse_args():
    parser = argparse.ArgumentParser('GroupViT segmentation evaluation and visualization')
    parser.add_argument(
        '--cfg',
        type=str,
        required=True,
        help='path to config file',
    )
    parser.add_argument(
        '--opts',
        help="Modify config options by adding 'KEY VALUE' pairs. ",
        default=None,
        nargs='+',
    )

    parser.add_argument('--resume', help='resume from checkpoint')
    parser.add_argument(
        '--output', type=str, help='root of output folder, '
        'the full path is <output>/<model_name>/<tag>')
    parser.add_argument('--tag', help='tag of experiment')
    parser.add_argument(
        '--vis',
        help='Specify the visualization mode, '
        'could be a list, support input, pred, input_seg, input_pred_seg_label, all_groups, first_group, last_group',
        default=None,
        nargs='+')

    # distributed training
    parser.add_argument('--local_rank', type=int, required=False, default=0, help='local rank for DistributedDataParallel')

    args = parser.parse_args()

    return args


def inference(cfg):
    device = torch.device(cfg.device)

    logger = get_logger()

    # TODO: multi datasets zero shot oad val loaders
    datasets_oad = build_oad_dataset(cfg.evaluate.oad)
    data_loaders_oad = build_oad_dataloaders(datasets_oad)
    print(f'Evaluating dataset: {data_loaders_oad.keys()}')

    logger.info(f'Creating model:{cfg.model.type}/{cfg.model_name}')

    model = build_model(cfg.model)
    model = model.to(device)
    logger.info(str(model))

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f'number of params: {n_parameters}')

    load_checkpoint(cfg, model, None, None, None)
    
    global tokenizer
    tokenizer = tokenizer_dict[cfg.model.text_encoder.type]

    if cfg.model.text_encoder.type == 'Roberta':  # False
        raise NotImplementedError  # TODO
    
    if 'oad' in cfg.evaluate.task:
        ## multi eval datasets
        mAPs = dict()
        cmAPs = dict()
        for data_name, dataloader in data_loaders_oad.items():
            logger.info(f'Evaluate OVOADor on {data_name} oad val set. ')
            mAP, cmAP = validate_oad(cfg, device, dataloader, model, epoch=0, writer=None, tokenizer=tokenizer)
            logger.info(f'mAP of the {data_name} on the {len(datasets_oad[data_name])} test instances: {mAP:.1f}%, {cmAP:.1f}%')
            mAPs[data_name] = mAP
            cmAPs[data_name] = cmAP

        logger.info(f'\n mAPs: {mAPs}, \n cmAPs: {cmAPs}.')
    else:
        logger.info('No OAD evaluation specified')
        raise NotImplementedError

def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print

def init_distributed_mode(args):
    # launched with torch.distributed.launch
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.gpu = int(os.environ['LOCAL_RANK'])
    # launched with submitit on a slurm cluster
    elif 'SLURM_PROCID' in os.environ:
        #args.rank = int(os.environ['SLURM_PROCID'])
        #args.gpu = args.rank % torch.cuda.device_count()
        proc_id = int(os.environ['SLURM_PROCID'])
        ntasks = os.environ['SLURM_NTASKS']
        node_list = os.environ['SLURM_NODELIST']
        num_gpus = torch.cuda.device_count()
        addr = subprocess.getoutput(
            'scontrol show hostname {} | head -n1'.format(node_list)
        )
        master_port = os.environ.get('MASTER_PORT', '29499')
        os.environ['MASTER_PORT'] = master_port
        os.environ['MASTER_ADDR'] = addr
        os.environ['WORLD_SIZE'] = str(ntasks)
        os.environ['RANK'] = str(proc_id)
        os.environ['LOCAL_RANK'] = str(proc_id % num_gpus)
        os.environ['LOCAL_SIZE'] = str(num_gpus)
        args.dist_url = 'env://'
        args.world_size = int(ntasks)
        args.rank = int(proc_id)
        args.gpu = int(proc_id % num_gpus)
        print(f'SLURM MODE: proc_id: {proc_id}, ntasks: {ntasks}, node_list: {node_list}, num_gpus:{num_gpus}, addr:{addr}, master port:{master_port}' )
        
    # launched naively with `python main_dino.py`
    # we manually add MASTER_ADDR and MASTER_PORT to env variables
    elif torch.cuda.is_available():
        print('Will run the code on one GPU.')
        args.rank, args.gpu, args.world_size = 0, 0, 1
        os.environ['MASTER_ADDR'] = '127.0.0.1'
        os.environ['MASTER_PORT'] = '29500'
    else:
        print('Does not support training without GPU.')
        sys.exit(1)

    dist.init_process_group(
        backend="nccl",
        init_method=args.dist_url,
        world_size=args.world_size,
        rank=args.rank,
    )

    torch.cuda.set_device(args.gpu)
    print('| distributed init (rank {}): {}'.format(
        args.rank, args.dist_url), flush=True)
    dist.barrier()
    setup_for_distributed(args.rank == 0)

def main():
    args = parse_args()
    cfg = get_config(args)

    with read_write(cfg):
        cfg.evaluate.eval_only = True
    
    init_distributed_mode(args)
    rank, world_size = args.rank, args.world_size

    set_random_seed(cfg.seed, use_rank_shift=True) # True
    cudnn.benchmark = True

    os.makedirs(cfg.output, exist_ok=True)
    logger = get_logger(cfg)

    if dist.get_rank() == 0:
        path = os.path.join(cfg.output, 'config.json')
        OmegaConf.save(cfg, path)
        logger.info(f'Full config saved to {path}')


    # print config
    logger.info(OmegaConf.to_yaml(cfg))

    inference(cfg)
    dist.barrier()


if __name__ == '__main__':
    main()
