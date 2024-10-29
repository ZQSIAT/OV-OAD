# -------------------------------------------------------------------------
# Swin Transformer
# Copyright (c) 2021 Microsoft
#
# MIT License
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE
#
# Written by Ze Liu, Zhenda Xie
# Modified by Jiarui Xu
# Modified by Jilan Xu
# Modified by Qingsong Zhao
# -------------------------------------------------------------------------

import argparse
import datetime
import os
import os.path as osp
import time
from collections import defaultdict
import subprocess
import time
import pickle
import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torch.distributed as dist
import torch.multiprocessing as mp
from datasets import build_text_transform, imagenet_classes, build_pretrain_loader
from mmcv.parallel import MMDistributedDataParallel
from mmcv.runner import get_dist_info, init_dist, set_random_seed
from mmcv.utils import collect_env, get_git_hash
from mmseg.apis import multi_gpu_test
from models import build_model
from omegaconf import OmegaConf, read_write
from oad.evaluation import build_oad_dataloaders, build_oad_dataset, build_oad_inference, build_single_frame_oad_inference
from oad.datasets import frame_level_map_n_cap
from timm.utils import AverageMeter, accuracy
from utils import (auto_resume_helper, build_dataset_class_tokens, build_optimizer, build_scheduler, data2cuda,
                   get_config, get_grad_norm, get_logger, load_checkpoint, parse_losses, reduce_tensor, save_checkpoint, momentum_update,
                   load_checkpoint_stage1, build_dataset_class_lists,cdist_,
                   )

from ipdb import set_trace
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from transformers import AutoTokenizer, RobertaTokenizer
from einops import rearrange
from IPython import embed
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import torch.nn.utils as nn_utils
import clip
# torch.autograd.set_detect_anomaly(True)


tokenizer_dict = {
    # 'Bert': AutoTokenizer.from_pretrained('distilbert-base-uncased', TOKENIZERS_PARALLELISM=False,),
    'Bert': AutoTokenizer.from_pretrained('/mnt/petrelfs/zhaoqingsong/code/ovoad/models/pretrained_models/bert-base-uncased', TOKENIZERS_PARALLELISM=False,),  # load tokenizer from local
    'TextTransformer': None,
    'CLIPTransformer': clip.tokenize,
}

def parse_args():
    parser = argparse.ArgumentParser('GroupViT training and evaluation script')
    parser.add_argument('--cfg', type=str, required=True, help='path to config file')
    parser.add_argument('--opts', help="Modify config options by adding 'KEY=VALUE' list. ", default=None, nargs='+')

    # easy config modification
    parser.add_argument('--batch-size', type=int, help='batch size for single GPU')
    parser.add_argument('--resume', help='resume from checkpoint')
    parser.add_argument(
        '--output', type=str, help='root of output folder, '
        'the full path is <output>/<model_name>/<tag>')
    parser.add_argument('--tag', type=str, help='tag of experiment')
    parser.add_argument('--eval', action='store_true', help='Perform evaluation only')
    parser.add_argument('--wandb', action='store_true', help='Use W&B to log experiments')
    parser.add_argument('--keep', type=int, help='Maximum checkpoint to keep')

    # distributed training
    parser.add_argument('--local_rank', type=int, required=False, default=0, help='local rank for DistributedDataParallel')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    args = parser.parse_args()

    return args


def check_max_map(mAPs, max_mAP):
    max_key = None
    exists_greater = False

    for key, value in max_mAP.items():
        if mAPs.get(key, float('-inf')) > value:
            max_key = key
            exists_greater = True
            break
    return [exists_greater, max_key]
    

def train(cfg):
    device = torch.device(cfg.device)
    if cfg.wandb and dist.get_rank() == 0:
        import wandb
        wandb.init(
            project='group_vit',
            name=osp.join(cfg.model_name, cfg.tag),
            dir=cfg.output,
            config=OmegaConf.to_container(cfg, resolve=True),
            resume=cfg.checkpoint.auto_resume)
    else:
        wandb = None
    # waiting wandb init
    dist.barrier()

    logger = get_logger()
    if dist.get_rank() == 0:
        writer = SummaryWriter(cfg.output)
    else:
        writer = None

    ############# set tokenizer ##############
    global tokenizer
    tokenizer = tokenizer_dict[cfg.model.text_encoder.type]

    # TODO: single dataset zero shot oad val loader
    # data_loader_oad = build_oad_dataloader(build_oad_dataset(cfg.evaluate.oad))
    # dataset_oad = data_loader_oad.dataset
    # print(f'Evaluating dataset: {dataset_oad}')

    # TODO: multi datasets zero shot oad val loaders
    datasets_oad = build_oad_dataset(cfg.evaluate.oad)
    data_loaders_oad = build_oad_dataloaders(datasets_oad)
    print(f'Evaluating dataset: {data_loaders_oad.keys()}')

    # a = datasets_oad['tvseries'].__getitem__(0)
    # embed()
    # exit()

    ## perform single frame eval
    if cfg.single_eval:
        for data_name, dataloader in data_loaders_oad.items():
            mAP, cmAP = single_frame_validate_oad(config=cfg, device=device, data_loader=dataloader, tokenizer=tokenizer)
            logger.info(f'mAP of the network on the {len(datasets_oad[data_name])} test instances: {mAP:.1f}%, {cmAP:.1f}%')
        return 0
    
    # logger.info(f'Creating model:{cfg.model.type}/{cfg.model_name}')
    model = build_model(cfg.model)
    # model.cuda()
    model = model.to(device)
    # logger.info(str(model))

    ## build loaders
    dataset_train, data_loader_train = build_pretrain_loader(cfg.data)
    print('Done pretrain loader')

    # dataset_train.__getitem__(206)
    # embed()
    # exit()

    optimizer = build_optimizer(cfg.train, model)
    scaler = torch.cuda.amp.GradScaler(enabled=True)  # True : fp16

    model = MMDistributedDataParallel(model, device_ids=[torch.cuda.current_device()], broadcast_buffers=False, find_unused_parameters=True)
    model_without_ddp = model.module
        
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f'number of params: {n_parameters}')

    # print(len(data_loader_train))

    lr_scheduler = build_scheduler(cfg.train, optimizer, len(data_loader_train))

    ##### load init params from stage 1 here, before auto resuming ######
    if cfg.checkpoint.stage1_checkpoint:
        load_checkpoint_stage1(cfg, model_without_ddp)

    if cfg.checkpoint.auto_resume:
        resume_file = auto_resume_helper(cfg.output)
        if resume_file:
            if cfg.checkpoint.resume:
                logger.warning(f'auto-resume changing resume file from {cfg.checkpoint.resume} to {resume_file}')
            with read_write(cfg):
                cfg.checkpoint.resume = resume_file
            logger.info(f'auto resuming from {resume_file}')
        else:
            logger.info(f'no checkpoint found in {cfg.output}, ignoring auto resume')

    ## init metrics
    max_mAP = dict()
    max_cmAP = dict()
    for dataname in datasets_oad.keys():
        max_mAP[dataname] = 0.0
        max_cmAP[dataname] = 0.0
    max_metrics = {'max_mAP': max_mAP, 'max_cmAP': max_cmAP}

    if cfg.checkpoint.resume:
        max_metrics = load_checkpoint(cfg, model_without_ddp, optimizer, lr_scheduler, scaler)
        max_mAP, max_cmAP = max_metrics['max_mAP'], max_metrics['max_cmAP']
    
    tensorbd_logdir = cfg.output + "/logs"
    logger.info('Start training')
    start_time = time.time()
    
    for epoch in range(cfg.train.start_epoch, cfg.train.epochs):
        ### train model ###
        loss_train_dict = train_one_epoch(cfg, device, model, data_loader_train, optimizer, scaler, epoch, lr_scheduler, writer)
        # if dist.get_rank() == 0 and (epoch % cfg.checkpoint.save_freq == 0 or epoch == (cfg.train.epochs - 1)):
        #     save_checkpoint(cfg, epoch, model_without_ddp, {
        #         'max_cmAP': max_cmAP,
        #         'max_mAP': max_mAP,
        #     }, optimizer, lr_scheduler, scaler)
        # dist.barrier()
        loss_train = loss_train_dict['total_loss']
        logger.info(f'Avg loss of the network on the {len(dataset_train)} train images: {loss_train:.2f}')
                
        # evaluate TODO: not eval mode
        if (epoch % cfg.evaluate.eval_freq == 0 or epoch == (cfg.train.epochs - 1)):
            if 'oad' in cfg.evaluate.task:
                # mAP, cmAP = validate_oad(cfg, data_loader_oad, model_without_ddp, epoch, writer, tokenizer=tokenizer)
                # logger.info(f'mAP of the network on the {len(dataset_oad)} test instances: {mAP:.1f}%, {cmAP:.1f}%')
                mAPs = dict()
                cmAPs = dict()
                for data_name, dataloader in data_loaders_oad.items():
                    logger.info(f'Evaluate OVOADor on {data_name} oad val set. ')
                    mAP, cmAP = validate_oad(cfg, device, dataloader, model_without_ddp, epoch, writer, tokenizer)
                    logger.info(f'mAP of the {data_name} on the {len(datasets_oad[data_name])} test instances: {mAP:.1f}%, {cmAP:.1f}%')
                    mAPs[data_name] = mAP
                    cmAPs[data_name] = cmAP
                    
                max_metrics['max_mAP'] = {key: max(max_mAP[key], mAPs[key]) for key in max_mAP} 
                max_metrics['max_cmAP'] = {key: max(max_cmAP[key], cmAPs[key]) for key in max_cmAP} 
                # max_metrics['max_mAP'] = max(max_metrics['max_mAP'], mAP)
                # exists_max_map = any(mAPs.get(key, float('-inf')) > value for key, value in max_mAP.items())
                # exists_max_cmap = any(cmAPs.get(key, float('-inf')) > value for key, value in max_cmAP.items())
                # exists_max_map, max_data_name = check_max_map(mAPs, max_mAP)
                # exists_max_cmap, max_key = check_max_map(cmAPs, max_cmAP)
                # if cfg.evaluate.oad.save_best and dist.get_rank() == 0 and mAP >= max_mAP:
                # if cfg.evaluate.oad.save_best and dist.get_rank() == 0 and exists_max_map:
                #     save_checkpoint(cfg, epoch, model_without_ddp, max_metrics, optimizer, lr_scheduler, scaler, suffix=f'{max_data_name}_best_map')

                if cfg.evaluate.oad.save_best and dist.get_rank() == 0:
                    for key, value in max_mAP.items():
                        if mAPs.get(key, float('-inf')) > value:
                            save_checkpoint(cfg, epoch, model_without_ddp, max_metrics, optimizer, lr_scheduler, scaler, suffix=f'{key}_best_map')
                dist.barrier()
                # embed()
                # exit()
                max_mAP = max_metrics['max_mAP']
                max_cmAP = max_metrics['max_cmAP']
                # logger.info(f'Max_mAP: {max_mAP:.4f}%, Max_cmAP: {max_cmAP:.4f}%.')
                logger.info(f'\n Max_mAP: {max_mAP}, \n Max_cmAP: {max_cmAP}.')

        if wandb is not None:  # default : False
            log_stat = {f'epoch/train_{k}': v for k, v in loss_train_dict.items()}
            log_stat.update({
                'epoch/val_acc1': acc1,
                'epoch/val_acc5': acc5,
                'epoch/val_loss': loss,
                'epoch/val_miou': miou,
                'epoch/epoch': epoch,
                'epoch/n_parameters': n_parameters
            })
            wandb.log(log_stat)

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    logger.info('Training time {}'.format(total_time_str))
    dist.barrier()
    # writer.flush()

def process_text(cfg, text_data):
    # embed()
    # exit()
    if cfg.model.text_encoder['type'] in ['DistilBert','Bert','BertMedium','Roberta']:
        ### we run all the exps with padding=True, meaning padding to the longest caption ###
        # text_data = tokenizer(text_data, return_tensors='pt', padding=True,
        #                         truncation=True, max_length=77)

        ### this is more memory friendly/load balance if we chunk the padding size to max_length ###
        text_data = tokenizer(text_data, return_tensors='pt', padding='max_length',
                            truncation=True, max_length=77)
    
        text_data = {key: val.cuda().contiguous() for key, val in text_data.items()}
    elif cfg.model.text_encoder['type'] in ['CLIPTransformer'] and type(text_data) is list:
        text_data = tokenizer(text_data, truncate=True, context_length=77).cuda() # [256, 77]
    else:   
        text_data = text_data.cuda()
    # embed()
    # exit()
    return text_data

                    
def generate_entity_masks(text_data):
    # embed()
    # exit()
    if type(text_data) is not dict:
        text = text_data
    else:
        text = text_data['input_ids'] # [256, 77]
    # [b, L]
    entity_masks = text.clone()
    entity_masks[entity_masks != 103] = 0
    entity_masks[entity_masks == 103] = 1 # [MASK]的位置设置为1 TODO CLIP tokenize时确认下
    
    entity_masks  = entity_masks.to(text.device)
    return entity_masks

def train_one_epoch(config, device, model, data_loader, optimizer, scaler, epoch, lr_scheduler, writer):
    logger = get_logger()
    dist.barrier()
    model.train()
    optimizer.zero_grad()
    if config.wandb and dist.get_rank() == 0:
        import wandb
    else:
        wandb = None

    num_steps = len(data_loader)
    batch_time = AverageMeter()
    loss_meter = AverageMeter()
    norm_meter = AverageMeter()
    log_vars_meters = defaultdict(AverageMeter)

    start = time.time()
    end = time.time()

    # text_transform = build_text_transform(False, config.data.text_aug, with_dc=False)
    
    for idx, samples in enumerate(data_loader):        
        batch_size = config.data.train.batch_size
        # all_images = samples['camera_inputs'].cuda()
        all_images = samples['camera_inputs'].to(device, non_blocking=True)
        enc_caption_mask = samples['enc_caption_mask'].to(device, non_blocking=True)
        all_questions = None
        enc_saliency_caption = None
        entity_labels = entity_masks =  None
        all_answers = None

        ## 默认只执行第一个if语句
        # if config.model.text_encoder['type'] in ['DistilBert','Bert','BertMedium','Roberta','CLIPTransformer']:
        # embed()
        # exit()

        all_texts = process_text(cfg=config, text_data=samples['enc_raw_caption']) # for multi captions text_data: [256, 3, 77]
        if config.data.train.use_entity is True:  # default: False
            all_questions = process_text(config, samples['raw_question'])
            all_answers= process_text(config, samples['raw_answer'])
            # embed()
            # exit()
            # entity_masks = generate_entity_masks(all_questions)
            entity_masks= None # not used
        if config.data.train.use_saliency or config.data.train.use_enc_feat:
            # embed()
            # exit()
            enc_saliency_caption = process_text(config, samples['enc_saliency_caption'])

        # else:
        #     raise NotImplementedError
        # elif config.model.text_encoder['type'] not in ['TextTransformer'] and config.data.train.use_entity is True:
        #     all_texts = samples['caption'].cuda()
        #     all_questions = samples['question'].cuda()
        #     all_answers = samples['answer'].cuda()
        # else:
        #     all_texts = samples['caption'].cuda()
        ### for cross-image mask consistency loss ###
        # all_crossimage = samples['cross_image'].cuda() if 'cross_image' in samples and samples['cross_image'] is not None else None
        # question_masks = samples['question_mask'].cuda() if 'question_mask' in samples else None
        # cross_entity = process_text(samples['cross_entity']) if 'cross_entity' in samples and samples['cross_entity'] is not None else None

        ### forward and compute loss ###
        # losses = model(image=all_images, text=all_texts, cross_image=all_crossimage, cross_entity=cross_entity, \
        #                 question=all_questions, answer=all_answers, entity_masks=entity_masks, question_masks=question_masks)
            
        ## TODO: default: only contras loss
        with torch.cuda.amp.autocast(enabled=True):  # fp16 
            # losses = model(image=all_images, text=all_texts, enc_targets=enc_caption_mask)
            losses = model(image=all_images, text=all_texts, enc_saliency_caption=enc_saliency_caption, question=all_questions, answer=all_answers, entity_masks=entity_masks, enc_targets=enc_caption_mask)
            # embed()
            # exit()
            loss, log_vars = parse_losses(losses)
        
        if dist.get_rank() == 0:
            writer.add_scalar("Total loss", loss, len(data_loader) * epoch + idx)
            writer.add_scalar("contrastive loss", losses['matching_loss'], len(data_loader) * epoch + idx)
            if 'entity' in losses:
                writer.add_scalar("entity loss", losses['entity_loss'], len(data_loader) * epoch + idx)
            if 'mask' in losses:
                writer.add_scalar("Mask loss", losses['mask_loss'], len(data_loader) * epoch + idx)
            writer.add_scalar("lr",  optimizer.param_groups[0]['lr'], len(data_loader) * epoch + idx)


        optimizer.zero_grad()
        # embed()
        # exit()
        scaler.scale(loss).backward()

        # # 裁剪梯度 TODO
        if config.train.clip_grad:
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.train.clip_grad)

        scaler.step(optimizer)
        scaler.update()
        # optimizer.step()
        lr_scheduler.step_update(epoch * num_steps + idx)
        # if config.model.use_maskloss:  # default: False
        #         maskloss_coeff = 0.99
        #         momentum_update(model.module.img_encoder, model.module.img_encoder_momentum, maskloss_coeff)

        torch.cuda.synchronize()
        loss_meter.update(loss.item(), batch_size)
        for loss_name in log_vars:
            log_vars_meters[loss_name].update(log_vars[loss_name], batch_size)
        norm_meter.update(grad_norm)
        batch_time.update(time.time() - end)
        end = time.time()

        if idx % config.print_freq == 0:
            lr = optimizer.param_groups[0]['lr']
            memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
            etas = batch_time.avg * (num_steps - idx)
            log_vars_str = '\t'.join(f'{n} {m.val:.4f} ({m.avg:.4f})' for n, m in log_vars_meters.items())
            logger.info(f'Train: [{epoch}/{config.train.epochs}][{idx}/{num_steps}]\t'
                        f'eta {datetime.timedelta(seconds=int(etas))} lr {lr:.6f}\t'
                        f'time {batch_time.val:.4f} ({batch_time.avg:.4f})\t'
                        f'total_loss {loss_meter.val:.4f} ({loss_meter.avg:.4f})\t'
                        f'{log_vars_str}\t'
                        f'grad_norm {norm_meter.val:.4f} ({norm_meter.avg:.4f})\t'
                        f'mem {memory_used:.0f}MB')
            if wandb is not None:
                log_stat = {f'iter/train_{n}': m.avg for n, m in log_vars_meters.items()}
                log_stat['iter/train_total_loss'] = loss_meter.avg
                log_stat['iter/learning_rate'] = lr
                wandb.log(log_stat)
        
        if config.debug:
            break

    epoch_time = time.time() - start
    logger.info(f'EPOCH {epoch} training takes {datetime.timedelta(seconds=int(epoch_time))}')
    result_dict = dict(total_loss=loss_meter.avg)
    for n, m in log_vars_meters.items():
        result_dict[n] = m.avg
    dist.barrier()
    return result_dict

@torch.no_grad()
def validate_oad(config, device, data_loader, model, epoch=0, writer=None, tokenizer=None):
    # device = torch.device(config.device)
    num_tasks = dist.get_world_size()
    logger = get_logger()
    dist.barrier()
    model.eval()

    batch_time = AverageMeter()

    if hasattr(model, 'module'):
        model_without_ddp = model.module
    else:
        model_without_ddp = model

    # text_transform传入build_oad_inference但不会使用
    text_transform = build_text_transform(False, config.data.text_aug, with_dc=False)  
    if config.model.text_encoder['type'] in ['DistilBert', 'Bert','BertMedium','Roberta', 'CLIPTransformer']:
        oad_model = build_oad_inference(model_without_ddp, data_loader.dataset, text_transform, config.evaluate.oad, tokenizer)
    else:
        oad_model = build_oad_inference(model_without_ddp, data_loader.dataset, text_transform, config.evaluate.oad)

    mmddp_model = MMDistributedDataParallel(
        oad_model, device_ids=[torch.cuda.current_device()], broadcast_buffers=True)
    mmddp_model.eval()

    end = time.time()
    logger.info('Zero shot oad model built!')
    
    all_probs, all_classes, all_fixed_saliency, all_online_saliency, all_enc_feat = [], [], [], [], []
    for idx, (camera_inputs, targets, saliency_current) in enumerate(data_loader):
        all_images = camera_inputs.to(device, non_blocking=True)  # torch.Size([B, 128, 512])
        # print("oad:", all_images.shape)
        target_val = targets.to(device, non_blocking=True) # torch.Size([B, 128, 21])
        saliency_current = saliency_current.to(device, non_blocking=True)
        # target_val  = target_val.view(-1, 21).contiguous()
        target_val = targets[:, -1, :].squeeze(1).to(device, non_blocking=True).contiguous().float() # get last frame

        with torch.cuda.amp.autocast(enabled=True):
            # logits_dict = mmddp_model(feats=all_images[:,-24:,:])
            logits_dict = mmddp_model(feats=all_images) # {"temporal": outputs, "saliency": saliency_logits} # B L C -> [256, 128, 21], B C -> [256, 21]
        logits = logits_dict["temporal"]
        saliency_logits = logits_dict["saliency"]
        enc_feat_logits = logits_dict["enc_feat_logits"]
        # embed()
        # exit()
        # target_val  = target_val.max(1).values
        # logits = logits.max(1).values
        # logits = logits.mean(1).contiguous()
        logits = logits[:, -1, :].squeeze(1).contiguous()
        # logits  = logits.view(-1, 21).contiguous()

        logits_gather_list = [torch.zeros_like(logits) for _ in range(num_tasks)]
        torch.distributed.all_gather(logits_gather_list, logits)
        logits = torch.cat(logits_gather_list, dim=0)

        targets_gather_list = [torch.zeros_like(target_val) for _ in range(num_tasks)]
        torch.distributed.all_gather(targets_gather_list, target_val)
        target_val = torch.cat(targets_gather_list, dim=0)

        saliency_gather_list = [torch.zeros_like(saliency_current) for _ in range(num_tasks)]
        torch.distributed.all_gather(saliency_gather_list, saliency_current)
        saliency_current = torch.cat(saliency_gather_list, dim=0)

        if saliency_logits is not None:
            saliency_logits_list = [torch.zeros_like(saliency_logits) for _ in range(num_tasks)]
            torch.distributed.all_gather(saliency_logits_list, saliency_logits)
            saliency_logits = torch.cat(saliency_logits_list, dim=0)
            all_online_saliency.extend(saliency_logits.detach().cpu().numpy())

        if enc_feat_logits is not None:
            enc_feat_logits_list = [torch.zeros_like(enc_feat_logits) for _ in range(num_tasks)]
            torch.distributed.all_gather(enc_feat_logits_list, enc_feat_logits)
            enc_feat_logits = torch.cat(enc_feat_logits_list, dim=0)
            all_enc_feat.extend(enc_feat_logits.detach().cpu().numpy())
        

        all_probs.extend(logits.detach().cpu().numpy())  
        all_classes.extend(target_val.detach().cpu().numpy())
        all_fixed_saliency.extend(saliency_current.detach().cpu().numpy())

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        if idx % config.print_freq == 0:
            memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
            logger.info(f'Test: [{idx}/{len(data_loader)}]\t'
                        f'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                        f'Mem {memory_used:.0f}MB')

    # embed()
    # exit()
    # targets = torch.argmax(torch.tensor(np.asarray(all_classes)), dim=1)
    # ce_loss = F.cross_entropy(torch.tensor(np.asarray(all_probs)).float(), targets)
    # logger.info(f"ce_loss: {ce_loss.item()}")

    all_probs = np.asarray(all_probs).T 
    logger.info(f"all_probs.shape {str(all_probs.shape)}") 
    all_classes = np.asarray(all_classes).T 
    logger.info(f"all_classes.shape {str(all_classes.shape)}")
    all_fixed_saliency = np.asarray(all_fixed_saliency).T 
    logger.info(f"all_fixed_saliency.shape {str(all_fixed_saliency.shape)}")

    mean_fusion = np.mean(np.stack([all_probs, all_fixed_saliency]), axis=0)

    if len(all_online_saliency) > 0:
        all_online_saliency = np.asarray(all_online_saliency).T 
        logger.info(f"all_online_saliency.shape {str(all_online_saliency.shape)}")
        mean_fusion = np.mean(np.stack([all_probs, all_fixed_saliency, all_online_saliency]), axis=0) # 27.2
        # mean_fusion = np.mean(np.stack([all_probs, all_online_saliency]), axis=0)

    if len(all_enc_feat) > 0:
        all_enc_feat = np.asarray(all_enc_feat).T 
        logger.info(f"all_enc_feat.shape {str(all_enc_feat.shape)}")
        mean_fusion = np.mean(np.stack([all_probs, all_fixed_saliency, all_enc_feat]), axis=0) # 23.3
        # mean_fusion = np.mean(np.stack([all_probs, 0.4*all_fixed_saliency, 0.1*all_enc_feat]), axis=0)  # 30.15->30.89
        # mean_fusion = np.mean(np.stack([1.5*all_probs, 0.2*all_fixed_saliency, 0.1*all_enc_feat]), axis=0)  # 35.3->36.3
        # mean_fusion = np.mean(np.stack([1.6*all_probs, 0.2*all_fixed_saliency, 0.1*all_enc_feat, 0.75*all_online_saliency]), axis=0)  # 35.3 -> 37.5
    
    results = {'probs': mean_fusion, 'labels': all_classes}
    # results = {'probs': all_probs, 'labels': all_classes}
    embed()
    exit()

    # measure map cmap
    metric = frame_level_map_n_cap(results=results, with_bg=False) # 不统计BG类别

    ap_nan_count = np.sum(np.isnan(metric['all_cls_ap']))
    # ap_mean_value = np.nanmean(metric['all_cls_ap'])
    cap_nan_count = np.sum(np.isnan(metric['all_cls_acp']))
    # cap_mean_value = np.nanmean(metric['all_cls_acp'])
    len_classes = len(data_loader.dataset.CLASSES)

    if ap_nan_count > 0:
        metric['map'] = np.nansum(metric['all_cls_ap']) / len_classes
    if cap_nan_count > 0:
        metric['cap'] = np.nansum(metric['all_cls_acp']) / len_classes

    for i, ap in enumerate(metric['all_cls_ap']):
        cls_name = data_loader.dataset.CLASSES[i+1] # with_bg=False
        logger.info('{}: {:.4f}'.format(cls_name, ap))

    # for i, cap in enumerate(metric['all_cls_acp']):
    #     cls_name = data_loader.dataset.CLASSES[i+1]
    #     logger.info('{}: {:.4f}'.format(cls_name, cap))
    # embed()
    # exit()
        
    mAP = metric['map']*100
    cmAP = metric['cap']*100
    logger.info('[Epoch-{}] [Dataset-{}] mAP: {:.2f}/% means map: {:.2f}/%\n'.format(epoch, data_loader.dataset.data_name, mAP, cmAP))
    # embed()
    # exit()
    out = [mAP, cmAP]
    torch.cuda.empty_cache()
    logger.info('Clearing zero shot classifier')
    if writer is not None and dist.get_rank() == 0:
        writer.add_scalar("mAP", mAP, epoch)
        writer.add_scalar("cmAP", cmAP, epoch)
    dist.barrier()
    return out


@torch.no_grad()
def validate_oad_results(config, data_loader, model, epoch=0, writer=None, tokenizer=None):
    device = torch.device(config.device)
    num_tasks = dist.get_world_size()
    logger = get_logger()
    dist.barrier()
    model.eval()

    batch_time = AverageMeter()

    if hasattr(model, 'module'):
        model_without_ddp = model.module
    else:
        model_without_ddp = model

    # text_transform传入build_oad_inference但不会使用
    text_transform = build_text_transform(False, config.data.text_aug, with_dc=False)  
    if config.model.text_encoder['type'] in ['DistilBert', 'Bert','BertMedium','Roberta', 'CLIPTransformer']:
        oad_model = build_oad_inference(model_without_ddp, data_loader.dataset, text_transform, config.evaluate.oad, tokenizer)
    else:
        oad_model = build_oad_inference(model_without_ddp, data_loader.dataset, text_transform, config.evaluate.oad)

    mmddp_model = MMDistributedDataParallel(
        oad_model, device_ids=[torch.cuda.current_device()], broadcast_buffers=True)
    mmddp_model.eval()
    end = time.time()
    logger.info('Zero shot oad model built!')

    all_probs, all_classes = [], []
    for idx, (camera_inputs, targets) in enumerate(data_loader):
        all_images = camera_inputs.to(device, non_blocking=True)  # torch.Size([B, 128, 512])
        target_val = targets.to(device, non_blocking=True) # torch.Size([B, 128, 21])
        # target_val  = target_val.view(-1, 21).contiguous()
        target_val = targets[:, -1, :].squeeze(1).to(device, non_blocking=True).contiguous() # get last frame

        logits = mmddp_model(feats=all_images)
        # embed()
        # exit()
        # target_val  = target_val.max(1).values
        # logits = logits.max(1).values
        # logits = logits.mean(1).contiguous()
        logits = logits[:, -1, :].squeeze(1).contiguous()
        # logits  = logits.view(-1, 21).contiguous()

        logits_gather_list = [torch.zeros_like(logits) for _ in range(num_tasks)]
        torch.distributed.all_gather(logits_gather_list, logits)
        logits = torch.cat(logits_gather_list, dim=0)
        targets_gather_list = [torch.zeros_like(target_val) for _ in range(num_tasks)]
        torch.distributed.all_gather(targets_gather_list, target_val)
        target_val = torch.cat(targets_gather_list, dim=0)
        all_probs.extend(logits.detach().cpu().numpy())  
        all_classes.extend(target_val.detach().cpu().numpy())

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        if idx % config.print_freq == 0:
            memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
            logger.info(f'Test: [{idx}/{len(data_loader)}]\t'
                        f'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                        f'Mem {memory_used:.0f}MB')

        # break # for debug
    # embed()
    # exit()
            
    # 将one-hot向量转换为类别索引的长整型张量
    targets = torch.argmax(torch.tensor(np.asarray(all_classes)), dim=1)

    # 计算损失
    ce_loss = F.cross_entropy(torch.tensor(np.asarray(all_probs)).float(), targets)

    # 打印损失
    logger.info(f"ce_loss: {ce_loss.item()}")
    all_probs = np.asarray(all_probs).T 
    logger.info(f"all_probs.shape {str(all_probs.shape)}") 
    import pickle
    with open('thumos_clip_b16_enc32_all_probs_saliency.pickle', 'rb') as file:
        saliency_probs = pickle.load(file)
    logger.info(f"pickle load thumos_clip_b16_enc32_all_probs_saliency.pickle succes!")
    mean_fusion = np.mean(np.stack([all_probs, saliency_probs]), axis=0)
    all_classes = np.asarray(all_classes).T 
    logger.info(f"all_classes.shape {str(all_classes.shape)}")
    results = {'probs': mean_fusion, 'labels': all_classes}
    torch.cuda.empty_cache()
    logger.info('Clearing zero shot classifier')
    dist.barrier()
    return results


@torch.no_grad() # trun to 00_vindlu/exp/zero-shot-oad/CLIP_ZS_THUMOS14_V10/zero_shot_oad_trainer.py
def single_frame_validate_oad(config, device, data_loader, epoch=0, writer=None, tokenizer=None):
    num_tasks = dist.get_world_size()
    logger = get_logger()
    dist.barrier()

    batch_time = AverageMeter()

    if config.model.text_encoder['type'] in ['DistilBert', 'Bert','BertMedium','Roberta', 'CLIPTransformer']:
        oad_model = build_single_frame_oad_inference(data_loader.dataset, config.evaluate.oad, tokenizer)
    else:
        raise NotImplementedError

    mmddp_model = MMDistributedDataParallel(
        oad_model, device_ids=[torch.cuda.current_device()], broadcast_buffers=True) 
    mmddp_model.eval()
    end = time.time()
    logger.info('Zero shot oad model built!')

    all_probs, all_classes = [], []
    for idx, (camera_inputs, targets) in enumerate(data_loader):
        # embed()
        # exit()
        if config.data.train.long_term_steps > 0:
            camera_inputs = camera_inputs[:,config.data.train.long_term_steps:,:].contiguous()

        if config.evaluate.oad.eval_type == "single":
            all_images = camera_inputs[:, -1, :].squeeze(1).to(device, non_blocking=True)  # torch.Size([B, 128, 512])
        elif "means" in config.evaluate.oad.eval_type:
            all_images = camera_inputs.to(device, non_blocking=True)
        else:
            raise NotImplementedError
        # target_val = targets.to(device, non_blocking=True) # torch.Size([B, 128, 21])
        # target_val  = target_val.view(-1, 21).contiguous()
        target_val = targets[:, -1, :].squeeze(1).to(device, non_blocking=True).contiguous().float() # get last frame

        with torch.cuda.amp.autocast(enabled=True): 
            logits = mmddp_model(all_images)

        # embed()
        # exit()

        if config.evaluate.oad.eval_type == "single":
            logits = logits.squeeze(1).contiguous()
        elif config.evaluate.oad.eval_type == "means":
            logits = logits.mean(1).contiguous()
        elif config.evaluate.oad.eval_type == "kmeans":
            logits = logits[:,-16:,:].mean(1).contiguous()
        else:
            raise NotImplementedError
        # logits  = logits.view(-1, 21).contiguous()


        logits_gather_list = [torch.zeros_like(logits) for _ in range(num_tasks)]
        torch.distributed.all_gather(logits_gather_list, logits)
        logits = torch.cat(logits_gather_list, dim=0)

        targets_gather_list = [torch.zeros_like(target_val) for _ in range(num_tasks)]
        torch.distributed.all_gather(targets_gather_list, target_val)
        target_val = torch.cat(targets_gather_list, dim=0)

        all_probs.extend(logits.detach().cpu().numpy())  
        all_classes.extend(target_val.detach().cpu().numpy())

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        if idx % config.print_freq == 0:
            memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
            logger.info(f'Test: [{idx}/{len(data_loader)}]\t'
                        f'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                        f'Mem {memory_used:.0f}MB')
    # embed()
    # exit()
    all_probs = np.asarray(all_probs).T 
    logger.info(f"all_probs.shape {str(all_probs.shape)}") 
    if config.single_eval:
        # save_name = f"models/{config.evaluate.oad.eval_type}_saliency_probs/{data_loader.dataset.data_name}_clip_b16_enc{config.data.train.enc_steps}_long{config.data.train.long_term_steps}_{data_loader.dataset.class_type}_probs_saliency.npy"
        # np.save(save_name, all_probs)
        # np.save('ek100_clip_b16_enc32_all_probs_saliency.npy', all_probs) 
        # /mnt/petrelfs/heyinan/00_zqs/code/ovoad/models/means_saliency_probs
        # /mnt/petrelfs/heyinan/00_zqs/code/ovoad/models/single_saliency_probs
        pass
    all_classes = np.asarray(all_classes).T 
    logger.info(f"all_classes.shape {str(all_classes.shape)}")
    results = {'probs': all_probs, 'labels': all_classes}

    # measure map cmap
    metric = frame_level_map_n_cap(results=results, with_bg=False) # with_bg=False 不统计BG类别

    ap_nan_count = np.sum(np.isnan(metric['all_cls_ap']))
    # ap_mean_value = np.nanmean(metric['all_cls_ap'])
    cap_nan_count = np.sum(np.isnan(metric['all_cls_acp']))
    # cap_mean_value = np.nanmean(metric['all_cls_acp'])
    len_classes = len(data_loader.dataset.CLASSES)
    if ap_nan_count > 0:
        metric['map'] = np.nansum(metric['all_cls_ap']) / len_classes
    if cap_nan_count > 0:
        metric['cap'] = np.nansum(metric['all_cls_acp']) / len_classes
    for i, ap in enumerate(metric['all_cls_ap']):
        cls_name = data_loader.dataset.CLASSES[i+1] # with_bg=False
        logger.info('{}: {:.4f}'.format(cls_name, ap))

    for i, cap in enumerate(metric['all_cls_acp']):
        cls_name = data_loader.dataset.CLASSES[i+1]
        logger.info('{}: {:.4f}'.format(cls_name, cap))
    # embed()
    # exit()

    mAP = metric['map']*100
    cmAP = metric['cap']*100
    torch.cuda.empty_cache()
    logger.info('[Epoch-{}] [Dataset-{}] mAP: {:.2f}/% means map: {:.2f}/%\n'.format(epoch, data_loader.dataset, mAP, cmAP))
    logger.info('Clearing zero shot classifier')
    if writer is not None and dist.get_rank() == 0:
        writer.add_scalar("mAP", mAP, epoch)
        writer.add_scalar("cmAP", cmAP, epoch)
    dist.barrier()
    return mAP, cmAP


@torch.no_grad()
def validate_seg(config, data_loader, model, epoch=0, writer=None, tokenizer=None):
    logger = get_logger()
    dist.barrier()
    model.eval()

    if hasattr(model, 'module'):
        model_without_ddp = model.module
    else:
        model_without_ddp = model

    text_transform = build_text_transform(False, config.data.text_aug, with_dc=False)
    if config.model.text_encoder['type'] in ['DistilBert', 'Bert','BertMedium','Roberta']:
        seg_model = build_seg_inference(model_without_ddp, data_loader.dataset, text_transform, config.evaluate.seg, tokenizer)
    else:
        seg_model = build_seg_inference(model_without_ddp, data_loader.dataset, text_transform, config.evaluate.seg)

    mmddp_model = MMDistributedDataParallel(
        seg_model, device_ids=[torch.cuda.current_device()], broadcast_buffers=False)
    mmddp_model.eval()
    results = multi_gpu_test(
        model=mmddp_model,
        data_loader=data_loader,
        tmpdir=None,
        gpu_collect=True,
        efficient_test=False,
        pre_eval=True,
        format_only=False)

    if dist.get_rank() == 0:
        metric = [data_loader.dataset.evaluate(results, metric='mIoU')]
    else:
        metric = [None]
    dist.broadcast_object_list(metric)
    miou_result = metric[0]['mIoU'] * 100

    torch.cuda.empty_cache()
    logger.info(f'Eval Seg mIoU {miou_result:.2f}')
    if writer is not None and dist.get_rank() == 0:
        writer.add_scalar("mIoU", miou_result, epoch)
    dist.barrier()
    return miou_result

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
    import random
    # launched with torch.distributed.launch
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.gpu = int(os.environ['LOCAL_RANK'])
    # launched with submitit on a slurm cluster
    elif 'SLURM_PROCID' in os.environ:
        proc_id = int(os.environ['SLURM_PROCID'])
        ntasks = os.environ['SLURM_NTASKS']
        node_list = os.environ['SLURM_NODELIST']
        num_gpus = torch.cuda.device_count()
        addr = subprocess.getoutput(
            'scontrol show hostname {} | head -n1'.format(node_list)
        )

        # master_port = os.environ.get('MASTER_PORT', str(29537+random.randint(0, 37)))
        master_port = os.environ.get('MASTER_PORT', '29547')
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
        os.environ['MASTER_PORT'] = '29537'
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
    # embed()
    # exit()
    args = parse_args()
    cfg = get_config(args)
    '''
    # start faster ref: https://github.com/open-mmlab/mmdetection/pull/7036
    mp.set_start_method('fork', force=True)
    init_dist('pytorch')
    rank, world_size = get_dist_info()
    print(f'RANK and WORLD_SIZE in environ: {rank}/{world_size}')
    dist.barrier()
    '''
    init_distributed_mode(args)
    rank, world_size = args.rank, args.world_size

    set_random_seed(cfg.seed, use_rank_shift=True)
    cudnn.benchmark = True

    os.makedirs(cfg.output, exist_ok=True)
    logger = get_logger(cfg)

    # linear scale the learning rate according to total batch size, may not be optimal
    linear_scaled_lr = cfg.train.base_lr * cfg.data.train.batch_size * world_size / 4096.0
    linear_scaled_warmup_lr = cfg.train.warmup_lr * cfg.data.train.batch_size * world_size / 4096.0
    linear_scaled_min_lr = cfg.train.min_lr * cfg.data.train.batch_size * world_size / 4096.0


    # gradient accumulation also need to scale the learning rate
    if cfg.train.accumulation_steps > 1:
        linear_scaled_lr = linear_scaled_lr * cfg.train.accumulation_steps
        linear_scaled_warmup_lr = linear_scaled_warmup_lr * cfg.train.accumulation_steps
        linear_scaled_min_lr = linear_scaled_min_lr * cfg.train.accumulation_steps

    with read_write(cfg):
        logger.info(f'Scale base_lr from {cfg.train.base_lr} to {linear_scaled_lr}')
        logger.info(f'Scale warmup_lr from {cfg.train.warmup_lr} to {linear_scaled_warmup_lr}')
        logger.info(f'Scale min_lr from {cfg.train.min_lr} to {linear_scaled_min_lr}')
        cfg.train.base_lr = linear_scaled_lr
        cfg.train.warmup_lr = linear_scaled_warmup_lr
        cfg.train.min_lr = linear_scaled_min_lr

    if dist.get_rank() == 0:
        path = os.path.join(cfg.output, 'config.json')
        OmegaConf.save(cfg, path)
        logger.info(f'Full config saved to {path}')

    # log env info
    env_info_dict = collect_env()
    env_info = '\n'.join([f'{k}: {v}' for k, v in env_info_dict.items()])
    dash_line = '-' * 60 + '\n'
    logger.info('Environment info:\n' + dash_line + env_info + '\n' + dash_line)

    logger.info(f'Git hash: {get_git_hash(digits=7)}')

    # print config
    logger.info(OmegaConf.to_yaml(cfg))

    train(cfg)
    dist.barrier()

if __name__ == '__main__':
    main()
