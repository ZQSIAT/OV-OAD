# -------------------------------------------------------------------------
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
# -------------------------------------------------------------------------
# Modified by Jilan Xu
# -------------------------------------------------------------------------

import mmcv
from mmseg.datasets import build_dataloader, build_dataset
from mmseg.datasets.pipelines import Compose
from omegaconf import OmegaConf
from utils import build_dataset_class_tokens, build_dataset_class_lists
from oad.datasets import *
from .group_vit_oad import GroupViTOadInference
from IPython import embed
import torch.nn as nn
import torch
import clip as CLIP


def build_oad_dataset(config):
    """Build a dataset from config."""
    assert len(config.cfg) == len(config.datasets), f"Please check settings !!"
    datasets = dict()
    for idx, dataset_name in enumerate(config.datasets):
        cfg = mmcv.Config.fromfile(config.cfg[idx])
        cfg.enc_steps = config.enc_steps
        cfg.single_eval = config.single_eval
        if dataset_name == "thomus":
            dataset = ThumosImgLoader(args=cfg, flag='test')
        datasets[dataset_name] = dataset
    return datasets


def build_oad_dataloaders(datasets):
    data_loaders = dict()
    assert type(datasets) is dict, f"We use multi val dataset !! "
    for k, v in datasets.items():
        dataset = v
        eval_batch_size = dataset.batch_size
        data_loader = build_dataloader(
            dataset,
            samples_per_gpu=eval_batch_size,  # batch size
            workers_per_gpu=4,  # num work
            dist=True,
            shuffle=False,
            persistent_workers=True,
            pin_memory=False)
        data_loaders[k] = data_loader
    return data_loaders

def build_oad_inference(model, dataset, text_transform, config, tokenizer=None):
    with_bg = False   
    if with_bg:
        classnames = dataset.CLASSES[1:]
    else:
        classnames = dataset.CLASSES
    
    if tokenizer is not None:
        text_tokens = build_dataset_class_lists(config.template, classnames)  # [str1, str2, str3, ...], len 20
        text_embedding = model.build_text_embedding(text_tokens, tokenizer, num_classes=len(classnames))  # torch.Size([20, 256])
    else:
        raise NotImplementedError
    
    kwargs = dict(with_bg=with_bg)

    saliency_text_embedding = None
    if config.use_saliency or config.use_enc_feat:
        ## class templates
        saliency_text_embedding = model.build_text_embedding_saliency_prediction(text_tokens, tokenizer)
    
    oad_model = GroupViTOadInference(model=model, 
                                    text_embedding=text_embedding,
                                    saliency_text_embedding=saliency_text_embedding,
                                    config=config,
                                    **kwargs)
    return oad_model

class LoadImage:
    """A simple pipeline to load image."""
    cnt = 0
    def __call__(self, results):
        """Call function to load images into results.

        Args:
            results (dict): A result dict contains the file name
                of the image to be read.

        Returns:
            dict: ``results`` will be returned containing loaded image.
        """

        if isinstance(results['img'], str):
            results['filename'] = results['img']
            results['ori_filename'] = results['img']
        else:
            results['filename'] = None
            results['ori_filename'] = None
        img = mmcv.imread(results['img'])
        results['img'] = img
        results['img_shape'] = img.shape
        results['ori_shape'] = img.shape
        return results

