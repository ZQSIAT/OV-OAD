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
        # cfg.dec_steps = config.dec_steps
        # cfg.long_term_steps = config.long_term_steps
        # cfg.eval_type = config.eval_type # if you want cheange all dataset eval type here !! 
        cfg.single_eval = config.single_eval
        if dataset_name == "thomus":
            dataset = ThumosImgLoader(args=cfg, flag='test')
        elif dataset_name == "tvseries":
            dataset = TVSeriesImgLoader(args=cfg, flag='test')
        elif dataset_name == "epic":
            dataset = EK100ImgLoader(args=cfg, flag='test')
        elif dataset_name == "anet":
            dataset = ANetImgLoader(args=cfg, flag='test')
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

# def build_oad_dataloader(dataset):
#     eval_batch_size = dataset.batch_size
#     data_loader = build_dataloader(
#         dataset,
#         samples_per_gpu=eval_batch_size,  # batch size
#         workers_per_gpu=4,  # num work
#         dist=True,
#         shuffle=False,
#         persistent_workers=True,
#         pin_memory=False)
#     return data_loader

def build_oad_inference(model, dataset, text_transform, config, tokenizer=None):
    # cfg = mmcv.Config.fromfile(config.cfg)
    # if len(config.opts):  # default: False
    #     cfg.merge_from_dict(OmegaConf.to_container(OmegaConf.from_dotlist(OmegaConf.to_container(config.opts))))
    # with_bg = dataset.CLASSES[0] == 'background'
    with_bg = False   
    # embed()
    # exit()
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


# turn to 00_vindlu
class CLIPOadInference(nn.Module):
    def __init__(self, model, text_embedding):  
        super().__init__()
        self.model = model
        # [N, C]
        self.register_buffer('text_embedding', text_embedding)
        self.register_buffer('proj', self.model.visual.proj)
        # embed()
        # exit()

    def forward(self, image_features):
        image_features /= image_features.norm(dim=-1, keepdim=True)
        image_features = image_features @ self.proj
        logits = self.model.logit_scale.exp() * image_features @ self.text_embedding  
        logits = logits.softmax(dim=-1)
        return logits
    
# turn to 00_vindlu
class CLIPOadThumosInference(nn.Module):
    def __init__(self, model, text_embedding):  
        super().__init__()
        self.model = model
        # [N, C]
        self.register_buffer('text_embedding', text_embedding)
        # self.register_buffer('proj', self.model.visual.proj)
        # embed()
        # exit()

    def forward(self, image_features):
        image_features /= image_features.norm(dim=-1, keepdim=True)
        # image_features = image_features @ self.proj
        logits = self.model.logit_scale.exp() * image_features @ self.text_embedding  
        logits = logits.softmax(dim=-1)
        return logits

# turn to 00_vindlu
def zeroshot_classifier(model, texts, tokenizer):
    with torch.no_grad():
        zeroshot_weights = []
        # for classname in tqdm(classnames):
        for classname in texts:
            # text_data = tokenizer(classname, return_tensors='pt', padding='max_length',
            #                 truncation=True, max_length=77)
            # text_data = {key: val.cuda() for key, val in text_data.items()}
            # class_embeddings = model.encode_text(text_data['input_ids']) 
            texts = CLIP.tokenize(classname).cuda()
            class_embeddings = model.encode_text(texts)
            class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
            # embed()
            # exit()
            class_embedding = class_embeddings.mean(dim=0)
            class_embedding /= class_embedding.norm()
            zeroshot_weights.append(class_embedding)
        zeroshot_weights = torch.stack(zeroshot_weights, dim=1).cuda()
    return zeroshot_weights

# turn to 00_vindlu
def build_single_frame_oad_inference(dataset, config, tokenizer=None):
    # cfg = mmcv.Config.fromfile(config.cfg)
    # if len(config.opts):  # default: False
    #     cfg.merge_from_dict(OmegaConf.to_container(OmegaConf.from_dotlist(OmegaConf.to_container(config.opts))))
    # with_bg = dataset.CLASSES[0] == 'background'
    # classnames = dataset.CLASSES # mAP: 28.45/% means map: 93.34/%
    # classnames = all_class_name # 23.69/% means map: 90.72/%
    # embed()
    # exit()
    # with_bg = dataset.with_bg
    # if not with_bg:
    #     classnames = dataset.CLASSES[1:]
    # else:
    #     classnames = dataset.CLASSES
    classnames = dataset.CLASSES
    print(f"Using those {len(classnames)} classes: \n {classnames}.")

    models_name = "ViT-B/16"
    clip_model, preprocess = CLIP.load(models_name)

    if tokenizer is not None:
        texts = build_dataset_class_lists(config.template, classnames)  # [str1, str2, str3, ...], len 20
        text_embedding = zeroshot_classifier(clip_model, texts, tokenizer)  # [512, 21]
    else:
        raise NotImplementedError
    
    # embed()
    # exit()
    # if hasattr(dataset, 'out_dim768'):
    if not dataset.out_dim768:
        oad_model = CLIPOadThumosInference(clip_model, text_embedding)
    else:
        oad_model = CLIPOadInference(clip_model, text_embedding)

    print('Evaluate CLIP oad inference')
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


# def build_seg_demo_pipeline():
#     """Build a demo pipeline from config."""
#     img_norm_cfg = dict(mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
#     test_pipeline = Compose([
#         LoadImage(),
#         dict(
#             type='MultiScaleFlipAug',
#             img_scale=(2048, 448),
#             flip=False,
#             transforms=[
#                 dict(type='Resize', keep_ratio=True),
#                 dict(type='RandomFlip'),
#                 dict(type='Normalize', **img_norm_cfg),
#                 dict(type='ImageToTensor', keys=['img']),
#                 dict(type='Collect', keys=['img']),
#             ])
#     ])
#     return test_pipeline
