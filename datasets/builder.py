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
# -------------------------------------------------------------------------
# Modified by Jilan Xu
# -------------------------------------------------------------------------


import os.path as osp
import random
import warnings
from functools import partial

import nltk
import spacy
import numpy as np
import torch
import torch.distributed as dist
import webdataset as wds
from braceexpand import braceexpand
from mmcv.parallel import collate
from timm.data import create_transform
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
import timm
if timm.__version__ == '0.6.12':
    from timm.data.transforms import str_to_pil_interp as _pil_interp
else:
    from timm.data.transforms import _pil_interp
# this works for timm==0.3.2
# from timm.data.transforms import _pil_interp 
from torchvision import transforms
import torch.nn as nn
from PIL import ImageFilter,Image
from torch import Tensor
from typing import Tuple, List, Optional
import numbers
import math
import torchvision.transforms.functional as F
import shutil

from .formatting import ToDataContainer
from .tokenizer import SimpleTokenizer
# from .clip_dataset import ClipDataset
from .video_clip_dataset import VideoClipDataset
# from ipdb import set_trace
from IPython import embed

def worker_init_fn(worker_id, num_workers, rank, seed):
    # The seed of each worker equals to
    # num_worker * rank + worker_id + user_seed
    worker_seed = num_workers * rank + worker_id + seed
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def collate_fn(batch):  
    camera_inputs = torch.stack([b['camera_inputs'] for b in batch])
    enc_raw_caption = [b['enc_raw_caption'] for b in batch] if type(batch[0]['enc_raw_caption']) is str else torch.stack([b['enc_raw_caption'] for b in batch]) # caption is string
    enc_saliency_caption = torch.stack([b['enc_saliency_caption'] for b in batch]) if 'enc_saliency_caption' in batch[0].keys() else None # ret_info['enc_saliency_caption'] [b['enc_saliency_caption'] for b in batch]
    # enc_raw_caption = torch.stack([b['enc_raw_caption'] for b in batch]) # tokenizerd captions, stack [3, 77] tensors
    # dec_raw_caption = [b['dec_raw_caption'] for b in batch] 
    enc_caption_mask = torch.stack([b['enc_caption_mask'] for b in batch]) 

    raw_question = [b['raw_question'] for b in batch] if 'raw_question' in batch[0].keys() else None
    raw_answer = [b['raw_answer'] for b in batch] if 'raw_answer' in batch[0].keys() else None
    
    return {    
        'camera_inputs':camera_inputs,
        'enc_raw_caption':enc_raw_caption,
        'enc_caption_mask': enc_caption_mask,
        'enc_saliency_caption': enc_saliency_caption,
        # 'dec_raw_caption' : dec_raw_caption,
        'raw_question': raw_question,
        'raw_answer': raw_answer,
    }

def build_pretrain_loader(config):
    local_rank = dist.get_rank() % torch.cuda.device_count() if dist.is_initialized() else 0

    dataset_train = build_videoclip_dataset(is_train=True, config=config)
    print(f'local rank {local_rank} / global rank {dist.get_rank()} \
        successfully build train dataset')
    # embed()
    # exit()
    sampler_train = torch.utils.data.DistributedSampler(dataset_train, shuffle=True)        
    print('train batch size: ', config.train.batch_size)
    data_loader_train = torch.utils.data.DataLoader(
        dataset_train,
        sampler=sampler_train,
        batch_size=config.train.batch_size,
        num_workers=config.num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=True,
        collate_fn=collate_fn, ### NOTEL THIS ###
        #shuffle=False,
    )
    return dataset_train, data_loader_train


def build_videoclip_dataset(is_train, config):
    # text_transform = None
    text_transform = build_text_transform(is_train, config.text_aug, config.with_dc)
    split = 'train' if is_train else 'val'

    dataset = VideoClipDataset(
        root_dir=config[split]['root_dir'],
        meta_file=config[split]['meta_file'],
        anno_file=config[split]['anno_file'],
        img_transform=None,  # we load feats tensor do not need img transform
        text_transform=text_transform,  # TODO: entity loss need aug from verbs not nouns
        read_from=config[split]['read_from'],
        evaluator=None, # no evaluator for now
        split=split,
        use_nvideos=config[split]['use_nvideos'],
        nonzero=config[split]['nonzero'],
        enc_steps=config[split]['enc_steps'],
        long_term_steps=config[split]['long_term_steps'],
        dec_steps=config[split]['dec_steps'],
        caption_pick=config[split]['caption_pick'],
        mask_type=config[split].get('mask_type', 'class'),
        use_entity=config[split]['use_entity'],
        verb_aug=config[split]['verb_aug'],
        verb_filter=config[split]['verb_filter'],
        use_saliency=config[split]['use_saliency'],
        use_enc_feat=config[split]['use_enc_feat'],
        saliency_steps=config[split]['saliency_steps'],
        use_distilbert=config[split].get('use_distilbert', True),
        multi_label=config.text_aug.multi_label,
        use_image = config[split]['use_image']
    )
    print('dataset len: ', len(dataset))
    return dataset


def build_img_transform(is_train, config, with_dc=True):
    if not config.deit_aug:
        if is_train:
            transform = transforms.Compose([
                transforms.RandomResizedCrop(config.img_size, scale=config.img_scale),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD)
            ])
        else:
            transform = transforms.Compose([
                transforms.Resize(config.img_size + 32),
                transforms.CenterCrop(config.img_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD)
            ])
        return transform

    if is_train:
        # this should always dispatch to transforms_imagenet_train
        transform = create_transform(
            input_size=config.img_size,
            is_training=True,
            color_jitter=config.color_jitter if config.color_jitter > 0 else None,
            auto_augment=config.auto_augment if config.auto_augment != 'none' else None,
            re_prob=config.re_prob,
            re_mode=config.re_mode,
            re_count=config.re_count,
            interpolation=config.interpolation,
        )
    else:
        size = int((256 / 224) * config.img_size)
        transform = transforms.Compose([
            transforms.Resize(size, interpolation=_pil_interp(config.interpolation)),
            transforms.CenterCrop(config.img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD)
        ])
    if with_dc:
        transform = transforms.Compose([*transform.transforms, ToDataContainer()])

    return transform

def build_text_transform(is_train, config, with_dc=True):
    local_rank = dist.get_rank() % torch.cuda.device_count() if dist.is_initialized() else 0
    if is_train:
        ### only on local rank 0 ###
        if local_rank == 0:
            ### download itself or pre-download and give the nltk dir ###
            # nltk.download('popular')
            nltk.data.path.append('/mnt/petrelfs/heyinan/00_zqs/code/ovoad/models/pretrained_models/nltk_data')


        ## TODO:  word_type must be verb!!! not noun!!!
        transform = WordAugTokenizeWrapper(
            Tokenize(SimpleTokenizer(), max_seq_len=config.max_seq_len),
            max_word=config.multi_label,
            word_type=config.word_type)
    else:
        transform = Tokenize(SimpleTokenizer(), max_seq_len=config.max_seq_len)
            
    if with_dc:
        transform = transforms.Compose([transform, ToDataContainer()])
    return transform
    
class Tokenize:

    def __init__(self, tokenizer, max_seq_len=77, truncate=True):
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.truncate = truncate

    def __call__(self, texts):
        expanded_dim = False
        if isinstance(texts, str):
            texts = [texts]
            expanded_dim = True
        
        sot_token = self.tokenizer.encoder['<|startoftext|>']
        eot_token = self.tokenizer.encoder['<|endoftext|>']
        all_tokens = [[sot_token] + self.tokenizer.encode(text) + [eot_token] for text in texts]
        result = torch.zeros(len(all_tokens), self.max_seq_len, dtype=torch.long)

        for i, tokens in enumerate(all_tokens):
            if len(tokens) > self.max_seq_len:
                if self.truncate:
                    tokens = tokens[:self.max_seq_len]
                    tokens[-1] = eot_token
                else:
                    raise RuntimeError(f'Input {texts[i]} is too long for context length {self.max_seq_len}')
            result[i, :len(tokens)] = torch.tensor(tokens)

        if expanded_dim:
            return result[0]

        return result

TOP_UNIQUE_VERB_CLASSES = [
    'hold', 'stand', 'sit', 'wear', 'play', 'make', 'cut', 'walk', 'write', 'put', 'drive', 
    'eat', 'ride', 'fly', 'fill',  'run', 'clean', 'lay', 'pour', 'draw', 
    'dance', 'singe', 'open', 'cook', 'hang', 'paint', 'go', 'give', 'touch', 
    'fight', 'shoot', 'pick', 'tie', 'repair', 'park', 'wash', 'jump', 'place', 'mix', 'dress', 'lie', 'swim', 
    'fry', 'fix', 'drink', 'throw', 'carry', 'stir', 'hit', 'stuff',  'climb',
    'fall', 'move', 'surround', 'slice', 'pull', 'sew', 'dig', 'plant',  'burn', 'kick', 'launch',
    'push', 'blow', 'spray', 'type', 'feed', 'shake', 'catch',  
    'hug', 'chop', 'wrap', 'fold', 'pet','sing', 'connect', 
    'brush', 'wave', 'cross', 'turn', 'celebrate', 'find', 'lift', 
    'break', 'raise', 'decorate', 'weld', 'bend', 'kiss', 'fish', 'sign', 'color', 'carve', 
    'skate', 'seat', 'box', 'stick',  'glow', 'kneel', 'reveal', 'film', 'rise', 'call', 'ski',
    'cry', 'stack', 'line', 'serve', 'attack', 'knit','leave',
    'drill', 'clap', 'destroy', 
    'crash', 'hand', 'crochet', 'flow', 'trim', 'assemble', 'smoke', 'roll', 'arrest', 'press', 
    'lean', 'deliver', 'mount', 'punch', 'swinge',
    'shave', 'grill', 'arrange', 'water', 
    'flood', 'slide', 'wrestle', 'damage', 'boil', 'protest', 
    'sand', 'bake', 'exercise', 'pray', 'stretch', 'skateboard', 'crouch', 
    'dip', 'peel', 'sink', 'chase', 'grind', 'march', 'capture', 'crowd', 'scoop',
    'wipe', 'clash',  'cause', 'comb', 'block', 'crawl',
    'grab', 'unveil', 'curl',  'plug', 'dive', 'follow', 'pile', 'vote',
    'prune', 'squat', 'surf', 'explode', 'polish', 'click', 'iron', 'stitch', 'smash', 'cheer', 'advertise', 'groom', 'whip', 
    'plow', 'melt', 'injure', 'step',  'rub',  'beat', 'spin', 
    'steal', 'knead', 'insert', 'glue', 'snowboard',  'paddle', 
    'collect', 'massage', 'drop', 'squeeze', 'graze', 'hike',  'karate', 
    'braid', 'dump', 'pack',  'bow', 'land', 'sweep', 'sail', 'embrace',
    'yell', 'lick', 'drain', 'pump', 'crack', 
    'hammer', 'pierce', 'squash', 'rip', 
    'shovel', 'swing',  'paraglide', 'cast', 'embroider',  'switch',   'mow', 'shred',
    'bathe',  'scrub', 'battle', 'chew', 'craft', 
    'slam', 'tow', 'suck', 'dunk',  'translate', 'splash',
    'juggle',  'bloom', 'kayak', 
]

class WordAugTokenizeWrapper:

    def __init__(self, tokenize, max_word=3, template_set='full', word_type='noun'):
        self.tokenize = tokenize
        self.max_word = max_word
        from .imagenet_template import (kinetics_templates, sub_kinetics_template, simple_kinetics_template,
                                        identity_template)
        assert template_set in ['full', 'subset', 'simple', 'identity']
        if template_set == 'full':
            templates = kinetics_templates
        elif template_set == 'subset':
            templates = sub_kinetics_template
        elif template_set == 'simple':
            templates = simple_kinetics_template
        elif template_set == 'identity':
            templates = identity_template
        else:
            raise ValueError
        self.templates = templates
        assert word_type in ['verb', 'dobj'] # TODO
        self.word_type = word_type 
        self.nlp = spacy.load("en_core_web_sm")  

    def get_tag(self, tokenized, tags):
        if not isinstance(tags, (list, tuple)):
            tags = [tags]
        ret = []
        for (word, pos) in nltk.pos_tag(tokenized):
            for tag in tags:
                if pos == tag:
                    ret.append(word)
        return ret
    
    def get_verbs(self, doc):
        ret = []
        for token in doc:
            word = token.text
            if token.pos_ == 'VERB' and token.lemma_ in TOP_UNIQUE_VERB_CLASSES:
                verb_phrase = word
                for child in token.children:
                    if child.dep_ == 'dobj':  # 直接宾语
                        object_phrase = child.text
                        verb_phrase = " ".join([word, object_phrase])
                ret.append(verb_phrase)
        return ret

    def get_tag_with_loc(self, tokenized, tags):
        if not isinstance(tags, (list, tuple)):
            tags = [tags]
        ret = []
        loc = []
        for i, (word, pos) in enumerate(nltk.pos_tag(tokenized)):
            for tag in tags:
                if pos == tag:
                    ret.append(word)
                    loc.append(i)
        return ret, loc
    
    def get_noun_phrase(self, tokenized):
        # Taken from Su Nam Kim Paper...
        grammar = r"""
            NBAR:
                {<NN.*|JJ>*<NN.*>}  # Nouns and Adjectives, terminated with Nouns

            NP:
                {<NBAR>}
                {<NBAR><IN><NBAR>}  # Above, connected with in/of/etc...
        """
        chunker = nltk.RegexpParser(grammar)

        chunked = chunker.parse(nltk.pos_tag(tokenized))
        continuous_chunk = []
        current_chunk = []

        for subtree in chunked:
            if isinstance(subtree, nltk.Tree):
                current_chunk.append(' '.join([token for token, pos in subtree.leaves()]))
            elif current_chunk:
                named_entity = ' '.join(current_chunk)
                if named_entity not in continuous_chunk:
                    continuous_chunk.append(named_entity)
                    current_chunk = []
            else:
                continue

        return continuous_chunk

    def __call__(self, texts, common_caption):
        """
        Args:
            text: List: [strs]
        
        """
        assert isinstance(texts, list)
        ### By default, we use 3 captions ###
        assert self.max_word >= 3, "By default, we use >= 3 captions"

        if len(texts) >= self.max_word:
            return self.tokenize(texts[:self.max_word])
        else:
            len_prompts = self.max_word - len(texts)
            captions = ", ".join(texts)
            doc = self.nlp(captions)
            # embed()
            # exit()
            verbs = []
            if self.word_type == 'verb':
                verbs = self.get_verbs(doc)
            else:
                raise ValueError('word_type must be verb or verb_phrase')
            
            prompt_texts = []
            if len(verbs) > 0:
                select_verbs = np.random.choice(verbs, len_prompts, replace=True)
                prompt_texts = [np.random.choice(self.templates).format(verb) for verb in select_verbs]
            else:
                prompt_texts += [random.choice(texts+["there is background"])] * len_prompts

            texts = texts + prompt_texts
            if common_caption in texts: # 将出现次数最多的caption放在首位！
                texts.remove(common_caption)  # 先从列表中移除重点元素
                texts.insert(0, common_caption)  # 将重点元素插入到列表的第一个位置
            return self.tokenize(texts) # [3, 77]
