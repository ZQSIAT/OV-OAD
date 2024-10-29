# -------------------------------------------------------------------------
# Rewritten by Qingsong Zhao
# -------------------------------------------------------------------------
import io
from re import L
import torch
import json
import os.path as osp
import requests
import numpy as np
import time
from typing import List
# from .base_dataset import BaseDataset
from torch.utils.data import Dataset
# from prototype.data.image_reader import build_image_reader
# from .image_reader import build_image_reader
# import linklink as link
import random
import os
import omegaconf
import clip
# from ipdb import set_trace
from .tokenizer import SimpleTokenizer
from .imagenet_template import full_imagenet_templates, kinetics_templates
from nltk.stem import WordNetLemmatizer
from PIL import Image
import pickle
lemmatizer = WordNetLemmatizer()
from petrel_client.client import Client
from collections import Counter
# import logging
from IPython import embed
# logger = logging.getLogger(__name__)
import spacy
from torchvision.transforms import PILToTensor


### frequently appeared 100 entities ###
TOP_CLASSES_1=[
    'people', 'man', 'men', 'woman', 'women', 'girl', 'boy', 'lady', 'kid', 'child', 'children', 'baby', 'student', 'bride', 'groom', 'couple', 'prince', 'princess', \
    'car', 'bus', 'truck', 'motorcycle', 'train', 'bicycle', 'boat', 'aeroplane', 'airplane', 'motorbike', 'bike',\
    'cup', 'bottle', 'bowl', 'knife', 'spoon',  'glass', 'fork',\
    'chair', 'table', 'bench', 'clock', 'laptop', 'light', 'vase', 'plant', 'remote', 'microwave', 'toaster', 'oven','mouse', 'keyboard','sofa', 'monitor','desk', 'tv','TV', 'couch', 'flower','refrigerator', \
    'house', 'building', 'hotel',\
    'handbag', 'umbrella','book', 'backpack', 'phone', 'shirt', 'tie', 'suitcase','T-shirt', 'bag',  'box', \
    'sink','bed','toilet',\
    'cat','dog',  'horse', 'bird','cow', 'sheep' ,'elephant', 'bear', 'zebra', 'giraffe', \
    'ball', 'racket', 'skateboard', 'skis', 'snowboard', 'surfboard', 'kite', \
    'pizza', 'cake', 'apple', 'banana', 'sandwich', 'orange', 'carrot', 'donut' ,\
]

### some of the entities are similar, map them to a single one ###
syn_dict = {
    'people':'people', 'man':'people', 'men':'people', 'woman':'people', 'women':'people', 'girl':'people', 'boy':'people', 'lady':'people', 'kid':'people', 'child':'people', 'children':'people', 'baby':'people', 'student':'people', 'bride':'people', 'groom':'people', 'couple':'people', 'prince':'people', 'princess':'people',\
    'airplane': 'aeroplane','motorbike': 'motorcycle','bike': 'bicycle',\
    'TV':'tv', 'desk': 'table', 'couch':'sofa',\
    'building': 'house', 'hotel': 'house', \
    'T-shirt': 'shirt','T-Shirt': 'shirt', 'handbag': 'bag', \
}

### unique entities ###
TOP_UNIQUE_CLASSES = [
    'people', 'car', 'bus', 'truck', 'motorcycle', \
    'train', 'bicycle', 'boat', 'aeroplane', 'cup', \
    'bottle', 'bowl', 'knife', 'spoon',  'glass', \
    'fork', 'chair', 'table', 'bench', 'clock', \
    'laptop', 'light', 'vase', 'plant', 'remote',\
    'microwave', 'toaster', 'oven','mouse', 'keyboard',\
    'sofa', 'monitor', 'tv', 'flower','refrigerator', \
    'house', 'bag', 'umbrella','book', 'backpack', \
    'phone', 'shirt', 'tie', 'suitcase', 'box',\
    'sink','bed','toilet', 'cat','dog', \
    'horse', 'bird','cow', 'sheep' ,'elephant', \
    'bear', 'zebra', 'giraffe',  'ball', 'racket', \
    'skateboard', 'skis', 'snowboard', 'surfboard', 'kite',\
    'pizza', 'cake', 'apple', 'banana', 'sandwich',\
    'orange', 'carrot', 'donut' ,\
]

TOP_UNIQUE_CLASSES_IDX = {}
for i, x in enumerate(TOP_UNIQUE_CLASSES):
    TOP_UNIQUE_CLASSES_IDX[x] = i

TOP200_UNIQUE_VERB_CLASSES = [
    'hold', 'stand', 'sit', 'wear', 'use', 'play', 'make', 'have', 'talk', 'cut', 'walk', 'look', 'speak', 'write', 'work', 'put', 'drive', 
    'eat', 'ride', 'do', 'animate', 'take', 'fly', 'fill', 'watch', 'run', 'clean', 'read', 'lay', 'point', 'pour', 'draw', 
    'get', 'dance', 'singe', 'open', 'perform', 'grow', 'prepare', 'cook', 'smile', 'cover', 'feature', 'hang', 'paint', 'go', 'remove', 'give', 'touch', 
    'fight', 'close', 'shoot', 'pick', 'tie', 'repair', 'build', 'park', 'wash', 'jump', 'set', 'pose', 'place', 'mix', 'dress', 'lie', 'reach', 'swim', 
    'fry', 'fix', 'drink', 'throw', 'come', 'carry', 'stir', 'hit', 'stuff', 'laugh', 'climb', 'attach', 'sleep',
    'fall', 'move', 'create', 'surround', 'slice', 'pull', 'sew', 'dig', 'plant',  'burn', 'kick', 'practice', 'launch',
    'push', 'adjust', 'blow', 'spray', 'learn', 'interview', 'type', 'feed', 'travel', 'shake', 'race', 'light', 'catch',  
    'hug', 'present', 'float', 'replace', 'chop', 'wrap', 'sell', 'fold', 'check', 'pet','sing', 'connect', 'instal', 'lead',
    'install', 'demonstrate', 'shape', 'brush', 'measure', 'wave', 'gather', 'win', 'cross', 'change', 'turn', 'celebrate', 'find', 'lift', 'buy', 
    'meet', 'break', 'raise', 'decorate', 'load', 'weld', 'discuss', 'bend', 'kiss', 'fish', 'add', 'sign', 'color', 'carve', 
    'skate', 'print', 'wait', 'seat', 'box', 'stick',  'glow', 'kneel', 'reveal', 'film', 'face', 'rise', 'shop', 'chat', 'call', 'teach', 'ski',
    'cry', 'stack', 'line', 'serve', 'attack', 'dry', 'shine', 'explain', 'knit','leave',
]

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

RAW_TOP_UNIQUE_VERB_CLASSES_500 = ['show', 'hold', 'stand', 'sit', 'wear', 'use', 'play', 'make', 'have', 'talk', 'cut', 'walk', 'look', 'speak', 'write', 'work', 'put', 'drive', 'say', 'eat', 'ride', 
    'do', 'animate', 'display', 'take', 'fly', 'fill', 'watch', 'run', 'clean', 'read', 'lay', 'point', 'pour', 'see', 'draw', 'get', 'dance', 'singe', 'open', 'perform',
    'grow', 'prepare', 'cook', 'smile', 'cover', 'feature', 'hang', 'paint', 'go', 'remove', 'give', 'touch', 'fight', 'close', 'shoot', 'pick', 'tie', 'repair', 'build',
    'park', 'wash', 'jump', 'set', 'pose', 'place', 'mix', 'dress', 'lie', 'reach', 'swim', 'fry', 'fix', 'drink', 'throw', 'come', 'try', 'contain', 'screenshot', 'be', 
    'carry', 'stir', 'hit', 'stuff', 'laugh', 'climb', 'attach', 'sleep', 'fall', 'move', 'create', 'surround', 'slice', 'pull', 'window', 'sew', 'dig', 'plant', 
    'broadcast', 'burn', 'kick', 'hd', 'practice', 'launch', 'push', 'apply', 'adjust', 'blow', 'spray', 'help', 'learn', 'interview', 'type', 'feed', 'travel', 'shake', 
    'race', 'light', 'catch', 'live', 'subscribe', 'hug', 'present', 'float', 'replace', 'chop', 'wrap', 'sell', 'fold', 'check', 'pet', 'thank', 'sing', 'connect', 
    'instal', 'lead', 'install', 'demonstrate', 'shape', 'brush', 'measure', 'wave', 'gather', 'win', 'cross', 'change', 'turn', 'celebrate', 'find', 'lift', 'buy', 
    'meet', 'break', 'raise', 'decorate', 'announce', 'load', 'weld', 'discuss', 'bend', 'kiss', 'fish', 'add', 'sign', 'color', 'carve', 'include', 'skate', 'print', 
    'wait', 'seat', 'box', 'stick', 'appear', 'glow', 'kneel', 'reveal', 'film', 'face', 'rise', 'shop', 'chat', 'call', 'teach', 'ski', 'cry', 'youtube', 'base', 
    'stack', 'line', 'stop', 'serve', 'attack', 'dry', 'shine', 'explain', 'knit', 'start', 'love', 'leave', 'train', 'kill', 'drill', 'report', 'clap', 'destroy', 
    'crash', 'highlight', 'inspect', 'hand', 'crochet', 'flow', 'vend', 'picture', 'trim', 'assemble', 'smoke', 'roll', 'charge', 'depict', 'arrest', 'press', 'receive',
    'want', 'lean', 'deliver', 'mount', 'promote', 'punch', 'swinge', 'enter', 'shave', 'grill', 'arrange', 'ask', 'water', 'test', 'keep', 'code', 'visit', 
    'flood', 'share', 'fire', 'know', 'slide', 'save', 'wrestle', 'damage', 'title', 'need', 'aim', 'warn', 'release', 'view', 'boil', 'protest', 'let', 'handle', 
    'sand', 'excel', 'abandon', 'pay', 'treat', 'bake', 'stare', 'exercise', 'pray', 'edit', 'tell', 'design', 'produce', 'stretch', 'skateboard', 'crouch', 
    'listen', 'dip', 'peel', 'sink', 'rescue', 'arrive', 'chase', 'grind', 'operate', 'hack', 'march', 'capture', 'lose', 'examine', 'pass', 'crowd', 'scoop',
    'wipe', 'clash', 'hide', 'address', 'style', 'enjoy', 'die', 'illuminate', 'cause', 'comb', 'record', 'automate', 'block', 'crawl', 'form', 'like',
    'grab', 'attend', 'harvest', 'think', 'unveil', 'curl', 'offer', 'become', 'plug', 'dive', 'follow', 'pile', 'miss', 'send', 'control', 'feel', 'vote',
    'prune', 'squat', 'surf', 'explode', 'polish', 'click', 'iron', 'locate', 'stitch', 'protect', 'smash', 'cheer', 'advertise', 'groom', 'whip', 
    'construct', 'plow', 'spread', 'describe', 'melt', 'injure', 'm', 'step', 'extend', 'rub', 'reflect', 'continue', 'trade', 'note', 'beat', 'spin', 
    'bear', 'steal', 'compete', 'cool', 'screenshote', 'knead', 'forget', 'insert', 'glue', 'top', 'pot', 'snowboard', 'blur', 'paddle', 'rest', 'score', 
    'collect', 'woodworke', 'massage', 'fence', 'kpop', 'wire', 'choose', 'drop', 'balance', 'scream', 'store', 'squeeze', 'graze', 'hike', 'frost', 'karate', 
    'braid', 'overlook', 'dump', "'", 'stripe', 'pack', 'attempt', 'bow', 'study', 'outbreak', 'land', 'sweep', 'dock', 'develop', 'overwatch', 'post', 'sail', 'embrace', 'salad',
    'yell', 'bring', 'weave', 'position', 'introduce', 'begin', 'process', 'lock', 'lick', 'bowl', 'drain', 'pump', 'crack', 'investigate', 'increase', 'mask', 'allow',
    'hammer', 'render', 'monitor', 'theme', 'pierce', 'stay', 'checker', 'improve', 'spot', 'organize', 'manufacture', 'state', 'squash', 'rip', 'enclose', 'ban', 'weigh',
    'dribble', 'shovel', 'swing', 'facebook', 'conduct', 'end', 'plan', 'paraglide', 'cast', 'join', 'embroider', 'tv', 'power', 'switch', 'reduce', 'hear', 'mow', 'shred',
    'list', 'greet', 'bathe', 'magnify', 'happen', 'scrub', 'floor', 'battle', 'hunt', 'rust', 'sauce', 'chew', 'select', 'star', 'rail', 'craft', 'download', 'indicate',
    'slam', 'bitcoin', 'tow', 'provide', 'frame', 'update', 'focus', 'count', 'support', 'milk', 'ai', 'suck', 'showcase', 'dunk', 'flash', 'avoid', 'translate', 'splash',
    'juggle', 'sle', 'bloom', 'kayak', 'search', 'calculate'
]


class VideoClipDataset(Dataset):
    """
    Video Clip Dataset.
    sample_meta = torch.load(io.BytesIO(memoryview(client.get(f"pvideo:s3://internvid_extc_feats_4fps/CLIP_ViT_B_16/wV2WNFpaBdw.pth"))))
    local_instance = "/mnt/petrelfs/heyinan/00_zqs/data/tmp/1ezrj3GQWG0.pth"
    Arguments:
        - root_dir (:obj:`str`): root directory of dataset
        - meta_file (:obj:`str`): name of meta file
        - transform (list of ``Transform`` objects): list of transforms
        - read_from (:obj:`str`): read type from the original meta_file
        - evaluator (:obj:`Evaluator`): evaluate to get metrics
        - osg_server (:obj:`str`): '10.198.3.28:30080/components/osg-default/v1'
        - topnoun: 'none' / 'coco_top50' / 'cc3m_top50' / ...
    Metafile example::
        "{"filename": "n01440764/n01440764_10026.JPEG", "label": 0, "label_name": "dog"}\n"
    """

    def __init__(self, 
                root_dir, 
                meta_file, 
                anno_file, 
                img_transform=None,
                text_transform=None,
                read_from='dir', 
                use_nvideos=-1,
                nonzero=1,
                enc_steps=128,
                long_term_steps=512, 
                dec_steps=8,
                evaluator=None,
                split='train',
                caption_pick='random',
                use_entity=False,
                mask_type='class',
                use_distilbert=True,
                verb_aug=False,
                verb_filter=False,
                use_saliency=False,
                use_enc_feat=False,
                saliency_steps=1,
                multi_label=3,
                use_image = False # whether to use image as inputs instead to feature
                ):
        super().__init__()
        assert caption_pick in ["random", "most_common", "txt_cat", "token_cat"], f"\"{caption_pick}\" not implement!"
        self.root_dir = root_dir
        self.data_name = root_dir.split('/')[2].split('_')[0]  # TODO !! root_dir should be 'pxxx:s3://{xxxx}_extc_feats_4fps/CLIP_ViT_B_16_768/'
        self.caption_pick = caption_pick
        self.training = split == 'train'
        self.meta_file = meta_file
        self.anno_file = anno_file
        self.enc_steps = enc_steps
        self.long_term_steps = long_term_steps
        self.dec_steps = dec_steps
        self.nonzero = nonzero
        self.read_from = read_from
        self.multi_label = multi_label
        self.img_transform = img_transform
        self.text_transform = text_transform
        self.evaluator = evaluator
        self.initialized = False
        self.split=split
        self.use_entity = use_entity
        # self.tokenizer = SimpleTokenizer()
        self.mask_type = mask_type
        self.use_distilbert = use_distilbert     
        self.use_saliency = use_saliency
        self.saliency_steps = saliency_steps
        self.use_enc_feat = use_enc_feat
        self.use_iamge = use_image

        self.metas = []
        self.client = Client("/mnt/petrelfs/zhaoqingsong/petreloss.conf")
        self.tokenizer = clip.tokenize

        ## load all sessions and targets
        
        if self.data_name == "internvid":
            with open(meta_file, 'r') as f: 
                self.sessions_all = json.load(f)
            ## read from local anno file and load all metafile info 
            target_all = pickle.load(open(self.anno_file, 'rb'))
        elif self.data_name == "anet": 
            with open(meta_file, 'r') as f: 
                data_info = json.load(f)['Anet']

            ## we use all train and val set samples 14950 videos for pretraining.
            train_session_set = data_info['train_session_set']
            test_session_set = data_info['test_session_set']
            self.sessions_all = train_session_set + test_session_set

            self.CLASSES = data_info["class_names"]

            ## we use all train and val set samples 14950 videos for pretraining.
            train_annos_dict = pickle.load(open(osp.join(self.anno_file, 'anet_annos_train.pickle'), 'rb')) 
            test_annos_dict = pickle.load(open(osp.join(self.anno_file, 'anet_annos_val.pickle'), 'rb')) 
            target_all = {**train_annos_dict, **test_annos_dict}
        else:
            raise NotImplementedError
        
        ## pick front N videos
        if use_nvideos > 0:
            get_sessions = self.sessions_all[:use_nvideos]
            # get_sessions = random.sample(self.sessions_all, use_nvideos)  # random choice n videos
        else:
            get_sessions = self.sessions_all


        self.inputs_feats = dict()
        self.inputs_captions = dict()
        self.inputs_images = dict()

        self.nlp = spacy.load("en_core_web_sm")  
        self.verb_aug = verb_aug
        self.verb_filter = verb_filter
        assert not (self.verb_filter and self.data_name == "anet"), f"ANet dataset can not use verb_filter"

        for session in get_sessions:
            try:
                ## load feats from s3 xxxxx.pth
                if self.read_from == 'petrel':
                    if use_image:
                        session_s3_path = osp.join(self.root_dir, f"v_{session}")
                        torch_images = self._read_frame(session_s3_path)
                        self.inputs_images[session] = torch_images

                    else:
                        session_s3_path = osp.join(self.root_dir, f"{session}.pth")
                        sample_meta = torch.load(io.BytesIO(self.client.get(session_s3_path)))
                        self.inputs_feats[session] = sample_meta['rgb']
                else:
                    raise NotImplementedError 
                if self.data_name == "internvid":
                    self.inputs_captions[session] = sample_meta['captions'] 
            except Exception as e:
                print(e)
                continue

            if self.data_name == "internvid":
                target = target_all[session]
            else:
                onehot_target = target_all[session]['anno']
                target = onehot_target.argmax(axis=1)

            vlen = target.shape[0]
            # seed = np.random.randint(self.enc_steps) if self.training else 0
            seed = 0

            if self.verb_filter:
                enc_clses = np.unique(target[np.nonzero(target)])
                ## 去除不包含verb的caption，生成新enc_target, enc_captions
                for cls in enc_clses:
                    caption  = list(self.inputs_captions[session][cls-1].values())[0][2]
                    if self.not_contains_verb(caption):
                        target[target == cls] = 0

            for start, end in zip(
                    range(seed, vlen, 1),
                    range(seed+self.enc_steps+self.long_term_steps, vlen-self.dec_steps, 1)):
                    # range(seed+self.enc_steps, vlen-self.dec_steps, 1)):
                # enc_target = target[start:end]
                enc_target = target[start+self.long_term_steps:end]  # for long term memory

                # dec_target = target[end:end+self.dec_steps]  ## todo, decoder's nonzero number must be set!!!
                
                if np.count_nonzero(enc_target) < self.nonzero:  ## zero means BG, we get 2 non zero frame at least!
                    continue
                
                # print(f"enc_target: {target[start:end]}, enc_clses: {enc_clses} '\n' enc_captions: {enc_captions}, enc_caption_mask: {enc_caption_mask}")

                self.metas.append([session, start, end, enc_target])
                # self.metas.append([session, start, end, enc_raw_caption, enc_caption_mask])

        # we have 106176 sessions, use 10000 videos, set enc nonzero: 8, get 2339673 metas
        # we have 106176 sessions, use 10000 videos, set enc nonzero: 8, get 3161088 metas 
        # we have 106176 sessions, use 5000 videos, set enc nonzero: 0, get 12424546 metas        
        # we have 14950 sessions, use all videos, set enc nonzero: 0, get 6435178 metas.
        print(f"we have {len(self.sessions_all)} sessions, use {use_nvideos} videos, set enc nonzero: {nonzero}, get {len(self.metas)} metas.")
        # embed()
        # exit()

    def __len__(self):        
        return len(self.metas)

    def _str2list(self, x):
        if type(x) is list:
            return x
        elif type(x) is str:
            return [x]
        else:
            raise RuntimeError(
                "unknown value for _str2list: {}".format(type(x)))

    def _load_cls_id(self, target):
        nonzero = np.nonzero(target)
        nonzero_indices = nonzero[0]
        nonzero_target = target[nonzero]
        if self.caption_pick == "random":
            if len(nonzero_target) > 0:
                enc_cls = random.choice(np.unique(nonzero_target))
                counter = Counter(nonzero_target)
                common_cls = counter.most_common(1)[0][0]
            else: 
                enc_cls = 0
                common_cls = 0
        # elif self.caption_pick == "most_common":
        #     counter = Counter(nonzero_target)
        #     return counter.most_common(1)[0][0]
        elif self.caption_pick == "txt_cat": # use the multi labels contrastive loss
            enc_cls = np.unique(nonzero_target)
            if len(enc_cls) > 0:
                counter = Counter(nonzero_target)
                common_cls = counter.most_common(1)[0][0]
            else:
                common_cls = 0
        else:
            raise NotImplementedError
        return {"enc_cls": enc_cls, "common_cls": common_cls, "nonzero_indices": nonzero_indices}
    
    def _read_frame(self, video_path, bound=None):
        # video_path = os.path.join(video_path, str(self.num_segments))
        # print(video_path)
        if os.path.exists(video_path):
            frame_list = [p for p in os.listdir(video_path)]
        else:
            frame_list = [p for p in self.client.list(video_path)]
            
        images_group = list()
        
        for frame_name in frame_list:
            if "s3://" in video_path:
                img_bytes = self.client.get(os.path.join(video_path, frame_name))
                img = Image.open(io.BytesIO(img_bytes))
            else:
                img = Image.open(os.path.join(video_path, frame_name))
            img = clip._transform(224)(img)
            # img = PILToTensor()(img).unsqueeze(0)
            img = img.float()
            # print(img.shape)  # 3, 224, 224
            images_group.append(img)
        # print(img.shape)
        # embed()
        # exit()
        torch_imgs = torch.stack(images_group)
        # print(torch_imgs.shape)  # [32, 3, 224, 224]
        return torch_imgs

    def get_anet_item(self, idx):
        session, start, end, enc_target = self.metas[idx]
        ret_info = dict()
        try:
            camera_inputs = self.inputs_feats[session][start:end]

            ## get raw captions
            # enc_cls = self._load_cls_id(enc_target)
            # dec_cls = self._load_cls_id(dec_target)
            parse_cls_dict = self._load_cls_id(enc_target)
            enc_cls = parse_cls_dict["enc_cls"]
            common_cls = parse_cls_dict["common_cls"]
            nonzero_indices = parse_cls_dict["nonzero_indices"]

            ## todo make sure correctness
            if self.caption_pick == "txt_cat":
                if len(enc_cls) == 0:
                    ## TODO offline model prediction
                    # [3 x "this is background"]
                    enc_raw_caption = [random.choice(kinetics_templates).format(self.CLASSES[0]) for _ in range(self.multi_label)] 
                    enc_caption_mask = np.zeros_like(enc_target) # TODO  np.zeros_like / np.ones_like ?
                    enc_caption_mask = torch.Tensor(enc_caption_mask)
                elif len(enc_cls) > 0:
                    # list_enc_cls = list(enc_cls)
                    enc_raw_caption = [random.choice(kinetics_templates).format(self.CLASSES[common_cls])]
                    # len_prompts = self.multi_label - len(enc_cls)
                    len_prompts = self.multi_label - 1
                    select_cls = np.random.choice(enc_cls, len_prompts, replace=True)
                    enc_raw_caption += [random.choice(kinetics_templates).format(self.CLASSES[i]) for i in select_cls]  # TODO here
                    # remove_common_cls = list_enc_cls.remove(common_cls)
                    # if remove_common_cls is not None:
                    # else:
                    #     enc_raw_caption += [random.choice(kinetics_templates).format(self.CLASSES[common_cls]) for _ in range(len_prompts)]

                    enc_caption_mask = np.zeros_like(enc_target)
                    enc_caption_mask[nonzero_indices] = 1
                    enc_caption_mask = torch.Tensor(enc_caption_mask)
                    # embed()
                    # exit()
            elif self.caption_pick == "random":
                if enc_cls > 0:
                    enc_raw_caption = random.choice(kinetics_templates).format(self.CLASSES[enc_cls])
                    enc_caption_mask = np.zeros_like(enc_target)
                    enc_caption_mask[enc_target == enc_cls] = 1
                    enc_caption_mask = torch.Tensor(enc_caption_mask)
                elif enc_cls == 0:
                    enc_raw_caption = "this is background"
                    enc_caption_mask = np.zeros_like(enc_target) # TODO  np.zeros_like / np.ones_like ?
                    enc_caption_mask = torch.Tensor(enc_caption_mask)
            else:
                raise NotImplementedError
            
            # dec_raw_caption = list(targets[dec_cls-1].values())[0][2]
            # enc_raw_caption = next(item[enc_cls] for item in targets if enc_cls in item)  # check it again

            if self.caption_pick == "txt_cat": # return tokenized captions [3, 77]
                enc_raw_caption = self.tokenizer(enc_raw_caption)
                # embed()
                # exit()
            if self.use_entity:  # only used in one caption setting 
                raw_question = random.choice(kinetics_templates)
                raw_answer = raw_question.format(self.CLASSES[common_cls])

                ret_info['raw_question'] = raw_question.format('[MASK]')
                ret_info['raw_answer'] = raw_answer

            if self.use_saliency or self.use_enc_feat:
                enc_saliency_caption = [random.choice(kinetics_templates).format(self.CLASSES[i]) for i in enc_target[-self.saliency_steps:]]
                ret_info['enc_saliency_caption'] = self.tokenizer(enc_saliency_caption)

            ret_info['camera_inputs'] = camera_inputs
            ret_info['enc_raw_caption'] = enc_raw_caption
            ret_info['enc_caption_mask'] = enc_caption_mask
            # ret_info['dec_raw_caption'] = dec_raw_caption
            return ret_info         
        except Exception as e:          
            print(e)

    def get_intern_item(self, idx):
        session, start, end, enc_target = self.metas[idx]
        ret_info = dict()
        try:
            # ## load feats from s3 xxxxx.pth
            # if self.read_from == 'petrel':
            #     session_s3_path = osp.join(self.root_dir, f"{session}.pth")
            #     sample_meta = torch.load(io.BytesIO(self.client.get(session_s3_path)))
            # else:
            #     raise NotImplementedError
            # camera_inputs = sample_meta['rgb'][start:end]
            camera_inputs = self.inputs_feats[session][start:end]

            ## get raw captions
            parse_cls_dict = self._load_cls_id(enc_target)
            enc_cls = parse_cls_dict["enc_cls"]
            common_cls = parse_cls_dict["common_cls"]
            nonzero_indices = parse_cls_dict["nonzero_indices"]

            # dec_cls = self._load_cls_id(dec_target)
            # targets = sample_meta['captions']  
            targets = self.inputs_captions[session]
            ## todo make sure correctness
            if self.caption_pick == "txt_cat":
                if len(enc_cls) > 0:
                    # enc_raw_caption = ", ".join([list(targets[i-1].values())[0][2] for i in enc_cls])
                    enc_raw_caption = [list(targets[i-1].values())[0][2] for i in enc_cls]
                    enc_caption_mask = np.zeros_like(enc_target)
                    enc_caption_mask[nonzero_indices] = 1
                    enc_caption_mask = torch.Tensor(enc_caption_mask)
                    # embed()
                    # exit()
                elif len(enc_cls) == 0:
                    enc_raw_caption = ["this is background"]
                    enc_caption_mask = np.zeros_like(enc_target) # TODO  np.zeros_like / np.ones_like ?
                    enc_caption_mask = torch.Tensor(enc_caption_mask)
            elif self.caption_pick == "random":
                if enc_cls > 0:
                    enc_raw_caption = list(targets[enc_cls-1].values())[0][2]
                    enc_caption_mask = np.zeros_like(enc_target)
                    enc_caption_mask[enc_target == enc_cls] = 1
                    enc_caption_mask = torch.Tensor(enc_caption_mask)
                elif enc_cls == 0:
                    enc_raw_caption = "this is background"
                    enc_caption_mask = np.zeros_like(enc_target) # TODO  np.zeros_like / np.ones_like ?
                    enc_caption_mask = torch.Tensor(enc_caption_mask)        
            else:
                raise NotImplementedError
            
            # dec_raw_caption = list(targets[dec_cls-1].values())[0][2]
            # enc_raw_caption = next(item[enc_cls] for item in targets if enc_cls in item)  # check it again
            # assert self.is_contains_chinese(", ".join([enc_raw_caption, dec_raw_caption])) == False

            ### for clip TextTransformer, captions are here tokenised ###
            ### for bert/distilbert, text transform are used to select nouns, captions will be tokensized later ###
            if self.caption_pick == "txt_cat":
                common_caption = list(targets[common_cls-1].values())[0][2] if common_cls > 0 else None # 使用出现次数最多的caption计算matching loss
                enc_raw_caption = self.text_transform(enc_raw_caption, common_caption) # 0：caption，1-2: other caption or prompts 

                if self.use_entity:
                    raw_question, raw_answer = self.extract_verb_phrases(common_caption)

            elif self.caption_pick == "random":
                if self.verb_aug:
                    enc_raw_caption = self.verb_phrases_aug(enc_raw_caption) # only used in one caption setting
                if self.use_entity:
                    raw_question, raw_answer = self.extract_verb_phrases(enc_raw_caption)
            else:
                raise NotImplementedError
            
            if self.use_entity:
                ret_info['raw_question'] = raw_question
                ret_info['raw_answer'] = raw_answer

            if self.use_saliency or self.use_enc_feat:
                enc_saliency_caption = [list(targets[i-1].values())[0][2] if i > 0 else "this is background" for i in enc_target[-self.saliency_steps:]]
                ret_info['enc_saliency_caption'] = self.tokenizer(enc_saliency_caption)

            ret_info['camera_inputs'] = camera_inputs
            ret_info['enc_raw_caption'] = enc_raw_caption
            ret_info['enc_caption_mask'] = enc_caption_mask
            # ret_info['dec_raw_caption'] = dec_raw_caption
            return ret_info         
        except Exception as e:          
            print(e)
    
    def get_anet_image_item(self, idx):
        session, start, end, enc_target = self.metas[idx]
        ret_info = dict()
        try:
            camera_inputs = self.inputs_images[session][start:end]
            parse_cls_dict = self._load_cls_id(enc_target)
            enc_cls = parse_cls_dict["enc_cls"]
            common_cls = parse_cls_dict["common_cls"]
            nonzero_indices = parse_cls_dict["nonzero_indices"]

            if self.caption_pick == "txt_cat":
                if len(enc_cls) == 0:
                    # [3 x "this is background"]
                    enc_raw_caption = [random.choice(kinetics_templates).format(self.CLASSES[0]) for _ in range(self.multi_label)] 
                    enc_caption_mask = np.zeros_like(enc_target) # TODO  np.zeros_like / np.ones_like ?
                    enc_caption_mask = torch.Tensor(enc_caption_mask)
                elif len(enc_cls) > 0:
                    # list_enc_cls = list(enc_cls)
                    enc_raw_caption = [random.choice(kinetics_templates).format(self.CLASSES[common_cls])]
                    # len_prompts = self.multi_label - len(enc_cls)
                    len_prompts = self.multi_label - 1
                    select_cls = np.random.choice(enc_cls, len_prompts, replace=True)
                    enc_raw_caption += [random.choice(kinetics_templates).format(self.CLASSES[i]) for i in select_cls]  # TODO here
                    # remove_common_cls = list_enc_cls.remove(common_cls)
                    # if remove_common_cls is not None:
                    # else:
                    #     enc_raw_caption += [random.choice(kinetics_templates).format(self.CLASSES[common_cls]) for _ in range(len_prompts)]

                    enc_caption_mask = np.zeros_like(enc_target)
                    enc_caption_mask[nonzero_indices] = 1
                    enc_caption_mask = torch.Tensor(enc_caption_mask)
                    # embed()
                    # exit()
            elif self.caption_pick == "random":
                if enc_cls > 0:
                    enc_raw_caption = random.choice(kinetics_templates).format(self.CLASSES[enc_cls])
                    enc_caption_mask = np.zeros_like(enc_target)
                    enc_caption_mask[enc_target == enc_cls] = 1
                    enc_caption_mask = torch.Tensor(enc_caption_mask)
                elif enc_cls == 0:
                    enc_raw_caption = "this is background"
                    enc_caption_mask = np.zeros_like(enc_target) # TODO  np.zeros_like / np.ones_like ?
                    enc_caption_mask = torch.Tensor(enc_caption_mask)
            else:
                raise NotImplementedError
            
            # dec_raw_caption = list(targets[dec_cls-1].values())[0][2]
            # enc_raw_caption = next(item[enc_cls] for item in targets if enc_cls in item)  # check it again

            if self.caption_pick == "txt_cat": # return tokenized captions [3, 77]
                enc_raw_caption = self.tokenizer(enc_raw_caption)

            if self.use_entity:  # only used in one caption setting 
                raw_question = random.choice(kinetics_templates)
                raw_answer = raw_question.format(self.CLASSES[common_cls])

                ret_info['raw_question'] = raw_question.format('[MASK]')
                ret_info['raw_answer'] = raw_answer

            if self.use_saliency or self.use_enc_feat:
                enc_saliency_caption = [random.choice(kinetics_templates).format(self.CLASSES[i]) for i in enc_target[-self.saliency_steps:]]
                ret_info['enc_saliency_caption'] = self.tokenizer(enc_saliency_caption)

            ret_info['camera_inputs'] = camera_inputs
            ret_info['enc_raw_caption'] = enc_raw_caption
            ret_info['enc_caption_mask'] = enc_caption_mask
            # ret_info['dec_raw_caption'] = dec_raw_caption
            return ret_info         
        except Exception as e:          
            print(e)



    def __getitem__(self, index):
        if self.data_name == "internvid": # lazy code 
            return self.get_intern_item(idx=index) 
        elif self.data_name == "anet":   
            # return self.get_anet_item(idx=index)
            return self.get_anet_image_item(idx=index)
        else:
            raise NotImplementedError
        
    def judge_noun(self, n):
        n = n.replace('.', '')
        # ans = n.split("'s")[0].split(',')[0]
        # ans = n.strip("'s").strip(",")
        ans = n
        ### conduct Lemmatization ###
        # ans = nlp(ans.lower())[0].lemma_
        ans = lemmatizer.lemmatize(ans.lower())
        
        if ans in syn_dict:
            ans = syn_dict[ans]
        
        if ans in TOP_UNIQUE_CLASSES:
            return 1, ans
        return 0, n      

    def not_contains_verb(self, caption):
        doc = self.nlp(caption)
        for token in doc:
            if token.pos_ == "VERB":
                return False
        return True
    
    def build_question_and_answer(self, caption, nouns=None):
        words = caption.split(' ')
        question = ''
        ans_list = []

        token_mapper = {}
        word_mapper = {}
        assert self.mask_type == 'class'
        for word in words:
            word_after = word
            word_flag, newword = self.judge_noun(word)
            if word_flag == 1:
                question = question + newword + ' '
                ans_list.append(newword)
                token_id = self.tokenizer.encode(newword)[0]
                token_mapper[token_id] = TOP_UNIQUE_CLASSES_IDX[newword]
                word_mapper[token_id] = 332   ### this is 'M'
            else:
                question = question + word + ' '
                    
        question = question.replace("'", '').strip()
        raw_question = question
        
        question, _, _, _ = self.text_transform(raw_question)
        question = torch.tensor([word_mapper[int(word)] if int(word) in word_mapper else word for word in question])
        # raw_answer = 'A photo of ' + ' and '.join(list(set(ans_list))) ## unique words
        raw_answer = random.choice(full_imagenet_templates).split('{}')[0] + ' and '.join(list(set(ans_list)))
        answer, _, _, _ = self.text_transform(raw_answer)
        
        return raw_question, question, raw_answer, answer


    def build_question_and_answer_for_distilbert(self, caption, nouns):
        words = caption.split(' ')
        question = ''
        entity_list = []

        ### default, mask all entites ###
        assert self.mask_type == 'class'
        for word in words:
            word_after = word
            word_flag, newword = self.judge_noun(word)
            if word_flag == 1:
                question = question + '[MASK]' + ' '
                entity_list.append(newword)
            else:
                question = question + word + ' '
    
        question = question.replace("'", '').strip()
        raw_question = question
        #### build and transform answers ###
        # raw_answer = 'A photo of ' + ' and '.join(list(set(ans_list))) ## unique words
        raw_answer = random.choice(full_imagenet_templates).split('{}')[0] + ' and '.join(list(set(entity_list)))    
        return raw_question, None, raw_answer, None

    def extract_verb_phrases(self, caption):
        doc = self.nlp(caption)
        question = ''
        entity_list = []
        ## default, mask only verbs
        if self.mask_type == 'verb':  # dobj
            for token in doc:
                word = token.text
                if token.pos_ == 'VERB' and token.lemma_ in TOP_UNIQUE_VERB_CLASSES:
                    question = question + '[MASK]' + ' '
                    entity_list.append(token.lemma_)
                else:
                    question = question + word + ' '
            raw_question = question.replace("'", '').strip()

        elif self.mask_type == 'dobj':  
            entity_list_ = []
            for token in doc:
                word = token.text
                if token.pos_ == 'VERB' and token.lemma_ in TOP_UNIQUE_VERB_CLASSES:
                    verb_phrase = token.lemma_
                    verb_phrase_ = [word]
                    for child in token.children:
                        if child.dep_ == 'dobj':  # 直接宾语
                            object_phrase = child.text
                            verb_phrase = " ".join([token.lemma_, object_phrase])
                            verb_phrase_ = [word, object_phrase]
                    entity_list.append(verb_phrase)
                    entity_list_ += verb_phrase_
            
            question = ' '.join([token.text if token.text not in entity_list_ else '[MASK]' for token in doc])
            raw_question = question.replace("'", '').replace(" ,", ',').strip()
        else:
            raise NotImplementedError
    
        if len(entity_list) > 0:
            raw_answer = random.choice(kinetics_templates).format(" and ".join(list(set(entity_list))))
        else:
            raw_answer = caption
        # embed()
        # exit()
        return raw_question, raw_answer
    

    def verb_phrases_aug(self, caption):
        doc = self.nlp(caption)
        entity_list = []
        ## default, mask only verbs
        if self.mask_type == 'verb':  # dobj
            for token in doc:
                if token.pos_ == 'VERB' and token.lemma_ in TOP_UNIQUE_VERB_CLASSES:
                    entity_list.append(token.lemma_)

        elif self.mask_type == 'dobj':  
            for token in doc:
                if token.pos_ == 'VERB' and token.lemma_ in TOP_UNIQUE_VERB_CLASSES:
                    verb_phrase = token.lemma_
                    for child in token.children:
                        if child.dep_ == 'dobj':  # 直接宾语
                            object_phrase = child.text
                            verb_phrase = " ".join([token.lemma_, object_phrase])
                    entity_list.append(verb_phrase)
        else:
            raise NotImplementedError
    
        if len(entity_list) > 0:
            return random.choice(kinetics_templates).format(" and ".join(list(set(entity_list))))
        else:
            return caption

    def is_contains_chinese(self, strs):
        for _char in strs:
            if '\u4e00' <= _char <= '\u9fa5':
                return True
        return False

    def _get_label_text(self, text):
        # label_text = ['a photo of ' + text + '.']
        if self.label_texts_ensemble == 'prompt6':
            f = f'{osp.abspath(os.getcwd())}/../../prototype/data/datasets/prompts/query_pattern_prompt6'
        elif self.label_texts_ensemble == 'prompt8':
            f = f'{osp.abspath(os.getcwd())}/../../prototype/data/datasets/prompts/query_pattern_prompt8'
        elif self.label_texts_ensemble == 'prompt80':
            f = f'{osp.abspath(os.getcwd())}/../../prototype/data/datasets/prompts/query_pattern_prompt80'
        elif self.label_texts_ensemble == 'cc':
            return [text]
        else:
            f = f'{osp.abspath(os.getcwd())}/../../prototype/data/datasets/prompts/query_pattern_prompt1'
        label_text = []
        with open(f) as fin:
            for line in fin.readlines():
                label_text.append(line.replace('{0}', text))
        return label_text

    def get_label_texts(self,):
        label_to_name = {}
        for curr_meta in self.metas:
            label = int(curr_meta['label']) if 'label' in curr_meta else None
            label_name = curr_meta['label_name'] if 'label_name' in curr_meta else None
            if label is not None and label_name is not None:
                label_to_name[label] = label_name
        labels = list(label_to_name.keys())
        labels.sort()

        label_texts = []
        label_text_len = []
        for label in labels:
            label_name = label_to_name[label]
            label_text = self._get_label_text(label_name)
            label_texts.extend(label_text)
            label_text_len.append(len(label_text))

        all_len = sum(label_text_len)
        offset = 0
        label_num = len(labels)
        label_texts_ensemble_matrix = torch.zeros(all_len, label_num)
        for lbl, ltl in enumerate(label_text_len):
            label_texts_ensemble_matrix[offset: offset + ltl, lbl] = 1
            offset += ltl

        return label_texts, label_texts_ensemble_matrix

    def dump(self, writer, output):
        filenames = output['filenames']
        image_ids = output['image_ids']
        label_names = output['label_names']
        captions = output['captions']
        tags = output['tags']
        prediction = self.tensor2numpy(output['prediction'])
        score = self.tensor2numpy(output['score'])
        labels = self.tensor2numpy(output['labels'])
        for _idx in range(len(filenames)):
            res = {
                'image_id': int(image_ids[_idx]),
                'filename': filenames[_idx],
                'label': int(labels[_idx]),
                'label_name': label_names[_idx],
                'caption': captions[_idx],
                'tag': tags[_idx],
                'prediction': int(prediction[_idx]),
                'score': [float('%.8f' % s) for s in score[_idx]]
            }
            writer.write(json.dumps(res, ensure_ascii=False) + '\n')
        writer.flush()