import numpy as np
import random
from copy import deepcopy
from scipy.ndimage import label
import os
import io
import os.path as osp
import subprocess
import time
from collections import defaultdict, deque
import datetime
import pickle
import json
from typing import Optional, List
import torch
import torch.distributed as dist
from torch.utils.data import Dataset
from PIL import Image
from IPython import embed
from petrel_client.client import Client
from torchvision.transforms import PILToTensor
# import models.backbones.clip as CLIP

# thumos classes
all_class_name = [
    "Background",
    "BaseballPitch",
    "BasketballDunk",
    "Billiards",
    "CleanAndJerk",
    "CliffDiving",
    "CricketBowling",
    "CricketShot",
    "Diving",
    "FrisbeeCatch",
    "GolfSwing",
    "HammerThrow",
    "HighJump",
    "JavelinThrow",
    "LongJump",
    "PoleVault",
    "Shotput",
    "SoccerPenalty",
    "TennisSwing",
    "ThrowDiscus",
    "VolleyballSpiking"]


## zero shot mode using only val sessions
class ANetImgLoader(Dataset):
    def __init__(self, args, flag='train'):
        super().__init__()
        assert flag in ['train', 'test', 'valid']
        self.data_name = 'anet'
        self.read_from = args.read_from
        self.anno_root = osp.join(args.anno_root, 'anno')  
        self.data_root = args.data_root
        self.nonzero = args.nonzero  # 128-->8
        self.training = flag == 'train'
        self.subnet = 'train' if self.training else 'val'
        self.single_eval = args.single_eval
        self.client = Client("/mnt/petrelfs/heyinan/petreloss.conf")
        self.inputs_feats = dict()
        self.class_type = args.class_type
        self.eval_type = args.eval_type

        ## get dataset info
        with open(args.dataset_file, 'r') as f:
            data_info = json.load(f)['Anet']

        self.ignore_index = data_info["ignore_index"] # 0 过滤backgorund
        self.with_bg = True if self.ignore_index != 0 else False # with_bg = True metas不过滤background， with_bg = False 过滤backgroud

        self.CLASSES = data_info["class_names"]

        args.train_session_set = data_info['train_session_set']
        args.test_session_set = data_info['test_session_set']

        self.sessions = getattr(args, flag + '_session_set')

        self.enc_steps = args.enc_steps
        self.dec_steps = args.dec_steps
        self.batch_size = args.batch_size

        self.inputs = []

        if not self.single_eval:
            saliency_npy_path = f"models/{self.eval_type}_saliency_probs/{self.data_name}_clip_b16_enc{self.enc_steps}_{self.class_type}_probs_saliency.npy" 
            if osp.exists(saliency_npy_path):
                self.saliency_npy_all = np.load(saliency_npy_path)
                print(f'load {saliency_npy_path}!')
            else:
                raise FileNotFoundError
        else:
            print(f'Let get the all_probs_saliency!')
    
        target_all = pickle.load(open(osp.join(self.anno_root, 'anet_annos_' + self.subnet + '.pickle'), 'rb')) 

        for session in self.sessions:
            try:
                ## load feats from s3 xxxxx.pth
                if self.read_from == 'petrel':
                    session_s3_path = osp.join(self.data_root, f"{session}.pth")
                    sample_meta = torch.load(io.BytesIO(self.client.get(session_s3_path)))
                else:
                    raise NotImplementedError
                self.inputs_feats[session] = sample_meta['rgb']
            except Exception as e:
                print(e)
                continue
            # embed()
            # exit()
            target = target_all[session]['anno']

            vlen = target.shape[0]
            seed = 0

            for start, end in zip(
                    range(seed, vlen, 1),
                    range(seed+self.enc_steps, vlen-self.dec_steps, 1)):
                enc_target = target[start:end]
                dec_target = target[end:end + self.dec_steps]
                class_enc_target = enc_target.argmax(axis=1)

                if np.count_nonzero(class_enc_target) >= self.nonzero and class_enc_target[-1] != self.ignore_index:
                    self.inputs.append([session, start, end, enc_target, dec_target])

            # if len(self.inputs) >= 2048: # for debug happy
            #     break

        print(f"we have {len(self.sessions)} sessions, set enc nonzero: {self.nonzero}, get {len(self.inputs)} metas.")
        # embed()
        # exit()

    
    ## if dataset is .pth files
    def getfeatstargets(self, index):
        # embed()
        # exit()
        session, start, end, enc_target, dec_target = self.inputs[index]
        camera_inputs = self.inputs_feats[session][start:end]
        # camera_inputs = torch.tensor(camera_inputs)
        enc_target = torch.tensor(enc_target)
        # dec_target = torch.tensor(dec_target)
        # print(camera_inputs.shape, enc_target.shape)
        return camera_inputs, enc_target,  # torch.Size([128, 512]) torch.Size([32, 31])

    ## if dataset is .pth files
    def getfeatstargetssaliency(self, index):
        session, start, end, enc_target, dec_target = self.inputs[index]
        camera_inputs = self.inputs_feats[session][start:end]
        # camera_inputs = torch.tensor(camera_inputs)
        enc_target = torch.tensor(enc_target)
        # dec_target = torch.tensor(dec_target)
        # print(camera_inputs.shape, enc_target.shape)
        saliency_current = self.saliency_npy_all[:,index]
        return camera_inputs, enc_target, saliency_current # torch.Size([128, 512]) torch.Size([128, 21])

    ## get item
    def __getitem__(self, index):
        if self.read_from == "petrel" and self.single_eval: # lazy code TODO
            return self.getfeatstargets(index=index) 
        elif self.read_from == "petrel" and not self.single_eval:    
            return self.getfeatstargetssaliency(index=index)
        else:
            raise NotImplementedError


    def __len__(self):
        return len(self.inputs)
        # return len(self.inputs[:2048]) # todo for happy debug, using 2048
    

## zero shot mode using only val sessions
class EK100ImgLoader(Dataset):
    def __init__(self, args, flag='train'):
        super().__init__()
        assert flag in ['train', 'test', 'valid']
        self.data_name = 'epic'
        self.read_from = args.read_from
        self.anno_root = osp.join(args.anno_root, 'anno')  
        self.data_root = args.data_root
        self.nonzero = args.nonzero  # 128-->8
        self.training = flag == 'train'
        self.subnet = 'train' if self.training else 'val'
        self.single_eval = args.single_eval
        self.client = Client("/mnt/petrelfs/heyinan/petreloss.conf")
        self.inputs_feats = dict()
        self.class_type = args.class_type
        # class_type = 'action_perframe' # noun_perframe, verb_perframe
        self.eval_type = args.eval_type

        ## get dataset info
        with open(args.dataset_file, 'r') as f:
            data_info = json.load(f)['EK100']

        if self.class_type == 'action_perframe':
            self.CLASSES = data_info["class_names"]
        elif self.class_type == 'verb_perframe':
            self.CLASSES = data_info["verb_class"]
        elif self.class_type == 'noun_perframe':
            self.CLASSES = data_info["noun_class"]
        else:
            raise NotImplementedError
        
        self.ignore_index = data_info["ignore_index"] # 0 过滤backgorund
        self.with_bg = True if self.ignore_index != 0 else False # with_bg = True metas不过滤background， with_bg = False 过滤backgroud
        # embed()
        # exit()

        args.train_session_set = data_info['train_session_set']
        args.test_session_set = data_info['test_session_set']

        self.sessions = getattr(args, flag + '_session_set')

        self.enc_steps = args.enc_steps
        self.dec_steps = args.dec_steps
        self.batch_size = args.batch_size

        self.inputs = []

        if not self.single_eval:
            saliency_npy_path = f"models/{self.eval_type}_saliency_probs/{self.data_name}_clip_b16_enc{self.enc_steps}_{self.class_type}_probs_saliency.npy" 
            if osp.exists(saliency_npy_path):
                self.saliency_npy_all = np.load(saliency_npy_path)
                print(f'load {saliency_npy_path}!')
            else:
                raise FileNotFoundError
        else:
            print(f'Let get the all_probs_saliency!')
    
        target_all = pickle.load(open(osp.join(self.anno_root, 'epic_annos_' + self.subnet + '.pickle'), 'rb')) # epic_annos_val.pickle

        for session in self.sessions:
            try:
                ## load feats from s3 xxxxx.pth
                if self.read_from == 'petrel':
                    session_s3_path = osp.join(self.data_root, f"{session}.pth")
                    sample_meta = torch.load(io.BytesIO(self.client.get(session_s3_path)))
                else:
                    raise NotImplementedError
                self.inputs_feats[session] = sample_meta['rgb']
            except Exception as e:
                print(e)
                continue
            # embed()
            # exit()
            target = target_all[session][self.class_type]

            vlen = target.shape[0]
            seed = 0

            for start, end in zip(
                    range(seed, vlen, 1),
                    range(seed+self.enc_steps, vlen-self.dec_steps, 1)):
                enc_target = target[start:end]
                dec_target = target[end:end + self.dec_steps]
                class_enc_target = enc_target.argmax(axis=1)

                if np.count_nonzero(class_enc_target) >= self.nonzero and class_enc_target[-1] != self.ignore_index:
                    self.inputs.append([session, start, end, enc_target, dec_target])

        print(f"we have {len(self.sessions)} sessions, set enc nonzero: {self.nonzero}, get {len(self.inputs)} metas.")
        # embed()
        # exit()

    
    ## if dataset is .pth files
    def getfeatstargets(self, index):
        # embed()
        # exit()
        session, start, end, enc_target, dec_target = self.inputs[index]
        camera_inputs = self.inputs_feats[session][start:end]
        # camera_inputs = torch.tensor(camera_inputs)
        enc_target = torch.tensor(enc_target)
        # dec_target = torch.tensor(dec_target)
        # print(camera_inputs.shape, enc_target.shape)
        return camera_inputs, enc_target,  # torch.Size([128, 512]) torch.Size([32, 31])

    ## if dataset is .pth files
    def getfeatstargetssaliency(self, index):
        session, start, end, enc_target, dec_target = self.inputs[index]
        camera_inputs = self.inputs_feats[session][start:end]
        # camera_inputs = torch.tensor(camera_inputs)
        enc_target = torch.tensor(enc_target)
        # dec_target = torch.tensor(dec_target)
        # print(camera_inputs.shape, enc_target.shape)
        saliency_current = self.saliency_npy_all[:,index]
        return camera_inputs, enc_target, saliency_current # torch.Size([128, 512]) torch.Size([128, 21])

    ## get item
    def __getitem__(self, index):
        if self.read_from == "petrel" and self.single_eval: # lazy code TODO
            return self.getfeatstargets(index=index) 
        elif self.read_from == "petrel" and not self.single_eval:    
            return self.getfeatstargetssaliency(index=index)
        else:
            raise NotImplementedError


    def __len__(self):
        return len(self.inputs)
        # return len(self.inputs[:2048])  # todo for happy debug, using 2048
    

## zero shot mode using only val sessions
class TVSeriesImgLoader(Dataset):
    def __init__(self, args, flag='train'):
        super().__init__()
        assert flag in ['train', 'test', 'valid']
        self.data_name = 'tvseries'
        self.read_from = args.read_from
        self.anno_root = osp.join(args.anno_root, 'anno')  
        self.data_root = args.data_root
        self.nonzero = args.nonzero  # 128-->8
        self.training = flag == 'train'
        self.subnet = 'train' if self.training else 'val'
        self.single_eval = args.single_eval
        self.client = Client("/mnt/petrelfs/heyinan/petreloss.conf")
        self.inputs_feats = dict()
        self.class_type = args.class_type
        self.eval_type = args.eval_type
        self.out_dim768 = True

        ## get dataset info
        with open(args.dataset_file, 'r') as f:
            data_info = json.load(f)['TVSeries']

        self.ignore_index = data_info["ignore_index"] # 0 过滤backgorund
        self.with_bg = True if self.ignore_index != 0 else False # with_bg = True metas不过滤background， with_bg = False 过滤backgroud

        args.train_session_set = data_info['train_session_set']
        args.test_session_set = data_info['test_session_set']

        self.sessions = getattr(args, flag + '_session_set')

        self.enc_steps = args.enc_steps
        self.long_term_steps = args.long_term_steps
        self.dec_steps = args.dec_steps
        self.batch_size = args.batch_size

        self.inputs = []
        # embed()
        # exit()
        if not self.single_eval:
            saliency_npy_path = f"models/{self.eval_type}_saliency_probs/{self.data_name}_clip_b16_enc{self.enc_steps}_long{self.long_term_steps}_{self.class_type}_probs_saliency.npy" 
            if osp.exists(saliency_npy_path):
                self.saliency_npy_all = np.load(saliency_npy_path)
                print(f'load {saliency_npy_path}!')
            else:
                raise FileNotFoundError
        else:
            print(f'Let get the all_probs_saliency!')
    
        target_all = pickle.load(open(osp.join(self.anno_root, 'tvseries_annos_' + self.subnet + '.pickle'), 'rb'))

        for session in self.sessions:
            try:
                ## load feats from s3 xxxxx.pth
                if self.read_from == 'petrel':
                    session_s3_path = osp.join(self.data_root, f"{session}.pth")
                    sample_meta = torch.load(io.BytesIO(self.client.get(session_s3_path)))
                else:
                    raise NotImplementedError
                self.inputs_feats[session] = sample_meta['rgb']
            except Exception as e:
                print(e)
                continue
            # embed()
            # exit()
            target = target_all[session]['anno']
            vlen = target.shape[0]
            seed = 0

            for start, end in zip(
                    range(seed, vlen, 1),
                    # range(seed+self.enc_steps, vlen-self.dec_steps, 1)):
                    range(seed+self.enc_steps+self.long_term_steps, vlen-self.dec_steps, 1)):
                # enc_target = target[start:end]
                enc_target = target[start+self.long_term_steps:end]

                dec_target = target[end:end + self.dec_steps]
                class_enc_target = enc_target.argmax(axis=1)

                if np.count_nonzero(class_enc_target) >= self.nonzero and self.ignore_index not in class_enc_target:
                    self.inputs.append([session, start, end, enc_target, dec_target])

        print(f"we have {len(self.sessions)} sessions, set enc nonzero: {self.nonzero}, get {len(self.inputs)} metas.")
        # embed()
        # exit()

    CLASSES = ["Background", "Pick something up", "Point", "Drink", "Stand up", "Run", "Sit down", "Read", "Smoke", "Drive car", "Open door", "Give something",
        "Use computer", "Write", "Go down stairway", "Close door", "Throw something", "Go up stairway", "Get in/out of car", "Hang up phone", "Eat", "Answer phone",
        "Dress up", "Clap", "Undress", "Kiss", "Fall/trip", "Wave", "Pour", "Punch", "Fire weapon"]
    
    ## if dataset is .pth files
    def getfeatstargets(self, index):
        # embed()
        # exit()
        session, start, end, enc_target, dec_target = self.inputs[index]
        camera_inputs = self.inputs_feats[session][start:end]
        # camera_inputs = torch.tensor(camera_inputs)
        enc_target = torch.tensor(enc_target)
        # dec_target = torch.tensor(dec_target)
        # print(camera_inputs.shape, enc_target.shape)
        return camera_inputs, enc_target,  # torch.Size([128, 512]) torch.Size([32, 31])

    ## if dataset is .pth files
    def getfeatstargetssaliency(self, index):
        session, start, end, enc_target, dec_target = self.inputs[index]
        camera_inputs = self.inputs_feats[session][start:end]
        # camera_inputs = torch.tensor(camera_inputs)
        enc_target = torch.tensor(enc_target)
        # dec_target = torch.tensor(dec_target)
        # print(camera_inputs.shape, enc_target.shape)
        saliency_current = self.saliency_npy_all[:,index]
        return camera_inputs, enc_target, saliency_current # torch.Size([128, 512]) torch.Size([128, 21])

    ## get item
    def __getitem__(self, index):
        if self.read_from == "petrel" and self.single_eval: # lazy code TODO
            return self.getfeatstargets(index=index) 
        elif self.read_from == "petrel" and not self.single_eval:    
            return self.getfeatstargetssaliency(index=index)
        else:
            raise NotImplementedError


    def __len__(self):
        return len(self.inputs)
        # return len(self.inputs[:10240])  # todo for happy debug, using 2048
    

## zero shot mode using only val sessions
class ThumosImgLoader(Dataset):
    def __init__(self, args, preprocess=None, flag='train'):
        super().__init__()
        assert flag in ['train', 'test', 'valid']
        self.data_name = 'thumos'
        self.read_from = args.read_from
        self.anno_root = osp.join(args.data_root, 'anno')  
        self.pickle_root = osp.join(args.data_root, 'pickles')
        self.nonzero = args.nonzero  # 128-->8
        self.training = flag == 'train'
        self.subnet = 'val' if self.training else 'test'
        self.single_eval = args.single_eval
        self.class_type = args.class_type
        self.out_dim768 = args.out_dim768
        self.eval_type = args.eval_type
        ## eval class
        # self.novel5 = [0, 3, 9, 20, 10, 16]

        ## get dataset info
        with open(args.dataset_file, 'r') as f:
            data_info = json.load(f)['THUMOS']

        self.ignore_index = data_info["ignore_index"] # 0 过滤backgorund
        self.with_bg = True if self.ignore_index != 0 else False # with_bg = True metas不过滤background， with_bg = False 过滤backgroud

        args.train_session_set = data_info['train_session_set']
        args.test_session_set = data_info['test_session_set']

        self.sessions = getattr(args, flag + '_session_set')

        self.enc_steps = args.enc_steps
        self.long_term_steps = args.long_term_steps
        self.dec_steps = args.dec_steps
        self.batch_size = args.batch_size

        self.inputs = []
        model_name = args.models_name.replace('-','_').replace('/','_')

        if not self.single_eval:
            saliency_npy_path = f"models/{self.eval_type}_saliency_probs/{self.data_name}_clip_b16_enc{self.enc_steps}_long{self.long_term_steps}_{self.class_type}_probs_saliency.npy" 
            if osp.exists(saliency_npy_path):
                self.saliency_npy_all = np.load(saliency_npy_path)
                print(f'load {saliency_npy_path}!')
            else:
                raise FileNotFoundError
        else:
            print(f'Let get the all_probs_saliency!')

        if self.read_from == "pickle":
            if self.out_dim768:
                pickle_path = osp.join(self.pickle_root, args.feature_type, f"thumos_all_feature_{flag}_{model_name}_d768.pickle")
            else:
                pickle_path = osp.join(self.pickle_root, args.feature_type, f"thumos_all_feature_{flag}_{model_name}.pickle")
            if osp.exists(pickle_path):
                self.feature_all = pickle.load(open(pickle_path, 'rb'))
                print(f'load {pickle_path}!')
            else:
                raise FileNotFoundError
        elif self.read_from == "jpg":
            self.transform = preprocess  # transforms.Compose([transforms.ToTensor(),])
            self.imgs_root = osp.join(args.data_root, self.subnet)
            self.img_format = "img{:05d}.jpg"
        else:
            raise NotImplementedError

        target_all = pickle.load(open(osp.join(self.anno_root, 'thumos_' + self.subnet + '_anno.pickle'), 'rb'))
        for session in self.sessions:
            target = target_all[session]['anno']
            # seed = np.random.randint(self.enc_steps) if self.training else 0
            seed = 0  # for dist.all_reduce, seed set at 0
            for start, end in zip(
                    range(seed, target.shape[0], 1),
                    range(seed+self.enc_steps+self.long_term_steps, target.shape[0]-self.dec_steps, 1)):
                    # range(seed + self.enc_steps, target.shape[0]-self.dec_steps, 1)):
                    # range(seed + self.enc_steps, target.shape[0], 1)):  # target.shape[0] self.long_term_steps
                # enc_target = target[start:end]
                enc_target = target[start+self.long_term_steps:end]
                
                dec_target = target[end:end + self.dec_steps]
                class_enc_target = enc_target.argmax(axis=1)
                # if 21 not in class_enc_target: #  class_enc_target[-1] != self.ignore_index:
                if np.count_nonzero(class_enc_target) >= self.nonzero and self.ignore_index not in class_enc_target:
                    self.inputs.append([session, start, end, enc_target, dec_target])
                # if class_enc_target[-1] in self.novel5:
                #     self.inputs.append([session, start, end, enc_target, dec_target])
                # if class_enc_target[-1] != 21: 
                #     self.inputs.append([session, start, end, enc_target, dec_target])

        print(f"we have {len(self.sessions)} sessions, set enc nonzero: {self.nonzero}, get {len(self.inputs)} metas.")
        
    CLASSES = [
        "background",
        "baseball pitch",
        "basketball dunk",
        "billiards",
        "clean and jerk",
        "cliff diving",
        "cricket bowling",
        "cricket shot",
        "diving",
        "frisbee catch",
        "golf swing",
        "hammer throw",
        "high jump",
        "javelin throw",
        "long jump",
        "pole vault",
        "shotput",
        "soccer penalty",
        "tennis swing",
        "throw discus",
        "volleyball spiking",
        ]

    ## if dataset is .pickle files
    def getfeatstargets(self, index):
        session, start, end, enc_target, dec_target = self.inputs[index]
        camera_inputs = self.feature_all[session]['rgb'][start:end]
        # camera_inputs = torch.tensor(camera_inputs)
        enc_target = torch.tensor(enc_target)
        # dec_target = torch.tensor(dec_target)
        # targets = torch.cat([enc_target, dec_target], dim=0)  # [9, 22] targets[:,:-1]
        # print(camera_inputs.shape, enc_target.shape)
        return camera_inputs, enc_target[:,:-1],  # torch.Size([128, 512]) torch.Size([128, 21])

    ## if dataset is .pickle files
    def getfeatstargetssaliency(self, index):
        session, start, end, enc_target, dec_target = self.inputs[index]
        camera_inputs = self.feature_all[session]['rgb'][start:end]
        # camera_inputs = torch.tensor(camera_inputs)
        enc_target = torch.tensor(enc_target)
        dec_target = torch.tensor(dec_target)
        # targets = torch.cat([enc_target, dec_target], dim=0)  # [9, 22] targets[:,:-1]
        # print(camera_inputs.shape, enc_target.shape)
        saliency_current = self.saliency_npy_all[:,index]
        return camera_inputs, enc_target[:,:-1], saliency_current # torch.Size([128, 512]) torch.Size([128, 21])

    ## if dataset is .pickle files, get (features, txts)
    def getfeatstxts(self, index):
        session, start, end, enc_target, dec_target = self.inputs[index]
        camera_inputs = self.feature_all[session]['rgb'][start:end]
        camera_inputs = torch.tensor(camera_inputs)
        enc_target = torch.tensor(enc_target[-1:])  # last frame is current frame
        dec_target = torch.tensor(dec_target)
        targets = torch.cat([enc_target, dec_target], dim=0)[:,:21]   # [9, 22]
        targets_index = torch.argmax(targets, dim=1)
        if self.template_random:
            template = random.choice(kinetics_templates)  # todo using template?
        else:
            template = kinetics_templates[0]  # "A photo of action {}."
        texts = [template.format(all_class_name_[i]) for i in targets_index]
        ## tokenize the text caption
        text_input = self.tokenizer(texts)  # [9, 77] 
        ## retturn
        if self.training:
            return camera_inputs, text_input  # torch.Size([64, dim]), len 9
        else:  ## 22 -> 11 cls names for zero shot eval
            targets = targets[:, self.noval_class]
            return camera_inputs, targets
        
    ## if dataset is .png files
    def getimgstargets(self, index):
        session, start, end, enc_target, dec_target = self.inputs[index]
        imgs_path = osp.join(self.imgs_root, session)
        imgs = []
        for i in range(start, end):
            i_path = osp.join(imgs_path, self.img_format.format(i))
            img = clip._transform(224)(Image.open(i_path))
            # img = PILToTensor()(img).unsqueeze(0)
            img = img.float()
            imgs.append(img)
        camera_inputs = torch.stack(imgs)
        enc_target = torch.tensor(enc_target[-1:])
        dec_target = torch.tensor(dec_target)
        targets = torch.cat([enc_target, dec_target], dim=0) 
        saliency_current = self.saliency_npy_all[:,index]
        # print(camera_inputs.shape, targets.shape)
        return camera_inputs, targets[:,:-1], saliency_current  # torch.Size([96, 3, 224, 224]) torch.Size([9, 22])
    

    ## get (img txt)
    def getimgstxts(self, index):
        session, start, end, enc_target, dec_target = self.inputs[index]
        set_path = "test" if session.split('_')[1]=="test" else "val"
        imgs_path = osp.join(self.imgs_root, set_path, session)
        imgs = []
        for i in range(start, end):
            i_path = osp.join(imgs_path, self.img_format.format(i))
            img = self.transform(Image.open(i_path))  # .convert('RGB') 198 (320, 180) RGB torch.Size([3, 180, 320])
            imgs.append(img)
        camera_inputs = torch.stack(imgs)  # [64, 3, 224, 224]
        enc_target = torch.tensor(enc_target[-1:])  # last frame is current frame
        dec_target = torch.tensor(dec_target)
        targets = torch.cat([enc_target, dec_target], dim=0)  # [9, 22]
        targets_index = torch.argmax(targets, dim=1)
        template = random.choice(kinetics_templates)  # todo using template?
        texts = [template.format(all_class_name_[i]) for i in targets_index]
        ## tokenize the text caption
        text_input = self.tokenizer(texts)  # [9, 77] 
        ## retturn
        if self.training:
            return camera_inputs, text_input  # torch.Size([64, 3, 224, 224]), len 9
        else:  ## 22 -> 11 cls names for zero shot eval
            targets = targets[:,self.noval_class]
            return camera_inputs, targets
    
    """
    ## get last frame
    def getlastframe(self, index):
        session, start, end, enc_target = self.inputs[index]
        img_path = osp.join(self.imgs_root, session, self.img_format.format(end-1))
        img = self.transform(Image.open(img_path))
        
        camera_inputs = img
        enc_target = torch.tensor(enc_target)[-1:].squeeze()
        # print(camera_inputs.shape, enc_target.shape)dataset_val
        return camera_inputs, enc_target  # torch.Size([64, 3, 180, 320]) torch.Size([64, 22])
    """

    ## getitem
    def __getitem__(self, index):
        if self.read_from == "pickle" and self.single_eval: # lazy code TODO
            return self.getfeatstargets(index=index) 
        elif self.read_from == "pickle" and not self.single_eval:    
            return self.getfeatstargetssaliency(index=index)
        elif self.read_from == 'jpg':
            return self.getimgstargets(index = index)
        else:
            raise NotImplementedError

    def __len__(self):
        return len(self.inputs)
        # return len(self.inputs[:10240])  # todo for happy debug, using 2048


def frame_level_map_n_cap(results, with_bg=False):
    all_probs = results['probs']
    all_labels = results['labels']   
    # print(all_probs.shape, all_labels.shape)

    n_classes = all_labels.shape[0]
    all_cls_ap, all_cls_acp = list(), list()
    from_index = 0 if with_bg else 1  # default: cut off background label
    for i in range(from_index, n_classes):
        this_cls_prob = all_probs[i, :]
        this_cls_gt = all_labels[i, :]
        w = np.sum(this_cls_gt == 0) / np.sum(this_cls_gt == 1)

        indices = np.argsort(-this_cls_prob)
        tp, psum, cpsum = 0, 0., 0.
        for k, idx in enumerate(indices):
            if this_cls_gt[idx] == 1:
                tp += 1
                wtp = w * tp
                fp = (k + 1) - tp
                psum += tp / (tp + fp)
                cpsum += wtp / (wtp + fp)
        this_cls_ap = psum/np.sum(this_cls_gt)
        this_cls_acp = cpsum / np.sum(this_cls_gt)

        all_cls_ap.append(this_cls_ap)
        all_cls_acp.append(this_cls_acp)

    map = sum(all_cls_ap) / len(all_cls_ap)
    cap = sum(all_cls_acp) / len(all_cls_acp)

    # print(len(all_cls_ap), len(all_cls_acp)) # thumos 20 20
    # embed()
    # exit()

    return {'map': map, 'all_cls_ap': all_cls_ap, 'cap': cap, 'all_cls_acp': all_cls_acp}

