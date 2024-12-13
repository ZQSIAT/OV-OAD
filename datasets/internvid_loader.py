import numpy as np
import random
from copy import deepcopy
from scipy.ndimage import label
import os
import os.path as osp
import subprocess
import time
from collections import defaultdict, deque
import datetime
import pickle
from typing import Optional, List
import torch
import torch.distributed as dist
from torch.utils.data import Dataset
from PIL import Image
from IPython import embed
# import models.backbones.clip as CLIP
import av
from petrel_client.client import Client
client = Client("/mnt/petrelfs/xxx/petreloss.conf")

# len 28
kinetics_templates = [
    "A photo of action {}.", 
    "A video of action {}.", 
    "He or she is {}.", 
    "A person is doing {}.",
    "Look, the human is {}.",
    "Human action of {}.", 
    "Playing action of {}.", 
    "Video classification of {}.", 
    "Doing a kind of action, {}.", 
    "Playing a kind of action, {}.", 
    "Can you recognize the action of {}?", 
    "{}, an action.", 
    "{} this is an action.",  
    "{}, a video of action.", 
    "An action of {} is in the video.", 
    "There is a person doing {} in the video.",
    'A photo of a person doing {}.',
    'A photo of a person performing {}.',
    'A photo of a person practicing {}.',
    'A video of a person doing {}.',
    'A video of a person performing {}.',
    'A video of a person practicing {}.',
    'A example of a person doing {}.',
    'A example of a person performing {}.',
    'A example of a person practicing {}.',
    'A demonstration of a person doing {}.',
    'A demonstration of a person performing {}.',
    'A demonstration of a person practicing {}.',
]

## len 80
imagenet_templates = [
    'a bad photo of a {}.',
    'a photo of many {}.',
    'a sculpture of a {}.',
    'a photo of the hard to see {}.',
    'a low resolution photo of the {}.',
    'a rendering of a {}.',
    'graffiti of a {}.',
    'a bad photo of the {}.',
    'a cropped photo of the {}.',
    'a tattoo of a {}.',
    'the embroidered {}.',
    'a photo of a hard to see {}.',
    'a bright photo of a {}.',
    'a photo of a clean {}.',
    'a photo of a dirty {}.',
    'a dark photo of the {}.',
    'a drawing of a {}.',
    'a photo of my {}.',
    'the plastic {}.',
    'a photo of the cool {}.',
    'a close-up photo of a {}.',
    'a black and white photo of the {}.',
    'a painting of the {}.',
    'a painting of a {}.',
    'a pixelated photo of the {}.',
    'a sculpture of the {}.',
    'a bright photo of the {}.',
    'a cropped photo of a {}.',
    'a plastic {}.',
    'a photo of the dirty {}.',
    'a jpeg corrupted photo of a {}.',
    'a blurry photo of the {}.',
    'a photo of the {}.',
    'a good photo of the {}.',
    'a rendering of the {}.',
    'a {} in a video game.',
    'a photo of one {}.',
    'a doodle of a {}.',
    'a close-up photo of the {}.',
    'a photo of a {}.',
    'the origami {}.',
    'the {} in a video game.',
    'a sketch of a {}.',
    'a doodle of the {}.',
    'a origami {}.',
    'a low resolution photo of a {}.',
    'the toy {}.',
    'a rendition of the {}.',
    'a photo of the clean {}.',
    'a photo of a large {}.',
    'a rendition of a {}.',
    'a photo of a nice {}.',
    'a photo of a weird {}.',
    'a blurry photo of a {}.',
    'a cartoon {}.',
    'art of a {}.',
    'a sketch of the {}.',
    'a embroidered {}.',
    'a pixelated photo of a {}.',
    'itap of the {}.',
    'a jpeg corrupted photo of the {}.',
    'a good photo of a {}.',
    'a plushie {}.',
    'a photo of the nice {}.',
    'a photo of the small {}.',
    'a photo of the weird {}.',
    'the cartoon {}.',
    'art of the {}.',
    'a drawing of the {}.',
    'a photo of the large {}.',
    'a black and white photo of a {}.',
    'the plushie {}.',
    'a dark photo of a {}.',
    'itap of a {}.',
    'graffiti of the {}.',
    'a toy {}.',
    'itap of my {}.',
    'a photo of a cool {}.',
    'a photo of a small {}.',
    'a tattoo of the {}.',
]

## zero shot mode using all sessions from train and val set.
class InternVidOadLoader(Dataset):
    def __init__(self, args, preprocess, flag='train'):
        super().__init__()
        assert flag in ['train', 'test', 'valid']
        self.zero_shot = args.zero_shot  
        self.pickle_root = osp.join(args.data_root, 'anno')  
        self.base_class = args.base_class  # for train [0, 5, 19, 4, 2, 15, 8, 14, 13, 12, 11]
        self.noval_class = args.noval_class  #  for test [0, 1, 18, 17, 7, 6, 3, 9, 20, 10, 16]
        if self.zero_shot:
            train_sessions = getattr(args, 'train_session_set')
            test_sessions = getattr(args, 'test_session_set')
            self.sessions = train_sessions + test_sessions
        else:
            self.sessions = getattr(args, flag + '_session_set')
        self.enc_steps = args.enc_layers
        self.dec_steps = args.dec_query
        self.training = flag == 'train'
        self.inputs = []
        self.subnet = 'val' if self.training else 'test'

        self.imgs_root = args.data_root

        self.img_format = "img{:05d}.jpg"

        if self.zero_shot:
            target_train = pickle.load(open(osp.join(self.pickle_root, 'thumos_val_anno.pickle'), 'rb'))
            target_test = pickle.load(open(osp.join(self.pickle_root, 'thumos_test_anno.pickle'), 'rb'))
            target_all = {**target_train, **target_test}
        else:
            target_all = pickle.load(open(osp.join(self.pickle_root, 'thumos_' + self.subnet + '_anno.pickle'), 'rb'))

        for session in self.sessions:
            target = target_all[session]['anno']
            # seed = np.random.randint(self.enc_steps) if self.training else 0
            seed = 0  # for dist.all_reduce, seed set at 0
            for start, end in zip(
                    range(seed, target.shape[0], 1),
                    range(seed + self.enc_steps, target.shape[0]-self.dec_steps, 1)):
                    # range(seed + self.enc_steps, target.shape[0], 1)):  # target.shape[0]
                enc_target = target[start:end]
                dec_target = target[end:end + self.dec_steps]
                class_h_target = enc_target[self.enc_steps - 1]
                class_h_target_index = class_h_target.argmax()
                # embed()
                # exit()
                if self.training and class_h_target_index in self.base_class:
                    self.inputs.append([session, start, end, enc_target, dec_target])
                elif not self.training and class_h_target_index in self.noval_class:
                    self.inputs.append([session, start, end, enc_target, dec_target])
                else:
                    continue
        self.transform = preprocess  # transforms.Compose([transforms.ToTensor(),])
        # self.tokenizer = CLIP.tokenize

    def getimgtxt(self, index):
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
            return camera_inputs, text_input, targets
        
    ## get frames from ceph
    def lazy_load_s3video(self, client, s3path_video, start, end):
        # load video from ceph
        video_bytes_stream = client.get(s3path_video, enable_stream_lazyloding=True)
        container = av.open(video_bytes_stream)
        stream = container.streams.video[0]
        # duration = stream.duration
        real_fps = container.streams.video[0].average_rate
        time_base = container.streams.video[0].time_base

        # Convert time to pts
        start_pts = int((start/real_fps) / time_base)
        end_pts = int((end/real_fps) / time_base)
        # mid_pts = int(stream.frames // real_fps //2 / time_base)
        # container.seek(mid_pts, stream=stream)

        # Seek to nearest key frame from the start
        container.seek(max(start_pts, 0), stream=stream)
        
        frames = []
        for frame in container.decode(**{"video":0}):
            if frame.pts < start_pts:
                continue
            if frame.pts <= end_pts:
                frames.append(frame)
            else:
                break
        frames_idx = #请补充sample帧index的函数
        # print(frames)
        frames = [frames[idx].to_rgb().to_image() for idx in frames_idx]
        return frames
        
    ## getitem
    def __getitem__(self, index):
        return self.getimgtxt(index=index)

    def __len__(self):
        return len(self.inputs)
        # return len(self.inputs[:2048])  # todo for happy debug, using 2048
    

def frame_level_map_n_cap(results):
    all_probs = results['probs']
    all_labels = results['labels']   

    n_classes = all_labels.shape[0]
    all_cls_ap, all_cls_acp = list(), list()
    for i in range(0, n_classes):
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
    return map, all_cls_ap, cap, all_cls_acp

