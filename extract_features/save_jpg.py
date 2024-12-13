import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# import clip
import torch
import os
import os.path as osp
import io
import av
import cv2
import json
import jsonlines
import random
import numpy as np
import logging
import subprocess
import torch
from tqdm import tqdm
from PIL import Image, ImageSequence
from decord import VideoReader, cpu
from torchvision import transforms
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize, ToPILImage
from torchvision.transforms import InterpolationMode
from torchvision.transforms import PILToTensor
from IPython import embed
from datetime import datetime
import clip as CLIP
import math
import argparse
from internvid_extract_loader import InternvidImgLoader
from torch.utils.data import DataLoader, DistributedSampler
import pickle 
BICUBIC = InterpolationMode.BICUBIC
BILINEAR = InterpolationMode.BILINEAR

def clip_transform(n_px):
    return Compose([
        Resize((n_px, n_px), interpolation=BICUBIC),
        # CenterCrop(n_px),
        transforms.Lambda(lambda x: x.float().div(255.0)),
        Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])

from petrel_client.client import Client
client = Client("/mnt/petrelfs/xxx/petreloss.conf")

device = "cuda" if torch.cuda.is_available() else "cpu"


def load_image_from_path(image_path, client=None):
    if client:
        value = client.Get(image_path)
        img_bytes = np.frombuffer(value, dtype=np.uint8)
        buff = io.BytesIO(img_bytes)
        image = Image.open(buff).convert('RGB')
    else:
        image = Image.open(image_path).convert('RGB')  # PIL Image
    image = PILToTensor()(image)  # (C, H, W), torch.uint8
    return image


def get_frame_indices(num_frames=8, vlen=100, sample='fps4.0', fix_start=None, input_fps=24.0, max_num_frames=-1):
    if sample in ["rand", "middle"]: # uniform sampling
        acc_samples = min(num_frames, vlen)
        # split the video into `acc_samples` intervals, and sample from each interval.
        intervals = np.linspace(start=0, stop=vlen, num=acc_samples + 1).astype(int)
        ranges = []
        for idx, interv in enumerate(intervals[:-1]):
            ranges.append((interv, intervals[idx + 1] - 1))
        if sample == 'rand':
            try:
                frame_indices = [random.choice(range(x[0], x[1])) for x in ranges]
            except:
                frame_indices = np.random.permutation(vlen)[:acc_samples]
                frame_indices.sort()
                frame_indices = list(frame_indices)
        elif fix_start is not None:
            frame_indices = [x[0] + fix_start for x in ranges]
        elif sample == 'middle':
            frame_indices = [(x[0] + x[1]) // 2 for x in ranges]
        else:
            raise NotImplementedError

        if len(frame_indices) < num_frames:  # padded with last frame
            padded_frame_indices = [frame_indices[-1]] * num_frames
            padded_frame_indices[:len(frame_indices)] = frame_indices
            frame_indices = padded_frame_indices
    elif "fps" in sample:  # fps0.5, sequentially sample frames at 0.5 fps
        output_fps = float(sample[3:])
        duration = float(vlen) / input_fps
        delta = 1 / output_fps  # gap between frames, this is also the clip length each frame represents
        frame_seconds = np.arange(0 + delta / 2, duration + delta / 2, delta)
        frame_indices = np.around(frame_seconds * input_fps).astype(int)
        frame_indices = [e for e in frame_indices if e < vlen]
        if max_num_frames > 0 and len(frame_indices) > max_num_frames:
            frame_indices = frame_indices[:max_num_frames]
            # frame_indices = np.linspace(0 + delta / 2, duration + delta / 2, endpoint=False, num=max_num_frames)
    else:
        raise ValueError
    return frame_indices

def read_frames_decord_by_fps(
                            video_path,
                            sample='fps4.0',
                            fix_start=None, 
                            max_num_frames=-1,
                            trimmed30=False,
                            num_frames=8
                            ):
    import decord
    decord.bridge.set_bridge("torch")
    
    video_bytes = client.get(video_path)
    # if video_bytes is None:
    #     logger.warning(f"Failed to load {video_path}")
    video_reader = VideoReader(io.BytesIO(video_bytes), num_threads=1)
    vlen = len(video_reader)
    fps = video_reader.get_avg_fps()
    duration = vlen / float(fps)

    if trimmed30 and duration > 30:
        duration = 30
        vlen = int(30 * float(fps))

    frame_indices = get_frame_indices(
        num_frames=num_frames, vlen=vlen, sample=sample, fix_start=fix_start,
        input_fps=fps, max_num_frames=max_num_frames
    )
    frames = video_reader.get_batch(frame_indices)  # (T, H, W, C), torch.uint8
    frames = frames.permute(0, 3, 1, 2)  # (T, C, H, W), torch.uint8
    return frames, fps

def get_vlen_4fps(
                video_path,
                sample='fps4.0',
                fix_start=None, 
                max_num_frames=-1,
                num_frames=8
                ):
    import decord
    decord.bridge.set_bridge("torch")
    filename, _ = osp.splitext(osp.basename(video_path))
    video_bytes = client.get(video_path)
    # if video_bytes is None:
    #     logger.warning(f"Failed to load {video_path}")
    video_reader = VideoReader(io.BytesIO(video_bytes), num_threads=1)
    vlen = len(video_reader)
    fps = video_reader.get_avg_fps()
    # duration = vlen / float(fps)
    frame_indices = get_frame_indices(
        num_frames=num_frames, vlen=vlen, sample=sample, fix_start=fix_start,
        input_fps=fps, max_num_frames=max_num_frames
    )
    vlen_4fps = len(frame_indices)
    return vlen_4fps, vlen, fps


def resize_image(image, short_edge):
    # 计算缩放比例
    width, height = image.size
    aspect_ratio = max(short_edge / width, short_edge / height)
    new_width = int(width * aspect_ratio)
    new_height = int(height * aspect_ratio)
    
    # 缩放图像
    resized_image = image.resize((new_width, new_height), Image.LANCZOS)
    
    return resized_image

def time_str_to_seconds(time_str):
    # 按照":"和"."进行分割
    hours, minutes, seconds = time_str.split(":")
    seconds, milliseconds = seconds.split(".")
    # 将各个部分转换为整数或浮点数
    hours = int(hours)
    minutes = int(minutes)
    seconds = int(seconds)
    milliseconds = int(milliseconds) / 1000
    # 计算总秒数
    total_seconds = (hours * 3600) + (minutes * 60) + seconds + milliseconds
    return total_seconds

def generatetargets(input, output):
    filenames = dict()
    # 打开 JSONL 文件并逐行处理
    with jsonlines.open(input) as reader:
        for line in tqdm(reader):
            filename = line["YoutubeID"]
            if filename not in filenames:
                filenames[filename] = dict()
            start_times = line["Start_timestamp"]
            end_times = line["End_timestamp"]
            try:
                start = time_str_to_seconds(start_times)
                end = time_str_to_seconds(end_times)
            except Exception as e:
                print(e, line)
                continue
            caption = line["Caption"]
            filenames[filename][start] = [end, caption]
    ## generate targets
    # 将filenames字典保存为 JSON 文件
    with open(output, 'w') as json_file:
        json.dump(filenames, json_file)

def generatesortedeffectargets(input, output):
    out_effec_frames = dict()
    ## 通过有效帧帧数排名，后检索，累加至2M，10M
    for filename, targets in tqdm(input.items()):
        sorted_targets = dict(sorted(targets.items()))
        num_frames = 0
        for target, value in sorted_targets.items():
            start_time = float(target)
            end_time = value[0]
            (start_4fps, end_4fps) = int(start_time*4), math.ceil(end_time*4)
            # num_frames += round((end_time-start_time)*4.0)
            num_frames += (end_4fps-start_4fps)
        out_effec_frames[filename] = [sorted_targets, num_frames]
        # break ## debug

    sorted_effec_videos = dict(sorted(out_effec_frames.items(), key=lambda x: x[1][1], reverse=True))  ## 降序

    with open(output, 'w') as json_file:
        json.dump(sorted_effec_videos, json_file)

def instancefindvideos(filenames, instances=2e6, per_caption_min_frames=8):
    out_videos_list = []
    num_videos = 0
    num_instances = 0
    for filename, targets in tqdm(filenames.items()):
        if targets[1] <= targets[2]*per_caption_min_frames:
            continue
        out_videos_list.append(filename)
        num_instances += targets[1]
        num_videos += 1
        if num_instances >= instances:
            return num_videos, out_videos_list

def parse_args():
    parser = argparse.ArgumentParser('GroupViT training and evaluation script')
    parser.add_argument('--range', type=int, nargs='+', default=[0, 200],
                    help='from 0 to 100')
    args = parser.parse_args()
    return args
  
image_transform = clip_transform(224)

# Load 4060598 sample ids success!
print(f"Load {len(filenames)} samples success!", ) 
print(f"will save to ", f"{save_s3_root}/")

def save_imgs(start, end):
    for idx in tqdm(range(start, end)):
        filename = filenames[idx]
        video_s3_path = fineaction_s3_root.format(filename)
        try:
            read_frames_decord_save_by_s3(video_s3_path)
        except Exception as e:
            print(e, f"{video_s3_path} extract failed!")
            continue


if __name__ == "__main__":
    args = parse_args()
    start, end = args.range
    save_imgs(start, end)
    exit()
