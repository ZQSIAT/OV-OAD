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
from internvid_extract_loader import InternvidImgLoader, TVSeriesImgLoader, EpicImgLoader, AnetImgLoader, FineActionImgLoader
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

def read_frames_decord_save_by_s3(
                            video_path,
                            sample='fps4.0',
                            fix_start=None, 
                            max_num_frames=-1,
                            trimmed30=False,
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

    # 获取张量形状信息
    T, C, H, W = frames.shape
    img_format = "img_{:08d}.jpg"

    # 逐帧保存为 JPG 文件
    for t in range(T):
        # 获取当前帧
        frame = frames[t]
        img_name = img_format.format(t)
        # 创建 PIL 图像对象
        img = Image.fromarray(frame.permute(1, 2, 0).numpy())
        # 缩放图像到短边为 224
        img = resize_image(img, 224)
        # 创建 BytesIO 对象
        with io.BytesIO() as f:
            # 将图像保存到 BytesIO 对象
            img.save(f, format='JPEG')
            # 将 BytesIO 对象的内容写入文件
            client.put(f"{save_s3_root}/src_img/{filename}/{img_name}", f.getvalue())
            
def get_vid_features(model, input_frames):
    with torch.no_grad():
        clip_feat = model.encode_vision(input_frames,test=True).float()
        # clip_feat /= clip_feat.norm(dim=-1, keepdim=True)    
    return clip_feat

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

## 实际抽帧的结果
filenames = {}
with open(fineaction_src_imgs_list, 'r') as f:
    for line in tqdm(f.readlines()):
        filename = line.strip()[31:].split('/')[1]
        if filename not in filenames:
            filenames[filename] = {"vlen": 1}
        else:
            filenames[filename]["vlen"] += 1

print(f" We have {len(list(filenames.keys()))} filenames to ext.")



# 先单独保存每个视频的pth，需要的时候再合并
def ext_feats(start, end):
    ## ========= load model =========
    model_name = "ViT-B/16" 
    print(f"Loading CLIP (backbone: {model_name})")  
    clip_model, _ = CLIP.load(model_name)
    model = clip_model.cuda()
    model.eval()
    model_name = model_name.replace('-','_').replace('/','_')

    ## load meta targets
    list_filenames = [{k: v} for k,v in filenames.items()]
    ## build loader

    with open(effec_caption_json, 'r') as f:
        targets_all = json.load(f)

    for idx in tqdm(range(start, end)):
        filename_dict = list_filenames[idx]
        filename, videoinfo = next(iter(filename_dict.items()))
        targets = targets_all[filename][0]  # targets_all[filename] = [sorted_targets, num_frames(targets), len(targets)]
        # 按照键的浮点数值大小排序的新字典 
        sorted_targets = {k: v for k, v in sorted(targets.items(), key=lambda item: float(item[0]))} 
        vlen, vlen_src, fps = videoinfo

        ## get captions
        anno = torch.zeros(vlen, dtype=torch.long)
        captions = []
        cls_id = 0
        for target, value in sorted_targets.items():
            cls_id += 1
            start_time = float(target)
            end_time = value[0]
            caption = value[1]
            captions.append({cls_id: [start_time, end_time, caption]})   
            ## get anno
            (start_4fps, end_4fps) = int(start_time*4), math.ceil(end_time*4)  # 向下，向上取整
            end_4fps = vlen if end_4fps > vlen else end_4fps
            start_4fps = 1 if start_4fps < 1 else start_4fps
            duration_indexs = range(start_4fps-1, end_4fps)
            anno[duration_indexs] = cls_id   

        jpg_s3_path = jpg_s3_root.format(filename)
        try:  
            ## load frames
            images_feats = []
            for i in range(vlen):
                image_path = osp.join(jpg_s3_path, img_format.format(i))
                image_feat = load_image_from_path(image_path, client=client)  # (C, H, W), torch.uint8
                images_feats.append(image_feat)
            images_feats = torch.stack(images_feats, dim=0)  

            ## ext feats
            input_frames = image_transform(images_feats).cuda()
            with torch.no_grad():
                video_feature = model.encode_image(input_frames).float()

            sample_features = {
                "rgb": video_feature.cpu(),
                "anno": anno.cpu(),
                "captions": captions,
                "fps": fps,
                "vlen_src": vlen_src}

            ## save 
            with io.BytesIO() as f:
                torch.save(sample_features, f)
                client.put(f"pvideo:s3://internvid_extc_feats_4fps/CLIP_{model_name}/{filename}.pth", f.getvalue())
        except Exception as e:
            ## make sure load jpgs good
            print(e, f"{filename} has {vlen} frames, extracted failed, when load {img_format.format(i)}!")
            continue

def getpthanno2pickle(start, end):
    anno = dict()
    for idx in tqdm(range(start, end)):
        filename = filenames[idx]
        pth_s3_path = pth_s3_format.format(filename)
        try:
            sample_meta = torch.load(io.BytesIO(memoryview(client.get(pth_s3_path))))
            anno_array = sample_meta["anno"].contiguous().numpy()
            anno[filename] = anno_array.astype(np.int16)
        except Exception as e:
            print(e, f"read {filename}.pth failed!")
        # break
    
    with open(anno_pickle.format(start), 'wb') as file:
        pickle.dump(anno, file)
    print(f"Save annos {len(anno)} done")
    # embed()
    # exit()


def ext_annos():
    with open(effec_caption_json, 'r') as f:
        targets_all = json.load(f)

    anno = dict()
    for filename, videoinfo in tqdm(filenames.items()):
        targets = targets_all[filename][0]  # targets_all[filename] = [sorted_targets, num_frames(targets), len(targets)]
        # 按照键的浮点数值大小排序的新字典 
        sorted_targets = {k: v for k, v in sorted(targets.items(), key=lambda item: float(item[0]))} 
        vlen, vlen_src, fps = videoinfo

        anno_array = np.zeros(vlen, dtype=np.int16)
        cls_id = 0
        for target, value in sorted_targets.items():
            cls_id += 1
            start_time = float(target)
            end_time = value[0]
            ## get anno
            (start_4fps, end_4fps) = int(start_time*4), math.ceil(end_time*4)  # 向下，向上取整
            end_4fps = vlen if end_4fps > vlen else end_4fps
            start_4fps = 1 if start_4fps < 1 else start_4fps
            duration_indexs = range(start_4fps-1, end_4fps)
            anno_array[duration_indexs] = cls_id

        anno[filename] = anno_array
    with open(anno_pickle, 'wb') as file:
        pickle.dump(anno, file)
    print(f"Save annos {len(anno)} done")


if __name__ == "__main__":
    args = parse_args()
    start, end = args.range
    ext_batch_feats(start, end)
