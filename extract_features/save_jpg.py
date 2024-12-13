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
client = Client("/mnt/petrelfs/heyinan/petreloss.conf")

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

    # embed()
    # exit()

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
            # embed()
            # exit()
            # 将 BytesIO 对象的内容写入文件
            client.put(f"{save_s3_root}/src_img/{filename}/{img_name}", f.getvalue())

    #     if t == 16: ## debug
    #         break  
    # embed()
    # exit()

# read_frames_decord_save_by_s3('phdd:s3://youtubeBucket/videos/QMAHmFAWfrc.mp4')
# exit()

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

    # # 将时间字符串解析为datetime对象
    # time_obj = datetime.strptime(time_str, "%H:%M:%S.%f")
    # # 将datetime对象转换为总秒数
    # total_seconds = time_obj.hour * 3600 + time_obj.minute * 60 + time_obj.second + time_obj.microsecond / 1000000
    
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

    # ## sorted by number captions
    # out_effec_frames = dict()
    # ## 通过有效帧帧数排名，后检索，累加至2M，10M
    # for filename, targets in tqdm(filenames.items()):
    #     sorted_targets = dict(sorted(targets.items()))
    #     num_frames = 0
    #     clen = len(sorted_targets)
    #     for target, value in sorted_targets.items():
    #         start_time = float(target)
    #         end_time = value[0]
    #         (start_4fps, end_4fps) = int(start_time*4), math.ceil(end_time*4)
    #         # num_frames += round((end_time-start_time)*4.0)
    #         num_frames += (end_4fps-start_4fps)
    #     out_effec_frames[filename] = [sorted_targets, num_frames, clen]
    #     # break ## debug
    # sorted_effec_videos = dict(sorted(out_effec_frames.items(), key=lambda x: x[1][2], reverse=True))  ## 降序

    # with open(effec_caption_json, 'w') as json_file:
    #     json.dump(sorted_effec_videos, json_file)

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

json_root = "/mnt/petrelfs/heyinan/00_zqs/data/internvid"
internvid_flt_10m = osp.join(json_root, "InternVid-10M-flt.jsonl")
video_s3_root = "phdd:s3://youtubeBucket/videos/{:}.mp4"
jpg_s3_root = "pvideo:s3://internvid_extc_feats_4fps/src_img/{}/"
img_format = "img_{:08d}.jpg"
pth_s3_format = "pvideo:s3://internvid_extc_feats_4fps/CLIP_ViT_B_16/{}.pth"
epic_s3_root = "pssd:s3://epic/epic_video_320p/{:}.mp4"
anet_s3_root = "pssd:s3://ANet/ANet_320p/{:}/v_{:}.mp4" # train / val 
anet_meta_json = "/mnt/petrelfs/heyinan/00_zqs/code/ovoad/extract_features/data_list/activity_net.v1-3.min.json"
tv_s3_root = "pvideo:s3://tv_series/mkv_videos/{:}.mkv"

fineaction_s3_root = "pssd:s3://fineaction/fineaction_video/{:}" # v_00000902.webm, v_00000901.mp4
fineaction_meta_json = "/mnt/cache/heyinan/00_zqs/code/ovoad/extract_features/data_list/fineaction_annotations_gt.json"


# test_json = "/mnt/petrelfs/heyinan/00_zqs/code/ovoad/extract_features/test.jsonl"
# filenames_json = "/mnt/petrelfs/heyinan/00_zqs/code/ovoad/extract_features/internvid_10m_targets.json" # source targets
# effec_frame_json = "/mnt/petrelfs/heyinan/00_zqs/code/ovoad/extract_features/internvid_10m_effec_frames.json"  # sorted by effec frames
# statistic_10w_videos_format = "/mnt/petrelfs/heyinan/00_zqs/code/ovoad/extract_features/statistis_split/internvid10w_statistics_{}.json"  
effec_caption_json = "/mnt/petrelfs/heyinan/00_zqs/code/ovoad/extract_features/internvid_10m_effec_captions.json"  # sorted by number captions
effec_10w_videos_json = "/mnt/petrelfs/heyinan/00_zqs/code/ovoad/extract_features/internvid10w_20m_oad_instances.json"  # videos list
statistic_10w = "/mnt/petrelfs/heyinan/00_zqs/code/ovoad/extract_features/internvid10w_statistics.json"  # videos list and vlen4fps, vlen, fps
anno_pickle = "/mnt/petrelfs/heyinan/00_zqs/code/ovoad/extract_features/internvid10w_anno_106176.pickle"
epic_val_src = "/mnt/petrelfs/heyinan/00_zqs/code/ovoad/extract_features/data_list/EPIC_100_validation.csv"
epic_val_sessions = ["P01_11", "P01_12", "P01_13", "P01_14", "P01_15", "P02_12", "P02_13", "P02_14", "P02_15", "P03_21", "P03_22", "P03_23", "P03_24", "P03_25",
                    "P03_26", "P04_24", "P04_25", "P04_26", "P04_27", "P04_28", "P04_29", "P04_30", "P04_31", "P04_32", "P04_33", "P05_07", "P05_09", "P06_10",
                    "P06_11", "P06_12", "P06_13", "P06_14", "P07_12", "P07_13", "P07_14", "P07_15", "P07_16", "P07_17", "P07_18", "P08_09", "P08_10", "P08_14",
                    "P08_15", "P08_16", "P08_17", "P09_07", "P09_08", "P10_03", "P11_17", "P11_18", "P11_19", "P11_20", "P11_21", "P11_22", "P11_23", "P11_24",
                    "P12_03", "P12_08", "P13_01", "P13_02", "P13_03", "P14_06", "P14_08", "P15_04", "P15_05", "P15_06", "P16_04", "P17_02", "P18_01", "P18_02", 
                    "P18_03", "P18_04", "P18_05", "P18_06", "P18_07", "P18_08", "P18_09", "P18_10", "P18_11", "P18_12", "P19_05", "P19_06", "P20_05", "P20_06", 
                    "P20_07", "P21_02", "P22_01", "P22_02", "P22_03", "P22_04", "P23_05", "P24_09", "P25_06", "P25_07", "P25_08", "P26_30", "P26_31", "P26_32", 
                    "P26_33", "P26_34", "P26_35", "P26_36", "P26_37", "P26_38", "P26_39", "P26_40", "P26_41", "P27_05", "P28_15", "P28_16", "P28_17", "P28_18", 
                    "P28_19", "P28_20", "P28_21", "P28_22", "P28_23", "P28_24", "P28_25", "P28_26", "P29_05", "P29_06", "P30_07", "P30_08", "P30_09", "P31_10", 
                    "P31_11", "P31_12", "P32_01", "P32_02", "P32_03", "P32_04", "P32_05", "P32_06", "P32_07", "P32_08", "P32_09", "P32_10"]

tv_val_sessions = ["24_ep4", "Breaking_Bad_ep3", "Mad_Men_ep3", "How_I_Met_Your_Mother_ep7", "How_I_Met_Your_Mother_ep8", "Modern_Family_ep5", "Sons_of_Anarchy_ep3"]


# for epic test
# filenames = epic_val_sessions
# save_s3_root = "phdd:s3://epic_extc_feats_4fps"

# for tv series test
# filenames = tv_val_sessions
# save_s3_root = "phdd:s3://tv_series_extc_feats_4fps"

# for anet train and validation 
## load all targets
# with open(anet_meta_json, 'r') as f:
#     filenames = json.load(f)

# train_list = [["train", k] for k, v in filenames['database'].items() if v["subset"] == "training"] 
# val_list = [["val", k] for k, v in filenames['database'].items() if v["subset"] == "validation"] 
# filenames = train_list + val_list
# save_s3_root = "phdd:s3://anet_extc_feats_4fps"
# Load 14950 samples success!


## for FineAction  TODO 
with open(fineaction_meta_json, 'r') as f:
    filenames = json.load(f)
val_list = [v["filename"] for k, v in filenames['database'].items() if v["subset"] == "validation"] # only use val set
filenames = val_list  # Load 4174 samples success!
save_s3_root = "phdd2:s3://fineaction_extc_feats_4fps"


# Load 4060598 sample ids success!
print(f"Load {len(filenames)} samples success!", ) 
print(f"will save to ", f"{save_s3_root}/")

# embed()
# exit()

def save_imgs(start, end):
    for idx in tqdm(range(start, end)):
        filename = filenames[idx]
        # video_s3_path = epic_s3_root.format(filename)
        # video_s3_path = tv_s3_root.format(filename)
        # video_s3_path = anet_s3_root.format(filename[0], filename[1])
        video_s3_path = fineaction_s3_root.format(filename)
        # video_s3_path = video_s3_root.format(filename)
        try:
            read_frames_decord_save_by_s3(video_s3_path)
        except Exception as e:
            print(e, f"{video_s3_path} extract failed!")
            continue


def statisticvideos(start, end):
    out_json = statistic_10w_videos_format.format(start)
    out_videos_list = dict()
    for idx in tqdm(range(start, end)):
        filename = filenames[idx]
        video_s3_path = video_s3_root.format(filename)
        try:
            vlen_4fps, vlen, fps = get_vlen_4fps(video_s3_path)
            out_videos_list[filename] = [vlen_4fps, vlen, fps]
        except Exception as e:
            print(e, f"read {video_s3_path} failed!")
            continue
        # break
    with open(out_json, 'w') as json_file:
        json.dump(out_videos_list, json_file)
    # embed()
    # exit()

def readsplitjson(root_path):
    count = 0
    total_list = dict()
    # total_list = []
    file_list = os.listdir(root_path)
    # file_list.sort()
    for file in tqdm(file_list):
        json_path = osp.join(root_path, file)
        # with open(json_path, 'r') as f:
        #     videos_list = json.load(f)
        with open(json_path, 'rb') as file:
            videos_list = pickle.load(file)
        count += len(videos_list)
        print(f"current video list has {len(videos_list)} samples")
        total_list = {**total_list, **videos_list}
        # total_list = total_list+videos_list

    print(f"count video: {count}, len(total_list): {len(total_list)} \n", total_list)

    with open(anno_pickle, 'wb') as json_file:
        # json.dump(total_list, json_file)
        pickle.dump(total_list, json_file)
    embed()
    exit()

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
    #     break  # debug
    # embed()
    # exit()
        
# 先单独保存每个视频的pth，需要的时候再合并
def ext_batch_feats(start, end):
    ## ========= load model =========
    model_name = "ViT-B/16" 
    print(f"Loading CLIP (backbone: {model_name})")  
    clip_model, _ = CLIP.load(model_name)
    model = clip_model.cuda()
    model.eval()
    model_name = model_name.replace('-','_').replace('/','_')

    ## load meta targets
    with open(effec_caption_json, 'r') as f:
        targets_all = json.load(f)
    list_filenames = [{k: v} for k,v in filenames.items()]
    list_filenames = list_filenames[start:end]

    ## build loader
    batch_size = 1
    num_workers = 16
    dataset = InternvidImgLoader(video_list=list_filenames,
                                targets_json=effec_caption_json,)
    # batch = dataset.__getitem__(0)
    # embed()
    # exit()
    sampler = torch.utils.data.SequentialSampler(dataset)
    data_loader = DataLoader(dataset, batch_size, sampler=sampler,
                            drop_last=False, pin_memory=False, num_workers=num_workers)
    
    num_batches = len(data_loader)
    num_samples = dataset.__len__()
    print(f"{num_batches} batches, {num_samples} num_samples") 
    for batch in tqdm(data_loader):
        input_frames, filename = batch
        filename = filename[0]
        input_frames = input_frames.squeeze(0)
        vlen, vlen_src, fps = filenames[filename]
        targets = targets_all[filename][0]
        sorted_targets = {k: v for k, v in sorted(targets.items(), key=lambda item: float(item[0]))}
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

        try:  
            ## ext feats
            input_frames = input_frames.cuda()
            with torch.no_grad():
                video_feature = model.encode_image(input_frames)
                video_feature = video_feature.half() # .float() # [2962, 768]
            sample_features = {
                "rgb": video_feature.cpu(),
                # "anno": anno.cpu(),
                "captions": captions,
                "fps": round(fps, 2),
                "vlen_src": vlen_src}

            ## save 
            with io.BytesIO() as f:
                torch.save(sample_features, f)
                client.put(f"phdd:s3://internvid_extc_feats_4fps/CLIP_{model_name}_768/{filename}.pth", f.getvalue())
        except Exception as e:
            print(e, f"{filename} extracted failed!")
            continue
        # break  # debug
    # embed()
    # exit()
        

def check_jpg_openable(start, end):
    error_list = []
    list_filenames = [{k: v} for k,v in filenames.items()]
    for idx in tqdm(range(start, end)):
        filename_dict = list_filenames[idx]
        filename, videoinfo = next(iter(filename_dict.items()))
        vlen_4fps, vlen, fps = videoinfo
        img_format = "img_{:08d}.jpg"
        jpg_s3_path = jpg_s3_root.format(filename)
        try:
            command = f"aws --endpoint-url=\"http://p-ceph-norm-outside.pjlab.org.cn\" --profile=pvideo s3 ls s3://internvid_extc_feats_4fps/src_img/{filename}/ | grep jpg | wc -l"
            practice_vlen = int(subprocess.check_output(command, shell=True, encoding='utf-8').strip())
            if practice_vlen != vlen_4fps:
                error_list.append(filename)
                continue
        except Exception as e:
            print(e, f"read {jpg_s3_path} failed!")
            error_list.append(filename)
            continue
        for i in range(vlen_4fps):
            image_path = osp.join(jpg_s3_path, img_format.format(i))
            try:
                # 使用PIL尝试打开图像
                value = client.Get(image_path)
                img_bytes = np.frombuffer(value, dtype=np.uint8)
                buff = io.BytesIO(img_bytes)
                img_pil = Image.open(buff)
                img_pil.verify()  # 验证图像文件的完整性
                    # img_pil.load()  # 加载图像像素数据
                    # print("PIL: Image opened successfully")
                # # 使用OpenCV尝试打开图像
                # img_cv = cv2.imread(image_path)
                # if img_cv is not None:
                #     print("OpenCV: Image opened successfully")
                # else:
                #     print("OpenCV: Failed to open image")
            except Exception as e:
                print("Error while opening image:", str(e))
                error_list.append(filename)
                break
        # embed()
        # exit()
        # print(f"{filename} done!")
    out_json = f"./split_error_list/error_list_{start}.json"
    with open(out_json, 'w') as json_file:
        json.dump(error_list, json_file)
    # embed()
    # exit()


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
    #     break
    # embed()
    # exit()
    with open(anno_pickle, 'wb') as file:
        pickle.dump(anno, file)
    print(f"Save annos {len(anno)} done")


if __name__ == "__main__":
    # ext_annos()
    # readsplitjson("/mnt/petrelfs/heyinan/00_zqs/code/ovoad/extract_features/anno_split")
    args = parse_args()
    start, end = args.range
    # ext_batch_feats(start, end)
    # check_jpg_openable(start, end)
    save_imgs(start, end)
    # statisticvideos(start, end)
    # getanno2pickle(start, end)
    exit()

## sorted by number captions 
# In [1]: instancefindvideos(filenames, instances=2e6)                                                                                                                                                                      | 10170/4060598 [00:00<00:02, 1372283.87it/s]
# Out[1]: 10171
# In [2]: instancefindvideos(filenames, instances=10e6)                                                                                                                                                                    | 55103/4060598 [00:00<00:02, 1434593.39it/s]
# Out[2]: 55104
# In [3]: instancefindvideos(filenames, instances=20e6)                                                                                                                                                                | 115288/4060598 [00:00<00:02, 1542625.46it/s]
# Out[3]: 115289

## sorted by number captions filterd by per_caption_min_frames
# In [5]: instancefindvideos(filenames, instances=2e6)                                                                                                                                                                    | 10538/4060598 [00:00<00:02, 1421528.18it/s]
# Out[5]: 9416
# In [6]: instancefindvideos(filenames, instances=10e6)                                                                                                                                                                     | 56756/4060598 [00:00<00:02, 1495911.76it/s]
# Out[6]: 50816
# In [7]: instancefindvideos(filenames, instances=20e6)                                                                                                                                                                 | 118591/4060598 [00:00<00:02, 1386692.80it/s]
# Out[7]: 106176

## sorted by number frames 
# instancefindvideos(filenames, i nstances=2e6)
# Out[1]: 1644
# In [2]: instancefindvideos(filenames, instances=10e6)                                                                                                                                                                    | 10509/4060598 [00:00<00:02, 1982813.35it/s]
# Out[2]: 10510
# In [3]: instancefindvideos(filenames, instances=20e6)                                                                                                                                                                  | 24032/4060598 [00:00<00:02, 2013051.48it/s]
# Out[3]: 24033
"""
In [1]: sample_features
Out[1]: 
{'rgb': tensor([[ 0.2922, -0.3398,  0.6045,  ...,  0.4287,  0.1490,  0.3855],
         [-0.0729, -0.4753,  0.5425,  ...,  0.4429,  0.1837,  0.4678],
         [-0.0393, -0.2983,  0.5947,  ...,  0.4316,  0.1633,  0.3635],
         ...,
         [ 0.1087, -0.6338,  0.0684,  ...,  0.0982,  0.3311,  0.2477],
         [ 0.0172, -0.7085,  0.0496,  ...,  0.0740,  0.3315,  0.2148],
         [-0.0151, -0.7319,  0.0210,  ...,  0.0549,  0.3184,  0.2057]]),
 'anno': tensor([1, 1, 1,  ..., 0, 0, 0]),
 'captions': [{1: [0.0, 3.737, 'an artist drawing a circle in pencil']},
  {2: [187.888, 192.592, 'a woman is writing on an ipad to design a door']},
  {3: [348.748, 418.585, 'painting a sweet bell pepper']},
  {4: [5.939, 9.71, 'close up of a woman reading an open book']},
  {5: [501.801, 587.72, 'a woman in a room with plants at her desk']},
  {6: [9.71, 13.313, 'a woman paints her own art']}],
 'fps': 29.97002997002997,
 'vlen_src': 22190}
"""
# embed()
# exit()
# for filename, targets in tqdm(filenames.items()):
#         video_s3_path = video_s3_root.format(filename)
#         try:
#             ## load frames
#             input_frames, fps = read_frames_decord_by_fps(video_s3_path)

#             vlen = input_frames.shape[0]
#             anno = torch.zeros(vlen, dtype=torch.long)
#             sorted_targets = dict(sorted(targets.items()))

#             ## get captions
#             captions = []
#             cls_id = 0
#             for target, value in sorted_targets.items():
#                 cls_id += 1
#                 start_time = float(target)
#                 end_time = value[0]
#                 caption = value[1]
#                 captions.append({cls_id: [start_time, end_time, caption]})

#                 ## get anno
#                 (start_4fps, end_4fps) = int(start_time*4), math.ceil(end_time*4)  # 向下，向上取整
#                 end_4fps = vlen if end_4fps > vlen else end_4fps
#                 start_4fps = 1 if start_4fps < 1 else start_4fps
#                 duration_indexs = range(start_4fps-1, end_4fps)
#                 anno[duration_indexs] = cls_id

#             ## ext feats
#             input_frames = input_frames.cuda()
#             input_frames = image_transform(input_frames)
#             with torch.no_grad():
#                 video_feature = model.encode_image(input_frames).float()
#                 # video_feature = get_vid_features(model, input_frames.unsqueeze(0))
#                 sample_features[filename] = {
#                     "rgb": video_feature.cpu(),
#                     "anno": anno.cpu(),
#                     "captions": captions,
#                     "fps": fps,}
#             with io.BytesIO() as f:
#                 torch.save(sample_features, f)
#                 client.put(f"pvideo:s3://internvid_extc_feats_4fps/CLIP_{model_name}/{filename}.pth", f.getvalue())
#         except Exception as e:
#             print(e, f"{filename} has {vlen} frames, but extract failed!")
#             continue

def get_text_features(model, input_text, tokenizer, text_feature_dict={}):
    if input_text in text_feature_dict:
        return text_feature_dict[input_text]
    text_template= f"{input_text}"
    with torch.no_grad():
        text_features = model.encode_text(text_template).float()
        # text_features /= text_features.norm(dim=-1, keepdim=True)      
        text_feature_dict[input_text] = text_features
    return text_features

def get_predict_label(clip_feature, text_feats_tensor, top=5):
    label_probs = (100.0 * clip_feature @ text_feats_tensor.T).softmax(dim=-1)
    top_probs, top_labels = label_probs.cpu().topk(top, dim=-1)
    return top_probs, top_labels