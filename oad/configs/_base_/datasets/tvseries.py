# ========================= tvseries ==========================
single_eval = False # when train must set it False
class_type = 'actions'
eval_type = "single"   # single, means
data_root = 'phdd:s3://tv_series_extc_feats_4fps/src_img/'
anno_root = '/mnt/petrelfs/zhaoqingsong/data/tvseries'
dataset_file = '/mnt/petrelfs/zhaoqingsong/code/ovoad/extract_features/data_list/new_data_info.json'
enc_steps = 32
dec_steps = 8
nonzero = 0 # not use !!!
numclass = 22
batch_size = 256
eval_batch_size = 256
num_workers = 4
feature_type = 'CLIP'
input_resolution = 224  # model.visual.input_resolution
debug = False
log_freq = 10
read_from = "petrel"
# ========================= OAD CLIP ==========================
models_name = "ViT-B/16"  #  ViT-L-14, ViT-B/16
max_txt_l = 77


