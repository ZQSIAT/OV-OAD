# ========================= anet ==========================
single_eval = False # when train must set it False
eval_type = "single"
class_type = 'actions' # noun_perframe, verb_perframe, action_perframe
data_root = 'phdd:s3://anet_extc_feats_4fps/CLIP_ViT_B_16_768/'
anno_root = '/mnt/petrelfs/heyinan/00_zqs/data/anet'
dataset_file = '/mnt/petrelfs/heyinan/00_zqs/code/ovoad/extract_features/data_list/new_data_info.json'
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


