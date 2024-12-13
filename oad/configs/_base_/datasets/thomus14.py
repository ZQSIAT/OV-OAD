# ========================= thumos14 ==========================
single_eval = False 
eval_type = "means"  # single, means
out_dim768 = True
class_type = 'actions'
data_root = '/mnt/petrelfs/xxxx/data/thumos_imgs' # ?
dataset_file = '/mnt/petrelfs/xxxxxx/code/ovoad/extract_features/data_list/new_data_info.json'
enc_steps = 32
dec_steps = 8
long_term_steps = 64
nonzero = 0 # not use !!!
numclass = 22
batch_size = 256
eval_batch_size = 256
num_workers = 4
feature_type = 'CLIP'
input_resolution = 224  # model.visual.input_resolution
debug = False
log_freq = 10
read_from = "jpg"
# ========================= OAD CLIP ==========================
models_name = "ViT-B/16"  #  ViT-L-14, ViT-B/16
max_txt_l = 77


