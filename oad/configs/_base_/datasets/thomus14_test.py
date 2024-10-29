# ========================= thumos14 ==========================
mode = "zsoad"  # for thumos online action detection only!
data_root = '/mnt/petrelfs/heyinan/00_zqs/data/thumos_imgs'
dataset_file = '/mnt/petrelfs/heyinan/00_zqs/code/ovoad/extract_features/data_list/new_data_info.json'
enc_steps = 32
dec_steps = 8
nonzero = 0
numclass = 22
batch_size = 256 
eval_batch_size = 256
num_workers = 4
feature_type = 'CLIP'
input_resolution = 224  # model.visual.input_resolution
debug = False
log_freq = 10
# ========================= CLIP ==========================
models_name = "ViT-B/16"  #  ViT-L-14, ViT-B/16
read_from = "pickle"
out_dim768 = True
# ========================= OAD CLIP ==========================
models_name = "ViT-B/16"  #  ViT-L-14, ViT-B/16
max_txt_l = 77


