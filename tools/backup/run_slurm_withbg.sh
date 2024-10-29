PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
srun -p Gvlab-S1 -N1 -n4 --gres=gpu:4 --job-name=internvid-ovoad --cpus-per-task=12 \
python -u -m main_pretrain \
    --cfg configs/ovoador/enc32_nv2w_withbg_vl_adapter.yml \
    # --cfg configs/ovoador/enc32_nv2w_withbg_multilabels.yml \
    # --cfg configs/ovoador/enc32_nv2w_withbg.yml \
    # --cfg configs/ovoador/enc32_nv2k_withbg.yml \
    # --cfg configs/ovoador/enc128_nv2k_withbg.yml \







# PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
# srun -p video -N1 -n8 --gres=gpu:8 --job-name=4m --quotatype=auto --cpus-per-task=12 \
# python -u -m main_pretrain \ /mnt/petrelfs/heyinan/00_zqs/code/ovoad/configs/ovoador/ovoador_pretrain_vit_clip_enc32_stages2.yml 
#     --cfg configs/ovsegmentor/ovsegmentor_pretrain_vit_bert_stage1.yml
