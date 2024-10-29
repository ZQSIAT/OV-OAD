PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
srun -p Gvlab-S1 --pty -N1 -n2 --gres=gpu:2 --job-name=internvid-debug --quotatype=auto --cpus-per-task=12 \
python -u -m main_pretrain \
    --cfg configs/ovoador/3layers/enc32_anet_3layers_lsxattn_copy.yml \

# PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
# srun -p video3 -N1 -n2 --gres=gpu:2 --job-name=4m --quotatype=auto --cpus-per-task=12 \
# python -u -m main_pretrain \
#     --cfg configs/ovsegmentor/ovsegmentor_pretrain_vit_bert_stage2.yml \
#     --amp-opt-level O0 