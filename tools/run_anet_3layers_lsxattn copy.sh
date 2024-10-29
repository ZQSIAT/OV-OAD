PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
# --quotatype=spot
# --cfg configs/ovoador/3layers/enc32_anet_3layers_lsxattn.yml \
srun -p Gvlab-S1 -N1 -n4 --gres=gpu:4 --job-name=internvid-ovoad --quotatype=auto --cpus-per-task=12 \
python -u -m main_pretrain \
    --cfg configs/ovoador/3layers/enc32_anet_3layers_lsxattn.yml \

