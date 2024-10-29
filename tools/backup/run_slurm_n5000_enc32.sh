PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
srun -p Gvlab-S1 -N1 -n4 --gres=gpu:4 --job-name=internvid-ovoad -x SH-IDC1-10-140-1-28 --quotatype=auto --cpus-per-task=12 \
python -u -m main_pretrain \
    --cfg configs/ovoador/ovoador_pretrain_vit_bert_enc32.yml \





# PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
# srun -p video -N1 -n8 --gres=gpu:8 --job-name=4m --quotatype=auto --cpus-per-task=12 \
# python -u -m main_pretrain \
#     --cfg configs/ovsegmentor/ovsegmentor_pretrain_vit_bert_stage1.yml
