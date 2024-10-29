PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
# --quotatype=spot
# --cfg configs/ovoador/3layers/enc32_anet_3layers_lsxattn.yml \
# debug --pty  -n2 others -n8 --gres=gpu:8
# srun -p s2_bigdata --pty --preempt -N1 -n2 --gres=gpu:2 --job-name=internvid-ovoad --quotatype=spot --cpus-per-task=12 \
srun -p s2_bigdata --preempt -N1 -n8 --gres=gpu:8 --job-name=internvid-ovoad --quotatype=auto --cpus-per-task=12 \
python -u -m main_pretrain \
    --cfg configs/ovoador/3layers/enc32_anet_3layers_lsxattn_Q5.yml \

