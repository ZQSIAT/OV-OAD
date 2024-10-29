PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
srun -p Gvlab-S1 -N1 -n4 --gres=gpu:4 --job-name=internvid-ovoad  --cpus-per-task=12 \
python -u -m main_pretrain \
    --cfg configs/ovoador/enc128_anet_withbg.yml \
