# CUDA 配置
export CUDA_VISIBLE_DEVICES=0
export CUDA_LAUNCH_BLOCKING=0

export NCCL_SOCKET_IFNAME=eth2
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=0
export NCCL_SHM_DISABLE=0
export NCCL_DEBUG=WARN
export NCCL_TIMEOUT=1800
export NCCL_SOCKET_NTHREADS=8
export NCCL_NSOCKS_PERTHREAD=8

nnodes=1
nproc_per_node=1  # 与GPU数量匹配
node_rank=0
master_addr="localhost"
master_port=55618

torchrun \
--nnodes=$nnodes \
--nproc_per_node=$nproc_per_node \
--node_rank=$node_rank \
--master_addr=$master_addr \
--master_port=$master_port \
main_pretrain.py \
--cfg configs/ovoador/token_strategy/configs/enc32_anet_3layers_lsxattn_only_inference.yml
