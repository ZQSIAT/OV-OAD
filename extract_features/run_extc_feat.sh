#!/bin/bash

# 函数名：split_integer_range
# 参数：
#   - number: 要分割的整数
#   - parts: 分割的份数
function split_integer_range() {
    local number=$1
    local parts=$2
    local script_file=$3
    # 计算每份的大小
    local increment=$((number / parts))
    
    for ((i = 0; i < parts; i++)); do
        local start=$((i * increment))
        local end=$((start + increment))
        
        # 最后一份的结束范围修正
        if ((i == parts - 1)); then
            end=$number
        fi
        srun -p xxx --job-name=extc_feats -n1 --gres=gpu:1 --cpus-per-task=16 -N1 python "$script_file" --range $start $end &
        sleep 1.0
        echo "srun -p xxx --job-name=extc_feats -n1 --gres=gpu:1 --cpus-per-task=16 -N1 python "$script_file" --range $start $end "
    done
}

split_integer_range 4115 8 "internvid_extract.py"
