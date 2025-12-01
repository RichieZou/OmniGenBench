#!/bin/bash

# 选择 GPU（0 或 1）
GPU_ID=0
SCRIPT="triclass_te_na_as_2_tno0_1.py"

# 需要测试的 weight_2 列表
WEIGHTS=(0.01 0.1 0.3 0.5 0.7 1.0)

# 逐个 weight 循环执行
for w in "${WEIGHTS[@]}"; do
    while true; do
        USED=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | sed -n "$((GPU_ID+1))p")

        # 判断显存是否空闲（小于 2000MiB 认为空闲）
        if [ "$USED" -lt 2000 ]; then
            echo "✅ GPU $GPU_ID 空闲 ($USED MiB)，开始运行 weight_2=$w ..."
            export CUDA_VISIBLE_DEVICES=$GPU_ID
            python "$SCRIPT" --weight_2 "$w"
            echo "🏁 weight_2=$w 任务完成！"
            break
        else
            echo "⏳ GPU $GPU_ID 正忙 ($USED MiB)，等待中..."
            sleep 20  # 每20秒检查一次
        fi
    done
done

echo "🎉 所有 weight_2 任务执行完成！"
