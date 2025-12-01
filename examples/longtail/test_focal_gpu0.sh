#!/bin/bash
# GPU 0 上的训练任务
# gamma = 1.0, 2.0

echo "=========================================="
echo "GPU 0 上的 Focal Loss 训练任务"
echo "测试 gamma = 1.0, 2.0"
echo "=========================================="
echo ""

# 激活 conda 环境（如果需要）
# source /home/yingjie/anaconda3/etc/profile.d/conda.sh
# conda activate omni

# GPU 0 的任务列表
gamma_values=(1.0 2.0)

for gamma in "${gamma_values[@]}"
do
    echo "=========================================="
    echo "🚀 开始训练: gamma = $gamma (GPU 0)"
    echo "=========================================="
    echo ""
    
    python biclass_focal.py --gamma $gamma --gpu 0
    
    echo ""
    echo "✅ gamma = $gamma (GPU 0) 的训练完成"
    echo ""
    echo "----------------------------------------"
    echo ""
done

echo "=========================================="
echo "🎉 GPU 0 上的所有任务完成！"
echo "=========================================="
echo ""
echo "结果保存在以下目录："
for gamma in "${gamma_values[@]}"
do
    gamma_suffix=$(echo "gamma$gamma" | sed 's/\./_/g')
    echo "  - split_8_1_1_focal_${gamma_suffix}/"
done
echo ""

