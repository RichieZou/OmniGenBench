#!/bin/bash

# 超参数扰动脚本 - 基于 triclass_te.py
# 测试 OmniGenome-186M
# 在 cuda:0 上运行学习率扰动和批次大小扰动

echo "🚀 开始在 CUDA:0 上的超参数扰动实验..."

# 设置基础参数
SCRIPT_PATH="/home/yingjie/OmniGenBench/examples/9TE/tri25102025/triclass_te.py"
BASE_DIR="/data/yingjie/omni"
EPOCHS=12
DEVICE="cuda:0"

# 定义要测试的模型列表
MODELS=("OmniGenome-186M")

# 定义超参数扫描范围
LEARNING_RATES=(5e-4 5e-5 5e-6)
BATCH_SIZES=(8 4 2)  # 为OmniGenome-186M使用更小的批次大小
GRAD_ACCUMS=(64 2 1)

# ===== 学习率扰动实验 =====
echo "📱 在 CUDA:0 上运行学习率扰动实验..."

for model in "${MODELS[@]}"; do
    echo "🤖 测试模型: $model"
    
    # 学习率扰动 (固定 batch_size=16, grad_accum=8)
    for lr in "${LEARNING_RATES[@]}"; do
        echo "🔬 运行学习率: $lr (batch_size=16, grad_accum=8, model=$model)"
        python $SCRIPT_PATH \
            --epochs $EPOCHS \
            --learning_rate $lr \
            --batch_size 8 \
            --grad_accum 8 \
            --base_dir $BASE_DIR \
            --device $DEVICE \
            --model_name $model
        echo "✅ 学习率 $lr 实验完成 (模型: $model)"
        echo "----------------------------------------"
    done
done

echo "📱 在 CUDA:0 上运行批次大小扰动实验..."

# 批次大小扰动 (固定 lr=5e-5, grad_accum=8)
for model in "${MODELS[@]}"; do
    echo "🤖 测试模型: $model"
    
    for bs in "${BATCH_SIZES[@]}"; do
        echo "🔬 运行批次大小: $bs (lr=5e-5, grad_accum=8, model=$model)"
        python $SCRIPT_PATH \
            --epochs $EPOCHS \
            --learning_rate 5e-5 \
            --batch_size $bs \
            --grad_accum 8 \
            --base_dir $BASE_DIR \
            --device $DEVICE \
            --model_name $model
        echo "✅ 批次大小 $bs 实验完成 (模型: $model)"
        echo "----------------------------------------"
    done
done

echo "📱 在 CUDA:0 上运行梯度累积扰动实验..."

# 梯度累积扰动 (固定 lr=5e-5, batch_size=24)
for model in "${MODELS[@]}"; do
    echo "🤖 测试模型: $model"
    
    for grad_accum in "${GRAD_ACCUMS[@]}"; do
        echo "🔬 运行梯度累积: $grad_accum (lr=5e-5, batch_size=16, model=$model)"
        python $SCRIPT_PATH \
            --epochs $EPOCHS \
            --learning_rate 5e-5 \
            --batch_size 8 \
            --grad_accum $grad_accum \
            --base_dir $BASE_DIR \
            --device $DEVICE \
            --model_name $model
        echo "✅ 梯度累积 $grad_accum 实验完成 (模型: $model)"
        echo "----------------------------------------"
    done
done

echo "🎉 CUDA:0 上的所有超参数扰动实验完成！"
echo "📊 实验结果保存在: $BASE_DIR"
