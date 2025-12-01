# -*- coding: utf-8 -*-
# file: split_data.py
# 功能：将 9tissue_dot_all.csv 按照 8:1:1 的比例划分为训练集、验证集和测试集

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# 设置随机种子以确保可重复性
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# 数据文件路径
data_dir = "/home/yingjie/OmniGenBench/examples/second_structure_for_classification/data"
input_file = os.path.join(data_dir, "9tissue_dot_all.csv")
output_dir = os.path.join(data_dir, "8_1_1")

# 创建输出目录
os.makedirs(output_dir, exist_ok=True)
print(f"📁 创建输出目录: {output_dir}")

# 读取数据
print(f"📖 正在读取数据文件: {input_file}")
df = pd.read_csv(input_file)

print(f"📊 数据总行数: {len(df)} (包含表头)")
print(f"📊 数据样本数: {len(df) - 1} (排除表头)")

# 检查必要的列是否存在
required_columns = ['ID', 'SEQ', 'DOT', 'TE', 'tissue', 'LABLE']
missing_columns = [col for col in required_columns if col not in df.columns]
if missing_columns:
    raise ValueError(f"❌ 缺少必要的列: {missing_columns}")

print(f"✅ 所有必要的列都存在")

# 显示数据基本信息
print(f"\n📋 数据列信息:")
print(f"  列名: {list(df.columns)}")
print(f"\n📊 标签分布 (LABLE 列):")
print(df['LABLE'].value_counts().sort_index())

# 移除表头行（如果有的话，确保只处理数据行）
# 如果第一行是表头，数据从第二行开始
data_df = df.copy()

# 按照 8:1:1 的比例划分
# 首先按照 8:2 划分（训练集:验证+测试集）
# 然后再将验证+测试集按照 1:1 划分

print(f"\n🔄 开始数据划分...")
print(f"   比例: 训练集 80% : 验证集 10% : 测试集 10%")

# 检查每个标签的样本数，决定是否使用分层抽样
label_counts = data_df['LABLE'].value_counts()
min_samples = label_counts.min()
print(f"   最小类别样本数: {min_samples}")

# 如果最小类别样本数 >= 2，使用分层抽样；否则不使用
use_stratify = min_samples >= 2

if use_stratify:
    print(f"   ✅ 使用分层抽样（保持类别比例）")
    # 第一次划分：训练集 80%，临时集 20%
    train_df, temp_df = train_test_split(
        data_df,
        test_size=0.2,
        random_state=RANDOM_SEED,
        stratify=data_df['LABLE']  # 按标签分层，保持类别比例
    )
    
    # 检查临时集的标签分布
    temp_label_counts = temp_df['LABLE'].value_counts()
    temp_min_samples = temp_label_counts.min()
    
    # 第二次划分：临时集 20% 分为验证集 10% 和测试集 10%
    if temp_min_samples >= 2:
        valid_df, test_df = train_test_split(
            temp_df,
            test_size=0.5,
            random_state=RANDOM_SEED,
            stratify=temp_df['LABLE']  # 按标签分层
        )
    else:
        print(f"   ⚠️  临时集最小类别样本数 ({temp_min_samples}) < 2，第二次划分不使用分层抽样")
        valid_df, test_df = train_test_split(
            temp_df,
            test_size=0.5,
            random_state=RANDOM_SEED
        )
else:
    print(f"   ⚠️  最小类别样本数 ({min_samples}) < 2，不使用分层抽样")
    # 第一次划分：训练集 80%，临时集 20%
    train_df, temp_df = train_test_split(
        data_df,
        test_size=0.2,
        random_state=RANDOM_SEED
    )
    
    # 第二次划分：临时集 20% 分为验证集 10% 和测试集 10%
    valid_df, test_df = train_test_split(
        temp_df,
        test_size=0.5,
        random_state=RANDOM_SEED
    )

print(f"\n✅ 数据划分完成:")
print(f"   训练集: {len(train_df)} 个样本 ({len(train_df)/len(data_df)*100:.1f}%)")
print(f"   验证集: {len(valid_df)} 个样本 ({len(valid_df)/len(data_df)*100:.1f}%)")
print(f"   测试集: {len(test_df)} 个样本 ({len(test_df)/len(data_df)*100:.1f}%)")

# 显示各数据集的标签分布
print(f"\n📊 训练集标签分布:")
print(train_df['LABLE'].value_counts().sort_index())
print(f"\n📊 验证集标签分布:")
print(valid_df['LABLE'].value_counts().sort_index())
print(f"\n📊 测试集标签分布:")
print(test_df['LABLE'].value_counts().sort_index())

# 保存数据
train_file = os.path.join(output_dir, "train.csv")
valid_file = os.path.join(output_dir, "valid.csv")
test_file = os.path.join(output_dir, "test.csv")

print(f"\n💾 正在保存数据...")
train_df.to_csv(train_file, index=False)
print(f"   ✅ 训练集已保存: {train_file}")

valid_df.to_csv(valid_file, index=False)
print(f"   ✅ 验证集已保存: {valid_file}")

test_df.to_csv(test_file, index=False)
print(f"   ✅ 测试集已保存: {test_file}")

print(f"\n🎉 数据划分完成！所有文件已保存到: {output_dir}")

