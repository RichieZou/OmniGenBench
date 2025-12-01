#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
生成训练数据标签统计图表
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial Unicode MS', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False
sns.set_style("whitegrid")

# 读取数据
print("正在读取数据...")
df = pd.read_csv('train.csv')
print(f"数据集大小: {len(df)} 条样本")

# 定义标签列
label_columns = [
    'root_TE_label',
    'seedling_TE_label', 
    'leaf_TE_label',
    'FMI_TE_label',
    'FOD_TE_label',
    'Prophase-I-pollen_TE_label',
    'Tricellular-pollen_TE_label',
    'flag_TE_label',
    'grain_TE_label'
]

# 标签中文名称映射
label_names_cn = {
    'root_TE_label': 'Root (根)',
    'seedling_TE_label': 'Seedling (幼苗)',
    'leaf_TE_label': 'Leaf (叶片)',
    'FMI_TE_label': 'FMI',
    'FOD_TE_label': 'FOD',
    'Prophase-I-pollen_TE_label': 'Prophase-I Pollen (前期I花粉)',
    'Tricellular-pollen_TE_label': 'Tricellular Pollen (三细胞花粉)',
    'flag_TE_label': 'Flag (旗叶)',
    'grain_TE_label': 'Grain (籽粒)'
}

output_dir = Path('label_statistics')
output_dir.mkdir(exist_ok=True)

print("\n开始生成单个标签分布图...")

# 1. 为每个标签生成单独的分布图
for i, col in enumerate(label_columns, 1):
    print(f"[{i}/9] 生成 {col} 的分布图...")
    
    # 统计各类别数量
    value_counts = df[col].value_counts(dropna=False)
    nan_count = df[col].isna().sum()
    
    # 创建统计数据
    categories = ['0.0', '1.0', '2.0', 'Missing']
    counts = [
        value_counts.get(0.0, 0),
        value_counts.get(1.0, 0),
        value_counts.get(2.0, 0),
        nan_count
    ]
    
    # 创建图表
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # 左图：柱状图
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#CCCCCC']
    bars = ax1.bar(categories, counts, color=colors, alpha=0.8, edgecolor='black')
    
    # 添加数值标签
    for bar, count in zip(bars, counts):
        height = bar.get_height()
        percentage = (count / len(df)) * 100
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(count)}\n({percentage:.1f}%)',
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    ax1.set_xlabel('Label Value', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Count', fontsize=12, fontweight='bold')
    ax1.set_title(f'{label_names_cn[col]} - Distribution', fontsize=14, fontweight='bold')
    ax1.grid(axis='y', alpha=0.3)
    
    # 右图：饼图（排除缺失值）
    valid_categories = []
    valid_counts = []
    valid_colors = []
    for cat, count, color in zip(categories[:3], counts[:3], colors[:3]):
        if count > 0:
            valid_categories.append(cat)
            valid_counts.append(count)
            valid_colors.append(color)
    
    if valid_counts:
        wedges, texts, autotexts = ax2.pie(valid_counts, labels=valid_categories, 
                                            colors=valid_colors, autopct='%1.1f%%',
                                            startangle=90, textprops={'fontsize': 11})
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontweight('bold')
    
    ax2.set_title(f'{label_names_cn[col]} - Valid Labels Only', 
                  fontsize=14, fontweight='bold')
    
    # 添加统计信息
    valid_count = len(df) - nan_count
    info_text = f'Total: {len(df)}\nValid: {valid_count}\nMissing: {nan_count}'
    fig.text(0.98, 0.02, info_text, ha='right', va='bottom', 
             fontsize=10, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(output_dir / f'{i}_{col}_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()

print("\n开始生成联合标签统计图...")

# 2. 联合标签统计
# 统计每个样本有多少个非空标签
df['num_valid_labels'] = df[label_columns].notna().sum(axis=1)

# 统计每个样本中各类别的数量
df['num_class_0'] = (df[label_columns] == 0.0).sum(axis=1)
df['num_class_1'] = (df[label_columns] == 1.0).sum(axis=1)
df['num_class_2'] = (df[label_columns] == 2.0).sum(axis=1)

# 创建联合统计图（2x2布局）
fig = plt.figure(figsize=(16, 12))

# 子图1：每个样本的有效标签数量分布
ax1 = plt.subplot(2, 2, 1)
valid_label_counts = df['num_valid_labels'].value_counts().sort_index()
bars1 = ax1.bar(valid_label_counts.index, valid_label_counts.values, 
                color='steelblue', alpha=0.7, edgecolor='black')
for bar in bars1:
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height,
            f'{int(height)}',
            ha='center', va='bottom', fontsize=9, fontweight='bold')
ax1.set_xlabel('Number of Valid Labels per Sample', fontsize=11, fontweight='bold')
ax1.set_ylabel('Count', fontsize=11, fontweight='bold')
ax1.set_title('Distribution of Valid Labels per Sample', fontsize=13, fontweight='bold')
ax1.grid(axis='y', alpha=0.3)

# 子图2：每个标签列的完整度
ax2 = plt.subplot(2, 2, 2)
completeness = []
for col in label_columns:
    valid_ratio = (1 - df[col].isna().sum() / len(df)) * 100
    completeness.append(valid_ratio)

label_names_short = [label_names_cn[col].split('(')[0].strip() for col in label_columns]
bars2 = ax2.barh(label_names_short, completeness, color='coral', alpha=0.7, edgecolor='black')
for i, (bar, val) in enumerate(zip(bars2, completeness)):
    ax2.text(val, bar.get_y() + bar.get_height()/2.,
            f'{val:.1f}%',
            ha='left', va='center', fontsize=9, fontweight='bold', 
            bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.3))
ax2.set_xlabel('Completeness (%)', fontsize=11, fontweight='bold')
ax2.set_title('Label Completeness by Tissue Type', fontsize=13, fontweight='bold')
ax2.set_xlim(0, 105)
ax2.grid(axis='x', alpha=0.3)

# 子图3：总体标签值分布热图
ax3 = plt.subplot(2, 2, 3)
label_dist_matrix = []
for col in label_columns:
    counts = [
        (df[col] == 0.0).sum(),
        (df[col] == 1.0).sum(),
        (df[col] == 2.0).sum(),
        df[col].isna().sum()
    ]
    label_dist_matrix.append(counts)

label_dist_matrix = np.array(label_dist_matrix)
im = ax3.imshow(label_dist_matrix, cmap='YlOrRd', aspect='auto')
ax3.set_xticks(range(4))
ax3.set_xticklabels(['0.0', '1.0', '2.0', 'Missing'], fontsize=10)
ax3.set_yticks(range(len(label_columns)))
ax3.set_yticklabels(label_names_short, fontsize=9)
ax3.set_title('Label Distribution Heatmap', fontsize=13, fontweight='bold')

# 添加数值标注
for i in range(len(label_columns)):
    for j in range(4):
        text = ax3.text(j, i, int(label_dist_matrix[i, j]),
                       ha="center", va="center", color="black", fontsize=8)

plt.colorbar(im, ax=ax3, label='Count')

# 子图4：标签值组合的整体分布
ax4 = plt.subplot(2, 2, 4)
class_counts = {
    'Class 0.0': df['num_class_0'].sum(),
    'Class 1.0': df['num_class_1'].sum(),
    'Class 2.0': df['num_class_2'].sum(),
    'Missing': df[label_columns].isna().sum().sum()
}

colors_pie = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#CCCCCC']
wedges, texts, autotexts = ax4.pie(class_counts.values(), labels=class_counts.keys(),
                                     colors=colors_pie, autopct='%1.1f%%',
                                     startangle=90, textprops={'fontsize': 11})
for autotext in autotexts:
    autotext.set_color('white')
    autotext.set_fontweight('bold')

# 添加图例
total_labels = sum(class_counts.values())
legend_labels = [f'{k}: {v:,} ({v/total_labels*100:.1f}%)' 
                 for k, v in class_counts.items()]
ax4.legend(legend_labels, loc='upper left', bbox_to_anchor=(1, 1), fontsize=9)
ax4.set_title('Overall Label Value Distribution', fontsize=13, fontweight='bold')

plt.suptitle('Joint Label Statistics Analysis', fontsize=16, fontweight='bold', y=0.995)
plt.tight_layout()
plt.savefig(output_dir / '10_joint_label_statistics.png', dpi=300, bbox_inches='tight')
plt.close()

print("\n完成！所有图表已保存到 label_statistics/ 目录")

# 生成统计报告
print("\n" + "="*60)
print("标签统计摘要")
print("="*60)
print(f"总样本数: {len(df):,}")
print(f"\n各标签列的完整度:")
for col, comp in zip(label_columns, completeness):
    print(f"  {label_names_cn[col]:35s}: {comp:5.1f}%")

print(f"\n整体标签值分布:")
for k, v in class_counts.items():
    print(f"  {k:15s}: {v:8,} ({v/total_labels*100:5.1f}%)")

print(f"\n每个样本的有效标签数量分布:")
for num, count in valid_label_counts.items():
    print(f"  {num} 个标签: {count:6,} 样本 ({count/len(df)*100:5.1f}%)")

print("="*60)

