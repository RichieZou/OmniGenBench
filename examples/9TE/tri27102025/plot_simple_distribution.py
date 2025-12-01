#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
简单绘制标签组合分布的柱状图
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
from collections import Counter

# 设置样式
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

print("\n正在计算标签组合...")
# 将每个样本的标签组合转换为元组
def labels_to_tuple(row):
    """将一行的标签转换为元组，NA用'NA'表示"""
    result = []
    for col in label_columns:
        val = row[col]
        if pd.isna(val):
            result.append('NA')
        else:
            result.append(str(int(val)))
    return tuple(result)

# 获取每个样本的标签组合
label_combinations = df[label_columns].apply(labels_to_tuple, axis=1)

# 统计每种组合出现的次数
combination_counts = Counter(label_combinations)
print(f"实际出现的不同组合数: {len(combination_counts)} / {4**9} (理论最大值)")

# 转换为列表并排序（从大到小）
sorted_counts = sorted(combination_counts.values(), reverse=True)
print(f"最大出现次数: {sorted_counts[0]}")
print(f"最小出现次数: {sorted_counts[-1]}")
print(f"平均出现次数: {np.mean(sorted_counts):.2f}")
print(f"中位数出现次数: {np.median(sorted_counts):.0f}")

output_dir = Path('label_statistics')
output_dir.mkdir(exist_ok=True)

# 创建图表 - 完整版
print("\n正在生成完整分布图...")
fig, ax = plt.subplots(figsize=(20, 8))

x = np.arange(len(sorted_counts))
bars = ax.bar(x, sorted_counts, width=1.0, color='steelblue', alpha=0.8, edgecolor='none')

ax.set_xlabel('Label Combination Index (sorted by count, descending)', fontsize=14, fontweight='bold')
ax.set_ylabel('Sample Count', fontsize=14, fontweight='bold')
ax.set_title(f'Distribution of Label Combinations\n({len(sorted_counts):,} unique combinations, {len(df):,} total samples)', 
             fontsize=16, fontweight='bold')
ax.grid(axis='y', alpha=0.3, linestyle='--', linewidth=0.8)
ax.set_xlim(-100, len(sorted_counts)+100)

# 添加统计信息文本框
stats_text = f'Total Combinations: {len(sorted_counts):,}\n'
stats_text += f'Max Count: {sorted_counts[0]:,}\n'
stats_text += f'Min Count: {sorted_counts[-1]:,}\n'
stats_text += f'Mean: {np.mean(sorted_counts):.1f}\n'
stats_text += f'Median: {np.median(sorted_counts):.0f}'

ax.text(0.98, 0.97, stats_text, transform=ax.transAxes,
        fontsize=11, verticalalignment='top', horizontalalignment='right',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

plt.tight_layout()
plt.savefig(output_dir / '12_simple_distribution_full.png', dpi=300, bbox_inches='tight')
print(f"完整分布图已保存: {output_dir / '12_simple_distribution_full.png'}")
plt.close()

# 创建图表 - 前1000个组合
print("正在生成Top 1000组合分布图...")
fig, ax = plt.subplots(figsize=(20, 8))

top_n = min(1000, len(sorted_counts))
x_top = np.arange(top_n)
bars = ax.bar(x_top, sorted_counts[:top_n], width=1.0, color='coral', alpha=0.8, edgecolor='none')

ax.set_xlabel('Label Combination Index (Top 1000, sorted by count)', fontsize=14, fontweight='bold')
ax.set_ylabel('Sample Count', fontsize=14, fontweight='bold')
ax.set_title(f'Distribution of Top {top_n} Label Combinations', 
             fontsize=16, fontweight='bold')
ax.grid(axis='y', alpha=0.3, linestyle='--', linewidth=0.8)
ax.set_xlim(-10, top_n+10)

# 添加统计信息
coverage = sum(sorted_counts[:top_n]) / len(df) * 100
stats_text = f'Top {top_n} Combinations:\n'
stats_text += f'Coverage: {coverage:.1f}%\n'
stats_text += f'Max Count: {sorted_counts[0]:,}\n'
stats_text += f'Min in Top {top_n}: {sorted_counts[top_n-1]:,}'

ax.text(0.98, 0.97, stats_text, transform=ax.transAxes,
        fontsize=11, verticalalignment='top', horizontalalignment='right',
        bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))

plt.tight_layout()
plt.savefig(output_dir / '13_simple_distribution_top1000.png', dpi=300, bbox_inches='tight')
print(f"Top 1000分布图已保存: {output_dir / '13_simple_distribution_top1000.png'}")
plt.close()

# 创建图表 - 前100个组合（更详细）
print("正在生成Top 100组合分布图...")
fig, ax = plt.subplots(figsize=(20, 8))

top_n = min(100, len(sorted_counts))
x_top = np.arange(top_n)
bars = ax.bar(x_top, sorted_counts[:top_n], width=0.8, color='green', alpha=0.7, edgecolor='black', linewidth=0.5)

# 为前10个添加数值标签
for i in range(min(10, top_n)):
    ax.text(i, sorted_counts[i], f'{sorted_counts[i]}', 
            ha='center', va='bottom', fontsize=9, fontweight='bold')

ax.set_xlabel('Label Combination Index (Top 100, sorted by count)', fontsize=14, fontweight='bold')
ax.set_ylabel('Sample Count', fontsize=14, fontweight='bold')
ax.set_title(f'Distribution of Top {top_n} Label Combinations (with value labels for top 10)', 
             fontsize=16, fontweight='bold')
ax.grid(axis='y', alpha=0.3, linestyle='--', linewidth=0.8)
ax.set_xlim(-1, top_n+1)

# 添加统计信息
coverage = sum(sorted_counts[:top_n]) / len(df) * 100
stats_text = f'Top {top_n} Combinations:\n'
stats_text += f'Coverage: {coverage:.1f}%\n'
stats_text += f'Samples: {sum(sorted_counts[:top_n]):,} / {len(df):,}'

ax.text(0.98, 0.97, stats_text, transform=ax.transAxes,
        fontsize=11, verticalalignment='top', horizontalalignment='right',
        bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))

plt.tight_layout()
plt.savefig(output_dir / '14_simple_distribution_top100.png', dpi=300, bbox_inches='tight')
print(f"Top 100分布图已保存: {output_dir / '14_simple_distribution_top100.png'}")
plt.close()

# 创建一个综合对比图
print("正在生成综合对比图...")
fig = plt.figure(figsize=(20, 12))

# 子图1: 完整分布（线性）
ax1 = plt.subplot(2, 2, 1)
x_full = np.arange(len(sorted_counts))
ax1.bar(x_full, sorted_counts, width=1.0, color='steelblue', alpha=0.8, edgecolor='none')
ax1.set_xlabel('Combination Index', fontsize=11, fontweight='bold')
ax1.set_ylabel('Sample Count', fontsize=11, fontweight='bold')
ax1.set_title(f'Full Distribution (Linear Scale)\n{len(sorted_counts):,} combinations', fontsize=12, fontweight='bold')
ax1.grid(axis='y', alpha=0.3)

# 子图2: 完整分布（对数）
ax2 = plt.subplot(2, 2, 2)
ax2.bar(x_full, sorted_counts, width=1.0, color='coral', alpha=0.8, edgecolor='none')
ax2.set_xlabel('Combination Index', fontsize=11, fontweight='bold')
ax2.set_ylabel('Sample Count (log scale)', fontsize=11, fontweight='bold')
ax2.set_title('Full Distribution (Log Scale)', fontsize=12, fontweight='bold')
ax2.set_yscale('log')
ax2.grid(axis='y', alpha=0.3, which='both')

# 子图3: Top 1000
ax3 = plt.subplot(2, 2, 3)
top_1000 = min(1000, len(sorted_counts))
x_1000 = np.arange(top_1000)
ax3.bar(x_1000, sorted_counts[:top_1000], width=1.0, color='green', alpha=0.7, edgecolor='none')
ax3.set_xlabel('Combination Index', fontsize=11, fontweight='bold')
ax3.set_ylabel('Sample Count', fontsize=11, fontweight='bold')
coverage_1000 = sum(sorted_counts[:top_1000]) / len(df) * 100
ax3.set_title(f'Top 1000 Combinations\n(Coverage: {coverage_1000:.1f}%)', fontsize=12, fontweight='bold')
ax3.grid(axis='y', alpha=0.3)

# 子图4: Top 100
ax4 = plt.subplot(2, 2, 4)
top_100 = min(100, len(sorted_counts))
x_100 = np.arange(top_100)
bars4 = ax4.bar(x_100, sorted_counts[:top_100], width=0.8, color='purple', alpha=0.7, edgecolor='black', linewidth=0.5)
ax4.set_xlabel('Combination Index', fontsize=11, fontweight='bold')
ax4.set_ylabel('Sample Count', fontsize=11, fontweight='bold')
coverage_100 = sum(sorted_counts[:top_100]) / len(df) * 100
ax4.set_title(f'Top 100 Combinations\n(Coverage: {coverage_100:.1f}%)', fontsize=12, fontweight='bold')
ax4.grid(axis='y', alpha=0.3)

plt.suptitle('Label Combination Distribution Analysis - Multiple Views', 
             fontsize=16, fontweight='bold', y=0.995)
plt.tight_layout()
plt.savefig(output_dir / '15_distribution_comparison.png', dpi=300, bbox_inches='tight')
print(f"综合对比图已保存: {output_dir / '15_distribution_comparison.png'}")
plt.close()

print("\n" + "="*60)
print("分布图生成完成！")
print("="*60)
print(f"生成的图表:")
print(f"  1. 12_simple_distribution_full.png - 完整分布（所有{len(sorted_counts):,}个组合）")
print(f"  2. 13_simple_distribution_top1000.png - Top 1000组合")
print(f"  3. 14_simple_distribution_top100.png - Top 100组合")
print(f"  4. 15_distribution_comparison.png - 综合对比视图")
print("="*60)

