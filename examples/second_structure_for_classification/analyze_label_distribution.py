#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
分析9tissue_dot_all.csv文件的标签分布
- 以序列(SEQ)为主的label分布
- 以dot(DOT)为主的label分布
- 按组织(tissue)的label分布
- 整体label分布
- 序列-结构组合的label分布
"""

import pandas as pd
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial Unicode MS', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 读取数据
print("正在读取数据...")
df = pd.read_csv('/home/yingjie/OmniGenBench/examples/second_structure_for_classification/data/9tissue_dot_all.csv')

print(f"总数据量: {len(df):,} 行")
print(f"列名: {df.columns.tolist()}")

# 检查LABLE列的情况
print(f"\nLABLE列的唯一值: {df['LABLE'].unique()}")
print(f"LABLE列的数据类型: {df['LABLE'].dtype}")

# 将LABLE转换为字符串以便处理空值
df['LABLE'] = df['LABLE'].astype(str)
df['LABLE'] = df['LABLE'].replace('nan', 'Empty')
df['LABLE'] = df['LABLE'].replace('', 'Empty')
df['LABLE'] = df['LABLE'].replace(' ', 'Empty')
df['LABLE'] = df['LABLE'].str.strip()
df['LABLE'] = df['LABLE'].replace('', 'Empty')

# 创建输出目录
output_dir = Path('/home/yingjie/OmniGenBench/examples/second_structure_for_classification/label_distribution_analysis')
output_dir.mkdir(exist_ok=True)

# ==================== 1. 整体标签分布 ====================
print("\n" + "="*60)
print("1. 整体标签分布")
print("="*60)

overall_dist = df['LABLE'].value_counts()
print("\n整体标签分布:")
for label, count in overall_dist.items():
    percentage = count / len(df) * 100
    print(f"  {label}: {count:,} ({percentage:.2f}%)")

# 可视化整体分布
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# 柱状图
colors = {'0': '#FF6B6B', '1': '#4ECDC4', 'Empty': '#CCCCCC'}
bars = ax1.bar(overall_dist.index, overall_dist.values, 
               color=[colors.get(str(label), '#95A5A6') for label in overall_dist.index],
               alpha=0.8, edgecolor='black')
for bar, count in zip(bars, overall_dist.values):
    height = bar.get_height()
    percentage = count / len(df) * 100
    ax1.text(bar.get_x() + bar.get_width()/2., height,
            f'{int(count):,}\n({percentage:.1f}%)',
            ha='center', va='bottom', fontsize=11, fontweight='bold')
ax1.set_xlabel('Label', fontsize=12, fontweight='bold')
ax1.set_ylabel('Count', fontsize=12, fontweight='bold')
ax1.set_title('Overall Label Distribution', fontsize=14, fontweight='bold')
ax1.grid(axis='y', alpha=0.3)

# 饼图
wedges, texts, autotexts = ax2.pie(overall_dist.values, labels=overall_dist.index,
                                   autopct='%1.1f%%', startangle=90,
                                   colors=[colors.get(str(label), '#95A5A6') for label in overall_dist.index])
for autotext in autotexts:
    autotext.set_color('white')
    autotext.set_fontweight('bold')
ax2.set_title('Overall Label Distribution (Pie Chart)', fontsize=14, fontweight='bold')

plt.tight_layout()
plt.savefig(output_dir / '01_overall_label_distribution.png', dpi=300, bbox_inches='tight')
plt.close()
print(f"已保存: {output_dir / '01_overall_label_distribution.png'}")

# ==================== 2. 以序列(SEQ)为主的label分布 ====================
print("\n" + "="*60)
print("2. 以序列(SEQ)为主的label分布")
print("="*60)

# 按SEQ分组统计
def get_label_stats(x):
    stats = {
        'total': len(x),
        'label_0': int((x == '0').sum()),
        'label_1': int((x == '1').sum()),
        'label_empty': int((x == 'Empty').sum()),
        'label_distribution': dict(Counter(x))
    }
    return stats

seq_label_stats_list = []
for seq, group in df.groupby('SEQ')['LABLE']:
    stats = get_label_stats(group)
    seq_label_stats_list.append({'SEQ': seq, 'stats': stats})

seq_label_stats = pd.DataFrame(seq_label_stats_list)

print(f"唯一序列数量: {len(seq_label_stats):,}")

# 统计每个序列的标签分布情况
def has_multiple_labels(stats_dict):
    if isinstance(stats_dict, dict):
        return len(stats_dict.get('label_distribution', {})) > 1
    return False

seq_with_multiple_labels = seq_label_stats[seq_label_stats['stats'].apply(has_multiple_labels)]
print(f"具有多种标签的序列数量: {len(seq_with_multiple_labels):,}")

# 统计每个序列的主要标签
seq_main_labels = []
for idx, row in seq_label_stats.iterrows():
    dist = row['stats']
    if not isinstance(dist, dict):
        continue
    if 'label_distribution' not in dist or len(dist['label_distribution']) == 0:
        continue
    main_label = max(dist['label_distribution'].items(), key=lambda x: x[1])[0]
    seq_main_labels.append({
        'SEQ': row['SEQ'],
        'main_label': main_label,
        'total_count': dist.get('total', 0),
        'label_0_count': dist.get('label_0', 0),
        'label_1_count': dist.get('label_1', 0),
        'label_empty_count': dist.get('label_empty', 0)
    })

seq_main_df = pd.DataFrame(seq_main_labels)
seq_main_label_dist = seq_main_df['main_label'].value_counts()

print("\n以序列为主的主要标签分布:")
for label, count in seq_main_label_dist.items():
    percentage = count / len(seq_main_df) * 100
    print(f"  {label}: {count:,} 个序列 ({percentage:.2f}%)")

# 可视化
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# 子图1: 序列的主要标签分布
ax1 = axes[0, 0]
bars = ax1.bar(seq_main_label_dist.index, seq_main_label_dist.values,
               color=[colors.get(str(label), '#95A5A6') for label in seq_main_label_dist.index],
               alpha=0.8, edgecolor='black')
for bar, count in zip(bars, seq_main_label_dist.values):
    height = bar.get_height()
    percentage = count / len(seq_main_df) * 100
    ax1.text(bar.get_x() + bar.get_width()/2., height,
            f'{int(count):,}\n({percentage:.1f}%)',
            ha='center', va='bottom', fontsize=10, fontweight='bold')
ax1.set_xlabel('Main Label', fontsize=12, fontweight='bold')
ax1.set_ylabel('Number of Sequences', fontsize=12, fontweight='bold')
ax1.set_title('Main Label Distribution by Sequence', fontsize=13, fontweight='bold')
ax1.grid(axis='y', alpha=0.3)

# 子图2: 序列的标签一致性分布
ax2 = axes[0, 1]
seq_label_consistency = []
for idx, row in seq_label_stats.iterrows():
    dist = row['stats']
    if not isinstance(dist, dict) or 'label_distribution' not in dist:
        continue
    if dist.get('total', 0) == 0:
        continue
    max_count = max(dist['label_distribution'].values())
    consistency = max_count / dist['total']
    seq_label_consistency.append(consistency)

ax2.hist(seq_label_consistency, bins=50, color='steelblue', alpha=0.7, edgecolor='black')
ax2.axvline(np.mean(seq_label_consistency), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(seq_label_consistency):.3f}')
ax2.set_xlabel('Label Consistency (Main Label Ratio in Same Sequence)', fontsize=11, fontweight='bold')
ax2.set_ylabel('Number of Sequences', fontsize=12, fontweight='bold')
ax2.set_title('Sequence Label Consistency Distribution', fontsize=13, fontweight='bold')
ax2.legend()
ax2.grid(axis='y', alpha=0.3)

# 子图3: 每个序列的样本数量分布
ax3 = axes[1, 0]
seq_sample_counts = seq_label_stats['stats'].apply(lambda x: x['total'] if isinstance(x, dict) else 0)
ax3.hist(seq_sample_counts, bins=50, color='green', alpha=0.7, edgecolor='black')
ax3.set_xlabel('Number of Samples per Sequence', fontsize=12, fontweight='bold')
ax3.set_ylabel('Number of Sequences', fontsize=12, fontweight='bold')
ax3.set_title('Sample Count Distribution per Sequence', fontsize=13, fontweight='bold')
ax3.grid(axis='y', alpha=0.3)

# 子图4: 序列标签分布的详细统计
ax4 = axes[1, 1]
# 统计每个序列中0,1,空标签的数量
label_0_counts = seq_label_stats['stats'].apply(lambda x: x.get('label_0', 0) if isinstance(x, dict) else 0).sum()
label_1_counts = seq_label_stats['stats'].apply(lambda x: x.get('label_1', 0) if isinstance(x, dict) else 0).sum()
label_empty_counts = seq_label_stats['stats'].apply(lambda x: x.get('label_empty', 0) if isinstance(x, dict) else 0).sum()

label_counts = [label_0_counts, label_1_counts, label_empty_counts]
label_names = ['0', '1', 'Empty']
bars = ax4.bar(label_names, label_counts,
               color=[colors.get(str(label), '#95A5A6') for label in label_names],
               alpha=0.8, edgecolor='black')
for bar, count in zip(bars, label_counts):
    height = bar.get_height()
    percentage = count / sum(label_counts) * 100
    ax4.text(bar.get_x() + bar.get_width()/2., height,
            f'{int(count):,}\n({percentage:.1f}%)',
            ha='center', va='bottom', fontsize=11, fontweight='bold')
ax4.set_xlabel('Label', fontsize=12, fontweight='bold')
ax4.set_ylabel('Total Sample Count', fontsize=12, fontweight='bold')
ax4.set_title('All Label Statistics by Sequence', fontsize=13, fontweight='bold')
ax4.grid(axis='y', alpha=0.3)

plt.suptitle('Label Distribution Analysis by Sequence (SEQ)', fontsize=16, fontweight='bold', y=0.995)
plt.tight_layout()
plt.savefig(output_dir / '02_seq_based_label_distribution.png', dpi=300, bbox_inches='tight')
plt.close()
print(f"已保存: {output_dir / '02_seq_based_label_distribution.png'}")

# ==================== 3. 以dot(DOT)为主的label分布 ====================
print("\n" + "="*60)
print("3. 以dot(DOT)为主的label分布")
print("="*60)

# 按DOT分组统计
dot_label_stats_list = []
for dot, group in df.groupby('DOT')['LABLE']:
    stats = get_label_stats(group)
    dot_label_stats_list.append({'DOT': dot, 'stats': stats})

dot_label_stats = pd.DataFrame(dot_label_stats_list)

print(f"唯一结构数量: {len(dot_label_stats):,}")

# 统计每个结构的标签分布情况
dot_with_multiple_labels = dot_label_stats[dot_label_stats['stats'].apply(has_multiple_labels)]
print(f"具有多种标签的结构数量: {len(dot_with_multiple_labels):,}")

# 统计每个结构的主要标签
dot_main_labels = []
for idx, row in dot_label_stats.iterrows():
    dist = row['stats']
    if not isinstance(dist, dict):
        continue
    if 'label_distribution' not in dist or len(dist['label_distribution']) == 0:
        continue
    main_label = max(dist['label_distribution'].items(), key=lambda x: x[1])[0]
    dot_main_labels.append({
        'DOT': row['DOT'],
        'main_label': main_label,
        'total_count': dist.get('total', 0),
        'label_0_count': dist.get('label_0', 0),
        'label_1_count': dist.get('label_1', 0),
        'label_empty_count': dist.get('label_empty', 0)
    })

dot_main_df = pd.DataFrame(dot_main_labels)
dot_main_label_dist = dot_main_df['main_label'].value_counts()

print("\n以结构为主的主要标签分布:")
for label, count in dot_main_label_dist.items():
    percentage = count / len(dot_main_df) * 100
    print(f"  {label}: {count:,} 个结构 ({percentage:.2f}%)")

# 可视化
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# 子图1: 结构的主要标签分布
ax1 = axes[0, 0]
bars = ax1.bar(dot_main_label_dist.index, dot_main_label_dist.values,
               color=[colors.get(str(label), '#95A5A6') for label in dot_main_label_dist.index],
               alpha=0.8, edgecolor='black')
for bar, count in zip(bars, dot_main_label_dist.values):
    height = bar.get_height()
    percentage = count / len(dot_main_df) * 100
    ax1.text(bar.get_x() + bar.get_width()/2., height,
            f'{int(count):,}\n({percentage:.1f}%)',
            ha='center', va='bottom', fontsize=10, fontweight='bold')
ax1.set_xlabel('Main Label', fontsize=12, fontweight='bold')
ax1.set_ylabel('Number of Structures', fontsize=12, fontweight='bold')
ax1.set_title('Main Label Distribution by Structure', fontsize=13, fontweight='bold')
ax1.grid(axis='y', alpha=0.3)

# 子图2: 结构的标签一致性分布
ax2 = axes[0, 1]
dot_label_consistency = []
for idx, row in dot_label_stats.iterrows():
    dist = row['stats']
    if not isinstance(dist, dict) or 'label_distribution' not in dist:
        continue
    if dist.get('total', 0) == 0:
        continue
    max_count = max(dist['label_distribution'].values())
    consistency = max_count / dist['total']
    dot_label_consistency.append(consistency)

ax2.hist(dot_label_consistency, bins=50, color='steelblue', alpha=0.7, edgecolor='black')
ax2.axvline(np.mean(dot_label_consistency), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(dot_label_consistency):.3f}')
ax2.set_xlabel('Label Consistency (Main Label Ratio in Same Structure)', fontsize=11, fontweight='bold')
ax2.set_ylabel('Number of Structures', fontsize=12, fontweight='bold')
ax2.set_title('Structure Label Consistency Distribution', fontsize=13, fontweight='bold')
ax2.legend()
ax2.grid(axis='y', alpha=0.3)

# 子图3: 每个结构的样本数量分布
ax3 = axes[1, 0]
dot_sample_counts = dot_label_stats['stats'].apply(lambda x: x['total'] if isinstance(x, dict) else 0)
ax3.hist(dot_sample_counts, bins=50, color='green', alpha=0.7, edgecolor='black')
ax3.set_xlabel('Number of Samples per Structure', fontsize=12, fontweight='bold')
ax3.set_ylabel('Number of Structures', fontsize=12, fontweight='bold')
ax3.set_title('Sample Count Distribution per Structure', fontsize=13, fontweight='bold')
ax3.grid(axis='y', alpha=0.3)

# 子图4: 结构标签分布的详细统计
ax4 = axes[1, 1]
# 统计每个结构中0,1,空标签的数量
label_0_counts = dot_label_stats['stats'].apply(lambda x: x.get('label_0', 0) if isinstance(x, dict) else 0).sum()
label_1_counts = dot_label_stats['stats'].apply(lambda x: x.get('label_1', 0) if isinstance(x, dict) else 0).sum()
label_empty_counts = dot_label_stats['stats'].apply(lambda x: x.get('label_empty', 0) if isinstance(x, dict) else 0).sum()

label_counts = [label_0_counts, label_1_counts, label_empty_counts]
label_names = ['0', '1', 'Empty']
bars = ax4.bar(label_names, label_counts,
               color=[colors.get(str(label), '#95A5A6') for label in label_names],
               alpha=0.8, edgecolor='black')
for bar, count in zip(bars, label_counts):
    height = bar.get_height()
    percentage = count / sum(label_counts) * 100
    ax4.text(bar.get_x() + bar.get_width()/2., height,
            f'{int(count):,}\n({percentage:.1f}%)',
            ha='center', va='bottom', fontsize=11, fontweight='bold')
ax4.set_xlabel('Label', fontsize=12, fontweight='bold')
ax4.set_ylabel('Total Sample Count', fontsize=12, fontweight='bold')
ax4.set_title('All Label Statistics by Structure', fontsize=13, fontweight='bold')
ax4.grid(axis='y', alpha=0.3)

plt.suptitle('Label Distribution Analysis by Structure (DOT)', fontsize=16, fontweight='bold', y=0.995)
plt.tight_layout()
plt.savefig(output_dir / '03_dot_based_label_distribution.png', dpi=300, bbox_inches='tight')
plt.close()
print(f"已保存: {output_dir / '03_dot_based_label_distribution.png'}")

# ==================== 4. 按组织(tissue)的label分布 ====================
print("\n" + "="*60)
print("4. 按组织(tissue)的label分布")
print("="*60)

tissue_label_dist = df.groupby(['tissue', 'LABLE']).size().unstack(fill_value=0)
print("\n各组织的标签分布:")
print(tissue_label_dist)

# 计算每个组织的标签比例
tissue_label_percent = tissue_label_dist.div(tissue_label_dist.sum(axis=1), axis=0) * 100
print("\n各组织的标签比例 (%):")
print(tissue_label_percent.round(2))

# 可视化
fig, axes = plt.subplots(2, 2, figsize=(18, 12))

# 子图1: 各组织的标签数量堆叠柱状图
ax1 = axes[0, 0]
tissue_label_dist.plot(kind='bar', stacked=True, ax=ax1, 
                        color=[colors.get(str(col), '#95A5A6') for col in tissue_label_dist.columns],
                        alpha=0.8, edgecolor='black')
ax1.set_xlabel('Tissue', fontsize=12, fontweight='bold')
ax1.set_ylabel('Sample Count', fontsize=12, fontweight='bold')
ax1.set_title('Label Count Distribution by Tissue', fontsize=13, fontweight='bold')
ax1.legend(title='Label', fontsize=10)
ax1.tick_params(axis='x', rotation=45)
ax1.grid(axis='y', alpha=0.3)

# 子图2: 各组织的标签比例堆叠柱状图
ax2 = axes[0, 1]
tissue_label_percent.plot(kind='bar', stacked=True, ax=ax2,
                          color=[colors.get(str(col), '#95A5A6') for col in tissue_label_percent.columns],
                          alpha=0.8, edgecolor='black')
ax2.set_xlabel('Tissue', fontsize=12, fontweight='bold')
ax2.set_ylabel('Percentage (%)', fontsize=12, fontweight='bold')
ax2.set_title('Label Percentage Distribution by Tissue', fontsize=13, fontweight='bold')
ax2.legend(title='Label', fontsize=10)
ax2.tick_params(axis='x', rotation=45)
ax2.grid(axis='y', alpha=0.3)

# 子图3: 热力图 - 各组织的标签数量
ax3 = axes[1, 0]
sns.heatmap(tissue_label_dist, annot=True, fmt='d', cmap='YlOrRd', ax=ax3,
            cbar_kws={'label': 'Sample Count'}, linewidths=0.5, linecolor='gray')
ax3.set_xlabel('Label', fontsize=12, fontweight='bold')
ax3.set_ylabel('Tissue', fontsize=12, fontweight='bold')
ax3.set_title('Label Distribution Heatmap by Tissue (Count)', fontsize=13, fontweight='bold')

# 子图4: 热力图 - 各组织的标签比例
ax4 = axes[1, 1]
sns.heatmap(tissue_label_percent, annot=True, fmt='.1f', cmap='YlOrRd', ax=ax4,
            cbar_kws={'label': 'Percentage (%)'}, linewidths=0.5, linecolor='gray')
ax4.set_xlabel('Label', fontsize=12, fontweight='bold')
ax4.set_ylabel('Tissue', fontsize=12, fontweight='bold')
ax4.set_title('Label Distribution Heatmap by Tissue (Percentage)', fontsize=13, fontweight='bold')

plt.suptitle('Label Distribution Analysis by Tissue', fontsize=16, fontweight='bold', y=0.995)
plt.tight_layout()
plt.savefig(output_dir / '04_tissue_based_label_distribution.png', dpi=300, bbox_inches='tight')
plt.close()
print(f"已保存: {output_dir / '04_tissue_based_label_distribution.png'}")

# ==================== 5. 序列-结构组合的label分布 ====================
print("\n" + "="*60)
print("5. 序列-结构组合(SEQ-DOT)的label分布")
print("="*60)

# 按SEQ-DOT组合分组统计
seq_dot_label_stats_list = []
for (seq, dot), group in df.groupby(['SEQ', 'DOT'])['LABLE']:
    stats = get_label_stats(group)
    seq_dot_label_stats_list.append({'SEQ': seq, 'DOT': dot, 'LABLE': stats})

seq_dot_label_stats = pd.DataFrame(seq_dot_label_stats_list)

print(f"唯一序列-结构组合数量: {len(seq_dot_label_stats):,}")

# 统计每个组合的主要标签
seq_dot_main_labels = []
for idx, row in seq_dot_label_stats.iterrows():
    dist = row['LABLE']
    if not isinstance(dist, dict):
        continue
    if 'label_distribution' not in dist or len(dist['label_distribution']) == 0:
        continue
    main_label = max(dist['label_distribution'].items(), key=lambda x: x[1])[0]
    seq_dot_main_labels.append({
        'SEQ': row['SEQ'],
        'DOT': row['DOT'],
        'main_label': main_label,
        'total_count': dist.get('total', 0),
        'label_0_count': dist.get('label_0', 0),
        'label_1_count': dist.get('label_1', 0),
        'label_empty_count': dist.get('label_empty', 0)
    })

seq_dot_main_df = pd.DataFrame(seq_dot_main_labels)
seq_dot_main_label_dist = seq_dot_main_df['main_label'].value_counts()

print("\n以序列-结构组合为主的主要标签分布:")
for label, count in seq_dot_main_label_dist.items():
    percentage = count / len(seq_dot_main_df) * 100
    print(f"  {label}: {count:,} 个组合 ({percentage:.2f}%)")

# 可视化
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# 子图1: 组合的主要标签分布
ax1 = axes[0, 0]
bars = ax1.bar(seq_dot_main_label_dist.index, seq_dot_main_label_dist.values,
               color=[colors.get(str(label), '#95A5A6') for label in seq_dot_main_label_dist.index],
               alpha=0.8, edgecolor='black')
for bar, count in zip(bars, seq_dot_main_label_dist.values):
    height = bar.get_height()
    percentage = count / len(seq_dot_main_df) * 100
    ax1.text(bar.get_x() + bar.get_width()/2., height,
            f'{int(count):,}\n({percentage:.1f}%)',
            ha='center', va='bottom', fontsize=10, fontweight='bold')
ax1.set_xlabel('Main Label', fontsize=12, fontweight='bold')
ax1.set_ylabel('Number of Combinations', fontsize=12, fontweight='bold')
ax1.set_title('Main Label Distribution by SEQ-DOT Combination', fontsize=13, fontweight='bold')
ax1.grid(axis='y', alpha=0.3)

# 子图2: 组合的标签一致性分布
ax2 = axes[0, 1]
seq_dot_label_consistency = []
for idx, row in seq_dot_label_stats.iterrows():
    dist = row['LABLE']
    if not isinstance(dist, dict) or 'label_distribution' not in dist:
        continue
    if dist.get('total', 0) == 0:
        continue
    max_count = max(dist['label_distribution'].values())
    consistency = max_count / dist['total']
    seq_dot_label_consistency.append(consistency)

ax2.hist(seq_dot_label_consistency, bins=50, color='steelblue', alpha=0.7, edgecolor='black')
ax2.axvline(np.mean(seq_dot_label_consistency), color='red', linestyle='--', linewidth=2, 
            label=f'Mean: {np.mean(seq_dot_label_consistency):.3f}')
ax2.set_xlabel('Label Consistency (Main Label Ratio in Same Combination)', fontsize=11, fontweight='bold')
ax2.set_ylabel('Number of Combinations', fontsize=12, fontweight='bold')
ax2.set_title('SEQ-DOT Combination Label Consistency Distribution', fontsize=13, fontweight='bold')
ax2.legend()
ax2.grid(axis='y', alpha=0.3)

# 子图3: 每个组合的样本数量分布
ax3 = axes[1, 0]
seq_dot_sample_counts = seq_dot_label_stats['LABLE'].apply(lambda x: x.get('total', 0) if isinstance(x, dict) else 0)
ax3.hist(seq_dot_sample_counts, bins=50, color='green', alpha=0.7, edgecolor='black')
ax3.set_xlabel('Number of Samples per Combination', fontsize=12, fontweight='bold')
ax3.set_ylabel('Number of Combinations', fontsize=12, fontweight='bold')
ax3.set_title('Sample Count Distribution per SEQ-DOT Combination', fontsize=13, fontweight='bold')
ax3.grid(axis='y', alpha=0.3)

# 子图4: 组合标签分布的详细统计
ax4 = axes[1, 1]
label_0_counts = seq_dot_label_stats['LABLE'].apply(lambda x: x.get('label_0', 0) if isinstance(x, dict) else 0).sum()
label_1_counts = seq_dot_label_stats['LABLE'].apply(lambda x: x.get('label_1', 0) if isinstance(x, dict) else 0).sum()
label_empty_counts = seq_dot_label_stats['LABLE'].apply(lambda x: x.get('label_empty', 0) if isinstance(x, dict) else 0).sum()

label_counts = [label_0_counts, label_1_counts, label_empty_counts]
label_names = ['0', '1', 'Empty']
bars = ax4.bar(label_names, label_counts,
               color=[colors.get(str(label), '#95A5A6') for label in label_names],
               alpha=0.8, edgecolor='black')
for bar, count in zip(bars, label_counts):
    height = bar.get_height()
    percentage = count / sum(label_counts) * 100
    ax4.text(bar.get_x() + bar.get_width()/2., height,
            f'{int(count):,}\n({percentage:.1f}%)',
            ha='center', va='bottom', fontsize=11, fontweight='bold')
ax4.set_xlabel('Label', fontsize=12, fontweight='bold')
ax4.set_ylabel('Total Sample Count', fontsize=12, fontweight='bold')
ax4.set_title('All Label Statistics by SEQ-DOT Combination', fontsize=13, fontweight='bold')
ax4.grid(axis='y', alpha=0.3)

plt.suptitle('Label Distribution Analysis by SEQ-DOT Combination', fontsize=16, fontweight='bold', y=0.995)
plt.tight_layout()
plt.savefig(output_dir / '05_seq_dot_combined_label_distribution.png', dpi=300, bbox_inches='tight')
plt.close()
print(f"已保存: {output_dir / '05_seq_dot_combined_label_distribution.png'}")

# ==================== 6. 综合统计报告 ====================
print("\n" + "="*60)
print("6. 综合统计报告")
print("="*60)

report_file = output_dir / 'label_distribution_report.txt'
with open(report_file, 'w', encoding='utf-8') as f:
    f.write("="*60 + "\n")
    f.write("标签分布统计报告\n")
    f.write("="*60 + "\n\n")
    
    f.write(f"总数据量: {len(df):,} 行\n")
    f.write(f"唯一序列数量: {len(seq_label_stats):,}\n")
    f.write(f"唯一结构数量: {len(dot_label_stats):,}\n")
    f.write(f"唯一序列-结构组合数量: {len(seq_dot_label_stats):,}\n")
    f.write(f"组织数量: {df['tissue'].nunique()}\n")
    f.write(f"组织列表: {', '.join(sorted(df['tissue'].unique()))}\n\n")
    
    f.write("="*60 + "\n")
    f.write("1. 整体标签分布\n")
    f.write("="*60 + "\n")
    for label, count in overall_dist.items():
        percentage = count / len(df) * 100
        f.write(f"  {label}: {count:,} ({percentage:.2f}%)\n")
    
    f.write("\n" + "="*60 + "\n")
    f.write("2. 以序列(SEQ)为主的标签分布\n")
    f.write("="*60 + "\n")
    f.write(f"唯一序列数量: {len(seq_label_stats):,}\n")
    f.write(f"具有多种标签的序列数量: {len(seq_with_multiple_labels):,}\n")
    f.write(f"序列标签一致性均值: {np.mean(seq_label_consistency):.3f}\n")
    f.write(f"每个序列的平均样本数: {seq_sample_counts.mean():.2f}\n")
    f.write("\n序列的主要标签分布:\n")
    for label, count in seq_main_label_dist.items():
        percentage = count / len(seq_main_df) * 100
        f.write(f"  {label}: {count:,} 个序列 ({percentage:.2f}%)\n")
    
    f.write("\n" + "="*60 + "\n")
    f.write("3. 以结构(DOT)为主的标签分布\n")
    f.write("="*60 + "\n")
    f.write(f"唯一结构数量: {len(dot_label_stats):,}\n")
    f.write(f"具有多种标签的结构数量: {len(dot_with_multiple_labels):,}\n")
    f.write(f"结构标签一致性均值: {np.mean(dot_label_consistency):.3f}\n")
    f.write(f"每个结构的平均样本数: {dot_sample_counts.mean():.2f}\n")
    f.write("\n结构的主要标签分布:\n")
    for label, count in dot_main_label_dist.items():
        percentage = count / len(dot_main_df) * 100
        f.write(f"  {label}: {count:,} 个结构 ({percentage:.2f}%)\n")
    
    f.write("\n" + "="*60 + "\n")
    f.write("4. 按组织(tissue)的标签分布\n")
    f.write("="*60 + "\n")
    f.write("\n各组织的标签数量:\n")
    f.write(tissue_label_dist.to_string())
    f.write("\n\n各组织的标签比例 (%):\n")
    f.write(tissue_label_percent.round(2).to_string())
    
    f.write("\n\n" + "="*60 + "\n")
    f.write("5. 序列-结构组合(SEQ-DOT)的标签分布\n")
    f.write("="*60 + "\n")
    f.write(f"唯一序列-结构组合数量: {len(seq_dot_label_stats):,}\n")
    f.write(f"组合标签一致性均值: {np.mean(seq_dot_label_consistency):.3f}\n")
    f.write(f"每个组合的平均样本数: {seq_dot_sample_counts.mean():.2f}\n")
    f.write("\n组合的主要标签分布:\n")
    for label, count in seq_dot_main_label_dist.items():
        percentage = count / len(seq_dot_main_df) * 100
        f.write(f"  {label}: {count:,} 个组合 ({percentage:.2f}%)\n")

print(f"\n统计报告已保存: {report_file}")

print("\n" + "="*60)
print("分析完成！")
print("="*60)
print(f"所有结果已保存到: {output_dir}")
print("\n生成的文件:")
print("  1. 01_overall_label_distribution.png - 整体标签分布")
print("  2. 02_seq_based_label_distribution.png - 以序列为主的标签分布")
print("  3. 03_dot_based_label_distribution.png - 以结构为主的标签分布")
print("  4. 04_tissue_based_label_distribution.png - 按组织的标签分布")
print("  5. 05_seq_dot_combined_label_distribution.png - 序列-结构组合的标签分布")
print("  6. label_distribution_report.txt - 详细统计报告")

