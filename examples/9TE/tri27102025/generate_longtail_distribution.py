#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
分析标签组合的长尾分布
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

# 转换为DataFrame并排序（从多到少）
combinations_df = pd.DataFrame([
    {'combination': comb, 'count': count} 
    for comb, count in combination_counts.items()
])
combinations_df = combinations_df.sort_values('count', ascending=False).reset_index(drop=True)

# 计算累积百分比
combinations_df['cumulative_count'] = combinations_df['count'].cumsum()
combinations_df['cumulative_percentage'] = (combinations_df['cumulative_count'] / len(df)) * 100
combinations_df['percentage'] = (combinations_df['count'] / len(df)) * 100

print(f"\n前10个最常见的组合:")
for i, row in combinations_df.head(10).iterrows():
    comb_str = ','.join(row['combination'])
    print(f"  {i+1}. [{comb_str}]: {row['count']:5d} 样本 ({row['percentage']:.2f}%)")

print(f"\n长尾统计:")
# 计算有多少组合只出现1次
single_occurrence = (combinations_df['count'] == 1).sum()
print(f"  只出现1次的组合: {single_occurrence} ({single_occurrence/len(combinations_df)*100:.1f}%)")

# 计算80%的数据由多少个组合构成
threshold_80 = combinations_df[combinations_df['cumulative_percentage'] <= 80].shape[0]
print(f"  前{threshold_80}个组合覆盖80%的数据")

threshold_90 = combinations_df[combinations_df['cumulative_percentage'] <= 90].shape[0]
print(f"  前{threshold_90}个组合覆盖90%的数据")

threshold_95 = combinations_df[combinations_df['cumulative_percentage'] <= 95].shape[0]
print(f"  前{threshold_95}个组合覆盖95%的数据")

# 创建图表
output_dir = Path('label_statistics')
output_dir.mkdir(exist_ok=True)

# 创建多个子图来展示长尾分布
fig = plt.figure(figsize=(20, 12))

# 子图1: 完整的长尾分布（线性坐标）
ax1 = plt.subplot(3, 2, 1)
x = range(len(combinations_df))
y = combinations_df['count'].values
ax1.plot(x, y, linewidth=1.5, color='steelblue', alpha=0.8)
ax1.fill_between(x, y, alpha=0.3, color='steelblue')
ax1.set_xlabel('Label Combination Index (sorted by frequency)', fontsize=11, fontweight='bold')
ax1.set_ylabel('Sample Count', fontsize=11, fontweight='bold')
ax1.set_title(f'Long-tail Distribution of Label Combinations\n({len(combinations_df)} unique combinations out of {4**9:,} possible)', 
              fontsize=12, fontweight='bold')
ax1.grid(True, alpha=0.3)
ax1.set_xlim(0, len(combinations_df))

# 子图2: 对数坐标查看长尾
ax2 = plt.subplot(3, 2, 2)
ax2.plot(x, y, linewidth=1.5, color='coral', alpha=0.8)
ax2.fill_between(x, y, alpha=0.3, color='coral')
ax2.set_xlabel('Label Combination Index (sorted by frequency)', fontsize=11, fontweight='bold')
ax2.set_ylabel('Sample Count (log scale)', fontsize=11, fontweight='bold')
ax2.set_title('Long-tail Distribution (Log Scale)', fontsize=12, fontweight='bold')
ax2.set_yscale('log')
ax2.grid(True, alpha=0.3, which='both')
ax2.set_xlim(0, len(combinations_df))

# 子图3: Top 100 组合
ax3 = plt.subplot(3, 2, 3)
top_n = min(100, len(combinations_df))
x_top = range(top_n)
y_top = combinations_df['count'].values[:top_n]
bars = ax3.bar(x_top, y_top, color='green', alpha=0.7, edgecolor='black', linewidth=0.5)
ax3.set_xlabel('Top 100 Combinations', fontsize=11, fontweight='bold')
ax3.set_ylabel('Sample Count', fontsize=11, fontweight='bold')
ax3.set_title(f'Top {top_n} Most Frequent Combinations', fontsize=12, fontweight='bold')
ax3.grid(axis='y', alpha=0.3)

# 子图4: 累积分布曲线
ax4 = plt.subplot(3, 2, 4)
ax4.plot(x, combinations_df['cumulative_percentage'].values, 
         linewidth=2, color='purple', alpha=0.8)
# 添加参考线
ax4.axhline(y=80, color='r', linestyle='--', alpha=0.5, label='80% threshold')
ax4.axhline(y=90, color='orange', linestyle='--', alpha=0.5, label='90% threshold')
ax4.axhline(y=95, color='yellow', linestyle='--', alpha=0.5, label='95% threshold')
# 添加垂直参考线
ax4.axvline(x=threshold_80, color='r', linestyle=':', alpha=0.3)
ax4.axvline(x=threshold_90, color='orange', linestyle=':', alpha=0.3)
ax4.axvline(x=threshold_95, color='yellow', linestyle=':', alpha=0.3)
ax4.set_xlabel('Number of Combinations', fontsize=11, fontweight='bold')
ax4.set_ylabel('Cumulative Percentage (%)', fontsize=11, fontweight='bold')
ax4.set_title('Cumulative Distribution', fontsize=12, fontweight='bold')
ax4.legend(loc='lower right', fontsize=9)
ax4.grid(True, alpha=0.3)
ax4.set_xlim(0, len(combinations_df))
ax4.set_ylim(0, 100)

# 子图5: 分桶统计
ax5 = plt.subplot(3, 2, 5)
max_count = max(combinations_df['count'])
if max_count >= 1000:
    bins = [1, 2, 5, 10, 20, 50, 100, 500, 1000, max_count+1]
    bin_labels = ['1', '2-4', '5-9', '10-19', '20-49', '50-99', '100-499', '500-999', '1000+']
elif max_count >= 500:
    bins = [1, 2, 5, 10, 20, 50, 100, 500, max_count+1]
    bin_labels = ['1', '2-4', '5-9', '10-19', '20-49', '50-99', '100-499', f'500-{max_count}']
elif max_count >= 100:
    bins = [1, 2, 5, 10, 20, 50, 100, max_count+1]
    bin_labels = ['1', '2-4', '5-9', '10-19', '20-49', '50-99', f'100-{max_count}']
else:
    bins = [1, 2, 5, 10, 20, 50, max_count+1]
    bin_labels = ['1', '2-4', '5-9', '10-19', '20-49', f'50-{max_count}']
combinations_df['count_bin'] = pd.cut(combinations_df['count'], bins=bins, labels=bin_labels, right=False)
bin_counts = combinations_df['count_bin'].value_counts().sort_index()
bars5 = ax5.bar(range(len(bin_counts)), bin_counts.values, color='teal', alpha=0.7, edgecolor='black')
ax5.set_xticks(range(len(bin_counts)))
ax5.set_xticklabels(bin_counts.index, rotation=45, ha='right')
ax5.set_xlabel('Sample Count Range', fontsize=11, fontweight='bold')
ax5.set_ylabel('Number of Combinations', fontsize=11, fontweight='bold')
ax5.set_title('Distribution of Combination Frequencies', fontsize=12, fontweight='bold')
# 添加数值标签
for bar in bars5:
    height = bar.get_height()
    ax5.text(bar.get_x() + bar.get_width()/2., height,
            f'{int(height)}',
            ha='center', va='bottom', fontsize=9)
ax5.grid(axis='y', alpha=0.3)

# 子图6: 基尼系数和帕累托图
ax6 = plt.subplot(3, 2, 6)
# 计算基尼系数 (需要从小到大排序)
sorted_counts = np.sort(combinations_df['count'].values)
n = len(sorted_counts)
index = np.arange(1, n + 1)
gini = (2 * np.sum(index * sorted_counts)) / (n * np.sum(sorted_counts)) - (n + 1) / n

# 绘制洛伦兹曲线
cumsum = np.cumsum(sorted_counts)
ax6.plot(np.arange(n) / n * 100, cumsum / cumsum[-1] * 100, 
         linewidth=2, color='darkred', label=f'Lorenz Curve (Gini={gini:.3f})')
ax6.plot([0, 100], [0, 100], 'k--', alpha=0.5, label='Perfect Equality')
ax6.fill_between(np.arange(n) / n * 100, cumsum / cumsum[-1] * 100, 
                  np.arange(n) / n * 100, alpha=0.2, color='darkred')
ax6.set_xlabel('Cumulative % of Combinations', fontsize=11, fontweight='bold')
ax6.set_ylabel('Cumulative % of Samples', fontsize=11, fontweight='bold')
ax6.set_title('Lorenz Curve & Gini Coefficient', fontsize=12, fontweight='bold')
ax6.legend(loc='lower right', fontsize=9)
ax6.grid(True, alpha=0.3)
ax6.set_xlim(0, 100)
ax6.set_ylim(0, 100)

plt.suptitle('Label Combination Long-tail Distribution Analysis', 
             fontsize=16, fontweight='bold', y=0.995)
plt.tight_layout()
plt.savefig(output_dir / '11_longtail_distribution.png', dpi=300, bbox_inches='tight')
print(f"\n图表已保存到: {output_dir / '11_longtail_distribution.png'}")
plt.close()

# 保存详细统计数据
print("\n正在保存详细统计数据...")
combinations_df['combination_str'] = combinations_df['combination'].apply(lambda x: ','.join(x))
output_csv = combinations_df[['combination_str', 'count', 'percentage', 'cumulative_percentage']]
output_csv.columns = ['Label_Combination', 'Sample_Count', 'Percentage', 'Cumulative_Percentage']
output_csv.to_csv(output_dir / 'combination_statistics.csv', index=False)
print(f"详细统计已保存到: {output_dir / 'combination_statistics.csv'}")

print("\n" + "="*60)
print("长尾分布分析完成！")
print("="*60)
print(f"理论组合数: {4**9:,}")
print(f"实际组合数: {len(combinations_df):,} ({len(combinations_df)/(4**9)*100:.2f}%)")
print(f"基尼系数: {gini:.3f} (越接近1表示分布越不均匀)")
print(f"最多出现: {combinations_df['count'].max()} 次")
print(f"最少出现: {combinations_df['count'].min()} 次")
print(f"平均出现: {combinations_df['count'].mean():.1f} 次")
print(f"中位数: {combinations_df['count'].median():.0f} 次")
print("="*60)

