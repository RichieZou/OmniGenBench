#!/usr/bin/env python3
"""
分析tri27102025实验结果的完整性并绘制图表
"""

import os
import re
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict

# 配置
base_dir = "/home/yingjie/OmniGenBench/examples/9TE/tri27102025"
expected_weights = [0.01, 0.1, 0.3, 0.5, 0.7, 1.0]
expected_epochs = 15

print("=" * 80)
print("实验完整性分析")
print("=" * 80)

# 获取所有ogb开头的文件夹
all_items = os.listdir(base_dir)
folders = [f for f in all_items if f.startswith('ogb_te_3class_finetuned') and 
           os.path.isdir(os.path.join(base_dir, f))]

print(f"\n总共找到 {len(folders)} 个实验文件夹")

# 解析文件夹名称
pattern_epoch = r'epoch_(\d+)_seed_42_accuracy_score_([\d.]+)_seed_42_f1_score_([\d.]+)'
pattern_final = r'final_seed_42_accuracy_score_([\d.]+)_seed_42_f1_score_([\d.]+)'

data = []
for folder in folders:
    if 'final' in folder:
        match = re.search(pattern_final, folder)
        if match:
            accuracy = float(match.group(1))
            f1 = float(match.group(2))
            data.append({
                'folder': folder,
                'epoch': 'final',
                'epoch_num': 16,  # 用于排序
                'accuracy': accuracy,
                'f1': f1
            })
    else:
        match = re.search(pattern_epoch, folder)
        if match:
            epoch = int(match.group(1))
            accuracy = float(match.group(2))
            f1 = float(match.group(3))
            data.append({
                'folder': folder,
                'epoch': epoch,
                'epoch_num': epoch,
                'accuracy': accuracy,
                'f1': f1
            })

df = pd.DataFrame(data)

# 统计每个epoch的实验数量
print("\n" + "=" * 80)
print("各Epoch实验数量统计")
print("=" * 80)

epoch_counts = df[df['epoch'] != 'final'].groupby('epoch').size()
for epoch in range(1, expected_epochs + 1):
    count = epoch_counts.get(epoch, 0)
    status = "✓" if count == len(expected_weights) else f"✗ (实际: {count}, 预期: {len(expected_weights)})"
    print(f"Epoch {epoch:2d}: {count} 个实验 {status}")

final_count = len(df[df['epoch'] == 'final'])
status = "✓" if final_count == len(expected_weights) else f"✗ (实际: {final_count}, 预期: {len(expected_weights)})"
print(f"Final:     {final_count} 个实验 {status}")

# 识别不同的权重组
# 对于每个epoch，找出所有不同的accuracy/f1组合
print("\n" + "=" * 80)
print("识别不同的权重组（基于Epoch 1数据）")
print("=" * 80)

epoch1_data = df[df['epoch'] == 1].sort_values('accuracy')
print(f"\nEpoch 1 找到 {len(epoch1_data)} 个不同的实验:")
for idx, (_, row) in enumerate(epoch1_data.iterrows(), 1):
    print(f"  组{idx}: accuracy={row['accuracy']:.4f}, f1={row['f1']:.4f}")

# 使用epoch 1的数据作为基准，为每个实验分配组别
# 通过在每个epoch中匹配相似的accuracy和f1模式
def assign_group(row, reference_data):
    """根据参考数据为当前行分配组别"""
    if len(reference_data) == 0:
        return 0
    
    # 计算与每个参考点的距离
    distances = []
    for ref_idx, (_, ref_row) in enumerate(reference_data.iterrows()):
        # 使用加权欧氏距离
        dist = np.sqrt((row['accuracy'] - ref_row['accuracy'])**2 + 
                      (row['f1'] - ref_row['f1'])**2)
        distances.append((dist, ref_idx))
    
    # 返回最近的组
    return min(distances)[1]

# 为所有非final数据分配组别
df_non_final = df[df['epoch'] != 'final'].copy()
if len(epoch1_data) > 0:
    df_non_final['group'] = df_non_final.apply(
        lambda row: assign_group(row, epoch1_data), axis=1
    )
    
    # 为每个组分配一个颜色和标签
    group_labels = {}
    if len(epoch1_data) <= len(expected_weights):
        for idx, (_, row) in enumerate(epoch1_data.iterrows()):
            group_labels[idx] = f"weight_2={expected_weights[idx]}"
    else:
        for idx in range(len(epoch1_data)):
            group_labels[idx] = f"实验组{idx+1}"
    
    # 检查每个组的完整性
    print("\n" + "=" * 80)
    print("各权重组实验完整性")
    print("=" * 80)
    
    for group_id in sorted(df_non_final['group'].unique()):
        group_data = df_non_final[df_non_final['group'] == group_id]
        epochs_present = sorted(group_data['epoch'].unique())
        missing_epochs = set(range(1, expected_epochs + 1)) - set(epochs_present)
        
        status = "✓ 完整" if len(missing_epochs) == 0 else f"✗ 缺少epoch: {sorted(missing_epochs)}"
        print(f"\n{group_labels.get(group_id, f'组{group_id}')}:")
        print(f"  完成的epochs: {len(epochs_present)}/{expected_epochs} {status}")
    
    # 绘图
    print("\n" + "=" * 80)
    print("生成可视化图表")
    print("=" * 80)
    
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    
    # 使用不同的颜色和标记
    colors = plt.cm.tab10(np.linspace(0, 1, 10))
    markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h']
    
    for group_id in sorted(df_non_final['group'].unique()):
        group_data = df_non_final[df_non_final['group'] == group_id].sort_values('epoch')
        
        label = group_labels.get(group_id, f'组{group_id}')
        color = colors[group_id % len(colors)]
        marker = markers[group_id % len(markers)]
        
        # Accuracy曲线
        axes[0].plot(group_data['epoch'], group_data['accuracy'], 
                    marker=marker, label=label, color=color, 
                    linewidth=2, markersize=8, alpha=0.8)
        
        # F1 Score曲线
        axes[1].plot(group_data['epoch'], group_data['f1'], 
                    marker=marker, label=label, color=color, 
                    linewidth=2, markersize=8, alpha=0.8)
    
    # 设置Accuracy图表
    axes[0].set_xlabel('Epoch', fontsize=13, fontweight='bold')
    axes[0].set_ylabel('Accuracy', fontsize=13, fontweight='bold')
    axes[0].set_title('Accuracy vs Epoch for Different Weight_2 Values', 
                     fontsize=15, fontweight='bold', pad=20)
    axes[0].legend(fontsize=11, loc='best', framealpha=0.9)
    axes[0].grid(True, alpha=0.3, linestyle='--')
    axes[0].set_xlim(0.5, expected_epochs + 0.5)
    
    # 设置F1 Score图表
    axes[1].set_xlabel('Epoch', fontsize=13, fontweight='bold')
    axes[1].set_ylabel('F1 Score', fontsize=13, fontweight='bold')
    axes[1].set_title('F1 Score vs Epoch for Different Weight_2 Values', 
                     fontsize=15, fontweight='bold', pad=20)
    axes[1].legend(fontsize=11, loc='best', framealpha=0.9)
    axes[1].grid(True, alpha=0.3, linestyle='--')
    axes[1].set_xlim(0.5, expected_epochs + 0.5)
    
    plt.tight_layout()
    
    # 保存图表
    output_path = os.path.join(base_dir, 'weight_experiment_analysis.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n✓ 图表已保存至: {output_path}")
    
    # 保存数据到CSV
    csv_path = os.path.join(base_dir, 'experiment_results_summary.csv')
    df_non_final['weight_label'] = df_non_final['group'].map(group_labels)
    df_non_final[['weight_label', 'epoch', 'accuracy', 'f1']].to_csv(
        csv_path, index=False
    )
    print(f"✓ 数据摘要已保存至: {csv_path}")
    
    # 不显示图表，只保存
    # plt.show()

else:
    print("\n⚠ 未找到有效的实验数据")

# 总结
print("\n" + "=" * 80)
print("总结")
print("=" * 80)
print(f"预期权重数量: {len(expected_weights)} ({', '.join(map(str, expected_weights))})")
print(f"预期每个权重的epochs: {expected_epochs}")
print(f"预期每个权重的final: 1")
print(f"预期总文件夹数: {len(expected_weights)} × ({expected_epochs} + 1) = {len(expected_weights) * (expected_epochs + 1)}")
print(f"实际总文件夹数: {len(df)}")

if len(df) == len(expected_weights) * (expected_epochs + 1):
    print("\n✓ 所有实验已完成！")
else:
    completed_groups = len(epoch1_data)
    remaining_groups = len(expected_weights) - completed_groups
    print(f"\n✗ 实验未完成")
    print(f"  已完成: {completed_groups} 个权重")
    print(f"  待完成: {remaining_groups} 个权重")
    if remaining_groups > 0:
        completed_weights = expected_weights[:completed_groups]
        remaining_weights = expected_weights[completed_groups:]
        print(f"  已完成的权重可能是: {completed_weights}")
        print(f"  待运行的权重可能是: {remaining_weights}")

print("=" * 80)

