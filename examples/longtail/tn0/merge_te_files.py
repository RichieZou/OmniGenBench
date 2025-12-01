#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
合并9个TE CSV文件，相同序列合并标签
"""

import pandas as pd
import os
from collections import defaultdict

# 定义文件路径和对应的标签名称
input_dir = "/home/yingjie/OmniGenBench/examples/longtail/tn0/original_allTE_extracted"
output_file = "/home/yingjie/OmniGenBench/examples/longtail/tn0/merged_allTE.csv"

# 9个文件及其对应的标签名称
file_mapping = {
    "Tricellular-pollen_TE.csv": "Tricellular-pollen",
    "seedling_TE.csv": "seedling",
    "root_TE.csv": "root",
    "Prophase-I-pollen_TE.csv": "Prophase-I-pollen",
    "leaf_TE.csv": "leaf",
    "grain_TE.csv": "grain",
    "FOD_TE.csv": "FOD",
    "FMI_TE.csv": "FMI",
    "flag_TE.csv": "flag"
}

def merge_te_files():
    """
    合并9个TE文件，相同序列合并标签
    """
    # 使用字典存储每个序列的信息
    # key: Seq, value: dict包含ID列表和9个标签值
    seq_dict = defaultdict(lambda: {
        'IDs': [],
        'labels': {}
    })
    
    # 读取每个文件
    for filename, label_name in file_mapping.items():
        filepath = os.path.join(input_dir, filename)
        print(f"正在处理: {filename}")
        
        # 分块读取大文件
        chunk_size = 10000
        for chunk in pd.read_csv(filepath, chunksize=chunk_size):
            for _, row in chunk.iterrows():
                seq = row['Seq']
                seq_id = row['ID']
                label = row['label']
                
                # 如果label为空或NaN，设置为空字符串
                if pd.isna(label):
                    label = ''
                else:
                    label = str(label)
                
                # 存储序列信息
                if seq_id not in seq_dict[seq]['IDs']:
                    seq_dict[seq]['IDs'].append(seq_id)
                
                # 存储该文件对应的标签
                seq_dict[seq]['labels'][label_name] = label
    
    print(f"\n总共找到 {len(seq_dict)} 个唯一序列")
    
    # 构建输出数据
    output_data = []
    for seq, info in seq_dict.items():
        # 使用第一个ID作为主ID（或者可以合并所有ID）
        main_id = info['IDs'][0] if info['IDs'] else ''
        
        # 构建行数据
        row_data = {
            'ID': main_id,
            'Seq': seq
        }
        
        # 添加9个标签列
        for label_name in file_mapping.values():
            row_data[label_name] = info['labels'].get(label_name, '')
        
        output_data.append(row_data)
    
    # 创建DataFrame并保存
    df_output = pd.DataFrame(output_data)
    
    # 确保列的顺序：ID, Seq, 然后是9个标签列
    columns_order = ['ID', 'Seq'] + list(file_mapping.values())
    df_output = df_output[columns_order]
    
    # 保存到CSV
    df_output.to_csv(output_file, index=False)
    print(f"\n合并完成！输出文件: {output_file}")
    print(f"总行数: {len(df_output)}")
    print(f"列名: {', '.join(df_output.columns)}")
    
    # 显示一些统计信息
    print("\n各标签列的统计信息:")
    for label_name in file_mapping.values():
        non_empty = (df_output[label_name] != '').sum()
        print(f"  {label_name}: {non_empty} 个非空标签")

if __name__ == "__main__":
    merge_te_files()

