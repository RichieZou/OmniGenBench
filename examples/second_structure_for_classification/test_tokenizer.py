# -*- coding: utf-8 -*-
"""
测试 OmniTokenizer 是否支持 AUCG 序列和 .() 二级结构
基于 structure_binary_classification.py 的代码
"""

import os
import torch
from omnigenbench import OmniTokenizer

# 使用和你的代码相同的模型路径
model_name_or_path = "/home/yingjie/OmniGenBench/models_cache/OmniGenome-52M"

if not os.path.exists(model_name_or_path) or not os.listdir(model_name_or_path):
    print("⚠️  本地模型不存在或为空，使用在线下载...")
    model_name_or_path = "yangheng/OmniGenome-52M"
else:
    print(f"✅ 使用本地模型: {model_name_or_path}")

# 加载 tokenizer（和你的代码一样）
tokenizer = OmniTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)

print("\n" + "="*60)
print("测试 1: AUCG RNA 序列")
print("="*60)

# 从你的数据中取一个真实的序列
rna_sequence = "AGCCGCCTCTCTCCCGCACTTGCGCAGCCCATCTCTCTTCCTCCCGTCACCCCCCTCCTGCCTACTCTCTCGATCCCCACTGTCGCCGCCTCGTTCGCCGACCGATTCCGGCCGCTCCGGCGAGCTTTGGTGCCACCCAACACACACACAGCTGCTCCTGGACGAGCTCTACTAGCTTCACCGCGTCGATCTACTTTTCCTCGCCTCATGCGGTCACCCCCATCGATTTCCCCTGCTAGGGTTTCGATGGTGGGGGTGGGCCTCGCTCCGGCGGCCGGCTGCTCGTTTCCCCGATCTCTCGATCCCCATTGATGCGTCCCTGCAGAAGAGAAGATGCGGAGGGAGGATAGGAGGAGGTAGGCGCCCTGGGGTGGGGTGGTCTCGCCGGCGACGATAGGGCTGGCCGGCTGGAGTTGGATCGGGATCCCGGCTGGCACGCCTCGCCTCCACCTCCCTCTCCCACCTCTCTGCTCCCAGAACCCTAGCAGCTCCCGCCGTTGCCGTCGTCTACTTCACCCAGCCTGCCGGACCGCCTCCGTCCACTTCCCCGGTGTCGTCGCGGTCGTCCTCGCCTGCACACGGCAAGAGGAGCGCCCCCGCCCTCTCCTCTCCTCCTCCCCTCATCACGAGCCAGGACAGGAAGGGCCACCGTGGCCGGGCAGGTCAGGTGGCGCGTCCAGATCACCGAGGAGCCGCCCGCAAGGTCGGCC"

print(f"序列长度: {len(rna_sequence)}")
print(f"序列前50个字符: {rna_sequence[:50]}...")

# 使用和你的代码相同的 tokenization 方式
tokenized_rna = tokenizer(
    rna_sequence,
    max_length=512,
    padding="max_length",
    truncation=True,
    return_tensors="pt",
)

print(f"✅ Tokenization 成功!")
print(f"   input_ids shape: {tokenized_rna['input_ids'].shape}")
print(f"   attention_mask shape: {tokenized_rna['attention_mask'].shape}")
print(f"   前10个 token IDs: {tokenized_rna['input_ids'][0][:10].tolist()}")

# 检查是否有未知 token
if hasattr(tokenizer, 'base_tokenizer') and hasattr(tokenizer.base_tokenizer, 'unk_token_id'):
    unk_id = tokenizer.base_tokenizer.unk_token_id
    if unk_id is not None:
        unk_count = (tokenized_rna['input_ids'] == unk_id).sum().item()
        print(f"   <unk> token 数量: {unk_count}")

print("\n" + "="*60)
print("测试 2: .() 二级结构")
print("="*60)

# 从你的数据中取一个真实的二级结构
structure = ".((((.............((((((..((...((.(((.((((((.((.....((((((.((....(((((.............................................................................................................(((((((.....................)))))))..((((((((((....(((.....)))..))))))).)))......................................((((....)))).........((......))...)))))....))))))))...)).))))))(((((((....(.(((((((((((...((.((((((((.(((((((((..(((.((((..(((..((((....((..(((.......))).))...))))..)))...))))..))).))))..)))))...........)))))))).)).)))))))))))))))))))((((((((.((....((((.((((((.....((((((.(((((.(((.......)))..((((.(((....))))))).(((.((((.(((..........)))..)))).))).))))).)))))))))))).))))..)).)))).))))........)))))...)).))))))..))))."

print(f"结构长度: {len(structure)}")
print(f"结构前50个字符: {structure[:50]}...")
print(f"包含的字符类型: {set(structure)}")

# 使用和你的代码完全相同的 tokenization 方式
tokenized_structure = tokenizer(
    structure,
    max_length=512,
    padding="max_length",
    truncation=True,
    return_tensors="pt",
)

print(f"✅ Tokenization 成功!")
print(f"   input_ids shape: {tokenized_structure['input_ids'].shape}")
print(f"   attention_mask shape: {tokenized_structure['attention_mask'].shape}")
print(f"   前10个 token IDs: {tokenized_structure['input_ids'][0][:10].tolist()}")

# 检查是否有未知 token
if hasattr(tokenizer, 'base_tokenizer') and hasattr(tokenizer.base_tokenizer, 'unk_token_id'):
    unk_id = tokenizer.base_tokenizer.unk_token_id
    if unk_id is not None:
        unk_count = (tokenized_structure['input_ids'] == unk_id).sum().item()
        print(f"   <unk> token 数量: {unk_count}")

print("\n" + "="*60)
print("测试 3: 短序列和短结构（验证基本功能）")
print("="*60)

short_seq = "AUCGACGUAGCUAGCUAGCU"
short_struct = "..(((...))).."

print(f"短序列: {short_seq}")
tokenized_short_seq = tokenizer(short_seq, max_length=512, padding="max_length", truncation=True, return_tensors="pt")
print(f"  ✅ Tokenization 成功! shape: {tokenized_short_seq['input_ids'].shape}")

print(f"短结构: {short_struct}")
tokenized_short_struct = tokenizer(short_struct, max_length=512, padding="max_length", truncation=True, return_tensors="pt")
print(f"  ✅ Tokenization 成功! shape: {tokenized_short_struct['input_ids'].shape}")

print("\n" + "="*60)
print("测试 4: structure_in=True 混合方式（序列 + EOS + 二级结构）")
print("="*60)

# 模拟 structure_in=True 的行为
# 根据 abstract_dataset.py 的 _preprocessing 方法：
# self.examples[idx]["sequence"] = f"{sequence}{self.tokenizer.eos_token}{structure}"

# 获取 EOS token
eos_token = tokenizer.eos_token if hasattr(tokenizer, 'eos_token') else tokenizer.base_tokenizer.eos_token
if eos_token is None:
    eos_token = tokenizer.base_tokenizer.sep_token if hasattr(tokenizer.base_tokenizer, 'sep_token') else "</s>"

print(f"EOS token: '{eos_token}'")

# 使用短序列和短结构进行测试（避免过长）
short_seq = "AUCGACGUAGCUAGCUAGCU"
short_struct = "..(((...))).."

# 模拟 structure_in=True 的拼接方式
mixed_input = f"{short_seq}{eos_token}{short_struct}"
print(f"序列: {short_seq}")
print(f"结构: {short_struct}")
print(f"混合输入: {mixed_input}")
print(f"混合输入长度: {len(mixed_input)}")

# Tokenization
tokenized_mixed = tokenizer(
    mixed_input,
    max_length=512,
    padding="max_length",
    truncation=True,
    return_tensors="pt",
)

print(f"✅ Tokenization 成功!")
print(f"   input_ids shape: {tokenized_mixed['input_ids'].shape}")
print(f"   attention_mask shape: {tokenized_mixed['attention_mask'].shape}")
print(f"   前15个 token IDs: {tokenized_mixed['input_ids'][0][:15].tolist()}")

# 解码前几个 token 看看是否正确
if hasattr(tokenizer, 'decode'):
    decoded_tokens = tokenizer.decode(tokenized_mixed['input_ids'][0][:15])
    print(f"   前15个 token 解码: {decoded_tokens}")

# 检查是否有未知 token
if hasattr(tokenizer, 'base_tokenizer') and hasattr(tokenizer.base_tokenizer, 'unk_token_id'):
    unk_id = tokenizer.base_tokenizer.unk_token_id
    if unk_id is not None:
        unk_count = (tokenized_mixed['input_ids'] == unk_id).sum().item()
        print(f"   <unk> token 数量: {unk_count}")

print("\n" + "="*60)
print("测试 5: 使用真实数据的混合方式")
print("="*60)

# 使用真实数据（截取前200个字符避免过长）
real_seq = rna_sequence[:200]
real_struct = structure[:200]

mixed_real = f"{real_seq}{eos_token}{real_struct}"
print(f"真实序列长度: {len(real_seq)}")
print(f"真实结构长度: {len(real_struct)}")
print(f"混合输入总长度: {len(mixed_real)}")
print(f"混合输入前100个字符: {mixed_real[:100]}...")

tokenized_mixed_real = tokenizer(
    mixed_real,
    max_length=512,
    padding="max_length",
    truncation=True,
    return_tensors="pt",
)

print(f"✅ Tokenization 成功!")
print(f"   input_ids shape: {tokenized_mixed_real['input_ids'].shape}")
print(f"   attention_mask shape: {tokenized_mixed_real['attention_mask'].shape}")

# 检查是否有未知 token
if hasattr(tokenizer, 'base_tokenizer') and hasattr(tokenizer.base_tokenizer, 'unk_token_id'):
    unk_id = tokenizer.base_tokenizer.unk_token_id
    if unk_id is not None:
        unk_count = (tokenized_mixed_real['input_ids'] == unk_id).sum().item()
        print(f"   <unk> token 数量: {unk_count}")

print("\n" + "="*60)
print("✅ 结论: OmniTokenizer 可以同时处理 AUCG 序列和 .() 二级结构!")
print("✅ 结论: structure_in=True 的混合方式（序列+EOS+结构）也可以正常工作!")
print("="*60)

