# -*- coding: utf-8 -*-
# file: structure_te_regression.py
# time: 2025-01-XX
# author: Generated for secondary structure TE regression
# Copyright (C) 2019-2025. All Rights Reserved.

"""
基于二级结构的TE回归模型
====================================
本脚本基于 OmniGenome-52M 预训练模型，实现了使用二级结构（DOT列）作为输入预测翻译效率（TE）的回归任务

任务说明：
- 输入：RNA 二级结构（dot-bracket notation，DOT 列）
- 输出：翻译效率（TE）的连续数值
- 数据来源：9tissue_dot_filtered.csv

模型架构：
- 基础模型：OmniGenome-52M（52M 参数的基因组预训练模型）
- 任务类型：序列级回归
- 损失函数：MSELoss（均方误差）
"""

import os
import warnings
import torch
import numpy as np
from typing import Dict, Any
os.environ["NCCL_P2P_DISABLE"] = "1"
os.environ["NCCL_IB_DISABLE"] = "1"

from omnigenbench import (
    RegressionMetric,
    AccelerateTrainer,
    ModelHub,
    OmniTokenizer,
    OmniDatasetForSequenceRegression,
    OmniModelForSequenceRegression,
)
from omnigenbench.src.misc.utils import fprint
from tqdm import tqdm


# ============================================================================
# 第一部分：模型和分词器加载
# ============================================================================

model_name_or_path = "/home/yingjie/OmniGenBench/models_cache/OmniGenome-52M"

if not os.path.exists(model_name_or_path) or not os.listdir(model_name_or_path):
    print("⚠️  本地模型不存在或为空，使用在线下载...")
    model_name_or_path = "yangheng/OmniGenome-52M"
else:
    print(f"✅ 使用本地模型: {model_name_or_path}")
    required_files = ["config.json", "tokenizer_config.json", "vocab.txt"]
    missing_files = [
        f for f in required_files if not os.path.exists(os.path.join(model_name_or_path, f))
    ]
    if missing_files:
        print(f"⚠️  缺少关键文件: {missing_files}")
        print("🔄 回退到在线下载...")
        model_name_or_path = "yangheng/OmniGenome-52M"

tokenizer = OmniTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)


# ============================================================================
# 第二部分：数据集类定义
# ============================================================================

class StructureTERegressionDataset(OmniDatasetForSequenceRegression):
    """
    基于二级结构的TE回归数据集类

    数据格式要求：
    - 输入列：使用 'DOT' 列（二级结构 dot-bracket notation）
    - 标签列：使用 'TE' 列（翻译效率的连续数值）

    标签处理规则：
    - 将TE值转换为float类型
    - 空值、NaN、None -> -100（训练时忽略）
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def prepare_input(self, instance, **kwargs):
        def safe_te_mapping(te_value):
            """
            安全地映射TE值，处理各种输入格式
            
            参数：
                te_value: 原始TE值（可能是数字、字符串、None等）
                
            返回：
                映射后的TE值（float）或 -100（表示忽略）
            """
            # 处理空值情况
            if te_value is None or te_value == '' or str(te_value).strip() == '':
                return -100  # 空值用-100表示，在MSELoss中会被忽略
            
            # 转换为字符串并去除空格
            te_str = str(te_value).strip()
            
            # 检查是否为缺失值标记
            if te_str.lower() in ['nan', 'na', 'null', 'none']:
                return -100  # 明确的缺失值标记
            
            try:
                # 尝试转换为float
                te_float = float(te_str)
                # 检查是否为NaN
                if np.isnan(te_float):
                    return -100
                return te_float
            except (ValueError, TypeError):
                # 转换失败，视为缺失值
                return -100

        # 获取TE标签
        te_label = safe_te_mapping(instance.get("TE"))

        # 使用 DOT 列作为输入（二级结构）
        structure = instance.get("DOT", "")
        
        if not structure:
            warnings.warn(f"Empty structure for instance {instance.get('ID', 'unknown')}")
            structure = "."  # 使用默认值避免错误

        # 对二级结构进行分词（tokenization）
        tokenized_inputs = self.tokenizer(
            structure,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        # 将标签添加到 tokenized_inputs 中
        # 回归任务使用float类型的标量张量
        # DataLoader 的 collate_fn 会将标量标签堆叠成 [batch] 形状
        tokenized_inputs["labels"] = torch.tensor(te_label, dtype=torch.float32)
        
        return tokenized_inputs


# ============================================================================
# 第三部分：模型类定义（使用基础的序列回归模型）
# ============================================================================

class OmniModelForStructureTERegression(OmniModelForSequenceRegression):
    """
    OmniGenome 二级结构TE回归模型

    输出：
    - logits: [batch_size, 1] (回归值)
    - predictions: [batch_size] (预测的TE值)
    """

    def __init__(self, config_or_model, tokenizer, num_labels=1, *args, **kwargs):
        # 在调用父类之前处理 dataset_class，避免 kwargs 被父类处理
        self.dataset_class = kwargs.pop("dataset_class", StructureTERegressionDataset)
        # 设置 num_labels 为 1（回归任务输出单个连续值）
        super().__init__(config_or_model, tokenizer, num_labels=num_labels, *args, **kwargs)
        
        self.metadata["model_name"] = self.__class__.__name__
        # 父类已经设置了MSELoss，但我们需要确保支持ignore_index
        # 注意：MSELoss不直接支持ignore_index，我们需要在forward中手动处理

    def forward(self, input_ids, attention_mask=None, labels=None, **kwargs):
        """
        前向传播，计算回归预测和损失
        
        注意：MSELoss不直接支持ignore_index，我们需要手动过滤掉-100的标签
        """
        labels_input = labels
        inputs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            **kwargs
        }
        
        # 调用父类的 forward 方法获取输出
        outputs = super().forward(**inputs, labels=labels_input)
        
        logits = outputs["logits"]  # [batch_size, 1]
        last_hidden_state = outputs["last_hidden_state"]
        
        # 计算损失（需要手动处理ignore_index=-100）
        if labels_input is not None:
            # 将logits从[batch_size, 1]压缩为[batch_size]
            logits_flat = logits.squeeze(-1)  # [batch_size]
            labels_flat = labels_input.view(-1)  # [batch_size]
            
            # 过滤掉-100的标签（忽略的样本）
            valid_mask = labels_flat != -100
            if valid_mask.any():
                valid_logits = logits_flat[valid_mask]
                valid_labels = labels_flat[valid_mask]
                loss = self.loss_fn(valid_logits, valid_labels)
            else:
                # 如果所有标签都是-100，损失为0
                loss = torch.tensor(0.0, device=logits.device, requires_grad=True)
            
            outputs["loss"] = loss
        
        return outputs

    def predict(self, sequence_or_inputs, **kwargs):
        """Prediction for TE regression"""
        raw_outputs = self._forward_from_raw_input(sequence_or_inputs, **kwargs)

        logits = raw_outputs["logits"]  # [batch_size, 1]
        last_hidden_state = raw_outputs["last_hidden_state"]

        # 将logits从[batch_size, 1]压缩为[batch_size]
        predictions = logits.squeeze(-1)  # [batch_size]

        outputs = {
            "predictions": predictions,
            "logits": logits,
            "last_hidden_state": last_hidden_state,
        }

        return outputs

    def inference(self, sequence_or_inputs, **kwargs):
        raw_outputs = self._forward_from_raw_input(sequence_or_inputs, **kwargs)

        logits = raw_outputs["logits"]  # [batch_size, 1]
        last_hidden_state = raw_outputs["last_hidden_state"]

        # 将logits从[batch_size, 1]压缩为[batch_size]
        predictions = logits.squeeze(-1)  # [batch_size]

        if not isinstance(sequence_or_inputs, list):
            return {
                "predictions": predictions[0].item(),
                "logits": logits[0],
                "last_hidden_state": last_hidden_state[0] if last_hidden_state is not None else None,
            }

        return {
            "predictions": predictions,
            "logits": logits,
            "last_hidden_state": last_hidden_state,
        }


# ============================================================================
# 第四部分：数据加载
# ============================================================================

print("📊 正在加载数据集...")

# 从目录中手动加载 train.csv, valid.csv, test.csv
data_dir = "/home/yingjie/OmniGenBench/examples/second_structure_for_classification/data/8_1_1_filter"

print(f"📁 数据目录: {data_dir}")

# 从指定目录加载已分割的数据集
datasets = StructureTERegressionDataset.from_hub(
    data_dir,  # 指定数据目录（包含 train.csv, valid.csv, test.csv）
    tokenizer=tokenizer,
    max_length=512,
    force_padding=False
)

print("📝 数据加载完成！")
print(f"📊 已加载的数据集: {list(datasets.keys())}")
for split, dataset in datasets.items():
    print(f"  - {split}: {len(dataset)} 个样本")
    # 显示TE值的统计信息
    if len(dataset) > 0:
        te_values = []
        for example in dataset.examples[:100]:  # 采样前100个样本
            te = example.get("TE")
            if te is not None and str(te).strip() not in ['', 'nan', 'na', 'null']:
                try:
                    te_values.append(float(te))
                except:
                    pass
        if te_values:
            print(f"    TE值范围: {min(te_values):.4f} ~ {max(te_values):.4f}, 均值: {np.mean(te_values):.4f}")


# ============================================================================
# 第五部分：模型初始化
# ============================================================================

print("\n🚀 正在初始化模型...")
model = OmniModelForStructureTERegression(
    model_name_or_path,
    tokenizer,
    num_labels=1,  # 回归任务输出单个连续值
    trust_remote_code=True,
)


# ============================================================================
# 第六部分：评估指标定义
# ============================================================================

metric_functions = [
    RegressionMetric(ignore_y=-100).mean_squared_error,
    RegressionMetric(ignore_y=-100).root_mean_squared_error,
    RegressionMetric(ignore_y=-100).mean_absolute_error,
    RegressionMetric(ignore_y=-100).r2_score,
]


# ============================================================================
# 第七部分：训练器配置和训练
# ============================================================================

# 🔑 关键修复：使用 drop_last=True 解决形状不一致问题
# 训练、验证、测试都使用 drop_last=True，丢弃最后一个不完整的 batch
# 这样就不需要自定义 collate_fn 了，代码更简洁
from torch.utils.data import DataLoader

batch_size = 24
train_loader = DataLoader(
    datasets["train"],
    batch_size=batch_size,
    shuffle=True,
    drop_last=True  # 丢弃最后一个不完整的 batch
)
eval_loader = DataLoader(
    datasets["valid"],
    batch_size=batch_size,
    shuffle=False,
    drop_last=True  # 丢弃最后一个不完整的 batch
)
test_loader = DataLoader(
    datasets["test"],
    batch_size=batch_size,
    shuffle=False,
    drop_last=True  # 丢弃最后一个不完整的 batch
)

trainer = AccelerateTrainer(
    model=model,
    epochs=20,
    learning_rate=2e-5,
    train_loader=train_loader,
    eval_loader=eval_loader,
    test_loader=test_loader,
    compute_metrics=metric_functions,
    gradient_accumulation_steps=2,
)

print("\n🚀 开始训练...")
# 创建保存目录
save_dir = "structure_te_regression_model_filtered"
os.makedirs(save_dir, exist_ok=True)
model_save_path = os.path.join(save_dir, "ogb_structure_te_52m")
print(f"📁 模型将保存到: {model_save_path}")

metrics = trainer.train(
    path_to_save=model_save_path,
    dataset_class=StructureTERegressionDataset,
)
print("\n📊 最终指标:", metrics)
print("\n🎉 训练完成！")


# ============================================================================
# 第八部分：推理示例（已注释）
# ============================================================================

"""
推理流程示例（使用训练好的模型）：

1. 使用 ModelHub.load 加载模型
2. 准备测试样本（二级结构字符串）
3. 调用 model.inference() 获取预测

输出内容：
- predictions: 预测的TE值（连续数值）
- logits: 模型原始输出
"""

# print("\n🔮 开始在测试样本上进行推理...")
# inference_model = ModelHub.load("path/to/saved/model")
# sample_structures = datasets['test'].examples[:10]
# 
# with torch.no_grad():
#     for row in sample_structures:
#         structure = row.get("DOT", "")
#         print(f"\n{'='*60}")
#         print(f"🧬 样本ID: {row.get('ID', 'N/A')}")
#         print(f"📏 二级结构长度: {len(structure)} bp")
#         print(f"📊 二级结构: {structure[:100]}..." if len(structure) > 100 else f"📊 二级结构: {structure}")
# 
#         outputs = inference_model.inference(structure)
#         pred_te = outputs['predictions']
# 
#         gt_te = row.get("TE", "N/A")
#         try:
#             gt_te_float = float(gt_te)
#             error = abs(pred_te - gt_te_float)
#             print(f"  📈 预测TE={pred_te:.4f} [真实TE: {gt_te_float:.4f}, 误差: {error:.4f}]")
#         except:
#             print(f"  📈 预测TE={pred_te:.4f} [真实TE: {gt_te}]")
# 
# print("\n🎉 所有任务完成！")

