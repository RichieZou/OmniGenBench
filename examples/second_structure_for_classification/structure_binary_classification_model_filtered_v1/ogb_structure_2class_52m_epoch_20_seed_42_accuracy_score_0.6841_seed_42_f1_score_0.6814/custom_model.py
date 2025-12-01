# -*- coding: utf-8 -*-
# file: structure_binary_classification.py
# time: 2025-01-XX
# author: Generated for secondary structure classification
# Copyright (C) 2019-2025. All Rights Reserved.

"""
基于二级结构的二分类模型
====================================
本脚本基于 OmniGenome-52M 预训练模型，实现了使用二级结构（DOT列）作为输入的二分类任务

任务说明：
- 输入：RNA 二级结构（dot-bracket notation，DOT 列）
- 输出：二分类标签（0=低表达, 1=高表达）
- 数据来源：9tissue_dot_filtered.csv

模型架构：
- 基础模型：OmniGenome-52M（52M 参数的基因组预训练模型）
- 任务类型：单标签二分类
- 损失函数：CrossEntropyLoss
"""

import os
import warnings
import torch
os.environ["NCCL_P2P_DISABLE"] = "1"
os.environ["NCCL_IB_DISABLE"] = "1"

from omnigenbench import (
    ClassificationMetric,
    AccelerateTrainer,
    ModelHub,
    OmniTokenizer,
    OmniDatasetForSequenceClassification,
    OmniModelForSequenceClassification,
)


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

class StructureBinaryClassificationDataset(OmniDatasetForSequenceClassification):
    """
    基于二级结构的二分类数据集类

    数据格式要求：
    - 输入列：使用 'DOT' 列（二级结构 dot-bracket notation）
    - 标签列：使用 'LABLE' 列（0=低表达, 1=高表达）

    标签映射规则：
    - 0 / 0.0 -> 0 (低表达)
    - 1 / 1.0 -> 1 (高表达)
    - 其他情况（空值、NaN、None） -> -100（训练时忽略）
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def prepare_input(self, instance, **kwargs):
        def safe_label_mapping(label_value):
            """
            安全地映射标签值，处理各种输入格式
            
            参数：
                label_value: 原始标签值（可能是数字、字符串、None等）
                
            返回：
                映射后的标签索引 (0/1) 或 -100（表示忽略）
            """
            # 处理空值情况
            if label_value is None or label_value == '' or str(label_value).strip() == '':
                return -100  # 空值用-100表示，在CrossEntropyLoss中会被忽略
            
            # 转换为字符串并去除空格
            label_str = str(label_value).strip()
            
            # 映射到对应的类别索引
            if label_str in ['0.0', '0']:
                return 0  # 低表达
            elif label_str in ['1.0', '1']:
                return 1  # 高表达
            elif label_str.lower() in ['nan', 'na', 'null']:
                return -100  # 明确的缺失值标记
            else:
                # 其他未知情况，视为缺失值
                return -100

        # 获取标签（LABLE 列）
        label = safe_label_mapping(instance.get("LABLE"))

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
        # 注意：单标签分类的标签应该是1D张量 [1]，这样 DataLoader 的 collate_fn 才能正确堆叠
        # 如果创建为标量（0维），不同批次大小会导致连接时形状不匹配
        tokenized_inputs["labels"] = torch.tensor([label], dtype=torch.long)
        
        return tokenized_inputs


# ============================================================================
# 第三部分：模型类定义（使用基础的单标签分类模型）
# ============================================================================

class OmniModelForStructureBinaryClassification(OmniModelForSequenceClassification):
    """
    OmniGenome 二级结构二分类模型

    输出：
    - logits: [batch_size, 2]
    - predictions: [batch_size]
    - probabilities: [batch_size, 2]
    """

    def __init__(self, config_or_model, tokenizer, num_labels=2, *args, **kwargs):
        # 在调用父类之前处理 dataset_class，避免 kwargs 被父类处理
        self.dataset_class = kwargs.pop("dataset_class", StructureBinaryClassificationDataset)
        # 设置 num_labels 为 2（二分类）
        super().__init__(config_or_model, tokenizer, num_labels=num_labels, *args, **kwargs)
        
        self.metadata["model_name"] = self.__class__.__name__
        # 更新损失函数以支持 ignore_index
        self.loss_fn = torch.nn.CrossEntropyLoss(ignore_index=-100, reduction="mean")

    def forward(self, input_ids, attention_mask=None, labels=None, **kwargs):
        # 注意：父类的 forward 方法会对 logits 应用 softmax，但 CrossEntropyLoss 需要原始 logits
        # 因此我们需要直接调用父类的内部方法获取原始 logits，或者重新计算
        labels_input = labels
        inputs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            **kwargs
        }
        
        # 调用父类的 last_hidden_state_forward 获取隐藏状态
        last_hidden_state = self.last_hidden_state_forward(**inputs)
        last_hidden_state = self.dropout(last_hidden_state)
        last_hidden_state = self.activation(last_hidden_state)
        last_hidden_state = self.pooler(inputs, last_hidden_state)
        
        # 获取原始 logits（未应用 softmax）
        raw_logits = self.classifier(last_hidden_state)
        
        # 应用 softmax 用于输出（保持与父类行为一致）
        logits = self.softmax(raw_logits)
        
        outputs = {
            "logits": logits,
            "last_hidden_state": last_hidden_state,
            "labels": labels_input,
        }
        
        # 计算损失（使用原始 logits，因为 CrossEntropyLoss 内部会应用 softmax）
        if labels_input is not None:
            # 确保 labels 的形状正确：[batch_size]
            # CrossEntropyLoss 要求 labels 是 1D 张量，形状为 [batch_size]
            if isinstance(labels_input, torch.Tensor):
                # 展平 labels 到 1D，确保形状为 [batch_size]
                labels_flat = labels_input.view(-1)
                # 确保 batch_size 匹配
                batch_size = raw_logits.size(0)
                if labels_flat.size(0) != batch_size:
                    # 如果 batch_size 不匹配，取前 batch_size 个元素或重复
                    if labels_flat.size(0) == 1:
                        # 如果只有一个标签，扩展到整个 batch（这种情况不应该发生，但处理一下）
                        labels_flat = labels_flat.expand(batch_size)
                    else:
                        # 取前 batch_size 个元素
                        labels_flat = labels_flat[:batch_size]
                outputs["loss"] = self.loss_fn(raw_logits, labels_flat)
            else:
                # 如果不是张量，转换为张量
                labels_tensor = torch.tensor(labels_input, dtype=torch.long, device=raw_logits.device)
                outputs["loss"] = self.loss_fn(raw_logits, labels_tensor.view(-1))
        
        return outputs

    def predict(self, sequence_or_inputs, **kwargs):
        raw_outputs = self._forward_from_raw_input(sequence_or_inputs, **kwargs)

        logits = raw_outputs["logits"]
        last_hidden_state = raw_outputs["last_hidden_state"]

        probabilities = torch.softmax(logits, dim=-1)
        predictions = torch.argmax(probabilities, dim=-1)

        return {
            "predictions": predictions,
            "logits": logits,
            "probabilities": probabilities,
            "last_hidden_state": last_hidden_state,
        }

    def inference(self, sequence_or_inputs, **kwargs):
        raw_outputs = self._forward_from_raw_input(sequence_or_inputs, **kwargs)

        logits = raw_outputs["logits"]
        last_hidden_state = raw_outputs["last_hidden_state"]

        probabilities = torch.softmax(logits, dim=-1)
        predictions = torch.argmax(probabilities, dim=-1)
        confidence, _ = torch.max(probabilities, dim=-1)

        if not isinstance(sequence_or_inputs, list):
            return {
                "predictions": predictions[0].item(),
                "logits": logits[0],
                "probabilities": probabilities[0],
                "confidence": confidence[0].item(),
                "last_hidden_state": last_hidden_state[0] if last_hidden_state is not None else None,
            }

        return {
            "predictions": predictions,
            "logits": logits,
            "probabilities": probabilities,
            "confidence": confidence,
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
datasets = StructureBinaryClassificationDataset.from_hub(
    data_dir,  # 指定数据目录（包含 train.csv, valid.csv, test.csv）
    tokenizer=tokenizer,
    max_length=512,
    force_padding=False
)

print("📝 数据加载完成！")
print(f"📊 已加载的数据集: {list(datasets.keys())}")
for split, dataset in datasets.items():
    print(f"  - {split}: {len(dataset)} 个样本")


# ============================================================================
# 第五部分：模型初始化
# ============================================================================

print("\n🚀 正在初始化模型...")
model = OmniModelForStructureBinaryClassification(
    model_name_or_path,
    tokenizer,
    num_labels=2,  # 二分类
    trust_remote_code=True,
)


# ============================================================================
# 第六部分：评估指标定义
# ============================================================================

metric_functions = [
    ClassificationMetric(ignore_y=-100).accuracy_score,
    ClassificationMetric(ignore_y=-100, average="macro").f1_score,
    ClassificationMetric(ignore_y=-100).classification_report,
]


# ============================================================================
# 第七部分：训练器配置和训练
# ============================================================================

trainer = AccelerateTrainer(
    model=model,
    epochs=20,
    learning_rate=2e-5,
    batch_size=12,
    train_dataset=datasets["train"],
    eval_dataset=datasets["valid"],
    test_dataset=datasets["test"],
    compute_metrics=metric_functions,
    gradient_accumulation_steps=2,
    max_grad_norm=1.0,
)

print("\n🚀 开始训练...")
# 创建保存目录
save_dir = "structure_binary_classification_model_filtered"
os.makedirs(save_dir, exist_ok=True)
model_save_path = os.path.join(save_dir, "ogb_structure_2class_52m")
print(f"📁 模型将保存到: {model_save_path}")

metrics = trainer.train(
    path_to_save=model_save_path,
    dataset_class=StructureBinaryClassificationDataset,
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
- predictions: 预测类别（0或1）
- probabilities: 每个类别的概率分布
- confidence: 预测置信度
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
#         pred_class = outputs['predictions']
#         conf = outputs['confidence']
#         probs = outputs['probabilities']
# 
#         gt_label = row.get("LABLE", "N/A")
#         match_emoji = "✅" if str(pred_class) == str(gt_label) else "❌"
#         print(f"  {match_emoji} 预测={pred_class} (置信度: {conf:.3f}) [真实值: {gt_label}]")
#         print(f"      概率分布 - 类别0: {probs[0]:.3f}, 类别1: {probs[1]:.3f}")
# 
# print("\n🎉 所有任务完成！")

