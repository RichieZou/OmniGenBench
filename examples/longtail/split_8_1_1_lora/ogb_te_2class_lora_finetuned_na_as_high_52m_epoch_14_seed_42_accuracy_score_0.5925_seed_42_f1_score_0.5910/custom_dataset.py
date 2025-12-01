# -*- coding: utf-8 -*-
# file: biclass_te_na_as_high_lora.py
# time: 10:15 11/11/2025
# author: YANG, HENG <hy345@exeter.ac.uk> (杨恒)
# homepage: https://yangheng95.github.io
# github: https://github.com/yangheng95
# huggingface: https://huggingface.co/yangheng
# google scholar: https://scholar.google.com/citations?user=NPq5a_0AAAAJ&hl=en
# Copyright (C) 2019-2025. All Rights Reserved.

"""
转座元件表达二分类模型（LoRA 微调版本）
====================================
本脚本基于 OmniGenome-52M 预训练模型，使用 LoRA（Low-Rank Adaptation）进行参数高效微调，
实现了 9 个组织的转座元件表达量二分类任务

任务说明：
- 输入：DNA 序列
- 输出：9 个组织对应的表达量类别（0=低表达, 1=高表达/NA）
- 组织列表：root, seedling, leaf, FMI, FOD, Prophase-I-pollen, Tricellular-pollen, flag, grain

模型架构：
- 基础模型：OmniGenome-52M（52M 参数的基因组预训练模型）
- 微调方式：LoRA（Low-Rank Adaptation）- 只训练少量参数，大幅降低显存需求
- 任务类型：多标签二分类（每个标签有 2 个类别）
- 损失函数：CrossEntropyLoss

LoRA 优势：
- 显存占用大幅降低（通常可降低 50-80%）
- 训练速度更快
- 可训练参数量显著减少（通常只训练 1-5% 的参数）
- 保持与全量微调相近的性能
"""

import os
import math
import torch
os.environ["NCCL_P2P_DISABLE"] = "1"
os.environ["NCCL_IB_DISABLE"] = "1"

# 设置使用 GPU ID=1
if torch.cuda.is_available():
    torch.cuda.set_device(1)
    print(f"✅ 已设置使用 GPU: {torch.cuda.current_device()} (GPU ID=1)")
else:
    print("⚠️  CUDA 不可用，将使用 CPU")

from omnigenbench import (
    ClassificationMetric,
    AccelerateTrainer,
    Trainer,
    ModelHub,
    OmniTokenizer,
    OmniDatasetForMultiLabelClassification,
    OmniModelForMultiLabelSequenceClassification,
    OmniPooling,
    OmniLoraModel,  # 导入 LoRA 模型包装类
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

class BiClassTEDataset(OmniDatasetForMultiLabelClassification):
    """
    转座元件表达二分类数据集类

    数据格式要求：
    - 序列列：使用 'sequence' 列
    - 标签列：使用组织名称作为列名（root, seedling, leaf, FMI, FOD, Prophase-I-pollen, Tricellular-pollen, flag, grain）

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

        # 统一使用组织名称作为列名（适配 CSV 文件格式）
        labels = [
            safe_label_mapping(instance.get("root")),
            safe_label_mapping(instance.get("seedling")),
            safe_label_mapping(instance.get("leaf")),
            safe_label_mapping(instance.get("FMI")),
            safe_label_mapping(instance.get("FOD")),
            safe_label_mapping(instance.get("Prophase-I-pollen")),
            safe_label_mapping(instance.get("Tricellular-pollen")),
            safe_label_mapping(instance.get("flag")),
            safe_label_mapping(instance.get("grain")),
        ]

        # 使用 sequence 列（CSV 文件中已删除重复的 Seq 列）
        sequence = instance.get("sequence", "")

        tokenized_inputs = self.tokenizer(
            sequence,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        tokenized_inputs["labels"] = torch.tensor(labels, dtype=torch.long)
        return tokenized_inputs


# ============================================================================
# 第三部分：模型类定义
# ============================================================================

class OmniModelForBiClassTESequenceClassification(OmniModelForMultiLabelSequenceClassification):
    """
    OmniGenome TE 二分类模型

    输出：
    - logits: [batch_size, 9, 2]
    - predictions: [batch_size, 9]
    - probabilities: [batch_size, 9, 2]
    """

    def __init__(self, config_or_model, tokenizer, num_labels=9, num_classes=2, *args, **kwargs):
        super().__init__(config_or_model, tokenizer, num_labels=num_labels * num_classes, *args, **kwargs)

        self.metadata["model_name"] = self.__class__.__name__

        self.num_labels = num_labels
        self.num_classes = num_classes

        self.pooler = OmniPooling(self.config)
        self.classifier = torch.nn.Linear(self.config.hidden_size, self.num_labels * self.num_classes)
        self.loss_fn = torch.nn.CrossEntropyLoss(ignore_index=-100, reduction="mean")
        self.dataset_class = kwargs.pop("dataset_class", BiClassTEDataset)

    def forward(self, input_ids, attention_mask=None, labels=None, **kwargs):
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **kwargs,
        )

        logits = outputs.logits if hasattr(outputs, "logits") else outputs[0]
        logits = self.classifier(self.pooler(input_ids, logits))

        batch_size = logits.shape[0]
        logits = logits.view(batch_size, self.num_labels, self.num_classes)

        loss = None
        if labels is not None:
            logits_flat = logits.view(-1, self.num_classes)
            labels_flat = labels.view(-1)
            loss = self.loss_fn(logits_flat, labels_flat)

        return {
            "loss": loss,
            "logits": logits,
            "last_hidden_state": getattr(outputs, "last_hidden_state", None),
        }

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
                "predictions": predictions[0],
                "logits": logits[0],
                "probabilities": probabilities[0],
                "confidence": confidence[0],
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

# 使用 from_hub 方法从目录中自动加载 train.csv, valid.csv, test.csv
data_dir = "/home/yingjie/OmniGenBench/examples/longtail/tn0/biclassdata/8_1_1"
datasets = BiClassTEDataset.from_hub(
    data_dir,  # 指定数据目录
    tokenizer=tokenizer,
    max_length=512,
    force_padding=False
)

print("📝 数据加载完成！")
print(f"📊 已加载的数据集: {list(datasets.keys())}")
for split, dataset in datasets.items():
    print(f"  - {split}: {len(dataset)} 个样本")


# ============================================================================
# 第五部分：模型初始化（LoRA 版本）
# ============================================================================

print("\n🚀 正在初始化模型（LoRA 微调模式）...")

# 第一步：创建基础模型
base_model = OmniModelForBiClassTESequenceClassification(
    model_name_or_path,
    tokenizer,
    num_labels=9,
    num_classes=2,
    trust_remote_code=True,
)

# 第二步：配置 LoRA 参数
# 根据 OmniGenome-52M 的推荐配置
lora_config = {
    "r": 8,                          # LoRA 的秩（rank），控制适配器的容量
    "lora_alpha": 32,                # LoRA 的缩放因子，通常设置为 r 的 2-4 倍
    "lora_dropout": 0.1,             # LoRA 层的 dropout 率
    "target_modules": ["key", "value", "dense"],  # 目标模块：注意力层的 key、value 和 dense 层
    "bias": "none",                  # 不训练偏置参数
    "use_rslora": True,              # 使用 RSLoRA（Rank-Stabilized LoRA）
}

print("📋 LoRA 配置:")
for key, value in lora_config.items():
    print(f"  - {key}: {value}")

# 第三步：使用 OmniLoraModel 包装基础模型
model = OmniLoraModel(
    base_model,
    lora_config=lora_config
)

print("✅ LoRA 模型初始化完成！")
print("💡 提示：LoRA 只训练少量参数，显存占用和训练时间都会大幅降低")


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

# LoRA 微调时，学习率可以设置得稍高一些（因为只训练少量参数）
# 通常 LoRA 的学习率是全量微调的 2-10 倍
trainer = AccelerateTrainer(
    model=model,
    epochs=30,
    learning_rate=5e-5,  # LoRA 可以使用更高的学习率（原版是 2e-5）
    batch_size=16,
    train_dataset=datasets["train"],
    eval_dataset=datasets["valid"],
    test_dataset=datasets["test"],
    compute_metrics=metric_functions,
    gradient_accumulation_steps=4,
    max_grad_norm=1.0,
    autocast="float16",
    # 可按需启用以下高级配置：
    # max_grad_norm=1.0,
    # weight_decay=0.01,
    # warmup_steps=100,
    # eval_steps=50,
    # save_strategy="steps",
    # save_steps=50,
    # load_best_model_at_end=True,
    # metric_for_best_model="accuracy_score",
    # greater_is_better=True,
    # save_total_limit=3,
)

print("\n🚀 开始训练（LoRA 微调模式）...")
# 创建 split_8_1_1_lora 目录，模型文件保存在该目录下
split_dir = "split_8_1_1_lora"
os.makedirs(split_dir, exist_ok=True)
model_save_path = os.path.join(split_dir, "ogb_te_2class_lora_finetuned_na_as_high_52m")
print(f"📁 模型将保存到: {model_save_path}")

metrics = trainer.train(
    path_to_save=model_save_path,
    dataset_class=BiClassTEDataset,
)
print("\n📊 最终指标:", metrics)
print("\n🎉 训练完成！")


# ============================================================================
# 第八部分：推理示例（已注释）
# ============================================================================

"""
推理流程示例（使用训练好的 LoRA 模型）：

1. 使用 ModelHub.load 加载模型（LoRA 适配器会自动加载）
2. 准备测试样本
3. 调用 model.inference() 获取预测

注意：LoRA 模型的保存和加载方式与全量微调模型相同，框架会自动处理 LoRA 适配器。

输出内容：
- predictions: 9 个组织的预测类别（0/1）
- probabilities: 每个类别的概率分布
- confidence: 预测置信度
"""

# print("\n🔮 开始在测试样本上进行推理...")
# inference_model = ModelHub.load("path/to/saved/lora/model")
# sample_sequences = datasets['test'].examples[:10]
# label_names = ['0', '1']
# tissue_names = [
#     'root', 'seedling', 'leaf', 'FMI', 'FOD',
#     'Prophase-I-pollen', 'Tricellular-pollen', 'flag', 'grain'
# ]
#
# with torch.no_grad():
#     for row in sample_sequences:
#         sequence = row.get("sequence", "")
#         print(f"\n{'='*60}")
#         print(f"🧬 样本ID: {row.get('ID', 'N/A')}")
#         print(f"📏 序列长度: {len(sequence)} bp")
#
#         outputs = inference_model.inference(sequence, **row)
#         predictions = outputs['predictions'].cpu().numpy()
#        probabilities = outputs['probabilities'].cpu().numpy()
#         confidence = outputs['confidence'].cpu().numpy()
#
#         print(f"\n📊 9个组织的预测结果:")
#         for i, tissue in enumerate(tissue_names):
#             pred_class = predictions[i]
#             pred_label = label_names[pred_class]
#             conf = confidence[i]
#             probs = probabilities[i]
#
#             gt_col = tissue if tissue in row else f"{tissue}_TE_label"
#             if gt_col in row:
#                 gt_value = row[gt_col]
#                 if gt_value is None or str(gt_value).strip() == "":
#                     gt_label = "NA"
#                 else:
#                     try:
#                         gt_label = str(int(float(gt_value)))
#                     except (ValueError, TypeError):
#                         gt_label = "NA"
#                 match_emoji = "✅" if pred_label == gt_label else "❌"
#                 print(f"  {match_emoji} {tissue:25s}: 预测={pred_label} (置信度: {conf:.3f}) [真实值: {gt_label}]")
#             else:
#                 print(f"  🔹 {tissue:25s}: 预测={pred_label} (置信度: {conf:.3f})")
#
#             print(f"      概率分布 - 类别0: {probs[0]:.3f}, 类别1: {probs[1]:.3f}")
#
# print("\n🎉 所有任务完成！")

