# -*- coding: utf-8 -*-
# file: triclass_te_na_as_2_tno0_1_no_weight.py
# time: 09:35 07/10/2025
# author: YANG, HENG <hy345@exeter.ac.uk> (杨恒)
# homepage: https://yangheng95.github.io
# github: https://github.com/yangheng95
# huggingface: https://huggingface.co/yangheng
# google scholar: https://scholar.google.com/citations?user=NPq5a_0AAAAJ&hl=en
# Copyright (C) 2019-2025. All Rights Reserved.

"""
三分类TE（转座元件）表达预测模型
====================================
本脚本实现了基于OmniGenome-52M预训练模型的转座元件表达量三分类任务

任务说明：
- 输入：DNA序列
- 输出：9个组织中每个组织的TE表达量分类（0=低表达, 1=中表达, 2=高表达/NA）
- 9个组织：root, seedling, leaf, FMI, FOD, Prophase-I-pollen, Tricellular-pollen, flag, grain

模型架构：
- 基础模型：OmniGenome-52M（52M参数的基因组预训练模型）
- 任务类型：多标签多分类（每个标签有3个类别）
- 损失函数：CrossEntropyLoss（交叉熵损失，适用于多分类任务）
"""

import torch
import math
import os

from omnigenbench import (
    ClassificationMetric,
    Trainer,
    ModelHub,
    OmniTokenizer,
    OmniDatasetForMultiLabelClassification,
    OmniModelForMultiLabelSequenceClassification,
    OmniPooling,
)


# ============================================================================
# 第一部分：模型和分词器加载
# ============================================================================

# 使用本地模型路径
model_name_or_path = "/home/yingjie/OmniGenBench/models_cache/OmniGenome-52M"

# 检查本地模型是否存在
if not os.path.exists(model_name_or_path) or not os.listdir(model_name_or_path):
    print("⚠️  本地模型不存在或为空，使用在线下载...")
    model_name_or_path = "yangheng/OmniGenome-52M"
else:
    print(f"✅ 使用本地模型: {model_name_or_path}")
    # 检查关键文件是否存在
    required_files = ['config.json', 'tokenizer_config.json', 'vocab.txt']
    missing_files = [f for f in required_files if not os.path.exists(os.path.join(model_name_or_path, f))]
    if missing_files:
        print(f"⚠️  缺少关键文件: {missing_files}")
        print("🔄 回退到在线下载...")
        model_name_or_path = "yangheng/OmniGenome-52M"

# 加载分词器
tokenizer = OmniTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)


# ============================================================================
# 第二部分：数据集类定义
# ============================================================================

class TriClassTEDataset(OmniDatasetForMultiLabelClassification):
    """
    三分类TE数据集类
    
    功能说明：
    - 继承自 OmniDatasetForMultiLabelClassification，获得多标签分类数据集的基础功能
    - 重写 prepare_input() 方法以适配 TE 3分类任务的特殊需求
    - 处理9个组织的TE表达量标签
    
    标签映射规则：
    - 0 -> 0 (低表达)
    - 1 -> 1 (中表达)
    - 2 -> 2 (高表达/NA)
    - 空值/NaN/缺失值 -> -100 (在损失计算中被忽略)
    """

    def __init__(self, **kwargs):
        """初始化数据集，调用父类构造函数"""
        super().__init__(**kwargs)

    def prepare_input(self, instance, **kwargs):
        """
        准备单个样本的输入
        
        参数：
            instance: 包含序列和标签信息的字典
            
        返回：
            tokenized_inputs: 包含input_ids, attention_mask和labels的字典
        """
        
        def safe_label_mapping(label_value):
            """
            安全地映射标签值，处理各种输入格式
            
            参数：
                label_value: 原始标签值（可能是数字、字符串、None等）
                
            返回：
                映射后的标签索引 (0/1/2) 或 -100（表示忽略）
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
                return 1  # 中表达
            elif label_str in ['2.0', '2']:
                return 2  # 高表达/NA
            elif label_str.lower() in ['nan', 'na', 'null']:
                return -100  # 明确的缺失值标记
            else:
                # 其他未知情况，视为缺失值
                return -100

        # 从数据实例中提取9个组织的标签
        # 适配 merged_allTE.csv 的列名格式
        root_TE_label = safe_label_mapping(instance.get("root", instance.get("root_TE_label")))
        seedling_TE_label = safe_label_mapping(instance.get("seedling", instance.get("seedling_TE_label")))
        leaf_TE_label = safe_label_mapping(instance.get("leaf", instance.get("leaf_TE_label")))
        FMI_TE_label = safe_label_mapping(instance.get("FMI", instance.get("FMI_TE_label")))
        FOD_TE_label = safe_label_mapping(instance.get("FOD", instance.get("FOD_TE_label")))
        Prophase_I_pollen_TE_label = safe_label_mapping(instance.get("Prophase-I-pollen", instance.get("Prophase-I-pollen_TE_label")))
        Tricellular_pollen_TE_label = safe_label_mapping(instance.get("Tricellular-pollen", instance.get("Tricellular-pollen_TE_label")))
        flag_TE_label = safe_label_mapping(instance.get("flag", instance.get("flag_TE_label")))
        grain_TE_label = safe_label_mapping(instance.get("grain", instance.get("grain_TE_label")))
        
        # 提取DNA序列（适配 merged_allTE.csv 的列名：Seq）
        sequence = instance.get("Seq", instance.get("sequence", ""))

        # 对序列进行分词（tokenization）
        # - max_length: 限制序列最大长度为512个token
        # - padding: 填充到max_length长度
        # - truncation: 超过max_length的部分截断
        # - return_tensors: 返回PyTorch张量格式
        tokenized_inputs = self.tokenizer(
            sequence,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        # 将所有9个标签堆叠成一个张量
        # 形状：[9] - 每个元素对应一个组织的标签（0/1/2/-100）
        labels = torch.tensor([
            root_TE_label,
            seedling_TE_label,
            leaf_TE_label,
            FMI_TE_label,
            FOD_TE_label,
            Prophase_I_pollen_TE_label,
            Tricellular_pollen_TE_label,
            flag_TE_label,
            grain_TE_label,
        ], dtype=torch.long)  # 使用long类型以兼容CrossEntropyLoss

        # 将标签添加到tokenized_inputs中
        tokenized_inputs["labels"] = labels

        return tokenized_inputs


# ============================================================================
# 第三部分：模型类定义
# ============================================================================

class OmniModelForTriClassTESequenceClassification(OmniModelForMultiLabelSequenceClassification):
    """
    三分类TE序列分类模型
    
    模型结构：
    - 基础编码器：继承自OmniModelForMultiLabelSequenceClassification
    - 池化层：OmniPooling（提取序列的固定长度表示）
    - 分类头：线性层，输出维度为 num_labels × num_classes = 9 × 3 = 27
    
    损失函数：
    - CrossEntropyLoss：标准的多分类交叉熵损失
    - ignore_index=-100：忽略值为-100的标签（缺失值）
    - reduction="mean"：对所有有效样本取平均损失
    
    输出格式：
    - logits: [batch_size, num_labels, num_classes] = [batch, 9, 3]
    - predictions: [batch_size, num_labels] = [batch, 9]，每个组织的预测类别
    - probabilities: [batch_size, num_labels, num_classes] = [batch, 9, 3]，每个类别的概率
    """

    def __init__(self, config_or_model, tokenizer, num_labels=9, num_classes=3, *args, **kwargs):
        """
        初始化模型
        
        参数：
            config_or_model: 预训练模型配置或模型路径
            tokenizer: 分词器
            num_labels: 标签数量（9个组织）
            num_classes: 每个标签的类别数（3类：0/1/2）
        """
        # 调用父类构造函数
        # num_labels参数设置为 9×3=27，用于初始化输出层维度
        super().__init__(config_or_model, tokenizer, num_labels=num_labels * num_classes, *args, **kwargs)
        
        # 更新模型元数据
        self.metadata["model_name"] = self.__class__.__name__
        
        # 保存标签和类别数量
        self.num_labels = num_labels  # 9个组织
        self.num_classes = num_classes  # 3个类别（0/1/2）
        
        # 初始化池化层：将变长序列转换为固定长度向量
        self.pooler = OmniPooling(self.config)
        
        # 初始化分类头：将hidden_size维度映射到 9×3=27 维度
        self.classifier = torch.nn.Linear(self.config.hidden_size, self.num_classes * self.num_labels)
        
        # 定义损失函数
        # - ignore_index=-100: 忽略标签值为-100的样本（缺失值不参与损失计算）
        # - reduction="mean": 对batch内所有有效样本的损失取平均
        self.loss_fn = torch.nn.CrossEntropyLoss(ignore_index=-100, reduction="mean")

        # 保存数据集类引用，用于模型保存时的完整性
        self.dataset_class = kwargs.pop('dataset_class', TriClassTEDataset)

    def forward(self, input_ids, attention_mask=None, labels=None, **kwargs):
        """
        模型前向传播
        
        参数：
            input_ids: 输入的token ID序列 [batch_size, seq_length]
            attention_mask: 注意力掩码 [batch_size, seq_length]
            labels: 真实标签 [batch_size, num_labels] = [batch, 9]
            
        返回：
            字典，包含loss, logits, last_hidden_state
        """
        # 通过基础编码器获取序列表示
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **kwargs
        )

        # 提取logits（如果有的话，否则使用outputs[0]）
        logits = outputs.logits if hasattr(outputs, 'logits') else outputs[0]
        
        # 通过池化层和分类头得到最终预测
        logits = self.classifier(self.pooler(input_ids, logits))
        
        # 重塑logits：从 [batch, 27] -> [batch, 9, 3]
        # 方便后续按每个组织独立计算损失和预测
        batch_size = logits.shape[0]
        logits = logits.view(batch_size, self.num_labels, self.num_classes)

        # 计算损失（如果提供了标签）
        loss = None
        if labels is not None:
            # 展平logits和labels以适配CrossEntropyLoss的输入格式
            # logits: [batch, 9, 3] -> [batch×9, 3] = [total_samples, num_classes]
            # labels: [batch, 9] -> [batch×9] = [total_samples]
            logits_flat = logits.view(-1, self.num_classes)  # [batch×9, 3]
            labels_flat = labels.view(-1)  # [batch×9]

            # 计算交叉熵损失
            # CrossEntropyLoss会：
            # 1. 自动应用softmax到logits
            # 2. 计算负对数似然损失
            # 3. 忽略label=-100的样本
            # 4. 对所有有效样本取平均
            loss = self.loss_fn(logits_flat, labels_flat)

        return {
            "loss": loss,
            "logits": logits,  # [batch, 9, 3]
            "last_hidden_state": outputs.last_hidden_state if hasattr(outputs, 'last_hidden_state') else None,
        }

    def predict(self, sequence_or_inputs, **kwargs):
        """
        预测方法（用于训练过程中的验证）
        
        参数：
            sequence_or_inputs: 输入序列或tokenized inputs
            
        返回：
            字典，包含predictions, logits, probabilities, last_hidden_state
        """
        # 调用内部方法获取原始输出
        raw_outputs = self._forward_from_raw_input(sequence_or_inputs, **kwargs)

        logits = raw_outputs["logits"]  # [batch, 9, 3]
        last_hidden_state = raw_outputs["last_hidden_state"]

        # 应用softmax获取概率分布
        # dim=-1表示在最后一维（类别维度）上做softmax
        probabilities = torch.softmax(logits, dim=-1)  # [batch, 9, 3]

        # 获取预测类别（概率最大的类别）
        predictions = torch.argmax(probabilities, dim=-1)  # [batch, 9]

        outputs = {
            "predictions": predictions,  # 预测的类别索引 [batch, 9]
            "logits": logits,  # 原始logits [batch, 9, 3]
            "probabilities": probabilities,  # 类别概率 [batch, 9, 3]
            "last_hidden_state": last_hidden_state,  # 隐藏状态
        }

        return outputs

    def inference(self, sequence_or_inputs, **kwargs):
        """
        推理方法（用于模型部署和实际应用）
        
        与predict方法类似，但额外返回置信度信息
        
        参数：
            sequence_or_inputs: 输入序列或tokenized inputs
            
        返回：
            字典，包含predictions, logits, probabilities, confidence, last_hidden_state
        """
        # 获取原始输出
        raw_outputs = self._forward_from_raw_input(sequence_or_inputs, **kwargs)

        logits = raw_outputs["logits"]
        last_hidden_state = raw_outputs["last_hidden_state"]

        # 应用softmax获取概率
        probabilities = torch.softmax(logits, dim=-1)  # [batch, 9, 3]

        # 获取预测类别
        predictions = torch.argmax(probabilities, dim=-1)  # [batch, 9]

        # 计算置信度（每个标签的最大概率）
        confidence, _ = torch.max(probabilities, dim=-1)  # [batch, 9]

        # 根据输入是单个样本还是批量样本，调整输出格式
        if not isinstance(sequence_or_inputs, list):
            # 单个样本：去除batch维度
            outputs = {
                "predictions": predictions[0],  # [9]
                "logits": logits[0],  # [9, 3]
                "probabilities": probabilities[0],  # [9, 3]
                "confidence": confidence[0],  # [9]
                "last_hidden_state": last_hidden_state[0] if last_hidden_state is not None else None,
            }
        else:
            # 批量样本：保留batch维度
            outputs = {
                "predictions": predictions,  # [batch, 9]
                "logits": logits,  # [batch, 9, 3]
                "probabilities": probabilities,  # [batch, 9, 3]
                "confidence": confidence,  # [batch, 9]
                "last_hidden_state": last_hidden_state,
            }

        return outputs


# ============================================================================
# 第四部分：数据加载
# ============================================================================

print("📊 正在加载数据集...")

# 数据文件路径配置
# 用户需要自己将 merged_allTE.csv 划分为 train.csv, valid.csv, test.csv
data_dir = "/home/yingjie/OmniGenBench/examples/longtail/tn0"
train_file = os.path.join(data_dir, "train.csv")
valid_file = os.path.join(data_dir, "valid.csv")
test_file = os.path.join(data_dir, "test.csv")

# 检查文件是否存在
if not os.path.exists(train_file):
    print(f"⚠️  训练文件不存在: {train_file}")
    print("💡 请先将 merged_allTE.csv 划分为 train.csv, valid.csv, test.csv")
    raise FileNotFoundError(f"训练文件不存在: {train_file}")

# 从CSV文件加载数据集
datasets = {}
for split, file_path in [("train", train_file), ("valid", valid_file), ("test", test_file)]:
    if os.path.exists(file_path):
        print(f"📂 正在加载 {split} 数据集: {file_path}")
        datasets[split] = TriClassTEDataset(
            data_source=file_path,
            tokenizer=tokenizer,
            max_length=512,  # 序列最大长度
            force_padding=False  # 不强制填充（在collate时动态填充）
        )
    else:
        print(f"⚠️  {split} 文件不存在: {file_path}，跳过...")

print("📝 数据加载完成！")
print(f"📊 已加载的数据集: {list(datasets.keys())}")
for split, dataset in datasets.items():
    print(f"  - {split}: {len(dataset)} 个样本")


# ============================================================================
# 第五部分：模型初始化
# ============================================================================

print("\n🚀 正在初始化模型...")
model = OmniModelForTriClassTESequenceClassification(
    model_name_or_path,
    tokenizer,
    num_labels=9,  # 9个组织
    num_classes=3,  # 3个类别: 0, 1, 2
    trust_remote_code=True
)


# ============================================================================
# 第六部分：评估指标定义
# ============================================================================

"""
定义训练过程中使用的评估指标

1. accuracy_score（准确率）：
   - 计算公式：正确预测数 / 总样本数
   - 忽略标签为-100的样本（缺失值）

2. f1_score（F1分数）：
   - 计算公式：2 × (精确率 × 召回率) / (精确率 + 召回率)
   - 使用macro平均：对每个类别分别计算F1，然后取平均
   - 适合处理类别不平衡的情况

3. classification_report（分类报告）：
   - 提供每个类别的详细指标（精确率、召回率、F1分数、支持数）
   - 有助于分析模型在各个类别上的表现

这些指标会在训练过程中应用于验证集，在训练结束后应用于测试集
"""
metric_functions = [
    ClassificationMetric(ignore_y=-100).accuracy_score,
    ClassificationMetric(ignore_y=-100, average='macro').f1_score,
    ClassificationMetric(ignore_y=-100).classification_report,
]


# ============================================================================
# 第七部分：训练器配置和训练
# ============================================================================

"""
训练器（Trainer）配置说明

关键参数：
- epochs: 训练轮数（30轮）
- learning_rate: 学习率（2e-5，适合BERT类模型的微调）
- batch_size: 批次大小（8个样本/批次）
- gradient_accumulation_steps: 梯度累积步数（4步）
  * 实际有效batch size = 8 × 4 = 32
  * 作用：在GPU内存有限时，通过累积梯度模拟大batch训练
  * 工作原理：
    1. 前向传播和反向传播计算梯度（但不更新参数）
    2. 将梯度累加4次
    3. 第4次后使用累积的梯度更新参数，然后清零梯度

数据集说明：
- train_dataset: 训练集，用于更新模型参数
- eval_dataset: 验证集，用于训练过程中的性能监控和早停
- test_dataset: 测试集，仅在训练完成后用于最终评估

可选参数（已注释）：
- max_grad_norm: 梯度裁剪，防止梯度爆炸
- weight_decay: 权重衰减，L2正则化
- warmup_steps: 学习率预热步数
- eval_steps: 验证频率
- save_strategy: 保存策略
- load_best_model_at_end: 训练结束时加载最佳模型
"""
trainer = Trainer(
    model=model,
    epochs=30,
    learning_rate=2e-5,
    batch_size=8,
    train_dataset=datasets["train"],
    eval_dataset=datasets.get("valid"),  # 如果valid不存在则为None
    test_dataset=datasets.get("test"),  # 如果test不存在则为None
    compute_metrics=metric_functions,
    gradient_accumulation_steps=4,
    # 以下为可选的高级配置（根据需要启用）：
    # max_grad_norm=1.0,  # 梯度裁剪
    # weight_decay=0.01,  # 权重衰减
    # warmup_steps=100,  # 学习率预热
    # eval_steps=50,  # 更频繁的验证
    # save_strategy="steps",
    # save_steps=50,
    # load_best_model_at_end=True,
    # metric_for_best_model="accuracy_score",
    # greater_is_better=True,
    # save_total_limit=3,
)

# 开始训练并保存模型
print("\n🚀 开始训练...")
metrics = trainer.train(
    path_to_save="ogb_te_3class_finetuned_na_as_2_tno0_52m",
    dataset_class=TriClassTEDataset
)
print('\n📊 最终指标:', metrics)
print('\n🎉 训练完成！')


# ============================================================================
# 第八部分：推理示例（已注释）
# ============================================================================

"""
以下代码展示了如何使用训练好的模型进行推理

推理流程：
1. 从ModelHub加载训练好的模型
2. 准备测试样本
3. 调用model.inference()方法获取预测结果
4. 解析输出：
   - predictions: 每个组织的预测类别（0/1/2）
   - probabilities: 每个类别的概率分布
   - confidence: 预测的置信度（最大概率值）

输出解释：
- predictions: tensor([0, 2, 1, ...]) - 9个组织的预测类别
- probabilities: shape [9, 3] - 每个组织在3个类别上的概率
- confidence: tensor([0.999, 1.000, ...]) - 每个组织预测的置信度
"""

# print("\n🔮 开始在测试样本上进行推理...")
#
# # 加载训练好的模型
# inference_model = ModelHub.load("path/to/saved/model")
#
# # 获取测试样本
# sample_sequences = datasets['test'].examples[:10]
#
# # 定义类别和组织名称
# label_names = ['0', '1', '2']
# tissue_names = [
#     'root', 'seedling', 'leaf', 'FMI', 'FOD',
#     'Prophase-I-pollen', 'Tricellular-pollen', 'flag', 'grain'
# ]
#
# # 对每个样本进行推理
# with torch.no_grad():
#     for row in sample_sequences:
#         # 适配 merged_allTE.csv 的列名格式
#         sequence = row.get("Seq", row.get("sequence", ""))
#         print(f"\n{'='*60}")
#         print(f"🧬 样本ID: {row.get('ID', 'N/A')}")
#         print(f"📏 序列长度: {len(sequence)} bp")
#
#         # 执行推理
#         outputs = inference_model.inference(sequence, **row)
#         predictions = outputs['predictions'].cpu().numpy()  # [9]
#         probabilities = outputs['probabilities'].cpu().numpy()  # [9, 3]
#         confidence = outputs['confidence'].cpu().numpy()  # [9]
#
#         # 显示预测结果
#         print(f"\n📊 9个组织的预测结果:")
#         for i, tissue in enumerate(tissue_names):
#             pred_class = predictions[i]
#             pred_label = label_names[pred_class]
#             conf = confidence[i]
#             probs = probabilities[i]
#
#             # 获取真实标签（如果有）- 适配 merged_allTE.csv 的列名格式
#             # 优先使用组织名称作为列名（merged_allTE.csv格式），否则使用旧格式
#             gt_col = tissue if tissue in row else f"{tissue}_TE_label"
#             if gt_col in row:
#                 gt_value = row[gt_col]
#                 if gt_value is None or gt_value == '' or str(gt_value).strip() == '':
#                     gt_label = 'NA'
#                 else:
#                     try:
#                         gt_label = str(int(float(gt_value)))
#                     except (ValueError, TypeError):
#                         gt_label = 'NA'
#                 match_emoji = "✅" if pred_label == gt_label else "❌"
#                 print(f"  {match_emoji} {tissue:25s}: 预测={pred_label} (置信度: {conf:.3f}) [真实值: {gt_label}]")
#             else:
#                 print(f"  🔹 {tissue:25s}: 预测={pred_label} (置信度: {conf:.3f})")
#
#             # 显示概率分布
#             print(f"      概率分布 - 类别0: {probs[0]:.3f}, 类别1: {probs[1]:.3f}, 类别2: {probs[2]:.3f}")
#
# print("\n🎉 所有任务完成！")

