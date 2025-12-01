# -*- coding: utf-8 -*-
# file: triclass_te_focal_loss.py
# time: 2025
# author: YANG, HENG <hy345@exeter.ac.uk> (杨恒)
# Copyright (C) 2019-2025. All Rights Reserved.

"""
三分类TE（转座元件）表达预测模型 - 使用 Focal Loss
====================================
本脚本实现了基于OmniGenome-52M预训练模型的转座元件表达量三分类任务
使用 Focal Loss 处理类别不平衡和大量缺失标签的问题

任务说明：
- 输入：DNA序列
- 输出：9个组织中每个组织的TE表达量分类（0=低表达, 1=中表达, 2=高表达/NA）
- 9个组织：root, seedling, leaf, FMI, FOD, Prophase-I-pollen, Tricellular-pollen, flag, grain

模型架构：
- 基础模型：OmniGenome-52M（52M参数的基因组预训练模型）
- 任务类型：多标签多分类（每个标签有3个类别）
- 损失函数：Focal Loss（用于处理类别不平衡和难样本）
"""

import os
import math
import sys
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
os.environ["NCCL_P2P_DISABLE"] = "1"
os.environ["NCCL_IB_DISABLE"] = "1"

# 解析命令行参数（需要在导入其他模块之前，以便设置 GPU）
parser = argparse.ArgumentParser(description='Train model with Focal Loss')
parser.add_argument('--gamma', type=float, default=2.5, 
                    help='Focal Loss gamma parameter (default: 2.5)')
parser.add_argument('--alpha', type=str, default=None,
                    help='Focal Loss alpha parameter: None or "0.33,0.33,0.33" (default: None)')
parser.add_argument('--gpu', type=int, default=None,
                    help='GPU ID to use (default: None, will use default GPU)')
args = parser.parse_args()

# 设置使用指定的 GPU
if args.gpu is not None:
    if torch.cuda.is_available():
        if args.gpu < torch.cuda.device_count():
            torch.cuda.set_device(args.gpu)
            print(f"✅ 已设置使用 GPU: {torch.cuda.current_device()} (GPU ID={args.gpu})")
        else:
            print(f"⚠️  警告: GPU {args.gpu} 不存在，可用 GPU 数量: {torch.cuda.device_count()}，使用默认 GPU")
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
)


# ============================================================================
# 第一部分：Focal Loss 实现
# ============================================================================

class FocalLoss(nn.Module):
    """
    Focal Loss 实现（支持多类别）
    
    Focal Loss 公式：FL(p_t) = -α_t * (1 - p_t)^γ * log(p_t)
    
    其中：
    - p_t: 模型对真实类别的预测概率
    - α_t: 类别权重（用于处理类别不平衡）
    - γ: 聚焦参数（用于降低易分类样本的权重，关注难分类样本）
    
    参数：
        alpha (float, list, or None): 类别权重
            - None: 不使用类别权重
            - float: 单一权重值（会应用到所有类别）
            - list: 每个类别的权重，长度应该等于 num_classes
        gamma (float): 聚焦参数，默认 2.0
            - gamma=0: 退化为加权交叉熵
            - gamma=1: 中等聚焦
            - gamma=2: 常用值，较强聚焦
            - gamma=3: 更强聚焦，适合极端不平衡
        ignore_index (int): 要忽略的标签索引，默认 -100
        reduction (str): 损失归约方式，'mean' 或 'sum'，默认 'mean'
    """
    
    def __init__(self, alpha=None, gamma=2.0, ignore_index=-100, reduction='mean', num_classes=3):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ignore_index = ignore_index
        self.reduction = reduction
        self.num_classes = num_classes
        
        # 处理 alpha 参数
        if alpha is not None:
            if isinstance(alpha, (float, int)):
                # 单一值，转换为列表
                self.alpha = torch.tensor([alpha] * num_classes, dtype=torch.float32)
            elif isinstance(alpha, (list, tuple)):
                # 列表，转换为张量
                if len(alpha) != num_classes:
                    raise ValueError(f"alpha 列表长度 ({len(alpha)}) 必须等于 num_classes ({num_classes})")
                self.alpha = torch.tensor(alpha, dtype=torch.float32)
            else:
                raise ValueError(f"alpha 必须是 float, list 或 None，当前类型: {type(alpha)}")
        else:
            self.alpha = None
    
    def forward(self, logits, targets):
        """
        计算 Focal Loss
        
        参数：
            logits: [N, num_classes] 模型输出的 logits
            targets: [N] 真实标签（整数索引）
        
        返回：
            loss: 标量损失值
        """
        # 确保 alpha 在正确的设备上
        if self.alpha is not None:
            self.alpha = self.alpha.to(logits.device)
        
        # 处理 ignore_index：创建 mask 并临时替换无效标签
        if self.ignore_index is not None:
            mask = (targets != self.ignore_index)
            # 将 ignore_index 临时替换为 0，避免 gather 索引越界
            targets_safe = targets.clone()
            targets_safe[~mask] = 0
        else:
            mask = torch.ones_like(targets, dtype=torch.bool)
            targets_safe = targets
        
        # 计算交叉熵损失（数值稳定版本）
        # log_probs: [N, num_classes]
        log_probs = F.log_softmax(logits, dim=1)
        
        # 提取真实类别对应的 log 概率
        # 使用 gather 提取每个样本对应类别的 log 概率
        # targets_safe: [N] -> [N, 1]
        targets_expanded = targets_safe.unsqueeze(1)
        # log_probs_selected: [N, 1] -> [N]
        log_probs_selected = log_probs.gather(1, targets_expanded).squeeze(1)
        
        # 计算概率 p_t
        # probs: [N, num_classes]
        probs = F.softmax(logits, dim=1)
        # 提取真实类别的概率
        # probs_selected: [N]
        probs_selected = probs.gather(1, targets_expanded).squeeze(1)
        
        # 计算 focal weight: (1 - p_t)^gamma
        focal_weight = (1 - probs_selected) ** self.gamma
        
        # 计算基础损失: -log(p_t)
        base_loss = -log_probs_selected
        
        # 应用 focal weight
        focal_loss = focal_weight * base_loss
        
        # 应用类别权重 alpha
        if self.alpha is not None:
            # 为每个样本选择对应的 alpha 值
            # alpha_selected: [N]
            alpha_selected = self.alpha.gather(0, targets_safe)
            focal_loss = alpha_selected * focal_loss
        
        # 处理 ignore_index：将忽略的样本损失设为 0
        focal_loss = focal_loss * mask.float()
        
        # 归约损失
        if self.reduction == 'mean':
            if self.ignore_index is not None:
                # 只对有效样本求平均
                valid_samples = (targets != self.ignore_index).sum().float()
                if valid_samples > 0:
                    loss = focal_loss.sum() / valid_samples
                else:
                    loss = torch.tensor(0.0, device=logits.device, requires_grad=True)
            else:
                loss = focal_loss.mean()
        elif self.reduction == 'sum':
            if self.ignore_index is not None:
                # 只对有效样本求和
                loss = focal_loss.sum()
            else:
                loss = focal_loss.sum()
        else:
            loss = focal_loss
        
        return loss


# ============================================================================
# 第二部分：模型和分词器加载
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
# 第三部分：数据集类定义
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
                return -100  # 空值用-100表示，在FocalLoss中会被忽略
            
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
        tokenized_inputs = self.tokenizer(
            sequence,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        # 将所有9个标签堆叠成一个张量
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
        ], dtype=torch.long)

        # 将标签添加到tokenized_inputs中
        tokenized_inputs["labels"] = labels

        return tokenized_inputs


# ============================================================================
# 第四部分：模型类定义
# ============================================================================

class OmniModelForTriClassTESequenceClassification(OmniModelForMultiLabelSequenceClassification):
    """
    三分类TE序列分类模型 - 使用 Focal Loss
    
    模型结构：
    - 基础编码器：继承自OmniModelForMultiLabelSequenceClassification
    - 池化层：OmniPooling（提取序列的固定长度表示）
    - 分类头：线性层，输出维度为 num_labels × num_classes = 9 × 3 = 27
    
    损失函数：
    - Focal Loss：用于处理类别不平衡和难样本
    - ignore_index=-100：忽略值为-100的标签（缺失值）
    - reduction="mean"：对所有有效样本取平均损失
    
    输出格式：
    - logits: [batch_size, num_labels, num_classes] = [batch, 9, 3]
    - predictions: [batch_size, num_labels] = [batch, 9]，每个组织的预测类别
    - probabilities: [batch_size, num_labels, num_classes] = [batch, 9, 3]，每个类别的概率
    
    超参数说明：
    - focal_alpha: 类别权重，None 或 [weight_0, weight_1, weight_2]
    - focal_gamma: 聚焦参数，建议 2.0-3.0
    """

    def __init__(self, config_or_model, tokenizer, num_labels=9, num_classes=3, 
                 focal_alpha=None, focal_gamma=2.5, *args, **kwargs):
        """
        初始化模型
        
        参数：
            config_or_model: 预训练模型配置或模型路径
            tokenizer: 分词器
            num_labels: 标签数量（9个组织）
            num_classes: 每个标签的类别数（3类：0/1/2）
            focal_alpha: Focal Loss 的类别权重
            focal_gamma: Focal Loss 的聚焦参数
        """
        # 调用父类构造函数
        super().__init__(config_or_model, tokenizer, num_labels=num_labels * num_classes, *args, **kwargs)
        
        # 更新模型元数据
        self.metadata["model_name"] = self.__class__.__name__
        
        # 保存标签和类别数量
        self.num_labels = num_labels  # 9个组织
        self.num_classes = num_classes  # 3个类别（0/1/2）
        
        # 初始化池化层
        self.pooler = OmniPooling(self.config)
        
        # 初始化分类头
        self.classifier = torch.nn.Linear(self.config.hidden_size, self.num_classes * self.num_labels)
        
        # 使用 Focal Loss 替代 CrossEntropyLoss
        self.loss_fn = FocalLoss(
            alpha=focal_alpha,
            gamma=focal_gamma,
            ignore_index=-100,
            reduction="mean",
            num_classes=num_classes
        )
        
        # 保存数据集类引用
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

        # 提取logits
        logits = outputs.logits if hasattr(outputs, 'logits') else outputs[0]
        
        # 通过池化层和分类头得到最终预测
        logits = self.classifier(self.pooler(input_ids, logits))
        
        # 重塑logits：从 [batch, 27] -> [batch, 9, 3]
        batch_size = logits.shape[0]
        logits = logits.view(batch_size, self.num_labels, self.num_classes)

        # 计算损失（如果提供了标签）
        loss = None
        if labels is not None:
            # 展平logits和labels以适配FocalLoss的输入格式
            logits_flat = logits.view(-1, self.num_classes)  # [batch×9, 3]
            labels_flat = labels.view(-1)  # [batch×9]

            # 计算 Focal Loss
            loss = self.loss_fn(logits_flat, labels_flat)

        return {
            "loss": loss,
            "logits": logits,  # [batch, 9, 3]
            "last_hidden_state": outputs.last_hidden_state if hasattr(outputs, 'last_hidden_state') else None,
        }

    def predict(self, sequence_or_inputs, **kwargs):
        """预测方法（用于训练过程中的验证）"""
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
        """推理方法（用于模型部署和实际应用）"""
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
# 第五部分：数据加载
# ============================================================================

print("📊 正在加载数据集...")

# 使用 from_hub 方法从目录中自动加载 train.csv, valid.csv, test.csv
data_dir = "/home/yingjie/OmniGenBench/examples/longtail/tn0/triclassdata/8_1_1"
datasets = TriClassTEDataset.from_hub(
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
# 第六部分：模型初始化
# ============================================================================

print("\n🚀 正在初始化模型...")

# 超参数设置（从命令行参数获取）
FOCAL_ALPHA = None  # 或 [0.33, 0.33, 0.33]
if args.alpha is not None:
    if args.alpha == "0.33,0.33,0.33":
        FOCAL_ALPHA = [0.33, 0.33, 0.33]
    else:
        try:
            # 尝试解析为逗号分隔的列表
            alpha_list = [float(x.strip()) for x in args.alpha.split(',')]
            if len(alpha_list) == 3:
                FOCAL_ALPHA = alpha_list
            else:
                print(f"⚠️  警告: alpha 必须有3个值，当前有 {len(alpha_list)} 个，使用默认值 None")
        except ValueError:
            print(f"⚠️  警告: 无法解析 alpha 值 {args.alpha}，使用默认值 None")

FOCAL_GAMMA = args.gamma  # 从命令行参数获取

print(f"📌 Focal Loss 超参数:")
print(f"   - alpha: {FOCAL_ALPHA}")
print(f"   - gamma: {FOCAL_GAMMA}")

model = OmniModelForTriClassTESequenceClassification(
    model_name_or_path,
    tokenizer,
    num_labels=9,  # 9个组织
    num_classes=3,  # 3个类别: 0, 1, 2
    focal_alpha=FOCAL_ALPHA,
    focal_gamma=FOCAL_GAMMA,
    trust_remote_code=True
)


# ============================================================================
# 第七部分：评估指标定义
# ============================================================================

metric_functions = [
    ClassificationMetric(ignore_y=-100).accuracy_score,
    ClassificationMetric(ignore_y=-100, average='macro').f1_score,
    ClassificationMetric(ignore_y=-100).classification_report,
]


# ============================================================================
# 第八部分：训练器配置和训练
# ============================================================================

trainer = AccelerateTrainer(
    model=model,
    epochs=30,
    learning_rate=2e-5,
    batch_size=16,
    train_dataset=datasets["train"],
    eval_dataset=datasets.get("valid"),
    test_dataset=datasets.get("test"),
    compute_metrics=metric_functions,
    gradient_accumulation_steps=2,
    autocast="float16",
    max_grad_norm=1.0,
)

print("\n🚀 开始训练...")
# 创建目录，模型文件保存在该目录下，文件夹名称包含 gamma 和 alpha 参数
gamma_suffix = f"gamma{FOCAL_GAMMA}".replace(".", "_")  # 将 2.5 转换为 gamma2_5
if FOCAL_ALPHA is None:
    alpha_suffix = "noalpha"
else:
    # 将 [0.33, 0.33, 0.33] 转换为 alpha0_33_0_33_0_33
    alpha_str = "_".join([str(a).replace(".", "_") for a in FOCAL_ALPHA])
    alpha_suffix = f"alpha{alpha_str}"
split_dir = f"triclass_focal_{gamma_suffix}_{alpha_suffix}"
os.makedirs(split_dir, exist_ok=True)
model_save_path = os.path.join(split_dir, "ogb_te_3class_focal_loss_52m")
print(f"📁 模型将保存到: {model_save_path}")

metrics = trainer.train(
    path_to_save=model_save_path,
    dataset_class=TriClassTEDataset,
)
print("\n📊 最终指标:", metrics)
print("\n🎉 训练完成！")

