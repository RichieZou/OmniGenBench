

# -*- coding: utf-8 -*-
# file: triclass_te.py
# time: 09:35 07/10/2025
# author: YANG, HENG <hy345@exeter.ac.uk> (杨恒)
# homepage: https://yangheng95.github.io
# github: https://github.com/yangheng95
# huggingface: https://huggingface.co/yangheng
# google scholar: https://scholar.google.com/citations?user=NPq5a_0AAAAJ&hl=en
# Copyright (C) 2019-2025. All Rights Reserved.

import os
import warnings
import torch
import math
import numpy as np
from typing import Dict, Any
from tqdm import tqdm

os.environ["NCCL_P2P_DISABLE"] = "1"
os.environ["NCCL_IB_DISABLE"] = "1"


from omnigenbench import (
    ClassificationMetric,
    AccelerateTrainer,
    ModelHub,
    OmniTokenizer,
    OmniDatasetForSequenceClassification,
    OmniModelForSequenceClassification,
    OmniPooling,
)
from omnigenbench.src.misc.utils import fprint

# model_name_or_path = "yangheng/OmniGenome-52M"
# model_name_or_path = "yangheng/OmniGenome-v1.5"
# model_name_or_path = "SpliceBERT-510nt"
# model_name_or_path = "InstaDeepAI/nucleotide-transformer-v2-100m-multi-species"
model_name_or_path = "/home/yingjie/OmniGenBench/models_cache/OmniGenome-52M"


# Load tokenizer
tokenizer = OmniTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)


class BiClassTEDataset(OmniDatasetForSequenceClassification):
    """Dataset for binary classification (0/1) single-label classification
    
    继承说明：
    - 继承自 OmniDatasetForSequenceClassification，用于单标签分类任务
    - 重写 prepare_input() 方法以适配二分类任务的特殊需求
    - 输入：DOT列（二级结构序列）
    - 标签：LABLE列（0或1）
    """

    def __init__(self, **kwargs):
        # 调用父类的初始化方法，继承父类的属性和行为
        super().__init__(**kwargs)

    def prepare_input(self, instance, **kwargs):
        # Map labels to indices: 0=0, 1=1, nan=-100
        label2idx = {'0': 0, '1': 1, 'nan': -100}

        # Extract label from LABLE column
        label_value = str(instance["LABLE"])
        label = label2idx.get(label_value, -100)
        
        # Extract sequence from DOT column
        sequence = instance["DOT"]

        # Tokenize sequence
        tokenized_inputs = self.tokenizer(
            sequence,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        # Create single label tensor for binary classification
        # Use scalar tensor so DataLoader stacks to [batch_size] shape (not [batch_size, 1])
        # This ensures consistent 1D shape [batch_size] matching predictions
        tokenized_inputs["labels"] = torch.tensor(label, dtype=torch.long)  # Shape: () -> DataLoader stacks to [batch_size]

        return tokenized_inputs


# 直接使用父类，无需创建子类


# Load datasets
print("📊 Loading datasets...")
datasets = BiClassTEDataset.from_hub(
    "/home/yingjie/OmniGenBench/examples/second_structure_for_classification/data/8_1_1_filter/",  # 指定具体的数据目录
    tokenizer=tokenizer,
    max_length=512,
    force_padding=False
)

print("📝 Data loading completed!")
print(f"📊 Loaded datasets: {list(datasets.keys())}")
for split, dataset in datasets.items():
    print(f"  - {split}: {len(dataset)} samples")

# Initialize model
print("\n🚀 Initializing model...")
model = OmniModelForSequenceClassification(
    model_name_or_path,
    tokenizer,
    num_labels=2,  # 2 classes: 0, 1
    trust_remote_code=True
)

# 设置数据集类和损失函数
model.dataset_class = BiClassTEDataset
model.loss_fn = torch.nn.CrossEntropyLoss(ignore_index=-100, reduction="mean")

# 重写 loss_function 方法以使用更新后的 loss_fn
def loss_function_with_ignore_index(logits, labels):
    """损失函数，支持 ignore_index=-100"""
    return model.loss_fn(logits.view(-1, model.config.num_labels), labels.view(-1))

model.loss_function = loss_function_with_ignore_index

# 重写 predict 方法以确保 predictions 形状与 labels 一致
# 注意：labels 从 DataLoader 来可能是 [batch_size] 或 [batch_size, 1]
# 我们需要确保 predictions 的形状与 labels 完全一致
def predict_with_consistent_shape(sequence_or_inputs, **kwargs):
    """重写 predict 方法，确保 predictions 形状与 labels 一致"""
    raw_outputs = model._forward_from_raw_input(sequence_or_inputs, **kwargs)
    logits = raw_outputs["logits"]  # [batch_size, num_labels]
    last_hidden_state = raw_outputs["last_hidden_state"]
    
    # 直接对 logits 应用 argmax，得到 [batch_size] 形状
    predictions = torch.argmax(logits, dim=-1)  # [batch_size]
    
    # 确保predictions是1D，与labels的形状一致
    # 如果labels是 [batch_size, 1]，我们需要unsqueeze；如果是 [batch_size]，保持原样
    # 但为了兼容性，我们确保predictions始终是1D [batch_size]
    if predictions.dim() == 0:
        predictions = predictions.unsqueeze(0)
    elif predictions.dim() > 1:
        predictions = predictions.squeeze()
    
    # 确保predictions是1D [batch_size]，这样与labels的 [batch_size] 形状一致
    # 即使labels被堆叠成 [batch_size, 1]，numpy连接时也会自动处理
    return {
        "predictions": predictions,
        "logits": logits,
        "last_hidden_state": last_hidden_state,
    }

model.predict = predict_with_consistent_shape

# 创建一个包装类，重写 evaluate 方法以确保形状一致
class FixedAccelerateTrainer(AccelerateTrainer):
    """修复评估时形状不一致问题的Trainer"""
    
    def evaluate(self) -> Dict[str, Any]:
        """重写evaluate方法，确保所有数组在连接前都是1D"""
        if self.eval_loader is None:
            return {}
        
        self.model.eval()
        all_preds = []
        all_truth = []
        
        it = tqdm(
            self.eval_loader,
            desc="Evaluating",
            disable=not self.accelerator.is_main_process,
        )
        
        with torch.no_grad():
            for batch in it:
                output = self.accelerator.unwrap_model(self.model).predict(batch)
                predictions = output["predictions"]
                labels = batch["labels"]
                
                # Gather predictions and labels from all processes
                gathered_predictions = self.accelerator.gather(predictions)
                gathered_labels = self.accelerator.gather(labels)
                
                # Only main process processes gathered data
                if self.accelerator.is_main_process:
                    gathered_predictions = gathered_predictions.float().cpu().numpy()
                    gathered_labels = gathered_labels.float().cpu().numpy()
                    
                    # 确保所有数组都是1D，展平2D数组
                    if gathered_predictions.ndim > 1:
                        gathered_predictions = gathered_predictions.flatten()
                    if gathered_labels.ndim > 1:
                        gathered_labels = gathered_labels.flatten()
                    
                    all_preds.append(gathered_predictions)
                    all_truth.append(gathered_labels)
        
        # Only main process computes metrics
        if self.accelerator.is_main_process:
            all_preds = np.concatenate(all_preds, axis=0)
            all_truth = np.concatenate(all_truth, axis=0)
            
            if not np.all(all_truth == -100):
                valid_metrics = {}
                for metric_func in self.compute_metrics:
                    valid_metrics.update(metric_func(all_truth, all_preds))
            else:
                valid_metrics = {
                    "Validation labels predictions may be NaN. No metrics calculated.": 0
                }
            fprint(valid_metrics)
        else:
            valid_metrics = None
        
        self.predictions.update({"valid": {"pred": all_preds, "true": all_truth}})
        return valid_metrics
    
    def test(self) -> Dict[str, Any]:
        """重写test方法，确保所有数组在连接前都是1D"""
        if self.test_loader is None:
            return {}
        
        self.model.eval()
        all_preds = []
        all_truth = []
        
        it = tqdm(
            self.test_loader,
            desc="Testing",
            disable=not self.accelerator.is_main_process,
        )
        
        with torch.no_grad():
            for batch in it:
                output = self.accelerator.unwrap_model(self.model).predict(batch)
                predictions = output["predictions"]
                labels = batch["labels"]
                
                gathered_predictions = self.accelerator.gather(predictions)
                gathered_labels = self.accelerator.gather(labels)
                
                if self.accelerator.is_main_process:
                    gathered_predictions = gathered_predictions.float().cpu().numpy()
                    gathered_labels = gathered_labels.float().cpu().numpy()
                    
                    # 确保所有数组都是1D，展平2D数组
                    if gathered_predictions.ndim > 1:
                        gathered_predictions = gathered_predictions.flatten()
                    if gathered_labels.ndim > 1:
                        gathered_labels = gathered_labels.flatten()
                    
                    all_preds.append(gathered_predictions)
                    all_truth.append(gathered_labels)
        
        # Only main process computes metrics
        if self.accelerator.is_main_process:
            all_preds = np.concatenate(all_preds, axis=0)
            all_truth = np.concatenate(all_truth, axis=0)
            
            if not np.all(all_truth == -100):
                test_metrics = {}
                for metric_func in self.compute_metrics:
                    test_metrics.update(metric_func(all_truth, all_preds))
            else:
                test_metrics = {
                    "Test labels predictions may be NaN. No metrics calculated.": 0
                }
            fprint(test_metrics)
        else:
            test_metrics = None
        
        self.predictions.update({"test": {"pred": all_preds, "true": all_truth}})
        return test_metrics

# Define metrics: accuracy and F1 score
# - accuracy_score: 计算整体分类准确率，忽略标签为-100的样本（通常用于padding或无效标签）
# - f1_score: 计算F1分数（精确率和召回率的调和平均），使用macro平均（对每个类别计算F1后取平均，适合类别不平衡的情况）
# 
# 计算时机：这些metrics会在训练过程中应用于eval_dataset（验证集）和test_dataset（测试集）
# 计算方式：
#   1. 模型对验证集/测试集进行前向传播，得到预测结果（logits）
#   2. 将logits转换为预测类别（argmax）
#   3. 将预测类别与真实标签进行比较，忽略标签为-100的位置
#   4. accuracy_score: 正确预测数 / 有效样本总数
#   5. f1_score (macro): 对每个类别分别计算F1值，然后取平均值
metric_functions = [
    ClassificationMetric(ignore_y=-100).accuracy_score,  # 准确率：正确预测的样本数 / 总样本数
    ClassificationMetric(ignore_y=-100, average='macro').f1_score,  # 宏平均F1：(TP) / (TP + 0.5*(FP+FN))，对所有类别取平均
]

# Initialize trainer
 # batch_size: 每次从数据集中加载并处理的样本数量
    # - 这里设置为16，表示每个训练步骤会处理16个序列样本
    # - 较小的batch_size可以减少GPU内存占用，但训练可能不够稳定
    # - 较大的batch_size可以提高训练稳定性和速度，但需要更多GPU内存

# gradient_accumulation_steps: 梯度累积步数
    # - 这里设置为4，表示每4个batch才进行一次参数更新
    # - 实际有效batch_size = batch_size × gradient_accumulation_steps = 16 × 4 = 64
    # - 作用：在GPU内存有限的情况下，通过累积多个小batch的梯度来模拟大batch训练
    # - 工作原理：
    #   1. 前向传播和反向传播计算梯度（但不更新参数）
    #   2. 将梯度累加到之前的梯度上
    #   3. 重复步骤1-2共4次
    #   4. 第4次后，使用累积的梯度更新模型参数，然后清零梯度
    # - 优点：可以用较小的GPU内存训练出与大batch相当的效果
    
# 训练时不会用到test_dataset，它仅在训练完成后用于最终评估
# - train_dataset: 用于模型训练，更新模型参数
# - eval_dataset: 用于训练过程中的验证，监控过拟合，选择最佳模型
# - test_dataset: 仅在训练完成后用于最终性能评估，不参与训练过程

trainer = FixedAccelerateTrainer(
    model=model,
    epochs=20,
    learning_rate=2e-5,
    batch_size=24,  # 每次训练的样本数量
    train_dataset=datasets["train"],
    eval_dataset=datasets["valid"],
    test_dataset=datasets["test"],  # 仅用于训练后的最终测试，不影响训练过程
    compute_metrics=metric_functions,
    gradient_accumulation_steps=2,
)


print("\n🚀 开始训练...")
# 创建保存目录
save_dir = "structure_binary_classification_model_filtered_from_heng_orig"
os.makedirs(save_dir, exist_ok=True)
model_save_path = os.path.join(save_dir, "ogb_structure_2class_52m")
print(f"📁 模型将保存到: {model_save_path}")

metrics = trainer.train(
    path_to_save=model_save_path,
    dataset_class=BiClassTEDataset,
)
print("\n📊 最终指标:", metrics)
print("\n🎉 训练完成！")



# === Model Inference ===
# print("\n🔮 Starting inference on test samples...")

# inference_model = ModelHub.load("/home/yingjie/OmniGenBench/examples/heng_orig/ogb_te_3class_finetuned_epoch_19_seed_42_accuracy_score_0.9900_seed_42_f1_score_0.9900")

# # Get some test samples
# # sample_sequences = datasets['test'].sample(1000).examples

# #sample_sequences = datasets['valid'].sample(1000).examples
# sample_sequences = datasets['train'].examples[:1]

# label_names = ['Low', 'Medium', 'High']
# tissue_names = [
#     'root', 'seedling', 'leaf', 'FMI', 'FOD',
#     'Prophase-I-pollen', 'Tricellular-pollen', 'flag', 'grain'
# ]

# with torch.no_grad():
#     for row in sample_sequences:
#         sequence = row["sequence"]
#         print(f"\n{'='*60}")
#         print(f"🧬 Sample ID: {row['ID']}")
#         print(f"📏 Sequence length: {len(sequence)} bp")

#         outputs = inference_model.inference(sequence, **row)
#         predictions = outputs['predictions'].cpu().numpy() # tensor([0, 2, 1, 2, 0, 2, 2, 1, 2], device='cuda:0') 9个tissue的预测类别
#         probabilities = outputs['probabilities'].cpu().numpy() # 9*3的tensor，每个tissue的3个类别的概率 （logits --> softmax）
#         confidence = outputs['confidence'].cpu().numpy() # 9个tissue的预测置信度 tensor([0.9990, 1.0000, 0.5112, 0.9834, 1.0000, 0.9985, 0.9990, 0.9995, 1.0000], probabilities中的最大值
#         # last_hidden_state = outputs['last_hidden_state'].cpu().numpy() # 9*512的tensor，每个tissue的512个token的隐藏状态


#         print(f"\n📊 Predictions for 9 tissues:")
#         for i, tissue in enumerate(tissue_names):
#             pred_class = predictions[i]
#             pred_label = label_names[pred_class]
#             conf = confidence[i]
#             probs = probabilities[i]

#             # Get ground truth if available
#             gt_col = f"{tissue}_TE_label"
#             if gt_col in row:
#                 gt_label = row[gt_col]
#                 if isinstance(gt_label, float) and math.isnan(gt_label):
#                     continue
#                 match_emoji = "✅" if pred_label == gt_label else "❌"
#                 print(f"  {match_emoji} {tissue:25s}: {pred_label:6s} (conf: {conf:.3f}) [GT: {gt_label}]")
#             else:
#                 print(f"  🔹 {tissue:25s}: {pred_label:6s} (conf: {conf:.3f})")

#             # Show probability distribution
#             print(f"      Probs - Low: {probs[0]:.3f}, Medium: {probs[1]:.3f}, High: {probs[2]:.3f}")

# print("\n🎉 All tasks completed!")