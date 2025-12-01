# Step 1: Data Preparation
from omnigenbench import (
    ClassificationMetric,
    AccelerateTrainer,
    ModelHub,
    OmniTokenizer,
    OmniDatasetForSequenceClassification,
    OmniModelForSequenceClassification,
)

import os
os.environ["NCCL_P2P_DISABLE"] = "1"
os.environ["NCCL_IB_DISABLE"] = "1"

# # 自定义数据集类：使用已有的结构信息，而不是自动预测
# class OmniDatasetWithExistingStructure(OmniDatasetForSequenceClassification):
#     """
#     使用数据中已有的结构信息的数据集类。
#     如果数据中包含 'structure' 字段，直接使用；否则回退到自动预测。
#     """
    
#     def _preprocessing(self):
#         """
#         重写预处理方法：优先使用数据中已有的结构信息。
#         """
#         for idx, ex in enumerate(self.examples):
#             # 处理不同的序列字段名（SEQ, seq, sequence, text）
#             if "SEQ" in self.examples[idx]:
#                 self.examples[idx]["sequence"] = self.examples[idx]["SEQ"]
#                 del self.examples[idx]["SEQ"]
#             if "seq" in self.examples[idx]:
#                 self.examples[idx]["sequence"] = self.examples[idx]["seq"]
#                 del self.examples[idx]["seq"]
#             if "text" in self.examples[idx]:
#                 self.examples[idx]["sequence"] = self.examples[idx]["text"]
#                 del self.examples[idx]["text"]

#             if "sequence" not in self.examples[idx]:
#                 import warnings
#                 warnings.warn("The 'sequence' field is missing in the raw dataset.")
        
#         if len(self.examples) > 0 and "sequence" in self.examples[0]:
#             sequences = [ex["sequence"] for ex in self.examples]
#             if self.structure_in:
#                 # 检查数据中是否已经包含结构信息（支持不同的大小写）
#                 has_structure = False
#                 structure_key = None
#                 for key in ["structure", "Structure", "STRUCTURE"]:
#                     if key in self.examples[0]:
#                         has_structure = True
#                         structure_key = key
#                         break
                
#                 if has_structure:
#                     # 使用已有的结构信息
#                     for idx, ex in enumerate(self.examples):
#                         structure = ex.get(structure_key, "")
#                         sequence = ex["sequence"]
#                         self.examples[idx]["sequence"] = f"{sequence}{self.tokenizer.eos_token}{structure}"
#                 else:
#                     # 如果没有结构信息，则自动预测（回退到原始行为）
#                     structures = self.rna2structure.fold(sequences)
#                     for idx, (sequence, structure) in enumerate(zip(sequences, structures)):
#                         self.examples[idx]["sequence"] = f"{sequence}{self.tokenizer.eos_token}{structure}"

model_name_or_path = "/home/yingjie/OmniGenBench/models_cache/OmniGenome-v1.5"
# dataset_name = "translation_efficiency_prediction"

# Model and Tokenizer

# We define the label mapping in the training
label2id = {"0": 0, "1": 1}  # 0: Low TE, 1: High TE

# Initialize tokenizer
tokenizer = OmniTokenizer.from_pretrained(model_name_or_path)

# Load datasets
# 使用自定义数据集类的 from_hub 方法，它会自动使用已有的结构信息
datasets = OmniDatasetForSequenceClassification.from_hub(
    "/home/yingjie/OmniGenBench/examples/second_structure_for_classification/data/8_1_1_filter", # 指定具体的数据目录
    tokenizer=tokenizer,
    max_length=512,
    label2id=label2id,
    # structure_in=True,  # 启用结构信息：使用数据中已有的结构信息（如果存在），否则自动预测
)
print(f"📊 Loaded datasets: {list(datasets.keys())}")
for split, dataset in datasets.items():
    print(f"  - {split}: {len(dataset)} samples")

# Step 2: Model Initialization
# === Model Initialization ===
# We support all genomic foundation models from Hugging Face Hub.
model = OmniModelForSequenceClassification(
    model_name_or_path,
    tokenizer,
    num_labels=len(list(label2id.keys())),  # Binary classification: Low TE vs High TE
)

# Step 3: Model Training
metric_functions = [
    ClassificationMetric().accuracy_score,  # 准确率：正确预测的样本数 / 总样本数
    ClassificationMetric().f1_score]

# 🔑 关键修复：使用 drop_last=True 解决形状不一致问题
# 训练、验证、测试都使用 drop_last=True，丢弃最后一个不完整的 batch
from torch.utils.data import DataLoader

batch_size = 4
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
    epochs=50,
    learning_rate=2e-5,
    train_loader=train_loader,
    eval_loader=eval_loader,
    test_loader=test_loader,
    compute_metrics=metric_functions,
    gradient_accumulation_steps=16,
)
print("🎓 Starting training...")

# metrics = trainer.train()
# trainer.save_model("ogb_te_finetuned")

save_dir = "OnlySeqInput"
os.makedirs(save_dir, exist_ok=True)
model_save_path = os.path.join(save_dir, "ogb_structure_te_v15")
print(f"📁 模型将保存到: {model_save_path}")

# # trainer.save_model(path_to_save="ogb_te_3class_finetuned", dataset_class=TriClassTEDataset)
metrics = trainer.train(path_to_save=model_save_path, dataset_class=OmniDatasetForSequenceClassification)
print('Final Metrics:', metrics)


# # Step 4: Model Inference and Interpretation
# inference_model = ModelHub.load("yangheng/ogb_te_finetuned")

# sample_sequences = {
#     "Optimized sequence": "AAACCAACAAAATGCAGTAGAAGTACTCTCGAGCTATAGTCGCGACGTGCTGCCCCGCAGGAGTACAGTAGTAGTACAACGTAAGCGGGAGCAACAGACTCCCCCCCTGCAACCCACTGTGCCTGTGCCCTCGACGCGTCTCCGTCGCTTTGGCAAATGTCACGTACATATTACCGTCTCAGGCTCTCAGCCATGCTCCCTACCACCCCTGCAGCGAAGCAAAAGCCACGCACGCGGCGCCTGACATGTAACAGGACTAGACCATCTTGTTCATTTCCCGCACCCCCTCCTCTCCTCTTCCTCCATCTGCCTCTTTAAAACAGTAAAAATAACCGTGCATCCCCTGGGCAAAATCTCTCCCATACATACACTACAGCGGCGAACCTTTCCTTATTCTCGCAACGCCTCGGTAACGGGCAGCGCCTGCTCCGCGCCGCGGTTGCGAGTTCGGGAAGGCGGCCGGAGTCGCGGGGAGGAGAGGGAGGATTCGATCGGCCAGA",
#     "Suboptimal sequence": "TGGAGATGGGCAGATGGCACACAAAACATGAATAGAAAACCCAAAAGGAAGGATGAAAAAAACACACACACACACACACACAAAACACAGAGAGAGAGAGAGAGAGAGCGAGAAAAGAAAAGAAAAAACCAATTCTTTTGGTCTCTTCCCTCTCCGTTTGTCGTGTCGAAGCCTTTGCCCCCACCACCTCCTCCTCTCCTCTCCCTTCCTCCCCTCCTCCCCATCTCGCTCTCCTCCCTCCTCTCTCCTCTCCTCGTCTCCTCTTCCTCTCCATTCCATTGGCCATTCCATTCCATTCCACCCCCCATGAAACCCCAAACCCTCGTCGGCCTCGCCGCGCTCGCGTAGCGCACCCGCCCTTCTCCTCTCGCCGGTGGTCCGCCGCCAGCCTCCCCCCACCCGATCCCGCCGCCCCCCCCGCCTTCACCCCGCCCACGCGGACGCATCCGATCCCGCCGCATCGCCGCGCGGGGGGGGGGGGGGGGGGGGGGGGGAGGGCACG",
#     "Random sequence": "AUGC" * (128 // 4),
# }
# for seq_name, sequence in sample_sequences.items():
#     outputs = inference_model.inference(sequence)

#     # —— Result Interpretation ——
#     prediction = outputs['predictions']
#     confidence = outputs['confidence']
#     print(f"  - Predicted Translation Efficiency: {prediction} (Confidence: {confidence:.2f})")
