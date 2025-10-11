# Step 1: Data gathering

from omnigenbench import (
    ClassificationMetric,#提供常见分类评估指标
    AccelerateTrainer,
    ModelHub,#访问和下载预训练模型。
    OmniTokenizer,#序列分词器，用于把 RNA 序列转为模型输入张量。
    OmniDatasetForSequenceClassification,
    OmniModelForSequenceClassification,
)

model_name_or_path = "yangheng/OmniGenome-186M" #来自huggingface的模型
dataset_name = "translation_efficiency_prediction"


#标签映射字典
label2id = {"0": 0, "1": 1}  # 0: Low TE, 1: High TE

# Initialize tokenizer
tokenizer = OmniTokenizer.from_pretrained(model_name_or_path)

# ---------------------------------------------
#将输入序列转换为张量字典传入模型。
# Method 1: From Hugging Face Hub (automatic download)
datasets = OmniDatasetForSequenceClassification.from_huggingface(
    dataset_name="translation_efficiency_prediction",
    tokenizer=tokenizer,
    max_length=512,
    label2id=label2id,
)

# # Method 2: From local directory (JSONL files)
# datasets = OmniDatasetForSequenceClassification(
#     "/home/yz1033/OmniGenBench/examples/translation_efficiency_prediction/__OMNIGENOME_DATA__/datasets/translation_efficiency_prediction",
#     tokenizer=tokenizer,
#     max_length=512,
#     label2id=label2id,
# )
# # Method 3: From CSV file
# datasets = OmniDatasetForSequenceClassification(
#     "/home/yz1033/OmniGenBench/examples/translation_efficiency_prediction/__OMNIGENOME_DATA__/datasets/translation_efficiency_prediction",
#     tokenizer=tokenizer,
#     sequence_column="sequence",
#     label_column="label",
#     max_length=512,
#     label2id=label2id,
# )
# ---------------------------------------------


#打印数据格式
print(f"📊 Loaded datasets: {list(datasets.keys())}")
for split, dataset in datasets.items():
    print(f"  - {split}: {len(dataset)} samples")


#构建模型
model = OmniModelForSequenceClassification(
    model_name_or_path,  # PlantRNA-FM
    tokenizer,
    num_labels=len(list(label2id.keys())),  # Binary classification: Low TE vs High TE
)

#指定训练器以及分类指标f1——score
metric_functions = [ClassificationMetric().f1_score]

#此处可以指定超参数
trainer = AccelerateTrainer(
    model=model,
    train_dataset=datasets["train"],
    eval_dataset=datasets["valid"],
    test_dataset=datasets["test"],
    compute_metrics=metric_functions,
)
print("🎓 Starting training...")

metrics = trainer.train()
trainer.save_model("ogb_te_finetuned")#保存训练好的模型

print('Metrics:', metrics)


#模型推理
inference_model = ModelHub.load("yangheng/ogb_te_finetuned")#加载微调后的模型

#传入三条序列，最后的是随机序列以测试模型
sample_sequences = {
    "Optimized sequence": "AAACCAACAAAATGCAGTAGAAGTACTCTCGAGCTATAGTCGCGACGTGCTGCCCCGCAGGAGTACAGTAGTAGTACAACGTAAGCGGGAGCAACAGACTCCCCCCCTGCAACCCACTGTGCCTGTGCCCTCGACGCGTCTCCGTCGCTTTGGCAAATGTCACGTACATATTACCGTCTCAGGCTCTCAGCCATGCTCCCTACCACCCCTGCAGCGAAGCAAAAGCCACGCACGCGGCGCCTGACATGTAACAGGACTAGACCATCTTGTTCATTTCCCGCACCCCCTCCTCTCCTCTTCCTCCATCTGCCTCTTTAAAACAGTAAAAATAACCGTGCATCCCCTGGGCAAAATCTCTCCCATACATACACTACAGCGGCGAACCTTTCCTTATTCTCGCAACGCCTCGGTAACGGGCAGCGCCTGCTCCGCGCCGCGGTTGCGAGTTCGGGAAGGCGGCCGGAGTCGCGGGGAGGAGAGGGAGGATTCGATCGGCCAGA",
    "Suboptimal sequence": "TGGAGATGGGCAGATGGCACACAAAACATGAATAGAAAACCCAAAAGGAAGGATGAAAAAAACACACACACACACACACACAAAACACAGAGAGAGAGAGAGAGAGAGCGAGAAAAGAAAAGAAAAAACCAATTCTTTTGGTCTCTTCCCTCTCCGTTTGTCGTGTCGAAGCCTTTGCCCCCACCACCTCCTCCTCTCCTCTCCCTTCCTCCCCTCCTCCCCATCTCGCTCTCCTCCCTCCTCTCTCCTCTCCTCGTCTCCTCTTCCTCTCCATTCCATTGGCCATTCCATTCCATTCCACCCCCCATGAAACCCCAAACCCTCGTCGGCCTCGCCGCGCTCGCGTAGCGCACCCGCCCTTCTCCTCTCGCCGGTGGTCCGCCGCCAGCCTCCCCCCACCCGATCCCGCCGCCCCCCCCGCCTTCACCCCGCCCACGCGGACGCATCCGATCCCGCCGCATCGCCGCGCGGGGGGGGGGGGGGGGGGGGGGGGGAGGGCACG",
    "Random sequence": "AUGC" * (128 // 4),
}

#给出上面三个序列的预测值，并返回一个 softmax 概率作为信心值
for seq_name, sequence in sample_sequences.items():
    outputs = inference_model.inference(sequence)

    # —— Result Interpretation ——
    prediction = outputs['predictions']
    confidence = outputs['confidence']
    print(f"  - Predicted Translation Efficiency: {prediction} (Confidence: {confidence:.2f})")

















