#!/usr/bin/env python3
"""
使用已加载的 OmniGenome 序列分类模型，直接提取输入序列在最后一层的隐藏表示。
无需训练，只要前向传播即可获得 hidden states。
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    adjusted_rand_score,
    normalized_mutual_info_score,
    homogeneity_score,
    completeness_score,
    v_measure_score,
    silhouette_score,
    accuracy_score,
    classification_report,
)
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np

from omnigenbench import (
    OmniDatasetForSequenceClassification,
    OmniModelForSequenceClassification,
    OmniTokenizer,
)


LABEL2ID = {"0": 0, "1": 1}


class OmniDatasetWithColumnMapping(OmniDatasetForSequenceClassification):
    """
    扩展数据集，自动将自定义列名映射为标准字段，避免手动修改原始 CSV。
    """

    def _preprocessing(self):
        for idx, ex in enumerate(self.examples):
            if "SEQ" in ex and "seq" not in ex:
                ex["seq"] = ex["SEQ"]
                del ex["SEQ"]
            if "LABLE" in ex and "label" not in ex:
                label_val = ex["LABLE"]
                # 处理空标签：空字符串、None、NaN 等都设为 None
                if label_val is None or (isinstance(label_val, str) and label_val.strip() == ""):
                    ex["label"] = "-100"
                else:
                    ex["label"] = label_val
            if "LABEL" in ex and "label" not in ex:
                label_val = ex["LABEL"]
                if label_val is None or (isinstance(label_val, str) and label_val.strip() == ""):
                    ex["label"] = "-100"
                else:
                    ex["label"] = label_val
            if "label" in ex and (ex["label"] is None or (isinstance(ex["label"], str) and ex["label"].strip() == "")):
                ex["label"] = "-100"
        super()._preprocessing()
    
    def print_label_distribution(self):
        """重写方法，跳过 -100 标签（缺失标签）以避免 KeyError。"""
        if not self.data or "labels" not in self.data[0]:
            return
        
        label_counts = {}
        for item in self.data:
            label = item["labels"].item() if hasattr(item["labels"], "item") else item["labels"]
            if label != -100:  # 跳过缺失标签
                label_counts[label] = label_counts.get(label, 0) + 1
        
        if label_counts:
            print(f"\n📊 Label Distribution:")
            for label in sorted(label_counts.keys()):
                label_name = self.id2label.get(label, f"Unknown({label})")
                count = label_counts[label]
                percentage = (count / len(self.data)) * 100
                print(f"  {label_name}: {count} ({percentage:.2f}%)")
            print(f"  Missing labels (-100): {len(self.data) - sum(label_counts.values())}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="提取最后一层隐藏表示（hidden states），无需训练。"
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default="/home/yingjie/OmniGenBench/models_cache/OmniGenome-v1.5",
        help="预训练模型的路径或名称。",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="/home/yingjie/OmniGenBench/examples/second_structure_for_classification/data/8_1_1_filter",
        help="包含 train/valid/test CSV 的数据目录。",
    )
    parser.add_argument(
        "--split",
        type=str,
        choices=["train", "valid", "test", "all"],
        default="all",
        help="选择要处理的数据划分。",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="前向推理的 batch 大小。",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=512,
        help="tokenizer 截断 / 填充的最大长度。",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="tsne/hidden_states.pt",
        help="保存 hidden states 的输出路径（.pt 文件）。",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="DataLoader 的 num_workers。",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="推理所使用的设备，例如 auto、cpu、cuda、cuda:1。",
    )
    parser.add_argument(
        "--tsne",
        action="store_true",
        help="提取 hidden states 后，使用标签执行 tSNE 降维并可视化。",
    )
    parser.add_argument(
        "--tsne-output",
        type=str,
        default="tsne/tsne.png",
        help="tSNE 可视化图像的输出路径。",
    )
    parser.add_argument(
        "--tsne-perplexity",
        type=float,
        default=30.0,
        help="tSNE 的 perplexity 参数。",
    )
    parser.add_argument(
        "--tsne-learning-rate",
        type=float,
        default=200.0,
        help="tSNE 的学习率。",
    )
    parser.add_argument(
        "--representation",
        type=str,
        choices=["all_tokens", "cls_token", "pooler_output", "mean_pooled", "max_pooled", "last_token"],
        default="cls_token",
        help="选择用于 tSNE 的序列表示方式。默认使用 cls_token。",
    )
    return parser.parse_args()


def prepare_datasets(
    data_dir: str,
    tokenizer: OmniTokenizer,
    max_length: int,
) -> Dict[str, OmniDatasetForSequenceClassification]:
    datasets = OmniDatasetWithColumnMapping.from_hub(
        data_dir,
        tokenizer=tokenizer,
        max_length=max_length,
        label2id=LABEL2ID,
    )
    return datasets


def move_to_device(batch: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    return {
        key: value.to(device) if hasattr(value, "to") else value
        for key, value in batch.items()
    }


@torch.no_grad()
def forward_hidden_states(
    model: OmniModelForSequenceClassification,
    dataloader: DataLoader,
    device: torch.device,
) -> Dict[str, torch.Tensor | None]:
    """
    提取多种序列表示方式。
    
    返回一个字典，包含：
    - all_tokens: 所有 token 的 hidden states [batch_size, seq_len, hidden_size]
    - cls_token: [CLS] token 的表示 [batch_size, hidden_size]
    - pooler_output: 经过 pooler 的表示（如果可用）[batch_size, hidden_size]
    - mean_pooled: 平均池化表示 [batch_size, hidden_size]
    - max_pooled: 最大池化表示 [batch_size, hidden_size]
    - last_token: 最后一个非 padding token 的表示 [batch_size, hidden_size]
    - labels: 标签 [batch_size]
    """
    all_tokens_chunks: List[torch.Tensor] = []
    cls_token_chunks: List[torch.Tensor] = []
    pooler_output_chunks: List[torch.Tensor] = []
    mean_pooled_chunks: List[torch.Tensor] = []
    max_pooled_chunks: List[torch.Tensor] = []
    last_token_chunks: List[torch.Tensor] = []
    label_chunks: List[torch.Tensor] = []

    for batch in tqdm(dataloader, desc="Extracting hidden states"):
        batch_on_device = move_to_device(batch, device)
        inputs = {k: v for k, v in batch_on_device.items() if k != "labels"}
        attention_mask = inputs.get("attention_mask", None)
        
        # 直接调用底层 base model 获取所有层的 hidden states
        base_model = model.model
        base_outputs = base_model(**inputs, output_hidden_states=True)
        
        # 获取最后一层的 hidden states
        if hasattr(base_outputs, "hidden_states") and base_outputs.hidden_states is not None:
            last_hidden = base_outputs.hidden_states[-1].detach().cpu()
        elif isinstance(base_outputs, dict) and "hidden_states" in base_outputs:
            last_hidden = base_outputs["hidden_states"][-1].detach().cpu()
        elif hasattr(base_outputs, "last_hidden_state"):
            last_hidden = base_outputs.last_hidden_state.detach().cpu()
        elif isinstance(base_outputs, dict) and "last_hidden_state" in base_outputs:
            last_hidden = base_outputs["last_hidden_state"].detach().cpu()
        else:
            raise ValueError("无法从模型输出中获取 hidden states")
        
        # 1. 所有 token 的 hidden states [batch_size, seq_len, hidden_size]
        all_tokens_chunks.append(last_hidden)
        
        # 2. [CLS] token 的表示 [batch_size, hidden_size]
        cls_token = last_hidden[:, 0, :]
        cls_token_chunks.append(cls_token)
        
        # 3. Pooler output（如果模型有提供）
        pooler_output = None
        if hasattr(base_outputs, "pooler_output") and base_outputs.pooler_output is not None:
            pooler_output = base_outputs.pooler_output.detach().cpu()
        elif isinstance(base_outputs, dict) and "pooler_output" in base_outputs:
            pooler_output = base_outputs["pooler_output"].detach().cpu()
        elif hasattr(model, "pooler"):
            # 使用模型的 pooler 来获取池化表示
            try:
                pooler_output = model.pooler(inputs, last_hidden.to(device)).detach().cpu()
            except:
                pooler_output = None
        
        if pooler_output is not None:
            pooler_output_chunks.append(pooler_output)
        
        # 4. 平均池化：对所有非 padding token 求平均
        if attention_mask is not None:
            attention_mask_cpu = attention_mask.detach().cpu()
            # 扩展 attention_mask 的维度以匹配 hidden states
            mask_expanded = attention_mask_cpu.unsqueeze(-1).expand_as(last_hidden)
            # 将 padding 位置设为 0
            masked_hidden = last_hidden * mask_expanded
            # 计算每个序列的有效长度
            seq_lengths = attention_mask_cpu.sum(dim=1, keepdim=True).float()
            # 避免除以 0
            seq_lengths = torch.clamp(seq_lengths, min=1.0)
            # 求平均
            mean_pooled = masked_hidden.sum(dim=1) / seq_lengths
        else:
            # 如果没有 attention_mask，直接对所有 token 求平均
            mean_pooled = last_hidden.mean(dim=1)
        mean_pooled_chunks.append(mean_pooled)
        
        # 5. 最大池化：对所有非 padding token 求最大
        if attention_mask is not None:
            attention_mask_cpu = attention_mask.detach().cpu()
            mask_expanded = attention_mask_cpu.unsqueeze(-1).expand_as(last_hidden)
            # 将 padding 位置设为很小的值，这样 max 操作会忽略它们
            masked_hidden = last_hidden.clone()
            masked_hidden[~mask_expanded.bool()] = float('-inf')
            max_pooled = masked_hidden.max(dim=1)[0]
            # 如果某个序列全是 padding，则设为 0
            all_padding = (attention_mask_cpu.sum(dim=1) == 0)
            if all_padding.any():
                max_pooled[all_padding] = 0.0
        else:
            max_pooled = last_hidden.max(dim=1)[0]
        max_pooled_chunks.append(max_pooled)
        
        # 6. 最后一个非 padding token 的表示
        if attention_mask is not None:
            attention_mask_cpu = attention_mask.detach().cpu()
            # 找到每个序列的最后一个非 padding token 的位置
            seq_lengths = attention_mask_cpu.sum(dim=1) - 1  # -1 因为索引从 0 开始
            seq_lengths = torch.clamp(seq_lengths, min=0)  # 确保至少是 0
            batch_indices = torch.arange(last_hidden.size(0), device=last_hidden.device)
            last_token = last_hidden[batch_indices, seq_lengths]
        else:
            # 如果没有 attention_mask，使用最后一个 token
            last_token = last_hidden[:, -1, :]
        last_token_chunks.append(last_token)

        if "labels" in batch_on_device:
            label_chunks.append(batch_on_device["labels"].detach().cpu())

    # 合并所有 batch
    results = {
        "all_tokens": torch.cat(all_tokens_chunks, dim=0) if all_tokens_chunks else None,
        "cls_token": torch.cat(cls_token_chunks, dim=0) if cls_token_chunks else None,
        "pooler_output": torch.cat(pooler_output_chunks, dim=0) if pooler_output_chunks else None,
        "mean_pooled": torch.cat(mean_pooled_chunks, dim=0) if mean_pooled_chunks else None,
        "max_pooled": torch.cat(max_pooled_chunks, dim=0) if max_pooled_chunks else None,
        "last_token": torch.cat(last_token_chunks, dim=0) if last_token_chunks else None,
        "labels": torch.cat(label_chunks, dim=0) if label_chunks else None,
    }
    
    # 打印维度信息
    print("\n" + "="*60)
    print("📊 提取的序列表示维度信息：")
    print("="*60)
    for key, value in results.items():
        if value is not None:
            print(f"  {key:20s}: {tuple(value.shape)}")
        else:
            print(f"  {key:20s}: None (不可用)")
    print("="*60 + "\n")
    
    return results


def ensure_output_dir(output_path: str) -> None:
    output_dir = Path(output_path).parent
    if output_dir and not output_dir.exists():
        output_dir.mkdir(parents=True, exist_ok=True)


def compute_clustering_metrics(
    embeddings_2d: np.ndarray,
    true_labels: np.ndarray,
) -> Tuple[Dict[str, float], np.ndarray]:
    """计算 tSNE 降维后的聚类指标，评估与真实标签的关系。"""
    metrics = {}
    
    # 1. 使用 KMeans 进行聚类（聚类数 = 真实标签数）
    n_clusters = len(np.unique(true_labels))
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(embeddings_2d)
    
    # 2. 聚类评估指标（比较聚类结果与真实标签）
    metrics["ARI"] = adjusted_rand_score(true_labels, cluster_labels)
    metrics["NMI"] = normalized_mutual_info_score(true_labels, cluster_labels)
    metrics["Homogeneity"] = homogeneity_score(true_labels, cluster_labels)
    metrics["Completeness"] = completeness_score(true_labels, cluster_labels)
    metrics["V-measure"] = v_measure_score(true_labels, cluster_labels)
    
    # 3. Silhouette Score（评估聚类质量）
    metrics["Silhouette"] = silhouette_score(embeddings_2d, true_labels)
    
    # 4. 在 tSNE 空间上的分类性能（使用 KNN）
    # 将数据分为训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(
        embeddings_2d, true_labels, test_size=0.2, random_state=42, stratify=true_labels
    )
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    metrics["KNN_Accuracy"] = accuracy_score(y_test, y_pred)
    
    return metrics, cluster_labels


def plot_tsne(
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    output_path: str,
    perplexity: float,
    learning_rate: float,
) -> Dict[str, float]:
    ensure_output_dir(output_path)

    embeddings_np = embeddings.numpy()
    labels_np = labels.numpy()

    print("🔍 正在执行 tSNE 降维...")
    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        learning_rate=learning_rate,
        init="random",
        random_state=42,
    )
    reduced = tsne.fit_transform(embeddings_np)

    # 计算聚类指标
    print("📊 正在计算聚类评估指标...")
    metrics, cluster_labels = compute_clustering_metrics(reduced, labels_np)
    
    # 打印指标
    print("\n" + "="*60)
    print("📈 tSNE 聚类与真实标签的关系指标：")
    print("="*60)
    print(f"  Adjusted Rand Index (ARI):        {metrics['ARI']:.4f}  [范围: -1 到 1, 1 表示完全一致]")
    print(f"  Normalized Mutual Info (NMI):     {metrics['NMI']:.4f}  [范围: 0 到 1, 1 表示完全一致]")
    print(f"  Homogeneity:                       {metrics['Homogeneity']:.4f}  [同质性，越高越好]")
    print(f"  Completeness:                     {metrics['Completeness']:.4f}  [完整性，越高越好]")
    print(f"  V-measure:                         {metrics['V-measure']:.4f}  [同质性和完整性的调和平均]")
    print(f"  Silhouette Score:                  {metrics['Silhouette']:.4f}  [范围: -1 到 1, 越高越好]")
    print(f"  KNN 分类准确率 (tSNE 空间):       {metrics['KNN_Accuracy']:.4f}  [在降维空间上的分类性能]")
    print("="*60 + "\n")

    # 绘制可视化图
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # 左图：按真实标签着色
    ax1 = axes[0]
    scatter1 = ax1.scatter(
        reduced[:, 0],
        reduced[:, 1],
        c=labels_np,
        cmap="viridis",
        alpha=0.7,
        s=10,
    )
    ax1.set_title("tSNE: 按真实标签着色", fontsize=14, fontweight="bold")
    ax1.set_xlabel("Dimension 1")
    ax1.set_ylabel("Dimension 2")
    plt.colorbar(scatter1, ax=ax1, ticks=sorted(set(labels_np)), label="真实标签")
    
    # 右图：按 KMeans 聚类结果着色
    ax2 = axes[1]
    scatter2 = ax2.scatter(
        reduced[:, 0],
        reduced[:, 1],
        c=cluster_labels,
        cmap="tab10",
        alpha=0.7,
        s=10,
    )
    ax2.set_title("tSNE: 按 KMeans 聚类结果着色", fontsize=14, fontweight="bold")
    ax2.set_xlabel("Dimension 1")
    ax2.set_ylabel("Dimension 2")
    plt.colorbar(scatter2, ax=ax2, ticks=sorted(set(cluster_labels)), label="聚类标签")
    
    # 在图上添加指标文本
    metrics_text = (
        f"ARI: {metrics['ARI']:.3f} | NMI: {metrics['NMI']:.3f}\n"
        f"V-measure: {metrics['V-measure']:.3f} | Silhouette: {metrics['Silhouette']:.3f}\n"
        f"KNN Accuracy: {metrics['KNN_Accuracy']:.3f}"
    )
    fig.suptitle("tSNE 降维可视化与聚类评估", fontsize=16, fontweight="bold", y=1.02)
    fig.text(0.5, 0.02, metrics_text, ha="center", fontsize=10, 
             bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"🖼️ tSNE 可视化已保存到: {output_path}")
    
    return metrics


def main() -> None:
    args = parse_args()

    # 显示系统信息（仅在指定了 CUDA 设备时）
    if args.device != "auto" and "cuda" in args.device.lower():
        print("🔍 系统 CUDA 信息：")
        print(f"   PyTorch 版本：{torch.__version__}")
        print(f"   CUDA 可用：{torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"   GPU 数量：{torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                print(f"   GPU {i}: {torch.cuda.get_device_name(i)}")
        else:
            if hasattr(torch.version, 'cuda'):
                print(f"   PyTorch 编译的 CUDA 版本：{torch.version.cuda}")
        print()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        try:
            device = torch.device(args.device)
        except (TypeError, RuntimeError) as exc:
            raise ValueError(f"无法解析设备标识 '{args.device}'") from exc

        if device.type == "cuda":
            if not torch.cuda.is_available():
                print("⚠️ 警告：PyTorch 未检测到 CUDA 支持。")
                print(f"   当前 PyTorch 版本：{torch.__version__}")
                print(f"   CUDA 可用性：{torch.cuda.is_available()}")
                if hasattr(torch.version, 'cuda'):
                    print(f"   PyTorch 编译的 CUDA 版本：{torch.version.cuda}")
                print("   将回退到 CPU 模式。")
                device = torch.device("cpu")
            elif device.index is not None and device.index >= torch.cuda.device_count():
                print(f"⚠️ 警告：指定的 GPU 索引 {device.index} 超出范围。")
                print(f"   当前共有 {torch.cuda.device_count()} 张 GPU（索引 0-{torch.cuda.device_count()-1}）。")
                print(f"   将使用 GPU 0 代替。")
                device = torch.device("cuda:0")

    print(f"🚀 使用设备: {device}")

    tokenizer = OmniTokenizer.from_pretrained(args.model_path)
    datasets = prepare_datasets(args.data_dir, tokenizer, args.max_length)

    model = OmniModelForSequenceClassification(
        args.model_path,
        tokenizer,
        num_labels=len(LABEL2ID),
    ).to(device)
    model.eval()

    splits: Iterable[str]
    if args.split == "all":
        splits = datasets.keys()
    else:
        splits = [args.split]

    results = {}
    for split in splits:
        if split not in datasets:
            raise ValueError(f"数据划分 '{split}' 不存在，可选值：{list(datasets.keys())}")

        dataloader = DataLoader(
            datasets[split],
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
        )

        split_results = forward_hidden_states(model, dataloader, device)
        results[split] = split_results
        print(f"✅ Split '{split}' 提取完成")

    ensure_output_dir(args.output)
    torch.save(
        {
            "model_path": args.model_path,
            "data_dir": args.data_dir,
            "max_length": args.max_length,
            "results": results,
        },
        args.output,
    )
    print(f"💾 Hidden states 已保存到: {args.output}")

    if args.tsne:
        concat_embeddings: List[torch.Tensor] = []
        concat_labels: List[torch.Tensor] = []

        for split, data in results.items():
            if data["labels"] is None:
                print(f"⚠️ Split '{split}' 缺失标签，跳过 tSNE 合并。")
                continue
            
            # 根据用户选择的表示方式提取 embeddings
            representation = data.get(args.representation)
            if representation is None:
                available = [k for k, v in data.items() if v is not None and k != "labels"]
                raise ValueError(
                    f"表示方式 '{args.representation}' 不可用。"
                    f"可用的表示方式：{available}"
                )
            
            # 如果选择了 all_tokens，需要先进行池化（使用平均池化）
            if args.representation == "all_tokens" and len(representation.shape) == 3:
                print(f"⚠️ 'all_tokens' 是 3D 张量，tSNE 需要 2D 输入。自动使用平均池化。")
                # 对每个序列的所有 token 求平均
                representation = representation.mean(dim=1)  # [batch_size, hidden_size]
            
            concat_embeddings.append(representation)
            concat_labels.append(data["labels"])

        if not concat_embeddings:
            raise ValueError("未找到带标签的数据，无法执行 tSNE。")

        all_embeddings = torch.cat(concat_embeddings, dim=0)
        all_labels = torch.cat(concat_labels, dim=0)
        print(f"📌 tSNE 使用的样本总数：{all_embeddings.shape[0]}")
        print(f"📌 使用的表示方式：{args.representation}，维度：{tuple(all_embeddings.shape)}")

        metrics = plot_tsne(
            embeddings=all_embeddings,
            labels=all_labels,
            output_path=args.tsne_output,
            perplexity=args.tsne_perplexity,
            learning_rate=args.tsne_learning_rate,
        )
        
        # 保存指标到文件
        metrics_output_path = args.tsne_output.replace(".png", "_metrics.txt")
        with open(metrics_output_path, "w", encoding="utf-8") as f:
            f.write("tSNE 聚类与真实标签的关系指标\n")
            f.write("="*60 + "\n\n")
            for key, value in metrics.items():
                f.write(f"{key}: {value:.6f}\n")
        print(f"💾 指标已保存到: {metrics_output_path}")


if __name__ == "__main__":
    main()
