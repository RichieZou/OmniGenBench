"""从已微调的 TE 模型中批量提取训练集 CLS 向量。"""

import argparse
import os
from typing import Dict, List

import numpy as np
import torch
from torch.utils.data import DataLoader

from omnigenbench import (
    OmniTokenizer,
    OmniDatasetForMultiLabelClassification,
    OmniModelForMultiLabelSequenceClassification,
    OmniPooling,
)


class TriClassTEDataset(OmniDatasetForMultiLabelClassification):
    """与 `triclass_te.py` 中保持一致的 3 分类数据集实现。"""

    def prepare_input(self, instance, **kwargs):
        label2idx = {"Low": 0, "Medium": 1, "High": 2, "nan": -100}

        labels = torch.tensor(
            [
                label2idx[str(instance["root_TE_label"])],
                label2idx[str(instance["seedling_TE_label"])],
                label2idx[str(instance["leaf_TE_label"])],
                label2idx[str(instance["FMI_TE_label"])],
                label2idx[str(instance["FOD_TE_label"])],
                label2idx[str(instance["Prophase-I-pollen_TE_label"])],
                label2idx[str(instance["Tricellular-pollen_TE_label"])],
                label2idx[str(instance["flag_TE_label"])],
                label2idx[str(instance["grain_TE_label"])],
            ],
            dtype=torch.long,
        )

        tokenized_inputs = self.tokenizer(
            instance["sequence"],
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        tokenized_inputs["labels"] = labels
        return tokenized_inputs


class OmniModelForTriClassTESequenceClassification(OmniModelForMultiLabelSequenceClassification):
    """复用训练脚本中的三分类模型定义，确保本地加载。"""

    def __init__(self, config_or_model, tokenizer, num_labels=9, num_classes=3, *args, **kwargs):
        dataset_class = kwargs.pop("dataset_class", TriClassTEDataset)
        super().__init__(config_or_model, tokenizer, num_labels=num_labels * num_classes, *args, **kwargs)
        self.metadata["model_name"] = self.__class__.__name__
        self.num_labels = num_labels
        self.num_classes = num_classes
        self.pooler = OmniPooling(self.config)
        self.classifier = torch.nn.Linear(self.config.hidden_size, self.num_classes * self.num_labels)
        self.loss_fn = torch.nn.CrossEntropyLoss(ignore_index=-100, reduction="mean")
        self.dataset_class = dataset_class

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
            "last_hidden_state": outputs.last_hidden_state if hasattr(outputs, "last_hidden_state") else None,
        }


def _move_to_device(batch: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    inputs = {}
    for key, value in batch.items():
        if key == "labels":
            continue
        if isinstance(value, torch.Tensor):
            inputs[key] = value.to(device)
    return inputs


def _load_local_datasets(
    data_dir: str,
    tokenizer: OmniTokenizer,
    max_length: int,
    force_padding: bool,
) -> Dict[str, TriClassTEDataset]:
    datasets: Dict[str, TriClassTEDataset] = {}
    for split in ["train", "valid", "test"]:
        file_path = os.path.join(data_dir, f"{split}.csv")
        if not os.path.isfile(file_path):
            continue
        datasets[split] = TriClassTEDataset(
            data_source=file_path,
            tokenizer=tokenizer,
            max_length=max_length,
            force_padding=force_padding,
        )
    if "train" not in datasets:
        raise FileNotFoundError(f"未找到训练集文件: {os.path.join(data_dir, 'train.csv')}")
    return datasets


def extract_cls_embeddings(
    data_dir: str,
    model_dir: str,
    tokenizer_path: str,
    output_path: str,
    batch_size: int = 32,
    max_length: int = 512,
    force_padding: bool = False,
):
    tokenizer = OmniTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)

    datasets = _load_local_datasets(
        data_dir=data_dir,
        tokenizer=tokenizer,
        max_length=max_length,
        force_padding=force_padding,
    )
    train_dataset = datasets["train"]

    dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=train_dataset.collate_fn,
    )

    model = OmniModelForTriClassTESequenceClassification(
        model_dir,
        tokenizer,
        num_labels=9,
        num_classes=3,
        trust_remote_code=True,
    )
    if torch.cuda.is_available():
        device = torch.device("cuda")
        model = model.to(device)
    else:
        device = torch.device("cpu")
        model = model.to(device)
    model.eval()

    cls_embeddings: List[torch.Tensor] = []

    with torch.no_grad():
        for batch in dataloader:
            inputs = _move_to_device(batch, device)
            outputs = model(**inputs)
            last_hidden_state = outputs.get("last_hidden_state")
            if last_hidden_state is None:
                raise RuntimeError("模型未返回 last_hidden_state，无法提取 CLS 向量。")

            cls_embeddings.append(last_hidden_state[:, 0, :].cpu())

    all_embeddings = torch.cat(cls_embeddings, dim=0).numpy()
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    np.save(output_path, all_embeddings)

    print(f"✅ CLS 向量已保存到: {output_path}")
    print(f"   形状: {all_embeddings.shape}")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        default="/home/yingjie/OmniGenBench/examples/heng_orig",
        help="包含 train.csv/valid.csv/test.csv 的目录",
    )
    parser.add_argument(
        "--model-dir",
        default="/home/yingjie/OmniGenBench/examples/heng_orig/ogb_te_3class_finetuned_epoch_19_seed_42_accuracy_score_0.9900_seed_42_f1_score_0.9900",
        help="已微调模型的本地目录 (包含 config 与权重)",
    )
    parser.add_argument(
        "--tokenizer-path",
        default="/home/yingjie/OmniGenBench/models_cache/OmniGenome-52M",
        help="Tokenizer 所在的模型名称或路径",
    )
    parser.add_argument(
        "--output",
        default="/home/yingjie/OmniGenBench/examples/heng_orig/train_cls_embeddings.npy",
        help="CLS 向量输出文件 (.npy)",
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument(
        "--force-padding",
        action="store_true",
        help="强制对输入进行 padding（与训练脚本默认一致）",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    extract_cls_embeddings(
        data_dir=args.data_dir,
        model_dir=args.model_dir,
        tokenizer_path=args.tokenizer_path,
        output_path=args.output,
        batch_size=args.batch_size,
        max_length=args.max_length,
        force_padding=args.force_padding,
    )


if __name__ == "__main__":
    main()