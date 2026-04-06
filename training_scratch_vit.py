import torch
from transformers import TrainingArguments, Trainer
from transformers import EarlyStoppingCallback
import numpy as np
import os
from sklearn.metrics import accuracy_score, f1_score, recall_score
import torch.nn as nn

# 数据加载 + 预处理
from dataloader import create_hf_dataset
from preprocessing import get_preprocessed_datasets
from dataloader import string_labels

# ViT 模型
from transformers import ViTConfig, ViTForImageClassification


def get_model(device):
    """
    🔥 ViT 从 Scratch 训练 + 标准 Kaiming Normal 初始化（最稳定）
    严禁纯随机初始化！
    """
    # 1. 配置 ViT 结构
    config = ViTConfig(
        image_size=224,
        patch_size=16,
        num_channels=3,
        hidden_size=768,
        num_attention_heads=12,
        num_hidden_layers=12,
        intermediate_size=3072,
        num_labels=len(string_labels),  # CK+48：7 分类
        hidden_act="gelu",
    )

    # 2. 随机初始化模型
    model = ViTForImageClassification(config)

    # --------------------------
    # 🔥 核心：Kaiming Normal 初始化（必须）
    # --------------------------
    for name, param in model.named_parameters():
        if "weight" in name and len(param.shape) >= 2:
            nn.init.kaiming_normal_(param, mode="fan_in", nonlinearity="relu")
        elif "bias" in name:
            nn.init.constant_(param, 0.0)

    model = model.to(device)
    print("✅ 模型初始化完成：Kaiming Normal + From Scratch")
    return model


def get_training_args():
    """
    Scratch 训练专用超参（更稳定、更慢、更严谨）
    """
    output_dir = "./vit_scratch_result"
    os.makedirs(output_dir, exist_ok=True)

    args = TrainingArguments(
        output_dir=output_dir,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        learning_rate=1e-4,       # Scratch 必须用这个 lr
        per_device_train_batch_size=8,
        per_device_eval_batch_size=16,
        num_train_epochs=100,     # 随机初始化需要更多轮
        weight_decay=0.01,
        warmup_ratio=0.1,
        fp16=torch.cuda.is_available(),
        dataloader_pin_memory=True,
        dataloader_num_workers=2,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        logging_dir=f"{output_dir}/logs",
        logging_steps=20,
        report_to="tensorboard",
        seed=42,
    )
    return args


def compute_metrics(eval_pred):
    """多分类评估指标：accuracy / F1 / UAR / WAR"""
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)

    accuracy = accuracy_score(labels, predictions)
    f1 = f1_score(labels, predictions, average="weighted")
    uar = recall_score(labels, predictions, average="macro")
    war = recall_score(labels, predictions, average="weighted")

    return {
        "accuracy": round(accuracy, 4),
        "f1": round(f1, 4),
        "uar": round(uar, 4),
        "war": round(war, 4),
    }


def train_model():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    # 1. 加载 8:1:1 划分的数据集
    print("加载数据集...")
    train_ds, val_ds, test_ds = create_hf_dataset()

    # 2. 预处理
    print("数据预处理...")
    pre_train, pre_val, pre_test = get_preprocessed_datasets(train_ds, val_ds, test_ds)

    # 3. 模型
    model = get_model(device)

    # 4. 训练参数
    training_args = get_training_args()

    # 5. 训练器（训练集训练 + 验证集评估）
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=pre_train,
        eval_dataset=pre_val,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=5)],
    )

    # 6. 开始训练
    print("开始训练 (From Scratch + Kaiming 初始化)...")
    trainer.train()

    # 7. 保存最优模型
    trainer.save_model("./vit_scratch_result/vit_scratch_best_model")
    print("训练完成！最优模型已保存")

    return trainer, pre_test


if __name__ == "__main__":
    trainer, test_ds = train_model()
    print("\n训练全部完成！请运行 evaluation.py 查看测试集最终性能")