import torch
from transformers import TrainingArguments, Trainer, EarlyStoppingCallback
import numpy as np
import os
from sklearn.metrics import accuracy_score, f1_score, recall_score, classification_report

from dataloader import create_hf_dataset
from preprocessing_ori import get_preprocessed_datasets
# 这里保持引用你的新版 ViT.py，它会自动加载固定位置编码
from model.Fixed import get_model


def get_training_args():
    """优化后的训练参数（针对固定位置编码与表情识别任务）"""
    output_dir = "./training_fixed_result"
    os.makedirs(output_dir, exist_ok=True)

    return TrainingArguments(
        output_dir=output_dir,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        # 💡 因为位置编码被固定，模型参数减少，可以尝试略微调大一点学习率或增加温热启动
        learning_rate=5e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=16,
        num_train_epochs=20,  # 适当增加轮次，因为固定编码可能收敛曲线不同
        weight_decay=0.01,
        warmup_ratio=0.1,
        fp16=torch.cuda.is_available(),
        gradient_accumulation_steps=1,
        dataloader_pin_memory=True,
        dataloader_num_workers=2,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        logging_dir=f"{output_dir}/logs",
        logging_steps=10,  # 缩短日志步长，更细致观察损失
        report_to="tensorboard",
        seed=42,
        # 💡 重要：避免保存没用的位置编码梯度状态
        remove_unused_columns=False
    )


def compute_metrics(eval_pred):
    """评估指标函数（保持不变）"""
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
        "war": round(war, 4)
    }


def train_model():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"🚀 使用设备: {device}")

    print("📦 加载数据集...")
    train_ds, val_ds, test_ds = create_hf_dataset()

    print("🛠 预处理数据...")
    pre_train_ds, pre_val_ds, pre_test_ds = get_preprocessed_datasets(train_ds, val_ds, test_ds)

    print("🧠 初始化模型 (固定位置编码已激活)...")
    model = get_model(device)

    # 💡 打印模型参数量，确认位置编码是否已被冻结
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"📊 总参数量: {total_params}")
    print(f"📉 可训练参数量: {trainable_params} (已减少 {total_params - trainable_params} 个固定参数)")

    training_args = get_training_args()

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=pre_train_ds,
        eval_dataset=pre_val_ds,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=5)]  # 容忍度稍微调高
    )

    print("🔥 开始训练...")
    trainer.train()

    print("💾 保存模型...")
    trainer.save_model("./training_fixed_result/vit_fixed_pos_model")

    # --- 新增：最后增加的评估代码 ---
    print("\n" + "=" * 30)
    print("测试集最终结果 (Final Test Metrics):")

    # 使用最佳模型在测试集上运行推理
    test_results = trainer.predict(pre_test_ds)

    # 直接提取并打印那四个指标
    metrics = test_results.metrics
    print(f"Accuracy: {metrics.get('test_accuracy')}")
    print(f"F1 Score: {metrics.get('test_f1')}")
    print(f"UAR:      {metrics.get('test_uar')}")
    print(f"WAR:      {metrics.get('test_war')}")
    print("=" * 30)

    return trainer


if __name__ == "__main__":
    trainer, test_ds = train_model()
    print("\n✅ 训练完成！建议通过 TensorBoard 查看固定位置编码对收敛的影响。")