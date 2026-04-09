import torch
import torch.nn as nn
from transformers import TrainingArguments, Trainer, EarlyStoppingCallback, ViTConfig
import numpy as np
import os
from sklearn.metrics import accuracy_score, f1_score, recall_score

from dataloader import create_hf_dataset
from preprocessing import get_preprocessed_datasets
from model.Fixed import ViTForImageClassification

def get_training_args():
    """优化后的训练参数（针对固定位置编码与表情识别任务）"""
    output_dir = "./training_fixed_scratch_result"
    os.makedirs(output_dir, exist_ok=True)

    return TrainingArguments(
        output_dir=output_dir,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        # 从零训练通常需要比微调稍大的学习率
        learning_rate=1e-4,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=16,
        # 从零训练收敛极慢，建议至少 100 轮
        num_train_epochs=100,
        weight_decay=0.05,
        warmup_ratio=0.1,
        fp16=torch.cuda.is_available(),
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        logging_dir=f"{output_dir}/logs",
        logging_steps=10,
        report_to="tensorboard",
        seed=42,
        remove_unused_columns=False
    )

def compute_metrics(eval_pred):
    """四个核心指标评估"""
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)

    return {
        "accuracy": round(accuracy_score(labels, predictions), 4),
        "f1": round(f1_score(labels, predictions, average="weighted"), 4),
        "uar": round(recall_score(labels, predictions, average="macro"), 4),
        "war": round(recall_score(labels, predictions, average="weighted"), 4)
    }


def train_model():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"🚀 [Mode: From Scratch + Kaiming] 使用设备: {device}")

    # 1. 加载数据
    print("📦 加载数据集...")
    train_ds, val_ds, test_ds = create_hf_dataset()
    pre_train_ds, pre_val_ds, pre_test_ds = get_preprocessed_datasets(train_ds, val_ds, test_ds)

    # 2. 初始化模型（不加载预训练权重）
    print("🧠 正在进行 Kaiming Normal 随机初始化...")
    # 直接实例化类（不通过 .from_pretrained），权重将按照 PyTorch 默认或自定义规则初始化
    model = ViTForImageClassification(num_labels=7)

    # --- 自定义 Kaiming Normal 初始化函数 ---
    def weights_init_kaiming(m):
        if isinstance(m, (nn.Linear, nn.Conv2d)):
            # 使用 kaiming_normal_ 维持前向传播方差，适用于 GELU/ReLU 激活
            nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0)

    # 应用初始化逻辑
    model.apply(weights_init_kaiming)

    # 打印参数确认：位置编码应仍为固定（Requires Grad = False）
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"📊 总参数量: {total_params}")
    print(f"📉 可训练参数量: {trainable_params}")

    model.to(device)

    # 3. 创建 Trainer
    trainer = Trainer(
        model=model,
        args=get_training_args(),
        train_dataset=pre_train_ds,
        eval_dataset=pre_val_ds,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=10)]
    )

    # 4. 开始训练
    print("🔥 开始从零训练...")
    trainer.train()

    # 5. 保存结果
    trainer.save_model("./training_fixed_scratch_result/vit_scratch_fixed_pos")

    # 6. 测试集评估
    print("\n" + "=" * 30)
    print("测试集最终结果 (Scratch + Fixed Pos + Kaiming):")
    test_results = trainer.predict(pre_test_ds)
    metrics = test_results.metrics
    print(f"Accuracy: {metrics.get('test_accuracy')}")
    print(f"F1 Score: {metrics.get('test_f1')}")
    print(f"UAR:      {metrics.get('test_uar')}")
    print(f"WAR:      {metrics.get('test_war')}")
    print("=" * 30)

    return trainer, pre_test_ds


if __name__ == "__main__":
    train_model()
    print("\n✅ 流程全部完成！")