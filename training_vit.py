import torch
from transformers import TrainingArguments, Trainer
from transformers import EarlyStoppingCallback
import numpy as np
import os
from sklearn.metrics import accuracy_score, f1_score, recall_score

from dataloader import create_hf_dataset
from preprocessing import get_preprocessed_datasets
from model.ViT import get_model

def get_training_args():
    """获取训练参数配置（适配RTX 3050 Ti + 1000条数据集）"""
    output_dir = "./training_vit_result"
    os.makedirs(output_dir, exist_ok=True)

    args = TrainingArguments(
        output_dir=output_dir,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        learning_rate=5e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=16,
        num_train_epochs=20,
        weight_decay=0.01,  # 新增：L2正则化，防止小数据集过拟合
        warmup_ratio=0.1,  # 新增：学习率预热（10%轮次），稳定训练
        fp16=torch.cuda.is_available(),  # 保留混合精度（3050 Ti支持，节省显存）
        gradient_accumulation_steps=1,  # 显存不足时设为2（批次8→等效16）
        dataloader_pin_memory=True,  # 新增：固定显存，提升数据加载速度
        dataloader_num_workers=2,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        logging_dir=f"{output_dir}/logs",
        logging_steps=50,
        report_to="tensorboard",
        seed=42,
    )
    return args

def compute_metrics(eval_pred):
    """包含accuracy/f1/uar/war的完整评估函数"""
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)  # 概率→类别

    # 1. 准确率
    accuracy = accuracy_score(labels, predictions)

    # 2. 加权F1分数（多分类默认weighted）
    f1 = f1_score(labels, predictions, average="weighted")

    # 3. UAR（未加权召回率=macro recall）
    uar = recall_score(labels, predictions, average="macro")

    # 4. WAR（加权召回率=weighted recall）
    war = recall_score(labels, predictions, average="weighted")

    return {
        "accuracy": round(accuracy, 4),
        "f1": round(f1, 4),
        "uar": round(uar, 4),
        "war": round(war, 4)
    }

def train_model():
    """完整训练流程"""
    # 1. 设备配置
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    # 2. 加载数据
    print("加载数据集...")
    train_ds, test_ds = create_hf_dataset()

    # 3. 数据预处理
    print("预处理数据...")
    pre_train_ds, pre_test_ds = get_preprocessed_datasets(train_ds, test_ds)

    # 4. 初始化模型
    print("初始化模型...")
    model = get_model(device)

    # 5. 配置训练参数
    training_args = get_training_args()

    # 6. 创建Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=pre_train_ds,
        eval_dataset=pre_test_ds,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
    )

    # 7. 开始训练
    print("开始训练...")
    trainer.train()

    # 8. 保存模型（本地路径）
    print("保存模型...")
    # 保存HuggingFace原生格式
    trainer.save_model("./training_vit_result/vit_emotion_model")

    return trainer, pre_test_ds


if __name__ == "__main__":
    # 执行训练
    trainer, test_ds = train_model()
    print("训练完成！")

    # 新增：TensorBoard可视化指引（对齐你的路径）
    print("\n=== TensorBoard可视化指引 ===")
    print(f"1. 终端执行：tensorboard --logdir=./training_vit_result/logs")
    print("2. 浏览器访问：http://localhost:6006")
    print("3. 可查看：损失曲线、准确率/F1/UAR/WAR等指标")