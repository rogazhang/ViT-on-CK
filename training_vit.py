import torch
from transformers import TrainingArguments, Trainer
from transformers import EarlyStoppingCallback
import numpy as np
from sklearn.metrics import accuracy_score, f1_score

from dataloader import create_hf_dataset
from preprocessing import get_preprocessed_datasets
from model.ViT import get_model

def get_training_args(output_dir="./training_local"):
    """获取训练参数配置（适配RTX 3050 Ti + 1000条数据集）"""
    args = TrainingArguments(
        output_dir=output_dir,
        evaluation_strategy="epoch",  # 小数据集按「轮次」评估，而非steps（避免频繁评估）
        save_strategy="epoch",  # 按轮次保存模型，减少显存占用
        save_total_limit=3,  # 只保留最近3个模型，避免磁盘占用
        learning_rate=5e-5,  # 小数据集适当提高学习率（2e-5→5e-5），加快收敛
        per_device_train_batch_size=8,  # 16→8（3050 Ti 4GB显存建议设4，6GB设8）
        per_device_eval_batch_size=16,  # 评估批次可稍大，不影响训练显存
        num_train_epochs=20,  # 10→20（小数据集需要更多轮次，但靠早停避免过拟合）
        weight_decay=0.01,  # 新增：L2正则化，防止小数据集过拟合
        warmup_ratio=0.1,  # 新增：学习率预热（10%轮次），稳定训练
        fp16=torch.cuda.is_available(),  # 保留混合精度（3050 Ti支持，节省显存）
        fp16_full_eval=False,  # 新增：评估时不使用全精度，进一步省显存
        gradient_accumulation_steps=1,  # 显存不足时设为2（批次8→等效16）
        dataloader_pin_memory=True,  # 新增：固定显存，提升数据加载速度
        dataloader_num_workers=2,  # 新增：CPU线程数（3050 Ti建议2-4，避免CPU瓶颈）
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,  # 新增：F1越高越好
        remove_unused_columns=False,
        logging_dir="./logs",
        logging_steps=50,  # 10→50（小数据集减少日志输出频率）
        report_to="none",
        seed=42,  # 新增：固定随机种子，结果可复现
    )
    return args


# 定义评估指标
def compute_metrics(eval_pred):
    """本地实现准确率和F1分数计算（无需evaluate库）"""
    # eval_pred包含预测值和真实标签：(predictions, labels)
    predictions, labels = eval_pred
    # 将预测概率转为类别（取最大值索引）
    predictions = np.argmax(predictions, axis=1)

    # 计算准确率
    accuracy = accuracy_score(labels, predictions)
    # 计算F1分数（适配多分类场景，设为"weighted"）
    f1 = f1_score(labels, predictions, average="weighted")

    # 返回指标字典（和原逻辑一致，保证Trainer能识别）
    return {
        "accuracy": accuracy,
        "f1": f1
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
    trainer.save_model("./vit_emotion_model")

    return trainer, pre_test_ds


if __name__ == "__main__":
    # 执行训练
    trainer, test_ds = train_model()
    print("训练完成！")