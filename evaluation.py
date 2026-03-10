import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from training_vit import train_model
from dataloader import string_labels

def evaluate_model(trainer, test_ds):
    """评估模型性能并可视化结果"""
    # 1. 执行预测
    print("执行模型预测...")
    outputs = trainer.predict(test_ds)
    print("预测完成，评估指标：")
    print(outputs.metrics)

    # 2. 解析预测结果
    predictions = np.argmax(outputs.predictions, axis=1)
    labels = outputs.label_ids

    # 3. 绘制混淆矩阵
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 解决中文显示问题（本地运行）
    plt.rcParams['axes.unicode_minus'] = False

    cm = confusion_matrix(labels, predictions)
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=string_labels,
        yticklabels=string_labels
    )
    plt.xlabel("预测标签")
    plt.ylabel("真实标签")
    plt.title("表情分类混淆矩阵")
    plt.tight_layout()
    plt.savefig("./confusion_matrix.png")  # 保存到本地
    plt.show()

    return outputs.metrics


if __name__ == "__main__":
    # 先训练模型，再评估
    trainer, test_ds = train_model()
    metrics = evaluate_model(trainer, test_ds)
    print(f"\n最终评估结果：")
    print(f"准确率: {metrics['test_accuracy']:.4f}")
    print(f"F1分数: {metrics['test_f1']:.4f}")