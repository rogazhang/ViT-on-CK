import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from training_vit import train_model
from dataloader import string_labels

def evaluate_model(trainer, test_ds):
    """仅使用测试集评估模型最终性能并可视化结果"""
    # 1. 执行测试集预测（训练阶段完全未接触测试集）
    print("执行测试集预测（最终性能检验）...")
    outputs = trainer.predict(test_ds)
    print("测试集评估指标：")
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
    plt.title("表情分类-测试集混淆矩阵")
    plt.tight_layout()
    plt.savefig("./test_confusion_matrix.png")  # 保存到本地
    plt.show()

    return outputs.metrics


if __name__ == "__main__":
    # 先训练模型（训练+验证），再用测试集做最终评估
    trainer, test_ds = train_model()
    metrics = evaluate_model(trainer, test_ds)
    print(f"\n=== 测试集最终评估结果 ===")
    print(f"准确率: {metrics['test_accuracy']:.4f}")
    print(f"F1分数: {metrics['test_f1']:.4f}")
    print(f"UAR: {metrics['test_uar']:.4f}")
    print(f"WAR: {metrics['test_war']:.4f}")