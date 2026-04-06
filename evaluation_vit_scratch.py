import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from training_scratch_vit import train_model
from dataloader import string_labels

def evaluate_model(trainer, test_ds):
    print("✅ 测试集最终评估...")
    outputs = trainer.predict(test_ds)
    print("测试集指标：", outputs.metrics)

    predictions = np.argmax(outputs.predictions, axis=1)
    labels = outputs.label_ids

    cm = confusion_matrix(labels, predictions)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=string_labels, yticklabels=string_labels)
    plt.xlabel("预测")
    plt.ylabel("真实")
    plt.title("测试集混淆矩阵")
    plt.tight_layout()
    plt.savefig("test_confusion_matrix.png")
    plt.show()

    return outputs.metrics

if __name__ == "__main__":
    trainer, test_ds = train_model()
    metrics = evaluate_model(trainer, test_ds)
    print(f"\n=== 最终测试集结果 ===")
    print(f"准确率: {metrics['test_accuracy']:.4f}")
    print(f"F1: {metrics['test_f1']:.4f}")
    print(f"UAR: {metrics['test_uar']:.4f}")