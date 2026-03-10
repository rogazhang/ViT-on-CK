import os
import pandas as pd
from datasets import Dataset, Features, Image, ClassLabel

# 配置参数
DATASET_PATH = r"C:\Users\49860\ViT on CK+\CK+48"
string_labels = ['anger', 'contempt', 'disgust', 'fear', 'happy', 'sadness', 'surprise']


def load_all_data(root_dir):
    """加载所有图像路径和标签，返回DataFrame"""
    data_list = []
    for emotion in string_labels:
        emotion_dir = os.path.join(root_dir, emotion)
        for img_name in os.listdir(emotion_dir):
            if img_name.lower().endswith(('.png')):
                data_list.append({
                    "image_path": os.path.join(emotion_dir, img_name),
                    "label": emotion
                })
    return pd.DataFrame(data_list)


def create_hf_dataset(root_dir=DATASET_PATH):
    """创建Hugging Face Dataset对象并划分训练/测试集"""
    # 读取数据
    df = load_all_data(root_dir)

    # 定义特征格式
    features = Features({
        "image": Image(),
        "label": ClassLabel(names=string_labels)
    })

    # 创建总数据集
    full_ds = Dataset.from_dict({
        "image": df["image_path"].tolist(),
        "label": df["label"].tolist()
    }, features=features)

    # 划分训练集和测试集
    ds_split = full_ds.train_test_split(test_size=0.2, seed=42)
    train_ds = ds_split['train']
    test_ds = ds_split['test']

    print(f"训练集数量: {len(train_ds)}, 测试集数量: {len(test_ds)}")
    return train_ds, test_ds


if __name__ == "__main__":
    # 测试数据加载
    train_ds, test_ds = create_hf_dataset()
    print("数据加载测试成功")