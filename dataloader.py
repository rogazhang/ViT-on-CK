import os
import numpy as np
import pandas as pd
import torch
from datasets import Dataset, Features, Image, ClassLabel, DatasetDict

# 配置参数
DATASET_PATH = r"C:\Users\49860\ViT on CK+\CK+48"
string_labels = ['anger', 'contempt', 'disgust', 'fear', 'happy', 'sadness', 'surprise']


def load_all_data(root_dir):
    """加载所有图像路径和标签，返回DataFrame"""
    data_list = []
    # 遍历情感文件夹
    for emotion in string_labels:
        emotion_dir = os.path.join(root_dir, emotion)
        if not os.path.exists(emotion_dir):
            continue

        for img_name in os.listdir(emotion_dir):
            if img_name.lower().endswith('.png'):
                img_path = os.path.join(emotion_dir, img_name)
                # 提取受试者ID: 假设格式为 S001_001_00000001.png -> 提取出 S001
                subject_id = img_name.split('_')[0]

                data_list.append({
                    "image": img_path,
                    "label": emotion,
                    "subject": subject_id
                })
    return pd.DataFrame(data_list)


def create_hf_dataset(root_dir=DATASET_PATH):
    """根据受试者ID (Subject ID) 划分训练/验证/测试集（8:1:1），并创建HF Dataset"""
    # 1. 加载数据到 DataFrame
    df = load_all_data(root_dir)

    # 2. 提取唯一的受试者列表并打乱
    subjects = sorted(df['subject'].unique())
    np.random.seed(42)
    np.random.shuffle(subjects)

    # 3. 按受试者 8:1:1 比例划分
    total_subs = len(subjects)
    train_split = int(total_subs * 0.8)
    val_split = int(total_subs * 0.9)  # 80%训练 + 10%验证 = 90%，剩余10%测试
    train_subs = subjects[:train_split]
    val_subs = subjects[train_split:val_split]
    test_subs = subjects[val_split:]

    # 4. 根据受试者名单过滤样本
    train_df = df[df['subject'].isin(train_subs)].reset_index(drop=True)
    val_df = df[df['subject'].isin(val_subs)].reset_index(drop=True)
    test_df = df[df['subject'].isin(test_subs)].reset_index(drop=True)

    # 5. 定义 Hugging Face Dataset 的特征格式
    features = Features({
        "image": Image(),
        "label": ClassLabel(names=string_labels),
        "subject": ClassLabel(names=list(df['subject'].unique()))  # 可选：保留受试者标签
    })

    # 6. 构建 Dataset 对象
    # 注意：这里直接传入包含文件路径的列表，HF 的 Image() 特征会自动处理读取
    train_ds = Dataset.from_dict({
        "image": train_df["image"].tolist(),
        "label": train_df["label"].tolist(),
        "subject": train_df["subject"].tolist()
    }, features=features)

    val_ds = Dataset.from_dict({
        "image": val_df["image"].tolist(),
        "label": val_df["label"].tolist(),
        "subject": val_df["subject"].tolist()
    }, features=features)

    test_ds = Dataset.from_dict({
        "image": test_df["image"].tolist(),
        "label": test_df["label"].tolist(),
        "subject": test_df["subject"].tolist()
    }, features=features)

    print(f"总受试者人数: {len(subjects)}")
    print(f"训练集: {len(train_ds)} 张图片 (来自 {len(train_subs)} 人)")
    print(f"验证集: {len(val_ds)} 张图片 (来自 {len(val_subs)} 人)")
    print(f"测试集: {len(test_ds)} 张图片 (来自 {len(test_subs)} 人)")

    return train_ds, val_ds, test_ds


if __name__ == "__main__":
    # 测试数据加载
    train_ds, val_ds, test_ds = create_hf_dataset()
    # 检查第一条数据，验证图片是否能正常解析
    print("训练集样例:", train_ds[0])
    print("验证集样例:", val_ds[0])
    print("测试集样例:", test_ds[0])
    print("数据加载及受试者划分成功")