from transformers import ViTImageProcessor
import torch
from torchvision import transforms
from typing import Tuple, Any
from datasets import Dataset  # 补充Dataset类型提示（需确保安装datasets库）

# 初始化特征提取器
local_vit_path = "./vit_pretrained/vit-base-patch16-224-in21k"
feature_extractor = ViTImageProcessor.from_pretrained(local_vit_path)

# --- 训练集预处理（含数据增强）---
train_transforms = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=10),  # 小幅旋转，减少数据分布偏移
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),  # 随机裁剪+缩放
    transforms.ColorJitter(brightness=0.2, contrast=0.2),  # 亮度/对比度增强
    transforms.ToTensor(),
    transforms.Normalize(mean=feature_extractor.image_mean, std=feature_extractor.image_std),
    transforms.RandomErasing(p=0.2, scale=(0.02, 0.1), ratio=(0.3, 3.3))  # 随机擦除
])

# --- 验证集预处理（仅基础变换，无增强）---
val_transforms = transforms.Compose([
    transforms.Resize((224, 224)),  # 固定尺寸缩放
    transforms.ToTensor(),
    transforms.Normalize(mean=feature_extractor.image_mean, std=feature_extractor.image_std)
])

# --- 测试集预处理（与验证集一致，严格无增强）---
test_transforms = transforms.Compose([
    transforms.Resize((224, 224)),  # 与验证集保持一致的尺寸策略
    transforms.ToTensor(),
    transforms.Normalize(mean=feature_extractor.image_mean, std=feature_extractor.image_std)
])


def preprocess_train(examples: dict) -> dict:
    """训练集预处理函数"""
    examples['pixel_values'] = [train_transforms(img.convert("RGB")) for img in examples['image']]
    return examples


def preprocess_val(examples: dict) -> dict:
    """验证集预处理函数"""
    examples['pixel_values'] = [val_transforms(img.convert("RGB")) for img in examples['image']]
    return examples


def preprocess_test(examples: dict) -> dict:
    """测试集预处理函数"""
    examples['pixel_values'] = [test_transforms(img.convert("RGB")) for img in examples['image']]
    return examples


def get_preprocessed_datasets(
        train_ds: Dataset,
        val_ds: Dataset,
        test_ds: Dataset
) -> Tuple[Dataset, Dataset, Dataset]:
    """
    批量预处理训练/验证/测试数据集

    Args:
        train_ds: 原始训练集
        val_ds: 原始验证集
        test_ds: 原始测试集

    Returns:
        预处理后的训练集、验证集、测试集（均为torch格式）
    """
    # 预处理训练集（批量处理，移除原始image列）
    preprocessed_train_ds = train_ds.map(
        preprocess_train,
        batched=True,
        remove_columns=['image']
    )
    # 预处理验证集
    preprocessed_val_ds = val_ds.map(
        preprocess_val,
        batched=True,
        remove_columns=['image']
    )
    # 预处理测试集
    preprocessed_test_ds = test_ds.map(
        preprocess_test,
        batched=True,
        remove_columns=['image']
    )

    # 设置数据集格式为PyTorch张量（仅保留模型所需列）
    preprocessed_train_ds.set_format(type='torch', columns=['pixel_values', 'label'])
    preprocessed_val_ds.set_format(type='torch', columns=['pixel_values', 'label'])
    preprocessed_test_ds.set_format(type='torch', columns=['pixel_values', 'label'])

    return preprocessed_train_ds, preprocessed_val_ds, preprocessed_test_ds