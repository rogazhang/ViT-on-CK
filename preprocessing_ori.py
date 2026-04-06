from transformers import ViTImageProcessor

# 初始化ViT特征提取器（全局单例，避免重复加载）
local_vit_path = "./vit_pretrained/vit-base-patch16-224-in21k"
feature_extractor = ViTImageProcessor.from_pretrained(local_vit_path)


def preprocess_images(examples):
    """
    图像预处理函数：
    1. 将图像转为RGB格式
    2. 使用ViT特征提取器处理为模型输入格式
    """
    # 将原始图像转为RGB格式
    rgb_images = [img.convert("RGB") for img in examples['image']]

    # 特征提取器处理：尺寸调整、归一化、生成张量
    inputs = feature_extractor(
        images=rgb_images,
        return_tensors="pt",  # 返回PyTorch张量
        padding=True,  # 补边
        truncation=True  # 截断
    )

    return {"pixel_values": inputs['pixel_values']}


def get_preprocessed_datasets(train_ds, val_ds, test_ds):
    """对训练/验证/测试集执行预处理，并设置张量格式"""
    # 映射预处理函数，移除原始image列节省内存
    preprocessed_train_ds = train_ds.map(
        preprocess_images,
        batched=True,
        remove_columns=['image']
    )
    # 修正：验证集用val_ds而不是test_ds
    preprocessed_val_ds = val_ds.map(
        preprocess_images,
        batched=True,
        remove_columns=['image']
    )
    preprocessed_test_ds = test_ds.map(
        preprocess_images,
        batched=True,
        remove_columns=['image']
    )

    # 设置为PyTorch张量格式
    preprocessed_train_ds.set_format(type='torch', columns=['pixel_values', 'label'])
    preprocessed_val_ds.set_format(type='torch', columns=['pixel_values', 'label'])
    preprocessed_test_ds.set_format(type='torch', columns=['pixel_values', 'label'])

    return preprocessed_train_ds, preprocessed_val_ds, preprocessed_test_ds


if __name__ == "__main__":
    # 测试预处理（需先运行dataloader）
    from dataloader import create_hf_dataset

    train_ds, val_ds, test_ds = create_hf_dataset()  # 修正：匹配3个返回值
    pre_train, pre_val, pre_test = get_preprocessed_datasets(train_ds, val_ds, test_ds)  # 传3个参数

    print(f"预处理后训练集第1个样本形状: {pre_train[0]['pixel_values'].shape}")
    print(f"预处理后验证集第1个样本形状: {pre_val[0]['pixel_values'].shape}")
    print(f"预处理后测试集第1个样本形状: {pre_test[0]['pixel_values'].shape}")
    print("预处理测试成功")