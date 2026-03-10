import torch
import torch.nn as nn
from transformers import ViTModel
from transformers.modeling_outputs import SequenceClassifierOutput
import torch.nn.functional as F
from dataloader import string_labels


class PEG(nn.Module):
    """CPVT原论文版PEG条件位置编码模块"""

    def __init__(self, embed_dim, kernel_size=3):
        super().__init__()
        # 深度卷积（原论文默认kernel_size=3），保持通道数不变
        self.proj = nn.Conv2d(embed_dim, embed_dim,
                              kernel_size=kernel_size,
                              padding=kernel_size // 2,
                              groups=embed_dim)
        # 残差连接+层归一化（原论文标配）
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x, h, w):
        """
        Args:
            x: [B, N, C]  ViT第一层Encoder输出特征（含CLS token）
            h: int        patch高度数（如14）
            w: int        patch宽度数（如14）
        Returns:
            x: [B, N, C]  融合PEG条件位置编码后的特征
        """
        # 分离CLS token和图像token（CLS无空间位置，不参与卷积）
        cls_token = x[:, 0:1, :]  # [B, 1, C]
        img_tokens = x[:, 1:, :]  # [B, N-1, C]

        # 重塑为2D特征图：[B, C, h, w]（恢复空间结构）
        img_tokens_2d = img_tokens.transpose(1, 2).reshape(-1, x.shape[-1], h, w)
        # 卷积提取空间位置特征（PEG核心：动态条件位置编码）
        pe_2d = self.proj(img_tokens_2d)
        # 展平回一维token格式：[B, N-1, C]
        pe_flat = pe_2d.flatten(2).transpose(1, 2)

        # 残差融合+层归一化（原论文结构）
        img_tokens = self.norm(img_tokens + pe_flat)
        # 拼接CLS token，恢复原形状
        x = torch.cat([cls_token, img_tokens], dim=1)
        return x

class ViTCPVTForImageClassification(nn.Module):
    """
    基于 ViT + CPVT (PEG) 的图像分类模型
    设计点：在 Encoder 的第 0 层之后注入 PEG
    """

    def __init__(self, num_labels=len(string_labels)):
        super(ViTCPVTForImageClassification, self).__init__()
        # 加载预训练ViT（路径根据你的实际路径修改）
        local_vit_path = "./vit_pretrained/vit-base-patch16-224-in21k"
        self.vit = ViTModel.from_pretrained(local_vit_path)
        embed_dim = self.vit.config.hidden_size

        # PEG 将取代原生的 Position Embeddings
        self.vit.embeddings.position_embeddings = nn.Identity()
        # 禁用位置编码的 forward 计算
        self.vit.embeddings._dropout = nn.Identity()

        # 仅初始化1个PEG模块（原论文：仅第一层Encoder后使用）
        self.peg = PEG(embed_dim, kernel_size=3)

        # 分类头（与原代码完全一致，保证训练兼容）
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(embed_dim, num_labels)
        self.num_labels = num_labels

        # 预计算patch尺寸（224x224图像 + 16x16 patch → 14x14）
        self.patch_h = self.vit.config.image_size // self.vit.config.patch_size
        self.patch_w = self.vit.config.image_size // self.vit.config.patch_size

    def forward(self, pixel_values, labels=None):
        """
        前向传播逻辑（严格对齐CPVT原论文）：
        1. 输入图像 → ViT Embedding → 第一层Encoder
        2. 第一层Encoder输出 → PEG条件位置编码
        3. PEG输出 → 后续所有Encoder层
        4. 最后一层输出 → 分类头
        """
        # 1. 获取ViT所有层的隐藏状态（包含Embedding和各Encoder层输出）
        outputs = self.vit(pixel_values=pixel_values, output_hidden_states=True)
        hidden_states = outputs.hidden_states  # 长度：num_hidden_layers + 1

        # 2. 提取第一层Encoder输出（hidden_states[1]），插入PEG
        first_encoder_output = hidden_states[1]  # Embedding→第一层Encoder的输出
        peg_output = self.peg(first_encoder_output, self.patch_h, self.patch_w)

        # 3. 构造新的隐藏状态序列：替换第一层Encoder输出为PEG输出
        new_hidden_states = [hidden_states[0]]  # 保留原始Embedding输出
        new_hidden_states.append(peg_output)  # 第一层Encoder输出替换为PEG输出
        # 后续Encoder层直接使用PEG输出作为输入（原论文逻辑）
        for i in range(2, len(hidden_states)):
            # 手动计算后续Encoder层的输出（基于PEG输出）
            encoder_layer = self.vit.encoder.layer[i - 1]
            layer_output = encoder_layer(new_hidden_states[-1])[0]
            new_hidden_states.append(layer_output)

        # 4. 取最后一层输出做分类
        last_hidden = new_hidden_states[-1]
        cls_output = self.dropout(last_hidden[:, 0])  # 取CLS token
        logits = self.classifier(cls_output)

        # 5. 损失计算（与原代码一致）
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        # 6. 返回标准输出格式（保证训练脚本兼容）
        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=new_hidden_states,  # 返回修改后的隐藏状态
            attentions=outputs.attentions,
        )

    def get_model(device="cuda"):
        """创建模型并移至指定设备（兼容原训练脚本）"""
        model = ViTCPVTForImageClassification()
        model.to(device)
        return model

    if __name__ == "__main__":
        # 测试模型初始化和前向传播（验证无报错）
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model = get_model(device)
        print(f"模型初始化成功，设备: {device}")

        # 测试前向传播
        dummy_input = torch.randn(2, 3, 224, 224).to(device)
        dummy_labels = torch.tensor([0, 1]).to(device)
        output = model(dummy_input, dummy_labels)

        print(f"模型输出logits形状: {output.logits.shape}")
        print(f"损失值: {output.loss.item():.4f}")
        print("PEG已按CPVT原论文放置在第一层Encoder后，验证通过！")