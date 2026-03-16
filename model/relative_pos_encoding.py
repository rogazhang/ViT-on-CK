import torch
import torch.nn as nn
from transformers import ViTForImageClassification, ViTConfig


class GlobalRelativePositionBias(nn.Module):
    def __init__(self, num_heads, h=14, w=14):
        super().__init__()
        self.num_heads = num_heads
        self.h, self.w = h, w
        self.num_patches = h * w

        # 1. 定义相对位置偏置表：(2h-1)*(2w-1) 是所有可能的相对位移组合
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * h - 1) * (2 * w - 1), num_heads)
        )

        # 2. 定义 CLS Token 专属的偏置参数
        # cls_to_patches: CLS 看 Patches 的偏置 (1, nH, 1, HW)
        # patches_to_cls: Patches 看 CLS 的偏置 (1, nH, HW, 1)
        # cls_to_cls: CLS 看自己的偏置 (1, nH, 1, 1)
        self.cls_to_patches = nn.Parameter(torch.zeros(1, num_heads, 1, self.num_patches))
        self.patches_to_cls = nn.Parameter(torch.zeros(1, num_heads, self.num_patches, 1))
        self.cls_to_cls = nn.Parameter(torch.zeros(1, num_heads, 1, 1))

        # 3. 计算 Patch 间的二维相对位置索引
        coords_h = torch.arange(h)
        coords_w = torch.arange(w)
        coords = torch.stack(torch.meshgrid(coords_h, coords_w, indexing='ij'))  # [2, h, w]
        coords_flatten = torch.flatten(coords, 1)  # [2, hw]
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # [2, hw, hw]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # [hw, hw, 2]

        relative_coords[:, :, 0] += h - 1  # 移至正数区间
        relative_coords[:, :, 1] += w - 1
        relative_coords[:, :, 0] *= 2 * w - 1
        relative_position_index = relative_coords.sum(-1)  # [hw, hw]
        self.register_buffer("relative_position_index", relative_position_index)

        # 为所有偏置参数添加截断正态初始化
        nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)
        nn.init.trunc_normal_(self.cls_to_patches, std=0.02)
        nn.init.trunc_normal_(self.patches_to_cls, std=0.02)
        nn.init.trunc_normal_(self.cls_to_cls, std=0.02)

    def forward(self):
        # 获取 Patch-to-Patch 的偏置
        patch_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)]
        patch_bias = patch_bias.view(self.num_patches, self.num_patches, self.num_heads)
        patch_bias = patch_bias.permute(2, 0, 1).contiguous().unsqueeze(0)  # [1, nH, HW, HW]

        # 拼接 CLS Token 相关的偏置，重组成 [1, nH, 1+HW, 1+HW]
        # 第一步：把 patches_to_cls 拼到 patch_bias 左边 -> [1, nH, HW, 1+HW]
        top_part = torch.cat([self.cls_to_cls, self.cls_to_patches], dim=-1)  # [1, nH, 1, 1+HW]
        bottom_part = torch.cat([self.patches_to_cls, patch_bias], dim=-1)  # [1, nH, HW, 1+HW]

        full_bias = torch.cat([top_part, bottom_part], dim=-2)  # [1, nH, 1+HW, 1+HW]
        return full_bias


class RelPosAttention(nn.Module):
    def __init__(self, original_attention_module):
        super().__init__()
        # 继承原模型的维度和权重
        old_self = original_attention_module.attention
        old_output = original_attention_module.output

        self.num_heads = old_self.num_attention_heads
        self.head_dim = old_self.attention_head_size
        self.scale = self.head_dim ** -0.5

        self.query = old_self.query
        self.key = old_self.key
        self.value = old_self.value
        self.attn_drop = old_self.dropout
        self.proj = old_output.dense
        self.proj_drop = old_output.dropout

        # 全局相对位置偏置模块
        self.rel_pos = GlobalRelativePositionBias(num_heads=self.num_heads, h=14, w=14)

    def forward(self, hidden_states,attention_mask=None, **kwargs):
        B, L, C = hidden_states.shape

        # QKV 计算
        q = self.query(hidden_states).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.key(hidden_states).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.value(hidden_states).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)

        # Scaled Dot-Product Attention
        attn = (q @ k.transpose(-2, -1)) * self.scale

        # 处理 attention_mask（兼容原逻辑，避免掩码失效）
        if attention_mask is not None:
            # 扩展 mask 维度以匹配注意力矩阵形状 [B, 1, L, L]
            attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
            attn = attn + attention_mask

        # 扩展相对位置偏置到batch维度，避免广播
        rel_bias = self.rel_pos()  # [1, nH, 1+HW, 1+HW]
        rel_bias = rel_bias.expand(B, -1, -1, -1)  # [B, nH, 1+HW, 1+HW]
        attn = attn + rel_bias

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        # 输出投影
        x = (attn @ v).transpose(1, 2).reshape(B, L, -1)
        x = self.proj(x)
        x = self.proj_drop(x)

        # 适配 Transformers 库的返回格式 (hidden_states, attn_weights)
        return (x, attn) if kwargs.get("output_attentions") else (x,)


def get_model(model_path, num_labels=7, device='cuda'):
    config = ViTConfig.from_pretrained(model_path, num_labels=num_labels)
    model = ViTForImageClassification.from_pretrained(model_path, config=config)

    # 遍历替换所有编码器层中的注意力模块
    for i, layer in enumerate(model.vit.encoder.layer):
        layer.attention = RelPosAttention(layer.attention)

    # 既然有了相对位置偏置，我们可以冻结或清空绝对位置编码（可选）
    # 实践证明保留微小的绝对位置信息有时也有益，但为了纯粹性，这里将其设为 0
    with torch.no_grad():
        model.vit.embeddings.position_embeddings.fill_(0)
        model.vit.embeddings.position_embeddings.requires_grad = False

    model.to(device)
    return model

