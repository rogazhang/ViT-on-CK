import torch
import torch.nn as nn
import math
from transformers import ViTModel
from transformers.modeling_outputs import SequenceClassifierOutput
from dataloader import string_labels

class ViTForImageClassification(nn.Module):
    def __init__(self, num_labels=len(string_labels)):
        super(ViTForImageClassification, self).__init__()
        local_vit_path = "./vit_pretrained/vit-base-patch16-224-in21k"

        # 1. 加载基础模型
        self.vit = ViTModel.from_pretrained(local_vit_path)

        # 2. 替换为固定位置编码
        self._replace_with_fixed_pos_embeddings()

        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(self.vit.config.hidden_size, num_labels)
        self.num_labels = num_labels

    def _replace_with_fixed_pos_embeddings(self):
        """生成并替换固定的正弦位置编码"""
        config = self.vit.config
        seq_len = config.hidden_size  # 维度 d_model
        max_position = self.vit.embeddings.position_embeddings.shape[1]  # 197
        hidden_size = config.hidden_size

        # 计算正弦编码矩阵
        pe = torch.zeros(max_position, hidden_size)
        position = torch.arange(0, max_position, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, hidden_size, 2).float() * (-math.log(10000.0) / hidden_size))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # 形状变为 (1, 197, 768)

        # 覆盖现有的 position_embeddings
        # 使用 nn.Parameter 包装，但设置 requires_grad=False 停用更新
        self.vit.embeddings.position_embeddings = nn.Parameter(pe, requires_grad=False)

        print(f"成功替换固定位置编码，形状: {pe.shape}")

    def forward(self, pixel_values, labels=None):
        outputs = self.vit(pixel_values=pixel_values)
        output = self.dropout(outputs.last_hidden_state[:, 0])
        logits = self.classifier(output)

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


def get_model(device="cuda"):
    model = ViTForImageClassification()
    model.to(device)
    return model


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_model(device)

    # 验证参数是否已冻结
    pos_embed_grad = model.vit.embeddings.position_embeddings.requires_grad
    print(f"位置编码是否参与训练: {pos_embed_grad}")  # 应输出 False

    dummy_input = torch.randn(2, 3, 224, 224).to(device)
    output = model(dummy_input)
    print(f"模型输出成功，Logits 形状: {output.logits.shape}")