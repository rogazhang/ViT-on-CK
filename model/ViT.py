import torch
import torch.nn as nn
from transformers import ViTModel
from transformers.modeling_outputs import SequenceClassifierOutput

# 导入标签列表（保持一致性）
from dataloader import string_labels

class ViTForImageClassification(nn.Module):
    """基于ViT的图像分类模型"""
    def __init__(self, num_labels=len(string_labels)):
        super(ViTForImageClassification, self).__init__()
        local_vit_path = "./vit_pretrained/vit-base-patch16-224-in21k"
        self.vit = ViTModel.from_pretrained(local_vit_path)
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(self.vit.config.hidden_size, num_labels)
        self.num_labels = num_labels

    def forward(self, pixel_values, labels=None):
        """前向传播，包含损失计算"""
        outputs = self.vit(pixel_values=pixel_values)
        # 取CLS token的输出
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
    """创建模型并移至指定设备"""
    model = ViTForImageClassification()
    model.to(device)
    return model

if __name__ == "__main__":
    # 测试模型初始化
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = get_model(device)
    print(f"模型初始化成功，设备: {device}")
    # 测试前向传播
    dummy_input = torch.randn(2, 3, 224, 224).to(device)
    dummy_labels = torch.tensor([0, 1]).to(device)
    output = model(dummy_input, dummy_labels)
    print(f"模型输出logits形状: {output.logits.shape}")
    print(f"损失值: {output.loss.item():.4f}")