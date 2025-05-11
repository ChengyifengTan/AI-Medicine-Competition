import torch
import torch.nn as nn
from transformers import QwenVisionModel
from peft import LoraConfig, get_peft_model, TaskType


class CervicalMRIMultiTaskModel(nn.Module):
    """
    多切片颈椎MRI多任务分类模型

    输入:
      - sag: (B, 3, C, H, W) 矢状位三切片
      - tra: (B, 11, C, H, W) 横轴十一切片

    输出 logits:
      - curvature: (B,3)
      - alignment: (B,2)
      - disc: (B,5,3)
      - canal: (B,11,2)

    架构:
      1. Qwen2.5-VL-3B 视觉编码器 (ViT) + LoRA 微调
      2. 融合 sag/tra 特征
      3. 多任务 MLP 分类头
    """
    def __init__(
        self,
        backbone_name: str = "Qwen/qwen2.5-vl-3b",
        lora_r: int = 8,
        lora_alpha: int = 16,
        freeze_backbone: bool = True,
    ):
        super().__init__()
        # 1. 加载视觉编码器
        self.backbone = QwenVisionModel.from_pretrained(backbone_name, add_pooling_layer=True)
        hidden_size = self.backbone.config.hidden_size

        # 冻结原始权重
        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

        # 2. 注入 LoRA
        peft_config = LoraConfig(
            task_type=TaskType.FEATURE_EXTRACTION,
            inference_mode=False,
            r=lora_r,
            lora_alpha=lora_alpha,
            target_modules=["q_proj","k_proj","v_proj","o_proj"],
            lora_dropout=0.05
        )
        self.backbone = get_peft_model(self.backbone, peft_config)

        # 3. 特征整合层
        # 将 sag/tra 两路切片展开后拼接池化
        self.fusion = nn.Sequential(
            nn.Linear(hidden_size*2, hidden_size),
            nn.ReLU()
        )

        # 4. 多任务分类头
        self.curvature_head = nn.Linear(hidden_size, 3)
        self.alignment_head = nn.Linear(hidden_size, 2)
        # 5 个椎间盘三分类
        self.disc_heads = nn.ModuleList([nn.Linear(hidden_size, 3) for _ in range(5)])
        # 11 个椎管二分类
        self.canal_heads = nn.ModuleList([nn.Linear(hidden_size, 2) for _ in range(11)])

    def forward(self, sag: torch.Tensor, tra: torch.Tensor):
        """
        sag: (B,3,C,H,W)
        tra: (B,11,C,H,W)
        """
        B = sag.size(0)
        # 将多个切片视作批量处理
        sag_flat = sag.view(-1, *sag.shape[2:])  # (B*3,C,H,W)
        tra_flat = tra.view(-1, *tra.shape[2:])  # (B*11,C,H,W)

        sag_out = self.backbone(pixel_values=sag_flat).pooler_output  # (B*3, H')
        tra_out = self.backbone(pixel_values=tra_flat).pooler_output  # (B*11, H')

        # 按原 batch 恢复维度，并池化
        sag_feat = sag_out.view(B, 3, -1).mean(dim=1)  # (B, H')
        tra_feat = tra_out.view(B, 11, -1).mean(dim=1) # (B, H')

        # 融合两个方向特征
        fusion_feat = self.fusion(torch.cat([sag_feat, tra_feat], dim=1))  # (B, H')

        # 多任务输出
        logits = {}
        logits['curvature'] = self.curvature_head(fusion_feat)
        logits['alignment'] = self.alignment_head(fusion_feat)
        logits['disc'] = torch.stack([h(fusion_feat) for h in self.disc_heads], dim=1)  # (B,5,3)
        logits['canal'] = torch.stack([h(fusion_feat) for h in self.canal_heads], dim=1)  # (B,11,2)
        return logits

    def freeze_backbone(self):
        for p in self.backbone.parameters(): p.requires_grad = False

    def unfreeze_backbone(self):
        for p in self.backbone.parameters(): p.requires_grad = True
