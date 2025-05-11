import os
import yaml
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from src.datasets import CervicalDataset
from src.models import CervicalMRIMultiTaskModel
from src.utils import load_yaml, get_logger, compute_multitask_metrics

def collate_fn(batch):
    # batch: list of dict {'sag','tra','labels','id'}
    sag = torch.stack([item['sag'] for item in batch], dim=0)
    tra = torch.stack([item['tra'] for item in batch], dim=0)
    ids = [item['id'] for item in batch]
    labels = {
        'curvature': torch.stack([item['labels']['curvature'] for item in batch]),
        'alignment': torch.stack([item['labels']['alignment'] for item in batch]),
        'disc': torch.stack([item['labels']['disc'] for item in batch]),
        'canal': torch.stack([item['labels']['canal'] for item in batch]),
    }
    return {'sag': sag, 'tra': tra, 'labels': labels, 'id': ids}


def train():
    # 加载配置
    cfg = load_yaml('configs/config.yaml')
    logger = get_logger('Trainer')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 数据集与 DataLoader
    data_cfg = cfg['data']
    train_ds = CervicalDataset(
        json_path=data_cfg['train_path'],
        image_dir=data_cfg['image_dir'],
        split='train',
        transform=None
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg['training']['batch_size'],
        shuffle=True,
        num_workers=4,
        collate_fn=collate_fn
    )
    # 如有验证集，可同理加载

    # 模型、优化器、scheduler
    model_cfg = cfg['model']
    model = CervicalMRIMultiTaskModel(
        backbone_name=model_cfg['backbone'],
        lora_r=model_cfg['lora_rank'],
        lora_alpha=model_cfg['lora_alpha'],
        freeze_backbone=model_cfg['freeze_backbone']
    ).to(device)
    optimizer = AdamW(model.parameters(), lr=cfg['training']['learning_rate'], weight_decay=cfg['training']['weight_decay'])
    total_steps = len(train_loader) * cfg['training']['epochs']
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=cfg['training']['warmup_steps'],
        num_training_steps=total_steps
    )

    best_f1 = 0.0
    # 训练循环
    for epoch in range(1, cfg['training']['epochs'] + 1):
        model.train()
        total_loss = 0.0

        for batch in train_loader:
            sag = batch['sag'].to(device)
            tra = batch['tra'].to(device)
            labels = {k: v.to(device) for k, v in batch['labels'].items()}
            optimizer.zero_grad()
            outputs = model(sag, tra)
            # 计算多任务总 loss
            loss = 0
            loss += torch.nn.functional.cross_entropy(outputs['curvature'], labels['curvature'])
            loss += torch.nn.functional.cross_entropy(outputs['alignment'], labels['alignment'])
            for i in range(outputs['disc'].shape[1]):
                loss += torch.nn.functional.cross_entropy(outputs['disc'][:, i], labels['disc'][:, i])
            for j in range(outputs['canal'].shape[1]):
                loss += torch.nn.functional.cross_entropy(outputs['canal'][:, j], labels['canal'][:, j])
            loss.backward()
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        logger.info(f"Epoch {epoch}/{cfg['training']['epochs']} - loss: {avg_loss:.4f}")

        # 评估环节（略，可调用验证集并使用 compute_multitask_metrics）
        # 保存最佳模型
        # if val_f1 > best_f1: ...
        save_path = os.path.join(cfg['logging']['save_dir'], 'best_model.pt')
        torch.save(model.state_dict(), save_path)
        logger.info(f"Saved model to {save_path}")

if __name__ == '__main__':
    train()
