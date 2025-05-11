import os
import torch
import yaml
from torch.utils.data import DataLoader
from src.datasets import CervicalDataset
from src.models import CervicalMRIMultiTaskModel
from src.utils import load_json, save_json


def run(config_path: str):
    # 读取配置
    cfg = yaml.safe_load(open(config_path, 'r', encoding='utf-8'))
    data_cfg = cfg['data']
    model_cfg = cfg['model']
    log_cfg = cfg['logging']

    # 加载测试集
    test_records = load_json(data_cfg['test_path'])
    test_dir = data_cfg.get('image_dir', os.path.dirname(data_cfg['test_path']))
    test_ds = CervicalDataset(
        json_path=data_cfg['test_path'],
        image_dir=test_dir,
        split='test'
    )
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False)

    # 初始化模型并加载权重
    model = CervicalMRIMultiTaskModel(
        backbone_name=model_cfg['backbone'],
        lora_r=model_cfg.get('lora_rank', 8),
        lora_alpha=model_cfg.get('lora_alpha', 16),
        freeze_backbone=model_cfg.get('freeze_backbone', True)
    )
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_path = os.path.join(log_cfg['save_dir'], 'best_model.pt')
    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state)
    model.to(device).eval()

    # 推理
    results = []
    with torch.no_grad():
        for batch in test_loader:
            sag = batch['sag'].to(device)      # (1,3,C,H,W)
            tra = batch['tra'].to(device)      # (1,11,C,H,W)
            ids = batch['id']
            outputs = model(sag, tra)

            # 采集预测结果
            qd = int(outputs['curvature'].argmax(dim=-1).item())
            sl = int(outputs['alignment'].argmax(dim=-1).item())
            zjppt = outputs['disc'].argmax(dim=-1).squeeze(0).tolist()  # length 5
            zyzg = outputs['canal'].argmax(dim=-1).squeeze(0).tolist()  # length 11

            results.append({
                'id': ids[0],
                'qd': qd,
                'sl': sl,
                'zjppt': zjppt,
                'zyzg': zyzg
            })

    # 保存预测
    save_json(results, data_cfg['output_predict'])
    print(f"Saved {len(results)} predictions to {data_cfg['output_predict']}")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/config.yaml',
                        help='配置文件路径')
    args = parser.parse_args()
    run(args.config)
