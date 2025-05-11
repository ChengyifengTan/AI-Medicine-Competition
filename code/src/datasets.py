import os
import json
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms

class CervicalDataset(Dataset):
    """
    通用颈椎MRI数据集
    - split='train' 返回 {'sag', 'tra', 'labels', 'id'}
    - split='test'  返回 {'sag', 'tra', 'id'}

    数据结构：每个样本为一个 id 文件夹，包含:
      {id}/
        sag/5.png,6.png,7.png   # 矢状位图像（3 张）
        tra/2.png,2-3.png,...,6-7.png,7.png  # 横轴位图像（11 张）
    JSON字段:
      id: str, 样本唯一 ID
      qd: int, 曲度标签 (0/1/2)
      sl: int, 对齐标签 (0/1)
      zjppt: List[int], 椎间盘状态 (长度=5，每项0/1/2)
      zyzg: List[int], 椎管状态 (长度=11，每项0/1)
    """
    def __init__(
        self,
        json_path: str,
        image_dir: str,
        split: str = 'train',
        transform: transforms.Compose = None
    ):
        assert split in ('train', 'test'), "split 必须为 'train' 或 'test'"
        self.split = split
        self.image_dir = image_dir.rstrip('/')
        with open(json_path, 'r', encoding='utf-8') as f:
            self.records = json.load(f)

        # 默认数据预处理：统一大小、ToTensor
        self.transform = transform or transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])

        # sagittal & transverse 文件名列表，确保顺序对应标签顺序
        self.sag_slices = ['5.png', '6.png', '7.png']
        self.tra_slices = [
            '2.png', '2-3.png', '3.png', '3-4.png', '4.png', '4-5.png',
            '5.png', '5-6.png', '6.png', '6-7.png', '7.png'
        ]

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        rec = self.records[idx]
        sample_id = rec['id']
        sample_dir = os.path.join(self.image_dir, sample_id)
        # sag images
        sag_imgs = []
        sag_dir = os.path.join(sample_dir, 'sag')
        for fname in self.sag_slices:
            path = os.path.join(sag_dir, fname)
            if not os.path.exists(path):
                raise FileNotFoundError(f"Missing sagittal slice {path}")
            img = Image.open(path).convert('RGB')
            sag_imgs.append(self.transform(img))
        sag = torch.stack(sag_imgs, dim=0)  # (3, C, H, W)

        # tra images
        tra_imgs = []
        tra_dir = os.path.join(sample_dir, 'tra')
        for fname in self.tra_slices:
            path = os.path.join(tra_dir, fname)
            if not os.path.exists(path):
                raise FileNotFoundError(f"Missing transverse slice {path}")
            img = Image.open(path).convert('RGB')
            tra_imgs.append(self.transform(img))
        tra = torch.stack(tra_imgs, dim=0)  # (11, C, H, W)

        sample = {'sag': sag, 'tra': tra, 'id': sample_id}

        if self.split == 'train':
            # 标签
            labels = {
                'curvature': torch.tensor(rec['qd'], dtype=torch.long),
                'alignment': torch.tensor(rec['sl'], dtype=torch.long),
                'disc': torch.tensor(rec['zjppt'], dtype=torch.long),
                'canal': torch.tensor(rec['zyzg'], dtype=torch.long),
            }
            sample['labels'] = labels

        return sample
