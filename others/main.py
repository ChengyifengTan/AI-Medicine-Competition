import os
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
task = 'qd' # qd, sl, ajppt, zyzg
if task == 'qd':
    num_classes = 3
elif task == 'sl':
    num_classes = 2
elif task == 'zjppt':
    num_classes = 4
elif task == 'zyzg':
    num_classes = 4


# 数据路径
TRAIN_IMAGE_DIR = f"e:\AI-Medicine-Competition\data\mri_images\train"
TRAIN_LABEL_PATH = f"e:\AI-Medicine-Competition\data\dataset\{task}_train.json"

# 加载标签数据
def load_labels(label_path):
    with open(label_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    labels = []
    id_to_index = {}  # 存储image_id到索引的映射
    index_to_id = {}  # 存储索引到image_id的映射
    image_ids = []    # 存储所有image_id
    
    for idx, item in enumerate(data):
        # print(f'==item={item}')
        image_id = item.get('id', '')
        answer = item.get('answer', '')
        image_paths = item.get('image', [])
        
        # 从图像路径中提取类型信息（sag或tra）和图像编号
        image_type = "unknown"
        image_number = "0"
        if image_paths and len(image_paths) > 0:
            path = image_paths[0]
            # 提取类型信息
            if "/sag/" in path or "\\sag\\" in path:
                image_type = "sag"
            elif "/tra/" in path or "\\tra\\" in path:
                image_type = "tra"
            
            # 提取图像编号
            import re
            match = re.search(r'/(sag|tra)/(\d+)\.png', path)
            if match:
                image_number = match.group(2)
        
        # 创建唯一标识符，结合ID、类型和图像编号
        unique_id = f"{image_id}_{image_type}_{image_number}"
        
        # 将标签转换为数字
        if task == 'qd':
            if "A" in answer:
                labels.append(0)
            elif "B" in answer:
                labels.append(1)
            elif "C" in answer:
                labels.append(2)
        elif task == 'sl':
            if "A" in answer:
                labels.append(0)
            elif "B" in answer:
                labels.append(1)
            
        # 建立image_id和索引的双向映射
        id_to_index[unique_id] = idx
        index_to_id[idx] = unique_id
        image_ids.append(unique_id)
        
        # print(f'image_id={unique_id}\nanswer={answer}\nindex={idx}\n')
    
    # # 检查是否有重复的image_id
    # unique_ids = set(image_ids)
    # if len(unique_ids) != len(image_ids):
    #     print(f"警告: 发现重复的image_id! 总数: {len(image_ids)}, 唯一数: {len(unique_ids)}")
    #     # 可以进一步分析重复的ID
    #     from collections import Counter
    #     id_counts = Counter(image_ids)
    #     duplicates = {id: count for id, count in id_counts.items() if count > 1}
    #     print(f"重复的image_id: {duplicates}")
    
    return labels, id_to_index, index_to_id, image_ids

# 自定义数据集
class MRIDataset(Dataset):
    def __init__(self, image_dir, labels, transform=None):
        self.image_dir = image_dir
        self.labels = labels
        self.transform = transform
        self.image_ids = list(labels.keys())
    
    def __len__(self):
        return len(self.image_ids)
    
    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        
        # 解析image_id以获取正确的路径
        # 假设image_id格式为: "01240522213019_sag_5"
        parts = image_id.split('_')
        if len(parts) >= 3:
            patient_id = parts[0]
            image_type = parts[1]  # sag或tra
            image_number = parts[2]
            
            # 构建正确的图像路径
            image_path = os.path.join(self.image_dir, patient_id, image_type, f"{image_number}.png")
        else:
            # 如果格式不符合预期，使用原始方式
            image_path = os.path.join(self.image_dir, image_id)
        
        # 确保文件存在
        if not os.path.exists(image_path):
            print(f"警告: 图像不存在 {image_path}")
            # 返回一个空白图像和标签
            image = Image.new('RGB', (224, 224), color='black')
            label = self.labels[image_id]
            if self.transform:
                image = self.transform(image)
            return image, label
        
        # 加载图像
        image = Image.open(image_path).convert('RGB')
        label = self.labels[image_id]
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

# 定义数据转换
data_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 加载标签
labels, id_to_index, index_to_id, image_ids = load_labels(TRAIN_LABEL_PATH)
print(f"加载了 {len(labels)} 个标签")
labels = np.array(labels)

# 划分训练集和验证集
train_indices, val_indices = train_test_split(range(len(labels)), test_size=0.2, shuffle=False)
train_indices = np.array(train_indices)
val_indices = np.array(val_indices)
print(f'train_indices.shape={train_indices.shape}, val_indices.shape={val_indices.shape}')

# 创建训练集和验证集的图像ID到标签的映射
train_id_label_dict = {index_to_id[idx]: labels[idx] for idx in train_indices}
val_id_label_dict = {index_to_id[idx]: labels[idx] for idx in val_indices}

print(f'训练集样本数: {len(train_id_label_dict)}, 验证集样本数: {len(val_id_label_dict)}')

# 创建数据集
train_dataset = MRIDataset(TRAIN_IMAGE_DIR, train_id_label_dict, transform=data_transforms)
val_dataset = MRIDataset(TRAIN_IMAGE_DIR, val_id_label_dict, transform=data_transforms)

# 创建数据加载器
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

# 定义模型 - 使用预训练的ResNet50
class MRIClassifier(nn.Module):
    def __init__(self, num_classes=3):
        super(MRIClassifier, self).__init__()
        self.model = models.resnet50(pretrained=True)
        # 冻结部分层以加快训练
        for param in list(self.model.parameters())[:-20]:
            param.requires_grad = False
        # 修改最后的全连接层
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Sequential(
            nn.Linear(num_ftrs, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        return self.model(x)

# 初始化模型
model = MRIClassifier(num_classes=3).to(device)

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)

# 训练函数
def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs=10):
    best_val_loss = float('inf')
    best_model_weights = None
    
    for epoch in range(num_epochs):
        # traning phase
        model.train()
        train_loss = 0.0
        train_preds = []
        train_targets = []
        
        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            
            outputs = model(inputs)
            # print(f'==outputs={outputs.shape}, {outputs}, labels={labels.shape}, {labels}')
            loss = criterion(outputs, labels)
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            train_preds.extend(preds.cpu().numpy())
            train_targets.extend(labels.cpu().numpy())
        
        train_loss = train_loss / len(train_loader.dataset)
        train_acc = accuracy_score(train_targets, train_preds)
        
        # validation phase
        model.eval()
        val_loss = 0.0
        val_preds = []
        val_targets = []
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)
                
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item() * inputs.size(0)
                _, preds = torch.max(outputs, 1)
                val_preds.extend(preds.cpu().numpy())
                val_targets.extend(labels.cpu().numpy())
        
        val_loss = val_loss / len(val_loader.dataset)
        val_acc = accuracy_score(val_targets, val_preds)
        val_precision = precision_score(val_targets, val_preds, average='weighted')
        val_recall = recall_score(val_targets, val_preds, average='weighted')
        val_f1 = f1_score(val_targets, val_preds, average='weighted')
        
        # 更新学习率
        scheduler.step(val_loss)
        
        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_weights = model.state_dict().copy()
        
        print(f'Epoch {epoch+1}/{num_epochs}:')
        print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}')
        print(f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, Precision: {val_precision:.4f}, Recall: {val_recall:.4f}, F1: {val_f1:.4f}')
        print('-' * 60)
    
    # 加载最佳模型权重
    model.load_state_dict(best_model_weights)
    return model

# 训练模型
print("开始训练模型...")
trained_model = train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs=15)

# 保存模型
torch.save(trained_model.state_dict(), 'mri_classifier_model.pth')
print("模型已保存到 'mri_classifier_model.pth'")

# 评估模型
def evaluate_model(model, data_loader):
    model.eval()
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(labels.cpu().numpy())
    
    accuracy = accuracy_score(all_targets, all_preds)
    precision = precision_score(all_targets, all_preds, average='weighted')
    recall = recall_score(all_targets, all_preds, average='weighted')
    f1 = f1_score(all_targets, all_preds, average='weighted')
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }

# 在验证集上评估模型
print("在验证集上评估模型...")
eval_metrics = evaluate_model(trained_model, val_loader)
print(f"验证集评估结果:")
print(f"准确率: {eval_metrics['accuracy']:.4f}")
print(f"精确率: {eval_metrics['precision']:.4f}")
print(f"召回率: {eval_metrics['recall']:.4f}")
print(f"F1分数: {eval_metrics['f1']:.4f}")