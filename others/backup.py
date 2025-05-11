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
import argparse
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_scorezy
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
import tensorflow as tf
import keras
# print(tf.__version__)
# print(keras.__version__)
# import keras_applications
# print(keras_applications.__version__)
import efficientnet
print(efficientnet.__version__)
import segmentation_models as sm
from imblearn.over_sampling import RandomOverSampler

def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")
parser = argparse.ArgumentParser(description='')
parser.add_argument('--task', type=str, required=True, choices=['qd', 'sl', 'zjppt', 'zyzg', 'positioning'], help='Task type: qd/sl/zjppt/zyzg')
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
task = args.task

if task == 'qd':
    num_classes = 3
elif task =='sl':
    num_classes = 2
elif task == 'zjppt' or task == 'zyzg':
    num_classes = 4
elif task == 'poisitioning':
    num_classes = 1

# Data paths 
TRAIN_IMAGE_DIR = os.path.dirname(__file__)+f"/../data/mri_images/train"
TRAIN_LABEL_PATH =  os.path.dirname(__file__)+f"/../data/dataset/{task}_train.json"

# Load label data
def load_labels(label_path):
    print(f'====label_path={label_path}')
    with open(label_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    labels = []
    path_to_index = {}  # Mapping from image_id to index
    index_to_path = {}  # Mapping from index to image_id 
    index_to_labels = {}  # Mapping from index to label
    image_ids = []    # Store all image_ids
    
    for idx, item in enumerate(data):
        # print(f'==item={item}')
        image_id = item.get('id', '')
        answer = item.get('answer', '')
        image_paths = item.get('image', [])
        # print(f'image_path={image_paths}')
        
        # Extract type info (sag or tra) and image number from image path
        image_type = "unknown"
        image_number = "0"
        if image_paths and len(image_paths) > 0:
            path = image_paths[0]
            # print(f'path={path}')
            
            # Extract image number
            import re
            match = re.search(r'/(sag|tra)/(\d+)\.png', path)
            if match:
                image_number = match.group(2)
        
            # Create unique identifier combining ID, type and image number
            # unique_id = f"{image_id}_{image_type}_{image_number}"
            
            adjusted_idx = idx + 1
            
            # Convert labels to numbers
            label_value = None
            if task == 'qd':
                if "A" in answer:
                    label_value = 0
                elif "B" in answer:
                    label_value = 1
                elif "C" in answer:
                    label_value = 2
            elif task == 'sl':
                if "A" in answer:
                    label_value = 0
                elif "B" in answer:
                    label_value = 1
            elif task == 'zjppt':
                if "A" in answer:
                    label_value = 0
                elif "B" in answer:
                    label_value = 1
                elif "C" in answer:
                    label_value = 2
                elif "D" in answer:
                    label_value = 3
            elif task == 'zyzg':
                if "A" in answer:
                    label_value = 0
                elif "B" in answer:
                    label_value = 1
                elif "C" in answer:
                    label_value = 2
                elif "D" in answer:
                    label_value = 3
            elif task == 'positioning':
                # 提取object字段（颈椎节段名称）
                object_name = item.get('object', '')
                
                # 从answer中提取边界框坐标
                box_match = re.search(r'<box>\((\d+),(\d+)\),\((\d+),(\d+)\)<box>', answer)
                if box_match:
                    x1, y1, x2, y2 = map(int, box_match.groups())
                    label_value = {
                        'object': object_name,
                        'bbox': [x1, y1, x2, y2]
                    }
            
            if label_value is not None:
                labels.append(label_value)
                index_to_labels[adjusted_idx] = label_value
                
            # Establish bidirectional mapping between image_id and index
            path_to_index[path] = adjusted_idx
            index_to_path[adjusted_idx] = path
            
            # print(f'image_path={path}\nanswer={answer}\nindex={idx}\n')
    
    index_list = list(range(1, len(labels) + 1))
    
    return path_to_index, index_to_path, index_to_labels, index_list


class MRIDataset(Dataset):
    def __init__(self, images, labels):
        self.images = images
        self.labels = labels

    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        return image, label


data_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

path_to_index, index_to_path, index_to_labels, index_list = load_labels(TRAIN_LABEL_PATH)
image_paths = [index_to_path[idx] for idx in index_list]

if task == 'positioning':
    # Create special data processing logic for segmentation tasks
    # Create a function to generate segmentation masks
    def create_mask_from_bbox(bbox, img_size=(224, 224)):
        """Create binary mask from bounding box"""
        x1, y1, x2, y2 = bbox
        mask = np.zeros(img_size, dtype=np.float32)
        
        # Adjust bounding box coordinates to fit resized image
        h_ratio = img_size[0] / 224
        w_ratio = img_size[1] / 224
        
        x1_scaled = int(x1 * w_ratio)
        y1_scaled = int(y1 * h_ratio)
        x2_scaled = int(x2 * w_ratio)
        y2_scaled = int(y2 * h_ratio)
        
        # Ensure coordinates are within image bounds
        x1_scaled = max(0, min(x1_scaled, img_size[1]-1))
        y1_scaled = max(0, min(y1_scaled, img_size[0]-1))
        x2_scaled = max(0, min(x2_scaled, img_size[1]-1))
        y2_scaled = max(0, min(y2_scaled, img_size[0]-1))
        
        # Fill the bounding box region
        mask[y1_scaled:y2_scaled+1, x1_scaled:x2_scaled+1] = 1.0
        return mask
    
    # Load images and create masks
    images = []
    masks = []
    object_names = []
    
    for idx in index_list:
        path = index_to_path[idx]
        label_info = index_to_labels[idx]
        
        # print(f'path={path}')
        img = Image.open(os.path.dirname(__file__)+'/../data'+path).convert('RGB')
        img = img.resize((224, 224))
        img_array = np.array(img) / 255.0  # 归一化到[0,1]
        
        # create masks
        bbox = label_info['bbox']
        mask = create_mask_from_bbox(bbox, img_size=(224, 224))
        
        images.append(img_array)
        masks.append(mask)
        object_names.append(label_info['object'])
    
    X = np.array(images)
    y = np.array(masks)
    if task != 'positioning':
        X = np.dot(X[...,:3], [0.299, 0.587, 0.114])
        X = np.expand_dims(X, axis=-1)
    y = np.expand_dims(y, axis=-1)  
    print(f'==X={X.shape}, y={y.shape}')
    
    # segmente the training set and validation set
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print(f'==X_train={X_train.shape}, y_train={y_train.shape}')
    
    sm.set_framework('tf.keras')
    tf.keras.backend.set_image_data_format('channels_last')
    
    BACKBONE = 'resnet34'
    preprocess_input = sm.get_preprocessing(BACKBONE)
    
    X_train = preprocess_input(X_train)
    X_val = preprocess_input(X_val)
    
    model = sm.Unet(
        BACKBONE,
        encoder_weights='imagenet',
        classes=1,
        activation='sigmoid'
    )
    
    model.compile(
        'Adam',
        loss=sm.losses.bce_jaccard_loss,
        metrics=[sm.metrics.iou_score],
    )
    
    history = model.fit(
        x=X_train,
        y=y_train,
        batch_size=16,
        epochs=50,
        validation_data=(X_val, y_val),
    )
    
    model.save(f'segmentation_model_{task}.h5')

    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper right')
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['iou_score'])
    plt.plot(history.history['val_iou_score'])
    plt.title('IoU Score')
    plt.ylabel('IoU')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='lower right')
    
    plt.tight_layout()
    plt.savefig(f'training_results_{task}.png')
    
    def visualize_predictions(model, X, y, num_samples=5):
        indices = np.random.choice(len(X), num_samples, replace=False)
        
        plt.figure(figsize=(15, 5*num_samples))
        for i, idx in enumerate(indices):
            img = X[idx]
            true_mask = y[idx]
            
            pred_mask = model.predict(np.expand_dims(img, axis=0))[0]
            
            plt.subplot(num_samples, 3, i*3+1)
            plt.imshow(img)
            plt.title('Original Image')
            plt.axis('off')
            
            plt.subplot(num_samples, 3, i*3+2)
            plt.imshow(true_mask[:,:,0], cmap='gray')
            plt.title('True Mask')
            plt.axis('off')
            
            plt.subplot(num_samples, 3, i*3+3)
            plt.imshow(pred_mask[:,:,0], cmap='gray')
            plt.title('Predicted Mask')
            plt.axis('off')
        
        plt.tight_layout()
        plt.savefig(f'prediction_samples_{task}.png')
    
    visualize_predictions(model, X_val, y_val)
    
    print(f"分割模型训练完成，已保存为 'segmentation_model_{task}.h5'")
else:
    x = torch.stack([
        data_transforms(Image.open(os.path.dirname(__file__)+'/../data'+path).convert('RGB')) 
        for path in image_paths
    ])
    y = torch.tensor([index_to_labels[idx] for idx in index_list])

    # 分析类别分布情况
    unique_labels, counts = np.unique(y.numpy(), return_counts=True)
    print("类别分布情况:")
    for label, count in zip(unique_labels, counts):
        print(f"类别 {label}: {count} 个样本")
    
    # 将数据转换为NumPy格式以便进行过采样
    x_np = x.numpy()
    y_np = y.numpy()
    
    # 应用随机过采样
    ros = RandomOverSampler(random_state=42)
    
    # 重塑数据以适应过采样器
    x_reshaped = x_np.reshape(x_np.shape[0], -1)
    x_resampled, y_resampled = ros.fit_resample(x_reshaped, y_np)
    
    # 将重塑的数据转换回原始形状
    x_resampled = x_resampled.reshape(-1, x_np.shape[1], x_np.shape[2], x_np.shape[3])
    
    # 转换回PyTorch张量
    x = torch.from_numpy(x_resampled)
    y = torch.from_numpy(y_resampled)
    
    # 打印过采样后的类别分布
    unique_labels, counts = np.unique(y.numpy(), return_counts=True)
    print("过采样后的类别分布情况:")
    for label, count in zip(unique_labels, counts):
        print(f"类别 {label}: {count} 个样本")

    
    x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.2, random_state=42)
    print(f'==x_train={x_train.shape}, y_train={y_train.shape}')

    # def imshow(img, title):
    #     mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    #     std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    #     img = img * std + mean  # 反标准化
    #     img = img.numpy().transpose((1, 2, 0))  # 转换维度顺序
    #     plt.imshow(img)
    #     plt.title(title)
    #     plt.axis('off')

    # label_map = {
    #     'qd': {0: 'A', 1: 'B', 2: 'C'},
    #     'sl': {0: 'A', 1: 'B'},
    #     'zjppt': {0: 'A', 1: 'B', 2: 'C', 3: 'D'},
    #     'zyzg': {0: 'A', 1: 'B', 2: 'C', 3: 'D'}
    # }

    # plt.figure(figsize=(12, 12))
    # for i in range(4):
    #     plt.subplot(4, 1, i+1)
    #     imshow(x[i], f'Label: {label_map[task][y[i].item()]} (idx:{y[i]}) (path:{image_paths[i]})')
    # plt.tight_layout()
    # plt.show()

    train_dataset = MRIDataset(x_train, y_train)
    val_dataset = MRIDataset(x_val, y_val)

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

    # Using pretrained ResNet50
    class MRIClassifier(nn.Module):
        def __init__(self, num_classes):
            super(MRIClassifier, self).__init__()
            self.model = models.resnet50(pretrained=True)
            # Freeze some layers to speed up training
            for param in list(self.model.parameters())[:-20]:
                param.requires_grad = False
            # Modify the final fully connected layer
            num_ftrs = self.model.fc.in_features
            self.model.fc = nn.Sequential(
                nn.Linear(num_ftrs, 256),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(256, num_classes)
            )
        
        def forward(self, x):
            return self.model(x)

    # Load model checkpoint
    def load_checkpoint(model_path, model, device):
        print(f"Loading model checkpoint: {model_path}")
        try:
            checkpoint = torch.load(model_path, map_location=device)
            # Check if checkpoint is in new format
            model.load_state_dict(checkpoint['model_state_dict'])
            metadata = {k: v for k, v in checkpoint.items() if k != 'model_state_dict'}
            print(f"Model checkpoint loaded successfully! Training info: {metadata}")
            return model, metadata
        except Exception as e:
            print(f"Error loading model checkpoint: {e}")
            return None, None

    # Initialize model
    model = MRIClassifier(num_classes=num_classes).to(device)

    # Check if checkpoint exists and load it
    checkpoint_path = f'mri_classifier_model_{task}.pth'
    metadata = {'epochs_trained': 0}
    if os.path.exists(checkpoint_path):
        print(f"Found existing model checkpoint: {checkpoint_path}")
        loaded_model, loaded_metadata = load_checkpoint(checkpoint_path, model, device)
        if loaded_model is not None:
            model = loaded_model
            metadata = loaded_metadata if loaded_metadata else metadata
            print(f"Loaded model has been trained for {metadata.get('epochs_trained', 0)} epochs")
    else:
        print("No model checkpoint found, will use newly initialized model")

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)

    # Training function
    def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs=10, start_epoch=0, visualize_batches=True, visualize_frequency=10):
        best_val_loss = float('inf')
        best_model_weights = None
        best_epoch = -1
        
        for epoch in range(start_epoch, start_epoch + num_epochs):
            # 训练阶段
            model.train()
            train_loss = 0.0
            train_preds = []
            train_targets = []
            
            for batch_idx, (inputs, labels) in enumerate(train_loader):
                inputs = inputs.to(device)
                labels = labels.to(device)
                
                optimizer.zero_grad()
                
                outputs = model(inputs)
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
                    # print(f'==outputs={outputs.shape}, labels={labels.shape}')
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

            cm = confusion_matrix(val_targets, val_preds)
            # Calculate F1 score for each class
            class_f1 = f1_score(val_targets, val_preds, average=None)
            print("F1 scores for each class:")
            for i, f1 in enumerate(class_f1):
                print(f"Class {i}: {f1:.4f}")
            print("Confusion Matrix:")
            print(cm)
            
            # Update learning rate
            scheduler.step(val_loss)
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_weights = model.state_dict().copy()
                best_epoch = epoch
            
            print(f'Epoch {epoch+1}/{start_epoch + num_epochs}:')
            print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}')
            print(f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, Precision: {val_precision:.4f}, Recall: {val_recall:.4f}, F1: {val_f1:.4f}')
            print('-' * 60)
        
        # Load best model weights
        model.load_state_dict(best_model_weights)
        return model, best_epoch

    # Determine if model needs training
    total_epochs = 30
    epochs_trained = metadata.get('epochs_trained', 0)
    epochs_to_train = max(0, total_epochs - epochs_trained)

    if epochs_to_train > 0:
        print(f"Starting model training... from epoch {epochs_trained}, planning to train for {epochs_to_train} epochs")
        trained_model, best_epoch = train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, 
                                            num_epochs=epochs_to_train, start_epoch=epochs_trained)
        
        # Save model
        checkpoint = {
            'model_state_dict': trained_model.state_dict(),
            'epochs_trained': best_epoch + 1,  # Record up to the best epoch
            'task': task,
            'num_classes': num_classes,
            'date_trained': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        torch.save(checkpoint, f'mri_classifier_model_{task}.pth')
        print(f"Model saved to 'mri_classifier_model_{task}.pth', trained for {best_epoch + 1} epochs")
    else:
        print(f"Model has already been trained for {epochs_trained} epochs, no further training needed")
        trained_model = model
