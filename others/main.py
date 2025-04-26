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
from sklearn.metrics import confusion_matrix

def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
task = 'zyzg' # qd, sl, ajppt, zyzg
if task == 'qd':
    num_classes = 3
elif task == 'sl':
    num_classes = 2
elif task == 'zjppt':
    num_classes = 4
elif task == 'zyzg':
    num_classes = 4


# Data paths
TRAIN_IMAGE_DIR = os.path.dirname(__file__)+f"/../data/mri_images/train"
TRAIN_LABEL_PATH =  os.path.dirname(__file__)+f"/../data/dataset/{task}_train.json"

# Load label data
def load_labels(label_path):
    with open(label_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    labels = []
    id_to_index = {}  # Mapping from image_id to index
    index_to_id = {}  # Mapping from index to image_id 
    image_ids = []    # Store all image_ids
    
    for idx, item in enumerate(data):
        # print(f'==item={item}')
        image_id = item.get('id', '')
        answer = item.get('answer', '')
        image_paths = item.get('image', [])
        
        # Extract type info (sag or tra) and image number from image path
        image_type = "unknown"
        image_number = "0"
        if image_paths and len(image_paths) > 0:
            path = image_paths[0]
            # Extract type info
            if "/sag/" in path or "\\sag\\" in path:
                image_type = "sag"
            elif "/tra/" in path or "\\tra\\" in path:
                image_type = "tra"
            
            # Extract image number
            import re
            match = re.search(r'/(sag|tra)/(\d+)\.png', path)
            if match:
                image_number = match.group(2)
        
        # Create unique identifier combining ID, type and image number
        unique_id = f"{image_id}_{image_type}_{image_number}"
        
        # Convert labels to numbers
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
        elif task == 'zjppt':
            if "A" in answer:
                labels.append(0)
            elif "B" in answer:
                labels.append(1)
            elif "C" in answer:
                labels.append(2)
            elif "D" in answer:
                labels.append(3)
        elif task == 'zyzg':
            if "A" in answer:
                labels.append(0)
            elif "B" in answer:
                labels.append(1)
            elif "C" in answer:
                labels.append(2)
            elif "D" in answer:
                labels.append(3)
            
        # Establish bidirectional mapping between image_id and index
        id_to_index[unique_id] = idx
        index_to_id[idx] = unique_id
        image_ids.append(unique_id)
        
        # print(f'image_id={unique_id}\nanswer={answer}\nindex={idx}\n')
    
    # # Check for duplicate image_ids
    # unique_ids = set(image_ids)
    # if len(unique_ids) != len(image_ids):
    #     print(f"Warning: Found duplicate image_ids! Total: {len(image_ids)}, Unique: {len(unique_ids)}")
    #     # Further analyze duplicates
    #     from collections import Counter
    #     id_counts = Counter(image_ids)
    #     duplicates = {id: count for id, count in id_counts.items() if count > 1}
    #     print(f"Duplicate image_ids: {duplicates}")
    
    return labels, id_to_index, index_to_id, image_ids


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
        
        parts = image_id.split('_')
        if len(parts) >= 3:
            patient_id = parts[0]
            image_type = parts[1]  
            image_number = parts[2]
            image_path = os.path.join(self.image_dir, patient_id, image_type, f"{image_number}.png")
        
        # Ensure file exists
        if not os.path.exists(image_path):
            print(f"Warning: Image does not exist {image_path}")
            # Return a blank image and label
            image = Image.new('RGB', (224, 224), color='black')
            label = self.labels[image_id]
            if self.transform:
                image = self.transform(image)
            return image, label
        
        # Load image
        image = Image.open(image_path).convert('RGB')
        label = self.labels[image_id]
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

# 定义数据转换 TODO: check这一点，以及看看是否要做过采样（数据不平衡）
data_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

labels, id_to_index, index_to_id, image_ids = load_labels(TRAIN_LABEL_PATH)
print(f"Loaded {len(labels)} labels")
labels = np.array(labels)
# Count number of samples for each class
unique_labels, counts = np.unique(labels, return_counts=True)
print("\nSample count statistics for each class:")
for label, count in zip(unique_labels, counts):
    print(f"Class {label}: {count} samples")

train_indices, val_indices = train_test_split(range(len(labels)), test_size=0.2, shuffle=True)
train_indices = np.array(train_indices)
val_indices = np.array(val_indices)
print(f'train_indices.shape={train_indices.shape}, val_indices.shape={val_indices.shape}')

train_id_label_dict = {index_to_id[idx]: labels[idx] for idx in train_indices}
val_id_label_dict = {index_to_id[idx]: labels[idx] for idx in val_indices}

print(f'Number of training samples: {len(train_id_label_dict)}, Number of validation samples: {len(val_id_label_dict)}')

train_dataset = MRIDataset(TRAIN_IMAGE_DIR, train_id_label_dict, transform=data_transforms)
val_dataset = MRIDataset(TRAIN_IMAGE_DIR, val_id_label_dict, transform=data_transforms)
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
def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs=10, start_epoch=0):
    best_val_loss = float('inf')
    best_model_weights = None
    best_epoch = -1
    
    for epoch in range(start_epoch, start_epoch + num_epochs):
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
total_epochs = 15
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


# Evaluate model
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
    
    # Calculate and print confusion matrix
    cm = confusion_matrix(all_targets, all_preds)
    print("Confusion Matrix:")
    print(cm)
    
    # Print normalized confusion matrix (row-normalized, showing recall distribution for each class)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    print("Normalized Confusion Matrix (by row):")
    print(np.round(cm_normalized, 2))
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'confusion_matrix': cm
    }

# Evaluate model on validation set
print("Evaluating model on validation set...")
eval_metrics = evaluate_model(trained_model, val_loader)
print(f"Validation Set Results:")
print(f"Accuracy: {eval_metrics['accuracy']:.4f}")
print(f"Precision: {eval_metrics['precision']:.4f}")
print(f"Recall: {eval_metrics['recall']:.4f}")
print(f"F1 Score: {eval_metrics['f1']:.4f}")
