import os
import json
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import glob

def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

class MRIClassifier(nn.Module): # keep the same as main.py
    def __init__(self, num_classes):
        super(MRIClassifier, self).__init__()
        self.model = models.resnet50(pretrained=True)
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
        model.load_state_dict(checkpoint['model_state_dict'])
        metadata = {k: v for k, v in checkpoint.items() if k != 'model_state_dict'}
        print(f"Model checkpoint loaded successfully! Training info: {metadata}")
        return model, metadata
    except Exception as e:
        print(f"Error loading model checkpoint: {e}")
        return None, None

# Data transformation
data_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def main():
    task = 'sl'  # can be 'qd', 'sl', 'zjppt', 'zyzg'
    
    # Set number of classes
    if task == 'qd':
        num_classes = 3
    elif task == 'sl':
        num_classes = 2
    elif task == 'zjppt' or task == 'zyzg':
        num_classes = 4
    
    TEST_IMAGE_DIR = os.path.dirname(__file__) + f"/../data/mri_images/test"
    TEST_PREDICT_PATH = os.path.dirname(__file__) + f"/../data/test_predict.json"
    checkpoint_path = os.path.dirname(__file__) + f"/mri_classifier_model_{task}.pth"
    
    model = MRIClassifier(num_classes=num_classes).to(device)
    
    loaded_model, _ = load_checkpoint(checkpoint_path, model, device)
    if loaded_model is not None:
        model = loaded_model
    else:
        print(f"Failed to load model, exiting program")
        return
    
    model.eval()
    
    try:
        with open(TEST_PREDICT_PATH, 'r', encoding='utf-8') as f:
            test_predict_data = json.load(f)
    except FileNotFoundError:
        print(f"Test prediction file not found: {TEST_PREDICT_PATH}")
        return
    except json.JSONDecodeError:
        print(f"Test prediction file format error")
        return
    
    for item in test_predict_data:
        patient_id = item['id']
        print(f"Processing patient ID: {patient_id}")
        
        if task == 'qd':
            image_pattern = os.path.join(TEST_IMAGE_DIR, patient_id, "sag", "*.png")
        elif task == 'sl':
            image_pattern = os.path.join(TEST_IMAGE_DIR, patient_id, "sag", "*.png")
        elif task == 'zjppt':
            image_pattern = os.path.join(TEST_IMAGE_DIR, patient_id, "tra", "*.png")
        elif task == 'zyzg':
            image_pattern = os.path.join(TEST_IMAGE_DIR, patient_id, "tra", "*.png")
        
        image_paths = glob.glob(image_pattern)
        
        if not image_paths:
            print(f"Warning: No images found for patient {patient_id}")
            continue
        
        predictions = []
        for img_path in image_paths:
            try:
                img = Image.open(img_path).convert('RGB')
                img_tensor = data_transforms(img).unsqueeze(0).to(device)
                
                with torch.no_grad():
                    outputs = model(img_tensor)
                    _, preds = torch.max(outputs, 1)
                    predictions.append(preds.item())
                    print(f'{preds.item()}')
            except Exception as e:
                print(f"Error processing image {img_path}: {e}")
        
        # If there are predictions
        if predictions:
            # For single-label tasks, use majority voting
            if task in ['qd', 'sl']:
                # Calculate the most common prediction
                unique, counts = np.unique(predictions, return_counts=True)
                most_common_idx = np.argmax(counts)
                final_prediction = unique[most_common_idx]
                
                # Update test prediction data
                item[task] = int(final_prediction)
            
            # For multi-label tasks, keep original format
            elif task in ['zjppt', 'zyzg']:
                # Here we assume multi-label predictions are a list
                # May need adjustment based on actual requirements
                item[task] = predictions
        
        print(f"Patient {patient_id} {task} prediction result: {item[task]}")
    
    # Save updated test prediction file
    with open(TEST_PREDICT_PATH, 'w', encoding='utf-8') as f:
        json.dump(test_predict_data, f, indent=4)
    
    print(f"Prediction results have been saved to {TEST_PREDICT_PATH}")

if __name__ == "__main__":
    main()