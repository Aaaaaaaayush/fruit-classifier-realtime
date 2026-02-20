import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
import cv2
import numpy as np
from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
import time
from datetime import datetime, timedelta

class FruitDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        """
        Args:
            root_dir (string): Directory with all the fruit images organized in subdirectories
            transform (callable, optional): Optional transform to be applied on a sample
        """
        self.root_dir = root_dir
        self.transform = transform
        self.classes = sorted(['Apple', 'Banana', 'avocado', 'cherry', 'kiwi', 
                             'mango', 'orange', 'pineapple', 'strawberries', 'watermelon'])
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(self.classes)}
        
        self.samples = []
        # Walk through all subdirectories
        for class_name in self.classes:
            class_dir = os.path.join(root_dir, class_name)
            if not os.path.isdir(class_dir):
                print(f"Warning: Directory {class_dir} not found")
                continue
            for img_name in os.listdir(class_dir):
                if img_name.endswith(('.jpg', '.jpeg', '.png')):
                    img_path = os.path.join(class_dir, img_name)
                    self.samples.append((img_path, self.class_to_idx[class_name]))
        
        print(f"Found {len(self.samples)} images across {len(self.classes)} classes")
        # Print class distribution
        class_counts = {}
        for _, class_idx in self.samples:
            class_name = self.classes[class_idx]
            class_counts[class_name] = class_counts.get(class_name, 0) + 1
        print("\nClass distribution:")
        for class_name, count in class_counts.items():
            print(f"{class_name}: {count} images")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

class FruitClassifier(nn.Module):
    def __init__(self, num_classes=10):
        super(FruitClassifier, self).__init__()
        # Load pre-trained ResNet50
        self.resnet = models.resnet50(weights='DEFAULT')
        
        # Freeze all layers except the last few
        for param in list(self.resnet.parameters())[:-20]:
            param.requires_grad = False
            
        # Replace the last fully connected layer
        num_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Sequential(
            nn.Linear(num_features, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        return self.resnet(x)

def plot_confusion_matrix(y_true, y_pred, classes):
    """Plot confusion matrix"""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=classes, yticklabels=classes)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f'confusion_matrix_epoch_{time.strftime("%Y%m%d_%H%M%S")}.png')
    plt.close()

def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=10, device='cuda', train_epoch_fn=None, scaler=None):
    best_acc = 0.0
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []
    start_time = time.time()
    
    # Get class names from the train loader
    class_names = train_loader.dataset.classes
    
    # Get available classes in validation set and create label mapping
    val_classes = set()
    for _, label in val_loader.dataset.samples:
        val_classes.add(val_loader.dataset.classes[label])
    val_class_names = [c for c in class_names if c in val_classes]
    val_class_to_idx = {c: i for i, c in enumerate(val_class_names)}
    orig_to_val_label = {i: val_class_to_idx[c] for i, c in enumerate(class_names) if c in val_classes}
    
    # Print validation class mapping
    print("\nValidation Class Mapping:")
    for orig_idx, val_idx in orig_to_val_label.items():
        print(f"{class_names[orig_idx]} -> {val_class_names[val_idx]}")
    
    for epoch in range(num_epochs):
        epoch_start = time.time()
        print(f'\nEpoch {epoch+1}/{num_epochs}')
        print('-' * 30)
        
        # Training phase
        model.train()
        running_loss = 0.0
        running_corrects = 0
        all_labels = []
        all_preds = []
        batch_times = []
        
        if train_epoch_fn:
            running_loss, running_corrects, all_labels, all_preds = train_epoch_fn(
                model, train_loader, optimizer, criterion, device, scaler
            )
        else:
            for batch_idx, (inputs, labels) in enumerate(tqdm(train_loader, desc='Training')):
                batch_start = time.time()
                
                inputs = inputs.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                
                optimizer.zero_grad()
                
                with torch.cuda.amp.autocast():
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)
                
                if scaler is not None:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()
                
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                
                all_labels.extend(labels.cpu().numpy())
                all_preds.extend(preds.cpu().numpy())
                
                batch_time = time.time() - batch_start
                batch_times.append(batch_time)
                
                if (batch_idx + 1) % 10 == 0:
                    avg_batch_time = np.mean(batch_times[-10:])
                    batches_left = len(train_loader) - (batch_idx + 1)
                    eta = timedelta(seconds=int(avg_batch_time * batches_left))
                    print(f'\nBatch {batch_idx + 1}/{len(train_loader)}:')
                    print(f'Batch Loss: {loss.item():.4f}')
                    print(f'Batch Accuracy: {torch.sum(preds == labels.data).item() / len(labels):.4f}')
                    print(f'Average Batch Time: {avg_batch_time:.2f}s')
                    print(f'ETA: {eta}')
        
        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = running_corrects.double() / len(train_loader.dataset)
        train_losses.append(epoch_loss)
        train_accs.append(epoch_acc.cpu().item())
        
        # Print training metrics
        print(f'\nTraining Metrics:')
        print(f'Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
        print('\nPer-class Training Report:')
        print(classification_report(all_labels, all_preds, target_names=class_names, zero_division=0))
        
        # Plot training confusion matrix
        plot_confusion_matrix(all_labels, all_preds, class_names)
        
        # Validation phase
        model.eval()
        running_loss = 0.0
        running_corrects = 0
        all_val_labels = []
        all_val_preds = []
        
        print('\nValidation Phase:')
        with torch.no_grad():
            with torch.cuda.amp.autocast():
                for inputs, labels in tqdm(val_loader, desc='Validation'):
                    inputs = inputs.to(device, non_blocking=True)
                    labels = labels.to(device, non_blocking=True)
                    
                    outputs = model(inputs)
                    # Zero out logits for classes not in validation set
                    for i in range(len(class_names)):
                        if i not in orig_to_val_label:
                            outputs[:, i] = float('-inf')
                    
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)
                    
                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)
                    
                    # Map labels to validation subset
                    val_labels = [orig_to_val_label[l.item()] for l in labels]
                    val_preds = [orig_to_val_label[p.item()] for p in preds]
                    all_val_labels.extend(val_labels)
                    all_val_preds.extend(val_preds)
        
        val_loss = running_loss / len(val_loader.dataset)
        val_acc = running_corrects.double() / len(val_loader.dataset)
        val_losses.append(val_loss)
        val_accs.append(val_acc.cpu().item())
        
        # Print validation metrics
        print(f'\nValidation Metrics:')
        print(f'Loss: {val_loss:.4f} Acc: {val_acc:.4f}')
        print('\nPer-class Validation Report:')
        print(classification_report(all_val_labels, all_val_preds, target_names=val_class_names, zero_division=0))
        
        # Plot validation confusion matrix
        plot_confusion_matrix(all_val_labels, all_val_preds, val_class_names)
        
        # Save best model
        if val_acc > best_acc:
            best_acc = val_acc
            print(f'\nSaving best model with validation accuracy: {val_acc:.4f}')
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': epoch_loss,
                'val_loss': val_loss,
                'train_acc': epoch_acc.cpu().item(),
                'val_acc': val_acc.cpu().item(),
            }, 'best_fruit_classifier.pth')
        
        # Print epoch summary
        epoch_time = time.time() - epoch_start
        total_time = time.time() - start_time
        avg_epoch_time = total_time / (epoch + 1)
        epochs_left = num_epochs - (epoch + 1)
        eta = timedelta(seconds=int(avg_epoch_time * epochs_left))
        
        print(f'\nEpoch Summary:')
        print(f'Time taken: {timedelta(seconds=int(epoch_time))}')
        print(f'Total training time: {timedelta(seconds=int(total_time))}')
        print(f'ETA for completion: {eta}')
        print('-' * 60)
    
    # Plot training history
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.title('Loss History')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(train_accs, label='Train Acc')
    plt.plot(val_accs, label='Val Acc')
    plt.title('Accuracy History')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('training_history.png')
    plt.close()

def main():
    # Set device and enable CUDA optimizations
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    else:
        print("Using CPU")
    
    print(f'Using device: {device}')
    
    # Data augmentation and normalization for training
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # Just normalization for validation/testing
    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # Create datasets with correct paths
    train_dataset = FruitDataset(root_dir=r'C:\Users\aayus\Downloads\archive\MY_data\train', transform=train_transform)
    val_dataset = FruitDataset(root_dir=r'C:\Users\aayus\Downloads\archive\MY_data\test', transform=val_transform)
    
    print(f"Number of training samples: {len(train_dataset)}")
    print(f"Number of validation samples: {len(val_dataset)}")
    
    # Optimize batch size for GPU
    batch_size = 64  # Increased for RTX 4050
    num_workers = 4   # Parallel data loading
    
    # Create data loaders with pinned memory for faster GPU transfer
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    # Create model and move to GPU
    model = FruitClassifier(num_classes=10)
    model = model.to(device)
    
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs!")
        model = nn.DataParallel(model)
    
    # Loss function and optimizer with gradient scaler for mixed precision
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
    
    # Enable automatic mixed precision for faster training
    scaler = torch.cuda.amp.GradScaler()
    
    # Train the model with mixed precision
    def train_epoch(model, loader, optimizer, criterion, device, scaler):
        model.train()
        running_loss = 0.0
        running_corrects = 0
        all_labels = []
        all_preds = []
        
        for inputs, labels in tqdm(loader, desc='Training'):
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            
            optimizer.zero_grad()
            
            # Runs the forward pass with autocasting
            with torch.cuda.amp.autocast():
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)
            
            # Scales loss and calls backward() to create scaled gradients
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)
            
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
        
        return running_loss, running_corrects, all_labels, all_preds
    
    # Train the model
    train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=20, device=device, train_epoch_fn=train_epoch, scaler=scaler)

if __name__ == '__main__':
    main() 