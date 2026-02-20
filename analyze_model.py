import torch
import torch.nn as nn
from torchvision import transforms
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from PIL import Image
import os
from train_fruit_classifier import FruitClassifier, FruitDataset
from sklearn.metrics import confusion_matrix
from collections import defaultdict
import random

def load_best_model(model_path='best_fruit_classifier.pth'):
    # Load model architecture
    model = FruitClassifier(num_classes=10)
    
    # Load the saved state
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Move to GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()
    
    return model, device

def visualize_predictions(model, dataset, num_samples=10):
    # Create figure
    fig = plt.figure(figsize=(20, 10))
    
    # Get random samples
    indices = random.sample(range(len(dataset)), num_samples)
    
    for idx, i in enumerate(indices):
        # Get image and label
        image, label = dataset[i]
        
        # Get prediction
        with torch.no_grad():
            output = model(image.unsqueeze(0).to(model.resnet.fc[-1].weight.device))
            pred = output.argmax(dim=1).item()
            probs = torch.softmax(output, dim=1)[0]
        
        # Convert tensor to image
        img = image.permute(1, 2, 0).numpy()
        img = (img * [0.229, 0.224, 0.225] + [0.485, 0.456, 0.406]).clip(0, 1)
        
        # Plot
        ax = fig.add_subplot(2, 5, idx + 1)
        ax.imshow(img)
        color = 'green' if pred == label else 'red'
        title = f'True: {dataset.classes[label]}\nPred: {dataset.classes[pred]}\nConf: {probs[pred]:.2f}'
        ax.set_title(title, color=color)
        ax.axis('off')
    
    plt.tight_layout()
    plt.savefig('prediction_samples.png')
    plt.close()

def analyze_confidence(model, dataset):
    all_confidences = []
    correct_confidences = []
    incorrect_confidences = []
    
    for i in range(len(dataset)):
        image, label = dataset[i]
        
        with torch.no_grad():
            output = model(image.unsqueeze(0).to(model.resnet.fc[-1].weight.device))
            pred = output.argmax(dim=1).item()
            prob = torch.softmax(output, dim=1)[0][pred].item()
        
        all_confidences.append(prob)
        if pred == label:
            correct_confidences.append(prob)
        else:
            incorrect_confidences.append(prob)
    
    # Plot confidence distributions
    plt.figure(figsize=(10, 6))
    plt.hist(correct_confidences, alpha=0.5, label='Correct Predictions', bins=50)
    plt.hist(incorrect_confidences, alpha=0.5, label='Incorrect Predictions', bins=50)
    plt.xlabel('Confidence')
    plt.ylabel('Count')
    plt.title('Distribution of Model Confidence')
    plt.legend()
    plt.savefig('confidence_distribution.png')
    plt.close()
    
    return np.mean(correct_confidences), np.mean(incorrect_confidences)

def analyze_errors(model, dataset):
    all_preds = []
    all_labels = []
    errors = []
    
    for i in range(len(dataset)):
        image, label = dataset[i]
        
        with torch.no_grad():
            output = model(image.unsqueeze(0).to(model.resnet.fc[-1].weight.device))
            pred = output.argmax(dim=1).item()
            prob = torch.softmax(output, dim=1)[0][pred].item()
        
        all_preds.append(pred)
        all_labels.append(label)
        
        if pred != label:
            errors.append({
                'index': i,
                'true': dataset.classes[label],
                'pred': dataset.classes[pred],
                'confidence': prob
            })
    
    # Plot confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=dataset.classes, yticklabels=dataset.classes)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('confusion_matrix.png')
    plt.close()
    
    return errors

def main():
    # Load the model
    model, device = load_best_model()
    print(f"Model loaded successfully. Using device: {device}")
    
    # Create validation dataset
    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    test_dir = r'C:\Users\aayus\Downloads\archive\MY_data\test'
    print(f"\nLoading test data from: {test_dir}")
    
    val_dataset = FruitDataset(
        root_dir=test_dir,
        transform=val_transform
    )
    
    print("\nAnalyzing model predictions...")
    
    # Calculate per-class and overall accuracy
    class_correct = defaultdict(int)
    class_total = defaultdict(int)
    all_preds = []
    all_labels = []
    
    for i in range(len(val_dataset)):
        image, label = val_dataset[i]
        class_name = val_dataset.classes[label]
        
        with torch.no_grad():
            output = model(image.unsqueeze(0).to(device))
            pred = output.argmax(dim=1).item()
            all_preds.append(pred)
            all_labels.append(label)
            
            if pred == label:
                class_correct[class_name] += 1
            class_total[class_name] += 1
    
    # Print per-class accuracy
    print("\nPer-class Accuracy:")
    total_correct = 0
    total_samples = 0
    
    for class_name in sorted(class_total.keys()):
        accuracy = 100 * class_correct[class_name] / class_total[class_name]
        print(f"{class_name}: {accuracy:.2f}% ({class_correct[class_name]}/{class_total[class_name]})")
        total_correct += class_correct[class_name]
        total_samples += class_total[class_name]
    
    # Print overall accuracy
    overall_accuracy = 100 * total_correct / total_samples
    print(f"\nOverall Test Accuracy: {overall_accuracy:.2f}%")
    
    # Visualize sample predictions
    print("\nGenerating prediction visualizations...")
    visualize_predictions(model, val_dataset)
    
    # Analyze confidence distributions
    print("\nAnalyzing confidence distributions...")
    correct_conf, incorrect_conf = analyze_confidence(model, val_dataset)
    print(f"Average confidence for correct predictions: {correct_conf:.3f}")
    print(f"Average confidence for incorrect predictions: {incorrect_conf:.3f}")
    
    # Analyze errors
    print("\nAnalyzing error cases...")
    errors = analyze_errors(model, val_dataset)
    print(f"\nFound {len(errors)} errors in test set")
    print("\nTop 10 most confident errors:")
    for err in sorted(errors, key=lambda x: x['confidence'], reverse=True)[:10]:
        print(f"True: {err['true']}, Predicted: {err['pred']}, Confidence: {err['confidence']:.3f}")

if __name__ == '__main__':
    main() 