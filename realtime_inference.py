import cv2
import torch
import numpy as np
from torchvision import transforms
from train_fruit_classifier import FruitClassifier
import time

def load_model(model_path=r'D:\code\best_fruit_classifier.pth'):
    # Load model architecture and determine target device
    model = FruitClassifier(num_classes=10)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load the saved state while mapping tensors to the correct device
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Move to GPU if available
    model = model.to(device)
    model.eval()
    
    return model, device

def preprocess_frame(frame):
    # Convert BGR to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Create transform pipeline
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # Apply transformations
    input_tensor = transform(frame_rgb)
    
    return input_tensor

def main():
    # Load the model
    model, device = load_model()
    print(f"Model loaded successfully. Using device: {device}")
    
    # Get class names
    class_names = ['Apple', 'Banana', 'avocado', 'cherry', 'kiwi', 
                   'mango', 'orange', 'pineapple', 'strawberries', 'watermelon']
    
    # Initialize webcam
    cap = cv2.VideoCapture(0)
    
    # Set frame size
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    print("\nPress 'q' to quit")
    
    # For FPS calculation
    fps_time = time.time()
    fps_frames = 0
    fps = 0
    
    while True:
        # Read frame
        ret, frame = cap.read()
        if not ret:
            break
        
        # Preprocess frame
        input_tensor = preprocess_frame(frame)
        
        # Get prediction
        with torch.no_grad():
            input_batch = input_tensor.unsqueeze(0).to(device)
            output = model(input_batch)
            probabilities = torch.softmax(output, dim=1)[0]
            pred_idx = output.argmax(dim=1).item()
            confidence = probabilities[pred_idx].item()
        
        # Calculate FPS
        fps_frames += 1
        if time.time() - fps_time >= 1.0:
            fps = fps_frames
            fps_frames = 0
            fps_time = time.time()
        
        # Draw prediction
        prediction = class_names[pred_idx]
        text = f"{prediction}: {confidence:.2f}"
        cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, f"FPS: {fps}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Show frame
        cv2.imshow('Fruit Classifier', frame)
        
        # Check for quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main() 
