import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import cv2
import numpy as np
from skimage.segmentation import slic
from skimage.segmentation import mark_boundaries
from train_fruit_classifier import FruitClassifier

class FruitSegmentationClassifier:
    def __init__(self, model_path='best_fruit_classifier.pth'):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize model
        self.model = FruitClassifier(num_classes=10)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()
        
        # Define classes
        self.classes = ['apple', 'banana', 'kiwi', 'orange', 'watermelon', 
                       'strawberry', 'cherry', 'avocado', 'mango', 'pineapple']
        
        # Setup image transformation
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def segment_image(self, image_path, n_segments=100):
        """
        Segment the image using SLIC
        """
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Could not read image at path: {image_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Apply SLIC segmentation
        segments = slic(image, n_segments=n_segments, compactness=10)
        
        # Create segmentation visualization
        segmented_image = mark_boundaries(image, segments)
        segmented_image = (segmented_image * 255).astype(np.uint8)
        
        return image, segments, segmented_image

    def classify_segment(self, segment_image):
        """
        Classify a single segment
        """
        # Convert to PIL Image
        segment_pil = Image.fromarray(segment_image)
        
        # Transform the image
        input_tensor = self.transform(segment_pil)
        input_batch = input_tensor.unsqueeze(0).to(self.device)
        
        # Get prediction
        with torch.no_grad():
            output = self.model(input_batch)
            probabilities = torch.nn.functional.softmax(output[0], dim=0)
            top_prob, top_class = torch.topk(probabilities, 1)
            
        return {
            'class': self.classes[top_class[0].item()],
            'probability': float(top_prob[0])
        }

    def process_image(self, image_path, confidence_threshold=0.7):
        """
        Process an image: segment it and classify each segment
        """
        # Segment the image
        original_image, segments, segmented_image = self.segment_image(image_path)
        
        # Classify each segment
        unique_segments = np.unique(segments)
        results = {}
        
        for segment_id in unique_segments:
            # Create a mask for the current segment
            mask = segments == segment_id
            
            # Extract the segment
            segment_image = original_image.copy()
            for c in range(3):
                segment_image[:,:,c] = segment_image[:,:,c] * mask
            
            # Classify the segment
            classification = self.classify_segment(segment_image)
            
            # Store results if confidence is above threshold
            if classification['probability'] > confidence_threshold:
                results[segment_id] = classification
        
        return original_image, segments, segmented_image, results

def main():
    # Initialize classifier
    classifier = FruitSegmentationClassifier()
    
    # Process an image
    image_path = "path/to/test/image.jpg"  # Replace with your test image
    _, _, segmented_image, results = classifier.process_image(image_path)
    
    # Save segmented image
    cv2.imwrite('segmented_output.jpg', cv2.cvtColor(segmented_image, cv2.COLOR_RGB2BGR))
    
    # Print results
    print("\nClassification Results:")
    for segment_id, result in results.items():
        print(f"Segment {segment_id}: {result['class']} ({result['probability']:.2f})")

if __name__ == "__main__":
    main() 