import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from src.model import MobileNetUNet
import torchvision.transforms as transforms
from skimage import measure
from scipy.ndimage import binary_opening, binary_closing, gaussian_filter
from skimage.measure import approximate_polygon
from scipy.interpolate import splprep, splev
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

def load_model(model_path, device):
    """Load model from checkpoint"""
    print(f"Using device: {device}")
    
    # Load model
    model = MobileNetUNet(img_ch=1, seg_ch=4, num_classes=4).to(device)
    
    # Load model weights
    checkpoint = torch.load(model_path, map_location=device)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model.eval()
    print(f"Model loaded from: {model_path}")
    return model

def load_and_preprocess_image(image_path):
    """Load and preprocess image"""
    # Load image
    img = Image.open(image_path)
    if img.mode != 'L':  # If not grayscale
        img = img.convert('L')  # Convert to grayscale

    # Resize to match training data size
    img = img.resize((256, 256))
    
    # Convert to numpy for visualization
    img_np = np.array(img)
    
    return img, img_np

def get_model_prediction(model, img, device):
    """Get model prediction for an image"""
    # Convert to tensor for model input
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])
    
    img_tensor = transform(img).unsqueeze(0).to(device)
    
    # Inference
    with torch.no_grad():
        seg_output, cls_output = model(img_tensor)
        
        # Get raw segmentation output for dice calculation
        seg_output_np = seg_output.squeeze().cpu().numpy()  # [C, H, W]
        
        # Process segmentation output for visualization
        seg_pred = torch.argmax(seg_output, dim=1).squeeze().cpu().numpy()
        
        # Process classification output
        cls_pred = torch.argmax(cls_output, dim=1).item()
        cls_probs = torch.softmax(cls_output, dim=1).squeeze().cpu().numpy()
    
    return seg_output_np, seg_pred, cls_pred, cls_probs

def create_visualization(img_np, seg_pred, cls_pred, cls_probs, output_path):
    """Create and save visualization of results"""
    # Define class names
    class_names = ['No Tumor', 'Glioma', 'Meningioma', 'Pituitary']
    
    # Create visualization
    plt.figure(figsize=(12, 10))
    
    # Display original image
    plt.imshow(img_np, cmap='gray')
    
    # Plot prediction mask and contours
    for i in range(1, 4):  # For each class (skipping background)
        pred_mask_class = (seg_pred == i)
        if np.any(pred_mask_class):
            plot_mask_and_contours(
                img_np, pred_mask_class, 
                color=(1, 0.5, 0),  # Orange
                sigma=0.2, 
                threshold=0.7,
                tolerance=0.1, 
                opacity=0.2, 
                line_alpha=0.4
            )
    
    # Add legend
    legend_elements = [
        Patch(facecolor='orange', alpha=0.5, label='Prediction (Fill)'),
        Line2D([0], [0], color='orange', linestyle='--', lw=1, alpha=0.9, label='Prediction (Contour)')
    ]
    plt.legend(handles=legend_elements, loc='upper right')
    
    # Set title with classification results
    plt.title(f'Class: {class_names[cls_pred]} ({cls_probs[cls_pred]*100:.1f}%)', fontsize=14)
    
    plt.axis('off')
    
    # Save and show result
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

def compare_predictions_with_gt_folder(model_path, mask_folder, output_base_folder, device='cuda'):
    """
    Compare model predictions with ground truth masks for all images in a folder.
    
    Args:
        model_path: Path to the saved model (.pt file)
        mask_folder: Path to the folder containing ground truth masks
        output_base_folder: Path to the base folder to save results
        device: Device to run inference on
    """
    # Set device
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    
    # Load model
    model = load_model(model_path, device)
    
    # Iterate through mask folder
    for mask_filename in os.listdir(mask_folder):
        if mask_filename.endswith('.png'):
            mask_path = os.path.join(mask_folder, mask_filename)
            
            # Load and preprocess image
            img, img_np = load_and_preprocess_image(mask_path)
            
            # Get model prediction
            seg_output_np, seg_pred, cls_pred, cls_probs = get_model_prediction(model, img, device)
            
            # Create output directory for the predicted class
            class_names = ['No Tumor', 'Glioma', 'Meningioma', 'Pituitary']
            output_dir = os.path.join(output_base_folder, class_names[cls_pred])
            os.makedirs(output_dir, exist_ok=True)
            
            # Create visualization
            output_path = os.path.join(output_dir, mask_filename.replace('.png', '_result.png'))
            create_visualization(img_np, seg_pred, cls_pred, cls_probs, output_path)
            
            print(f"Processed {mask_filename}: Predicted class {class_names[cls_pred]}")

if __name__ == "__main__":
    # Path to your trained model
    model_path = "checkpoint_new/best_model.pt"
    
    # Path to test ground truth mask folder
    mask_folder = "test/masks"
    
    # Output base folder
    output_base_folder = "results"
    
    # Run comparison for all images in the folder
    compare_predictions_with_gt_folder(model_path, mask_folder, output_base_folder)