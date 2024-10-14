import os
from PIL import Image
import torchvision.transforms as transforms

# Define paths
image_folder = 'image_dataset'  # Folder containing your images
output_folder = 'preprocessed_images'  # Folder to save processed images

# Ensure the output folder exists
os.makedirs(output_folder, exist_ok=True)

# Define transformations: Resizing, augmentation, and normalization
transform = transforms.Compose([
    transforms.Resize((512, 512)),  # Resize to a fixed size for consistency
    transforms.RandomHorizontalFlip(p=0.5),  # Random horizontal flip with 50% probability
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),  # Color jitter for robustness
    transforms.ToTensor(),  # Convert to Tensor
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])  # Normalize to [-1, 1] for training
])

# Additional transform to convert tensor back to image (if needed for saving as an image)
to_pil = transforms.ToPILImage()

# Process each image
for image_name in os.listdir(image_folder):
    if image_name.endswith(('.png', '.jpg', '.jpeg')):
        image_path = os.path.join(image_folder, image_name)
        image = Image.open(image_path).convert('RGB')  # Open and convert to RGB
        
        transformed_tensor = transform(image)  # Apply transformations
        
        # Convert back to a PIL image for saving (optional for debugging or verification)
        transformed_image = to_pil(transformed_tensor)
        
        # Save the preprocessed image
        output_image_path = os.path.join(output_folder, image_name)
        transformed_image.save(output_image_path)
        print(f"Saved preprocessed image: {output_image_path}")