import os
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from diffusers import StableDiffusionPipeline
from torch.utils.tensorboard import SummaryWriter
import torch.nn as nn
import torch.optim as optim

# Define Dataset class for preprocessed images
class PreprocessedImageDataset(Dataset):
    def __init__(self, image_folder, transform=None):
        self.image_folder = image_folder
        self.image_files = [f for f in os.listdir(image_folder) if f.endswith(('png', 'jpg', 'jpeg'))]
        self.transform = transform

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_folder, self.image_files[idx])
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image

# Load the latent diffusion model
model_id = "CompVis/ldm-text2img-large-256"
pipeline = StableDiffusionPipeline.from_pretrained(model_id)

# Set up the device (MPS for Mac, GPU, or CPU)
device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
pipeline.to(device)

# Image transformation (same as preprocessing)
transform = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

# Load the preprocessed dataset
dataset = PreprocessedImageDataset(image_folder="preprocessed_images", transform=transform)
dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

# Set up TensorBoard writer
writer = SummaryWriter('runs/CREAITVE_model')

# Define a simple MSE loss function and Adam optimizer
loss_fn = nn.MSELoss()
optimizer = optim.Adam(pipeline.parameters(), lr=1e-5)

# Training loop (with loss function and backpropagation)
num_epochs = 5
for epoch in range(num_epochs):
    epoch_loss = 0.0
    for images in dataloader:
        images = images.to(device)
        
        # Zero the gradients
        optimizer.zero_grad()

        # Define a text prompt
        prompt = "A beautiful landscape painting of mountains"
        
        # Generate an image using the model
        generated_images = pipeline(prompt, images).images
        
        # For simplicity, assume we want to match the generated images to the input images
        # In a real training scenario, this could be more complex
        loss = loss_fn(generated_images, images)
        loss.backward()  # Backpropagate the loss
        optimizer.step()  # Update model parameters
        
        # Accumulate the loss for logging
        epoch_loss += loss.item()

        # Log the first generated image and loss to TensorBoard
        writer.add_image(f'Epoch_{epoch}_generated_image', generated_images[0], epoch)
        writer.add_scalar('Loss/train', epoch_loss, epoch)

    # Save model checkpoint at the end of each epoch
    pipeline.save_pretrained(f'checkpoints/CREAITVE_model_epoch_{epoch}')
    print(f"Epoch {epoch}: Checkpoint saved, Loss: {epoch_loss}")

# Close TensorBoard writer after training
writer.close()