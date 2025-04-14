import os
import torch
from torchvision import transforms
from PIL import Image
from diffusers import AutoencoderKL
from tqdm import tqdm

# Load the VAE model of Stable Diffusion 2.1
vae = AutoencoderKL.from_pretrained("stabilityai/stable-diffusion-2-1-base", subfolder="vae").cuda()
vae.eval()

# Define the dataset directory and target directory
DATASET_DIR = "/your/path/DL3DV-ALL-960P"
TARGET_DIR = "/your/path/DL3DV-ALL-960P-latents"

# Define image preprocessing
preprocess = transforms.Compose([
    transforms.CenterCrop(540),         # Center crop to square (DL3DV images are 940x540)
    transforms.Resize((576, 576)),      # Resize to 512x512
    transforms.ToTensor(),              # Convert to tensor
    transforms.Normalize([0.5], [0.5])  # Normalize to [-1, 1]
])

# Function to process images and save latents
def process_and_save_latents(scene_path, target_scene_path):
    os.makedirs(target_scene_path, exist_ok=True)
    images_dir = os.path.join(scene_path, "images_4")
    target_images_dir = os.path.join(target_scene_path, "images_4")
    os.makedirs(target_images_dir, exist_ok=True)

    image_names = [name for name in os.listdir(images_dir) if name.lower().endswith(('.png', '.jpg', '.jpeg'))]
    batch_size = 32

    for i in range(0, len(image_names), batch_size):
        batch_names = image_names[i:i + batch_size]
        batch_images = []

        # Load and preprocess images in the batch
        for image_name in batch_names:
            image_path = os.path.join(images_dir, image_name)
            image = Image.open(image_path).convert("RGB")
            image_tensor = preprocess(image)
            batch_images.append(image_tensor)

        # Stack images into a batch tensor
        batch_tensor = torch.stack(batch_images).cuda()

        # Encode the batch to latents
        with torch.no_grad():
            latents = vae.encode(batch_tensor).latent_dist.sample()
            latents = latents.cpu()  # Move to CPU

        # Save each latent in the batch
        for j, image_name in enumerate(batch_names):
            latent_path = os.path.join(target_images_dir, f"{os.path.splitext(image_name)[0]}.pt")
            torch.save(latents[j], latent_path)

# Iterate through the dataset and process each scene
for subfolder_name in tqdm(os.listdir(DATASET_DIR), desc="Processing subfolders"):
    subfolder_path = os.path.join(DATASET_DIR, subfolder_name)
    if os.path.isdir(subfolder_path):
        for scene_name in tqdm(os.listdir(subfolder_path), desc=f"Processing scenes in {subfolder_name}", leave=False):
            scene_path = os.path.join(subfolder_path, scene_name)
            if os.path.isdir(scene_path):
                target_scene_path = os.path.join(TARGET_DIR, subfolder_name, scene_name)
                process_and_save_latents(scene_path, target_scene_path)

print("Processing complete. Latents saved.")