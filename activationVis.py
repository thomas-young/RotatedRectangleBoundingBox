import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO
from PIL import Image
import os

# Create directory for saving GIFs
output_dir = "./ACTIVATION"
os.makedirs(output_dir, exist_ok=True)

# Load your YOLOv8 model
model = YOLO('./runs/obb/train7/weights/best.pt').model
print("Model loaded")

# Check for MPS device
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
model.to(device)

# Function to normalize the image
def normalize(image):
    image = torch.tensor(image, dtype=torch.float32)  # Ensure float32
    return (image - image.min()) / (image.max() - image.min())

# Total Variation Loss for smoothness
def total_variation_loss(img):
    tv_loss = torch.sum(torch.abs(img[:, :, :-1] - img[:, :, 1:])) + torch.sum(torch.abs(img[:, :-1, :] - img[:, 1:, :]))
    return tv_loss.float()  # Ensure float32

# Custom Gaussian smoothing function
def gaussian_smoothing(img, kernel_size=3, sigma=0.5):
    channels = img.shape[1]
    kernel = torch.tensor(
        [[(1 / (2 * np.pi * sigma ** 2)) * np.exp(-((x - kernel_size // 2) ** 2 + (y - kernel_size // 2) ** 2) / (2 * sigma ** 2))
          for x in range(kernel_size)] for y in range(kernel_size)],
        dtype=torch.float32  # Ensure float32
    )
    kernel = kernel / torch.sum(kernel)
    kernel = kernel.view(1, 1, kernel_size, kernel_size)
    kernel = kernel.repeat(channels, 1, 1, 1)

    padding = kernel_size // 2
    img = F.conv2d(img, kernel.to(img.device), padding=padding, groups=channels)
    return img

# Function to visualize the activation maximization with regularization
def visualize_activation(model, target_layer, target_index, target_class, input_size=(224, 224), lr=0.0001, iterations=200, reg_lambda=1.0, smooth_kernel=5, smooth_sigma=0.9, smooth_interval=3):
    global activations
    activations = []  # Clear previous activations

    # Hook function to capture activations
    def hook_fn(module, input, output):
        if isinstance(output, tuple):
            activations.append(output[0].float())  # Ensure float32
        else:
            activations.append(output.float())

    # Register hook to the desired layer
    hook = target_layer.register_forward_hook(hook_fn)

    # Create a random image with slight noise
    input_image = torch.randn(1, 3, input_size[0], input_size[1], device=device, dtype=torch.float32) * 0.01
    input_image.requires_grad_()  # Make sure input_image is a leaf tensor
    optimizer = torch.optim.Adam([input_image], lr=lr)

    torch.autograd.set_detect_anomaly(True)  # Enable anomaly detection

    frames = []  # List to store frames for the GIF

    for i in range(iterations):
        optimizer.zero_grad()
        
        # Forward pass
        activations = []  # Clear previous activations before each forward pass
        model(input_image)
        
        # Get the activation from the hooked layer
        output_tensor = activations[0]  # The captured activations from the hook
        print(f"Output tensor shape: {output_tensor.shape}")
        
        # Maximize the activation of the target class with L2 and TV regularization
        loss = -F.relu(output_tensor[:, target_class]).sum()
        loss += reg_lambda * torch.norm(input_image) + reg_lambda * total_variation_loss(input_image)
        
        # Backward pass
        loss.backward(retain_graph=True)
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_([input_image], max_norm=1.0)
        # Update the image
        optimizer.step()

        # Apply Gaussian smoothing less frequently
        with torch.no_grad():
            input_image.data = torch.clamp(input_image.data, 0, 1)
            if (i + 1) % smooth_interval == 0:
                input_image.data = gaussian_smoothing(input_image.data, kernel_size=smooth_kernel, sigma=smooth_sigma)
        
        print(f"Iteration: {i+1}, Loss: {loss.item()}")
        
        # Save intermediate images every 80 iterations
        if (i + 1) % 40 == 0:
            optimized_image = input_image.detach().cpu().squeeze().permute(1, 2, 0).numpy()
            optimized_image = normalize(optimized_image).numpy()
            img = Image.fromarray((optimized_image * 255).astype(np.uint8))
            frames.append(img)
            
            # Save the current frame to the activations folder
            img_path = os.path.join(output_dir, f'{target_index}_current_activation.png')
            img.save(img_path)

    # Save the final optimized image
    optimized_image = input_image.detach().cpu().squeeze().permute(1, 2, 0).numpy()
    optimized_image = normalize(optimized_image).numpy()
    img = Image.fromarray((optimized_image * 255).astype(np.uint8))
    frames.append(img)

    # Save the frames as a GIF
    gif_path = os.path.join(output_dir, f'{target_index}_activation_maximization.gif')
    frames[0].save(gif_path, save_all=True, append_images=frames[1:], duration=200, loop=0)
    
    hook.remove()  # Remove the hook after processing the layer

    return optimized_image

# Iterate through all layers of the model
for idx, layer in enumerate(model.modules()):
    if idx == 47:
        if isinstance(layer, torch.nn.Module):
            print(f"Processing layer {idx}: {layer}")
            try:
                visualize_activation(model, layer, idx, target_class=0, input_size=(1280, 1280), lr=0.001, iterations=1000, reg_lambda=.8)
            except Exception as e:
                print(f"An error occurred while processing layer {idx}: {e}")



       
