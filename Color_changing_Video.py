import torch
import numpy as np
import cv2

# Parameters
width, height = 2560, 1440
fps = 60
duration = 360  # seconds
num_frames = fps * duration

# Create a video writer object
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
video_writer = cv2.VideoWriter('color_changing_video.mp4', fourcc, fps, (width, height))

# Check if CUDA is available and create a device object
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Use PyTorch tensors on CUDA
for frame_idx in range(num_frames):
    # Create an empty tensor on the specified device
    hsv_image = torch.zeros((height, width, 3), dtype=torch.uint8, device=device)
    
    # Calculate color components based on the frame index
    hue = int((frame_idx / num_frames) * 180)  # hue ranges from 0 to 179 (OpenCV's range for HSV)
    saturation = int((np.sin(frame_idx / num_frames * 2 * np.pi) * 128 + 128))  # Varies from 0 to 255
    value = int((np.cos(frame_idx / num_frames * 2 * np.pi) * 128 + 128))  # Varies from 0 to 255

    # Ensure saturation and value are within the valid range [0, 255]
    saturation = np.clip(saturation, 0, 255)
    value = np.clip(value, 0, 255)

    # Create tensors for hue, saturation, and value
    hue_tensor = torch.full((height, width), hue, dtype=torch.uint8, device=device)
    saturation_tensor = torch.full((height, width), saturation, dtype=torch.uint8, device=device)
    value_tensor = torch.full((height, width), value, dtype=torch.uint8, device=device)

    # Stack tensors to form the HSV image
    hsv_image = torch.stack([hue_tensor, saturation_tensor, value_tensor], dim=-1)
    
    # Convert HSV tensor to NumPy array (move to CPU first)
    hsv_image_np = hsv_image.cpu().numpy()
    
    # Convert HSV to RGB using OpenCV
    rgb_image = cv2.cvtColor(hsv_image_np, cv2.COLOR_HSV2BGR)

    # Write the frame to the video
    video_writer.write(rgb_image)

# Release the video writer
video_writer.release()

print("Video created successfully.")