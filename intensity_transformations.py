import torch
import torchvision.transforms.functional as F
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import glob

def load_image(path):
    """
    Loads an image, converts to grayscale, and converts to a PyTorch tensor.
    Returns the tensor (C, H, W) normalized to [0, 1] and the original PIL image for reference.
    """
    try:
        img = Image.open(path).convert('L') # Convert to grayscale for intensity transforms
        # ToTensor converts to [0, 1] and (C, H, W)
        tensor = F.to_tensor(img)
        return tensor
    except Exception as e:
        print(f"Error loading {path}: {e}")
        return None

def apply_gamma(tensor, gamma):
    """
    Applies Power-Law (Gamma) transformation: s = r^gamma
    """
    return torch.pow(tensor, gamma)

def apply_contrast_stretching(tensor, r1=0.2, r2=0.8):
    """
    Applies Piecewise Linear Contrast Stretching.
    Zone 1 (r < r1): 0
    Zone 2 (r1 <= r <= r2): (r - r1) / (r2 - r1)
    Zone 3 (r > r2): 1
    """
    # Clone to avoid modifying original in place if shared
    s = tensor.clone()
    
    # Zone 1
    s[tensor < r1] = 0.0
    
    # Zone 3
    s[tensor > r2] = 1.0
    
    # Zone 2
    mask = (tensor >= r1) & (tensor <= r2)
    # s = (r - r1) / (r2 - r1)
    s[mask] = (tensor[mask] - r1) / (r2 - r1)
    
    return s

def process_and_visualize(image_path, output_dir):
    """
    Applies transformations and saves visualization.
    """
    filename = os.path.basename(image_path)
    name, ext = os.path.splitext(filename)
    
    print(f"Processing {filename}...")
    
    original_tensor = load_image(image_path)
    if original_tensor is None:
        return

    # Transformations
    gamma_05 = apply_gamma(original_tensor, 0.5)
    gamma_20 = apply_gamma(original_tensor, 2.0)
    contrast_stretched = apply_contrast_stretching(original_tensor)

    # Plotting
    fig, axs = plt.subplots(1, 4, figsize=(20, 5))
    
    # Helper to display tensor
    def show_ax(ax, tensor, title):
        # CxHxW -> HxW for grayscale imshow
        img_np = tensor.squeeze(0).numpy()
        ax.imshow(img_np, cmap='gray', vmin=0, vmax=1)
        ax.set_title(title)
        ax.axis('off')

    show_ax(axs[0], original_tensor, 'Original')
    show_ax(axs[1], gamma_05, 'Gamma = 0.5 (Darker pixels boosted)')
    show_ax(axs[2], gamma_20, 'Gamma = 2.0 (Highlights boosted)')
    show_ax(axs[3], contrast_stretched, 'Contrast Stretching')

    plt.tight_layout()
    output_path = os.path.join(output_dir, f"{name}_results.png")
    plt.savefig(output_path)
    plt.close(fig)
    print(f"Saved results to {output_path}")

def main():
    # Setup directories
    sources_dir = os.path.join(os.getcwd(), 'Sources')
    results_dir = os.path.join(os.getcwd(), 'Results')
    
    if not os.path.exists(sources_dir):
        print(f"Directory not found: {sources_dir}")
        return

    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    # Find images
    image_extensions = ['*.png', '*.jpg', '*.jpeg', '*.bmp', '*.tif']
    image_files = []
    for ext in image_extensions:
        image_files.extend(glob.glob(os.path.join(sources_dir, ext)))
    
    if not image_files:
        print("No images found in Sources directory.")
        return

    print(f"Found {len(image_files)} images.")
    for img_path in image_files:
        process_and_visualize(img_path, results_dir)

if __name__ == "__main__":
    main()
