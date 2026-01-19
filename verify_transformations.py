import torch
import torchvision.transforms.functional as F
from PIL import Image
import os
import glob
import numpy as np
from intensity_transformations import apply_gamma, apply_contrast_stretching

def verify_image(path):
    print(f"\nVerifying {os.path.basename(path)}...")
    try:
        img = Image.open(path).convert('L')
        original = F.to_tensor(img)
    except Exception as e:
        print(f"FAILED to load: {e}")
        return

    # Apply transformations
    gamma_05 = apply_gamma(original, 0.5)
    gamma_20 = apply_gamma(original, 2.0)
    stretched = apply_contrast_stretching(original)

    # Calculate means
    mean_original = original.mean().item()
    mean_gamma_05 = gamma_05.mean().item()
    mean_gamma_20 = gamma_20.mean().item()
    std_original = original.std().item()
    std_stretched = stretched.std().item()

    print(f"  Mean Intensity: Original={mean_original:.4f}, Gamma0.5={mean_gamma_05:.4f}, Gamma2.0={mean_gamma_20:.4f}")
    print(f"  Contrast (Std Dev): Original={std_original:.4f}, Stretched={std_stretched:.4f}")

    # Assertions
    # Gamma 0.5 should brighten (mean increases) if image is not already max bright
    if mean_original < 0.99: 
        if mean_gamma_05 > mean_original:
            print("  [PASS] Gamma 0.5 increased brightness.")
        else:
            print(f"  [FAIL] Gamma 0.5 did NOT increase brightness ({mean_gamma_05} !> {mean_original}).")
    
    # Gamma 2.0 should darken (mean decreases) if image is not already pitch black
    if mean_original > 0.01:
        if mean_gamma_20 < mean_original:
            print("  [PASS] Gamma 2.0 decreased brightness.")
        else:
            print(f"  [FAIL] Gamma 2.0 did NOT decrease brightness ({mean_gamma_20} !< {mean_original}).")

    # Contrast Stretching (Piecewise)
    # Harder to assert generally without knowing distribution, but generally standard deviation might change.
    # We can at least check it ran and produced values in [0, 1]
    if stretched.min() >= 0.0 and stretched.max() <= 1.0:
        print("  [PASS] Contrast Stretching values in [0, 1].")
    else:
        print(f"  [FAIL] Contrast Stretching values out of range [{stretched.min()}, {stretched.max()}].")

def main():
    sources_dir = os.path.join(os.getcwd(), 'Sources')
    image_extensions = ['*.png', '*.jpg', '*.jpeg']
    image_files = []
    for ext in image_extensions:
        image_files.extend(glob.glob(os.path.join(sources_dir, ext)))
    
    if not image_files:
        print("No images found.")
        return

    print(f"Verifying {len(image_files)} images...")
    for img_path in image_files:
        verify_image(img_path)

if __name__ == "__main__":
    main()
