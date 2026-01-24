import cv2
import numpy as np
import matplotlib.pyplot as plt

def zoom_nearest_neighbor(image, scale):
    """
    (a) Zoom image using nearest-neighbor interpolation
    """
    height, width = image.shape[:2]
    new_height = int(height * scale)
    new_width = int(width * scale)
    
    # Create output image
    if len(image.shape) == 3:
        zoomed = np.zeros((new_height, new_width, image.shape[2]), dtype=image.dtype)
    else:
        zoomed = np.zeros((new_height, new_width), dtype=image.dtype)
    
    for i in range(new_height):
        for j in range(new_width):
            # Find corresponding source pixel
            src_i = int(i / scale)
            src_j = int(j / scale)
            
            # Boundary check
            src_i = min(src_i, height - 1)
            src_j = min(src_j, width - 1)
            
            zoomed[i, j] = image[src_i, src_j]
    
    return zoomed

def zoom_bilinear(image, scale):
    """
    (b) Zoom image using bilinear interpolation
    """
    height, width = image.shape[:2]
    new_height = int(height * scale)
    new_width = int(width * scale)
    
    # Create output image
    if len(image.shape) == 3:
        zoomed = np.zeros((new_height, new_width, image.shape[2]), dtype=image.dtype)
    else:
        zoomed = np.zeros((new_height, new_width), dtype=image.dtype)
    
    for i in range(new_height):
        for j in range(new_width):
            # Find corresponding source coordinates
            src_i = i / scale
            src_j = j / scale
            
            # Get integer and fractional parts
            i1 = int(np.floor(src_i))
            i2 = min(i1 + 1, height - 1)
            j1 = int(np.floor(src_j))
            j2 = min(j1 + 1, width - 1)
            
            # Compute weights
            di = src_i - i1
            dj = src_j - j1
            
            # Bilinear interpolation
            if len(image.shape) == 3:
                for c in range(image.shape[2]):
                    value = (1 - di) * (1 - dj) * image[i1, j1, c] + \
                            (1 - di) * dj * image[i1, j2, c] + \
                            di * (1 - dj) * image[i2, j1, c] + \
                            di * dj * image[i2, j2, c]
                    zoomed[i, j, c] = value
            else:
                value = (1 - di) * (1 - dj) * image[i1, j1] + \
                        (1 - di) * dj * image[i1, j2] + \
                        di * (1 - dj) * image[i2, j1] + \
                        di * dj * image[i2, j2]
                zoomed[i, j] = value
    
    return zoomed

def compute_normalized_ssd(img1, img2):
    """
    Compute normalized Sum of Squared Differences
    """
    if img1.shape != img2.shape:
        print(f"Warning: Image shapes don't match: {img1.shape} vs {img2.shape}")
        return None
    
    # Compute SSD
    ssd = np.sum((img1.astype(float) - img2.astype(float)) ** 2)
    
    # Normalize by number of pixels and channels
    if len(img1.shape) == 3:
        n_elements = img1.shape[0] * img1.shape[1] * img1.shape[2]
    else:
        n_elements = img1.shape[0] * img1.shape[1]
    
    normalized_ssd = ssd / n_elements
    
    return normalized_ssd

def main():
    print("Question 7: Image Zooming with Interpolation")
    print("="*60)
    
    # Load small and large images
    small_image_path = 'small_image.jpg'  # Replace with your small image
    large_image_path = 'large_image.jpg'  # Replace with your large original
    
    small_img = cv2.imread(small_image_path)
    large_img = cv2.imread(large_image_path)
    
    if small_img is None:
        print(f"Error: Could not load small image from {small_image_path}")
        print("Creating a test image instead...")
        # Create a simple test image
        small_img = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        large_img = None
    
    # Test different zoom factors
    zoom_factors = [2.0, 3.0, 4.0]
    
    # If we have a large image, calculate the zoom factor to match it
    if large_img is not None:
        calculated_scale = large_img.shape[0] / small_img.shape[0]
        print(f"Calculated zoom factor to match large image: {calculated_scale:.2f}")
        zoom_factors = [calculated_scale]
    
    for scale in zoom_factors:
        print(f"\n{'='*60}")
        print(f"Processing with zoom factor: {scale}x")
        print(f"{'='*60}")
        
        # (a) Nearest-neighbor interpolation
        print("(a) Applying nearest-neighbor interpolation...")
        zoomed_nn = zoom_nearest_neighbor(small_img, scale)
        
        # (b) Bilinear interpolation
        print("(b) Applying bilinear interpolation...")
        zoomed_bilinear = zoom_bilinear(small_img, scale)
        
        # Compare with OpenCV's resize for verification
        zoomed_opencv_nn = cv2.resize(small_img, None, fx=scale, fy=scale, 
                                      interpolation=cv2.INTER_NEAREST)
        zoomed_opencv_linear = cv2.resize(small_img, None, fx=scale, fy=scale, 
                                          interpolation=cv2.INTER_LINEAR)
        
        # Display results
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        axes[0, 0].imshow(cv2.cvtColor(small_img, cv2.COLOR_BGR2RGB))
        axes[0, 0].set_title(f'Original Small Image\n{small_img.shape[1]}×{small_img.shape[0]}')
        axes[0, 0].axis('off')
        
        axes[0, 1].imshow(cv2.cvtColor(zoomed_nn, cv2.COLOR_BGR2RGB))
        axes[0, 1].set_title(f'(a) Nearest-Neighbor (Manual)\n{zoomed_nn.shape[1]}×{zoomed_nn.shape[0]}')
        axes[0, 1].axis('off')
        
        axes[0, 2].imshow(cv2.cvtColor(zoomed_bilinear, cv2.COLOR_BGR2RGB))
        axes[0, 2].set_title(f'(b) Bilinear (Manual)\n{zoomed_bilinear.shape[1]}×{zoomed_bilinear.shape[0]}')
        axes[0, 2].axis('off')
        
        axes[1, 0].imshow(cv2.cvtColor(zoomed_opencv_nn, cv2.COLOR_BGR2RGB))
        axes[1, 0].set_title('OpenCV Nearest-Neighbor')
        axes[1, 0].axis('off')
        
        axes[1, 1].imshow(cv2.cvtColor(zoomed_opencv_linear, cv2.COLOR_BGR2RGB))
        axes[1, 1].set_title('OpenCV Bilinear')
        axes[1, 1].axis('off')
        
        # If large image exists, compare
        if large_img is not None and zoomed_nn.shape[:2] == large_img.shape[:2]:
            axes[1, 2].imshow(cv2.cvtColor(large_img, cv2.COLOR_BGR2RGB))
            axes[1, 2].set_title(f'Large Original\n{large_img.shape[1]}×{large_img.shape[0]}')
            axes[1, 2].axis('off')
            
            # Compute SSD
            ssd_nn = compute_normalized_ssd(zoomed_nn, large_img)
            ssd_bilinear = compute_normalized_ssd(zoomed_bilinear, large_img)
            
            print(f"\nNormalized SSD Results:")
            print(f"  Nearest-Neighbor SSD: {ssd_nn:.4f}")
            print(f"  Bilinear SSD: {ssd_bilinear:.4f}")
            print(f"  Difference: {abs(ssd_nn - ssd_bilinear):.4f}")
            
            if ssd_bilinear < ssd_nn:
                print(f"  → Bilinear is better (lower SSD by {ssd_nn - ssd_bilinear:.4f})")
            else:
                print(f"  → Nearest-Neighbor is better (lower SSD by {ssd_bilinear - ssd_nn:.4f})")
        else:
            axes[1, 2].text(0.5, 0.5, 'No large image\nfor comparison', 
                           ha='center', va='center', fontsize=12)
            axes[1, 2].axis('off')
        
        plt.tight_layout()
        plt.savefig(f'q7_results_scale_{scale:.1f}x.png', dpi=150, bbox_inches='tight')
        plt.show()
        
        # Save individual results
        cv2.imwrite(f'q7_nn_scale_{scale:.1f}x.png', zoomed_nn)
        cv2.imwrite(f'q7_bilinear_scale_{scale:.1f}x.png', zoomed_bilinear)
    
    print("\n" + "="*60)
    print("Analysis:")
    print("-"*60)
    print("Nearest-Neighbor Interpolation:")
    print("  + Fast computation (simple indexing)")
    print("  + Preserves sharp edges")
    print("  - Produces blocky, pixelated results")
    print("  - Aliasing artifacts visible")
    print("\nBilinear Interpolation:")
    print("  + Smoother results")
    print("  + Better visual quality")
    print("  + Lower SSD when compared to originals")
    print("  - Slightly slower")
    print("  - May blur sharp edges")
    print("\nConclusion: Bilinear interpolation generally produces")
    print("better results with lower SSD values.")
    print("="*60)

if __name__ == "__main__":
    main()