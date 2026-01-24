import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import glob

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
    
    zooming_dir = os.path.join('Sources', 'images_for_zooming')
    if not os.path.exists(zooming_dir):
        print(f"Error: Directory not found: {zooming_dir}")
        return

    # Find all small images
    small_images = []
    for ext in ['*.png', '*.jpg', '*.jpeg']:
        small_images.extend(glob.glob(os.path.join(zooming_dir, f"*small*{ext}")))
    
    if not small_images:
        print(f"No small images found in {zooming_dir}")
        return

    print(f"Found {len(small_images)} images to process.")

    for small_image_path in small_images:
        filename = os.path.basename(small_image_path)
        name, ext = os.path.splitext(filename)
        
        print(f"\nProcessing {filename}...")
        
        # Try to find corresponding large image
        # Remove "small", "_small", "_very_small" to find large version
        large_name = name.replace('very_small', '').replace('small', '').rstrip('_')
        large_image_path = os.path.join(zooming_dir, large_name + ext)
        
        small_img = cv2.imread(small_image_path)
        large_img = cv2.imread(large_image_path) if os.path.exists(large_image_path) else None
        
        if small_img is None:
            print(f"  Error: Could not load {small_image_path}")
            continue
            
        # Test different zoom factors
        zoom_factors = [2.0, 3.0, 4.0]
        
        # If we have a large image, calculate the zoom factor to match it
        if large_img is not None:
            calculated_scale = large_img.shape[0] / small_img.shape[0]
            print(f"  Found matching large image. Scale: {calculated_scale:.2f}x")
            zoom_factors = [calculated_scale]
        else:
            print(f"  No matching large image found for {large_name}{ext}. Using default scales.")
        
        for scale in zoom_factors:
            print(f"  Applying {scale:.1f}x zoom...")
            
            # (a) Nearest-neighbor interpolation
            zoomed_nn = zoom_nearest_neighbor(small_img, scale)
            
            # (b) Bilinear interpolation
            zoomed_bilinear = zoom_bilinear(small_img, scale)
            
            # Compare with OpenCV's resize for verification
            zoomed_opencv_nn = cv2.resize(small_img, None, fx=scale, fy=scale, 
                                          interpolation=cv2.INTER_NEAREST)
            zoomed_opencv_linear = cv2.resize(small_img, None, fx=scale, fy=scale, 
                                              interpolation=cv2.INTER_LINEAR)
            
            # Display results
            fig, axes = plt.subplots(2, 3, figsize=(18, 12))
            
            axes[0, 0].imshow(cv2.cvtColor(small_img, cv2.COLOR_BGR2RGB))
            axes[0, 0].set_title(f'Original Small\n{small_img.shape[1]}×{small_img.shape[0]}')
            axes[0, 0].axis('off')
            
            axes[0, 1].imshow(cv2.cvtColor(zoomed_nn, cv2.COLOR_BGR2RGB))
            axes[0, 1].set_title(f'NN (Manual)\n{zoomed_nn.shape[1]}×{zoomed_nn.shape[0]}')
            axes[0, 1].axis('off')
            
            axes[0, 2].imshow(cv2.cvtColor(zoomed_bilinear, cv2.COLOR_BGR2RGB))
            axes[0, 2].set_title(f'Bilinear (Manual)\n{zoomed_bilinear.shape[1]}×{zoomed_bilinear.shape[0]}')
            axes[0, 2].axis('off')
            
            axes[1, 0].imshow(cv2.cvtColor(zoomed_opencv_nn, cv2.COLOR_BGR2RGB))
            axes[1, 0].set_title('OpenCV NN')
            axes[1, 0].axis('off')
            
            axes[1, 1].imshow(cv2.cvtColor(zoomed_opencv_linear, cv2.COLOR_BGR2RGB))
            axes[1, 1].set_title('OpenCV Bilinear')
            axes[1, 1].axis('off')
            
            if large_img is not None and zoomed_nn.shape[:2] == large_img.shape[:2]:
                axes[1, 2].imshow(cv2.cvtColor(large_img, cv2.COLOR_BGR2RGB))
                axes[1, 2].set_title(f'Target Large\n{large_img.shape[1]}×{large_img.shape[0]}')
                axes[1, 2].axis('off')
                
                ssd_nn = compute_normalized_ssd(zoomed_nn, large_img)
                ssd_bilinear = compute_normalized_ssd(zoomed_bilinear, large_img)
                print(f"    SSD (NN): {ssd_nn:.4f}, SSD (Bilinear): {ssd_bilinear:.4f}")
            else:
                axes[1, 2].text(0.5, 0.5, 'No comparison\navailable', ha='center', va='center')
                axes[1, 2].axis('off')
            
            plt.tight_layout()
            output_name = f'zoom_result_{name}_{scale:.1f}x.png'
            plt.savefig(output_name, dpi=150)
            plt.close()
            print(f"    Result saved to {output_name}")

    print("\nBatch processing complete.")

if __name__ == "__main__":
    main()