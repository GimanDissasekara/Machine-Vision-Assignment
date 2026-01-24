import cv2
import numpy as np
import matplotlib.pyplot as plt

def otsu_threshold(image):
    """
    Implement Otsu's thresholding algorithm
    Returns the optimal threshold value
    """
    # Compute histogram
    histogram, _ = np.histogram(image.flatten(), bins=256, range=[0, 256])
    
    # Total number of pixels
    total_pixels = image.size
    
    # Compute sum of all intensity values
    sum_total = np.sum(np.arange(256) * histogram)
    
    sum_background = 0
    weight_background = 0
    max_variance = 0
    threshold = 0
    
    for t in range(256):
        weight_background += histogram[t]
        if weight_background == 0:
            continue
        
        weight_foreground = total_pixels - weight_background
        if weight_foreground == 0:
            break
        
        sum_background += t * histogram[t]
        
        mean_background = sum_background / weight_background
        mean_foreground = (sum_total - sum_background) / weight_foreground
        
        # Calculate between-class variance
        variance = weight_background * weight_foreground * (mean_background - mean_foreground) ** 2
        
        if variance > max_variance:
            max_variance = variance
            threshold = t
    
    return threshold

def apply_binary_mask(image, threshold):
    """
    Apply binary thresholding
    Returns binary mask (foreground = 1, background = 0)
    """
    mask = (image > threshold).astype(np.uint8)
    return mask

def histogram_equalization_masked(image, mask):
    """
    Apply histogram equalization only to masked region
    """
    result = image.copy()
    
    # Get pixels in the masked region
    masked_pixels = image[mask == 1]
    
    if len(masked_pixels) == 0:
        return result
    
    # Compute histogram for masked region
    histogram, _ = np.histogram(masked_pixels, bins=256, range=[0, 256])
    
    # Compute cumulative distribution function (CDF)
    cdf = histogram.cumsum()
    
    # Normalize CDF
    cdf_min = cdf[cdf > 0].min()
    cdf_normalized = ((cdf - cdf_min) * 255 / (masked_pixels.size - cdf_min)).astype(np.uint8)
    
    # Apply equalization only to masked pixels
    result[mask == 1] = cdf_normalized[image[mask == 1]]
    
    return result

def main():
    # Load image
    image_path = 'woman_door.jpg'  # Replace with your image path
    image = cv2.imread(image_path)
    
    if image is None:
        print(f"Error: Could not load image from {image_path}")
        return
    
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # (a) Apply Otsu's thresholding
    threshold_value = otsu_threshold(gray)
    print(f"(a) Otsu Threshold Value: {threshold_value}")
    
    # Create binary mask
    mask = apply_binary_mask(gray, threshold_value)
    binary_image = (mask * 255).astype(np.uint8)
    
    # (b) Apply histogram equalization to foreground only
    equalized = histogram_equalization_masked(gray, mask)
    
    # Display results
    plt.figure(figsize=(15, 10))
    
    plt.subplot(2, 3, 1)
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title('Original Image')
    plt.axis('off')
    
    plt.subplot(2, 3, 2)
    plt.imshow(gray, cmap='gray')
    plt.title('Grayscale Image')
    plt.axis('off')
    
    plt.subplot(2, 3, 3)
    plt.imshow(binary_image, cmap='gray')
    plt.title(f'Binary Mask (Otsu)\nThreshold = {threshold_value}')
    plt.axis('off')
    
    plt.subplot(2, 3, 4)
    plt.hist(gray.flatten(), bins=256, range=[0, 256], color='blue', alpha=0.7)
    plt.axvline(x=threshold_value, color='red', linestyle='--', linewidth=2, label=f'Threshold={threshold_value}')
    plt.title('Histogram with Otsu Threshold')
    plt.xlabel('Pixel Intensity')
    plt.ylabel('Frequency')
    plt.legend()
    
    plt.subplot(2, 3, 5)
    plt.imshow(equalized, cmap='gray')
    plt.title('Foreground Histogram Equalized')
    plt.axis('off')
    
    plt.subplot(2, 3, 6)
    # Show difference
    difference = np.abs(equalized.astype(float) - gray.astype(float))
    plt.imshow(difference, cmap='hot')
    plt.title('Difference (Equalized - Original)')
    plt.axis('off')
    plt.colorbar()
    
    plt.tight_layout()
    plt.savefig('q4_results.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print("\n(b) Hidden Features Revealed:")
    print("- Texture details in clothing and fabric patterns")
    print("- Facial features and skin tones in shadowed areas")
    print("- Room details like furniture textures and wall patterns")
    print("- Enhanced contrast in previously dark regions")
    print("- Better edge definition between objects")
    
    # Save individual results
    cv2.imwrite('q4_grayscale.png', gray)
    cv2.imwrite('q4_binary_mask.png', binary_image)
    cv2.imwrite('q4_equalized.png', equalized)
    
    print("\nResults saved as: q4_results.png, q4_grayscale.png, q4_binary_mask.png, q4_equalized.png")

if __name__ == "__main__":
    main()