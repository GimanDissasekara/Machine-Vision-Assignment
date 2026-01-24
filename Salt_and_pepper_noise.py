import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

def add_salt_pepper_noise(image, salt_prob=0.02, pepper_prob=0.02):
    """
    Add salt and pepper noise to an image for testing
    """
    noisy = image.copy()
    
    # Salt noise (white pixels)
    salt_mask = np.random.random(image.shape[:2]) < salt_prob
    noisy[salt_mask] = 255
    
    # Pepper noise (black pixels)
    pepper_mask = np.random.random(image.shape[:2]) < pepper_prob
    noisy[pepper_mask] = 0
    
    return noisy

def gaussian_filter_manual(image, kernel_size=5, sigma=1.5):
    """
    (a) Apply Gaussian smoothing manually
    """
    # Create Gaussian kernel
    kernel = np.zeros((kernel_size, kernel_size))
    center = kernel_size // 2
    
    for i in range(kernel_size):
        for j in range(kernel_size):
            x = i - center
            y = j - center
            kernel[i, j] = np.exp(-(x**2 + y**2) / (2 * sigma**2))
    
    # Normalize
    kernel = kernel / np.sum(kernel)
    
    # Apply filter
    filtered = cv2.filter2D(image, -1, kernel)
    
    return filtered

def median_filter_manual(image, kernel_size=5):
    """
    (b) Apply median filter manually
    """
    height, width = image.shape[:2]
    pad = kernel_size // 2
    
    # Pad image
    if len(image.shape) == 3:
        padded = np.pad(image, ((pad, pad), (pad, pad), (0, 0)), mode='edge')
        filtered = np.zeros_like(image)
        
        for i in range(height):
            for j in range(width):
                for c in range(image.shape[2]):
                    window = padded[i:i+kernel_size, j:j+kernel_size, c]
                    filtered[i, j, c] = np.median(window)
    else:
        padded = np.pad(image, pad, mode='edge')
        filtered = np.zeros_like(image)
        
        for i in range(height):
            for j in range(width):
                window = padded[i:i+kernel_size, j:j+kernel_size]
                filtered[i, j] = np.median(window)
    
    return filtered

def compute_psnr(original, filtered):
    """
    Compute Peak Signal-to-Noise Ratio
    """
    mse = np.mean((original.astype(float) - filtered.astype(float)) ** 2)
    if mse == 0:
        return float('inf')
    
    max_pixel = 255.0
    psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
    
    return psnr

def main():
    print("Question 8: Salt & Pepper Noise Filtering")
    print("="*60)
    
    # Load image
    image_path = os.path.join('Sources', 'emma_salt_pepper.jpg')  # Replace with your noisy image
    image = cv2.imread(image_path)
    
    # If no image, create a test image with noise
    if image is None:
        print(f"Could not load {image_path}, creating test image with noise...")
        # Load a clean image or create one
        clean_image_path = 'clean_image.jpg'
        image = cv2.imread(clean_image_path)
        
        if image is None:
            print("Creating synthetic test image...")
            image = np.random.randint(100, 200, (300, 300, 3), dtype=np.uint8)
            # Add some structure
            cv2.rectangle(image, (50, 50), (250, 250), (255, 255, 255), -1)
            cv2.circle(image, (150, 150), 50, (0, 0, 0), -1)
        
        # Add salt and pepper noise
        image = add_salt_pepper_noise(image, salt_prob=0.05, pepper_prob=0.05)
        cv2.imwrite('generated_noisy_image.jpg', image)
        print("Generated noisy image saved as 'generated_noisy_image.jpg'")
    
    # Convert to grayscale for processing
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    print(f"Image shape: {gray.shape}")
    
    # (a) Apply Gaussian smoothing
    print("\n(a) Applying Gaussian smoothing...")
    gaussian_filtered = gaussian_filter_manual(gray, kernel_size=5, sigma=1.5)
    
    # Also try OpenCV's GaussianBlur
    gaussian_opencv = cv2.GaussianBlur(gray, (5, 5), 1.5)
    
    # (b) Apply median filter
    print("(b) Applying median filter...")
    median_filtered = median_filter_manual(gray, kernel_size=5)
    
    # Also try OpenCV's medianBlur
    median_opencv = cv2.medianBlur(gray, 5)
    
    # Display results
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    axes[0, 0].imshow(gray, cmap='gray')
    axes[0, 0].set_title('Original (Noisy) Image')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(gaussian_filtered, cmap='gray')
    axes[0, 1].set_title('(a) Gaussian Filtered (Manual)')
    axes[0, 1].axis('off')
    
    axes[0, 2].imshow(median_filtered, cmap='gray')
    axes[0, 2].set_title('(b) Median Filtered (Manual)')
    axes[0, 2].axis('off')
    
    axes[1, 0].imshow(gaussian_opencv, cmap='gray')
    axes[1, 0].set_title('Gaussian (OpenCV)')
    axes[1, 0].axis('off')
    
    axes[1, 1].imshow(median_opencv, cmap='gray')
    axes[1, 1].set_title('Median (OpenCV)')
    axes[1, 1].axis('off')
    
    # Show difference between Gaussian and Median
    diff = np.abs(gaussian_filtered.astype(float) - median_filtered.astype(float))
    axes[1, 2].imshow(diff, cmap='hot')
    axes[1, 2].set_title('Difference (Gaussian - Median)')
    axes[1, 2].axis('off')
    plt.colorbar(axes[1, 2].imshow(diff, cmap='hot'), ax=axes[1, 2])
    
    plt.tight_layout()
    plt.savefig('q8_results.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # Detailed comparison plot
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Zoom in on a region to see detail
    h, w = gray.shape
    y1, y2 = h//3, 2*h//3
    x1, x2 = w//3, 2*w//3
    
    axes[0].imshow(gray[y1:y2, x1:x2], cmap='gray')
    axes[0].set_title('Original (Zoomed)')
    axes[0].axis('off')
    
    axes[1].imshow(gaussian_filtered[y1:y2, x1:x2], cmap='gray')
    axes[1].set_title('Gaussian Filtered (Zoomed)')
    axes[1].axis('off')
    
    axes[2].imshow(median_filtered[y1:y2, x1:x2], cmap='gray')
    axes[2].set_title('Median Filtered (Zoomed)')
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.savefig('q8_zoomed_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # Analysis
    print("\n" + "="*60)
    print("ANALYSIS: Gaussian vs Median Filtering for Salt & Pepper Noise")
    print("="*60)
    
    print("\n(a) Gaussian Smoothing:")
    print("  - Linear filter (weighted average)")
    print("  - Blurs noise but also blurs edges and details")
    print("  - Noise pixels affect neighboring pixels")
    print("  - Salt & pepper noise is reduced but NOT eliminated")
    print("  - Image appears overall blurry")
    
    print("\n(b) Median Filtering:")
    print("  - Non-linear filter (order statistic)")
    print("  - EXCELLENT for impulse noise (salt & pepper)")
    print("  - Replaces outliers with median value")
    print("  - Preserves edges while removing noise")
    print("  - Salt & pepper noise is effectively ELIMINATED")
    
    print("\n" + "-"*60)
    print("CONCLUSION:")
    print("-"*60)
    print("For salt and pepper noise, MEDIAN FILTER is clearly superior:")
    print("  ✓ Completely removes isolated noise pixels")
    print("  ✓ Preserves edges and important details")
    print("  ✓ Does not create blur artifacts")
    print("\nGaussian filter is better for:")
    print("  • Gaussian (random) noise")
    print("  • General smoothing")
    print("  • Pre-processing for other operations")
    print("="*60)
    
    # Save results
    cv2.imwrite('q8_gaussian_filtered.png', gaussian_filtered)
    cv2.imwrite('q8_median_filtered.png', median_filtered)
    
    print("\nResults saved as: q8_results.png, q8_zoomed_comparison.png")
    print("                  q8_gaussian_filtered.png, q8_median_filtered.png")

if __name__ == "__main__":
    main()