import cv2
import numpy as np
import matplotlib.pyplot as plt
import time
import os

def bilateral_filter_manual(image, d, sigma_space, sigma_color):
    """
    (a) Manual implementation of bilateral filter for grayscale images
    
    Parameters:
    - image: input grayscale image
    - d: diameter of pixel neighborhood
    - sigma_space: spatial standard deviation (σs)
    - sigma_color: range/intensity standard deviation (σr)
    
    Returns:
    - filtered image
    """
    print(f"  Applying bilateral filter: d={d}, σs={sigma_space}, σr={sigma_color}")
    
    height, width = image.shape
    filtered = np.zeros_like(image, dtype=np.float64)
    
    radius = d // 2
    
    # Pre-compute spatial Gaussian weights
    spatial_coeff = -0.5 / (sigma_space ** 2)
    range_coeff = -0.5 / (sigma_color ** 2)
    
    # Process each pixel
    for i in range(height):
        if i % 50 == 0:
            print(f"    Processing row {i}/{height}")
        
        for j in range(width):
            center_intensity = image[i, j]
            
            weighted_sum = 0.0
            weight_sum = 0.0
            
            # Iterate over neighborhood
            for ki in range(-radius, radius + 1):
                for kj in range(-radius, radius + 1):
                    # Get neighbor coordinates with boundary handling
                    ni = min(max(i + ki, 0), height - 1)
                    nj = min(max(j + kj, 0), width - 1)
                    
                    neighbor_intensity = image[ni, nj]
                    
                    # Compute spatial distance
                    spatial_dist = ki * ki + kj * kj
                    spatial_weight = np.exp(spatial_dist * spatial_coeff)
                    
                    # Compute intensity difference
                    intensity_diff = neighbor_intensity - center_intensity
                    range_weight = np.exp(intensity_diff * intensity_diff * range_coeff)
                    
                    # Combined weight
                    weight = spatial_weight * range_weight
                    
                    weighted_sum += neighbor_intensity * weight
                    weight_sum += weight
            
            filtered[i, j] = weighted_sum / weight_sum
    
    return filtered.astype(np.uint8)

def main():
    print("Question 10: Bilateral Filtering")
    print("="*60)
    
    # Load image
    image_path = os.path.join('Sources', 'runway.png')  # Replace with your image
    image = cv2.imread(image_path)
    
    if image is None:
        print(f"Could not load {image_path}, creating test image...")
        # Create a test image with edges and noise
        image = np.ones((200, 200, 3), dtype=np.uint8) * 128
        cv2.rectangle(image, (50, 50), (150, 150), (255, 255, 255), -1)
        cv2.circle(image, (100, 100), 30, (0, 0, 0), -1)
        
        # Add Gaussian noise
        noise = np.random.normal(0, 25, image.shape)
        image = np.clip(image + noise, 0, 255).astype(np.uint8)
        
        cv2.imwrite('test_noisy_image.jpg', image)
        print("Created test noisy image")
    
    # Convert to grayscale
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    print(f"Image shape: {gray.shape}")
    
    # Parameters for bilateral filter
    d = 9  # Diameter
    sigma_space = 75  # Spatial sigma
    sigma_color = 75  # Range/color sigma
    
    print("\n" + "="*60)
    print("BILATERAL FILTER THEORY")
    print("="*60)
    print("Formula: BF[I]_p = (1/W_p) * Σ G_s(||p-q||) * G_r(|I_p - I_q|) * I_q")
    print("\nWhere:")
    print("  G_s: Spatial Gaussian kernel (based on distance)")
    print("  G_r: Range Gaussian kernel (based on intensity difference)")
    print("  W_p: Normalization factor (sum of all weights)")
    print("\nKey Properties:")
    print("  • Non-linear filter")
    print("  • Edge-preserving smoothing")
    print("  • Combines spatial and range filtering")
    print("="*60)
    
    # (a) Manual bilateral filter
    print("\n(a) Applying manual bilateral filter...")
    start_time = time.time()
    bilateral_manual = bilateral_filter_manual(gray, d, sigma_space, sigma_color)
    manual_time = time.time() - start_time
    print(f"  Manual implementation time: {manual_time:.2f} seconds")
    
    # (b) Gaussian blur using OpenCV
    print("\n(b) Applying Gaussian smoothing (OpenCV)...")
    start_time = time.time()
    gaussian_opencv = cv2.GaussianBlur(gray, (9, 9), 0)
    gaussian_time = time.time() - start_time
    print(f"  Gaussian blur time: {gaussian_time:.4f} seconds")
    
    # (c) Bilateral filter using OpenCV
    print("\n(c) Applying bilateral filter (OpenCV)...")
    start_time = time.time()
    bilateral_opencv = cv2.bilateralFilter(gray, d, sigma_color, sigma_space)
    opencv_time = time.time() - start_time
    print(f"  OpenCV bilateral time: {opencv_time:.4f} seconds")
    
    # (d) Compare manual vs OpenCV bilateral
    print("\n(d) Comparing manual vs OpenCV implementation...")
    difference = np.abs(bilateral_manual.astype(float) - bilateral_opencv.astype(float))
    mean_diff = np.mean(difference)
    max_diff = np.max(difference)
    print(f"  Mean absolute difference: {mean_diff:.4f}")
    print(f"  Max absolute difference: {max_diff:.4f}")
    print(f"  Speedup (OpenCV vs Manual): {manual_time/opencv_time:.1f}x")
    
    # Display results - Main comparison
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    axes[0, 0].imshow(gray, cmap='gray')
    axes[0, 0].set_title('Original Grayscale Image')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(gaussian_opencv, cmap='gray')
    axes[0, 1].set_title(f'(b) Gaussian Blur (OpenCV)\nTime: {gaussian_time:.4f}s')
    axes[0, 1].axis('off')
    
    axes[0, 2].imshow(bilateral_manual, cmap='gray')
    axes[0, 2].set_title(f'(a,d) Bilateral Filter (Manual)\nTime: {manual_time:.2f}s')
    axes[0, 2].axis('off')
    
    axes[1, 0].imshow(bilateral_opencv, cmap='gray')
    axes[1, 0].set_title(f'(c) Bilateral Filter (OpenCV)\nTime: {opencv_time:.4f}s')
    axes[1, 0].axis('off')
    
    axes[1, 1].imshow(difference, cmap='hot')
    axes[1, 1].set_title(f'Difference (Manual - OpenCV)\nMean: {mean_diff:.2f}')
    axes[1, 1].axis('off')
    cbar1 = plt.colorbar(axes[1, 1].imshow(difference, cmap='hot'), ax=axes[1, 1])
    
    # Show difference between Gaussian and Bilateral
    diff_gaussian_bilateral = np.abs(gaussian_opencv.astype(float) - bilateral_opencv.astype(float))
    axes[1, 2].imshow(diff_gaussian_bilateral, cmap='hot')
    axes[1, 2].set_title('Difference (Gaussian - Bilateral)')
    axes[1, 2].axis('off')
    cbar2 = plt.colorbar(axes[1, 2].imshow(diff_gaussian_bilateral, cmap='hot'), ax=axes[1, 2])
    
    plt.tight_layout()
    plt.savefig('q10_bilateral_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # Zoomed comparison to see edge preservation
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    h, w = gray.shape
    y1, y2 = h//3, 2*h//3
    x1, x2 = w//3, 2*w//3
    
    axes[0].imshow(gray[y1:y2, x1:x2], cmap='gray')
    axes[0].set_title('Original (Zoomed)')
    axes[0].axis('off')
    
    axes[1].imshow(gaussian_opencv[y1:y2, x1:x2], cmap='gray')
    axes[1].set_title('Gaussian (Zoomed)\nBlurs edges')
    axes[1].axis('off')
    
    axes[2].imshow(bilateral_opencv[y1:y2, x1:x2], cmap='gray')
    axes[2].set_title('Bilateral (Zoomed)\nPreserves edges')
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.savefig('q10_edge_preservation.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # Analysis
    print("\n" + "="*60)
    print("ANALYSIS: GAUSSIAN vs BILATERAL FILTERING")
    print("="*60)
    
    print("\nGAUSSIAN FILTER:")
    print("  • Linear filter")
    print("  • Only considers spatial distance")
    print("  • Weight formula: w(p,q) = exp(-||p-q||²/(2σ²))")
    print("  • Smooths uniformly across entire image")
    print("  • BLURS EDGES along with noise")
    print("  • Fast computation (separable)")
    print("  • Use case: General smoothing, pre-processing")
    
    print("\nBILATERAL FILTER:")
    print("  • Non-linear filter")
    print("  • Considers BOTH spatial distance AND intensity similarity")
    print("  • Weight formula: w(p,q) = G_s(||p-q||) × G_r(|I_p - I_q|)")
    print("  • Smooths flat regions, PRESERVES edges")
    print("  • Slower computation (non-separable)")
    print("  • Use case: Noise reduction while keeping edges sharp")
    
    print("\n" + "-"*60)
    print("KEY INSIGHT:")
    print("-"*60)
    print("At an edge, pixels have DIFFERENT intensities.")
    print("• Gaussian: Averages them anyway → blurred edge")
    print("• Bilateral: Low range weight → edge preserved")
    print("\nAt a flat region, pixels have SIMILAR intensities.")
    print("• Both filters: High weights → effective smoothing")
    print("="*60)
    
    print("\nPARAMETER GUIDELINES:")
    print("-"*60)
    print("σ_space (Spatial):")
    print("  • Larger → considers more distant neighbors")
    print("  • Typical: 50-150")
    print("\nσ_color (Range/Intensity):")
    print("  • Larger → tolerates more intensity difference")
    print("  • Small σ_color → very strong edge preservation")
    print("  • Large σ_color → closer to Gaussian blur")
    print("  • Typical: 50-150")
    print("\nDiameter (d):")
    print("  • Defines neighborhood size")
    print("  • Typical: 5, 7, 9")
    print("="*60)
    
    # Save results
    cv2.imwrite('q10_bilateral_manual.png', bilateral_manual)
    cv2.imwrite('q10_bilateral_opencv.png', bilateral_opencv)
    cv2.imwrite('q10_gaussian_opencv.png', gaussian_opencv)
    
    print("\nResults saved!")
    print("\nNOTE: Manual implementation is much slower than OpenCV's")
    print("      optimized version, but produces equivalent results.")

if __name__ == "__main__":
    main()