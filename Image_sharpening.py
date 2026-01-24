import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

def unsharp_masking(image, sigma=1.0, strength=1.5):
    """
    Apply unsharp masking for image sharpening
    Formula: sharpened = original + strength * (original - blurred)
    """
    # Blur the image
    blurred = cv2.GaussianBlur(image, (0, 0), sigma)
    
    # Compute the detail (high-frequency component)
    detail = image.astype(float) - blurred.astype(float)
    
    # Add amplified detail back to original
    sharpened = image.astype(float) + strength * detail
    
    # Clip values to valid range
    sharpened = np.clip(sharpened, 0, 255).astype(np.uint8)
    
    return sharpened, blurred, detail

def laplacian_sharpening(image, strength=1.0):
    """
    Apply Laplacian-based sharpening
    """
    # Define Laplacian kernel
    laplacian_kernel = np.array([[0, -1, 0],
                                  [-1, 4, -1],
                                  [0, -1, 0]], dtype=np.float32)
    
    # Apply Laplacian
    laplacian = cv2.filter2D(image.astype(float), -1, laplacian_kernel)
    
    # Add to original
    sharpened = image.astype(float) + strength * laplacian
    
    # Clip values
    sharpened = np.clip(sharpened, 0, 255).astype(np.uint8)
    
    return sharpened, laplacian

def high_boost_filtering(image, sigma=1.0, A=1.5):
    """
    Apply high-boost filtering
    Formula: sharpened = A * original - blurred
    where A >= 1
    """
    # Blur the image
    blurred = cv2.GaussianBlur(image, (0, 0), sigma)
    
    # High-boost filter
    sharpened = A * image.astype(float) - blurred.astype(float)
    
    # Clip values
    sharpened = np.clip(sharpened, 0, 255).astype(np.uint8)
    
    return sharpened, blurred

def main():
    print("Question 9: Image Sharpening")
    print("="*60)
    
    # Load image
    image_path = os.path.join('Sources', 'looking_out.jpg')  # Replace with your image
    image = cv2.imread(image_path)
    
    if image is None:
        print(f"Could not load {image_path}, creating test image...")
        # Create a slightly blurred test image
        image = cv2.imread('sample_image.jpg')
        if image is None:
            # Create synthetic image
            image = np.zeros((400, 400, 3), dtype=np.uint8)
            cv2.circle(image, (200, 200), 100, (255, 255, 255), -1)
            cv2.rectangle(image, (100, 100), (300, 300), (128, 128, 128), 3)
        
        # Blur it to simulate a soft image
        image = cv2.GaussianBlur(image, (5, 5), 2.0)
        cv2.imwrite('test_blurred_image.jpg', image)
        print("Created test blurred image")
    
    # Convert to grayscale
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    print(f"Image shape: {gray.shape}")
    
    # Apply different sharpening methods
    print("\nApplying sharpening methods...")
    
    # Method 1: Unsharp Masking
    print("  1. Unsharp Masking...")
    sharpened_unsharp, blurred, detail = unsharp_masking(gray, sigma=1.0, strength=1.5)
    
    # Method 2: Laplacian Sharpening
    print("  2. Laplacian Sharpening...")
    sharpened_laplacian, laplacian = laplacian_sharpening(gray, strength=1.0)
    
    # Method 3: High-Boost Filtering
    print("  3. High-Boost Filtering...")
    sharpened_highboost, _ = high_boost_filtering(gray, sigma=1.0, A=2.0)
    
    # Display results - Main comparison
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    axes[0, 0].imshow(gray, cmap='gray')
    axes[0, 0].set_title('Original Image')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(sharpened_unsharp, cmap='gray')
    axes[0, 1].set_title('Unsharp Masking\n(σ=1.0, strength=1.5)')
    axes[0, 1].axis('off')
    
    axes[0, 2].imshow(sharpened_laplacian, cmap='gray')
    axes[0, 2].set_title('Laplacian Sharpening\n(strength=1.0)')
    axes[0, 2].axis('off')
    
    axes[1, 0].imshow(sharpened_highboost, cmap='gray')
    axes[1, 0].set_title('High-Boost Filtering\n(A=2.0)')
    axes[1, 0].axis('off')
    
    axes[1, 1].imshow(detail + 128, cmap='gray')
    axes[1, 1].set_title('Detail Component\n(Unsharp Mask)')
    axes[1, 1].axis('off')
    
    axes[1, 2].imshow(np.clip(laplacian + 128, 0, 255).astype(np.uint8), cmap='gray')
    axes[1, 2].set_title('Laplacian Response')
    axes[1, 2].axis('off')
    
    plt.tight_layout()
    plt.savefig('q9_sharpening_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # Zoomed comparison
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    
    # Select a region to zoom
    h, w = gray.shape
    y1, y2 = h//3, 2*h//3
    x1, x2 = w//3, 2*w//3
    
    axes[0].imshow(gray[y1:y2, x1:x2], cmap='gray')
    axes[0].set_title('Original (Zoomed)')
    axes[0].axis('off')
    
    axes[1].imshow(sharpened_unsharp[y1:y2, x1:x2], cmap='gray')
    axes[1].set_title('Unsharp Masking')
    axes[1].axis('off')
    
    axes[2].imshow(sharpened_laplacian[y1:y2, x1:x2], cmap='gray')
    axes[2].set_title('Laplacian')
    axes[2].axis('off')
    
    axes[3].imshow(sharpened_highboost[y1:y2, x1:x2], cmap='gray')
    axes[3].set_title('High-Boost')
    axes[3].axis('off')
    
    plt.tight_layout()
    plt.savefig('q9_zoomed_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # Different strength levels for unsharp masking
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    strengths = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
    
    for idx, strength in enumerate(strengths):
        sharpened, _, _ = unsharp_masking(gray, sigma=1.0, strength=strength)
        row = idx // 3
        col = idx % 3
        axes[row, col].imshow(sharpened, cmap='gray')
        axes[row, col].set_title(f'Unsharp Mask\nStrength = {strength}')
        axes[row, col].axis('off')
    
    plt.tight_layout()
    plt.savefig('q9_strength_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # Analysis
    print("\n" + "="*60)
    print("IMAGE SHARPENING METHODS - ANALYSIS")
    print("="*60)
    
    print("\n1. UNSHARP MASKING:")
    print("   Formula: Sharpened = Original + α × (Original - Blurred)")
    print("   How it works:")
    print("     - Blur the image to extract low-frequency components")
    print("     - Subtract blurred from original to get high-frequency details")
    print("     - Amplify and add back to original")
    print("   Advantages:")
    print("     ✓ Most natural-looking results")
    print("     ✓ Good control via σ and strength parameters")
    print("     ✓ Widely used in photo editing")
    print("   Best for: General-purpose sharpening, photos")
    
    print("\n2. LAPLACIAN SHARPENING:")
    print("   Formula: Sharpened = Original + α × Laplacian(Original)")
    print("   How it works:")
    print("     - Apply Laplacian operator (2nd derivative)")
    print("     - Detects rapid intensity changes (edges)")
    print("     - Add detected edges to original")
    print("   Advantages:")
    print("     ✓ Emphasizes edges strongly")
    print("     ✓ Simple and fast")
    print("   Disadvantages:")
    print("     ✗ More sensitive to noise")
    print("     ✗ Can create halo artifacts")
    print("   Best for: Technical images, when edge emphasis is important")
    
    print("\n3. HIGH-BOOST FILTERING:")
    print("   Formula: Sharpened = A × Original - Blurred")
    print("   How it works:")
    print("     - Similar to unsharp masking with different weighting")
    print("     - A >= 1 controls the boost amount")
    print("   Advantages:")
    print("     ✓ Can produce very sharp results")
    print("     ✓ Simple parameter (A)")
    print("   Best for: Images needing aggressive sharpening")
    
    print("\n" + "-"*60)
    print("RECOMMENDATIONS:")
    print("-"*60)
    print("• For portraits: Unsharp Masking (strength 0.5-1.0)")
    print("• For landscapes: Unsharp Masking (strength 1.0-2.0)")
    print("• For technical drawings: Laplacian (strength 0.5-1.5)")
    print("• For text/documents: High-Boost (A=1.5-2.5)")
    print("\nWARNINGS:")
    print("• Sharpening amplifies noise - use on good quality images")
    print("• Over-sharpening creates halos and artifacts")
    print("• Apply sharpening as final step after other processing")
    print("="*60)
    
    # Save results
    cv2.imwrite('q9_unsharp_masked.png', sharpened_unsharp)
    cv2.imwrite('q9_laplacian_sharpened.png', sharpened_laplacian)
    cv2.imwrite('q9_highboost_sharpened.png', sharpened_highboost)
    
    print("\nResults saved!")

if __name__ == "__main__":
    main()