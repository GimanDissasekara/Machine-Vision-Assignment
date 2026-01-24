import cv2
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os

def compute_derivative_gaussian_kernel(size, sigma, direction):
    """
    (b) Compute derivative of Gaussian kernel
    Based on: ∂G/∂x = -(x/σ²)G(x,y) and ∂G/∂y = -(y/σ²)G(x,y)
    """
    kernel = np.zeros((size, size))
    center = size // 2
    
    for i in range(size):
        for j in range(size):
            x = j - center
            y = i - center
            
            # Compute Gaussian
            gaussian = np.exp(-(x**2 + y**2) / (2 * sigma**2))
            
            # Apply derivative
            if direction == 'x':
                kernel[i, j] = -(x / (sigma**2)) * gaussian
            elif direction == 'y':
                kernel[i, j] = -(y / (sigma**2)) * gaussian
    
    # Normalize by sum of absolute values
    kernel = kernel / np.sum(np.abs(kernel))
    
    return kernel

def visualize_kernel_3d(kernel, title):
    """
    (c) Visualize derivative of Gaussian kernel as 3D surface plot
    """
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    size = kernel.shape[0]
    x = np.arange(size)
    y = np.arange(size)
    X, Y = np.meshgrid(x, y)
    
    surf = ax.plot_surface(X, Y, kernel, cmap='RdBu', edgecolor='none', alpha=0.9)
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Kernel Value')
    ax.set_title(title)
    
    fig.colorbar(surf, ax=ax, shrink=0.5)
    
    return fig

def apply_dog_filters(image, kernel_x, kernel_y):
    """
    (d) Apply derivative of Gaussian kernels to get image gradients
    """
    grad_x = cv2.filter2D(image, cv2.CV_64F, kernel_x)
    grad_y = cv2.filter2D(image, cv2.CV_64F, kernel_y)
    
    # Convert to displayable format (shift and scale)
    grad_x_display = np.uint8(np.clip(grad_x + 128, 0, 255))
    grad_y_display = np.uint8(np.clip(grad_y + 128, 0, 255))
    
    # Compute gradient magnitude
    magnitude = np.sqrt(grad_x**2 + grad_y**2)
    magnitude = np.uint8(np.clip(magnitude, 0, 255))
    
    return grad_x_display, grad_y_display, magnitude

def apply_sobel(image):
    """
    (e) Apply Sobel operator using OpenCV
    """
    sobel_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
    
    # Convert to displayable format
    sobel_x_display = np.uint8(np.clip(sobel_x + 128, 0, 255))
    sobel_y_display = np.uint8(np.clip(sobel_y + 128, 0, 255))
    
    # Compute magnitude
    magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
    magnitude = np.uint8(np.clip(magnitude, 0, 255))
    
    return sobel_x_display, sobel_y_display, magnitude

def main():
    print("Question 6: Derivative of Gaussian")
    print("="*60)
    
    # (a) Mathematical derivation
    print("\n(a) Mathematical Derivation:")
    print("Given: G(x,y) = (1/2πσ²) exp(-(x²+y²)/2σ²)")
    print("\nDerivative with respect to x:")
    print("∂G/∂x = (1/2πσ²) · exp(...) · ∂/∂x[-(x²+y²)/2σ²]")
    print("      = (1/2πσ²) · exp(...) · (-2x/2σ²)")
    print("      = -(x/σ²) · G(x,y)  ✓")
    print("\nSimilarly for y:")
    print("∂G/∂y = -(y/σ²) · G(x,y)  ✓")
    
    # Load image
    image_path = os.path.join('Sources', 'highlights_and_shadows.jpg')  # Replace with your image path
    image = cv2.imread(image_path)
    
    if image is None:
        print(f"\nError: Could not load image from {image_path}")
        return
    
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # (b) Compute 5x5 derivative of Gaussian kernels
    print("\n(b) Computing 5×5 Derivative of Gaussian kernels (σ=2)...")
    dog_kernel_x_5x5 = compute_derivative_gaussian_kernel(5, 2, 'x')
    dog_kernel_y_5x5 = compute_derivative_gaussian_kernel(5, 2, 'y')
    
    print("\n5×5 DoG Kernel (X-direction):")
    print(dog_kernel_x_5x5)
    print("\n5×5 DoG Kernel (Y-direction):")
    print(dog_kernel_y_5x5)
    
    # (c) Compute 51x51 kernel for visualization
    print("\n(c) Visualizing 51×51 DoG kernel as 3D surface...")
    dog_kernel_x_51x51 = compute_derivative_gaussian_kernel(51, 2, 'x')
    
    fig_3d = visualize_kernel_3d(dog_kernel_x_51x51, 
                                  '51×51 Derivative of Gaussian (X-direction, σ=2)')
    plt.savefig('q6_dog_kernel_3d.png', dpi=150, bbox_inches='tight')
    
    # (d) Apply DoG filters
    print("\n(d) Applying Derivative of Gaussian filters...")
    dog_grad_x, dog_grad_y, dog_magnitude = apply_dog_filters(gray, dog_kernel_x_5x5, dog_kernel_y_5x5)
    
    # (e) Apply Sobel operator
    print("\n(e) Applying Sobel operator for comparison...")
    sobel_grad_x, sobel_grad_y, sobel_magnitude = apply_sobel(gray)
    
    # Display results
    fig, axes = plt.subplots(3, 3, figsize=(15, 15))
    
    axes[0, 0].imshow(gray, cmap='gray')
    axes[0, 0].set_title('Original Grayscale')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(dog_kernel_x_5x5, cmap='RdBu')
    axes[0, 1].set_title('DoG Kernel (X-direction)')
    axes[0, 1].axis('off')
    plt.colorbar(axes[0, 1].images[0], ax=axes[0, 1])
    
    axes[0, 2].imshow(dog_kernel_y_5x5, cmap='RdBu')
    axes[0, 2].set_title('DoG Kernel (Y-direction)')
    axes[0, 2].axis('off')
    plt.colorbar(axes[0, 2].images[0], ax=axes[0, 2])
    
    axes[1, 0].imshow(dog_grad_x, cmap='gray')
    axes[1, 0].set_title('DoG Gradient X')
    axes[1, 0].axis('off')
    
    axes[1, 1].imshow(dog_grad_y, cmap='gray')
    axes[1, 1].set_title('DoG Gradient Y')
    axes[1, 1].axis('off')
    
    axes[1, 2].imshow(dog_magnitude, cmap='gray')
    axes[1, 2].set_title('DoG Gradient Magnitude')
    axes[1, 2].axis('off')
    
    axes[2, 0].imshow(sobel_grad_x, cmap='gray')
    axes[2, 0].set_title('Sobel Gradient X')
    axes[2, 0].axis('off')
    
    axes[2, 1].imshow(sobel_grad_y, cmap='gray')
    axes[2, 1].set_title('Sobel Gradient Y')
    axes[2, 1].axis('off')
    
    axes[2, 2].imshow(sobel_magnitude, cmap='gray')
    axes[2, 2].set_title('Sobel Gradient Magnitude')
    axes[2, 2].axis('off')
    
    plt.tight_layout()
    plt.savefig('q6_results.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # Comparison analysis
    print("\n(e) Comparison: DoG vs Sobel")
    print("-" * 60)
    print("Derivative of Gaussian (DoG):")
    print("  - Combines Gaussian smoothing with derivative operation")
    print("  - Better noise suppression (σ parameter controls smoothing)")
    print("  - Smoother gradient responses")
    print("  - Rotationally symmetric")
    print("  - More computationally expensive")
    print("\nSobel Operator:")
    print("  - Fixed 3×3 kernel, faster computation")
    print("  - Less smoothing, more sensitive to noise")
    print("  - Sharper edge localization")
    print("  - Good balance of speed and accuracy")
    print("\nObservations:")
    print("  - DoG produces smoother gradients with less noise")
    print("  - Sobel has crisper edges but more noise artifacts")
    print("  - DoG edges may be slightly less localized due to smoothing")
    
    # Save results
    cv2.imwrite('q6_dog_grad_x.png', dog_grad_x)
    cv2.imwrite('q6_dog_grad_y.png', dog_grad_y)
    cv2.imwrite('q6_dog_magnitude.png', dog_magnitude)
    cv2.imwrite('q6_sobel_magnitude.png', sobel_magnitude)
    
    print("\nResults saved as: q6_results.png, q6_dog_kernel_3d.png, and gradient images")

if __name__ == "__main__":
    main()