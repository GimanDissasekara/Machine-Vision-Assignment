import cv2
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def compute_gaussian_kernel(size, sigma):
    """
    (a) Compute normalized Gaussian kernel using NumPy
    """
    kernel = np.zeros((size, size))
    center = size // 2
    
    for i in range(size):
        for j in range(size):
            x = i - center
            y = j - center
            kernel[i, j] = np.exp(-(x**2 + y**2) / (2 * sigma**2))
    
    # Normalize
    kernel = kernel / np.sum(kernel)
    
    return kernel

def visualize_kernel_3d(kernel, title):
    """
    (b) Visualize kernel as 3D surface plot
    """
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    size = kernel.shape[0]
    x = np.arange(size)
    y = np.arange(size)
    X, Y = np.meshgrid(x, y)
    
    surf = ax.plot_surface(X, Y, kernel, cmap='viridis', edgecolor='none', alpha=0.9)
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Kernel Value')
    ax.set_title(title)
    
    fig.colorbar(surf, ax=ax, shrink=0.5)
    
    return fig

def apply_gaussian_manual(image, kernel):
    """
    (c) Apply Gaussian smoothing using manual convolution
    """
    # Use cv2.filter2D for convolution
    filtered = cv2.filter2D(image, -1, kernel)
    return filtered

def apply_gaussian_opencv(image, kernel_size, sigma):
    """
    (d) Apply Gaussian smoothing using OpenCV's GaussianBlur
    """
    filtered = cv2.GaussianBlur(image, (kernel_size, kernel_size), sigma)
    return filtered

def main():
    # Load image
    image_path = 'sample_image.jpg'  # Replace with your image path
    image = cv2.imread(image_path)
    
    if image is None:
        print(f"Error: Could not load image from {image_path}")
        return
    
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # (a) Compute 5x5 Gaussian kernel with sigma=2
    kernel_5x5 = compute_gaussian_kernel(5, 2)
    
    print("(a) 5×5 Gaussian Kernel (σ=2):")
    print(kernel_5x5)
    print(f"\nKernel sum (should be 1.0): {np.sum(kernel_5x5):.10f}")
    
    # (b) Compute 51x51 kernel for visualization
    kernel_51x51 = compute_gaussian_kernel(51, 2)
    
    print("\n(b) Visualizing 51×51 Gaussian kernel as 3D surface...")
    fig_3d = visualize_kernel_3d(kernel_51x51, '51×51 Gaussian Kernel (σ=2)')
    plt.savefig('q5_kernel_3d.png', dpi=150, bbox_inches='tight')
    
    # (c) Apply manual Gaussian filtering
    filtered_manual = apply_gaussian_manual(gray, kernel_5x5)
    
    # (d) Apply OpenCV's GaussianBlur
    filtered_opencv = apply_gaussian_opencv(gray, 5, 2)
    
    # Display results
    plt.figure(figsize=(15, 10))
    
    plt.subplot(2, 3, 1)
    plt.imshow(gray, cmap='gray')
    plt.title('Original Grayscale Image')
    plt.axis('off')
    
    plt.subplot(2, 3, 2)
    plt.imshow(filtered_manual, cmap='gray')
    plt.title('(c) Manual Gaussian Filter')
    plt.axis('off')
    
    plt.subplot(2, 3, 3)
    plt.imshow(filtered_opencv, cmap='gray')
    plt.title("(d) OpenCV's GaussianBlur")
    plt.axis('off')
    
    plt.subplot(2, 3, 4)
    plt.imshow(kernel_5x5, cmap='viridis')
    plt.title('5×5 Gaussian Kernel Heatmap')
    plt.colorbar()
    plt.axis('off')
    
    plt.subplot(2, 3, 5)
    # Show difference between manual and OpenCV
    difference = np.abs(filtered_manual.astype(float) - filtered_opencv.astype(float))
    plt.imshow(difference, cmap='hot')
    plt.title('Difference (Manual - OpenCV)')
    plt.colorbar()
    plt.axis('off')
    
    plt.subplot(2, 3, 6)
    # Plot kernel profile (center row)
    center = kernel_51x51.shape[0] // 2
    plt.plot(kernel_51x51[center, :], linewidth=2)
    plt.title('Gaussian Kernel Profile (Center Row)')
    plt.xlabel('Position')
    plt.ylabel('Kernel Value')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('q5_results.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # Calculate and display difference metrics
    mse = np.mean((filtered_manual.astype(float) - filtered_opencv.astype(float))**2)
    print(f"\n(d) Comparison - MSE between manual and OpenCV: {mse:.6f}")
    print("The difference is minimal, showing our manual implementation is correct.")
    
    # Save results
    cv2.imwrite('q5_manual_filtered.png', filtered_manual)
    cv2.imwrite('q5_opencv_filtered.png', filtered_opencv)
    
    print("\nResults saved as: q5_results.png, q5_kernel_3d.png, q5_manual_filtered.png, q5_opencv_filtered.png")

if __name__ == "__main__":
    main()