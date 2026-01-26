import numpy as np
from PIL import Image
import os

def zoom_nearest_neighbor(image, scale_factor):
    """
    Zoom an image using nearest-neighbor interpolation.
    
    Args:
        image: Input image as numpy array (H, W, C)
        scale_factor: Zoom factor (s > 0)
    
    Returns:
        Zoomed image as numpy array
    """
    if scale_factor <= 0 or scale_factor > 10:
        raise ValueError("Scale factor must be in range (0, 10]")
    
    h, w = image.shape[:2]
    new_h = int(h * scale_factor)
    new_w = int(w * scale_factor)
    
    # Handle grayscale and color images
    if len(image.shape) == 3:
        channels = image.shape[2]
        zoomed = np.zeros((new_h, new_w, channels), dtype=image.dtype)
    else:
        zoomed = np.zeros((new_h, new_w), dtype=image.dtype)
    
    # Create coordinate mapping
    for i in range(new_h):
        for j in range(new_w):
            # Map new coordinates to original coordinates
            src_i = int(i / scale_factor)
            src_j = int(j / scale_factor)
            
            # Ensure we don't go out of bounds
            src_i = min(src_i, h - 1)
            src_j = min(src_j, w - 1)
            
            zoomed[i, j] = image[src_i, src_j]
    
    return zoomed

def compute_normalized_ssd(img1, img2):
    """
    Compute normalized Sum of Squared Differences between two images.
    
    Args:
        img1, img2: Images as numpy arrays
    
    Returns:
        Normalized SSD value
    """
    if img1.shape != img2.shape:
        raise ValueError("Images must have the same dimensions")
    
    # Convert to float for computation
    img1_float = img1.astype(np.float64)
    img2_float = img2.astype(np.float64)
    
    # Compute SSD
    ssd = np.sum((img1_float - img2_float) ** 2)
    
    # Normalize by number of pixels and channels
    n_pixels = img1.size
    normalized_ssd = ssd / n_pixels
    
    return normalized_ssd

def test_zoom_algorithm():
    """
    Test the zoom algorithm by comparing zoomed small images with originals.
    """
    # Define image pairs (small, large)
    zooming_dir = os.path.join('Sources', 'images_for_zooming')
    output_subdir = 'Results'
    os.makedirs(output_subdir, exist_ok=True)
    image_pairs = [
        ('im01small.png', 'im01.png'),
      import numpy as np
from PIL import Image
import os

def zoom_nearest_neighbor(image, scale_factor):
    """
    Zoom an image using nearest-neighbor interpolation.
    
    Args:
        image: Input image as numpy array (H, W, C)
        scale_factor: Zoom factor (s > 0)
    
    Returns:
        Zoomed image as numpy array
    """
    if scale_factor <= 0 or scale_factor > 10:
        raise ValueError("Scale factor must be in range (0, 10]")
    
    h, w = image.shape[:2]
    new_h = int(h * scale_factor)
    new_w = int(w * scale_factor)
    
    # Handle grayscale and color images
    if len(image.shape) == 3:
        channels = image.shape[2]
        zoomed = np.zeros((new_h, new_w, channels), dtype=image.dtype)
    else:
        zoomed = np.zeros((new_h, new_w), dtype=image.dtype)
    
    # Create coordinate mapping
    for i in range(new_h):
        for j in range(new_w):
            # Map new coordinates to original coordinates
            src_i = int(i / scale_factor)
            src_j = int(j / scale_factor)
            
            # Ensure we don't go out of bounds
            src_i = min(src_i, h - 1)
            src_j = min(src_j, w - 1)
            
            zoomed[i, j] = image[src_i, src_j]
    
    return zoomed

def compute_normalized_ssd(img1, img2):
    """
    Compute normalized Sum of Squared Differences between two images.
    
    Args:
        img1, img2: Images as numpy arrays
    
    Returns:
        Normalized SSD value
    """
    if img1.shape != img2.shape:
        raise ValueError("Images must have the same dimensions")
    
    # Convert to float for computation
    img1_float = img1.astype(np.float64)
    img2_float = img2.astype(np.float64)
    
    # Compute SSD
    ssd = np.sum((img1_float - img2_float) ** 2)
    
    # Normalize by number of pixels and channels
    n_pixels = img1.size
    normalized_ssd = ssd / n_pixels
    
    return normalized_ssd

def test_zoom_algorithm():
    """
    Test the zoom algorithm by comparing zoomed small images with originals.
    """
    # Define image pairs (small, large)
    image_pairs = [
        ('im01small.png', 'im01.png'),
        ('im02small.png', 'im02.png'),
        ('im03small.png', 'im03.png'),
        ('taylor_small.jpg', 'taylor.jpg')
    ]
    
    print("Nearest-Neighbor Interpolation Test Results")
    print("=" * 60)
    
    for small_file, large_file in image_pairs:
        if not os.path.exists(small_file) or not os.path.exists(large_file):
            print(f"Skipping {small_file} - file not found")
            continue
        
        # Load images
        small_img = np.array(Image.open(small_file))
        large_img = np.array(Image.open(large_file))
        
        # Calculate scale factor
        scale_h = large_img.shape[0] / small_img.shape[0]
        scale_w = large_img.shape[1] / small_img.shape[1]
        scale_factor = (scale_h + scale_w) / 2  # Average scale
        
        print(f"\nProcessing: {small_file} -> {large_file}")
        print(f"Small image size: {small_img.shape}")
        print(f"Large image size: {large_img.shape}")
        print(f"Scale factor: {scale_factor:.2f}")
        
        # Zoom the small image
        zoomed_img = zoom_nearest_neighbor(small_img, scale_factor)
        
        # Resize zoomed image to exactly match large image size if needed
        if zoomed_img.shape[:2] != large_img.shape[:2]:
            zoomed_pil = Image.fromarray(zoomed_img)
import numpy as np
from PIL import Image
import os

def zoom_nearest_neighbor(image, scale_factor):
    """
    Zoom an image using nearest-neighbor interpolation.
    
    Args:
        image: Input image as numpy array (H, W, C)
        scale_factor: Zoom factor (s > 0)
    
    Returns:
        Zoomed image as numpy array
    """
    if scale_factor <= 0 or scale_factor > 10:
        raise ValueError("Scale factor must be in range (0, 10]")
    
    h, w = image.shape[:2]
    new_h = int(h * scale_factor)
    new_w = int(w * scale_factor)
    
    # Handle grayscale and color images
    if len(image.shape) == 3:
        channels = image.shape[2]
        zoomed = np.zeros((new_h, new_w, channels), dtype=image.dtype)
    else:
        zoomed = np.zeros((new_h, new_w), dtype=image.dtype)
    
    # Create coordinate mapping
    for i in range(new_h):
        for j in range(new_w):
            # Map new coordinates to original coordinates
            src_i = int(i / scale_factor)
            src_j = int(j / scale_factor)
            
            # Ensure we don't go out of bounds
            src_i = min(src_i, h - 1)
            src_j = min(src_j, w - 1)
            
            zoomed[i, j] = image[src_i, src_j]
    
    return zoomed

def compute_normalized_ssd(img1, img2):
    """
    Compute normalized Sum of Squared Differences between two images.
    
    Args:
        img1, img2: Images as numpy arrays
    
    Returns:
        Normalized SSD value
    """
    if img1.shape != img2.shape:
        raise ValueError("Images must have the same dimensions")
    
    # Convert to float for computation
    img1_float = img1.astype(np.float64)
    img2_float = img2.astype(np.float64)
    
    # Compute SSD
    ssd = np.sum((img1_float - img2_float) ** 2)
    
    # Normalize by number of pixels and channels
    n_pixels = img1.size
    normalized_ssd = ssd / n_pixels
    
    return normalized_ssd

def test_zoom_algorithm():
    """
    Test the zoom algorithm by comparing zoomed small images with originals.
    """
    # Define image pairs (small, large)
    image_pairs = [
        ('im01small.png', 'im01.png'),
        ('im02small.png', 'im02.png'),
        ('im03small.png', 'im03.png'),
        ('taylor_small.jpg', 'taylor.jpg')
    ]
    
    print("Nearest-Neighbor Interpolation Test Results")
    print("=" * 60)
    
    for small_file, large_file in image_pairs:
        if not os.path.exists(small_file) or not os.path.exists(large_file):
            print(f"Skipping {small_file} - file not found")
            continue
        
        # Load images
        small_img = np.array(Image.open(small_file))
        large_img = np.array(Image.open(large_file))
        
        # Calculate scale factor
        scale_h = large_img.shape[0] / small_img.shape[0]
        scale_w = large_img.shape[1] / small_img.shape[1]
        scale_factor = (scale_h + scale_w) / 2  # Average scale
        
        print(f"\nProcessing: {small_file} -> {large_file}")
        print(f"Small image size: {small_img.shape}")
        print(f"Large image size: {large_img.shape}")
        print(f"Scale factor: {scale_factor:.2f}")
        
        # Zoom the small image
        zoomed_img = zoom_nearest_neighbor(small_img, scale_factor)
        
        # Resize zoomed image to exactly match large image size if needed
        if zoomed_img.shape[:2] != large_img.shape[:2]:
            zoomed_pil = Image.fromarray(zoomed_img)
            zoomed_pil = zoomed_pil.resize((large_img.shape[1], large_img.shape[0]), 
                                           Image.NEAREST)
            zoomed_img = np.array(zoomed_pil)
        
        print(f"Zoomed image size: {zoomed_img.shape}")
        
        # Compute normalized SSD
import numpy as np
from PIL import Image
import os

def zoom_nearest_neighbor(image, scale_factor):
    """
    Zoom an image using nearest-neighbor interpolation.
    
    Args:
        image: Input image as numpy array (H, W, C)
        scale_factor: Zoom factor (s > 0)
    
    Returns:
        Zoomed image as numpy array
    """
    if scale_factor <= 0 or scale_factor > 10:
        raise ValueError("Scale factor must be in range (0, 10]")
    
    h, w = image.shape[:2]
    new_h = int(h * scale_factor)
    new_w = int(w * scale_factor)
    
    # Handle grayscale and color images
    if len(image.shape) == 3:
        channels = image.shape[2]
        zoomed = np.zeros((new_h, new_w, channels), dtype=image.dtype)
    else:
        zoomed = np.zeros((new_h, new_w), dtype=image.dtype)
    
    # Create coordinate mapping
    for i in range(new_h):
        for j in range(new_w):
            # Map new coordinates to original coordinates
            src_i = int(i / scale_factor)
            src_j = int(j / scale_factor)
            
            # Ensure we don't go out of bounds
            src_i = min(src_i, h - 1)
            src_j = min(src_j, w - 1)
            
            zoomed[i, j] = image[src_i, src_j]
    
    return zoomed

def compute_normalized_ssd(img1, img2):
    """
    Compute normalized Sum of Squared Differences between two images.
    
    Args:
        img1, img2: Images as numpy arrays
    
    Returns:
        Normalized SSD value
    """
    if img1.shape != img2.shape:
        raise ValueError("Images must have the same dimensions")
    
    # Convert to float for computation
    img1_float = img1.astype(np.float64)
    img2_float = img2.astype(np.float64)
    
    # Compute SSD
    ssd = np.sum((img1_float - img2_float) ** 2)
    
    # Normalize by number of pixels and channels
    n_pixels = img1.size
    normalized_ssd = ssd / n_pixels
    
    return normalized_ssd

def test_zoom_algorithm():
    """
    Test the zoom algorithm by comparing zoomed small images with originals.
    """
    # Define image pairs (small, large)
    image_pairs = [
        ('im01small.png', 'im01.png'),
        ('im02small.png', 'im02.png'),
        ('im03small.png', 'im03.png'),
        ('taylor_small.jpg', 'taylor.jpg')
    ]
    
    print("Nearest-Neighbor Interpolation Test Results")
    print("=" * 60)
    
    for small_file, large_file in image_pairs:
        if not os.path.exists(small_file) or not os.path.exists(large_file):
            print(f"Skipping {small_file} - file not found")
            continue
        
        # Load images
        small_img = np.array(Image.open(small_file))
        large_img = np.array(Image.open(large_file))
        
        # Calculate scale factor
        scale_h = large_img.shape[0] / small_img.shape[0]
        scale_w = large_img.shape[1] / small_img.shape[1]
        scale_factor = (scale_h + scale_w) / 2  # Average scale
        
        print(f"\nProcessing: {small_file} -> {large_file}")
        print(f"Small image size: {small_img.shape}")
        print(f"Large image size: {large_img.shape}")
        print(f"Scale factor: {scale_factor:.2f}")
        
        # Zoom the small image
        zoomed_img = zoom_nearest_neighbor(small_img, scale_factor)
        
        # Resize zoomed image to exactly match large image size if needed
        if zoomed_img.shape[:2] != large_img.shape[:2]:
            zoomed_pil = Image.fromarray(zoomed_img)
            zoomed_pil = zoomed_pil.resize((large_img.shape[1], large_img.shape[0]), 
                                           Image.NEAREST)
            zoomed_img = np.array(zoomed_pil)
        
        print(f"Zoomed image size: {zoomed_img.shape}")
        
        # Compute normalized SSD
        ssd = compute_normalized_ssd(zoomed_img, large_img)
        print(f"Normalized SSD: {ssd:.4f}")
        
        # Save zoomed image for inspection
        output_file = f"zoomed_nn_{small_file}"
        Image.fromarray(zoomed_img).save(output_file)
        print(f"Saved zoomed image: {output_file}")

if __name__ == "__main__":
    test_zoom_algorithm()22        ssd = compute_normalized_ssd(zoomed_img, large_img)
        print(f"Normalized SSD: {ssd:.4f}")
        
        # Save zoomed image for inspection
        output_file = f"zoomed_nn_{small_file}"
        Image.fromarray(zoomed_img).save(output_file)
        print(f"Saved zoomed image: {output_file}")

if __name__ == "__main__":
    test_zoom_algorithm()22            zoomed_pil = zoomed_pil.resize((large_img.shape[1], large_img.shape[0]), 
                                           Image.NEAREST)
            zoomed_img = np.array(zoomed_pil)
        
        print(f"Zoomed image size: {zoomed_img.shape}")
        
        # Compute normalized SSD
        ssd = compute_normalized_ssd(zoomed_img, large_img)
        print(f"Normalized SSD: {ssd:.4f}")
        
        # Save zoomed image for inspection
        output_file = f"zoomed_nn_{small_file}"
        Image.fromarray(zoomed_img).save(output_file)
        print(f"Saved zoomed image: {output_file}")

if __name__ == "__main__":
    test_zoom_algorithm()22  ('im02small.png', 'im02.png'),
        ('im03small.png', 'im03.png'),
        ('taylor_small.jpg', 'taylor.jpg')
    ]
    
    print("Nearest-Neighbor Interpolation Test Results")
    print("=" * 60)
    
    for small_file, large_file in image_pairs:
        small_path = os.path.join(zooming_dir, small_file)
        large_path = os.path.join(zooming_dir, large_file)
        
        if not os.path.exists(small_path) or not os.path.exists(large_path):
            print(f"Skipping {small_file} - file not found")
            continue
        
        # Load images
        small_img = np.array(Image.open(small_path))
        large_img = np.array(Image.open(large_path))
        
        # Calculate scale factor
        scale_h = large_img.shape[0] / small_img.shape[0]
        scale_w = large_img.shape[1] / small_img.shape[1]
        scale_factor = (scale_h + scale_w) / 2  # Average scale
        
        print(f"\nProcessing: {small_file} -> {large_file}")
        print(f"Small image size: {small_img.shape}")
        print(f"Large image size: {large_img.shape}")
        print(f"Scale factor: {scale_factor:.2f}")
        
        # Zoom the small image
        zoomed_img = zoom_nearest_neighbor(small_img, scale_factor)
        
        # Resize zoomed image to exactly match large image size if needed
        if zoomed_img.shape[:2] != large_img.shape[:2]:
            zoomed_pil = Image.fromarray(zoomed_img)
            zoomed_pil = zoomed_pil.resize((large_img.shape[1], large_img.shape[0]), 
                                           Image.NEAREST)
            zoomed_img = np.array(zoomed_pil)
        
        print(f"Zoomed image size: {zoomed_img.shape}")
        
        # Compute normalized SSD
        ssd = compute_normalized_ssd(zoomed_img, large_img)
        print(f"Normalized SSD: {ssd:.4f}")
        
        # Save zoomed image for inspection
        output_file = os.path.join(output_subdir, f"zoomed_nn_{small_file}")
        Image.fromarray(zoomed_img).save(output_file)
        print(f"Saved zoomed image: {output_file}")

if __name__ == "__main__":
    test_zoom_algorithm()