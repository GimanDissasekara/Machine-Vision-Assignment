# Machine Vision Assignment

This repository contains various implementations of image processing techniques as part of a Machine Vision assignment. The scripts cover filtering, edge detection, transformations, and sharpening.

## Project Structure

- `Sources/`: Contains input images for testing.
- `Results/`: Directory where output images and plots are saved.
- `Gaussian_filtering.py`: Manual and OpenCV implementation of Gaussian smoothing and kernel visualization.
- `Derivative_of_Gaussian.py`: Edge detection using Derivative of Gaussian (DoG) kernels compared with Sobel.
- `Bilateral_filtering.py`: Edge-preserving smoothing implementing a manual Bilateral filter.
- `Salt_and_pepper_noise.py`: Comparison between Gaussian and Median filters for removing impulsive noise.
- `Image_sharpening.py`: Techniques including Unsharp Masking, Laplacian sharpening, and High-Boost filtering.
- `zoom.py`: Image resizing using Nearest-Neighbor and Bilinear interpolation.
- `grayscale.py`: Otsu's thresholding and masked histogram equalization.
- `intensity_transformations.py`: Gamma correction and piecewise linear contrast stretching using PyTorch.
- `lab_gamma_and_hist_eq.py`: Gamma correction in L*a*b* space and histogram equalization.
- `verify_transformations.py`: Verification script for intensity transformations.

## Setup Instructions

### 1. Prerequisites
Ensure you have Python 3.8+ installed.

### 2. Virtual Environment
It is recommended to use a virtual environment:
```powershell
python -m venv .venv
.venv\Scripts\activate
```

### 3. Install Dependencies
Install the required packages using the provided `requirements.txt`:
```powershell
pip install -r requirements.txt
```

## How to Run

You can run any script individually to see its results and analysis. For example:
```powershell
python Gaussian_filtering.py
```
Most scripts will:
1. Load a sample image from the `Sources/` directory (or use a placeholder if not found).
2. Perform image processing operations.
3. Display a comparison plot using Matplotlib.
4. Save the resulting images and plots to the root or `Results/` directory.

## Dependencies

- **OpenCV**: Image loading and basic processing.
- **NumPy**: Numerical operations and manual filter implementations.
- **Matplotlib**: Visualization and plotting.
- **PyTorch & Torchvision**: Tensor-based image transformations.
- **Pillow**: Image reading and color space conversions.
