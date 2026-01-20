import os
import numpy as np
from PIL import Image
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt


def ensure_dir(path: str) -> None:
    if not os.path.exists(path):
        os.makedirs(path)


def _srgb_to_linear(c):
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def _linear_to_srgb(c):
    return np.where(c <= 0.0031308, 12.92 * c, 1.055 * (np.maximum(c, 0) ** (1 / 2.4)) - 0.055)


def rgb_to_lab(rgb: np.ndarray) -> np.ndarray:
    """Convert an sRGB image in [0, 1] to L*a*b* (D65)."""

    # sRGB to linear RGB
    rgb_linear = _srgb_to_linear(rgb)

    # Linear RGB to XYZ (D65)
    X = 0.4124564 * rgb_linear[..., 0] + 0.3575761 * rgb_linear[..., 1] + 0.1804375 * rgb_linear[..., 2]
    Y = 0.2126729 * rgb_linear[..., 0] + 0.7151522 * rgb_linear[..., 1] + 0.0721750 * rgb_linear[..., 2]
    Z = 0.0193339 * rgb_linear[..., 0] + 0.1191920 * rgb_linear[..., 1] + 0.9503041 * rgb_linear[..., 2]

    # Normalize by reference white
    Xn, Yn, Zn = 0.95047, 1.0, 1.08883
    x = X / Xn
    y = Y / Yn
    z = Z / Zn

    delta = 6 / 29

    def f(t):
        return np.where(t > delta ** 3, np.cbrt(t), (t / (3 * delta ** 2)) + (4 / 29))

    fx, fy, fz = f(x), f(y), f(z)

    L = (116 * fy) - 16
    a = 500 * (fx - fy)
    b = 200 * (fy - fz)

    lab = np.stack([L, a, b], axis=-1)
    return lab


def lab_to_rgb(lab: np.ndarray) -> np.ndarray:
    """Convert L*a*b* (D65) to sRGB in [0, 1]."""

    L, a, b = lab[..., 0], lab[..., 1], lab[..., 2]
    fy = (L + 16) / 116
    fx = fy + (a / 500)
    fz = fy - (b / 200)

    delta = 6 / 29

    def f_inv(t):
        return np.where(t > delta, t ** 3, 3 * (delta ** 2) * (t - (4 / 29)))

    Xn, Yn, Zn = 0.95047, 1.0, 1.08883
    X = Xn * f_inv(fx)
    Y = Yn * f_inv(fy)
    Z = Zn * f_inv(fz)

    # XYZ to linear RGB
    r_lin = 3.2404542 * X - 1.5371385 * Y - 0.4985314 * Z
    g_lin = -0.9692660 * X + 1.8760108 * Y + 0.0415560 * Z
    b_lin = 0.0556434 * X - 0.2040259 * Y + 1.0572252 * Z

    rgb_lin = np.stack([r_lin, g_lin, b_lin], axis=-1)
    rgb = _linear_to_srgb(rgb_lin)
    return np.clip(rgb, 0.0, 1.0)


def gamma_correct_lab(image_path: str, gamma: float, results_dir: str) -> None:
    """
    Apply gamma correction only on the L channel (0-100 range) in L*a*b* space.
    Saves the corrected image and a figure with original/corrected images and their L histograms.
    """

    rgb = Image.open(image_path).convert('RGB')
    rgb_np = np.asarray(rgb).astype(np.float32) / 255.0

    lab = rgb_to_lab(rgb_np)
    l_original = lab[..., 0] / 100.0  # normalize to [0, 1]

    l_corrected = np.power(np.clip(l_original, 0.0, 1.0), gamma)
    lab_corrected = lab.copy()
    lab_corrected[..., 0] = l_corrected * 100.0

    rgb_corrected = lab_to_rgb(lab_corrected)

    # Save corrected RGB image
    corrected_uint8 = np.clip(rgb_corrected * 255.0, 0, 255).astype(np.uint8)
    corrected_image = Image.fromarray(corrected_uint8)
    corrected_image_path = os.path.join(results_dir, 'gamma_corrected.png')
    corrected_image.save(corrected_image_path)

    # Plot original & corrected images + histograms of L channel
    fig, axs = plt.subplots(2, 2, figsize=(10, 8))

    axs[0, 0].imshow(rgb_np)
    axs[0, 0].set_title('Original RGB')
    axs[0, 0].axis('off')

    axs[0, 1].imshow(rgb_corrected)
    axs[0, 1].set_title(f'Gamma Corrected (γ={gamma})')
    axs[0, 1].axis('off')

    axs[1, 0].hist(l_original.ravel(), bins=256, range=(0, 1), color='blue', alpha=0.7)
    axs[1, 0].set_title('Original L Histogram')
    axs[1, 0].set_xlim(0, 1)

    axs[1, 1].hist(l_corrected.ravel(), bins=256, range=(0, 1), color='green', alpha=0.7)
    axs[1, 1].set_title('Gamma Corrected L Histogram')
    axs[1, 1].set_xlim(0, 1)

    plt.tight_layout()
    figure_path = os.path.join(results_dir, 'gamma_correction_overview.png')
    plt.savefig(figure_path, dpi=200)
    plt.close(fig)

    print(f"Gamma correction complete (gamma={gamma}).")
    print(f"  Corrected image saved to: {corrected_image_path}")
    print(f"  Figure saved to:         {figure_path}")


def manual_hist_equalization(gray: np.ndarray):
    """
    Perform histogram equalization on a grayscale image in [0, 1].
    Returns equalized image (float [0, 1]), original hist, equalized hist.
    """

    gray_clipped = np.clip(gray, 0.0, 1.0)
    gray_uint8 = (gray_clipped * 255).astype(np.uint8)

    hist = np.bincount(gray_uint8.flatten(), minlength=256)
    cdf = hist.cumsum()

    # Mask zeros to avoid divide-by-zero and scale CDF
    cdf_masked = np.ma.masked_equal(cdf, 0)
    cdf_min = cdf_masked.min()
    cdf_range = cdf_masked.max() - cdf_min
    cdf_scaled = (cdf_masked - cdf_min) * 255 / cdf_range
    cdf_filled = np.ma.filled(cdf_scaled, 0).astype(np.uint8)

    equalized_uint8 = cdf_filled[gray_uint8]
    equalized = equalized_uint8.astype(np.float32) / 255.0
    hist_eq = np.bincount(equalized_uint8.flatten(), minlength=256)

    return equalized, hist, hist_eq


def apply_hist_equalization(image_path: str, results_dir: str) -> None:
    gray = Image.open(image_path).convert('L')
    gray_np = np.asarray(gray).astype(np.float32) / 255.0

    eq_np, hist_orig, hist_eq = manual_hist_equalization(gray_np)
    eq_uint8 = np.clip(eq_np * 255.0, 0, 255).astype(np.uint8)
    eq_image = Image.fromarray(eq_uint8)

    equalized_image_path = os.path.join(results_dir, 'runway_hist_equalized.png')
    eq_image.save(equalized_image_path)

    fig, axs = plt.subplots(2, 2, figsize=(10, 8))

    axs[0, 0].imshow(gray_np, cmap='gray', vmin=0, vmax=1)
    axs[0, 0].set_title('Original Runway (Grayscale)')
    axs[0, 0].axis('off')

    axs[0, 1].imshow(eq_np, cmap='gray', vmin=0, vmax=1)
    axs[0, 1].set_title('Histogram Equalized')
    axs[0, 1].axis('off')

    axs[1, 0].bar(np.arange(256), hist_orig, color='blue')
    axs[1, 0].set_title('Original Histogram')
    axs[1, 0].set_xlim(0, 255)

    axs[1, 1].bar(np.arange(256), hist_eq, color='green')
    axs[1, 1].set_title('Equalized Histogram')
    axs[1, 1].set_xlim(0, 255)

    plt.tight_layout()
    figure_path = os.path.join(results_dir, 'runway_hist_equalization_overview.png')
    plt.savefig(figure_path, dpi=200)
    plt.close(fig)

    print("Histogram equalization complete.")
    print(f"  Equalized image saved to: {equalized_image_path}")
    print(f"  Figure saved to:         {figure_path}")


def main():
    root = os.getcwd()
    sources_dir = os.path.join(root, 'Sources')
    results_dir = os.path.join(root, 'Results')
    ensure_dir(results_dir)

    gamma_image = os.path.join(sources_dir, 'highlights_and_shadows.jpg')
    runway_image = os.path.join(sources_dir, 'runway.png')

    gamma_value = 0.6  # brighten dark tones while preserving highlights

    print("Applying gamma correction on L channel (L*a*b* space)...")
    gamma_correct_lab(gamma_image, gamma_value, results_dir)

    print("\nApplying custom histogram equalization on runway image...")
    apply_hist_equalization(runway_image, results_dir)


if __name__ == '__main__':
    main()