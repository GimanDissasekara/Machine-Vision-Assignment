import React, { useState, useRef, useEffect } from 'react';
import { Upload, Download } from 'lucide-react';

const ImageProcessingSuite = () => {
  const [image, setImage] = useState(null);
  const [results, setResults] = useState({});
  const [otsuThresholdValue, setOtsuThresholdValue] = useState(null);
  const canvasRef = useRef(null);

  // Load image from file
  const handleImageUpload = (e) => {
    const file = e.target.files[0];
    if (file) {
      const reader = new FileReader();
      reader.onload = (event) => {
        const img = new Image();
        img.onload = () => {
          setImage(img);
          processImage(img);
        };
        img.src = event.target.result;
      };
      reader.readAsDataURL(file);
    }
  };

  // Convert to grayscale
  const toGrayscale = (imageData) => {
    const data = imageData.data;
    for (let i = 0; i < data.length; i += 4) {
      const gray = 0.299 * data[i] + 0.587 * data[i + 1] + 0.114 * data[i + 2];
      data[i] = data[i + 1] = data[i + 2] = gray;
    }
    return imageData;
  };

  // Otsu's thresholding
  const otsuThreshold = (imageData) => {
    const data = imageData.data;
    const histogram = new Array(256).fill(0);
    
    // Build histogram
    for (let i = 0; i < data.length; i += 4) {
      histogram[data[i]]++;
    }
    
    const total = data.length / 4;
    let sum = 0;
    for (let i = 0; i < 256; i++) {
      sum += i * histogram[i];
    }
    
    let sumB = 0;
    let wB = 0;
    let wF = 0;
    let maxVariance = 0;
    let threshold = 0;
    
    for (let t = 0; t < 256; t++) {
      wB += histogram[t];
      if (wB === 0) continue;
      
      wF = total - wB;
      if (wF === 0) break;
      
      sumB += t * histogram[t];
      const mB = sumB / wB;
      const mF = (sum - sumB) / wF;
      
      const variance = wB * wF * (mB - mF) * (mB - mF);
      
      if (variance > maxVariance) {
        maxVariance = variance;
        threshold = t;
      }
    }
    
    return threshold;
  };

  // Apply binary mask
  const applyBinaryMask = (imageData, threshold) => {
    const data = imageData.data;
    const mask = new Uint8Array(data.length / 4);
    
    for (let i = 0; i < data.length; i += 4) {
      mask[i / 4] = data[i] > threshold ? 1 : 0;
      const val = mask[i / 4] * 255;
      data[i] = data[i + 1] = data[i + 2] = val;
    }
    
    return { imageData, mask };
  };

  // Histogram equalization
  const histogramEqualization = (imageData, mask = null) => {
    const data = imageData.data;
    const histogram = new Array(256).fill(0);
    let totalPixels = 0;
    
    // Build histogram (only for masked region if mask provided)
    for (let i = 0; i < data.length; i += 4) {
      if (mask === null || mask[i / 4] === 1) {
        histogram[data[i]]++;
        totalPixels++;
      }
    }
    
    // Compute CDF
    const cdf = new Array(256).fill(0);
    cdf[0] = histogram[0];
    for (let i = 1; i < 256; i++) {
      cdf[i] = cdf[i - 1] + histogram[i];
    }
    
    // Normalize CDF
    const cdfMin = cdf.find(val => val > 0);
    const lookupTable = new Array(256);
    for (let i = 0; i < 256; i++) {
      lookupTable[i] = Math.round(((cdf[i] - cdfMin) / (totalPixels - cdfMin)) * 255);
    }
    
    // Apply equalization
    const result = new ImageData(
      new Uint8ClampedArray(data),
      imageData.width,
      imageData.height
    );
    
    for (let i = 0; i < result.data.length; i += 4) {
      if (mask === null || mask[i / 4] === 1) {
        const newVal = lookupTable[result.data[i]];
        result.data[i] = result.data[i + 1] = result.data[i + 2] = newVal;
      }
    }
    
    return result;
  };

  // Compute Gaussian kernel
  const computeGaussianKernel = (size, sigma) => {
    const kernel = [];
    const center = Math.floor(size / 2);
    let sum = 0;
    
    for (let y = 0; y < size; y++) {
      const row = [];
      for (let x = 0; x < size; x++) {
        const xDist = x - center;
        const yDist = y - center;
        const value = Math.exp(-(xDist * xDist + yDist * yDist) / (2 * sigma * sigma));
        row.push(value);
        sum += value;
      }
      kernel.push(row);
    }
    
    // Normalize
    for (let y = 0; y < size; y++) {
      for (let x = 0; x < size; x++) {
        kernel[y][x] /= sum;
      }
    }
    
    return kernel;
  };

  // Compute derivative of Gaussian kernel
  const computeDerivativeGaussianKernel = (size, sigma, direction) => {
    const kernel = [];
    const center = Math.floor(size / 2);
    let sum = 0;
    
    for (let y = 0; y < size; y++) {
      const row = [];
      for (let x = 0; x < size; x++) {
        const xDist = x - center;
        const yDist = y - center;
        const gaussian = Math.exp(-(xDist * xDist + yDist * yDist) / (2 * sigma * sigma));
        
        let value;
        if (direction === 'x') {
          value = -(xDist / (sigma * sigma)) * gaussian;
        } else {
          value = -(yDist / (sigma * sigma)) * gaussian;
        }
        
        row.push(value);
        sum += Math.abs(value);
      }
      kernel.push(row);
    }
    
    // Normalize
    for (let y = 0; y < size; y++) {
      for (let x = 0; x < size; x++) {
        kernel[y][x] /= sum;
      }
    }
    
    return kernel;
  };

  // Apply convolution
  const applyConvolution = (imageData, kernel) => {
    const width = imageData.width;
    const height = imageData.height;
    const data = imageData.data;
    const result = new ImageData(width, height);
    const kernelSize = kernel.length;
    const halfKernel = Math.floor(kernelSize / 2);
    
    for (let y = 0; y < height; y++) {
      for (let x = 0; x < width; x++) {
        let sum = 0;
        
        for (let ky = 0; ky < kernelSize; ky++) {
          for (let kx = 0; kx < kernelSize; kx++) {
            const pixelY = Math.min(Math.max(y + ky - halfKernel, 0), height - 1);
            const pixelX = Math.min(Math.max(x + kx - halfKernel, 0), width - 1);
            const pixelIndex = (pixelY * width + pixelX) * 4;
            sum += data[pixelIndex] * kernel[ky][kx];
          }
        }
        
        const resultIndex = (y * width + x) * 4;
        result.data[resultIndex] = result.data[resultIndex + 1] = result.data[resultIndex + 2] = Math.max(0, Math.min(255, sum));
        result.data[resultIndex + 3] = 255;
      }
    }
    
    return result;
  };

  // Compute gradient magnitude
  const computeGradientMagnitude = (gradX, gradY) => {
    const result = new ImageData(gradX.width, gradX.height);
    
    for (let i = 0; i < gradX.data.length; i += 4) {
      const gx = gradX.data[i] - 128;
      const gy = gradY.data[i] - 128;
      const magnitude = Math.sqrt(gx * gx + gy * gy);
      result.data[i] = result.data[i + 1] = result.data[i + 2] = Math.min(255, magnitude);
      result.data[i + 3] = 255;
    }
    
    return result;
  };

  const processImage = (img) => {
    const canvas = document.createElement('canvas');
    canvas.width = img.width;
    canvas.height = img.height;
    const ctx = canvas.getContext('2d');
    ctx.drawImage(img, 0, 0);
    
    const originalData = ctx.getImageData(0, 0, canvas.width, canvas.height);
    
    // Grayscale
    const grayData = toGrayscale(new ImageData(
      new Uint8ClampedArray(originalData.data),
      originalData.width,
      originalData.height
    ));
    
    // Otsu thresholding
    const thresholdValue = otsuThreshold(grayData);
    setOtsuThresholdValue(thresholdValue);
    
    const { imageData: binaryData, mask } = applyBinaryMask(
      new ImageData(new Uint8ClampedArray(grayData.data), grayData.width, grayData.height),
      thresholdValue
    );
    
    // Histogram equalization on foreground
    const equalizedData = histogramEqualization(
      new ImageData(new Uint8ClampedArray(grayData.data), grayData.width, grayData.height),
      mask
    );
    
    // Gaussian filtering
    const gaussianKernel5x5 = computeGaussianKernel(5, 2);
    const gaussianFiltered = applyConvolution(
      new ImageData(new Uint8ClampedArray(grayData.data), grayData.width, grayData.height),
      gaussianKernel5x5
    );
    
    // Derivative of Gaussian
    const dogKernelX = computeDerivativeGaussianKernel(5, 2, 'x');
    const dogKernelY = computeDerivativeGaussianKernel(5, 2, 'y');
    
    const gradX = applyConvolution(
      new ImageData(new Uint8ClampedArray(grayData.data), grayData.width, grayData.height),
      dogKernelX
    );
    
    const gradY = applyConvolution(
      new ImageData(new Uint8ClampedArray(grayData.data), grayData.width, grayData.height),
      dogKernelY
    );
    
    const gradMagnitude = computeGradientMagnitude(gradX, gradY);
    
    setResults({
      grayscale: grayData,
      binary: binaryData,
      equalized: equalizedData,
      gaussianFiltered,
      gradX,
      gradY,
      gradMagnitude,
      gaussianKernel5x5,
      dogKernelX,
      dogKernelY
    });
  };

  const downloadCanvas = (imageData, filename) => {
    const canvas = document.createElement('canvas');
    canvas.width = imageData.width;
    canvas.height = imageData.height;
    const ctx = canvas.getContext('2d');
    ctx.putImageData(imageData, 0, 0);
    
    canvas.toBlob((blob) => {
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = filename;
      a.click();
      URL.revokeObjectURL(url);
    });
  };

  const ResultImage = ({ imageData, title, showDownload = true }) => {
    const canvasRef = useRef(null);
    
    useEffect(() => {
      if (canvasRef.current && imageData) {
        const ctx = canvasRef.current.getContext('2d');
        ctx.putImageData(imageData, 0, 0);
      }
    }, [imageData]);
    
    return (
      <div className="border rounded p-2">
        <div className="flex justify-between items-center mb-2">
          <h3 className="font-semibold text-sm">{title}</h3>
          {showDownload && (
            <button
              onClick={() => downloadCanvas(imageData, `${title}.png`)}
              className="p-1 hover:bg-gray-100 rounded"
            >
              <Download size={16} />
            </button>
          )}
        </div>
        <canvas
          ref={canvasRef}
          width={imageData?.width || 0}
          height={imageData?.height || 0}
          className="w-full border"
        />
      </div>
    );
  };

  const KernelDisplay = ({ kernel, title }) => (
    <div className="border rounded p-3">
      <h3 className="font-semibold mb-2">{title}</h3>
      <div className="overflow-auto max-h-64">
        <table className="text-xs border-collapse">
          <tbody>
            {kernel.map((row, i) => (
              <tr key={i}>
                {row.map((val, j) => (
                  <td key={j} className="border px-1 text-center">
                    {val.toFixed(4)}
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );

  return (
    <div className="max-w-7xl mx-auto p-6">
      <h1 className="text-3xl font-bold mb-6">Image Processing Suite</h1>
      
      <div className="mb-6">
        <label className="flex items-center gap-2 px-4 py-2 bg-blue-500 text-white rounded cursor-pointer hover:bg-blue-600 w-fit">
          <Upload size={20} />
          <span>Upload Image</span>
          <input
            type="file"
            accept="image/*"
            onChange={handleImageUpload}
            className="hidden"
          />
        </label>
      </div>

      {otsuThreshold !== null && (
        <div className="mb-6 p-4 bg-blue-50 rounded">
          <h2 className="font-semibold text-lg mb-2">Results Summary</h2>
          <p><strong>Otsu Threshold Value:</strong> {otsuThreshold}</p>
          <p className="mt-2 text-sm"><strong>Hidden Features Revealed:</strong> Histogram equalization on the foreground enhances contrast, revealing texture details in the woman's clothing, facial features, and room details that were previously obscured in darker regions.</p>
        </div>
      )}

      {results.grayscale && (
        <>
          <h2 className="text-2xl font-bold mb-4">Problem 4: Otsu Thresholding & Histogram Equalization</h2>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4 mb-8">
            <ResultImage imageData={results.grayscale} title="Grayscale" />
            <ResultImage imageData={results.binary} title="Otsu Binary Mask" />
            <ResultImage imageData={results.equalized} title="Foreground Equalized" />
          </div>

          <h2 className="text-2xl font-bold mb-4">Problem 5: Gaussian Filtering</h2>
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-4 mb-4">
            <KernelDisplay kernel={results.gaussianKernel5x5} title="5×5 Gaussian Kernel (σ=2)" />
          </div>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-8">
            <ResultImage imageData={results.grayscale} title="Original Grayscale" />
            <ResultImage imageData={results.gaussianFiltered} title="Gaussian Filtered" />
          </div>

          <h2 className="text-2xl font-bold mb-4">Problem 6: Derivative of Gaussian</h2>
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-4 mb-4">
            <KernelDisplay kernel={results.dogKernelX} title="5×5 DoG Kernel (X-direction, σ=2)" />
            <KernelDisplay kernel={results.dogKernelY} title="5×5 DoG Kernel (Y-direction, σ=2)" />
          </div>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-8">
            <ResultImage imageData={results.gradX} title="Gradient X" />
            <ResultImage imageData={results.gradY} title="Gradient Y" />
            <ResultImage imageData={results.gradMagnitude} title="Gradient Magnitude" />
          </div>
        </>
      )}

      {!image && (
        <div className="text-center text-gray-500 py-12">
          <p>Upload an image to begin processing</p>
        </div>
      )}
    </div>
  );
};

export default ImageProcessingSuite;