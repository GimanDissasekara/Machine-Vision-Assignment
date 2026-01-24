import React, { useState, useRef, useEffect } from 'react';
import { Upload, Download } from 'lucide-react';

const NoiseFiltering = () => {
  const [image, setImage] = useState(null);
  const [results, setResults] = useState({});
  const [kernelSize, setKernelSize] = useState(5);
  const [sigma, setSigma] = useState(1.5);

  const handleImageUpload = (e) => {
    const file = e.target.files[0];
    if (file) {
      const reader = new FileReader();
      reader.onload = (event) => {
        const img = new Image();
        img.onload = () => {
          setImage(img);
          processImage(img, kernelSize, sigma);
        };
        img.src = event.target.result;
      };
      reader.readAsDataURL(file);
    }
  };

  const handleParameterChange = (newKernelSize, newSigma) => {
    if (image) {
      processImage(image, newKernelSize, newSigma);
    }
  };

  const getImageData = (img) => {
    const canvas = document.createElement('canvas');
    canvas.width = img.width;
    canvas.height = img.height;
    const ctx = canvas.getContext('2d');
    ctx.drawImage(img, 0, 0);
    return ctx.getImageData(0, 0, canvas.width, canvas.height);
  };

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
    
    for (let y = 0; y < size; y++) {
      for (let x = 0; x < size; x++) {
        kernel[y][x] /= sum;
      }
    }
    
    return kernel;
  };

  const applyGaussianFilter = (imageData, kernelSize, sigma) => {
    const kernel = computeGaussianKernel(kernelSize, sigma);
    const width = imageData.width;
    const height = imageData.height;
    const data = imageData.data;
    const result = new ImageData(width, height);
    const halfKernel = Math.floor(kernelSize / 2);
    
    for (let y = 0; y < height; y++) {
      for (let x = 0; x < width; x++) {
        let r = 0, g = 0, b = 0;
        
        for (let ky = 0; ky < kernelSize; ky++) {
          for (let kx = 0; kx < kernelSize; kx++) {
            const pixelY = Math.min(Math.max(y + ky - halfKernel, 0), height - 1);
            const pixelX = Math.min(Math.max(x + kx - halfKernel, 0), width - 1);
            const pixelIndex = (pixelY * width + pixelX) * 4;
            const weight = kernel[ky][kx];
            
            r += data[pixelIndex] * weight;
            g += data[pixelIndex + 1] * weight;
            b += data[pixelIndex + 2] * weight;
          }
        }
        
        const resultIndex = (y * width + x) * 4;
        result.data[resultIndex] = r;
        result.data[resultIndex + 1] = g;
        result.data[resultIndex + 2] = b;
        result.data[resultIndex + 3] = 255;
      }
    }
    
    return result;
  };

  const applyMedianFilter = (imageData, kernelSize) => {
    const width = imageData.width;
    const height = imageData.height;
    const data = imageData.data;
    const result = new ImageData(width, height);
    const halfKernel = Math.floor(kernelSize / 2);
    
    for (let y = 0; y < height; y++) {
      for (let x = 0; x < width; x++) {
        const rValues = [];
        const gValues = [];
        const bValues = [];
        
        for (let ky = -halfKernel; ky <= halfKernel; ky++) {
          for (let kx = -halfKernel; kx <= halfKernel; kx++) {
            const pixelY = Math.min(Math.max(y + ky, 0), height - 1);
            const pixelX = Math.min(Math.max(x + kx, 0), width - 1);
            const pixelIndex = (pixelY * width + pixelX) * 4;
            
            rValues.push(data[pixelIndex]);
            gValues.push(data[pixelIndex + 1]);
            bValues.push(data[pixelIndex + 2]);
          }
        }
        
        rValues.sort((a, b) => a - b);
        gValues.sort((a, b) => a - b);
        bValues.sort((a, b) => a - b);
        
        const mid = Math.floor(rValues.length / 2);
        const resultIndex = (y * width + x) * 4;
        
        result.data[resultIndex] = rValues[mid];
        result.data[resultIndex + 1] = gValues[mid];
        result.data[resultIndex + 2] = bValues[mid];
        result.data[resultIndex + 3] = 255;
      }
    }
    
    return result;
  };

  const processImage = (img, kSize, sig) => {
    const imageData = getImageData(img);
    
    const gaussianFiltered = applyGaussianFilter(imageData, kSize, sig);
    const medianFiltered = applyMedianFilter(imageData, kSize);
    
    setResults({
      original: imageData,
      gaussian: gaussianFiltered,
      median: medianFiltered
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

  const ResultImage = ({ imageData, title }) => {
    const canvasRef = useRef(null);
    
    useEffect(() => {
      if (canvasRef.current && imageData) {
        const ctx = canvasRef.current.getContext('2d');
        ctx.putImageData(imageData, 0, 0);
      }
    }, [imageData]);
    
    return (
      <div className="border rounded p-3">
        <div className="flex justify-between items-center mb-2">
          <h3 className="font-semibold">{title}</h3>
          <button
            onClick={() => downloadCanvas(imageData, `${title}.png`)}
            className="p-1 hover:bg-gray-100 rounded"
            title="Download"
          >
            <Download size={16} />
          </button>
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

  return (
    <div className="max-w-7xl mx-auto p-6">
      <h1 className="text-3xl font-bold mb-2">Question 8: Salt & Pepper Noise Filtering</h1>
      <p className="text-gray-600 mb-6">
        Compare Gaussian smoothing and median filtering for removing salt and pepper noise.
      </p>

      <div className="mb-6">
        <label className="flex items-center gap-2 px-4 py-2 bg-blue-500 text-white rounded cursor-pointer hover:bg-blue-600 w-fit">
          <Upload size={20} />
          <span>Upload Noisy Image</span>
          <input
            type="file"
            accept="image/*"
            onChange={handleImageUpload}
            className="hidden"
          />
        </label>
      </div>

      <div className="mb-6 p-4 bg-gray-50 rounded space-y-4">
        <div>
          <label className="block mb-2 font-semibold">
            Kernel Size: {kernelSize}×{kernelSize}
          </label>
          <input
            type="range"
            min="3"
            max="11"
            step="2"
            value={kernelSize}
            onChange={(e) => {
              const val = parseInt(e.target.value);
              setKernelSize(val);
              handleParameterChange(val, sigma);
            }}
            className="w-full"
          />
          <div className="flex justify-between text-xs text-gray-600 mt-1">
            <span>3×3</span>
            <span>7×7</span>
            <span>11×11</span>
          </div>
        </div>

        <div>
          <label className="block mb-2 font-semibold">
            Gaussian Sigma (σ): {sigma.toFixed(1)}
          </label>
          <input
            type="range"
            min="0.5"
            max="5"
            step="0.1"
            value={sigma}
            onChange={(e) => {
              const val = parseFloat(e.target.value);
              setSigma(val);
              handleParameterChange(kernelSize, val);
            }}
            className="w-full"
          />
          <div className="flex justify-between text-xs text-gray-600 mt-1">
            <span>0.5</span>
            <span>2.5</span>
            <span>5.0</span>
          </div>
        </div>
      </div>

      {results.original && (
        <>
          <div className="mb-6 p-4 bg-blue-50 border-l-4 border-blue-500 rounded">
            <h2 className="font-semibold text-lg mb-2">Analysis</h2>
            <div className="text-sm space-y-2">
              <p><strong>(a) Gaussian Smoothing:</strong> Blurs the entire image uniformly, reducing noise but also softening edges. Salt and pepper noise is reduced but not completely eliminated as it averages noise pixels with neighbors.</p>
              <p><strong>(b) Median Filtering:</strong> Highly effective for salt and pepper noise. Replaces each pixel with the median value in its neighborhood, effectively removing isolated noise pixels while preserving edges better than Gaussian filtering.</p>
              <p className="font-semibold text-green-700">Winner for salt & pepper noise: Median Filter</p>
            </div>
          </div>

          <div className="grid grid-cols-1 lg:grid-cols-3 gap-4 mb-6">
            <ResultImage imageData={results.original} title="Original (Noisy)" />
            <ResultImage imageData={results.gaussian} title="(a) Gaussian Smoothed" />
            <ResultImage imageData={results.median} title="(b) Median Filtered" />
          </div>

          <div className="p-4 bg-yellow-50 border-l-4 border-yellow-500 rounded">
            <h3 className="font-semibold mb-2">Key Differences</h3>
            <div className="text-sm space-y-2">
              <p><strong>Gaussian Filter:</strong></p>
              <ul className="list-disc ml-6">
                <li>Linear filter (weighted average)</li>
                <li>Smooths all features uniformly</li>
                <li>Noise pixels affect neighboring pixels</li>
                <li>Less effective for impulse noise</li>
              </ul>
              <p className="mt-2"><strong>Median Filter:</strong></p>
              <ul className="list-disc ml-6">
                <li>Non-linear filter (order statistic)</li>
                <li>Preserves edges while removing noise</li>
                <li>Isolated noise pixels are replaced entirely</li>
                <li>Excellent for salt & pepper noise</li>
              </ul>
            </div>
          </div>
        </>
      )}

      {!image && (
        <div className="text-center text-gray-500 py-12 border-2 border-dashed rounded">
          <Upload size={48} className="mx-auto mb-4 opacity-50" />
          <p>Upload an image with salt and pepper noise to begin filtering</p>
        </div>
      )}
    </div>
  );
};

export default NoiseFiltering;