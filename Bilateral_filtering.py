import React, { useState, useRef, useEffect } from 'react';
import { Upload, Download } from 'lucide-react';

const BilateralFiltering = () => {
  const [image, setImage] = useState(null);
  const [results, setResults] = useState({});
  const [diameter, setDiameter] = useState(9);
  const [sigmaSpace, setSigmaSpace] = useState(75);
  const [sigmaColor, setSigmaColor] = useState(75);
  const [processing, setProcessing] = useState(false);

  const handleImageUpload = (e) => {
    const file = e.target.files[0];
    if (file) {
      const reader = new FileReader();
      reader.onload = (event) => {
        const img = new Image();
        img.onload = () => {
          setImage(img);
          processImage(img, diameter, sigmaSpace, sigmaColor);
        };
        img.src = event.target.result;
      };
      reader.readAsDataURL(file);
    }
  };

  const handleParameterChange = (d, sigS, sigC) => {
    if (image) {
      processImage(image, d, sigS, sigC);
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

  const toGrayscale = (imageData) => {
    const result = new ImageData(imageData.width, imageData.height);
    for (let i = 0; i < imageData.data.length; i += 4) {
      const gray = 0.299 * imageData.data[i] + 0.587 * imageData.data[i + 1] + 0.114 * imageData.data[i + 2];
      result.data[i] = result.data[i + 1] = result.data[i + 2] = gray;
      result.data[i + 3] = 255;
    }
    return result;
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

  const bilateralFilterManual = (imageData, d, sigmaS, sigmaR) => {
    const width = imageData.width;
    const height = imageData.height;
    const data = imageData.data;
    const result = new ImageData(width, height);
    const radius = Math.floor(d / 2);
    
    const spatialCoeff = -0.5 / (sigmaS * sigmaS);
    const rangeCoeff = -0.5 / (sigmaR * sigmaR);
    
    for (let y = 0; y < height; y++) {
      for (let x = 0; x < width; x++) {
        const centerIdx = (y * width + x) * 4;
        const centerR = data[centerIdx];
        const centerG = data[centerIdx + 1];
        const centerB = data[centerIdx + 2];
        
        let sumR = 0, sumG = 0, sumB = 0;
        let sumWeight = 0;
        
        for (let ky = -radius; ky <= radius; ky++) {
          for (let kx = -radius; kx <= radius; kx++) {
            const pixelY = Math.min(Math.max(y + ky, 0), height - 1);
            const pixelX = Math.min(Math.max(x + kx, 0), width - 1);
            const pixelIdx = (pixelY * width + pixelX) * 4;
            
            const spatialDist = kx * kx + ky * ky;
            const spatialWeight = Math.exp(spatialDist * spatialCoeff);
            
            const colorDistR = data[pixelIdx] - centerR;
            const colorDistG = data[pixelIdx + 1] - centerG;
            const colorDistB = data[pixelIdx + 2] - centerB;
            const colorDist = colorDistR * colorDistR + colorDistG * colorDistG + colorDistB * colorDistB;
            const rangeWeight = Math.exp(colorDist * rangeCoeff);
            
            const weight = spatialWeight * rangeWeight;
            
            sumR += data[pixelIdx] * weight;
            sumG += data[pixelIdx + 1] * weight;
            sumB += data[pixelIdx + 2] * weight;
            sumWeight += weight;
          }
        }
        
        result.data[centerIdx] = sumR / sumWeight;
        result.data[centerIdx + 1] = sumG / sumWeight;
        result.data[centerIdx + 2] = sumB / sumWeight;
        result.data[centerIdx + 3] = 255;
      }
    }
    
    return result;
  };

  const processImage = async (img, d, sigS, sigC) => {
    setProcessing(true);
    
    setTimeout(() => {
      const imageData = getImageData(img);
      const grayData = toGrayscale(imageData);
      
      const gaussianFiltered = applyGaussianFilter(
        new ImageData(new Uint8ClampedArray(grayData.data), grayData.width, grayData.height),
        9,
        2.0
      );
      
      const bilateralFiltered = bilateralFilterManual(
        new ImageData(new Uint8ClampedArray(grayData.data), grayData.width, grayData.height),
        d,
        sigS,
        sigC
      );
      
      setResults({
        original: imageData,
        grayscale: grayData,
        gaussian: gaussianFiltered,
        bilateral: bilateralFiltered
      });
      
      setProcessing(false);
    }, 100);
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
          <h3 className="font-semibold text-sm">{title}</h3>
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
      <h1 className="text-3xl font-bold mb-2">Question 10: Bilateral Filtering</h1>
      <p className="text-gray-600 mb-6">
        Edge-preserving smoothing using bilateral filter with spatial and range components.
      </p>

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

      <div className="mb-6 p-4 bg-gray-50 rounded space-y-4">
        <div>
          <label className="block mb-2 font-semibold">
            Kernel Diameter: {diameter}
          </label>
          <input
            type="range"
            min="3"
            max="15"
            step="2"
            value={diameter}
            onChange={(e) => {
              const val = parseInt(e.target.value);
              setDiameter(val);
              handleParameterChange(val, sigmaSpace, sigmaColor);
            }}
            className="w-full"
            disabled={processing}
          />
          <div className="flex justify-between text-xs text-gray-600 mt-1">
            <span>3</span>
            <span>9</span>
            <span>15</span>
          </div>
        </div>

        <div>
          <label className="block mb-2 font-semibold">
            Spatial Sigma (σₛ): {sigmaSpace}
          </label>
          <input
            type="range"
            min="10"
            max="200"
            step="5"
            value={sigmaSpace}
            onChange={(e) => {
              const val = parseInt(e.target.value);
              setSigmaSpace(val);
              handleParameterChange(diameter, val, sigmaColor);
            }}
            className="w-full"
            disabled={processing}
          />
          <div className="flex justify-between text-xs text-gray-600 mt-1">
            <span>10</span>
            <span>100</span>
            <span>200</span>
          </div>
          <p className="text-xs text-gray-600 mt-1">Controls spatial smoothing (distance-based)</p>
        </div>

        <div>
          <label className="block mb-2 font-semibold">
            Range/Intensity Sigma (σᵣ): {sigmaColor}
          </label>
          <input
            type="range"
            min="10"
            max="200"
            step="5"
            value={sigmaColor}
            onChange={(e) => {
              const val = parseInt(e.target.value);
              setSigmaColor(val);
              handleParameterChange(diameter, sigmaSpace, val);
            }}
            className="w-full"
            disabled={processing}
          />
          <div className="flex justify-between text-xs text-gray-600 mt-1">
            <span>10</span>
            <span>100</span>
            <span>200</span>
          </div>
          <p className="text-xs text-gray-600 mt-1">Controls edge preservation (intensity-based)</p>
        </div>

        {processing && (
          <div className="text-center text-blue-600 font-semibold">
            Processing... This may take a moment for large images.
          </div>
        )}
      </div>

      {results.original && (
        <>
          <div className="mb-6 p-4 bg-blue-50 border-l-4 border-blue-500 rounded">
            <h2 className="font-semibold text-lg mb-2">How Bilateral Filtering Works</h2>
            <div className="text-sm space-y-2">
              <p><strong>Formula:</strong> BF[I]ₚ = (1/Wₚ) Σ Gₛ(‖p-q‖) · Gᵣ(|Iₚ-Iᵧ|) · Iᵧ</p>
              <p><strong>Two Components:</strong></p>
              <ul className="list-disc ml-6">
                <li><strong>Spatial Weight Gₛ:</strong> Gaussian based on spatial distance (like regular Gaussian blur)</li>
                <li><strong>Range Weight Gᵣ:</strong> Gaussian based on intensity difference (preserves edges)</li>
              </ul>
              <p><strong>Key Insight:</strong> Pixels are averaged based on both proximity AND similarity in intensity. This smooths flat regions while preserving edges where intensity changes rapidly.</p>
            </div>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-6">
            <ResultImage imageData={results.original} title="Original Image" />
            <ResultImage imageData={results.grayscale} title="Grayscale" />
            <ResultImage imageData={results.gaussian} title="(b) Gaussian Smoothing" />
            <ResultImage imageData={results.bilateral} title="(a,d) Manual Bilateral Filter" />
          </div>

          <div className="p-4 bg-yellow-50 border-l-4 border-yellow-500 rounded">
            <h3 className="font-semibold mb-2">Comparison: Gaussian vs Bilateral</h3>
            <div className="text-sm space-y-2">
              <p><strong>Gaussian Filter:</strong></p>
              <ul className="list-disc ml-6">
                <li>Only considers spatial distance</li>
                <li>Blurs everything uniformly, including edges</li>
                <li>Faster to compute</li>
                <li>Good for uniform noise reduction</li>
              </ul>
              <p className="mt-2"><strong>Bilateral Filter:</strong></p>
              <ul className="list-disc ml-6">
                <li>Considers both spatial distance AND intensity similarity</li>
                <li>Preserves edges while smoothing flat regions</li>
                <li>Slower (non-linear filter)</li>
                <li>Excellent for noise reduction while maintaining detail</li>
                <li>Popular in computational photography and tone mapping</li>
              </ul>
              <p className="mt-2 font-semibold">Note: (c) OpenCV's cv.bilateralFilter() would produce similar results to our manual implementation with optimized performance.</p>
            </div>
          </div>
        </>
      )}

      {!image && (
        <div className="text-center text-gray-500 py-12 border-2 border-dashed rounded">
          <Upload size={48} className="mx-auto mb-4 opacity-50" />
          <p>Upload an image to apply bilateral filtering</p>
          <p className="text-sm mt-2">Best for edge-preserving noise reduction</p>
        </div>
      )}
    </div>
  );
};

export default BilateralFiltering;