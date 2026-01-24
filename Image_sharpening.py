import React, { useState, useRef, useEffect } from 'react';
import { Upload, Download } from 'lucide-react';

const ImageSharpening = () => {
  const [image, setImage] = useState(null);
  const [results, setResults] = useState({});
  const [sharpenStrength, setSharpenStrength] = useState(1.5);
  const [method, setMethod] = useState('unsharp');

  const handleImageUpload = (e) => {
    const file = e.target.files[0];
    if (file) {
      const reader = new FileReader();
      reader.onload = (event) => {
        const img = new Image();
        img.onload = () => {
          setImage(img);
          processImage(img, sharpenStrength, method);
        };
        img.src = event.target.result;
      };
      reader.readAsDataURL(file);
    }
  };

  const handleParameterChange = (strength, selectedMethod) => {
    if (image) {
      processImage(image, strength, selectedMethod);
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

  const applyConvolution = (imageData, kernel) => {
    const width = imageData.width;
    const height = imageData.height;
    const data = imageData.data;
    const result = new ImageData(width, height);
    const kernelSize = kernel.length;
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
        result.data[resultIndex] = Math.max(0, Math.min(255, r));
        result.data[resultIndex + 1] = Math.max(0, Math.min(255, g));
        result.data[resultIndex + 2] = Math.max(0, Math.min(255, b));
        result.data[resultIndex + 3] = 255;
      }
    }
    
    return result;
  };

  const unsharpMask = (imageData, strength) => {
    const gaussianKernel = computeGaussianKernel(5, 1.0);
    const blurred = applyConvolution(imageData, gaussianKernel);
    
    const result = new ImageData(imageData.width, imageData.height);
    
    for (let i = 0; i < imageData.data.length; i += 4) {
      for (let c = 0; c < 3; c++) {
        const original = imageData.data[i + c];
        const blur = blurred.data[i + c];
        const detail = original - blur;
        const sharpened = original + strength * detail;
        result.data[i + c] = Math.max(0, Math.min(255, sharpened));
      }
      result.data[i + 3] = 255;
    }
    
    return result;
  };

  const laplacianSharpen = (imageData, strength) => {
    const laplacianKernel = [
      [0, -1, 0],
      [-1, 4, -1],
      [0, -1, 0]
    ];
    
    const edges = applyConvolution(imageData, laplacianKernel);
    const result = new ImageData(imageData.width, imageData.height);
    
    for (let i = 0; i < imageData.data.length; i += 4) {
      for (let c = 0; c < 3; c++) {
        const original = imageData.data[i + c];
        const edge = edges.data[i + c] - 128;
        const sharpened = original + strength * edge;
        result.data[i + c] = Math.max(0, Math.min(255, sharpened));
      }
      result.data[i + 3] = 255;
    }
    
    return result;
  };

  const highBoostFilter = (imageData, strength) => {
    const gaussianKernel = computeGaussianKernel(5, 1.0);
    const blurred = applyConvolution(imageData, gaussianKernel);
    
    const A = 1 + strength;
    const result = new ImageData(imageData.width, imageData.height);
    
    for (let i = 0; i < imageData.data.length; i += 4) {
      for (let c = 0; c < 3; c++) {
        const original = imageData.data[i + c];
        const blur = blurred.data[i + c];
        const boosted = A * original - blur;
        result.data[i + c] = Math.max(0, Math.min(255, boosted));
      }
      result.data[i + 3] = 255;
    }
    
    return result;
  };

  const processImage = (img, strength, selectedMethod) => {
    const imageData = getImageData(img);
    
    let sharpened;
    switch (selectedMethod) {
      case 'unsharp':
        sharpened = unsharpMask(imageData, strength);
        break;
      case 'laplacian':
        sharpened = laplacianSharpen(imageData, strength);
        break;
      case 'highboost':
        sharpened = highBoostFilter(imageData, strength);
        break;
      default:
        sharpened = unsharpMask(imageData, strength);
    }
    
    setResults({
      original: imageData,
      sharpened: sharpened
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
      <h1 className="text-3xl font-bold mb-2">Question 9: Image Sharpening</h1>
      <p className="text-gray-600 mb-6">
        Enhance image details and edges using various sharpening techniques.
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
          <label className="block mb-2 font-semibold">Sharpening Method</label>
          <div className="space-y-2">
            <label className="flex items-center gap-2">
              <input
                type="radio"
                name="method"
                value="unsharp"
                checked={method === 'unsharp'}
                onChange={(e) => {
                  setMethod(e.target.value);
                  handleParameterChange(sharpenStrength, e.target.value);
                }}
              />
              <span>Unsharp Masking (Best overall)</span>
            </label>
            <label className="flex items-center gap-2">
              <input
                type="radio"
                name="method"
                value="laplacian"
                checked={method === 'laplacian'}
                onChange={(e) => {
                  setMethod(e.target.value);
                  handleParameterChange(sharpenStrength, e.target.value);
                }}
              />
              <span>Laplacian Sharpening</span>
            </label>
            <label className="flex items-center gap-2">
              <input
                type="radio"
                name="method"
                value="highboost"
                checked={method === 'highboost'}
                onChange={(e) => {
                  setMethod(e.target.value);
                  handleParameterChange(sharpenStrength, e.target.value);
                }}
              />
              <span>High-Boost Filtering</span>
            </label>
          </div>
        </div>

        <div>
          <label className="block mb-2 font-semibold">
            Sharpen Strength: {sharpenStrength.toFixed(1)}
          </label>
          <input
            type="range"
            min="0"
            max="3"
            step="0.1"
            value={sharpenStrength}
            onChange={(e) => {
              const val = parseFloat(e.target.value);
              setSharpenStrength(val);
              handleParameterChange(val, method);
            }}
            className="w-full"
          />
          <div className="flex justify-between text-xs text-gray-600 mt-1">
            <span>None (0)</span>
            <span>Moderate (1.5)</span>
            <span>Strong (3.0)</span>
          </div>
        </div>
      </div>

      {results.original && (
        <>
          <div className="mb-6 p-4 bg-blue-50 border-l-4 border-blue-500 rounded">
            <h2 className="font-semibold text-lg mb-2">How Image Sharpening Works</h2>
            <div className="text-sm space-y-2">
              <p><strong>Unsharp Masking:</strong> Subtracts a blurred version from the original, then adds the difference back with amplification. Formula: Sharpened = Original + α × (Original - Blurred)</p>
              <p><strong>Laplacian Sharpening:</strong> Uses second derivative (Laplacian operator) to detect edges, then adds them to the original. Enhances rapid intensity changes.</p>
              <p><strong>High-Boost Filtering:</strong> Weighted combination: A × Original - Blurred, where A ≥ 1. Similar to unsharp masking with different emphasis.</p>
            </div>
          </div>

          <div className="grid grid-cols-1 lg:grid-cols-2 gap-4 mb-6">
            <ResultImage imageData={results.original} title="Original Image" />
            <ResultImage imageData={results.sharpened} title={`Sharpened (${method})`} />
          </div>

          <div className="p-4 bg-yellow-50 border-l-4 border-yellow-500 rounded">
            <h3 className="font-semibold mb-2">Sharpening Tips</h3>
            <div className="text-sm space-y-2">
              <ul className="list-disc ml-6">
                <li>Sharpening enhances both details and noise - use on good quality images</li>
                <li>Over-sharpening creates halos and artifacts around edges</li>
                <li>Unsharp masking typically gives the most natural results</li>
                <li>Apply sharpening as the final step after other processing</li>
                <li>Different subjects need different strengths (portraits: 0.5-1.0, landscapes: 1.0-2.0)</li>
              </ul>
            </div>
          </div>
        </>
      )}

      {!image && (
        <div className="text-center text-gray-500 py-12 border-2 border-dashed rounded">
          <Upload size={48} className="mx-auto mb-4 opacity-50" />
          <p>Upload an image to apply sharpening</p>
          <p className="text-sm mt-2">Works best on slightly blurry or soft images</p>
        </div>
      )}
    </div>
  );
};

export default ImageSharpening;