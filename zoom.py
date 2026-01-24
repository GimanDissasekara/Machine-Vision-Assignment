import React, { useState, useRef, useEffect } from 'react';
import { Upload, Download } from 'lucide-react';

const ImageZooming = () => {
  const [smallImage, setSmallImage] = useState(null);
  const [largeImage, setLargeImage] = useState(null);
  const [zoomFactor, setZoomFactor] = useState(2.0);
  const [results, setResults] = useState({});

  const handleSmallImageUpload = (e) => {
    const file = e.target.files[0];
    if (file) {
      const reader = new FileReader();
      reader.onload = (event) => {
        const img = new Image();
        img.onload = () => {
          setSmallImage(img);
          if (largeImage) processImages(img, largeImage, zoomFactor);
        };
        img.src = event.target.result;
      };
      reader.readAsDataURL(file);
    }
  };

  const handleLargeImageUpload = (e) => {
    const file = e.target.files[0];
    if (file) {
      const reader = new FileReader();
      reader.onload = (event) => {
        const img = new Image();
        img.onload = () => {
          setLargeImage(img);
          if (smallImage) processImages(smallImage, img, zoomFactor);
        };
        img.src = event.target.result;
      };
      reader.readAsDataURL(file);
    }
  };

  const handleZoomFactorChange = (value) => {
    const factor = parseFloat(value);
    setZoomFactor(factor);
    if (smallImage && largeImage) {
      processImages(smallImage, largeImage, factor);
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

  const zoomNearestNeighbor = (imageData, scale) => {
    const srcWidth = imageData.width;
    const srcHeight = imageData.height;
    const dstWidth = Math.round(srcWidth * scale);
    const dstHeight = Math.round(srcHeight * scale);
    
    const result = new ImageData(dstWidth, dstHeight);
    
    for (let y = 0; y < dstHeight; y++) {
      for (let x = 0; x < dstWidth; x++) {
        const srcX = Math.floor(x / scale);
        const srcY = Math.floor(y / scale);
        
        const srcIdx = (srcY * srcWidth + srcX) * 4;
        const dstIdx = (y * dstWidth + x) * 4;
        
        result.data[dstIdx] = imageData.data[srcIdx];
        result.data[dstIdx + 1] = imageData.data[srcIdx + 1];
        result.data[dstIdx + 2] = imageData.data[srcIdx + 2];
        result.data[dstIdx + 3] = imageData.data[srcIdx + 3];
      }
    }
    
    return result;
  };

  const zoomBilinear = (imageData, scale) => {
    const srcWidth = imageData.width;
    const srcHeight = imageData.height;
    const dstWidth = Math.round(srcWidth * scale);
    const dstHeight = Math.round(srcHeight * scale);
    
    const result = new ImageData(dstWidth, dstHeight);
    
    for (let y = 0; y < dstHeight; y++) {
      for (let x = 0; x < dstWidth; x++) {
        const srcX = x / scale;
        const srcY = y / scale;
        
        const x1 = Math.floor(srcX);
        const y1 = Math.floor(srcY);
        const x2 = Math.min(x1 + 1, srcWidth - 1);
        const y2 = Math.min(y1 + 1, srcHeight - 1);
        
        const fx = srcX - x1;
        const fy = srcY - y1;
        
        const idx11 = (y1 * srcWidth + x1) * 4;
        const idx21 = (y1 * srcWidth + x2) * 4;
        const idx12 = (y2 * srcWidth + x1) * 4;
        const idx22 = (y2 * srcWidth + x2) * 4;
        
        const dstIdx = (y * dstWidth + x) * 4;
        
        for (let c = 0; c < 4; c++) {
          const top = imageData.data[idx11 + c] * (1 - fx) + imageData.data[idx21 + c] * fx;
          const bottom = imageData.data[idx12 + c] * (1 - fx) + imageData.data[idx22 + c] * fx;
          result.data[dstIdx + c] = top * (1 - fy) + bottom * fy;
        }
      }
    }
    
    return result;
  };

  const computeSSD = (img1, img2) => {
    if (img1.width !== img2.width || img1.height !== img2.height) {
      return null;
    }
    
    let ssd = 0;
    const n = img1.data.length / 4;
    
    for (let i = 0; i < img1.data.length; i += 4) {
      for (let c = 0; c < 3; c++) {
        const diff = img1.data[i + c] - img2.data[i + c];
        ssd += diff * diff;
      }
    }
    
    return ssd / (n * 3);
  };

  const processImages = (small, large, factor) => {
    const smallData = getImageData(small);
    
    const zoomedNN = zoomNearestNeighbor(smallData, factor);
    const zoomedBilinear = zoomBilinear(smallData, factor);
    
    let ssdNN = null;
    let ssdBilinear = null;
    
    if (large) {
      const largeData = getImageData(large);
      
      if (zoomedNN.width === largeData.width && zoomedNN.height === largeData.height) {
        ssdNN = computeSSD(zoomedNN, largeData);
        ssdBilinear = computeSSD(zoomedBilinear, largeData);
      }
    }
    
    setResults({
      original: smallData,
      zoomedNN,
      zoomedBilinear,
      largeOriginal: large ? getImageData(large) : null,
      ssdNN,
      ssdBilinear
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
          <h3 className="font-semibold text-sm">{title}</h3>
          <button
            onClick={() => downloadCanvas(imageData, `${title}.png`)}
            className="p-1 hover:bg-gray-100 rounded"
            title="Download"
          >
            <Download size={16} />
          </button>
        </div>
        <div className="text-xs text-gray-600 mb-1">
          Size: {imageData.width} × {imageData.height}
        </div>
        <canvas
          ref={canvasRef}
          width={imageData?.width || 0}
          height={imageData?.height || 0}
          className="w-full border"
          style={{ maxHeight: '400px', objectFit: 'contain' }}
        />
      </div>
    );
  };

  return (
    <div className="max-w-7xl mx-auto p-6">
      <h1 className="text-3xl font-bold mb-2">Question 7: Image Zooming with Interpolation</h1>
      <p className="text-gray-600 mb-6">
        Zoom images using nearest-neighbor and bilinear interpolation. Compare results using normalized SSD.
      </p>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-6">
        <div>
          <label className="block mb-2 font-semibold">Small Image (to zoom)</label>
          <label className="flex items-center gap-2 px-4 py-2 bg-blue-500 text-white rounded cursor-pointer hover:bg-blue-600 w-fit">
            <Upload size={20} />
            <span>Upload Small Image</span>
            <input
              type="file"
              accept="image/*"
              onChange={handleSmallImageUpload}
              className="hidden"
            />
          </label>
        </div>

        <div>
          <label className="block mb-2 font-semibold">Large Original (optional, for SSD)</label>
          <label className="flex items-center gap-2 px-4 py-2 bg-green-500 text-white rounded cursor-pointer hover:bg-green-600 w-fit">
            <Upload size={20} />
            <span>Upload Large Image</span>
            <input
              type="file"
              accept="image/*"
              onChange={handleLargeImageUpload}
              className="hidden"
            />
          </label>
        </div>
      </div>

      <div className="mb-6 p-4 bg-gray-50 rounded">
        <label className="block mb-2 font-semibold">
          Zoom Factor: {zoomFactor.toFixed(2)}x
        </label>
        <input
          type="range"
          min="0.1"
          max="10"
          step="0.1"
          value={zoomFactor}
          onChange={(e) => handleZoomFactorChange(e.target.value)}
          className="w-full"
        />
        <div className="flex justify-between text-xs text-gray-600 mt-1">
          <span>0.1x</span>
          <span>5.0x</span>
          <span>10.0x</span>
        </div>
      </div>

      {(results.ssdNN !== null || results.ssdBilinear !== null) && (
        <div className="mb-6 p-4 bg-blue-50 border-l-4 border-blue-500 rounded">
          <h2 className="font-semibold text-lg mb-2">Normalized SSD Results</h2>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div>
              <p className="font-semibold">Nearest-Neighbor:</p>
              <p className="text-2xl font-bold text-blue-600">
                {results.ssdNN?.toFixed(4) || 'N/A'}
              </p>
            </div>
            <div>
              <p className="font-semibold">Bilinear Interpolation:</p>
              <p className="text-2xl font-bold text-green-600">
                {results.ssdBilinear?.toFixed(4) || 'N/A'}
              </p>
            </div>
          </div>
          <p className="text-sm text-gray-600 mt-2">
            Lower SSD indicates better match with original. Bilinear typically produces smoother results with lower SSD.
          </p>
        </div>
      )}

      {results.original && (
        <div className="space-y-6">
          <div>
            <h2 className="text-xl font-bold mb-3">(a) Nearest-Neighbor Interpolation</h2>
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
              <ResultImage imageData={results.original} title="Original Small Image" />
              <ResultImage imageData={results.zoomedNN} title="Zoomed (Nearest-Neighbor)" />
            </div>
          </div>

          <div>
            <h2 className="text-xl font-bold mb-3">(b) Bilinear Interpolation</h2>
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
              <ResultImage imageData={results.original} title="Original Small Image" />
              <ResultImage imageData={results.zoomedBilinear} title="Zoomed (Bilinear)" />
            </div>
          </div>

          {results.largeOriginal && (
            <div>
              <h2 className="text-xl font-bold mb-3">Comparison with Large Original</h2>
              <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
                <ResultImage imageData={results.largeOriginal} title="Large Original" />
                <ResultImage imageData={results.zoomedNN} title="Nearest-Neighbor" />
                <ResultImage imageData={results.zoomedBilinear} title="Bilinear" />
              </div>
            </div>
          )}

          <div className="p-4 bg-yellow-50 border-l-4 border-yellow-500 rounded">
            <h3 className="font-semibold mb-2">Algorithm Comparison</h3>
            <div className="text-sm space-y-2">
              <p><strong>Nearest-Neighbor:</strong> Fast, preserves sharp edges, but produces blocky artifacts. Each output pixel takes the value of the nearest input pixel.</p>
              <p><strong>Bilinear Interpolation:</strong> Smoother results, weighted average of 4 nearest pixels. Better visual quality but slightly blurred edges. Generally lower SSD when compared to original high-resolution images.</p>
            </div>
          </div>
        </div>
      )}

      {!smallImage && (
        <div className="text-center text-gray-500 py-12 border-2 border-dashed rounded">
          <Upload size={48} className="mx-auto mb-4 opacity-50" />
          <p>Upload a small image to begin zooming</p>
          <p className="text-sm mt-2">Optionally upload a large original image to compute SSD metrics</p>
        </div>
      )}
    </div>
  );
};

export default ImageZooming;