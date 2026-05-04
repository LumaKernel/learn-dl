"use client";

import { useCallback, useRef, useState } from "react";
import type { GridSize, PixelValue } from "@/domain/pixel-grid";

type PixelCanvasProps = {
  readonly size: GridSize;
  readonly pixels: ReadonlyArray<PixelValue>;
  readonly onPixelsChange: (pixels: ReadonlyArray<PixelValue>) => void;
  readonly canvasSize?: number;
};

export function PixelCanvas({
  size,
  pixels,
  onPixelsChange,
  canvasSize = 400,
}: PixelCanvasProps) {
  const [isDrawing, setIsDrawing] = useState(false);
  const [drawColor, setDrawColor] = useState<PixelValue>(1);
  const svgRef = useRef<SVGSVGElement>(null);
  const pixelSize = canvasSize / size;

  const getGridCoords = useCallback(
    (e: React.MouseEvent<SVGSVGElement> | React.TouchEvent<SVGSVGElement>) => {
      const svg = svgRef.current;
      if (!svg) return null;
      const rect = svg.getBoundingClientRect();
      const clientX = "touches" in e ? (e.touches[0]?.clientX ?? 0) : e.clientX;
      const clientY = "touches" in e ? (e.touches[0]?.clientY ?? 0) : e.clientY;
      const x = Math.floor(((clientX - rect.left) / rect.width) * size);
      const y = Math.floor(((clientY - rect.top) / rect.height) * size);
      if (x < 0 || x >= size || y < 0 || y >= size) return null;
      return { x, y };
    },
    [size],
  );

  const paint = useCallback(
    (
      e: React.MouseEvent<SVGSVGElement> | React.TouchEvent<SVGSVGElement>,
    ) => {
      const coords = getGridCoords(e);
      if (!coords) return;
      const idx = coords.y * size + coords.x;
      if (pixels[idx] === drawColor) return;
      const next = [...pixels];
      next[idx] = drawColor;
      onPixelsChange(next);
    },
    [getGridCoords, size, pixels, drawColor, onPixelsChange],
  );

  const handlePointerDown = useCallback(
    (e: React.MouseEvent<SVGSVGElement>) => {
      setIsDrawing(true);
      paint(e);
    },
    [paint],
  );

  const handlePointerMove = useCallback(
    (e: React.MouseEvent<SVGSVGElement>) => {
      if (!isDrawing) return;
      paint(e);
    },
    [isDrawing, paint],
  );

  const handlePointerUp = useCallback(() => {
    setIsDrawing(false);
  }, []);

  const handleTouchStart = useCallback(
    (e: React.TouchEvent<SVGSVGElement>) => {
      e.preventDefault();
      setIsDrawing(true);
      paint(e);
    },
    [paint],
  );

  const handleTouchMove = useCallback(
    (e: React.TouchEvent<SVGSVGElement>) => {
      e.preventDefault();
      if (!isDrawing) return;
      paint(e);
    },
    [isDrawing, paint],
  );

  const handleTouchEnd = useCallback(
    (e: React.TouchEvent<SVGSVGElement>) => {
      e.preventDefault();
      setIsDrawing(false);
    },
    [],
  );

  return (
    <div className="flex flex-col gap-3">
      <div className="flex gap-2 items-center">
        <span className="text-sm text-zinc-500">Draw color:</span>
        <button
          type="button"
          className={`w-8 h-8 border-2 rounded ${(drawColor === 1 ? "border-blue-500" : "border-zinc-300") satisfies string}`}
          style={{ backgroundColor: "#000" }}
          onClick={() => setDrawColor(1)}
          aria-label="Black"
        />
        <button
          type="button"
          className={`w-8 h-8 border-2 rounded ${(drawColor === 0 ? "border-blue-500" : "border-zinc-300") satisfies string}`}
          style={{ backgroundColor: "#fff" }}
          onClick={() => setDrawColor(0)}
          aria-label="White (eraser)"
        />
        <button
          type="button"
          className="ml-4 px-3 py-1 text-sm bg-zinc-200 dark:bg-zinc-700 rounded hover:bg-zinc-300 dark:hover:bg-zinc-600"
          onClick={() => onPixelsChange(Array.from({ length: size * size }, () => 0 as const))}
        >
          Clear
        </button>
      </div>
      <svg
        ref={svgRef}
        width={canvasSize}
        height={canvasSize}
        viewBox={`0 0 ${String(canvasSize) satisfies string} ${String(canvasSize) satisfies string}`}
        className="border border-zinc-300 dark:border-zinc-600 cursor-crosshair select-none touch-none"
        onMouseDown={handlePointerDown}
        onMouseMove={handlePointerMove}
        onMouseUp={handlePointerUp}
        onMouseLeave={handlePointerUp}
        onTouchStart={handleTouchStart}
        onTouchMove={handleTouchMove}
        onTouchEnd={handleTouchEnd}
      >
        <rect width={canvasSize} height={canvasSize} fill="#fff" />
        {pixels.map((v, i) => {
          if (v === 0) return null;
          const x = (i % size) * pixelSize;
          const y = Math.floor(i / size) * pixelSize;
          return (
            <rect
              key={i}
              x={x}
              y={y}
              width={pixelSize}
              height={pixelSize}
              fill="#000"
            />
          );
        })}
        {/* grid lines */}
        {size <= 28 &&
          Array.from({ length: size + 1 }, (_, i) => (
            <g key={`grid-${String(i) satisfies string}`}>
              <line
                x1={i * pixelSize}
                y1={0}
                x2={i * pixelSize}
                y2={canvasSize}
                stroke="#ddd"
                strokeWidth={0.5}
              />
              <line
                x1={0}
                y1={i * pixelSize}
                x2={canvasSize}
                y2={i * pixelSize}
                stroke="#ddd"
                strokeWidth={0.5}
              />
            </g>
          ))}
      </svg>
    </div>
  );
}
