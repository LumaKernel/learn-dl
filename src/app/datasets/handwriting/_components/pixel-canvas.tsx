"use client";

import { useCallback, useRef, useState } from "react";
import type { GridSize, PixelValue } from "@/domain/pixel-grid";

type Coord = { readonly x: number; readonly y: number };

const bresenhamLine = (from: Coord, to: Coord): ReadonlyArray<Coord> => {
  const points: Coord[] = [];
  let x0 = from.x;
  let y0 = from.y;
  const x1 = to.x;
  const y1 = to.y;
  const dx = Math.abs(x1 - x0);
  const dy = Math.abs(y1 - y0);
  const sx = x0 < x1 ? 1 : -1;
  const sy = y0 < y1 ? 1 : -1;
  let err = dx - dy;

  for (;;) {
    points.push({ x: x0, y: y0 });
    if (x0 === x1 && y0 === y1) break;
    const e2 = 2 * err;
    if (e2 > -dy) {
      err -= dy;
      x0 += sx;
    }
    if (e2 < dx) {
      err += dx;
      y0 += sy;
    }
  }
  return points;
};

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
  const lastCoordRef = useRef<Coord | null>(null);
  const pixelSize = canvasSize / size;

  const getGridCoords = useCallback(
    (e: React.MouseEvent<SVGSVGElement> | React.TouchEvent<SVGSVGElement>): Coord | null => {
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

  const paintPoints = useCallback(
    (points: ReadonlyArray<Coord>) => {
      const next = [...pixels];
      let changed = false;
      for (const p of points) {
        const idx = p.y * size + p.x;
        if (next[idx] !== drawColor) {
          next[idx] = drawColor;
          changed = true;
        }
      }
      if (changed) onPixelsChange(next);
    },
    [size, pixels, drawColor, onPixelsChange],
  );

  const handlePointerDown = useCallback(
    (e: React.MouseEvent<SVGSVGElement>) => {
      const coord = getGridCoords(e);
      if (!coord) return;
      setIsDrawing(true);
      lastCoordRef.current = coord;
      paintPoints([coord]);
    },
    [getGridCoords, paintPoints],
  );

  const handlePointerMove = useCallback(
    (e: React.MouseEvent<SVGSVGElement>) => {
      if (!isDrawing) return;
      const coord = getGridCoords(e);
      if (!coord) return;
      const last = lastCoordRef.current;
      const points = last ? bresenhamLine(last, coord) : [coord];
      lastCoordRef.current = coord;
      paintPoints(points);
    },
    [isDrawing, getGridCoords, paintPoints],
  );

  const handlePointerUp = useCallback(() => {
    setIsDrawing(false);
    lastCoordRef.current = null;
  }, []);

  const handleTouchStart = useCallback(
    (e: React.TouchEvent<SVGSVGElement>) => {
      e.preventDefault();
      const coord = getGridCoords(e);
      if (!coord) return;
      setIsDrawing(true);
      lastCoordRef.current = coord;
      paintPoints([coord]);
    },
    [getGridCoords, paintPoints],
  );

  const handleTouchMove = useCallback(
    (e: React.TouchEvent<SVGSVGElement>) => {
      e.preventDefault();
      if (!isDrawing) return;
      const coord = getGridCoords(e);
      if (!coord) return;
      const last = lastCoordRef.current;
      const points = last ? bresenhamLine(last, coord) : [coord];
      lastCoordRef.current = coord;
      paintPoints(points);
    },
    [isDrawing, getGridCoords, paintPoints],
  );

  const handleTouchEnd = useCallback(
    (e: React.TouchEvent<SVGSVGElement>) => {
      e.preventDefault();
      setIsDrawing(false);
      lastCoordRef.current = null;
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
