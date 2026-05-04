"use client";

import type { GridSize, PixelValue } from "@/domain/pixel-grid";

type MiniPreviewProps = {
  readonly size: GridSize;
  readonly pixels: ReadonlyArray<PixelValue>;
  readonly previewSize?: number;
  readonly selected?: boolean;
  readonly onClick?: () => void;
};

export function MiniPreview({
  size,
  pixels,
  previewSize = 48,
  selected = false,
  onClick,
}: MiniPreviewProps) {
  const pixelSize = previewSize / size;

  return (
    <svg
      width={previewSize}
      height={previewSize}
      viewBox={`0 0 ${String(previewSize) satisfies string} ${String(previewSize) satisfies string}`}
      className={`border cursor-pointer shrink-0 ${(
        selected
          ? "border-blue-500 border-2"
          : "border-zinc-300 dark:border-zinc-600"
      ) satisfies string}`}
      onClick={onClick}
    >
      <rect width={previewSize} height={previewSize} fill="#fff" />
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
    </svg>
  );
}
