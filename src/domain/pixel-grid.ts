import { Data } from "effect";

export type PixelValue = 0 | 1;

export type GridSize = 28 | 64;

export const GridSize = {
  all: [28, 64] as const satisfies readonly GridSize[],
} as const;

export type PixelGrid = Data.TaggedEnum<{
  readonly PixelGrid: {
    readonly size: GridSize;
    readonly pixels: ReadonlyArray<PixelValue>;
  };
}>;

export const PixelGrid = Data.taggedEnum<PixelGrid>();

export const createEmpty = (size: GridSize): PixelGrid =>
  PixelGrid.PixelGrid({
    size,
    pixels: Array.from({ length: size * size }, () => 0 as const),
  });

export const getPixel = (
  grid: PixelGrid & { readonly _tag: "PixelGrid" },
  x: number,
  y: number,
): PixelValue => grid.pixels[y * grid.size + x] ?? 0;

export const setPixel = (
  grid: PixelGrid & { readonly _tag: "PixelGrid" },
  x: number,
  y: number,
  value: PixelValue,
): PixelGrid =>
  PixelGrid.PixelGrid({
    ...grid,
    pixels: grid.pixels.map((v, i) => (i === y * grid.size + x ? value : v)),
  });
