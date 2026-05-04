import { Data } from "effect";
import type { GridSize, PixelGrid } from "./pixel-grid";

export type Digit = 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9;

export const Digit = {
  all: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9] as const satisfies readonly Digit[],
} as const;

export type DatasetEntryId = string & { readonly _brand: unique symbol };

export const DatasetEntryId = (id: string): DatasetEntryId =>
  id as DatasetEntryId;

export type DatasetEntry = Data.TaggedEnum<{
  readonly DatasetEntry: {
    readonly id: DatasetEntryId;
    readonly digit: Digit;
    readonly grid: PixelGrid & { readonly _tag: "PixelGrid" };
  };
}>;

export const DatasetEntry = Data.taggedEnum<DatasetEntry>();

export type DatasetFolderPath = Data.TaggedEnum<{
  readonly DatasetFolderPath: {
    readonly size: GridSize;
    readonly folder: string;
  };
}>;

export const DatasetFolderPath = Data.taggedEnum<DatasetFolderPath>();

export type DatasetSummary = Data.TaggedEnum<{
  readonly DatasetSummary: {
    readonly path: DatasetFolderPath & { readonly _tag: "DatasetFolderPath" };
    readonly counts: Readonly<Record<Digit, number>>;
  };
}>;

export const DatasetSummary = Data.taggedEnum<DatasetSummary>();
