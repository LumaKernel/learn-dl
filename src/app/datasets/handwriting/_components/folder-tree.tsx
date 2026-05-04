"use client";

import type { Digit } from "@/domain/dataset";
import type { GridSize } from "@/domain/pixel-grid";
import type { FolderSummary } from "../actions";

type FolderTreeProps = {
  readonly folders: ReadonlyArray<FolderSummary>;
  readonly selectedSize: GridSize;
  readonly selectedFolder: string;
  readonly selectedDigit: Digit;
  readonly onSelect: (size: GridSize, folder: string, digit: Digit) => void;
};

export function FolderTree({
  folders,
  selectedSize,
  selectedFolder,
  selectedDigit,
  onSelect,
}: FolderTreeProps) {
  const grouped = new Map<GridSize, ReadonlyArray<FolderSummary>>();
  for (const f of folders) {
    const existing = grouped.get(f.size) ?? [];
    grouped.set(f.size, [...existing, f]);
  }

  return (
    <div className="text-sm">
      {([28, 64] as const).map((size) => {
        const sizeFolders = grouped.get(size) ?? [];
        return (
          <div key={size} className="mb-3">
            <div className="font-bold text-zinc-500 mb-1">
              {String(size)}x{String(size)}
            </div>
            {sizeFolders.length === 0 && (
              <div className="text-zinc-400 pl-3 text-xs">フォルダなし</div>
            )}
            {sizeFolders.map((f) => {
              const total = Object.values(f.counts).reduce((a, b) => a + b, 0);
              return (
                <div key={`${String(size) satisfies string}-${f.folder satisfies string}`} className="pl-3 mb-1">
                  <div className="font-medium text-zinc-700 dark:text-zinc-300 flex items-center gap-1">
                    <span>{f.folder}</span>
                    <span className="text-xs text-zinc-400">({String(total)})</span>
                  </div>
                  <div className="pl-3 flex flex-wrap gap-x-1">
                    {([0, 1, 2, 3, 4, 5, 6, 7, 8, 9] as const).map((d) => {
                      const isSelected =
                        size === selectedSize &&
                        f.folder === selectedFolder &&
                        d === selectedDigit;
                      const count = f.counts[d];
                      return (
                        <button
                          key={d}
                          type="button"
                          className={`px-1.5 py-0.5 rounded text-xs cursor-pointer ${(
                            isSelected
                              ? "bg-blue-500 text-white"
                              : "hover:bg-zinc-200 dark:hover:bg-zinc-700"
                          ) satisfies string}`}
                          onClick={() => onSelect(size, f.folder, d)}
                        >
                          {String(d)}
                          <span className="text-[10px] ml-0.5 opacity-60">
                            {String(count)}
                          </span>
                        </button>
                      );
                    })}
                  </div>
                </div>
              );
            })}
          </div>
        );
      })}
    </div>
  );
}
