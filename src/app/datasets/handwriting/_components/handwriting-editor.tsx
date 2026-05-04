"use client";

import { useCallback, useEffect, useState, useTransition } from "react";
import type { Digit } from "@/domain/dataset";
import type { GridSize, PixelValue } from "@/domain/pixel-grid";
import { PixelCanvas } from "./pixel-canvas";
import { MiniPreview } from "./mini-preview";
import { FolderTree } from "./folder-tree";
import {
  listFolders,
  listEntries,
  loadEntry,
  saveEntry,
  deleteEntry,
  duplicateEntry,
  ensureDefaultFolder,
  type FolderSummary,
  type EntryMeta,
} from "../actions";

type LoadedEntry = {
  readonly meta: EntryMeta;
  readonly pixels: ReadonlyArray<PixelValue>;
};

export function HandwritingEditor() {
  const [size, setSize] = useState<GridSize>(28);
  const [folder, setFolder] = useState("default");
  const [digit, setDigit] = useState<Digit>(0);
  const [folders, setFolders] = useState<ReadonlyArray<FolderSummary>>([]);
  const [loadedEntries, setLoadedEntries] = useState<
    ReadonlyArray<LoadedEntry>
  >([]);
  const [selectedEntryId, setSelectedEntryId] = useState<string | null>(null);
  const [currentPixels, setCurrentPixels] = useState<
    ReadonlyArray<PixelValue>
  >(() => Array.from({ length: 28 * 28 }, () => 0 as const));
  const [isNewEntry, setIsNewEntry] = useState(true);
  const [isPending, startTransition] = useTransition();

  const refreshFolders = useCallback(() => {
    startTransition(async () => {
      await ensureDefaultFolder(size);
      const f = await listFolders();
      setFolders(f);
    });
  }, [size]);

  const refreshEntries = useCallback(() => {
    startTransition(async () => {
      const e = await listEntries(size, folder, digit);
      const loaded: LoadedEntry[] = [];
      for (const meta of e) {
        const data = await loadEntry(size, folder, digit, meta.filename);
        if (data) {
          loaded.push({ meta, pixels: data.pixels });
        }
      }
      setLoadedEntries(loaded);
    });
  }, [size, folder, digit]);

  useEffect(() => {
    refreshFolders();
  }, [refreshFolders]);

  useEffect(() => {
    refreshEntries();
  }, [refreshEntries]);

  const handleSelectFolder = useCallback(
    (s: GridSize, f: string, d: Digit) => {
      setSize(s);
      setFolder(f);
      setDigit(d);
      setSelectedEntryId(null);
      setIsNewEntry(true);
      setCurrentPixels(Array.from({ length: s * s }, () => 0 as const));
    },
    [],
  );

  const handleSave = useCallback(() => {
    startTransition(async () => {
      await saveEntry(size, folder, digit, currentPixels);
      setCurrentPixels(Array.from({ length: size * size }, () => 0 as const));
      setIsNewEntry(true);
      setSelectedEntryId(null);
      refreshEntries();
      refreshFolders();
    });
  }, [size, folder, digit, currentPixels, refreshEntries, refreshFolders]);

  const handleDelete = useCallback(
    (meta: EntryMeta) => {
      startTransition(async () => {
        await deleteEntry(size, folder, digit, meta.filename);
        if (selectedEntryId === meta.id) {
          setSelectedEntryId(null);
          setIsNewEntry(true);
          setCurrentPixels(
            Array.from({ length: size * size }, () => 0 as const),
          );
        }
        refreshEntries();
        refreshFolders();
      });
    },
    [size, folder, digit, selectedEntryId, refreshEntries, refreshFolders],
  );

  const handleDuplicate = useCallback(
    (meta: EntryMeta) => {
      startTransition(async () => {
        await duplicateEntry(size, folder, digit, meta.filename);
        refreshEntries();
        refreshFolders();
      });
    },
    [size, folder, digit, refreshEntries, refreshFolders],
  );

  const handleSelectEntry = useCallback(
    (entry: LoadedEntry) => {
      setSelectedEntryId(entry.meta.id);
      setCurrentPixels(entry.pixels);
      setIsNewEntry(false);
    },
    [],
  );

  const handleNewEntry = useCallback(() => {
    setSelectedEntryId(null);
    setIsNewEntry(true);
    setCurrentPixels(Array.from({ length: size * size }, () => 0 as const));
  }, [size]);

  // Space key to save and start new
  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      if (e.code === "Space" && e.target === document.body) {
        e.preventDefault();
        handleSave();
      }
    };
    window.addEventListener("keydown", handler);
    return () => window.removeEventListener("keydown", handler);
  }, [handleSave]);

  return (
    <div className="flex h-full">
      {/* Left sidebar */}
      <div className="w-64 border-r border-zinc-200 dark:border-zinc-700 p-3 overflow-y-auto flex flex-col gap-4">
        <h2 className="font-bold text-lg">Dataset Explorer</h2>
        <FolderTree
          folders={folders}
          selectedSize={size}
          selectedFolder={folder}
          selectedDigit={digit}
          onSelect={handleSelectFolder}
        />

        <hr className="border-zinc-200 dark:border-zinc-700" />

        <div className="flex flex-col gap-1">
          <div className="font-medium text-sm text-zinc-600 dark:text-zinc-400 flex items-center justify-between">
            <span>
              Digit {String(digit)} entries ({String(loadedEntries.length)})
            </span>
            <button
              type="button"
              className="text-xs px-2 py-0.5 bg-blue-500 text-white rounded hover:bg-blue-600"
              onClick={handleNewEntry}
            >
              + New
            </button>
          </div>
          <div className="flex flex-wrap gap-1 mt-1">
            {loadedEntries.map((entry) => (
              <div key={entry.meta.id} className="relative group">
                <MiniPreview
                  size={size}
                  pixels={entry.pixels}
                  previewSize={48}
                  selected={selectedEntryId === entry.meta.id}
                  onClick={() => handleSelectEntry(entry)}
                />
                <div className="absolute -top-1 -right-1 hidden group-hover:flex gap-0.5">
                  <button
                    type="button"
                    className="w-4 h-4 bg-red-500 text-white text-[8px] rounded-full flex items-center justify-center"
                    onClick={(e) => {
                      e.stopPropagation();
                      handleDelete(entry.meta);
                    }}
                    title="Delete"
                  >
                    x
                  </button>
                  <button
                    type="button"
                    className="w-4 h-4 bg-zinc-500 text-white text-[8px] rounded-full flex items-center justify-center"
                    onClick={(e) => {
                      e.stopPropagation();
                      handleDuplicate(entry.meta);
                    }}
                    title="Duplicate"
                  >
                    d
                  </button>
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* Main area */}
      <div className="flex-1 p-6 flex flex-col items-center gap-4">
        <div className="flex items-center gap-4">
          <div className="flex items-center gap-2">
            <span className="text-sm text-zinc-500">Size:</span>
            {([28, 64] as const).map((s) => (
              <button
                key={s}
                type="button"
                className={`px-3 py-1 text-sm rounded ${(
                  size === s
                    ? "bg-blue-500 text-white"
                    : "bg-zinc-200 dark:bg-zinc-700"
                ) satisfies string}`}
                onClick={() => handleSelectFolder(s, folder, digit)}
              >
                {String(s)}x{String(s)}
              </button>
            ))}
          </div>
          <div className="flex items-center gap-1">
            <span className="text-sm text-zinc-500">Digit:</span>
            {([0, 1, 2, 3, 4, 5, 6, 7, 8, 9] as const).map((d) => (
              <button
                key={d}
                type="button"
                className={`w-8 h-8 text-sm rounded ${(
                  digit === d
                    ? "bg-blue-500 text-white"
                    : "bg-zinc-200 dark:bg-zinc-700 hover:bg-zinc-300 dark:hover:bg-zinc-600"
                ) satisfies string}`}
                onClick={() => handleSelectFolder(size, folder, d)}
              >
                {String(d)}
              </button>
            ))}
          </div>
        </div>

        <div className="text-sm text-zinc-500">
          {isNewEntry ? "New entry" : `Editing: ${(selectedEntryId ?? "") satisfies string}`}
          {isPending && (
            <span className="ml-2 text-blue-500">Saving...</span>
          )}
        </div>

        <PixelCanvas
          size={size}
          pixels={currentPixels}
          onPixelsChange={setCurrentPixels}
        />

        <div className="flex gap-2">
          <button
            type="button"
            className="px-4 py-2 bg-blue-500 text-white rounded hover:bg-blue-600"
            onClick={handleSave}
          >
            Save & New
          </button>
          <span className="text-xs text-zinc-400 self-center">
            or press Space
          </span>
        </div>
      </div>
    </div>
  );
}
