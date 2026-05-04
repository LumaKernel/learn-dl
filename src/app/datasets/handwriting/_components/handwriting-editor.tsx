"use client";

import { useCallback, useEffect, useRef, useState, useTransition } from "react";
import { useRouter, useSearchParams } from "next/navigation";
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

const parseSize = (v: string | null): GridSize => {
  if (v === "64") return 64;
  return 28;
};

const parseDigit = (v: string | null): Digit => {
  const n = Number(v);
  if (n >= 0 && n <= 9 && Number.isInteger(n)) return n as Digit;
  return 0;
};

const parseFolder = (v: string | null): string => v || "default";

export function HandwritingEditor() {
  const router = useRouter();
  const searchParams = useSearchParams();

  const size = parseSize(searchParams.get("size"));
  const folder = parseFolder(searchParams.get("folder"));
  const digit = parseDigit(searchParams.get("digit"));

  const [folders, setFolders] = useState<ReadonlyArray<FolderSummary>>([]);
  const [loadedEntries, setLoadedEntries] = useState<
    ReadonlyArray<LoadedEntry>
  >([]);
  const [selectedEntryId, setSelectedEntryId] = useState<string | null>(null);
  const [currentPixels, setCurrentPixels] = useState<
    ReadonlyArray<PixelValue>
  >(() => Array.from({ length: size * size }, () => 0 as const));
  const [isNewEntry, setIsNewEntry] = useState(true);
  const [isPending, startTransition] = useTransition();
  const undoStackRef = useRef<ReadonlyArray<ReadonlyArray<PixelValue>>>([]);
  const currentPixelsRef = useRef(currentPixels);
  useEffect(() => {
    currentPixelsRef.current = currentPixels;
  }, [currentPixels]);

  const pushUndo = useCallback(() => {
    undoStackRef.current = [...undoStackRef.current, currentPixelsRef.current];
    if (undoStackRef.current.length > 50) {
      undoStackRef.current = undoStackRef.current.slice(-50);
    }
  }, []);

  const handleUndo = useCallback(() => {
    const stack = undoStackRef.current;
    if (stack.length === 0) return;
    const prev = stack[stack.length - 1];
    undoStackRef.current = stack.slice(0, -1);
    if (prev) setCurrentPixels(prev);
  }, []);

  const updateUrl = useCallback(
    (s: GridSize, f: string, d: Digit) => {
      const params = new URLSearchParams();
      params.set("size", String(s));
      params.set("folder", f);
      params.set("digit", String(d));
      router.replace(`?${params.toString() satisfies string}`);
    },
    [router],
  );

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
      updateUrl(s, f, d);
      setSelectedEntryId(null);
      setIsNewEntry(true);
      setCurrentPixels(Array.from({ length: s * s }, () => 0 as const));
      undoStackRef.current = [];
    },
    [updateUrl],
  );

  const handleSave = useCallback(() => {
    startTransition(async () => {
      await saveEntry(size, folder, digit, currentPixels);
      setCurrentPixels(Array.from({ length: size * size }, () => 0 as const));
      setIsNewEntry(true);
      setSelectedEntryId(null);
      undoStackRef.current = [];
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
      undoStackRef.current = [];
    },
    [],
  );

  const handleNewEntry = useCallback(() => {
    setSelectedEntryId(null);
    setIsNewEntry(true);
    setCurrentPixels(Array.from({ length: size * size }, () => 0 as const));
    undoStackRef.current = [];
  }, [size]);

  // キーボードショートカット
  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      // Space: 保存して次へ
      if (e.code === "Space" && e.target === document.body) {
        e.preventDefault();
        handleSave();
        return;
      }
      // Cmd/Ctrl+Z: 元に戻す (undo)
      if (e.key === "z" && (e.metaKey || e.ctrlKey) && !e.shiftKey) {
        e.preventDefault();
        handleUndo();
      }
    };
    window.addEventListener("keydown", handler);
    return () => window.removeEventListener("keydown", handler);
  }, [handleSave, handleUndo]);

  return (
    <div className="flex h-full">
      {/* 左サイドバー */}
      <div className="w-64 border-r border-zinc-200 dark:border-zinc-700 p-3 overflow-y-auto flex flex-col gap-4">
        <h2 className="font-bold text-lg">データセット一覧</h2>
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
              数字 {String(digit)} のデータ ({String(loadedEntries.length)}件)
            </span>
            <button
              type="button"
              className="text-xs px-2 py-0.5 bg-blue-500 text-white rounded hover:bg-blue-600"
              onClick={handleNewEntry}
            >
              + 新規
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
                    title="削除"
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
                    title="複製"
                  >
                    d
                  </button>
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* メインエリア */}
      <div className="flex-1 p-6 flex flex-col items-center gap-4">
        <div className="flex items-center gap-4">
          <div className="flex items-center gap-2">
            <span className="text-sm text-zinc-500">サイズ:</span>
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
            <span className="text-sm text-zinc-500">数字:</span>
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
          {isNewEntry ? "新規エントリ" : `編集中: ${(selectedEntryId ?? "") satisfies string}`}
          {isPending && (
            <span className="ml-2 text-blue-500">保存中...</span>
          )}
        </div>

        <div className="flex items-start gap-4">
          <PixelCanvas
            size={size}
            pixels={currentPixels}
            onPixelsChange={setCurrentPixels}
            onStrokeStart={pushUndo}
          />
          <div className="flex flex-col items-center gap-1">
            <span className="text-xs text-zinc-400">プレビュー</span>
            <MiniPreview size={size} pixels={currentPixels} previewSize={size} />
            <MiniPreview size={size} pixels={currentPixels} previewSize={48} />
          </div>
        </div>

        <div className="flex gap-2">
          <button
            type="button"
            className="px-4 py-2 bg-blue-500 text-white rounded hover:bg-blue-600"
            onClick={handleSave}
          >
            保存して次へ
          </button>
          <span className="text-xs text-zinc-400 self-center">
            Space キーでも保存
          </span>
        </div>
      </div>
    </div>
  );
}
