"use server";

import * as fs from "node:fs/promises";
import * as path from "node:path";
import type { Digit } from "@/domain/dataset";
import type { GridSize, PixelValue } from "@/domain/pixel-grid";

const DATASETS_ROOT = path.join(process.cwd(), "datasets", "handwriting-numerics");

const entryDir = (size: GridSize, folder: string, digit: Digit): string =>
  path.join(DATASETS_ROOT, `size-${size satisfies number}`, folder, String(digit));

const ensureDir = async (dir: string): Promise<void> => {
  await fs.mkdir(dir, { recursive: true });
};

export type EntryMeta = {
  readonly id: string;
  readonly digit: Digit;
  readonly filename: string;
};

export type FolderSummary = {
  readonly size: GridSize;
  readonly folder: string;
  readonly counts: Readonly<Record<Digit, number>>;
};

export type EntryData = {
  readonly size: GridSize;
  readonly pixels: ReadonlyArray<PixelValue>;
};

export async function listFolders(): Promise<ReadonlyArray<FolderSummary>> {
  const results: FolderSummary[] = [];

  for (const size of [28, 64] as const) {
    const sizeDir = path.join(DATASETS_ROOT, `size-${size satisfies number}`);
    let folders: ReadonlyArray<string>;
    try {
      folders = await fs.readdir(sizeDir);
    } catch {
      continue;
    }

    for (const folder of folders) {
      const folderPath = path.join(sizeDir, folder);
      const stat = await fs.stat(folderPath);
      if (!stat.isDirectory()) continue;

      const counts = { 0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0, 9: 0 } as Record<Digit, number>;

      for (const d of [0, 1, 2, 3, 4, 5, 6, 7, 8, 9] as const) {
        const digitDir = path.join(folderPath, String(d));
        try {
          const files = await fs.readdir(digitDir);
          counts[d] = files.filter((f) => f.endsWith(".json")).length;
        } catch {
          // directory doesn't exist yet
        }
      }

      results.push({ size, folder, counts });
    }
  }

  return results;
}

export async function listEntries(
  size: GridSize,
  folder: string,
  digit: Digit,
): Promise<ReadonlyArray<EntryMeta>> {
  const dir = entryDir(size, folder, digit);
  let files: ReadonlyArray<string>;
  try {
    files = await fs.readdir(dir);
  } catch {
    return [];
  }

  return files
    .filter((f) => f.endsWith(".json"))
    .sort()
    .map((f) => ({
      id: `${size satisfies number}-${folder satisfies string}-${String(digit) satisfies string}-${f.replace(".json", "") satisfies string}`,
      digit,
      filename: f,
    }));
}

export async function loadEntry(
  size: GridSize,
  folder: string,
  digit: Digit,
  filename: string,
): Promise<EntryData | null> {
  const filePath = path.join(entryDir(size, folder, digit), filename);
  try {
    const raw = await fs.readFile(filePath, "utf-8");
    return JSON.parse(raw) as EntryData;
  } catch {
    return null;
  }
}

export async function saveEntry(
  size: GridSize,
  folder: string,
  digit: Digit,
  pixels: ReadonlyArray<PixelValue>,
): Promise<EntryMeta> {
  const dir = entryDir(size, folder, digit);
  await ensureDir(dir);

  let files: ReadonlyArray<string>;
  try {
    files = await fs.readdir(dir);
  } catch {
    files = [];
  }

  const existing = files.filter((f) => f.endsWith(".json")).sort();
  const nextNum =
    existing.length === 0
      ? 0
      : (() => {
          const last = existing[existing.length - 1];
          const n = last ? parseInt(last.replace(".json", ""), 10) : -1;
          return Number.isNaN(n) ? existing.length : n + 1;
        })();

  const filename = `${String(nextNum).padStart(4, "0") satisfies string}.json`;
  const data: EntryData = { size, pixels };
  await fs.writeFile(path.join(dir, filename), JSON.stringify(data));

  return {
    id: `${size satisfies number}-${folder satisfies string}-${String(digit) satisfies string}-${String(nextNum).padStart(4, "0") satisfies string}`,
    digit,
    filename,
  };
}

export async function deleteEntry(
  size: GridSize,
  folder: string,
  digit: Digit,
  filename: string,
): Promise<void> {
  const filePath = path.join(entryDir(size, folder, digit), filename);
  try {
    await fs.unlink(filePath);
  } catch {
    // already deleted
  }
}

export async function duplicateEntry(
  size: GridSize,
  folder: string,
  digit: Digit,
  filename: string,
): Promise<EntryMeta | null> {
  const data = await loadEntry(size, folder, digit, filename);
  if (!data) return null;
  return saveEntry(size, folder, digit, data.pixels);
}

export async function ensureDefaultFolder(size: GridSize): Promise<void> {
  for (const d of [0, 1, 2, 3, 4, 5, 6, 7, 8, 9] as const) {
    await ensureDir(entryDir(size, "default", d));
  }
}
