"use server";

import * as fs from "node:fs/promises";
import * as path from "node:path";
import type { Digit } from "@/domain/dataset";
import type { GridSize, PixelValue } from "@/domain/pixel-grid";
import { allActivations } from "@/domain/nn/activation";
import { allLossFunctions } from "@/domain/nn/loss";
import { allInitStrategies, initNetwork } from "@/domain/nn/init";
import { LayerDef } from "@/domain/nn/layer";
import { NetworkDef } from "@/domain/nn/network";
import { serializeNetworkDef, serializeLayerParams } from "@/domain/nn/serialize";

/**
 * ローカル開発環境でのみ書き込み可能。
 * デプロイ環境 (Vercel 等) ではファイルシステムへの書き込みが不可のため、
 * 書き込み系の操作は静かに失敗する。
 */
const MODELS_ROOT = path.join(process.cwd(), "models");
const DATASETS_ROOT = path.join(process.cwd(), "datasets", "handwriting-numerics");

const isWritable = (): boolean => process.env["NODE_ENV"] !== "production";

const ensureDir = async (dir: string): Promise<void> => {
  if (!isWritable()) return;
  await fs.mkdir(dir, { recursive: true });
};

/**
 * モデル (model) のシリアライズ可能な形式。
 * Effect TaggedClass は JSON.stringify でシリアライズ不可のため、
 * プレーンな構造に変換する。
 */
export type SerializedLayerDef = {
  readonly inputSize: number;
  readonly outputSize: number;
  readonly activationName: string;
};

export type SerializedLayerParams = {
  readonly weights: ReadonlyArray<ReadonlyArray<number>>;
  readonly bias: ReadonlyArray<number>;
};

export type SerializedModelEntry = {
  readonly id: string;
  readonly name: string;
  readonly createdAt: number;
  readonly taskName: string;
  readonly lossName: string;
  readonly layers: ReadonlyArray<SerializedLayerDef>;
  readonly params: ReadonlyArray<SerializedLayerParams>;
  readonly initStrategyName: string;
  readonly seed: number;
  readonly trainingStep: number;
  readonly lastLoss: number | null;
};

export type ModelListItem = {
  readonly id: string;
  readonly name: string;
  readonly createdAt: number;
  readonly taskName: string;
  readonly layerCount: number;
  readonly trainingStep: number;
  readonly lastLoss: number | null;
};

export async function listModels(): Promise<ReadonlyArray<ModelListItem>> {
  await ensureDir(MODELS_ROOT);
  let files: ReadonlyArray<string>;
  try {
    files = await fs.readdir(MODELS_ROOT);
  } catch {
    return [];
  }

  const results: ModelListItem[] = [];
  for (const f of files) {
    if (!f.endsWith(".json")) continue;
    try {
      const raw = await fs.readFile(path.join(MODELS_ROOT, f), "utf-8");
      const data = JSON.parse(raw) as SerializedModelEntry;
      results.push({
        id: data.id,
        name: data.name,
        createdAt: data.createdAt,
        taskName: data.taskName ?? "digitClassification",
        layerCount: data.layers.length,
        trainingStep: data.trainingStep,
        lastLoss: data.lastLoss,
      });
    } catch {
      // 壊れたファイルは無視
    }
  }

  return results.sort((a, b) => b.createdAt - a.createdAt);
}

export async function loadModel(id: string): Promise<SerializedModelEntry | null> {
  const filePath = path.join(MODELS_ROOT, `${id satisfies string}.json`);
  try {
    const raw = await fs.readFile(filePath, "utf-8");
    const data = JSON.parse(raw) as SerializedModelEntry;
    // taskName が無い古いモデルは digitClassification をデフォルトにする
    if (!data.taskName) {
      return { ...data, taskName: "digitClassification" };
    }
    return data;
  } catch {
    return null;
  }
}

export async function saveModel(entry: SerializedModelEntry): Promise<void> {
  if (!isWritable()) return;
  await ensureDir(MODELS_ROOT);
  const filePath = path.join(MODELS_ROOT, `${entry.id satisfies string}.json`);
  await fs.writeFile(filePath, JSON.stringify(entry, null, 2));
}

export async function deleteModel(id: string): Promise<void> {
  if (!isWritable()) return;
  const filePath = path.join(MODELS_ROOT, `${id satisfies string}.json`);
  try {
    await fs.unlink(filePath);
  } catch {
    // 既に削除済み
  }
}

/**
 * モデル作成リクエスト。クライアントからは軽量な設定のみ送信し、
 * 重み初期化はサーバー側で行う。
 */
export type CreateModelRequest = {
  readonly name: string;
  readonly taskName: string;
  readonly inputSize: number;
  readonly lossName: string;
  readonly initStrategyName: string;
  readonly seed: number;
  readonly hiddenLayers: ReadonlyArray<{
    readonly outputSize: number;
    readonly activationName: string;
  }>;
  readonly outputSize: number;
  readonly outputActivationName: string;
};

export async function createModel(req: CreateModelRequest): Promise<string> {
  const strategy = allInitStrategies.find((s) => s.name === req.initStrategyName);
  if (!strategy) throw new Error(`不明な初期化戦略: ${req.initStrategyName satisfies string}`);

  const lossFunction = allLossFunctions.find((l) => l.name === req.lossName);
  if (!lossFunction) throw new Error(`不明な損失関数: ${req.lossName satisfies string}`);

  const outputActivation = allActivations.find((a) => a.name === req.outputActivationName);
  if (!outputActivation) throw new Error(`不明な活性化関数: ${req.outputActivationName satisfies string}`);

  const layerDefs: LayerDef[] = [];
  let prevSize = req.inputSize;

  for (const hl of req.hiddenLayers) {
    const activation = allActivations.find((a) => a.name === hl.activationName);
    if (!activation) throw new Error(`不明な活性化関数: ${hl.activationName satisfies string}`);
    layerDefs.push(new LayerDef({ inputSize: prevSize, outputSize: hl.outputSize, activation }));
    prevSize = hl.outputSize;
  }

  layerDefs.push(new LayerDef({
    inputSize: prevSize,
    outputSize: req.outputSize,
    activation: outputActivation,
  }));

  const networkDef = new NetworkDef({ layers: layerDefs, lossFunction });
  const networkState = initNetwork(networkDef, strategy, req.seed);

  // eslint-disable-next-line @luma-dev/luma-ts/no-date -- Temporal の型定義が未導入のため
  const now = Date.now();
  const id = `model-${now.toString(36) satisfies string}`;
  const serializedDef = serializeNetworkDef(networkDef);

  const entry: SerializedModelEntry = {
    id,
    name: req.name,
    createdAt: now,
    taskName: req.taskName,
    lossName: serializedDef.lossName,
    layers: serializedDef.layers,
    params: networkState.params.map(serializeLayerParams),
    initStrategyName: req.initStrategyName,
    seed: req.seed,
    trainingStep: 0,
    lastLoss: null,
  };

  await ensureDir(MODELS_ROOT);
  const filePath = path.join(MODELS_ROOT, `${id satisfies string}.json`);
  await fs.writeFile(filePath, JSON.stringify(entry, null, 2));

  return id;
}

/**
 * データセット (dataset) からすべてのエントリを読み込む。
 * 学習 (training) に使用するため、ピクセル (pixel) データも含む。
 */
export type DatasetSample = {
  readonly digit: Digit;
  readonly pixels: ReadonlyArray<PixelValue>;
  readonly size: GridSize;
};

export async function loadDatasetSamples(
  size: GridSize,
  folder: string,
): Promise<ReadonlyArray<DatasetSample>> {
  const samples: DatasetSample[] = [];

  for (const d of [0, 1, 2, 3, 4, 5, 6, 7, 8, 9] as const) {
    const dir = path.join(DATASETS_ROOT, `size-${size satisfies number}`, folder, String(d));
    let files: ReadonlyArray<string>;
    try {
      files = await fs.readdir(dir);
    } catch {
      continue;
    }

    for (const f of files.filter((f) => f.endsWith(".json")).sort()) {
      try {
        const raw = await fs.readFile(path.join(dir, f), "utf-8");
        const data = JSON.parse(raw) as { readonly size: GridSize; readonly pixels: ReadonlyArray<PixelValue> };
        samples.push({ digit: d, pixels: data.pixels, size: data.size });
      } catch {
        // 壊れたファイルは無視
      }
    }
  }

  return samples;
}
