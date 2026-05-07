"use client";

import { useCallback, useState } from "react";
import { TrainingConfig } from "@/domain/nn/network";
import type { FolderSummary } from "@/app/datasets/handwriting/actions";
import type { GridSize } from "@/domain/pixel-grid";
import type { BuiltinDataset } from "@/domain/nn/task";

type TrainingProgress = {
  readonly current: number;
  readonly total: number;
};

type Props = {
  readonly sampleCount: number;
  readonly trainingStep: number;
  readonly lastLoss: number | null;
  readonly lossHistory: ReadonlyArray<number>;
  readonly isPending: boolean;
  readonly isTraining: boolean;
  readonly trainingProgress: TrainingProgress | null;
  readonly onTrainStep: (config: TrainingConfig, steps: number) => void;
  readonly onSave: () => void;
  readonly folders: ReadonlyArray<FolderSummary>;
  readonly gridSize: GridSize | null;
  readonly selectedFolder: string;
  readonly onSelectFolder: (folder: string) => void;
  readonly builtinDatasets: ReadonlyArray<BuiltinDataset> | null;
  readonly builtinDatasetName: string | null;
  readonly onSelectBuiltinDataset: (name: string) => void;
  readonly onCancel: () => void;
};

function LossGraph({ history }: { readonly history: ReadonlyArray<number> }) {
  if (history.length < 2) return null;

  const width = 600;
  const height = 150;
  const padding = 30;
  const graphWidth = width - padding * 2;
  const graphHeight = height - padding * 2;

  const maxLoss = history.reduce((m, v) => Math.max(m, v), 0);
  const minLoss = history.reduce((m, v) => Math.min(m, v), Infinity);
  const range = Math.max(maxLoss - minLoss, 1e-10);

  const points = history
    .map((loss, i) => {
      const x = padding + (i / (history.length - 1)) * graphWidth;
      const y = padding + (1 - (loss - minLoss) / range) * graphHeight;
      return `${x.toFixed(1) satisfies string},${y.toFixed(1) satisfies string}`;
    })
    .join(" ");

  return (
    <div className="flex flex-col gap-1">
      <span className="text-xs text-zinc-500">損失 (loss) の推移</span>
      <svg width={width} height={height} className="border border-zinc-200 dark:border-zinc-700 rounded">
        <rect width={width} height={height} fill="transparent" />
        {/* Y軸ラベル */}
        <text x={padding - 4} y={padding} textAnchor="end" fontSize="9" fill="#999">
          {maxLoss.toFixed(3)}
        </text>
        <text x={padding - 4} y={padding + graphHeight} textAnchor="end" fontSize="9" fill="#999">
          {minLoss.toFixed(3)}
        </text>
        {/* X軸ラベル */}
        <text x={padding} y={height - 4} fontSize="9" fill="#999">0</text>
        <text x={padding + graphWidth} y={height - 4} textAnchor="end" fontSize="9" fill="#999">
          {String(history.length)}
        </text>
        <polyline
          points={points}
          fill="none"
          stroke="#3b82f6"
          strokeWidth="1.5"
        />
      </svg>
    </div>
  );
}

export function TrainingPanel({
  sampleCount,
  trainingStep,
  lastLoss,
  lossHistory,
  isPending,
  isTraining,
  trainingProgress,
  onTrainStep,
  onSave,
  folders,
  gridSize,
  selectedFolder,
  onSelectFolder,
  builtinDatasets,
  builtinDatasetName,
  onSelectBuiltinDataset,
  onCancel,
}: Props) {
  const [learningRate, setLearningRate] = useState("0.01");
  // バッチサイズはサンプル数でcap
  const defaultBatchSize = Math.min(32, sampleCount > 0 ? sampleCount : 32);
  const [batchSize, setBatchSize] = useState(String(defaultBatchSize));
  const [mode, setMode] = useState<"sgd" | "minibatch">("minibatch");
  const [stepCount, setStepCount] = useState("100");
  const [inputNoise, setInputNoise] = useState(false);
  const [inputNoiseStd, setInputNoiseStd] = useState("0.1");
  const [paramError, setParamError] = useState<string | null>(null);

  const handleTrain = useCallback(
    (steps: number) => {
      setParamError(null);
      const lr = parseFloat(learningRate);
      const bs = parseInt(batchSize, 10);
      if (Number.isNaN(lr) || lr <= 0) {
        setParamError("学習率が不正です（正の数を入力してください）");
        return;
      }
      if (Number.isNaN(bs) || bs <= 0) {
        setParamError("バッチサイズが不正です（正の整数を入力してください）");
        return;
      }

      const noiseStd = inputNoise ? parseFloat(inputNoiseStd) : 0;
      if (inputNoise && (Number.isNaN(noiseStd) || noiseStd < 0)) {
        setParamError("ノイズ強度が不正です（0以上の数を入力してください）");
        return;
      }

      const config = new TrainingConfig({
        learningRate: lr,
        batchSize: bs,
        mode,
        inputNoiseStd: noiseStd,
      });
      onTrainStep(config, steps);
    },
    [learningRate, batchSize, mode, inputNoise, inputNoiseStd, onTrainStep],
  );

  return (
    <div className="border border-zinc-200 dark:border-zinc-700 rounded-lg p-4 flex flex-col gap-4">
      <div className="flex items-center justify-between">
        <h3 className="font-medium text-sm text-zinc-600 dark:text-zinc-400">
          学習 (training)
        </h3>
        <span className="text-xs text-zinc-500">
          データセット: {String(sampleCount)} サンプル
        </span>
      </div>

      {/* データセット選択 (dataset selection) */}
      {builtinDatasets ? (
        <div className="flex flex-wrap items-end gap-3">
          <label className="flex flex-col gap-1">
            <span className="text-xs text-zinc-500">データセット (論理ゲート)</span>
            <select
              className="border border-zinc-300 dark:border-zinc-600 rounded px-2 py-1 text-sm bg-transparent"
              value={builtinDatasetName ?? builtinDatasets[0]?.name ?? ""}
              onChange={(e) => onSelectBuiltinDataset(e.target.value)}
            >
              {builtinDatasets.map((d) => (
                <option key={d.name} value={d.name}>
                  {d.label}
                </option>
              ))}
            </select>
          </label>
          <span className="text-xs text-zinc-500 pb-1">
            {`${String(sampleCount) satisfies string} サンプル (全パターン網羅)`}
          </span>
        </div>
      ) : (
        <div className="flex flex-wrap items-end gap-3">
          <label className="flex flex-col gap-1">
            <span className="text-xs text-zinc-500">データセットフォルダ</span>
            <select
              className="border border-zinc-300 dark:border-zinc-600 rounded px-2 py-1 text-sm bg-transparent"
              value={selectedFolder}
              onChange={(e) => onSelectFolder(e.target.value)}
            >
              {folders
                .filter((f) => gridSize !== null && f.size === gridSize)
                .map((f) => {
                  const total = Object.values(f.counts).reduce((a, b) => a + b, 0);
                  return (
                    <option key={f.folder} value={f.folder}>
                      {`${f.folder satisfies string} (${String(total) satisfies string} サンプル)`}
                    </option>
                  );
                })}
              {gridSize !== null && folders.filter((f) => f.size === gridSize).length === 0 && (
                <option value="default">default (0 サンプル)</option>
              )}
            </select>
          </label>
          <span className="text-xs text-zinc-500 pb-1">
            {`読み込み済み: ${String(sampleCount) satisfies string} サンプル${(gridSize !== null ? ` (${String(gridSize) satisfies string}x${String(gridSize) satisfies string})` : "") satisfies string}`}
          </span>
        </div>
      )}

      {!builtinDatasets && sampleCount === 0 && (
        <p className="text-sm text-amber-500">
          データセットがありません。先に手書きデータを作成してください。
        </p>
      )}

      <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
        <label className="flex flex-col gap-1">
          <span className="text-xs text-zinc-500">学習率 (learning rate)</span>
          <input
            type="number"
            step="0.001"
            className="border border-zinc-300 dark:border-zinc-600 rounded px-2 py-1 text-sm bg-transparent"
            value={learningRate}
            onChange={(e) => setLearningRate(e.target.value)}
          />
        </label>

        <label className="flex flex-col gap-1">
          <span className="text-xs text-zinc-500">バッチサイズ (batch size)</span>
          <input
            type="number"
            className="border border-zinc-300 dark:border-zinc-600 rounded px-2 py-1 text-sm bg-transparent"
            value={batchSize}
            onChange={(e) => setBatchSize(e.target.value)}
          />
        </label>

        <label className="flex flex-col gap-1">
          <span className="text-xs text-zinc-500">モード (mode)</span>
          <select
            className="border border-zinc-300 dark:border-zinc-600 rounded px-2 py-1 text-sm bg-transparent"
            value={mode}
            onChange={(e) => {
              const v = e.target.value;
              if (v === "sgd" || v === "minibatch") setMode(v);
            }}
          >
            <option value="sgd">確率的勾配降下法 (SGD)</option>
            <option value="minibatch">ミニバッチ SGD</option>
          </select>
        </label>

        <label className="flex flex-col gap-1">
          <span className="text-xs text-zinc-500">ステップ数</span>
          <input
            type="number"
            className="border border-zinc-300 dark:border-zinc-600 rounded px-2 py-1 text-sm bg-transparent"
            value={stepCount}
            onChange={(e) => setStepCount(e.target.value)}
          />
        </label>
      </div>

      {/* 入力ノイズ (data augmentation) */}
      <div className="flex items-center gap-3">
        <label className="flex items-center gap-1.5 text-xs text-zinc-600 dark:text-zinc-400 cursor-pointer">
          <input
            type="checkbox"
            checked={inputNoise}
            onChange={(e) => setInputNoise(e.target.checked)}
            className="rounded"
          />
          入力ノイズ付与 (data augmentation)
        </label>
        {inputNoise && (
          <label className="flex items-center gap-1 text-xs">
            <span className="text-zinc-500">標準偏差 (σ):</span>
            <input
              type="number"
              step="0.01"
              className="w-20 border border-zinc-300 dark:border-zinc-600 rounded px-1.5 py-0.5 text-xs bg-transparent"
              value={inputNoiseStd}
              onChange={(e) => setInputNoiseStd(e.target.value)}
            />
          </label>
        )}
      </div>

      {paramError && (
        <p className="text-sm text-red-500">{paramError}</p>
      )}

      <div className="flex gap-2 items-center relative">
        <button
          type="button"
          className="px-3 py-1.5 text-sm bg-blue-500 text-white rounded hover:bg-blue-600 disabled:opacity-50"
          disabled={isPending || isTraining || (!builtinDatasets && sampleCount === 0)}
          onClick={() => handleTrain(1)}
        >
          1ステップ実行
        </button>
        <button
          type="button"
          className="px-3 py-1.5 text-sm bg-indigo-500 text-white rounded hover:bg-indigo-600 disabled:opacity-50"
          disabled={isPending || isTraining || (!builtinDatasets && sampleCount === 0)}
          onClick={() => handleTrain(parseInt(stepCount, 10) || 100)}
        >
          {`${String(parseInt(stepCount, 10) || 100) satisfies string}ステップ実行`}
        </button>
        <button
          type="button"
          className="px-3 py-1.5 text-sm bg-green-500 text-white rounded hover:bg-green-600 disabled:opacity-50"
          disabled={isPending || isTraining}
          onClick={onSave}
        >
          保存
        </button>
      </div>

      {/* 学習中のプログレス表示 */}
      {isTraining && trainingProgress && (
        <div className="flex flex-col gap-1">
          <div className="flex items-center gap-2 text-sm text-zinc-600 dark:text-zinc-400">
            <span className="animate-spin inline-block w-4 h-4 border-2 border-blue-500 border-t-transparent rounded-full" />
            <span>
              {`学習中... ${String(trainingProgress.current) satisfies string} / ${String(trainingProgress.total) satisfies string} (${String(Math.round((trainingProgress.current / trainingProgress.total) * 100)) satisfies string}%)`}
            </span>
            <button
              type="button"
              className="px-2 py-0.5 text-xs bg-red-500 text-white rounded hover:bg-red-600"
              onClick={onCancel}
            >
              中止
            </button>
          </div>
          <div className="w-full h-2 bg-zinc-200 dark:bg-zinc-700 rounded overflow-hidden">
            <div
              className="h-full bg-blue-500 transition-all duration-100"
              style={{ width: `${String((trainingProgress.current / trainingProgress.total) * 100) satisfies string}%` }}
            />
          </div>
        </div>
      )}

      <div className="text-sm text-zinc-500">
        ステップ: {String(trainingStep)}
        {lastLoss !== null && ` | 損失: ${lastLoss.toFixed(6) satisfies string}`}
      </div>

      <LossGraph history={lossHistory} />
    </div>
  );
}
