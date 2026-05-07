"use client";

import { type ReactNode, useCallback, useEffect, useMemo, useRef, useState, useTransition } from "react";
import {
  loadModel,
  saveModel,
  loadDatasetSamples,
  type SerializedModelEntry,
  type DatasetSample,
} from "../../actions";
import {
  listFolders,
  type FolderSummary,
} from "@/app/datasets/handwriting/actions";
import {
  deserializeNetworkDef,
  deserializeLayerParams,
  serializeLayerParams,
} from "@/domain/nn/serialize";
import { NetworkState } from "@/domain/nn/network";
import { TrainingConfig, TrainingSample } from "@/domain/nn/network";
import { predict } from "@/domain/nn/forward";
import { forwardNetwork } from "@/domain/nn/forward";
import { trainStep } from "@/domain/nn/training";
import * as V from "@/domain/math/vector";
import type { PixelValue, GridSize } from "@/domain/pixel-grid";
import { PixelCanvas } from "@/app/datasets/handwriting/_components/pixel-canvas";
import { findTask } from "@/domain/nn/task";
import { LayerVisualizer } from "./layer-visualizer";
import { TrainingPanel } from "./training-panel";
import { DecisionBoundaryVisualizer } from "./decision-boundary-visualizer";

/** Box-Muller 法によるガウス乱数生成 */
const gaussianRandom = (): number => {
  let u = 0;
  let v = 0;
  while (u === 0) u = Math.random();
  while (v === 0) v = Math.random();
  return Math.sqrt(-2.0 * Math.log(u)) * Math.cos(2.0 * Math.PI * v);
};

/**
 * ピクセル配列をループ付きで平行移動する。
 * dx > 0 は右, dy > 0 は下。はみ出した部分は反対側に回り込む。
 */
const translatePixels = (
  pixels: ReadonlyArray<PixelValue>,
  size: GridSize,
  dx: number,
  dy: number,
): ReadonlyArray<PixelValue> => {
  const result: PixelValue[] = Array.from({ length: size * size }, () => 0 as const);
  for (let y = 0; y < size; y++) {
    for (let x = 0; x < size; x++) {
      const srcIdx = y * size + x;
      if (pixels[srcIdx] === 0) continue;
      const nx = ((x + dx) % size + size) % size;
      const ny = ((y + dy) % size + size) % size;
      result[ny * size + nx] = 1;
    }
  }
  return result;
};

/**
 * 小型プレビュー表示。canvasの縮小版。
 */
function PixelPreview({ pixels, size, previewSize = 56 }: {
  readonly pixels: ReadonlyArray<PixelValue>;
  readonly size: GridSize;
  readonly previewSize?: number;
}): ReactNode {
  const pixelSize = previewSize / size;
  return (
    <svg
      width={previewSize}
      height={previewSize}
      viewBox={`0 0 ${String(previewSize) satisfies string} ${String(previewSize) satisfies string}`}
      className="border border-zinc-300 dark:border-zinc-600"
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

type Props = {
  readonly modelId: string;
};

export function ModelWorkspace({ modelId }: Props) {
  const [entry, setEntry] = useState<SerializedModelEntry | null>(null);
  const [networkState, setNetworkState] = useState<NetworkState | null>(null);
  const [isPending, startTransition] = useTransition();
  const [testPixels, setTestPixels] = useState<ReadonlyArray<PixelValue>>(() =>
    Array.from({ length: 784 }, () => 0 as const),
  );
  // 推論用: ストローク完了時にのみ更新される (描画中の再計算を避ける)
  const [committedPixels, setCommittedPixels] = useState<ReadonlyArray<PixelValue>>(() =>
    Array.from({ length: 784 }, () => 0 as const),
  );
  const [prevInputSize, setPrevInputSize] = useState(784);
  const [folders, setFolders] = useState<ReadonlyArray<FolderSummary>>([]);
  const [selectedFolder, setSelectedFolder] = useState("default");
  const [samples, setSamples] = useState<ReadonlyArray<DatasetSample>>([]);
  const [trainingStep, setTrainingStep] = useState(0);
  const [lastLoss, setLastLoss] = useState<number | null>(null);
  const [lossHistory, setLossHistory] = useState<ReadonlyArray<number>>([]);

  const networkStateRef = useRef(networkState);
  useEffect(() => {
    networkStateRef.current = networkState;
  }, [networkState]);

  const trainingStepRef = useRef(trainingStep);
  useEffect(() => {
    trainingStepRef.current = trainingStep;
  }, [trainingStep]);

  const lastLossRef = useRef(lastLoss);
  useEffect(() => {
    lastLossRef.current = lastLoss;
  }, [lastLoss]);

  const samplesRef = useRef(samples);
  useEffect(() => {
    samplesRef.current = samples;
  }, [samples]);

  const entryRef = useRef(entry);
  useEffect(() => {
    entryRef.current = entry;
  }, [entry]);

  const builtinDatasetNameRef = useRef<string | null>(null);

  const inputSize = entry ? (entry.layers[0]?.inputSize ?? 784) : 784;
  const task = entry ? findTask(entry.taskName) : null;
  const isBuiltinTask = task?.datasetType === "builtin";
  const gridSize: GridSize | null = isBuiltinTask ? null : (inputSize === 4096 ? 64 : 28);

  // モデル読み込み (load model)
  useEffect(() => {
    startTransition(async () => {
      const loaded = await loadModel(modelId);
      if (!loaded) {
        throw new Error(`モデル "${modelId satisfies string}" が見つかりません`);
      }
      setEntry(loaded);
      setTrainingStep(loaded.trainingStep);
      setLastLoss(loaded.lastLoss);

      const def = deserializeNetworkDef({
        layers: loaded.layers,
        lossName: loaded.lossName,
      });
      const params = loaded.params.map(deserializeLayerParams);
      const state = new NetworkState({ def, params });
      setNetworkState(state);
    });
  }, [modelId]);

  // 組み込みデータセット選択 (builtin dataset selection)
  const [builtinDatasetName, setBuiltinDatasetName] = useState<string | null>(null);
  useEffect(() => {
    builtinDatasetNameRef.current = builtinDatasetName;
  }, [builtinDatasetName]);

  // フォルダ一覧の読み込み (load folder list) — handwriting タスクのみ
  useEffect(() => {
    if (isBuiltinTask) return;
    startTransition(async () => {
      const f = await listFolders();
      setFolders(f);
    });
  }, [isBuiltinTask]);

  // データセット読み込み (load dataset) — handwriting タスクのみ
  useEffect(() => {
    if (isBuiltinTask || gridSize === null) return;
    startTransition(async () => {
      const loaded = await loadDatasetSamples(gridSize, selectedFolder);
      setSamples(loaded);
    });
  }, [gridSize, selectedFolder, isBuiltinTask]);

  // テスト入力のピクセルサイズ同期 (setState during render パターン) — handwriting タスクのみ
  if (!isBuiltinTask && inputSize !== prevInputSize) {
    setPrevInputSize(inputSize);
    setTestPixels(Array.from({ length: inputSize }, () => 0 as const));
    setCommittedPixels(Array.from({ length: inputSize }, () => 0 as const));
  }

  // ストロークやクリアで最新のピクセルを即座に追跡する
  const testPixelsRef = useRef(testPixels);

  const handlePixelsChange = useCallback((pixels: ReadonlyArray<PixelValue>) => {
    testPixelsRef.current = pixels;
    setTestPixels(pixels);
  }, []);

  // ストローク完了時に推論用ピクセルを更新する
  const handleStrokeEnd = useCallback(() => {
    setCommittedPixels(testPixelsRef.current);
  }, []);

  // 推論 (inference) — ストローク完了後の committedPixels に対してのみ実行
  const { predictionResult, layerOutputs } = useMemo(() => {
    if (!networkState) return { predictionResult: [] as ReadonlyArray<number>, layerOutputs: [] as ReadonlyArray<ReadonlyArray<number>> };
    const inputVec = V.from(committedPixels.map((p) => (p === 1 ? 1 : 0)));
    const output = predict(networkState, inputVec);
    const caches = forwardNetwork(networkState)(inputVec);
    return {
      predictionResult: [...output.data],
      layerOutputs: caches.map((c) => [...c.output.data]),
    };
  }, [networkState, committedPixels]);

  // モデル保存 (save model)
  const handleSave = useCallback(() => {
    if (!entry || !networkState) {
      throw new Error("モデルが未読み込みの状態で保存しようとしました");
    }
    startTransition(async () => {
      const updated: SerializedModelEntry = {
        ...entry,
        params: networkState.params.map(serializeLayerParams),
        trainingStep,
        lastLoss,
      };
      await saveModel(updated);
      setEntry(updated);
    });
  }, [entry, networkState, trainingStep, lastLoss]);

  // 学習中の状態管理
  const [isTraining, setIsTraining] = useState(false);
  const [trainingProgress, setTrainingProgress] = useState<{ readonly current: number; readonly total: number } | null>(null);
  const cancelTrainingRef = useRef(false);

  // 学習ステップ実行 (execute training step)
  // 10ステップごとにUIに制御を返し、プログレス表示を更新する
  const handleTrainStep = useCallback(
    (config: TrainingConfig, steps: number) => {
      const currentNetwork = networkStateRef.current;
      const currentSamples = samplesRef.current;
      if (!currentNetwork) {
        throw new Error("ネットワークが未初期化の状態で学習を実行しようとしました");
      }

      const currentEntry = entryRef.current;
      if (!currentEntry) {
        throw new Error("モデルが未読み込みの状態で学習を実行しようとしました");
      }
      const currentTask = findTask(currentEntry.taskName);

      let trainingSamples: ReadonlyArray<TrainingSample>;

      if (currentTask.datasetType === "builtin") {
        // 組み込みデータセットからサンプル生成
        const datasets = currentTask.builtinDatasets;
        if (!datasets) {
          throw new Error("組み込みデータセットが定義されていません");
        }
        const datasetName = builtinDatasetNameRef.current;
        const dataset = datasets.find((d) => d.name === datasetName) ?? datasets[0];
        if (!dataset) {
          throw new Error("組み込みデータセットが見つかりません");
        }
        trainingSamples = dataset.samples.map((s) =>
          new TrainingSample({ input: V.from(s.input), target: V.from(s.target) }),
        );
      } else {
        if (currentSamples.length === 0) {
          throw new Error("データセットが空の状態で学習を実行しようとしました");
        }
        const buildTarget = currentTask.buildTarget;
        if (!buildTarget) {
          throw new Error("buildTarget が定義されていません");
        }
        trainingSamples = currentSamples.map((s) => {
          const inputVec = V.from(s.pixels.map((p) => (p === 1 ? 1 : 0)));
          const target = V.from(buildTarget(s.digit));
          return new TrainingSample({ input: inputVec, target });
        });
      }

      cancelTrainingRef.current = false;
      setIsTraining(true);
      setTrainingProgress({ current: 0, total: steps });

      const CHUNK_SIZE = 10;
      let current = currentNetwork;
      let step = trainingStepRef.current;
      let loss = lastLossRef.current;
      const allNewLosses: number[] = [];
      let completed = 0;

      const finalize = () => {
        setLossHistory((prev) => [...prev, ...allNewLosses]);
        setIsTraining(false);
        setTrainingProgress(null);
      };

      const runChunk = () => {
        if (cancelTrainingRef.current) {
          finalize();
          return;
        }

        const end = Math.min(completed + CHUNK_SIZE, steps);
        for (let i = completed; i < end; i++) {
          const indices =
            config.mode === "sgd"
              ? [Math.floor(Math.random() * trainingSamples.length)]
              : Array.from(
                  { length: Math.min(config.batchSize, trainingSamples.length) },
                  () => Math.floor(Math.random() * trainingSamples.length),
                );

          // 入力ノイズ付与 (data augmentation)
          const samplesForStep = config.inputNoiseStd > 0
            ? trainingSamples.map((s) => new TrainingSample({
                input: V.map(s.input, (v) => v + gaussianRandom() * config.inputNoiseStd),
                target: s.target,
              }))
            : trainingSamples;

          const [updated, avgLoss] = trainStep(current, config, samplesForStep, indices);
          current = updated;
          step += 1;
          loss = avgLoss;
          allNewLosses.push(avgLoss);
        }
        completed = end;

        // UI更新
        setNetworkState(current);
        setTrainingStep(step);
        setLastLoss(loss);
        setTrainingProgress({ current: completed, total: steps });

        if (completed < steps) {
          // 次のチャンクをスケジュール (UIに制御を返す)
          requestAnimationFrame(runChunk);
        } else {
          finalize();
        }
      };

      requestAnimationFrame(runChunk);
    },
    [],
  );

  const handleCancelTraining = useCallback(() => {
    cancelTrainingRef.current = true;
  }, []);

  if (!entry || !networkState) {
    return (
      <div className="p-6 text-zinc-500">
        {isPending ? "読み込み中..." : "モデルが見つかりません"}
      </div>
    );
  }

  const topPrediction = predictionResult.length > 0
    ? predictionResult.reduce(
        (best, val, idx) => (val > best.val ? { idx, val } : best),
        { idx: 0, val: -Infinity },
      )
    : null;

  // 組み込みデータセットの現在の選択
  const currentBuiltinDataset = task?.builtinDatasets?.find((d) => d.name === builtinDatasetName) ?? task?.builtinDatasets?.[0] ?? null;
  const builtinSampleCount = currentBuiltinDataset?.samples.length ?? 0;

  return (
    <div className="flex flex-col gap-6">
      {/* ヘッダー情報 */}
      <div className="flex items-center justify-between">
        <div>
          <h2 className="font-bold text-lg">{entry.name}</h2>
          <span className="text-xs text-zinc-500">
            {`${findTask(entry.taskName).description satisfies string} | ${String(entry.layers.length) satisfies string}層 | ${entry.initStrategyName satisfies string} 初期化 | シード: ${String(entry.seed) satisfies string}`}
          </span>
        </div>
        <div className="flex items-center gap-2">
          <span className="text-sm text-zinc-500">
            ステップ: {String(trainingStep)}
            {lastLoss !== null && ` | 損失: ${lastLoss.toFixed(6) satisfies string}`}
          </span>
          <button
            type="button"
            className="px-3 py-1 text-sm bg-green-500 text-white rounded hover:bg-green-600 disabled:opacity-50"
            disabled={isPending}
            onClick={handleSave}
          >
            保存
          </button>
        </div>
      </div>

      {isBuiltinTask ? (
        /* 論理ゲート等の組み込みタスク用UI */
        <div className="flex flex-col gap-6">
          {/* 決定境界可視化 (decision boundary visualization) */}
          <DecisionBoundaryVisualizer
            networkState={networkState}
            dataset={currentBuiltinDataset}
            trainingStep={trainingStep}
            lossName={entry.lossName}
          />
        </div>
      ) : (
        <div className="flex gap-6 flex-wrap">
          {/* テスト入力 (test input) */}
          <div className="flex flex-col gap-3">
            <h3 className="font-medium text-sm text-zinc-600 dark:text-zinc-400">
              テスト入力 (test input)
            </h3>
            <div className="flex items-start gap-3">
              {gridSize !== null && (
                <>
                  <PixelCanvas
                    size={gridSize}
                    pixels={testPixels}
                    onPixelsChange={handlePixelsChange}
                    onStrokeEnd={handleStrokeEnd}
                    canvasSize={280}
                  />
                  <div className="flex flex-col gap-2">
                    <PixelPreview pixels={testPixels} size={gridSize} previewSize={56} />
                    {/* 平行移動 (translation) コントロール */}
                    <div className="flex flex-col items-center gap-0.5">
                      <span className="text-[10px] text-zinc-400">平行移動</span>
                      <button
                        type="button"
                        className="w-6 h-6 text-xs bg-zinc-200 dark:bg-zinc-700 rounded hover:bg-zinc-300 dark:hover:bg-zinc-600"
                        onClick={() => { const p = translatePixels(testPixels, gridSize, 0, -1); handlePixelsChange(p); setCommittedPixels(p); }}
                        title="上に移動"
                      >
                        ↑
                      </button>
                      <div className="flex gap-0.5">
                        <button
                          type="button"
                          className="w-6 h-6 text-xs bg-zinc-200 dark:bg-zinc-700 rounded hover:bg-zinc-300 dark:hover:bg-zinc-600"
                          onClick={() => { const p = translatePixels(testPixels, gridSize, -1, 0); handlePixelsChange(p); setCommittedPixels(p); }}
                          title="左に移動"
                        >
                          ←
                        </button>
                        <button
                          type="button"
                          className="w-6 h-6 text-xs bg-zinc-200 dark:bg-zinc-700 rounded hover:bg-zinc-300 dark:hover:bg-zinc-600"
                          onClick={() => { const p = translatePixels(testPixels, gridSize, 1, 0); handlePixelsChange(p); setCommittedPixels(p); }}
                          title="右に移動"
                        >
                          →
                        </button>
                      </div>
                      <button
                        type="button"
                        className="w-6 h-6 text-xs bg-zinc-200 dark:bg-zinc-700 rounded hover:bg-zinc-300 dark:hover:bg-zinc-600"
                        onClick={() => { const p = translatePixels(testPixels, gridSize, 0, 1); handlePixelsChange(p); setCommittedPixels(p); }}
                        title="下に移動"
                      >
                        ↓
                      </button>
                    </div>
                  </div>
                </>
              )}
            </div>
            {/* 推論結果 (prediction result) */}
            {predictionResult.length > 0 && (
              <div className="flex flex-col gap-1">
                <span className="text-xs text-zinc-500">推論結果 (prediction)</span>
                {topPrediction && (
                  <span className="text-2xl font-bold">
                    {String(topPrediction.idx)} ({(topPrediction.val * 100).toFixed(1) satisfies string}%)
                  </span>
                )}
                <div className="flex gap-0.5">
                  {predictionResult.map((val, idx) => (
                    <div key={idx} className="flex flex-col items-center text-[10px]">
                      <div
                        className="w-5 bg-blue-500 rounded-t"
                        style={{ height: `${String(Math.max(1, val * 60)) satisfies string}px` }}
                      />
                      <span>{String(idx)}</span>
                      <span className="text-zinc-400">{(val * 100).toFixed(0) satisfies string}</span>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>

          {/* レイヤー可視化 (layer visualization) */}
          <div className="flex-1 min-w-[300px]">
            <LayerVisualizer
              layers={entry.layers}
              params={networkState.params}
              layerOutputs={layerOutputs}
              inputPixels={committedPixels}
            />
          </div>
        </div>
      )}

      {/* 学習パネル (training panel) */}
      <TrainingPanel
        sampleCount={isBuiltinTask ? builtinSampleCount : samples.length}
        trainingStep={trainingStep}
        lastLoss={lastLoss}
        lossHistory={lossHistory}
        isPending={isPending}
        isTraining={isTraining}
        trainingProgress={trainingProgress}
        onTrainStep={handleTrainStep}
        onSave={handleSave}
        folders={folders}
        gridSize={gridSize}
        selectedFolder={selectedFolder}
        onSelectFolder={setSelectedFolder}
        builtinDatasets={task?.builtinDatasets ?? null}
        builtinDatasetName={builtinDatasetName ?? task?.builtinDatasets?.[0]?.name ?? null}
        onSelectBuiltinDataset={setBuiltinDatasetName}
        onCancel={handleCancelTraining}
      />
    </div>
  );
}
