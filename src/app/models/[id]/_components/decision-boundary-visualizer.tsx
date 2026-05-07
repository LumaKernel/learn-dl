"use client";

import { type ReactNode, memo, useEffect, useMemo, useRef, useState } from "react";
import type { NetworkState } from "@/domain/nn/network";
import type { BuiltinDataset } from "@/domain/nn/task";
import { predict } from "@/domain/nn/forward";
import * as V from "@/domain/math/vector";

type Props = {
  readonly networkState: NetworkState;
  readonly dataset: BuiltinDataset | null;
  readonly trainingStep: number;
  readonly lossName: string;
};

/**
 * 決定境界 (decision boundary) の直線パラメータ。
 * 1層ネットワークの場合: w1*x1 + w2*x2 + b = 0
 */
type BoundaryLine = {
  readonly w1: number;
  readonly w2: number;
  readonly b: number;
  readonly step: number;
};

/**
 * 2D座標上のネットワーク出力をサンプリングしたグリッド。
 */
type OutputGrid = {
  readonly values: ReadonlyArray<ReadonlyArray<number>>;
  readonly resolution: number;
  readonly rangeMin: number;
  readonly rangeMax: number;
};

/**
 * ネットワークが1層かどうか判定し、その場合の重みを取得する。
 */
const getSingleLayerWeights = (networkState: NetworkState): BoundaryLine | null => {
  if (networkState.def.layers.length !== 1) return null;
  const params = networkState.params[0];
  if (!params) return null;
  // 入力2, 出力1 の場合のみ直線表示
  if (params.weights.rows !== 1 || params.weights.cols !== 2) return null;
  const w1 = params.weights.at(0, 0);
  const w2 = params.weights.at(0, 1);
  const b = params.bias.at(0);
  return { w1, w2, b, step: 0 };
};

/**
 * 出力グリッドを計算する (ネットワークの出力を2D平面上でサンプリング)。
 */
const computeOutputGrid = (
  networkState: NetworkState,
  resolution: number,
  rangeMin: number,
  rangeMax: number,
): OutputGrid => {
  const step = (rangeMax - rangeMin) / resolution;
  const values: ReadonlyArray<number>[] = [];
  for (let yi = 0; yi < resolution; yi++) {
    const row: number[] = [];
    const y = rangeMax - yi * step; // 上が大きい値
    for (let xi = 0; xi < resolution; xi++) {
      const x = rangeMin + xi * step;
      const input = V.from([x, y]);
      const output = predict(networkState, input);
      row.push(output.at(0));
    }
    values.push(row);
  }
  return { values, resolution, rangeMin, rangeMax };
};

/**
 * ヒートマップ色生成。0=赤系、0.5=白、1=青系。
 */
const outputToColor = (value: number): string => {
  const clamped = Math.max(0, Math.min(1, value));
  if (clamped < 0.5) {
    // 0→赤, 0.5→白
    const t = clamped * 2;
    const r = 255;
    const g = Math.round(100 + 155 * t);
    const b = Math.round(100 + 155 * t);
    return `rgb(${String(r) satisfies string},${String(g) satisfies string},${String(b) satisfies string})`;
  }
  // 0.5→白, 1→青
  const t = (clamped - 0.5) * 2;
  const r = Math.round(255 - 155 * t);
  const g = Math.round(255 - 155 * t);
  const b = 255;
  return `rgb(${String(r) satisfies string},${String(g) satisfies string},${String(b) satisfies string})`;
};

/** ヒートマップ表示コンポーネント */
const Heatmap = memo(function Heatmap({
  grid,
  size,
}: {
  readonly grid: OutputGrid;
  readonly size: number;
}): ReactNode {
  const cellSize = size / grid.resolution;
  return (
    <g>
      {grid.values.map((row, yi) =>
        row.map((val, xi) => (
          <rect
            key={`${String(yi) satisfies string}-${String(xi) satisfies string}`}
            x={xi * cellSize}
            y={yi * cellSize}
            width={cellSize + 0.5}
            height={cellSize + 0.5}
            fill={outputToColor(val)}
            opacity={0.7}
          />
        )),
      )}
    </g>
  );
});

/** 決定境界の直線描画 */
const BoundaryLineComponent = memo(function BoundaryLineComponent({
  line,
  size,
  rangeMin,
  rangeMax,
  opacity,
  color,
  strokeWidth,
}: {
  readonly line: BoundaryLine;
  readonly size: number;
  readonly rangeMin: number;
  readonly rangeMax: number;
  readonly opacity: number;
  readonly color: string;
  readonly strokeWidth: number;
}): ReactNode {
  const { w1, w2, b } = line;
  const range = rangeMax - rangeMin;

  // Wx+b=0 の直線を座標→ピクセルに変換
  const toPixelX = (x: number) => ((x - rangeMin) / range) * size;
  const toPixelY = (y: number) => ((rangeMax - y) / range) * size; // Y軸反転

  // 直線の2点を求める（クリッピング含む）
  // w1*x + w2*y + b = 0 → y = -(w1*x + b) / w2 or x = -(w2*y + b) / w1
  const points: { x: number; y: number }[] = [];

  if (Math.abs(w2) > 1e-10) {
    // x = rangeMin での y
    const y1 = -(w1 * rangeMin + b) / w2;
    if (y1 >= rangeMin && y1 <= rangeMax) points.push({ x: rangeMin, y: y1 });
    // x = rangeMax での y
    const y2 = -(w1 * rangeMax + b) / w2;
    if (y2 >= rangeMin && y2 <= rangeMax) points.push({ x: rangeMax, y: y2 });
  }
  if (Math.abs(w1) > 1e-10) {
    // y = rangeMin での x
    const x1 = -(w2 * rangeMin + b) / w1;
    if (x1 > rangeMin && x1 < rangeMax) points.push({ x: x1, y: rangeMin });
    // y = rangeMax での x
    const x2 = -(w2 * rangeMax + b) / w1;
    if (x2 > rangeMin && x2 < rangeMax) points.push({ x: x2, y: rangeMax });
  }

  if (points.length < 2) return null;

  // 重複除去
  const uniquePoints = points.filter(
    (p, i) => points.findIndex((q) => Math.abs(q.x - p.x) < 1e-8 && Math.abs(q.y - p.y) < 1e-8) === i,
  );
  if (uniquePoints.length < 2) return null;

  const p0 = uniquePoints[0]!;
  const p1 = uniquePoints[1]!;

  return (
    <line
      x1={toPixelX(p0.x)}
      y1={toPixelY(p0.y)}
      x2={toPixelX(p1.x)}
      y2={toPixelY(p1.y)}
      stroke={color}
      strokeWidth={strokeWidth}
      opacity={opacity}
      strokeDasharray={opacity < 1 ? "4 2" : "none"}
    />
  );
});

/**
 * 各サンプルの損失計算式と値を生成する。
 */
const computeSampleLoss = (
  lossName: string,
  output: number,
  target: number,
): { readonly formula: string; readonly value: number } => {
  const y = Math.max(1e-15, Math.min(1 - 1e-15, output));
  const t = target;

  switch (lossName) {
    case "binaryCrossEntropy": {
      const value = -(t * Math.log(y) + (1 - t) * Math.log(1 - y));
      // 変数代入を見せる式を生成
      const tLogY = t === 0 ? "0" : `${String(t) satisfies string}·log(${y.toFixed(4) satisfies string})`;
      const oneMinusT = 1 - t;
      const oneMinusTLogOneMinusY = oneMinusT === 0
        ? "0"
        : `${String(oneMinusT) satisfies string}·log(${(1 - y).toFixed(4) satisfies string})`;
      const formula = `-[${tLogY satisfies string} + ${oneMinusTLogOneMinusY satisfies string}]`;
      return { formula, value };
    }
    case "mse": {
      const value = (y - t) ** 2;
      const formula = `(${y.toFixed(4) satisfies string} - ${String(t) satisfies string})²`;
      return { formula, value };
    }
    case "crossEntropy": {
      const value = t === 0 ? 0 : -t * Math.log(y);
      const formula = t === 0
        ? "0"
        : `-${String(t) satisfies string}·log(${y.toFixed(4) satisfies string})`;
      return { formula, value };
    }
    default: {
      const value = Math.abs(output - target);
      return { formula: `|${output.toFixed(4) satisfies string} - ${String(target) satisfies string}|`, value };
    }
  }
};

export function DecisionBoundaryVisualizer({ networkState, dataset, trainingStep, lossName }: Props): ReactNode {
  const SIZE = 400;
  const RANGE_MIN = -0.5;
  const RANGE_MAX = 1.5;
  const RESOLUTION = 40; // ヒートマップの解像度

  const [showGhosts, setShowGhosts] = useState(true);
  const [showHeatmap, setShowHeatmap] = useState(true);

  // 過去の決定境界の履歴 (残像用)
  const [boundaryHistory, setBoundaryHistory] = useState<ReadonlyArray<BoundaryLine>>([]);

  // 現在の決定境界
  const currentBoundary = useMemo(() => getSingleLayerWeights(networkState), [networkState]);

  // 履歴に追加 (ステップが進んだ場合のみ)
  const lastRecordedStepRef = useRef(-1);
  useEffect(() => {
    if (currentBoundary && trainingStep !== lastRecordedStepRef.current) {
      lastRecordedStepRef.current = trainingStep;
      setBoundaryHistory((prev) => {
        const newHistory = [...prev, { ...currentBoundary, step: trainingStep }];
        // 最大50個まで保持
        return newHistory.length > 50 ? newHistory.slice(newHistory.length - 50) : newHistory;
      });
    }
  }, [currentBoundary, trainingStep]);

  // ヒートマップ計算
  const outputGrid = useMemo(
    () => showHeatmap ? computeOutputGrid(networkState, RESOLUTION, RANGE_MIN, RANGE_MAX) : null,
    [networkState, showHeatmap, RESOLUTION, RANGE_MIN, RANGE_MAX],
  );

  // 各データ点のネットワーク出力
  const dataPointOutputs = useMemo(() => {
    if (!dataset) return [];
    return dataset.samples.map((s) => {
      const input = V.from(s.input);
      const output = predict(networkState, input);
      return output.at(0);
    });
  }, [networkState, dataset]);

  const range = RANGE_MAX - RANGE_MIN;
  const toPixelX = (x: number) => ((x - RANGE_MIN) / range) * SIZE;
  const toPixelY = (y: number) => ((RANGE_MAX - y) / range) * SIZE;

  const ghostLines = boundaryHistory;
  // 最新N個の残像を表示 (古いほど薄い)
  const maxGhosts = 20;
  const visibleGhosts = ghostLines.slice(Math.max(0, ghostLines.length - maxGhosts - 1), ghostLines.length - 1);

  return (
    <div className="flex flex-col gap-4">
      <div className="flex items-center justify-between">
        <h3 className="font-medium text-sm text-zinc-600 dark:text-zinc-400">
          決定境界の可視化 (decision boundary)
        </h3>
        <div className="flex gap-3">
          <label className="flex items-center gap-1 text-xs text-zinc-500 cursor-pointer">
            <input
              type="checkbox"
              checked={showHeatmap}
              onChange={(e) => setShowHeatmap(e.target.checked)}
              className="rounded"
            />
            ヒートマップ
          </label>
          {currentBoundary && (
            <label className="flex items-center gap-1 text-xs text-zinc-500 cursor-pointer">
              <input
                type="checkbox"
                checked={showGhosts}
                onChange={(e) => setShowGhosts(e.target.checked)}
                className="rounded"
              />
              残像表示
            </label>
          )}
        </div>
      </div>

      <div className="flex gap-6 flex-wrap">
        {/* メインの2Dプロット */}
        <svg
          width={SIZE}
          height={SIZE}
          className="border border-zinc-300 dark:border-zinc-600 rounded bg-white dark:bg-zinc-900"
        >
          {/* ヒートマップ (出力値の色表示) */}
          {outputGrid && <Heatmap grid={outputGrid} size={SIZE} />}

          {/* グリッド線 */}
          <line x1={toPixelX(0)} y1={0} x2={toPixelX(0)} y2={SIZE} stroke="#999" strokeWidth={0.5} opacity={0.5} />
          <line x1={toPixelX(1)} y1={0} x2={toPixelX(1)} y2={SIZE} stroke="#999" strokeWidth={0.5} opacity={0.5} />
          <line x1={0} y1={toPixelY(0)} x2={SIZE} y2={toPixelY(0)} stroke="#999" strokeWidth={0.5} opacity={0.5} />
          <line x1={0} y1={toPixelY(1)} x2={SIZE} y2={toPixelY(1)} stroke="#999" strokeWidth={0.5} opacity={0.5} />

          {/* 軸ラベル */}
          <text x={toPixelX(0) - 3} y={SIZE - 4} fontSize="10" fill="#666" textAnchor="middle">0</text>
          <text x={toPixelX(1) - 3} y={SIZE - 4} fontSize="10" fill="#666" textAnchor="middle">1</text>
          <text x={4} y={toPixelY(0) + 4} fontSize="10" fill="#666">0</text>
          <text x={4} y={toPixelY(1) + 4} fontSize="10" fill="#666">1</text>
          <text x={SIZE - 20} y={SIZE - 4} fontSize="10" fill="#666">x₁</text>
          <text x={4} y={14} fontSize="10" fill="#666">x₂</text>

          {/* 残像 (過去の決定境界) */}
          {showGhosts && visibleGhosts.map((ghost, i) => (
            <BoundaryLineComponent
              key={ghost.step}
              line={ghost}
              size={SIZE}
              rangeMin={RANGE_MIN}
              rangeMax={RANGE_MAX}
              opacity={0.1 + (i / visibleGhosts.length) * 0.3}
              color="#888"
              strokeWidth={1}
            />
          ))}

          {/* 現在の決定境界 */}
          {currentBoundary && (
            <BoundaryLineComponent
              line={currentBoundary}
              size={SIZE}
              rangeMin={RANGE_MIN}
              rangeMax={RANGE_MAX}
              opacity={1}
              color="#000"
              strokeWidth={2.5}
            />
          )}

          {/* データ点 */}
          {dataset?.samples.map((sample, i) => {
            const x = sample.input[0] ?? 0;
            const y = sample.input[1] ?? 0;
            const target = sample.target[0] ?? 0;
            const output = dataPointOutputs[i] ?? 0;
            const px = toPixelX(x);
            const py = toPixelY(y);
            // ターゲット: 1=青丸, 0=赤丸
            const fillColor = target > 0.5 ? "#2563eb" : "#dc2626";
            // 出力値に基づく枠の色
            const strokeColor = output > 0.5 ? "#2563eb" : "#dc2626";
            return (
              <g key={i}>
                <circle
                  cx={px}
                  cy={py}
                  r={14}
                  fill={fillColor}
                  stroke={strokeColor}
                  strokeWidth={3}
                  opacity={0.9}
                />
                {/* 出力値テキスト */}
                <text
                  x={px}
                  y={py + 4}
                  fontSize="10"
                  fill="white"
                  textAnchor="middle"
                  fontWeight="bold"
                >
                  {output.toFixed(2)}
                </text>
              </g>
            );
          })}
        </svg>

        {/* 右側の情報パネル */}
        <div className="flex flex-col gap-3 min-w-[200px]">
          {/* 真理値表 */}
          {dataset && (
            <div className="flex flex-col gap-1">
              <span className="text-xs font-medium text-zinc-500">真理値表 (truth table)</span>
              <table className="text-xs border-collapse">
                <thead>
                  <tr className="border-b border-zinc-300 dark:border-zinc-600">
                    <th className="px-2 py-1 text-left">x₁</th>
                    <th className="px-2 py-1 text-left">x₂</th>
                    <th className="px-2 py-1 text-left">目標 t</th>
                    <th className="px-2 py-1 text-left">出力 y</th>
                    <th className="px-2 py-1 text-left">損失 L</th>
                  </tr>
                </thead>
                <tbody>
                  {dataset.samples.map((sample, i) => {
                    const output = dataPointOutputs[i] ?? 0;
                    const target = sample.target[0] ?? 0;
                    const sampleLoss = computeSampleLoss(lossName, output, target);
                    const isCorrect = (output > 0.5) === (target > 0.5);
                    return (
                      <tr
                        key={i}
                        className={`border-b border-zinc-200 dark:border-zinc-700 ${(isCorrect ? "" : "bg-red-50 dark:bg-red-900/20") satisfies string}`}
                      >
                        <td className="px-2 py-1">{String(sample.input[0] ?? 0)}</td>
                        <td className="px-2 py-1">{String(sample.input[1] ?? 0)}</td>
                        <td className="px-2 py-1 font-medium">{String(target)}</td>
                        <td className="px-2 py-1">{output.toFixed(4)}</td>
                        <td className={`px-2 py-1 ${(sampleLoss.value > 0.5 ? "text-red-500" : sampleLoss.value > 0.1 ? "text-amber-500" : "text-green-500") satisfies string}`}>
                          {sampleLoss.value.toFixed(4)}
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
              {/* 損失関数の計算式展開 */}
              <div className="mt-1 flex flex-col gap-0.5">
                <span className="text-[10px] font-medium text-zinc-400">各サンプルの損失計算</span>
                {dataset.samples.map((sample, i) => {
                  const output = dataPointOutputs[i] ?? 0;
                  const target = sample.target[0] ?? 0;
                  const sampleLoss = computeSampleLoss(lossName, output, target);
                  return (
                    <div key={i} className="text-[10px] font-mono text-zinc-500">
                      <span className="text-zinc-400">
                        {`(${String(sample.input[0] ?? 0) satisfies string},${String(sample.input[1] ?? 0) satisfies string})`}
                      </span>
                      {" L = "}
                      {sampleLoss.formula}
                      {` = ${sampleLoss.value.toFixed(4) satisfies string}`}
                    </div>
                  );
                })}
              </div>
            </div>
          )}

          {/* ネットワークパラメータ表示 (1層の場合) */}
          {currentBoundary && (
            <div className="flex flex-col gap-1">
              <span className="text-xs font-medium text-zinc-500">パラメータ</span>
              <div className="text-xs font-mono bg-zinc-100 dark:bg-zinc-800 rounded p-2">
                <div>w₁ = {currentBoundary.w1.toFixed(4)}</div>
                <div>w₂ = {currentBoundary.w2.toFixed(4)}</div>
                <div>b = {currentBoundary.b.toFixed(4)}</div>
                <div className="mt-1 text-zinc-500">
                  決定境界: {currentBoundary.w1.toFixed(2)}x₁ + {currentBoundary.w2.toFixed(2)}x₂ + {currentBoundary.b.toFixed(2)} = 0
                </div>
              </div>
            </div>
          )}

          {/* 凡例 */}
          <div className="flex flex-col gap-1">
            <span className="text-xs font-medium text-zinc-500">凡例</span>
            <div className="flex flex-col gap-1 text-xs">
              <div className="flex items-center gap-2">
                <div className="w-4 h-4 rounded-full bg-blue-600" />
                <span>ターゲット = 1</span>
              </div>
              <div className="flex items-center gap-2">
                <div className="w-4 h-4 rounded-full bg-red-600" />
                <span>ターゲット = 0</span>
              </div>
              {currentBoundary && (
                <div className="flex items-center gap-2">
                  <div className="w-4 h-0.5 bg-zinc-800 dark:bg-zinc-200" />
                  <span>決定境界 (Wx+b=0)</span>
                </div>
              )}
              {showHeatmap && (
                <div className="flex items-center gap-1 mt-1">
                  <div className="flex h-3">
                    <div className="w-4 h-full" style={{ background: "rgb(255,100,100)" }} />
                    <div className="w-4 h-full" style={{ background: "rgb(255,255,255)" }} />
                    <div className="w-4 h-full" style={{ background: "rgb(100,100,255)" }} />
                  </div>
                  <span className="ml-1">出力 0 → 0.5 → 1</span>
                </div>
              )}
            </div>
          </div>

          {/* 多層ネットワークの注意書き */}
          {!currentBoundary && networkState.def.layers.length > 1 && (
            <div className="text-xs text-zinc-500 italic">
              多層ネットワークのため直線の決定境界は表示されません。
              ヒートマップで出力分布を確認してください。
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
