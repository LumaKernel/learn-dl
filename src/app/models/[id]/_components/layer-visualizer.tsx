"use client";

import { memo, useEffect, useMemo, useRef } from "react";
import type { LayerParams } from "@/domain/nn/layer";
import type { SerializedLayerDef } from "../../actions";

type Props = {
  readonly layers: ReadonlyArray<SerializedLayerDef>;
  readonly params: ReadonlyArray<LayerParams>;
  readonly layerOutputs: ReadonlyArray<ReadonlyArray<number>>;
  readonly inputPixels: ReadonlyArray<number>;
};

/**
 * 値を [-1, 1] の範囲の色に変換するヘルパー。
 * 正の値: 青、負の値: 赤、0: 白
 */
const valueToColor = (v: number, maxAbs: number): string => {
  if (maxAbs === 0) return "rgb(255, 255, 255)";
  const normalized = Math.max(-1, Math.min(1, v / maxAbs));
  if (normalized >= 0) {
    const intensity = Math.round(255 * (1 - normalized));
    return `rgb(${String(intensity) satisfies string}, ${String(intensity) satisfies string}, 255)`;
  }
  const intensity = Math.round(255 * (1 + normalized));
  return `rgb(255, ${String(intensity) satisfies string}, ${String(intensity) satisfies string})`;
};

/**
 * 値を [0, 1] の範囲のグレースケールに変換するヘルパー。
 * 活性化出力 (activation output) の可視化に使う。
 */
const activationToColor = (v: number, maxVal: number): string => {
  if (maxVal === 0) return "rgb(255, 255, 255)";
  const normalized = Math.max(0, Math.min(1, v / maxVal));
  const intensity = Math.round(255 * (1 - normalized));
  return `rgb(${String(intensity) satisfies string}, ${String(intensity) satisfies string}, ${String(intensity) satisfies string})`;
};

function WeightHeatmap({
  layerParams,
  maxSize,
}: {
  readonly layerParams: LayerParams;
  readonly maxSize: number;
}) {
  const { weights, bias } = layerParams;
  const allWeightValues = weights.data.flatMap((row) => [...row.data]);
  const maxAbs = allWeightValues.reduce((m, v) => Math.max(m, Math.abs(v)), 0);
  const weightMean = allWeightValues.reduce((a, b) => a + b, 0) / Math.max(allWeightValues.length, 1);

  const displayRows = Math.min(weights.rows, maxSize);
  const displayCols = Math.min(weights.cols, maxSize);
  const cellSize = Math.max(2, Math.min(8, Math.floor(200 / Math.max(displayRows, displayCols))));

  const biasMaxAbs = bias.data.reduce((m, v) => Math.max(m, Math.abs(v)), 0);

  return (
    <div className="flex flex-col gap-1">
      <svg
        width={displayCols * cellSize}
        height={displayRows * cellSize}
        className="border border-zinc-200 dark:border-zinc-600"
      >
        {Array.from({ length: displayRows }, (_, i) =>
          Array.from({ length: displayCols }, (_, j) => (
            <rect
              key={`${String(i) satisfies string}-${String(j) satisfies string}`}
              x={j * cellSize}
              y={i * cellSize}
              width={cellSize}
              height={cellSize}
              fill={valueToColor(weights.at(i, j), maxAbs)}
            />
          )),
        )}
      </svg>
      <span className="text-[10px] text-zinc-400">
        {`max|w|=${maxAbs.toFixed(4) satisfies string} mean=${weightMean.toFixed(4) satisfies string}`}
      </span>
      {/* バイアス (bias) の可視化 */}
      <div className="flex gap-px items-center">
        <span className="text-[10px] text-zinc-400 mr-1">bias:</span>
        {bias.data.slice(0, 32).map((v, i) => (
          <div
            key={i}
            className="rounded-sm"
            style={{
              width: `${String(Math.max(4, Math.min(10, Math.floor(120 / bias.length)))) satisfies string}px`,
              height: "12px",
              backgroundColor: valueToColor(v, biasMaxAbs),
              border: "1px solid #e5e5e5",
            }}
            title={`bias[${String(i) satisfies string}]=${v.toFixed(4) satisfies string}`}
          />
        ))}
        {bias.length > 32 && (
          <span className="text-[10px] text-zinc-400 ml-0.5">+{String(bias.length - 32)}</span>
        )}
      </div>
    </div>
  );
}

/**
 * 活性化値に応じた枠色を返す。
 * 強い反応: 緑、弱い/ゼロ: グレー
 */
const activationToBorderColor = (value: number, maxVal: number): string => {
  if (maxVal === 0) return "rgb(200, 200, 200)";
  const ratio = Math.min(1, value / maxVal);
  const r = Math.round(200 + (34 - 200) * ratio);
  const g = Math.round(200 + (197 - 200) * ratio);
  const b = Math.round(200 + (94 - 200) * ratio);
  return `rgb(${String(r) satisfies string}, ${String(g) satisfies string}, ${String(b) satisfies string})`;
};

/**
 * 重みベクトルを RGBA の ImageData に変換するヘルパー。
 * 正: 青、負: 赤、0: 白
 */
const weightsToImageData = (
  weightsArr: ReadonlyArray<number>,
  width: number,
  height: number,
  maxAbs: number,
): ImageData => {
  const data = new Uint8ClampedArray(width * height * 4);
  for (let i = 0; i < width * height; i++) {
    const v = weightsArr[i] ?? 0;
    const normalized = maxAbs === 0 ? 0 : Math.max(-1, Math.min(1, v / maxAbs));
    let r: number, g: number, b: number;
    if (normalized >= 0) {
      const intensity = Math.round(255 * (1 - normalized));
      r = intensity;
      g = intensity;
      b = 255;
    } else {
      const intensity = Math.round(255 * (1 + normalized));
      r = 255;
      g = intensity;
      b = intensity;
    }
    data[i * 4] = r;
    data[i * 4 + 1] = g;
    data[i * 4 + 2] = b;
    data[i * 4 + 3] = 255;
  }
  return new ImageData(data, width, height);
};

/**
 * 個別ニューロンの Canvas 描画コンポーネント。
 * memo化して重みが変わらない限り再描画しない。
 */
const NeuronCanvas = memo(function NeuronCanvas({
  weightsArr,
  inputWidth,
  inputHeight,
  globalMaxAbs,
  inputPixels,
  activation,
  maxActivation,
  neuronIdx,
  imageSize,
}: {
  readonly weightsArr: ReadonlyArray<number>;
  readonly inputWidth: number;
  readonly inputHeight: number;
  readonly globalMaxAbs: number;
  readonly inputPixels: ReadonlyArray<number>;
  readonly activation: number;
  readonly maxActivation: number;
  readonly neuronIdx: number;
  readonly imageSize: number;
}) {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  // 重み画像データ (重みが変わったときのみ再計算)
  const weightImageData = useMemo(
    () => weightsToImageData(weightsArr, inputWidth, inputHeight, globalMaxAbs),
    [weightsArr, inputWidth, inputHeight, globalMaxAbs],
  );

  // Canvas 描画
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    // 重み画像を描画
    ctx.putImageData(weightImageData, 0, 0);

    // 入力ピクセルオーバーレイ (うっすら半透明)
    ctx.fillStyle = "rgba(0, 0, 0, 0.3)";
    for (let i = 0; i < inputPixels.length; i++) {
      if (inputPixels[i] === 0) continue;
      const x = i % inputWidth;
      const y = Math.floor(i / inputWidth);
      ctx.fillRect(x, y, 1, 1);
    }
  }, [weightImageData, inputPixels, inputWidth]);

  const borderColor = activationToBorderColor(activation, maxActivation);

  return (
    <div
      className="relative"
      style={{
        border: `2px solid ${borderColor satisfies string}`,
        borderRadius: "2px",
      }}
      title={`ニューロン ${String(neuronIdx) satisfies string}: 出力=${activation.toFixed(3) satisfies string}`}
    >
      <canvas
        ref={canvasRef}
        width={inputWidth}
        height={inputHeight}
        style={{ width: `${String(imageSize) satisfies string}px`, height: `${String(imageSize) satisfies string}px`, imageRendering: "pixelated" }}
      />
      {activation > 0 && (
        <span className="absolute bottom-0 right-0 text-[8px] px-0.5 bg-black/60 text-white rounded-tl">
          {activation.toFixed(1) satisfies string}
        </span>
      )}
    </div>
  );
});

/**
 * 1層目のニューロンの重みを画像として個別可視化する (Canvas ベース)。
 */
function NeuronWeightImages({
  layerParams,
  inputWidth,
  activations,
  inputPixels,
}: {
  readonly layerParams: LayerParams;
  readonly inputWidth: number;
  readonly activations: ReadonlyArray<number> | undefined;
  readonly inputPixels: ReadonlyArray<number>;
}) {
  const { weights } = layerParams;
  const neuronCount = weights.rows;
  const inputHeight = Math.floor(weights.cols / inputWidth);
  const imageSize = 48;

  const globalMaxAbs = useMemo(
    () => weights.data.reduce(
      (m, row) => Math.max(m, row.data.reduce((rm, v) => Math.max(rm, Math.abs(v)), 0)),
      0,
    ),
    [weights],
  );

  const maxActivation = activations
    ? activations.reduce((m, v) => Math.max(m, v), 0)
    : 0;

  const hasInput = inputPixels.some((p) => p > 0);

  return (
    <div className="flex flex-col gap-2">
      <span className="text-[10px] text-zinc-400">
        {`ニューロン重み画像 (${String(neuronCount) satisfies string}個, ${String(inputWidth) satisfies string}x${String(inputHeight) satisfies string})`}
        {hasInput && " — 枠色: 反応の強さ, 薄黒: 入力"}
      </span>
      <div className="flex flex-wrap gap-1">
        {weights.data.map((row, neuronIdx) => (
          <NeuronCanvas
            key={neuronIdx}
            weightsArr={row.data}
            inputWidth={inputWidth}
            inputHeight={inputHeight}
            globalMaxAbs={globalMaxAbs}
            inputPixels={inputPixels}
            activation={activations?.[neuronIdx] ?? 0}
            maxActivation={maxActivation}
            neuronIdx={neuronIdx}
            imageSize={imageSize}
          />
        ))}
      </div>
    </div>
  );
}

function ActivationBar({
  values,
}: {
  readonly values: ReadonlyArray<number>;
}) {
  if (values.length === 0) return null;
  const maxVal = values.reduce((m, v) => Math.max(m, Math.abs(v)), 0);
  const displayCount = Math.min(values.length, 64);
  const barWidth = Math.max(3, Math.min(12, Math.floor(200 / displayCount)));

  return (
    <div className="flex gap-px items-end" style={{ height: "48px" }}>
      {values.slice(0, displayCount).map((v, i) => (
        <div
          key={i}
          className="rounded-t"
          style={{
            width: `${String(barWidth) satisfies string}px`,
            height: `${String(Math.max(1, (Math.abs(v) / Math.max(maxVal, 1e-10)) * 48)) satisfies string}px`,
            backgroundColor: activationToColor(Math.abs(v), maxVal),
            border: "1px solid #ccc",
          }}
          title={`[${String(i) satisfies string}] = ${v.toFixed(4) satisfies string}`}
        />
      ))}
      {values.length > displayCount && (
        <span className="text-[10px] text-zinc-400 self-center ml-1">
          +{String(values.length - displayCount)}
        </span>
      )}
    </div>
  );
}

export function LayerVisualizer({ layers, params, layerOutputs, inputPixels }: Props) {
  // 1層目の入力幅を計算（28x28=784 → 幅28, 64x64=4096 → 幅64）
  const firstLayerInputSize = layers[0]?.inputSize ?? 784;
  const inputWidth = Math.round(Math.sqrt(firstLayerInputSize));

  return (
    <div className="flex flex-col gap-4">
      <h3 className="font-medium text-sm text-zinc-600 dark:text-zinc-400">
        レイヤー可視化 (layer visualization)
      </h3>

      {layers.map((layerDef, idx) => {
        const layerParams = params[idx];
        const output = layerOutputs[idx];

        return (
          <div
            key={idx}
            className="border border-zinc-200 dark:border-zinc-700 rounded-lg p-3 flex flex-col gap-2"
          >
            <div className="flex items-center justify-between">
              <span className="text-sm font-medium">
                層 {String(idx + 1)}: {layerDef.activationName}
              </span>
              <span className="text-xs text-zinc-500">
                {`${String(layerDef.inputSize) satisfies string} → ${String(layerDef.outputSize) satisfies string}`}
              </span>
            </div>

            <div className="flex gap-4 flex-wrap">
              {/* 重みヒートマップ (weight heatmap) */}
              {layerParams && (
                <div className="flex flex-col gap-1">
                  <span className="text-[10px] text-zinc-400">重み (weights)</span>
                  <WeightHeatmap layerParams={layerParams} maxSize={50} />
                  <span className="text-[10px] text-zinc-400">
                    {`${String(layerParams.weights.rows) satisfies string}x${String(layerParams.weights.cols) satisfies string}`}
                  </span>
                </div>
              )}

              {/* 活性化出力 (activation output) */}
              {output && output.length > 0 && (
                <div className="flex flex-col gap-1">
                  <span className="text-[10px] text-zinc-400">出力 (output)</span>
                  <ActivationBar values={output} />
                </div>
              )}
            </div>

            {/* 1層目: ニューロンの重みを画像として可視化 */}
            {idx === 0 && layerParams && (
              <NeuronWeightImages
                layerParams={layerParams}
                inputWidth={inputWidth}
                activations={output}
                inputPixels={inputPixels}
              />
            )}
          </div>
        );
      })}
    </div>
  );
}
