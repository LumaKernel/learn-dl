import { Data } from "effect";
import type { Digit } from "../dataset";
import type { GridSize } from "../pixel-grid";

/**
 * 入力サイズの選択肢。タスクごとに利用可能な入力サイズが決まる。
 */
export class InputOption extends Data.TaggedClass("InputOption")<{
  readonly size: number;
  readonly gridSize: GridSize;
  readonly label: string;
}> {}

/**
 * タスク (問題設定) の定義。
 * ネットワーク構成の制約 (出力サイズ・活性化・損失関数) と、
 * 学習時のターゲットベクトル生成方法を決定する。
 */
export class TaskDef extends Data.TaggedClass("TaskDef")<{
  readonly name: string;
  readonly description: string;
  readonly inputOptions: ReadonlyArray<InputOption>;
  readonly outputSize: number;
  readonly outputActivationName: string;
  readonly defaultLossName: string;
  /**
   * データセットの digit からターゲットベクトルを生成する。
   * 例: 数字分類なら one-hot、偶奇分類なら [1,0]/[0,1]
   */
  readonly buildTarget: (digit: Digit) => ReadonlyArray<number>;
}> {}

/** 手書き数字分類 (0-9) */
const digitClassification = new TaskDef({
  name: "digitClassification",
  description: "手書き数字分類 (0-9): 10クラス分類",
  inputOptions: [
    new InputOption({ size: 784, gridSize: 28, label: "784 (28×28)" }),
    new InputOption({ size: 4096, gridSize: 64, label: "4096 (64×64)" }),
  ],
  outputSize: 10,
  outputActivationName: "softmax",
  defaultLossName: "crossEntropy",
  buildTarget: (digit) =>
    Array.from({ length: 10 }, (_, i) => (i === digit ? 1 : 0)),
});

/** 偶数/奇数の二値分類 */
const parityClassification = new TaskDef({
  name: "parityClassification",
  description: "偶数/奇数分類: 2クラス分類",
  inputOptions: [
    new InputOption({ size: 784, gridSize: 28, label: "784 (28×28)" }),
    new InputOption({ size: 4096, gridSize: 64, label: "4096 (64×64)" }),
  ],
  outputSize: 2,
  outputActivationName: "softmax",
  defaultLossName: "crossEntropy",
  buildTarget: (digit) =>
    digit % 2 === 0 ? [1, 0] : [0, 1],
});

/** 数字の値を回帰で予測 */
const digitRegression = new TaskDef({
  name: "digitRegression",
  description: "数字の値を回帰予測: 0〜9 を [0, 1] に正規化",
  inputOptions: [
    new InputOption({ size: 784, gridSize: 28, label: "784 (28×28)" }),
    new InputOption({ size: 4096, gridSize: 64, label: "4096 (64×64)" }),
  ],
  outputSize: 1,
  outputActivationName: "sigmoid",
  defaultLossName: "mse",
  buildTarget: (digit) => [digit / 9],
});

export const allTasks: ReadonlyArray<TaskDef> = [
  digitClassification,
  parityClassification,
  digitRegression,
] as const;

export const findTask = (name: string): TaskDef => {
  const found = allTasks.find((t) => t.name === name);
  if (!found) throw new Error(`不明なタスク (unknown task): ${name satisfies string}`);
  return found;
};
