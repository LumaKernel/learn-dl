import { Data } from "effect";
import type { Digit } from "../dataset";
import type { GridSize } from "../pixel-grid";

/**
 * 組み込みデータセットのサンプル (builtin dataset sample)。
 * 論理ゲート等、自動生成可能なタスクで使用する。
 */
export type BuiltinSample = {
  readonly input: ReadonlyArray<number>;
  readonly target: ReadonlyArray<number>;
};

/**
 * 組み込みデータセット定義。
 * 1つのタスクに複数のデータセット（例: OR, AND, XOR）を持てる。
 */
export type BuiltinDataset = {
  readonly name: string;
  readonly label: string;
  readonly samples: ReadonlyArray<BuiltinSample>;
};

/**
 * 入力サイズの選択肢。タスクごとに利用可能な入力サイズが決まる。
 * gridSize が null の場合はピクセルグリッドを使わない（例: 論理ゲート）。
 */
export class InputOption extends Data.TaggedClass("InputOption")<{
  readonly size: number;
  readonly gridSize: GridSize | null;
  readonly label: string;
}> {}

/**
 * タスク (問題設定) の定義。
 * ネットワーク構成の制約 (出力サイズ・活性化・損失関数) と、
 * 学習時のターゲットベクトル生成方法を決定する。
 *
 * datasetType:
 * - "handwriting": ファイルベースの手書きデータセットを使用
 * - "builtin": 組み込みデータセットを使用（自動生成）
 */
export class TaskDef extends Data.TaggedClass("TaskDef")<{
  readonly name: string;
  readonly description: string;
  readonly inputOptions: ReadonlyArray<InputOption>;
  readonly outputSize: number;
  readonly outputActivationName: string;
  readonly defaultLossName: string;
  readonly datasetType: "handwriting" | "builtin";
  /**
   * データセットの digit からターゲットベクトルを生成する。
   * handwriting タスクで使用。builtin タスクでは null。
   */
  readonly buildTarget: ((digit: Digit) => ReadonlyArray<number>) | null;
  /**
   * 組み込みデータセット一覧。builtin タスクで使用。
   * handwriting タスクでは null。
   */
  readonly builtinDatasets: ReadonlyArray<BuiltinDataset> | null;
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
  datasetType: "handwriting",
  buildTarget: (digit) =>
    Array.from({ length: 10 }, (_, i) => (i === digit ? 1 : 0)),
  builtinDatasets: null,
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
  datasetType: "handwriting",
  buildTarget: (digit) =>
    digit % 2 === 0 ? [1, 0] : [0, 1],
  builtinDatasets: null,
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
  datasetType: "handwriting",
  buildTarget: (digit) => [digit / 9],
  builtinDatasets: null,
});

/** 2入力論理ゲート学習 (binary logic gate) */
const binaryLogicGate = new TaskDef({
  name: "binaryLogicGate",
  description: "2オペランド論理ゲート学習 (OR, AND, XOR)",
  inputOptions: [
    new InputOption({ size: 2, gridSize: null, label: "2 (2ビット入力)" }),
  ],
  outputSize: 1,
  outputActivationName: "sigmoid",
  defaultLossName: "mse",
  datasetType: "builtin",
  buildTarget: null,
  builtinDatasets: [
    {
      name: "or",
      label: "OR ゲート",
      samples: [
        { input: [0, 0], target: [0] },
        { input: [0, 1], target: [1] },
        { input: [1, 0], target: [1] },
        { input: [1, 1], target: [1] },
      ],
    },
    {
      name: "and",
      label: "AND ゲート",
      samples: [
        { input: [0, 0], target: [0] },
        { input: [0, 1], target: [0] },
        { input: [1, 0], target: [0] },
        { input: [1, 1], target: [1] },
      ],
    },
    {
      name: "xor",
      label: "XOR ゲート",
      samples: [
        { input: [0, 0], target: [0] },
        { input: [0, 1], target: [1] },
        { input: [1, 0], target: [1] },
        { input: [1, 1], target: [0] },
      ],
    },
    {
      name: "nand",
      label: "NAND ゲート",
      samples: [
        { input: [0, 0], target: [1] },
        { input: [0, 1], target: [1] },
        { input: [1, 0], target: [1] },
        { input: [1, 1], target: [0] },
      ],
    },
    {
      name: "nor",
      label: "NOR ゲート",
      samples: [
        { input: [0, 0], target: [1] },
        { input: [0, 1], target: [0] },
        { input: [1, 0], target: [0] },
        { input: [1, 1], target: [0] },
      ],
    },
    {
      name: "xnor",
      label: "XNOR ゲート",
      samples: [
        { input: [0, 0], target: [1] },
        { input: [0, 1], target: [0] },
        { input: [1, 0], target: [0] },
        { input: [1, 1], target: [1] },
      ],
    },
  ],
});

export const allTasks: ReadonlyArray<TaskDef> = [
  digitClassification,
  parityClassification,
  digitRegression,
  binaryLogicGate,
] as const;

export const findTask = (name: string): TaskDef => {
  const found = allTasks.find((t) => t.name === name);
  if (!found) throw new Error(`不明なタスク (unknown task): ${name satisfies string}`);
  return found;
};
