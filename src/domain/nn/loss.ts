import { Data } from "effect";
import type { Vector } from "../math/vector";
import * as V from "../math/vector";

/** 損失関数 (loss function): スカラー損失 (scalar loss) と予測値に対する勾配 (gradient) を計算する */
export class LossFunction extends Data.TaggedClass("LossFunction")<{
  readonly name: string;
  readonly description: string;
  readonly loss: (predicted: Vector, target: Vector) => number;
  readonly gradient: (predicted: Vector, target: Vector) => Vector;
}> {}

/** 平均二乗誤差 (Mean Squared Error) */
export const mse: LossFunction = new LossFunction({
  name: "mse",
  description: "平均二乗誤差 (MSE): 回帰タスクの標準的な損失関数。予測値と正解の差の二乗の平均",
  loss: (predicted, target) => {
    const diff = V.sub(predicted, target);
    return V.dot(diff, diff) / predicted.length;
  },
  gradient: (predicted, target) =>
    V.scale(V.sub(predicted, target), 2 / predicted.length),
});

/**
 * 交差エントロピー損失 (cross-entropy loss)。ソフトマックス (softmax) 出力との組み合わせで使用する。
 * target はワンホット (one-hot) エンコーディングを前提とする。
 */
export const crossEntropy: LossFunction = new LossFunction({
  name: "crossEntropy",
  description: "交差エントロピー (Cross-Entropy): 分類タスクの標準的な損失関数。softmax 出力との組み合わせで使用",
  loss: (predicted, target) =>
    -V.dot(target, V.map(predicted, (p) => Math.log(Math.max(p, 1e-15)))),
  gradient: (predicted, target) =>
    V.sub(predicted, target),
});

/**
 * 二値交差エントロピー (Binary Cross-Entropy)。
 * シグモイド (sigmoid) 出力 + 二値分類で使用する。
 * L = -[t·log(y) + (1-t)·log(1-y)]
 *
 * 勾配 (gradient) は dL/dy = (y-t) / (y·(1-y))。
 * sigmoid の逆伝播 y·(1-y) と組み合わせると y-t に簡約される。
 */
export const binaryCrossEntropy: LossFunction = new LossFunction({
  name: "binaryCrossEntropy",
  description: "二値交差エントロピー (BCE): sigmoid 出力の二値分類向け損失関数",
  loss: (predicted, target) => {
    let sum = 0;
    for (let i = 0; i < predicted.length; i++) {
      const y = Math.max(1e-15, Math.min(1 - 1e-15, predicted.at(i)));
      const t = target.at(i);
      sum += -(t * Math.log(y) + (1 - t) * Math.log(1 - y));
    }
    return sum / predicted.length;
  },
  gradient: (predicted, target) => {
    const diff = V.sub(predicted, target);
    const denom = V.map(predicted, (y) => {
      const clamped = Math.max(1e-15, Math.min(1 - 1e-15, y));
      return clamped * (1 - clamped);
    });
    return V.scale(
      V.from(Array.from({ length: diff.length }, (_, i) => diff.at(i) / denom.at(i))),
      1 / predicted.length,
    );
  },
});

export const allLossFunctions = [mse, crossEntropy, binaryCrossEntropy] as const;
