import { Data } from "effect";
import type { Vector } from "../math/vector";
import * as V from "../math/vector";

/** 損失関数 (loss function): スカラー損失 (scalar loss) と予測値に対する勾配 (gradient) を計算する */
export class LossFunction extends Data.TaggedClass("LossFunction")<{
  readonly name: string;
  readonly loss: (predicted: Vector, target: Vector) => number;
  readonly gradient: (predicted: Vector, target: Vector) => Vector;
}> {}

/** 平均二乗誤差 (Mean Squared Error) */
export const mse: LossFunction = new LossFunction({
  name: "mse",
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
  loss: (predicted, target) =>
    -V.dot(target, V.map(predicted, (p) => Math.log(Math.max(p, 1e-15)))),
  gradient: (predicted, target) =>
    V.sub(predicted, target),
});

export const allLossFunctions = [mse, crossEntropy] as const;
