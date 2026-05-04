import { Data } from "effect";
import type { Vector } from "../math/vector";
import * as V from "../math/vector";

/** Loss function: computes scalar loss and gradient w.r.t. predicted */
export class LossFunction extends Data.TaggedClass("LossFunction")<{
  readonly name: string;
  readonly loss: (predicted: Vector, target: Vector) => number;
  readonly gradient: (predicted: Vector, target: Vector) => Vector;
}> {}

/** Mean Squared Error */
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
 * Cross-entropy loss (for use with softmax output).
 * target is one-hot encoded.
 */
export const crossEntropy: LossFunction = new LossFunction({
  name: "crossEntropy",
  loss: (predicted, target) =>
    -V.dot(target, V.map(predicted, (p) => Math.log(Math.max(p, 1e-15)))),
  gradient: (predicted, target) =>
    V.sub(predicted, target),
});

export const allLossFunctions = [mse, crossEntropy] as const;
