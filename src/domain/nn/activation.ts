import { Data } from "effect";
import { Vector } from "../math/vector";
import * as V from "../math/vector";

/** Activation function definition: forward and its derivative */
export class Activation extends Data.TaggedClass("Activation")<{
  readonly name: string;
  readonly forward: (input: Vector) => Vector;
  readonly backward: (input: Vector, output: Vector) => Vector;
}> {}

export const sigmoid: Activation = new Activation({
  name: "sigmoid",
  forward: (input) =>
    V.map(input, (x) => 1 / (1 + Math.exp(-x))),
  backward: (_input, output) => {
    void _input;
    return V.map(output, (o) => o * (1 - o));
  },
});

export const relu: Activation = new Activation({
  name: "relu",
  forward: (input) =>
    V.map(input, (x) => Math.max(0, x)),
  backward: (input, _output) => {
    void _output;
    return V.map(input, (x) => (x > 0 ? 1 : 0));
  },
});

export const identity: Activation = new Activation({
  name: "identity",
  forward: (input) => input,
  backward: (input) =>
    V.from(Array.from({ length: input.length }, () => 1)),
});

/**
 * Softmax: applied to entire vector.
 * Backward returns 1s because softmax gradient is handled
 * together with cross-entropy loss in the training step.
 */
export const softmax: Activation = new Activation({
  name: "softmax",
  forward: (input) => {
    const m = V.max(input);
    const exps = V.map(input, (x) => Math.exp(x - m));
    const s = V.sum(exps);
    return V.map(exps, (x) => x / s);
  },
  backward: (_input, output) => output,
});

export const allActivations = [sigmoid, relu, identity, softmax] as const;
