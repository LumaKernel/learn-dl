import { Data } from "effect";
import { Matrix } from "../math/matrix";
import { Vector } from "../math/vector";
import type { Activation } from "./activation";

/** Weights and bias for a single dense layer */
export class LayerParams extends Data.TaggedClass("LayerParams")<{
  readonly weights: Matrix;
  readonly bias: Vector;
}> {}

/** Layer definition (architecture, not weights) */
export class LayerDef extends Data.TaggedClass("LayerDef")<{
  readonly inputSize: number;
  readonly outputSize: number;
  readonly activation: Activation;
}> {}

/** Cached intermediate values from forward pass, needed for backward */
export class LayerCache extends Data.TaggedClass("LayerCache")<{
  readonly input: Vector;
  readonly preActivation: Vector;
  readonly output: Vector;
}> {}

/** Gradients for a single layer */
export class LayerGradients extends Data.TaggedClass("LayerGradients")<{
  readonly dWeights: Matrix;
  readonly dBias: Vector;
}> {}
