import { Data } from "effect";
import { Matrix } from "../math/matrix";
import { Vector } from "../math/vector";
import type { Activation } from "./activation";

/** 単一の全結合層 (dense layer) の重み (weights) とバイアス (bias) */
export class LayerParams extends Data.TaggedClass("LayerParams")<{
  readonly weights: Matrix;
  readonly bias: Vector;
}> {}

/** レイヤー定義 (layer definition): アーキテクチャ (architecture) のみで重み (weights) は含まない */
export class LayerDef extends Data.TaggedClass("LayerDef")<{
  readonly inputSize: number;
  readonly outputSize: number;
  readonly activation: Activation;
}> {}

/** 順伝播 (forward pass) で保存した中間値 (intermediate values)。逆伝播 (backward pass) に必要 */
export class LayerCache extends Data.TaggedClass("LayerCache")<{
  readonly input: Vector;
  readonly preActivation: Vector;
  readonly output: Vector;
}> {}

/** 単一レイヤーの勾配 (gradients) */
export class LayerGradients extends Data.TaggedClass("LayerGradients")<{
  readonly dWeights: Matrix;
  readonly dBias: Vector;
}> {}
