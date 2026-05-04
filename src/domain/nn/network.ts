import { Data } from "effect";
import type { LayerDef, LayerParams } from "./layer";
import type { LossFunction } from "./loss";

/** Network architecture definition */
export class NetworkDef extends Data.TaggedClass("NetworkDef")<{
  readonly layers: ReadonlyArray<LayerDef>;
  readonly lossFunction: LossFunction;
}> {}

/** Network with initialized weights */
export class NetworkState extends Data.TaggedClass("NetworkState")<{
  readonly def: NetworkDef;
  readonly params: ReadonlyArray<LayerParams>;
}> {}

/** Training configuration */
export class TrainingConfig extends Data.TaggedClass("TrainingConfig")<{
  readonly learningRate: number;
  readonly batchSize: number;
  readonly mode: "sgd" | "minibatch";
}> {}

/** A single training sample */
export class TrainingSample extends Data.TaggedClass("TrainingSample")<{
  readonly input: import("../math/vector").Vector;
  readonly target: import("../math/vector").Vector;
}> {}

/** Snapshot of training progress */
export class TrainingSnapshot extends Data.TaggedClass("TrainingSnapshot")<{
  readonly epoch: number;
  readonly step: number;
  readonly loss: number;
  readonly network: NetworkState;
}> {}

/** Saved model definition for persistence */
export class ModelEntry extends Data.TaggedClass("ModelEntry")<{
  readonly id: string;
  readonly name: string;
  readonly createdAt: number;
  readonly networkDef: NetworkDef;
  readonly latestSnapshot: TrainingSnapshot | null;
}> {}
