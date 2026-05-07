import { Data } from "effect";
import type { LayerDef, LayerParams } from "./layer";
import type { LossFunction } from "./loss";

/** ネットワーク (network) のアーキテクチャ (architecture) 定義 */
export class NetworkDef extends Data.TaggedClass("NetworkDef")<{
  readonly layers: ReadonlyArray<LayerDef>;
  readonly lossFunction: LossFunction;
}> {}

/** 重み (weights) を初期化済みのネットワーク (network) */
export class NetworkState extends Data.TaggedClass("NetworkState")<{
  readonly def: NetworkDef;
  readonly params: ReadonlyArray<LayerParams>;
}> {}

/** 学習 (training) の設定 (configuration) */
export class TrainingConfig extends Data.TaggedClass("TrainingConfig")<{
  readonly learningRate: number;
  readonly batchSize: number;
  readonly mode: "sgd" | "minibatch";
  /** 入力へのガウシアンノイズ (Gaussian noise) の標準偏差。0 でノイズなし。 */
  readonly inputNoiseStd: number;
}> {}

/** 1件の学習サンプル (training sample) */
export class TrainingSample extends Data.TaggedClass("TrainingSample")<{
  readonly input: import("../math/vector").Vector;
  readonly target: import("../math/vector").Vector;
}> {}

/** 学習進捗 (training progress) のスナップショット (snapshot) */
export class TrainingSnapshot extends Data.TaggedClass("TrainingSnapshot")<{
  readonly epoch: number;
  readonly step: number;
  readonly loss: number;
  readonly network: NetworkState;
}> {}

/** 永続化 (persistence) のために保存するモデル (model) 定義 */
export class ModelEntry extends Data.TaggedClass("ModelEntry")<{
  readonly id: string;
  readonly name: string;
  readonly createdAt: number;
  readonly networkDef: NetworkDef;
  readonly latestSnapshot: TrainingSnapshot | null;
}> {}
