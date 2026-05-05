/**
 * Effect TaggedClass ⇔ シリアライズ可能な (serializable) プレーンオブジェクトの変換。
 * サーバーアクション (server action) やローカルストレージ (localStorage) でやり取りするために使う。
 */
import * as V from "../math/vector";
import * as Mat from "../math/matrix";
import { LayerDef, LayerParams } from "./layer";
import { NetworkDef, NetworkState } from "./network";
import { allActivations } from "./activation";
import type { Activation } from "./activation";
import { allLossFunctions } from "./loss";
import type { LossFunction } from "./loss";

export type SerializedLayerDef = {
  readonly inputSize: number;
  readonly outputSize: number;
  readonly activationName: string;
};

export type SerializedLayerParams = {
  readonly weights: ReadonlyArray<ReadonlyArray<number>>;
  readonly bias: ReadonlyArray<number>;
};

export type SerializedNetworkDef = {
  readonly layers: ReadonlyArray<SerializedLayerDef>;
  readonly lossName: string;
};

export type SerializedNetworkState = {
  readonly def: SerializedNetworkDef;
  readonly params: ReadonlyArray<SerializedLayerParams>;
};

const findActivation = (name: string): Activation => {
  const found = allActivations.find((a) => a.name === name);
  if (!found) throw new Error(`不明な活性化関数 (unknown activation): ${name satisfies string}`);
  return found;
};

const findLoss = (name: string): LossFunction => {
  const found = allLossFunctions.find((l) => l.name === name);
  if (!found) throw new Error(`不明な損失関数 (unknown loss): ${name satisfies string}`);
  return found;
};

export const serializeLayerDef = (def: LayerDef): SerializedLayerDef => ({
  inputSize: def.inputSize,
  outputSize: def.outputSize,
  activationName: def.activation.name,
});

export const deserializeLayerDef = (s: SerializedLayerDef): LayerDef =>
  new LayerDef({
    inputSize: s.inputSize,
    outputSize: s.outputSize,
    activation: findActivation(s.activationName),
  });

export const serializeLayerParams = (params: LayerParams): SerializedLayerParams => ({
  weights: params.weights.data.map((row) => [...row.data]),
  bias: [...params.bias.data],
});

export const deserializeLayerParams = (s: SerializedLayerParams): LayerParams =>
  new LayerParams({
    weights: Mat.fromArrays(s.weights),
    bias: V.from(s.bias),
  });

export const serializeNetworkDef = (def: NetworkDef): SerializedNetworkDef => ({
  layers: def.layers.map(serializeLayerDef),
  lossName: def.lossFunction.name,
});

export const deserializeNetworkDef = (s: SerializedNetworkDef): NetworkDef =>
  new NetworkDef({
    layers: s.layers.map(deserializeLayerDef),
    lossFunction: findLoss(s.lossName),
  });

export const serializeNetworkState = (state: NetworkState): SerializedNetworkState => ({
  def: serializeNetworkDef(state.def),
  params: state.params.map(serializeLayerParams),
});

export const deserializeNetworkState = (s: SerializedNetworkState): NetworkState =>
  new NetworkState({
    def: deserializeNetworkDef(s.def),
    params: s.params.map(deserializeLayerParams),
  });
