import { Data } from "effect";
import * as V from "../math/vector";
import * as Rand from "../math/random";
import { LayerParams } from "./layer";
import type { LayerDef } from "./layer";
import { NetworkState } from "./network";
import type { NetworkDef } from "./network";
import type { Rng } from "../math/random";

export class InitStrategy extends Data.TaggedClass("InitStrategy")<{
  readonly name: string;
  readonly description: string;
  readonly getStd: (fanIn: number, fanOut: number) => number;
}> {}

/** すべて0で初期化する。学習の出発点を確認するためのベースライン */
export const zeroInit: InitStrategy = new InitStrategy({
  name: "zero",
  description: "ゼロ初期化: すべての重みを 0 にする（学習の対照実験用）",
  getStd: () => 0,
});

/** 小さな一様乱数で初期化する。最もシンプルなランダム初期化 */
export const smallRandom: InitStrategy = new InitStrategy({
  name: "smallRandom",
  description: "小さなランダム値: 標準偏差 0.01 の正規分布で初期化",
  getStd: () => 0.01,
});

/** Xavier/Glorot 初期化。sigmoid, tanh 向け */
export const xavier: InitStrategy = new InitStrategy({
  name: "xavier",
  description: "Xavier/Glorot: sigmoid/tanh 向け、入出力サイズに応じた標準偏差",
  getStd: Rand.xavierStd,
});

/** He 初期化。ReLU 向け */
export const he: InitStrategy = new InitStrategy({
  name: "he",
  description: "He: ReLU 向け、入力サイズに応じた標準偏差",
  getStd: (fanIn) => Rand.heStd(fanIn),
});

export const allInitStrategies = [zeroInit, smallRandom, xavier, he] as const;

export const initLayer = (
  def: LayerDef,
  strategy: InitStrategy,
  rng: Rng,
): readonly [LayerParams, Rng] => {
  const std = strategy.getStd(def.inputSize, def.outputSize);
  const [weights, rng1] = Rand.normalMatrix(rng, def.outputSize, def.inputSize, 0, std);
  const bias = V.zeros(def.outputSize);
  return [new LayerParams({ weights, bias }), rng1];
};

export const initNetwork = (
  def: NetworkDef,
  strategy: InitStrategy,
  seed: number,
): NetworkState => {
  let rng = Rand.fromSeed(seed);
  const params: LayerParams[] = [];
  for (const layerDef of def.layers) {
    const [layerParams, nextRng] = initLayer(layerDef, strategy, rng);
    params.push(layerParams);
    rng = nextRng;
  }
  return new NetworkState({ def, params });
};
