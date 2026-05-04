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

export const xavier: InitStrategy = new InitStrategy({
  name: "xavier",
  description: "Xavier/Glorot: good for sigmoid/tanh",
  getStd: Rand.xavierStd,
});

export const he: InitStrategy = new InitStrategy({
  name: "he",
  description: "He: good for ReLU",
  getStd: (fanIn) => Rand.heStd(fanIn),
});

export const allInitStrategies = [xavier, he] as const;

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
