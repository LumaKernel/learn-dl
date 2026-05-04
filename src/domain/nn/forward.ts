import { Vector } from "../math/vector";
import * as V from "../math/vector";
import * as Mat from "../math/matrix";
import { LayerCache } from "./layer";
import type { LayerParams, LayerDef } from "./layer";
import type { NetworkState } from "./network";

/** 単一レイヤーの順伝播 (forward pass) */
export const forwardLayer = (
  def: LayerDef,
  params: LayerParams,
  input: Vector,
): LayerCache => {
  const preActivation = V.add(Mat.mulVec(params.weights, input), params.bias);
  const output = def.activation.forward(preActivation);
  return new LayerCache({ input, preActivation, output });
};

/** ネットワーク全体の順伝播 (forward pass)。全レイヤーのキャッシュ (cache) を返す */
export const forwardNetwork = (
  network: NetworkState,
): ((input: Vector) => ReadonlyArray<LayerCache>) =>
  (input: Vector): ReadonlyArray<LayerCache> => {
    const caches: LayerCache[] = [];
    let current = input;

    for (let i = 0; i < network.def.layers.length; i++) {
      const layerDef = network.def.layers[i];
      const layerParams = network.params[i];
      if (!layerDef || !layerParams) break;
      const cache = forwardLayer(layerDef, layerParams, current);
      caches.push(cache);
      current = cache.output;
    }

    return caches;
  };

/** 順伝播 (forward pass) の最終出力を取得する */
export const predict = (
  network: NetworkState,
  input: Vector,
): Vector => {
  const caches = forwardNetwork(network)(input);
  const last = caches[caches.length - 1];
  return last?.output ?? input;
};
