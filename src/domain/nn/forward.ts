import { Vector } from "../math/vector";
import * as V from "../math/vector";
import * as Mat from "../math/matrix";
import { LayerCache } from "./layer";
import type { LayerParams, LayerDef } from "./layer";
import type { NetworkState } from "./network";

/** Forward pass through a single layer */
export const forwardLayer = (
  def: LayerDef,
  params: LayerParams,
  input: Vector,
): LayerCache => {
  const preActivation = V.add(Mat.mulVec(params.weights, input), params.bias);
  const output = def.activation.forward(preActivation);
  return new LayerCache({ input, preActivation, output });
};

/** Forward pass through entire network, returns all layer caches */
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

/** Get final output from forward pass */
export const predict = (
  network: NetworkState,
  input: Vector,
): Vector => {
  const caches = forwardNetwork(network)(input);
  const last = caches[caches.length - 1];
  return last?.output ?? input;
};
