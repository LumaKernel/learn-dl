import type { Vector } from "../math/vector";
import * as V from "../math/vector";
import * as Mat from "../math/matrix";
import { LayerGradients } from "./layer";
import type { LayerCache, LayerDef, LayerParams } from "./layer";
import { LayerParams as LayerParamsClass } from "./layer";
import { NetworkState } from "./network";

/**
 * 単一レイヤーの逆伝播。
 *
 * 1. activation の backward で dOutput → dPreActivation
 * 2. dWeights = outer(dPreActivation, input)
 * 3. dBias = dPreActivation
 * 4. dInput = W^T * dPreActivation
 */
export const backwardLayer = (
  def: LayerDef,
  params: LayerParams,
  cache: LayerCache,
  dOutput: Vector,
): readonly [LayerGradients, Vector] => {
  const dActivation = def.activation.backward(cache.preActivation, cache.output);
  const dPreActivation = V.hadamard(dOutput, dActivation);

  const dWeights = Mat.outer(dPreActivation, cache.input);
  const dBias = dPreActivation;
  const dInput = Mat.mulVec(Mat.transpose(params.weights), dPreActivation);

  return [new LayerGradients({ dWeights, dBias }), dInput];
};

/**
 * ネットワーク全体の逆伝播。
 * 最終層から入力層へ逆順に勾配を伝播する。
 */
export const backwardNetwork = (
  network: NetworkState,
  caches: ReadonlyArray<LayerCache>,
  lossGradient: Vector,
): ReadonlyArray<LayerGradients> => {
  const gradients: LayerGradients[] = Array.from(
    { length: caches.length },
    () => new LayerGradients({ dWeights: Mat.zeros(0, 0), dBias: V.zeros(0) }),
  );
  let dOutput = lossGradient;

  for (let i = caches.length - 1; i >= 0; i--) {
    const layerDef = network.def.layers[i];
    const layerParams = network.params[i];
    const cache = caches[i];
    if (!layerDef || !layerParams || !cache) continue;

    const [grad, dInput] = backwardLayer(layerDef, layerParams, cache, dOutput);
    gradients[i] = grad;
    dOutput = dInput;
  }

  return gradients;
};

/**
 * 勾配降下法でパラメータを更新する。
 * w_new = w_old - lr * dw
 */
export const applyGradients = (
  network: NetworkState,
  gradients: ReadonlyArray<LayerGradients>,
  learningRate: number,
): NetworkState => {
  const newParams = network.params.map((p, i) => {
    const grad = gradients[i];
    if (!grad) return p;
    return new LayerParamsClass({
      weights: Mat.sub(p.weights, Mat.scale(grad.dWeights, learningRate)),
      bias: V.sub(p.bias, V.scale(grad.dBias, learningRate)),
    });
  });
  return new NetworkState({ def: network.def, params: newParams });
};
