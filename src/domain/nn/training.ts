import * as V from "../math/vector";
import * as Mat from "../math/matrix";
import type { NetworkState, TrainingConfig, TrainingSample } from "./network";
import { LayerGradients } from "./layer";
import { forwardNetwork } from "./forward";
import { backwardNetwork, applyGradients } from "./backward";

/**
 * 1つのサンプルに対して forward → loss → backward で勾配を計算する。
 */
export const computeGradients = (
  network: NetworkState,
  sample: TrainingSample,
): readonly [ReadonlyArray<LayerGradients>, number] => {
  const caches = forwardNetwork(network)(sample.input);
  const lastCache = caches[caches.length - 1];
  const output = lastCache?.output ?? sample.input;

  const loss = network.def.lossFunction.loss(output, sample.target);
  const lossGrad = network.def.lossFunction.gradient(output, sample.target);
  const gradients = backwardNetwork(network, caches, lossGrad);

  return [gradients, loss];
};

/**
 * 複数サンプルの勾配を要素ごとに平均する。
 */
export const averageGradients = (
  allGradients: ReadonlyArray<ReadonlyArray<LayerGradients>>,
): ReadonlyArray<LayerGradients> => {
  const first = allGradients[0];
  if (!first || allGradients.length === 0) return [];

  const n = allGradients.length;

  return first.map((_, layerIdx) => {
    const layerGrads = allGradients.map((g) => g[layerIdx]);
    const firstGrad = layerGrads[0];
    if (!firstGrad) {
      return new LayerGradients({
        dWeights: Mat.zeros(0, 0),
        dBias: V.zeros(0),
      });
    }

    let sumWeights = firstGrad.dWeights;
    let sumBias = firstGrad.dBias;

    for (let i = 1; i < layerGrads.length; i++) {
      const g = layerGrads[i];
      if (!g) continue;
      sumWeights = Mat.add(sumWeights, g.dWeights);
      sumBias = V.add(sumBias, g.dBias);
    }

    return new LayerGradients({
      dWeights: Mat.scale(sumWeights, 1 / n),
      dBias: V.scale(sumBias, 1 / n),
    });
  });
};

/**
 * 1ステップの学習を実行する。
 *
 * sampleIndices で指定されたサンプルの勾配を計算し、
 * 平均して重みを更新する。
 */
export const trainStep = (
  network: NetworkState,
  config: TrainingConfig,
  samples: ReadonlyArray<TrainingSample>,
  sampleIndices: ReadonlyArray<number>,
): readonly [NetworkState, number] => {
  const allGradients: ReadonlyArray<LayerGradients>[] = [];
  let totalLoss = 0;

  for (const idx of sampleIndices) {
    const sample = samples[idx];
    if (!sample) continue;
    const [grads, loss] = computeGradients(network, sample);
    allGradients.push(grads);
    totalLoss += loss;
  }

  if (allGradients.length === 0) return [network, 0];

  const avgGrads = averageGradients(allGradients);
  const updated = applyGradients(network, avgGrads, config.learningRate);
  const avgLoss = totalLoss / allGradients.length;

  return [updated, avgLoss];
};
