/* eslint-disable @typescript-eslint/no-unused-vars */
import type { NetworkState, TrainingConfig, TrainingSample } from "./network";
import type { LayerGradients } from "./layer";
import { forwardNetwork } from "./forward";
import { backwardNetwork, applyGradients } from "./backward";

// ============================================================
// TODO: ここを自分で実装する
// ============================================================

/**
 * 1つのサンプルに対して勾配を計算してください。
 *
 * @param network - 現在のネットワーク状態
 * @param sample - 入力と正解のペア
 * @returns [各レイヤーの勾配, 損失値]
 *
 * ヒント:
 * 1. forwardNetwork で forward pass
 * 2. network.def.lossFunction.loss で損失を計算
 * 3. network.def.lossFunction.gradient で出力の勾配を計算
 * 4. backwardNetwork で全レイヤーの勾配を計算
 */
export const computeGradients: (
  network: NetworkState,
  sample: TrainingSample,
) => readonly [ReadonlyArray<LayerGradients>, number] = (_network, _sample) => {
  throw new Error("TODO: computeGradients を実装してください");
};

/**
 * ミニバッチの勾配を平均してください。
 *
 * @param allGradients - 各サンプルごとの勾配の配列
 * @returns 平均された勾配
 *
 * ヒント:
 * - 各レイヤーの dWeights と dBias をサンプル数で割る
 */
export const averageGradients: (
  allGradients: ReadonlyArray<ReadonlyArray<LayerGradients>>,
) => ReadonlyArray<LayerGradients> = (_allGradients) => {
  throw new Error("TODO: averageGradients を実装してください");
};

/**
 * 1ステップの学習を実行してください。
 *
 * SGDモード (batchSize=1): ランダムな1サンプルで勾配計算 → 更新
 * ミニバッチモード: batchSize個のサンプルで勾配平均 → 更新
 *
 * @param network - 現在のネットワーク状態
 * @param config - 学習設定
 * @param samples - 利用可能な全サンプル
 * @param sampleIndices - このステップで使うサンプルのインデックス
 * @returns [更新後のネットワーク, 平均損失]
 */
export const trainStep: (
  network: NetworkState,
  config: TrainingConfig,
  samples: ReadonlyArray<TrainingSample>,
  sampleIndices: ReadonlyArray<number>,
) => readonly [NetworkState, number] = (_network, _config, _samples, _sampleIndices) => {
  throw new Error("TODO: trainStep を実装してください");
};
