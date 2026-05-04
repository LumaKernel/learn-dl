/* eslint-disable @typescript-eslint/no-unused-vars */
import type { Vector } from "../math/vector";
import type { Matrix } from "../math/matrix";
import type { LayerCache, LayerDef, LayerGradients, LayerParams } from "./layer";
import type { NetworkState } from "./network";
import type { LossFunction } from "./loss";

// ============================================================
// TODO: ここを自分で実装する
// ============================================================

/**
 * 単一レイヤーの逆伝播を実装してください。
 *
 * @param def - レイヤーの定義（activation関数を含む）
 * @param params - レイヤーの重みとバイアス
 * @param cache - forward passで保存した中間値（input, preActivation, output）
 * @param dOutput - このレイヤーの出力に対する損失の勾配
 * @returns [レイヤーの勾配, 入力に対する勾配（前のレイヤーへ伝播する）]
 *
 * ヒント:
 * 1. activation の backward で dOutput から preActivation の勾配を得る
 * 2. dWeights = outer(dPreActivation, cache.input)
 * 3. dBias = dPreActivation
 * 4. dInput = transpose(weights) * dPreActivation
 */
export const backwardLayer: (
  def: LayerDef,
  params: LayerParams,
  cache: LayerCache,
  dOutput: Vector,
) => readonly [LayerGradients, Vector] = (_def, _params, _cache, _dOutput) => {
  throw new Error("TODO: backwardLayer を実装してください");
};

/**
 * ネットワーク全体の逆伝播を実装してください。
 *
 * @param network - ネットワークの状態（定義と重み）
 * @param caches - forward passの各レイヤーのキャッシュ
 * @param lossGradient - 最終出力に対する損失関数の勾配
 * @returns 各レイヤーの勾配の配列
 *
 * ヒント:
 * - 最後のレイヤーから最初のレイヤーへ逆順にbackwardLayerを呼ぶ
 * - 各レイヤーから返されるdInputを次の（前の）レイヤーのdOutputとして渡す
 */
export const backwardNetwork: (
  network: NetworkState,
  caches: ReadonlyArray<LayerCache>,
  lossGradient: Vector,
) => ReadonlyArray<LayerGradients> = (_network, _caches, _lossGradient) => {
  throw new Error("TODO: backwardNetwork を実装してください");
};

/**
 * 勾配を使ってパラメータを更新してください。
 *
 * @param network - 現在のネットワーク状態
 * @param gradients - 各レイヤーの勾配
 * @param learningRate - 学習率
 * @returns 更新されたネットワーク状態
 *
 * ヒント:
 * - weights_new = weights_old - learningRate * dWeights
 * - bias_new = bias_old - learningRate * dBias
 */
export const applyGradients: (
  network: NetworkState,
  gradients: ReadonlyArray<LayerGradients>,
  learningRate: number,
) => NetworkState = (_network, _gradients, _learningRate) => {
  throw new Error("TODO: applyGradients を実装してください");
};
