import { describe, it, expect } from "vitest";
import { LayerDef } from "./layer";
import { NetworkDef, NetworkState, TrainingConfig, TrainingSample } from "./network";
import { softmax } from "./activation";
import { crossEntropy } from "./loss";
import { initNetwork, he, smallRandom } from "./init";
import { predict } from "./forward";
import { trainStep } from "./training";
import * as V from "../math/vector";

/**
 * 学習 (training) の基本動作テスト。
 * 1層のネットワークで trainStep 実行後に重みが変化することを検証する。
 */
describe("trainStep", () => {
  const makeSimpleNetwork = (): NetworkState => {
    const def = new NetworkDef({
      layers: [
        new LayerDef({ inputSize: 4, outputSize: 3, activation: softmax }),
      ],
      lossFunction: crossEntropy,
    });
    return initNetwork(def, he, 42);
  };

  const makeSample = (input: ReadonlyArray<number>, label: number): TrainingSample =>
    new TrainingSample({
      input: V.from(input),
      target: V.from(Array.from({ length: 3 }, (_, i) => (i === label ? 1 : 0))),
    });

  it("1ステップで重みが変化する", () => {
    const network = makeSimpleNetwork();
    const config = new TrainingConfig({ learningRate: 0.1, batchSize: 1, mode: "sgd" });
    const samples = [makeSample([1, 0, 1, 0], 0)];

    const oldWeights = network.params[0]?.weights;

    const [updated, loss] = trainStep(network, config, samples, [0]);

    const newWeights = updated.params[0]?.weights;

    // 損失は正の値
    expect(loss).toBeGreaterThan(0);

    // 重みが実際に変化している
    expect(oldWeights).toBeDefined();
    expect(newWeights).toBeDefined();
    if (!oldWeights || !newWeights) return;

    let changed = false;
    for (let i = 0; i < oldWeights.rows; i++) {
      for (let j = 0; j < oldWeights.cols; j++) {
        if (oldWeights.at(i, j) !== newWeights.at(i, j)) {
          changed = true;
        }
      }
    }
    expect(changed).toBe(true);
  });

  it("複数ステップで損失が減少傾向になる", () => {
    const network = makeSimpleNetwork();
    const config = new TrainingConfig({ learningRate: 0.1, batchSize: 2, mode: "minibatch" });
    const samples = [
      makeSample([1, 0, 0, 0], 0),
      makeSample([0, 1, 0, 0], 1),
      makeSample([0, 0, 1, 0], 2),
    ];

    let current = network;
    const losses: number[] = [];

    for (let i = 0; i < 50; i++) {
      const indices = [i % samples.length, (i + 1) % samples.length];
      const [updated, loss] = trainStep(current, config, samples, indices);
      current = updated;
      losses.push(loss);
    }

    // 最初の10ステップの平均損失 > 最後の10ステップの平均損失
    const earlyAvg = losses.slice(0, 10).reduce((a, b) => a + b, 0) / 10;
    const lateAvg = losses.slice(-10).reduce((a, b) => a + b, 0) / 10;
    expect(lateAvg).toBeLessThan(earlyAvg);
  });

  it("学習後に予測が改善する", () => {
    const def = new NetworkDef({
      layers: [
        new LayerDef({ inputSize: 4, outputSize: 3, activation: softmax }),
      ],
      lossFunction: crossEntropy,
    });
    const network = initNetwork(def, smallRandom, 123);
    const config = new TrainingConfig({ learningRate: 0.5, batchSize: 3, mode: "minibatch" });
    const samples = [
      makeSample([1, 0, 0, 0], 0),
      makeSample([0, 1, 0, 0], 1),
      makeSample([0, 0, 1, 0], 2),
    ];

    let current = network;
    for (let i = 0; i < 200; i++) {
      const [updated] = trainStep(current, config, samples, [0, 1, 2]);
      current = updated;
    }

    // 学習後、各入力に対して正しいラベルが最大確率になっているか
    const pred0 = predict(current, V.from([1, 0, 0, 0]));
    const pred1 = predict(current, V.from([0, 1, 0, 0]));
    const pred2 = predict(current, V.from([0, 0, 1, 0]));

    expect(V.argmax(pred0)).toBe(0);
    expect(V.argmax(pred1)).toBe(1);
    expect(V.argmax(pred2)).toBe(2);
  });

  it("ゼロ入力でも重みが変化する", () => {
    const network = makeSimpleNetwork();
    const config = new TrainingConfig({ learningRate: 0.1, batchSize: 1, mode: "sgd" });
    // 全ゼロ入力 — バイアスのみが更新されるはず
    const samples = [makeSample([0, 0, 0, 0], 1)];

    const oldBias = network.params[0]?.bias;
    const [updated] = trainStep(network, config, samples, [0]);
    const newBias = updated.params[0]?.bias;

    expect(oldBias).toBeDefined();
    expect(newBias).toBeDefined();
    if (!oldBias || !newBias) return;

    let biasChanged = false;
    for (let i = 0; i < oldBias.length; i++) {
      if (oldBias.at(i) !== newBias.at(i)) biasChanged = true;
    }
    expect(biasChanged).toBe(true);
  });
});
