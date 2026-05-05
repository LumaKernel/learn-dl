"use client";

import { useCallback, useState, useTransition } from "react";
import { useRouter } from "next/navigation";
import { allActivations } from "@/domain/nn/activation";
import { allInitStrategies } from "@/domain/nn/init";
import { allTasks, findTask } from "@/domain/nn/task";
import { allLossFunctions } from "@/domain/nn/loss";
import { createModel } from "../actions";

type HiddenLayerConfig = {
  readonly outputSize: string;
  readonly activationName: string;
};

export function CreateModelForm() {
  const router = useRouter();
  const [isPending, startTransition] = useTransition();

  const [name, setName] = useState("モデル 1");
  const defaultTask = allTasks[0] ?? (() => { throw new Error("タスクが1つも定義されていません"); })();
  const [taskName, setTaskName] = useState(defaultTask.name);
  const task = findTask(taskName);

  const defaultInput = task.inputOptions[0] ?? (() => { throw new Error("入力オプションが定義されていません"); })();
  const [inputSizeStr, setInputSizeStr] = useState(String(defaultInput.size));
  const [lossName, setLossName] = useState(task.defaultLossName);
  const [initStrategyName, setInitStrategyName] = useState("he");
  const [seed, setSeed] = useState("42");
  const [hiddenLayers, setHiddenLayers] = useState<ReadonlyArray<HiddenLayerConfig>>([
    { outputSize: "32", activationName: "relu" },
  ]);

  const handleTaskChange = useCallback((newTaskName: string) => {
    const newTask = findTask(newTaskName);
    setTaskName(newTaskName);
    setLossName(newTask.defaultLossName);
    const firstInput = newTask.inputOptions[0] ?? (() => { throw new Error("入力オプションが定義されていません"); })();
    setInputSizeStr(String(firstInput.size));
  }, []);

  const addHiddenLayer = useCallback(() => {
    setHiddenLayers((prev) => [...prev, { outputSize: "16", activationName: "relu" }]);
  }, []);

  const removeHiddenLayer = useCallback((idx: number) => {
    setHiddenLayers((prev) => prev.filter((_, i) => i !== idx));
  }, []);

  const updateHiddenLayer = useCallback((idx: number, patch: Partial<HiddenLayerConfig>) => {
    setHiddenLayers((prev) =>
      prev.map((l, i) => (i === idx ? { ...l, ...patch } : l)),
    );
  }, []);

  const swapHiddenLayers = useCallback((idxA: number, idxB: number) => {
    setHiddenLayers((prev) => {
      const next = [...prev];
      const a = next[idxA];
      const b = next[idxB];
      if (a === undefined || b === undefined) {
        throw new Error(`レイヤーのインデックスが不正です: ${String(idxA) satisfies string}, ${String(idxB) satisfies string}`);
      }
      next[idxA] = b;
      next[idxB] = a;
      return next;
    });
  }, []);

  const [validationError, setValidationError] = useState<string | null>(null);

  const handleCreate = useCallback(() => {
    setValidationError(null);

    const parsedInputSize = parseInt(inputSizeStr, 10);
    if (!task.inputOptions.some((o) => o.size === parsedInputSize)) {
      setValidationError("入力サイズがタスクの選択肢にありません");
      return;
    }

    const parsedSeed = parseInt(seed, 10);
    if (Number.isNaN(parsedSeed)) {
      setValidationError("乱数シードが不正です");
      return;
    }

    // 隠れ層のバリデーション
    const parsedHiddenLayers: ReadonlyArray<{ readonly outputSize: number; readonly activationName: string }> = [];
    for (const config of hiddenLayers) {
      const outSize = parseInt(config.outputSize, 10);
      if (Number.isNaN(outSize) || outSize <= 0) {
        setValidationError(`隠れ層の出力サイズ "${config.outputSize satisfies string}" が不正です`);
        return;
      }
      (parsedHiddenLayers as { outputSize: number; activationName: string }[]).push({
        outputSize: outSize,
        activationName: config.activationName,
      });
    }

    startTransition(async () => {
      const id = await createModel({
        name,
        taskName: task.name,
        inputSize: parsedInputSize,
        lossName,
        initStrategyName,
        seed: parsedSeed,
        hiddenLayers: parsedHiddenLayers,
        outputSize: task.outputSize,
        outputActivationName: task.outputActivationName,
      });
      router.push(`/models/${id satisfies string}`);
    });
  }, [name, task, inputSizeStr, lossName, initStrategyName, seed, hiddenLayers, router]);

  // 表示用: 入力 → 隠れ層 → 出力
  const layerSizeFlow = [
    inputSizeStr,
    ...hiddenLayers.map((l) => l.outputSize),
    String(task.outputSize),
  ].join(" → ");

  return (
    <div className="border border-zinc-200 dark:border-zinc-700 rounded-lg p-4 flex flex-col gap-4">
      <h2 className="font-bold text-lg">新しいモデルを作成</h2>

      {/* タスク選択 */}
      <div className="flex flex-col gap-1">
        <span className="text-sm font-medium text-zinc-600 dark:text-zinc-400">
          タスク (問題設定)
        </span>
        <div className="flex flex-wrap gap-2">
          {allTasks.map((t) => (
            <button
              key={t.name}
              type="button"
              className={`px-3 py-2 text-sm rounded border transition-colors ${(taskName === t.name ? "border-blue-500 bg-blue-50 dark:bg-blue-900/30 text-blue-700 dark:text-blue-300" : "border-zinc-300 dark:border-zinc-600 hover:border-zinc-400") satisfies string}`}
              onClick={() => handleTaskChange(t.name)}
            >
              {t.description}
            </button>
          ))}
        </div>
      </div>

      <div className="grid grid-cols-2 gap-3">
        <label className="flex flex-col gap-1">
          <span className="text-sm text-zinc-600 dark:text-zinc-400">モデル名</span>
          <input
            type="text"
            className="border border-zinc-300 dark:border-zinc-600 rounded px-2 py-1 text-sm bg-transparent"
            value={name}
            onChange={(e) => setName(e.target.value)}
          />
        </label>

        <label className="flex flex-col gap-1">
          <span className="text-sm text-zinc-600 dark:text-zinc-400">入力サイズ (input size)</span>
          <select
            className="border border-zinc-300 dark:border-zinc-600 rounded px-2 py-1 text-sm bg-transparent"
            value={inputSizeStr}
            onChange={(e) => setInputSizeStr(e.target.value)}
          >
            {task.inputOptions.map((opt) => (
              <option key={opt.size} value={String(opt.size)}>{opt.label}</option>
            ))}
          </select>
        </label>

        <label className="flex flex-col gap-1">
          <span className="text-sm text-zinc-600 dark:text-zinc-400">損失関数 (loss function)</span>
          <select
            className="border border-zinc-300 dark:border-zinc-600 rounded px-2 py-1 text-sm bg-transparent"
            value={lossName}
            onChange={(e) => setLossName(e.target.value)}
          >
            {allLossFunctions.map((l) => (
              <option key={l.name} value={l.name}>{l.description}</option>
            ))}
          </select>
        </label>

        <label className="flex flex-col gap-1">
          <span className="text-sm text-zinc-600 dark:text-zinc-400">初期化戦略 (init strategy)</span>
          <select
            className="border border-zinc-300 dark:border-zinc-600 rounded px-2 py-1 text-sm bg-transparent"
            value={initStrategyName}
            onChange={(e) => setInitStrategyName(e.target.value)}
          >
            {allInitStrategies.map((s) => (
              <option key={s.name} value={s.name}>
                {s.name} — {s.description}
              </option>
            ))}
          </select>
        </label>

        <label className="flex flex-col gap-1">
          <span className="text-sm text-zinc-600 dark:text-zinc-400">乱数シード (random seed)</span>
          <input
            type="number"
            className="border border-zinc-300 dark:border-zinc-600 rounded px-2 py-1 text-sm bg-transparent"
            value={seed}
            onChange={(e) => setSeed(e.target.value)}
          />
        </label>
      </div>

      <div className="flex flex-col gap-2">
        <div className="flex items-center justify-between">
          <span className="text-sm font-medium text-zinc-600 dark:text-zinc-400">
            レイヤー構成 (layer configuration)
          </span>
          <button
            type="button"
            className="text-xs px-2 py-1 bg-zinc-200 dark:bg-zinc-700 rounded hover:bg-zinc-300 dark:hover:bg-zinc-600"
            onClick={addHiddenLayer}
          >
            + 隠れ層を追加
          </button>
        </div>

        <div className="text-xs text-zinc-400 mb-1">
          {`入力: ${layerSizeFlow satisfies string}`}
        </div>

        {hiddenLayers.map((config, idx) => (
          <div
            key={idx}
            className="flex items-center gap-2 bg-zinc-50 dark:bg-zinc-800 rounded px-3 py-2"
          >
            <span className="text-xs text-zinc-500 w-20 shrink-0">
              {`隠れ層 ${String(idx + 1) satisfies string}`}
            </span>
            <label className="flex items-center gap-1 text-xs">
              <span className="text-zinc-500">出力:</span>
              <input
                type="number"
                className="w-16 border border-zinc-300 dark:border-zinc-600 rounded px-1 py-0.5 text-xs bg-transparent"
                value={config.outputSize}
                onChange={(e) => updateHiddenLayer(idx, { outputSize: e.target.value })}
              />
            </label>
            <label className="flex items-center gap-1 text-xs">
              <span className="text-zinc-500">活性化:</span>
              <select
                className="border border-zinc-300 dark:border-zinc-600 rounded px-1 py-0.5 text-xs bg-transparent"
                value={config.activationName}
                onChange={(e) => updateHiddenLayer(idx, { activationName: e.target.value })}
              >
                {allActivations.map((a) => (
                  <option key={a.name} value={a.name}>{a.name}</option>
                ))}
              </select>
            </label>
            <div className="ml-auto flex items-center gap-1">
              <button
                type="button"
                className="text-xs text-zinc-400 hover:text-zinc-600 disabled:opacity-30"
                disabled={idx === 0}
                onClick={() => swapHiddenLayers(idx, idx - 1)}
                title="上へ移動"
              >
                ↑
              </button>
              <button
                type="button"
                className="text-xs text-zinc-400 hover:text-zinc-600 disabled:opacity-30"
                disabled={idx >= hiddenLayers.length - 1}
                onClick={() => swapHiddenLayers(idx, idx + 1)}
                title="下へ移動"
              >
                ↓
              </button>
              <button
                type="button"
                className="text-xs text-red-500 hover:text-red-700"
                onClick={() => removeHiddenLayer(idx)}
              >
                削除
              </button>
            </div>
          </div>
        ))}

        {hiddenLayers.length === 0 && (
          <div className="text-xs text-zinc-400 italic px-3 py-2">
            隠れ層なし（入力 → 出力層の直結）
          </div>
        )}

        {/* 出力層（タスクから自動決定、編集不可） */}
        <div className="flex items-center gap-2 rounded px-3 py-2 bg-blue-50 dark:bg-blue-900/20 border border-blue-200 dark:border-blue-800">
          <span className="text-xs text-blue-600 dark:text-blue-400 w-20 shrink-0 font-medium">
            出力層
          </span>
          <span className="text-xs text-zinc-500">
            {`出力: ${String(task.outputSize) satisfies string}`}
          </span>
          <span className="text-xs text-zinc-500">
            {`活性化: ${task.outputActivationName satisfies string}`}
          </span>
          <span className="text-xs text-zinc-400 ml-auto">
            タスクにより固定
          </span>
        </div>
      </div>

      {validationError && (
        <p className="text-sm text-red-500">{validationError}</p>
      )}

      <button
        type="button"
        className="self-start px-4 py-2 bg-blue-500 text-white rounded hover:bg-blue-600 disabled:opacity-50"
        disabled={isPending}
        onClick={handleCreate}
      >
        {isPending ? "作成中..." : "モデルを作成"}
      </button>
    </div>
  );
}
