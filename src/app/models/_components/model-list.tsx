"use client";

import { useCallback, useEffect, useState, useTransition } from "react";
import Link from "next/link";
import { listModels, deleteModel, type ModelListItem } from "../actions";
import { findTask } from "@/domain/nn/task";

export function ModelList() {
  const [models, setModels] = useState<ReadonlyArray<ModelListItem>>([]);
  const [isPending, startTransition] = useTransition();

  const refresh = useCallback(() => {
    startTransition(async () => {
      const m = await listModels();
      setModels(m);
    });
  }, []);

  useEffect(() => {
    refresh();
  }, [refresh]);

  const handleDelete = useCallback(
    (id: string) => {
      if (!confirm("このモデルを削除しますか？")) return;
      startTransition(async () => {
        await deleteModel(id);
        refresh();
      });
    },
    [refresh],
  );

  return (
    <div className="flex flex-col gap-3">
      <div className="flex items-center justify-between">
        <h2 className="font-bold text-lg">保存済みモデル</h2>
        {isPending && <span className="text-sm text-blue-500">読み込み中...</span>}
      </div>

      {models.length === 0 && !isPending && (
        <p className="text-zinc-500 text-sm">モデルがありません。上のフォームから作成してください。</p>
      )}

      <div className="grid gap-2">
        {models.map((m) => (
          <div
            key={m.id}
            className="border border-zinc-200 dark:border-zinc-700 rounded-lg p-3 flex items-center justify-between hover:bg-zinc-50 dark:hover:bg-zinc-800"
          >
            <Link
              href={`/models/${m.id satisfies string}`}
              className="flex-1 flex flex-col gap-0.5"
            >
              <span className="font-medium">{m.name}</span>
              <span className="text-xs text-zinc-500">
                {`${findTask(m.taskName).description satisfies string} | ${String(m.layerCount) satisfies string}層 | ステップ: ${String(m.trainingStep) satisfies string}`}
                {m.lastLoss !== null && ` | 損失: ${m.lastLoss.toFixed(4) satisfies string}`}
              </span>
              <span className="text-xs text-zinc-400">
                {/* eslint-disable-next-line @luma-dev/luma-ts/no-date -- Temporal の型定義が未導入のため */}
                {new Date(m.createdAt).toLocaleString("ja-JP")}
              </span>
            </Link>
            <button
              type="button"
              className="text-xs px-2 py-1 text-red-500 hover:bg-red-50 dark:hover:bg-red-900/20 rounded"
              onClick={() => handleDelete(m.id)}
            >
              削除
            </button>
          </div>
        ))}
      </div>
    </div>
  );
}
