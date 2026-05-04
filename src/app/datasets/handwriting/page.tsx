import { Suspense } from "react";
import Link from "next/link";
import { HandwritingEditor } from "./_components/handwriting-editor";

export default function HandwritingDatasetPage() {
  return (
    <div className="h-screen flex flex-col">
      <header className="border-b border-zinc-200 dark:border-zinc-700 px-4 py-2 flex items-center gap-4">
        <Link href="/" className="text-sm text-zinc-500 hover:text-zinc-700 dark:hover:text-zinc-300">
          &larr; トップ
        </Link>
        <h1 className="text-lg font-bold">手書きデータセット作成</h1>
      </header>
      <main className="flex-1 overflow-hidden">
        <Suspense>
          <HandwritingEditor />
        </Suspense>
      </main>
    </div>
  );
}
