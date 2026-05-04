import Link from "next/link";

export default function ModelsPage() {
  return (
    <div className="min-h-screen flex flex-col">
      <header className="border-b border-zinc-200 dark:border-zinc-700 px-4 py-2 flex items-center gap-4">
        <Link href="/" className="text-sm text-zinc-500 hover:text-zinc-700 dark:hover:text-zinc-300">
          &larr; トップ
        </Link>
        <h1 className="text-lg font-bold">モデル一覧</h1>
      </header>
      <main className="flex-1 p-6 max-w-4xl mx-auto w-full">
        <p className="text-zinc-500">準備中 — ニューラルネットワークの作成・学習・可視化機能をここに実装します</p>
      </main>
    </div>
  );
}
