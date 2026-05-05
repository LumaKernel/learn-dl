import Link from "next/link";
import { CreateModelForm } from "./_components/create-model-form";
import { ModelList } from "./_components/model-list";

export default function ModelsPage() {
  return (
    <div className="min-h-screen flex flex-col">
      <header className="border-b border-zinc-200 dark:border-zinc-700 px-4 py-2 flex items-center gap-4">
        <Link href="/" className="text-sm text-zinc-500 hover:text-zinc-700 dark:hover:text-zinc-300">
          &larr; トップ
        </Link>
        <h1 className="text-lg font-bold">モデル一覧</h1>
      </header>
      <main className="flex-1 p-6 max-w-4xl mx-auto w-full flex flex-col gap-8">
        <CreateModelForm />
        <ModelList />
      </main>
    </div>
  );
}
