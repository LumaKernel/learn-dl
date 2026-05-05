import Link from "next/link";
import { ModelWorkspace } from "./_components/model-workspace";

type Props = {
  readonly params: Promise<{ readonly id: string }>;
};

export default async function ModelDetailPage({ params }: Props) {
  const { id } = await params;
  return (
    <div className="min-h-screen flex flex-col">
      <header className="border-b border-zinc-200 dark:border-zinc-700 px-4 py-2 flex items-center gap-4">
        <Link href="/models" className="text-sm text-zinc-500 hover:text-zinc-700 dark:hover:text-zinc-300">
          &larr; モデル一覧
        </Link>
        <h1 className="text-lg font-bold">モデル詳細</h1>
      </header>
      <main className="flex-1 p-6 max-w-6xl mx-auto w-full">
        <ModelWorkspace modelId={id} />
      </main>
    </div>
  );
}
