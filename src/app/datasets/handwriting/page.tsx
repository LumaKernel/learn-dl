import { Suspense } from "react";
import { HandwritingEditor } from "./_components/handwriting-editor";

export default function HandwritingDatasetPage() {
  return (
    <div className="h-screen flex flex-col">
      <header className="border-b border-zinc-200 dark:border-zinc-700 px-4 py-2">
        <h1 className="text-lg font-bold">Handwriting Dataset Builder</h1>
      </header>
      <main className="flex-1 overflow-hidden">
        <Suspense>
          <HandwritingEditor />
        </Suspense>
      </main>
    </div>
  );
}
