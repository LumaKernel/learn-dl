import Link from "next/link";

const navItems = [
  {
    href: "/datasets/handwriting",
    title: "Handwriting Dataset Builder",
    description: "Draw and collect handwritten digit data for training",
    local: true,
  },
] as const;

export default function Home() {
  return (
    <div className="flex flex-col flex-1 min-h-screen">
      <header className="border-b border-zinc-200 dark:border-zinc-700 px-6 py-4">
        <h1 className="text-2xl font-bold">Learn Deep Learning</h1>
        <p className="text-sm text-zinc-500 mt-1">
          Hands-on deep learning from scratch
        </p>
      </header>
      <main className="flex-1 p-6 max-w-4xl mx-auto w-full">
        <section>
          <h2 className="text-lg font-semibold mb-4">Tools</h2>
          <div className="grid gap-3 sm:grid-cols-2">
            {navItems.map((item) => (
              <Link
                key={item.href}
                href={item.href}
                className="block p-4 rounded-lg border border-zinc-200 dark:border-zinc-700 hover:border-blue-400 dark:hover:border-blue-500 transition-colors"
              >
                <div className="font-medium">{item.title}</div>
                <div className="text-sm text-zinc-500 mt-1">
                  {item.description}
                </div>
                {item.local && (
                  <span className="inline-block mt-2 text-xs px-2 py-0.5 bg-amber-100 dark:bg-amber-900 text-amber-700 dark:text-amber-300 rounded">
                    Local only
                  </span>
                )}
              </Link>
            ))}
          </div>
        </section>
      </main>
    </div>
  );
}
