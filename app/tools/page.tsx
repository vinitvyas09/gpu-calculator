import Link from "next/link"
import { getTools } from "@/lib/utils/tools"

export default function ToolsPage() {
  const tools = getTools()

  return (
    <main className="min-h-screen">
      <div className="mx-auto max-w-6xl px-4 py-12 sm:px-6 lg:px-8">
        <h1 className="text-4xl text-foreground">Tools</h1>
        <p className="mt-3 max-w-2xl text-base leading-7 text-muted">
          Interactive tools and estimators built into this workspace.
        </p>

        <div className="mt-10 grid gap-4 md:grid-cols-2">
          {tools.map((tool) => (
            <Link
              key={tool.slug}
              href={`/tools/${tool.slug}`}
              className="rounded-3xl border border-border bg-surface/80 p-6 shadow-sm transition-transform hover:-translate-y-0.5"
            >
              <span className="inline-flex rounded-full bg-accent-soft px-3 py-1 text-xs font-semibold text-accent">
                {tool.category}
              </span>
              <h2 className="mt-4 text-2xl text-foreground">{tool.title}</h2>
              <p className="mt-3 text-sm leading-6 text-muted">{tool.summary}</p>
            </Link>
          ))}
        </div>
      </div>
    </main>
  )
}
