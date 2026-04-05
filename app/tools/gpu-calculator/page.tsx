import type { Metadata } from "next"
import Link from "next/link"
import { getTool } from "@/lib/utils/tools"
import GpuCalculatorEmbed from "./gpu-calculator-embed"
import ThemeToggle from "./theme-toggle"

const tool = getTool("gpu-calculator")

export const metadata: Metadata = {
  title: tool?.title ?? "LLM Training GPU Calculator",
  description: tool?.summary,
}

export default function GpuCalculatorPage() {
  if (!tool) {
    return null
  }

  return (
    <main className="min-h-screen">
      <div className="mx-auto max-w-7xl px-4 pb-14 pt-8 sm:px-6 lg:px-8">
        <nav className="mb-8 flex items-center justify-between">
          <div className="flex items-center gap-2 text-sm text-muted">
            <Link href="/tools" className="transition-colors hover:text-foreground">
              Tools
            </Link>
            <span>/</span>
            <span className="text-foreground">{tool.title}</span>
          </div>
          <ThemeToggle />
        </nav>

        <div className="mb-10 rounded-3xl border border-border bg-surface/80 p-8 shadow-sm backdrop-blur">
          <span className="inline-flex rounded-full bg-accent-soft px-3 py-1 text-xs font-semibold tracking-wide text-accent">
            {tool.category}
          </span>
          <h1 className="mt-4 text-4xl text-foreground sm:text-5xl">{tool.title}</h1>
          <p className="mt-4 max-w-3xl text-base leading-7 text-muted sm:text-lg">
            {tool.summary}
          </p>
          <div className="mt-6 flex flex-wrap gap-2">
            {tool.tags.map((tag) => (
              <span
                key={tag}
                className="rounded-full border border-border bg-background/70 px-3 py-1 text-xs text-muted"
              >
                {tag}
              </span>
            ))}
          </div>
        </div>

        <GpuCalculatorEmbed />
      </div>
    </main>
  )
}
