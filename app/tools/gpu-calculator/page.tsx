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
      <div className="mx-auto max-w-7xl px-5 pb-16 pt-10 sm:px-8 lg:px-10">
        {/* Sticky nav */}
        <nav className="mb-10 flex items-center justify-between">
          <div className="flex items-center gap-2.5 text-sm text-muted">
            <Link
              href="/tools"
              className="relative transition-colors duration-200 hover:text-foreground after:absolute after:-bottom-0.5 after:left-0 after:h-px after:w-0 after:bg-accent after:transition-all after:duration-200 hover:after:w-full"
            >
              Tools
            </Link>
            <span className="text-border">/</span>
            <span className="text-foreground font-medium">{tool.title}</span>
          </div>
          <ThemeToggle />
        </nav>

        {/* Hero */}
        <header className="mb-12 rounded-2xl border border-border bg-surface/70 p-8 sm:p-10 backdrop-blur-sm">
          <span className="inline-flex rounded-full bg-accent-soft px-3.5 py-1 text-xs font-medium tracking-wide text-accent">
            {tool.category}
          </span>
          <h1 className="mt-5 text-foreground">{tool.title}</h1>
          <p className="mt-4 max-w-[72ch] text-base leading-relaxed text-muted sm:text-lg" style={{ lineHeight: 1.85 }}>
            {tool.summary}
          </p>
          <div className="mt-7 flex flex-wrap gap-2">
            {tool.tags.map((tag) => (
              <span
                key={tag}
                className="rounded-full border border-border-subtle bg-background/60 px-3 py-1 text-xs text-muted"
              >
                {tag}
              </span>
            ))}
          </div>
        </header>

        <GpuCalculatorEmbed />
      </div>
    </main>
  )
}
