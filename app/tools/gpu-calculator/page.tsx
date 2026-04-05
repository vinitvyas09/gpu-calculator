import type { Metadata } from "next"
import { getTool } from "@/lib/utils/tools"
import GpuCalculatorEmbed from "./gpu-calculator-embed"

const tool = getTool("gpu-calculator")!

export const metadata: Metadata = {
  title: `${tool.title} | Tools`,
  description: tool.summary,
}

export default function GpuCalculatorPage() {
  return (
    <main className="min-h-screen">
      {/* Breadcrumb */}
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 pt-6">
        <nav className="text-sm text-gray-500 dark:text-gray-400">
          <a href="/tools" className="hover:underline">
            Tools
          </a>
          <span className="mx-2">/</span>
          <span className="text-gray-900 dark:text-gray-100">{tool.title}</span>
        </nav>
      </div>

      {/* Header */}
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-6">
        <span className="inline-block px-3 py-1 text-xs font-medium rounded-full bg-blue-100 text-blue-800 dark:bg-blue-900/30 dark:text-blue-300 mb-3">
          {tool.category}
        </span>
        <h1 className="text-3xl font-bold tracking-tight text-gray-900 dark:text-gray-100">
          {tool.title}
        </h1>
        <p className="mt-2 text-gray-600 dark:text-gray-400 max-w-2xl">
          {tool.summary}
        </p>
      </div>

      {/* Calculator */}
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 pb-12">
        <GpuCalculatorEmbed />
      </div>
    </main>
  )
}
