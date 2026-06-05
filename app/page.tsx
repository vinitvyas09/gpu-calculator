import type { Metadata } from "next"
import GpuCalculatorEmbed from "@/components/gpu-calculator-embed"

export const metadata: Metadata = {
  // `absolute` bypasses the layout's "%s | …" title template so the rendered
  // <title> is exactly this string (the value from the deleted lib/utils/tools.ts).
  title: { absolute: "LLM Training GPU Calculator" },
  description:
    "Estimate GPU requirements for LLM training — compute memory breakdown, parallelism strategy, training time, and cost across pretraining and post-training phases.",
}

export default function Home() {
  return (
    <main className="min-h-screen">
      <div className="mx-auto max-w-7xl px-5 pb-16 pt-10 sm:px-8 lg:px-10">
        <GpuCalculatorEmbed />
      </div>
    </main>
  )
}
