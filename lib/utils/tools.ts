export interface Tool {
  slug: string
  title: string
  summary: string
  category: string
  tags: string[]
  relatedPost?: string
}

const tools: Tool[] = [
  {
    slug: "gpu-calculator",
    title: "LLM Training GPU Calculator",
    summary:
      "Estimate GPU requirements for LLM training — compute memory breakdown, parallelism strategy, training time, and cost across pretraining and post-training phases.",
    category: "Planning & Estimation",
    tags: ["llm", "training", "gpu", "compute", "distributed-training"],
    relatedPost: undefined,
  },
]

export function getTools(): Tool[] {
  return tools
}

export function getTool(slug: string): Tool | undefined {
  return tools.find((t) => t.slug === slug)
}
