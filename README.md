# LLM Training GPU Calculator

What does it take to train an LLM? This is an interactive calculator that answers with numbers: how many GPUs you need, where the memory goes, how the model should be split across devices, how long the run takes, and what it costs.

It covers two phases:

- **Pretraining**: memory breakdown (weights, gradients, optimizer states, activations), a recommended parallelism layout (data / tensor / context / pipeline / expert), training time from a FLOPs model with realistic utilization, and cloud cost.
- **Post-training**: SFT, DPO, PPO, and GRPO, each as full fine-tuning, LoRA, QLoRA, or MeZO, including the extra policy / reference / reward model copies that RL methods carry.

Presets cover 14 models (GPT-2 124M up to DeepSeek V3 671B, dense and MoE) and 27 GPUs (V100 through B200 NVL72, plus AMD MI250X/MI300X, consumer cards, and Apple silicon). Everything is overridable: architecture dims, tokens, batch sizes, precision, optimizer (13 profiles), activation checkpointing, ZeRO stage, parallelism degrees, GPU prices. Results export as text or JSON.

## Run it

```bash
npm install
npm run dev
```

Open http://localhost:3000. There's no backend and no env vars; your inputs persist in localStorage. `npm run build` makes a production build.

Stack: Next.js 16, React 19, TypeScript (strict), Tailwind 4, Framer Motion.

## Where the numbers come from

Every formula in the app traces to `spec/llm-training-gpu-calculator-spec.md` (~2,200 lines): parameter counting, activation memory under each checkpointing mode, ZeRO stages, MoE expert parallelism, FLOPs and MFU, pricing. The spec is in turn built from 24 research deep-dives in `spec/research/` (Chinchilla and Kaplan scaling laws, ZeRO, Megatron-LM, FlashAttention and FlashAttention-3, EleutherAI's Transformer Math 101, the LLaMA 3 / PaLM / BLOOM training reports, and others). Formula changes go through the spec first; code follows.

## Repo layout

```
app/                          Next.js shell (single page)
components/gpu-calculator/
  formulas/                   pure math: memory, compute, parallelism, cost (+ validation)
  components/                 UI: hero bar, layer stack, memory bar, selectors, ...
  constants.ts                GPU specs, model presets, optimizer profiles
  types.ts                    all shared types
spec/                         the formula spec, implementation plan, research deep-dives
scripts/parity/               the numbers-parity gate (below)
```

## Numbers-parity gate

UI changes must not change the math. `scripts/parity/parity-check.mjs` drives the app with Playwright through 25 scenarios (model presets, tab switches, numeric edits), captures the JSON export of each, and diffs it byte-for-byte against `baseline-snapshots.json`. The baselines are never regenerated: if a snapshot differs, the numbers changed, and the fix goes in the code, not the gate.

```bash
npm i -D playwright-core --no-save   # once per checkout
npm run dev                          # in another terminal
node scripts/parity/parity-check.mjs
```
