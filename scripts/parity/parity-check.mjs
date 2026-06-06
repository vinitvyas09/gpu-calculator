// Numbers-parity gate for the UX redesign (spec/ux-redesign-plan.md §6).
//
// For each SCENARIO below: load the calculator fresh, perform the listed semantic
// interactions, capture the JSON export, and diff byte-for-byte against
// baseline-snapshots.json. The baseline was captured 2026-06-05 from commit
// f993952 (pre-redesign) and is the ground truth for "the numbers didn't change".
//
// RULES:
//   - NEVER regenerate or edit baseline-snapshots.json. If a snapshot differs,
//     the numbers changed — fix the code, never the gate.
//   - Selector fixes (a control moved/renamed) are allowed; numeric diffs are not.
//   - The ONLY diffs that may ever be allowed are the post-training keys affected
//     by the approved Phase 4 default-numGPUs seed (spec/ux-redesign-plan.md §0.1).
//     The script enforces this: ALLOWED_DIFF_KEYS rejects anything outside that set.
//     Every pretraining key must remain byte-identical FOREVER.
//
// Usage:
//   npm i -D playwright-core --no-save        (once per checkout)
//   npm run dev                                (in another terminal)
//   node scripts/parity/parity-check.mjs
// Env:
//   CALC_URL          override page URL (default: tries / then /tools/gpu-calculator)
//   CHROMIUM_PATH     override browser binary (default: newest ms-playwright headless shell)
//   ALLOWED_DIFF_KEYS comma-separated keys allowed to differ — restricted to SEED_AFFECTED_KEYS

import { chromium } from 'playwright-core'
import fs from 'fs'
import os from 'os'
import path from 'path'
import { fileURLToPath } from 'url'

const HERE = path.dirname(fileURLToPath(import.meta.url))
const BASELINE_PATH = path.join(HERE, 'baseline-snapshots.json')

// ── Scenarios ────────────────────────────────────────────────────────────────
// Each scenario starts from a FRESH page load (order-independent).
// step kinds:
//   {tab:'Post-Training'}                     click the phase tab
//   {select:'<exact option label>'}           pick an option in whichever select/combobox holds it
//   {number:[/label regex/, '<value>']}       set a numeric field located by accessible label
// export: 'json' (default) or 'text' — which export button to capture
const SCENARIOS = [
  { key: 'pretrain-default', steps: [] },
  { key: 'pretrain-GPT_2_Small_124M_', steps: [{ select: 'GPT-2 Small (124M)' }] },
  { key: 'pretrain-GPT_2_Medium_350M_', steps: [{ select: 'GPT-2 Medium (350M)' }] },
  { key: 'pretrain-GPT_2_Large_774M_', steps: [{ select: 'GPT-2 Large (774M)' }] },
  { key: 'pretrain-GPT_2_XL_1.56B_', steps: [{ select: 'GPT-2 XL (1.56B)' }] },
  { key: 'pretrain-LLaMA_2_70B_70B_', steps: [{ select: 'LLaMA 2 70B (70B)' }] },
  { key: 'pretrain-LLaMA_3_70B_70.6B_', steps: [{ select: 'LLaMA 3 70B (70.6B)' }] },
  { key: 'pretrain-LLaMA_3.1_405B_405B_', steps: [{ select: 'LLaMA 3.1 405B (405B)' }] },
  { key: 'pretrain-DeepSeekV3-MoE', steps: [{ select: 'DeepSeek V3 671B (671B)' }] },
  { key: 'pretrain-Mistral7B-H100default', steps: [{ select: 'Mistral 7B (7.2B)' }] },
  { key: 'pretrain-Mistral7B-A100-80GB', steps: [{ select: 'Mistral 7B (7.2B)' }, { select: 'A100 80GB (80 GB)' }] },
  // breadth: precision / parallelism / hardware-solver paths
  { key: 'pretrain-fp8', steps: [{ select: 'FP8' }] },
  { key: 'pretrain-manual-parallelism', steps: [{ select: 'Manual configuration' }] },
  { key: 'pretrain-targetdays-30', steps: [{ number: ['target training days', '30'] }] },
  { key: 'pretrain-64gpus', steps: [{ number: ['number of gpus', '64'] }] },
  // post-training methods / approaches
  { key: 'posttrain-default', steps: [{ tab: 'Post-Training' }] },
  { key: 'posttrain-dpo', steps: [{ tab: 'Post-Training' }, { select: 'DPO (Direct Preference)' }] },
  { key: 'posttrain-ppo', steps: [{ tab: 'Post-Training' }, { select: 'PPO (Proximal Policy)' }] },
  { key: 'posttrain-grpo', steps: [{ tab: 'Post-Training' }, { select: 'GRPO (Group Relative)' }] },
  { key: 'posttrain-qlora', steps: [{ tab: 'Post-Training' }, { select: 'QLoRA' }] },
  { key: 'posttrain-full-ft', steps: [{ tab: 'Post-Training' }, { select: 'Full fine-tuning' }] },
  // text/markdown export contract (pins WARNING_LABEL "Error", section labels, figures)
  { key: 'pretrain-default-text', steps: [], export: 'text' },
  { key: 'posttrain-default-text', steps: [{ tab: 'Post-Training' }], export: 'text' },
  // invalid-input behavior (pins what a blocked/invalid state serializes as);
  // the -text twin also pins WARNING_LABEL.critical ("Error") in the markdown export
  { key: 'pretrain-invalid-gpus-0', steps: [{ number: ['number of gpus', '0'] }] },
  { key: 'pretrain-invalid-gpus-0-text', steps: [{ number: ['number of gpus', '0'] }], export: 'text' },
]

// The only keys ALLOWED_DIFF_KEYS may name (Phase 4 numGPUs-seed blast radius).
const SEED_AFFECTED_KEYS = new Set([
  'posttrain-default', 'posttrain-dpo', 'posttrain-ppo', 'posttrain-grpo',
  'posttrain-qlora', 'posttrain-full-ft', 'posttrain-default-text',
])

// Validate ALLOWED_DIFF_KEYS up front, before spending minutes on captures.
{
  const requested = (process.env.ALLOWED_DIFF_KEYS || '').split(',').filter(Boolean)
  const illegal = requested.filter((k) => !SEED_AFFECTED_KEYS.has(k))
  if (illegal.length) {
    console.error(`ALLOWED_DIFF_KEYS rejected: ${illegal.join(', ')} — only the Phase-4 seed-affected post-training keys may ever be allow-diffed (plan §6). Pretraining keys are byte-frozen forever.`)
    process.exit(2)
  }
}

function findChromium() {
  if (process.env.CHROMIUM_PATH) return process.env.CHROMIUM_PATH
  const cache = path.join(os.homedir(), 'Library/Caches/ms-playwright')
  const candidates = []
  for (const dir of fs.readdirSync(cache)) {
    if (!/^chromium(_headless_shell)?-\d+$/.test(dir)) continue
    for (const sub of [
      'chrome-headless-shell-mac-arm64/chrome-headless-shell',
      'chrome-mac-arm64/Google Chrome for Testing.app/Contents/MacOS/Google Chrome for Testing',
    ]) {
      const p = path.join(cache, dir, sub)
      if (fs.existsSync(p)) candidates.push(p)
    }
  }
  if (!candidates.length) {
    console.error('No chromium found. Run: npx playwright install chromium  (or set CHROMIUM_PATH)')
    process.exit(2)
  }
  return candidates.sort().at(-1)
}

const browser = await chromium.launch({ executablePath: findChromium() })
const ctx = await browser.newContext({ viewport: { width: 1600, height: 1200 } })
await ctx.addInitScript(() => {
  window.__copies = []
  Object.defineProperty(navigator, 'clipboard', {
    value: { writeText: async (t) => { window.__copies.push(t) } },
    configurable: true,
  })
})
const page = await ctx.newPage()

async function resolveUrl() {
  const urls = process.env.CALC_URL
    ? [process.env.CALC_URL]
    : ['http://localhost:3000/', 'http://localhost:3000/tools/gpu-calculator']
  for (const url of urls) {
    await page.goto(url, { waitUntil: 'networkidle', timeout: 60000 }).catch(() => {})
    await page.waitForTimeout(1500)
    if (await page.getByRole('button', { name: /^JSON$/i }).count()) return url
  }
  console.error('Could not find the calculator (no JSON export button) at', urls.join(' or '))
  process.exit(2)
}

// After Phase 3, controls live inside layers; expanding all restores reachability.
async function expandAllIfPresent() {
  const btn = page.getByRole('button', { name: /expand all/i })
  if (await btn.count()) { await btn.first().click(); await page.waitForTimeout(400) }
}

async function grabExport(kind = 'json') {
  const name = kind === 'text' ? /^Text$/i : /^JSON$/i
  await page.getByRole('button', { name }).first().click()
  await page.waitForTimeout(350)
  const copies = await page.evaluate(() => window.__copies)
  return copies[copies.length - 1]
}

// Pick an option by exact visible label across native selects, with combobox fallback.
async function selectByLabel(optionText) {
  const selects = page.locator('select')
  const n = await selects.count()
  for (let i = 0; i < n; i++) {
    const opts = await selects.nth(i).locator('option').allTextContents()
    if (opts.some((o) => o.trim() === optionText)) {
      await selects.nth(i).selectOption({ label: optionText })
      await page.waitForTimeout(600)
      return
    }
  }
  const combos = page.getByRole('combobox')
  for (let i = 0; i < await combos.count(); i++) {
    await combos.nth(i).click()
    const esc = optionText.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')
    const opt = page.getByRole('option', { name: new RegExp(`^${esc}$`, 'i') })
    if (await opt.count()) { await opt.first().click(); await page.waitForTimeout(600); return }
    await page.keyboard.press('Escape')
  }
  throw new Error(`could not select option: ${optionText} — fix this script's selectors (do NOT skip or baseline-edit)`)
}

// Set a numeric field by accessible-label regex; Enter commits (NumberInput behavior).
async function setNumber(labelRe, value) {
  const re = new RegExp(labelRe, 'i')
  const field = page.getByLabel(re).first()
  if (!(await field.count())) throw new Error(`no field labeled ${labelRe} — fix selectors`)
  await field.fill(String(value))
  await field.press('Enter')
  await page.waitForTimeout(600)
}

const baseUrl = await resolveUrl()
console.log('driving', baseUrl)
const snaps = {}
for (const sc of SCENARIOS) {
  await page.goto(baseUrl, { waitUntil: 'networkidle', timeout: 60000 })
  await page.waitForTimeout(1200)
  for (const step of sc.steps) {
    if (step.tab) {
      // Phase 6 may give the phase tabs role="tab" (D.8 tablist a11y). Try that
      // first, then fall back to the original role="button" locator — the tab's
      // accessible name stays exactly "Pretraining"/"Post-Training" either way.
      const asTab = page.getByRole('tab', { name: step.tab })
      const tabLocator = (await asTab.count())
        ? asTab
        : page.getByRole('button', { name: step.tab })
      await tabLocator.first().click()
      await page.waitForTimeout(900)
      await expandAllIfPresent()
    } else if (step.select) {
      await expandAllIfPresent()
      await selectByLabel(step.select)
    } else if (step.number) {
      await expandAllIfPresent()
      await setNumber(step.number[0], step.number[1])
    }
  }
  if (!sc.steps.length) await expandAllIfPresent()
  snaps[sc.key] = await grabExport(sc.export)
  console.log('captured', sc.key)
}
await browser.close()

fs.writeFileSync(path.join(HERE, 'current-snapshots.json'), JSON.stringify(snaps, null, 2))
const baseline = JSON.parse(fs.readFileSync(BASELINE_PATH, 'utf8'))

if (process.env.WRITE_BASELINE === '1') {
  // One-time pre-redesign use only (already done 2026-06-05). Kept solely so a
  // maintainer on the PRE-redesign commit can re-derive the baseline; refuses to
  // run if the working tree differs from the pinned baseline for existing keys.
  for (const key of Object.keys(baseline)) {
    if (key in snaps && snaps[key] !== baseline[key]) {
      console.error(`WRITE_BASELINE refused: existing key ${key} differs — you are not on the pre-redesign code.`)
      process.exit(2)
    }
  }
  let added = 0
  for (const key of Object.keys(snaps)) {
    if (!(key in baseline)) { baseline[key] = snaps[key]; added++ }
  }
  fs.writeFileSync(BASELINE_PATH, JSON.stringify(baseline, null, 2))
  console.log(`WRITE_BASELINE: appended ${added} new key(s); existing keys untouched`)
}

const allowedDiffs = (process.env.ALLOWED_DIFF_KEYS || '').split(',').filter(Boolean)
// Every scenario must be pinned: a capture without a baseline key guards nothing.
let pass = true
for (const key of Object.keys(snaps)) {
  if (!(key in baseline)) { console.log(`FAIL ${key}: scenario has no baseline key — pin it from the pre-redesign commit before relying on this gate`); pass = false }
}
for (const key of Object.keys(baseline)) {
  if (!(key in snaps)) { console.log(`FAIL ${key}: missing from current capture`); pass = false; continue }
  if (snaps[key] === baseline[key]) { console.log(`PASS ${key}`); continue }
  if (allowedDiffs.includes(key)) { console.log(`ALLOWED-DIFF ${key} (documented intentional change)`); continue }
  pass = false
  const a = JSON.parse(baseline[key])
  const b = JSON.parse(snaps[key])
  const diffs = []
  ;(function walk(x, y, p) {
    if (diffs.length >= 5) return
    if (typeof x !== typeof y) { diffs.push(`${p}: type ${typeof x} -> ${typeof y}`); return }
    if (x && typeof x === 'object') {
      for (const k of new Set([...Object.keys(x), ...Object.keys(y || {})])) walk(x[k], (y || {})[k], `${p}.${k}`)
    } else if (x !== y) diffs.push(`${p}: ${x} -> ${y}`)
  })(a, b, '')
  console.log(`FAIL ${key}: ${diffs.join(' | ')}`)
}
console.log(pass ? 'PARITY: PASS' : 'PARITY: FAIL')
process.exit(pass ? 0 : 1)
