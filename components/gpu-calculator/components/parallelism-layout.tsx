"use client"

import { Fragment } from "react"
import { motion } from "framer-motion"
import type { ParallelismConfig } from "../types"

const MAX_VISIBLE = {
  dp: 4,
  pp: 4,
  tp: 6,
  ep: 3,
} as const

const DIMENSION_META = {
  dp: {
    label: "DP",
    name: "Data Parallel",
    light: "oklch(0.55 0.145 180)",
    dark: "oklch(0.72 0.12 180)",
  },
  tp: {
    label: "TP",
    name: "Tensor Parallel",
    light: "oklch(0.74 0.14 80)",
    dark: "oklch(0.82 0.12 80)",
  },
  cp: {
    label: "CP",
    name: "Context Parallel",
    light: "oklch(0.56 0.12 230)",
    dark: "oklch(0.72 0.10 230)",
  },
  pp: {
    label: "PP",
    name: "Pipeline Parallel",
    light: "oklch(0.60 0.13 150)",
    dark: "oklch(0.74 0.10 150)",
  },
  ep: {
    label: "EP",
    name: "Expert Parallel",
    light: "oklch(0.60 0.16 320)",
    dark: "oklch(0.72 0.14 320)",
  },
} as const

interface Props {
  config: ParallelismConfig
  isDark: boolean
}

function normalizeDegree(value: number): number {
  return Number.isFinite(value) && value > 0 ? Math.floor(value) : 1
}

function range(length: number): number[] {
  return Array.from({ length }, (_, index) => index)
}

function getDisplayLabel(prefix: string, index: number, visible: number, total: number) {
  const humanIndex = index + 1

  if (index === visible - 1 && visible < total) {
    return `${prefix}${humanIndex}+`
  }

  return `${prefix}${humanIndex}`
}

function getWorldSize(config: ParallelismConfig): number {
  return (
    normalizeDegree(config.N_dp) *
    normalizeDegree(config.N_tp) *
    normalizeDegree(config.N_cp) *
    normalizeDegree(config.N_pp) *
    normalizeDegree(config.N_ep)
  )
}

function LegendChip({
  label,
  value,
  color,
  isDark,
}: {
  label: string
  value: number
  color: string
  isDark: boolean
}) {
  return (
    <div
      className="inline-flex items-center gap-2 rounded-full border px-3 py-1 text-xs"
      style={{
        borderColor: isDark ? "oklch(0.36 0.012 260)" : "oklch(0.90 0.008 80)",
        backgroundColor: isDark ? "oklch(0.22 0.011 260)" : "oklch(0.98 0.004 80)",
      }}
    >
      <span
        className="inline-flex h-5 min-w-5 items-center justify-center rounded-full px-1 font-mono text-[10px] font-semibold"
        style={{
          backgroundColor: color,
          color: isDark ? "oklch(0.16 0.009 260)" : "oklch(0.99 0.002 80)",
        }}
      >
        {label}
      </span>
      <span className="font-mono tabular-nums text-foreground">{value}</span>
    </div>
  )
}

export default function ParallelismLayout({ config, isDark }: Props) {
  const degrees = {
    dp: normalizeDegree(config.N_dp),
    tp: normalizeDegree(config.N_tp),
    cp: normalizeDegree(config.N_cp),
    pp: normalizeDegree(config.N_pp),
    ep: normalizeDegree(config.N_ep),
  }
  const visible = {
    dp: Math.min(degrees.dp, MAX_VISIBLE.dp),
    pp: Math.min(degrees.pp, MAX_VISIBLE.pp),
    tp: Math.min(degrees.tp, MAX_VISIBLE.tp),
    ep: Math.min(degrees.ep, MAX_VISIBLE.ep),
  }
  const worldSize = getWorldSize(config)
  const hasTruncation =
    visible.dp < degrees.dp ||
    visible.pp < degrees.pp ||
    visible.tp < degrees.tp ||
    visible.ep < degrees.ep
  const cellBorder = isDark ? "oklch(0.34 0.012 260)" : "oklch(0.90 0.008 80)"
  const panelBackground = isDark ? "oklch(0.22 0.011 260)" : "oklch(0.98 0.004 80)"
  const stageBackground = isDark ? "oklch(0.25 0.012 260)" : "oklch(0.995 0.002 80)"
  const tileTextColor = isDark ? "oklch(0.16 0.009 260)" : "oklch(0.995 0.002 80)"
  const topologyLabel =
    `DP ${degrees.dp} x TP ${degrees.tp}` +
    (degrees.cp > 1 ? ` x CP ${degrees.cp}` : "") +
    ` x PP ${degrees.pp}` +
    (degrees.ep > 1 ? ` x EP ${degrees.ep}` : "")

  return (
    <div className="space-y-4">
      <div className="flex flex-col gap-3 lg:flex-row lg:items-start lg:justify-between">
        <div>
          <div className="font-mono text-sm font-semibold text-foreground">
            {topologyLabel}
          </div>
          <p className="mt-1 text-xs text-muted">
            {worldSize.toLocaleString()} total GPUs in the configured world size.
          </p>
        </div>

        <div className="flex flex-wrap gap-2">
          <LegendChip
            label={DIMENSION_META.dp.label}
            value={degrees.dp}
            color={DIMENSION_META.dp[isDark ? "dark" : "light"]}
            isDark={isDark}
          />
          <LegendChip
            label={DIMENSION_META.tp.label}
            value={degrees.tp}
            color={DIMENSION_META.tp[isDark ? "dark" : "light"]}
            isDark={isDark}
          />
          {degrees.cp > 1 && (
            <LegendChip
              label={DIMENSION_META.cp.label}
              value={degrees.cp}
              color={DIMENSION_META.cp[isDark ? "dark" : "light"]}
              isDark={isDark}
            />
          )}
          <LegendChip
            label={DIMENSION_META.pp.label}
            value={degrees.pp}
            color={DIMENSION_META.pp[isDark ? "dark" : "light"]}
            isDark={isDark}
          />
          {degrees.ep > 1 && (
            <LegendChip
              label={DIMENSION_META.ep.label}
              value={degrees.ep}
              color={DIMENSION_META.ep[isDark ? "dark" : "light"]}
              isDark={isDark}
            />
          )}
        </div>
      </div>

      <div className="space-y-3">
        {range(visible.ep).map((epIndex) => (
          <motion.section
            key={epIndex}
            className="rounded-xl border p-3"
            style={{
              borderColor:
                degrees.ep > 1
                  ? DIMENSION_META.ep[isDark ? "dark" : "light"]
                  : cellBorder,
              backgroundColor: panelBackground,
            }}
            initial={{ opacity: 0, y: 8 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: epIndex * 0.04, duration: 0.3 }}
          >
            {degrees.ep > 1 && (
              <div className="mb-3 flex items-center justify-between gap-3">
                <div
                  className="font-mono text-xs font-semibold uppercase tracking-[0.16em]"
                  style={{ color: DIMENSION_META.ep[isDark ? "dark" : "light"] }}
                >
                  {getDisplayLabel("EP ", epIndex, visible.ep, degrees.ep)}
                </div>
                <div className="text-[11px] text-muted">
                  {(
                    degrees.dp *
                    degrees.pp *
                    degrees.tp *
                    degrees.cp
                  ).toLocaleString()}{" "}
                  GPUs in this expert group
                </div>
              </div>
            )}

            <div
              className="grid gap-2"
              style={{
                gridTemplateColumns: `auto repeat(${visible.pp}, minmax(0, 1fr))`,
              }}
            >
              <div />

              {range(visible.pp).map((ppIndex) => (
                <div
                  key={`pp-header-${ppIndex}`}
                  className="px-2 text-center font-mono text-[10px] font-semibold uppercase tracking-[0.14em]"
                  style={{ color: DIMENSION_META.pp[isDark ? "dark" : "light"] }}
                >
                  {getDisplayLabel("PP ", ppIndex, visible.pp, degrees.pp)}
                </div>
              ))}

              {range(visible.dp).map((dpIndex) => (
                <Fragment key={dpIndex}>
                  <div
                    className="pr-2 text-right font-mono text-[10px] font-semibold uppercase tracking-[0.14em]"
                    style={{ color: DIMENSION_META.dp[isDark ? "dark" : "light"] }}
                  >
                    {getDisplayLabel("DP ", dpIndex, visible.dp, degrees.dp)}
                  </div>

                  {range(visible.pp).map((ppIndex) => (
                    <motion.div
                      key={`${epIndex}-${dpIndex}-${ppIndex}`}
                      className="relative overflow-hidden rounded-xl border p-2"
                      style={{
                        borderColor: cellBorder,
                        backgroundColor: stageBackground,
                      }}
                      initial={{ opacity: 0, scale: 0.98 }}
                      animate={{ opacity: 1, scale: 1 }}
                      transition={{
                        delay: (epIndex * visible.dp * visible.pp + dpIndex * visible.pp + ppIndex) * 0.02,
                        duration: 0.26,
                      }}
                    >
                      <div
                        className="absolute inset-x-0 top-0 h-1"
                        style={{
                          backgroundColor: DIMENSION_META.pp[isDark ? "dark" : "light"],
                        }}
                      />
                      <div
                        className="absolute inset-y-0 left-0 w-1"
                        style={{
                          backgroundColor: DIMENSION_META.dp[isDark ? "dark" : "light"],
                        }}
                      />
                      {degrees.ep > 1 && (
                        <div
                          className="absolute inset-y-0 right-0 w-1"
                          style={{
                            backgroundColor:
                              DIMENSION_META.ep[isDark ? "dark" : "light"],
                          }}
                        />
                      )}

                      <div className="relative pl-2 pt-1">
                        <div className="mb-2 flex items-center justify-between gap-2">
                          <div className="font-mono text-[10px] font-semibold uppercase tracking-[0.14em] text-foreground">
                            TP lanes
                          </div>
                          {degrees.cp > 1 && (
                            <div
                              className="rounded-full border px-2 py-0.5 font-mono text-[9px] font-semibold uppercase tracking-[0.12em]"
                              style={{
                                borderColor:
                                  DIMENSION_META.cp[isDark ? "dark" : "light"],
                                color: DIMENSION_META.cp[isDark ? "dark" : "light"],
                              }}
                            >
                              CP x{degrees.cp}
                            </div>
                          )}
                        </div>

                        <div className="flex flex-wrap gap-1">
                          {range(visible.tp).map((tpIndex) => (
                            <div
                              key={tpIndex}
                              className="flex h-6 w-6 items-center justify-center rounded-md border text-[9px] font-semibold"
                              style={{
                                borderColor: cellBorder,
                                backgroundColor:
                                  DIMENSION_META.tp[isDark ? "dark" : "light"],
                                color: tileTextColor,
                              }}
                              title={`EP ${epIndex + 1}, DP ${dpIndex + 1}, PP ${ppIndex + 1}, TP ${tpIndex + 1}${degrees.cp > 1 ? `, CP x${degrees.cp}` : ""}`}
                            >
                              {tpIndex + 1}
                            </div>
                          ))}

                          {visible.tp < degrees.tp && (
                            <div
                              className="flex h-6 min-w-6 items-center justify-center rounded-md border px-1 text-[9px] font-semibold"
                              style={{
                                borderColor: cellBorder,
                                backgroundColor: isDark
                                  ? "oklch(0.28 0.012 260)"
                                  : "oklch(0.95 0.005 80)",
                                color: isDark
                                  ? "oklch(0.63 0.015 260)"
                                  : "oklch(0.50 0.014 260)",
                              }}
                            >
                              +{degrees.tp - visible.tp}
                            </div>
                          )}
                        </div>
                      </div>
                    </motion.div>
                  ))}
                </Fragment>
              ))}
            </div>
          </motion.section>
        ))}
      </div>

      {degrees.cp > 1 && (
        <div className="rounded-xl border border-border bg-surface-elevated/50 px-3 py-2 text-xs text-muted">
          Context parallelism is applied across every tile. The spatial grid focuses on
          DP x TP x PP{degrees.ep > 1 ? " x EP" : ""}, while each TP lane spans a CP x
          {degrees.cp} sequence-sharding group.
        </div>
      )}

      {hasTruncation && (
        <p className="text-[11px] text-muted">
          Showing{" "}
          {[
            visible.dp < degrees.dp && `${visible.dp} of ${degrees.dp} DP rows`,
            visible.pp < degrees.pp && `${visible.pp} of ${degrees.pp} PP stages`,
            visible.tp < degrees.tp && `${visible.tp} of ${degrees.tp} TP lanes`,
            visible.ep < degrees.ep && `${visible.ep} of ${degrees.ep} EP groups`,
          ]
            .filter(Boolean)
            .join(", ")}
          .
        </p>
      )}
    </div>
  )
}
