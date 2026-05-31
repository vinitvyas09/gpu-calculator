"use client"

import { useSyncExternalStore } from "react"
import { useTheme } from "next-themes"
import { Moon, Sun } from "lucide-react"

export default function ThemeToggle() {
  const { resolvedTheme, setTheme } = useTheme()
  const mounted = useSyncExternalStore(
    () => () => {},
    () => true,
    () => false,
  )

  if (!mounted) {
    return <div className="h-9 w-9" />
  }

  const isDark = resolvedTheme === "dark"

  return (
    <button
      type="button"
      onClick={() => setTheme(isDark ? "light" : "dark")}
      className="no-theme-transition flex h-9 w-9 items-center justify-center rounded-xl border border-border bg-surface-elevated/70 text-muted backdrop-blur-sm hover:text-accent hover:border-accent/30 hover:-translate-y-px hover:shadow-sm focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-accent/40"
      style={{ transition: "transform 200ms cubic-bezier(0.22, 1, 0.36, 1), box-shadow 200ms ease, color 200ms ease, border-color 200ms ease" }}
      aria-label={isDark ? "Switch to light mode" : "Switch to dark mode"}
    >
      {isDark ? <Sun className="h-[18px] w-[18px]" /> : <Moon className="h-[18px] w-[18px]" />}
    </button>
  )
}
