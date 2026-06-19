"use client"

// Internal links here are intentionally <a> (hard navigation), not next/link:
// this calculator is a separate multi-zone deployment, so a soft <Link href="/">
// would resolve to THIS zone's own "/" (the calculator) instead of the portfolio
// home. Hard navigation crosses the zone boundary correctly under the proxy.
/* eslint-disable @next/next/no-html-link-for-pages */

// Site navigation header, replicated from the main vinitvyas.ai layout so the
// GPU calculator (a separate multi-zone deployment proxied under
// /tools/gpu-calculator) keeps the same masthead instead of rendering chrome-
// less. Links MUST hard-navigate with <a> — soft-navigating <Link> would try to
// resolve portfolio routes inside this zone's router. Non-sticky on purpose:
// the calculator's own HeroBar/VerdictBand are `sticky top-0`, so a sticky nav
// here would collide with their scroll-collapse behavior.

import { useEffect, useState, useSyncExternalStore } from "react"
import { useTheme } from "next-themes"
import { Moon, Sun } from "lucide-react"

const NAV_LINKS = [
  { href: "/", label: "Writing" },
  { href: "/tools", label: "Tools" },
  { href: "/tags", label: "Topics" },
  { href: "/about", label: "About" },
]

// Sigmoid mark — the single calligraphic activation curve used as the site icon.
function IconMark({ className }: { className?: string }) {
  return (
    <svg
      viewBox="0 0 64 64"
      xmlns="http://www.w3.org/2000/svg"
      className={className}
      aria-hidden="true"
      focusable="false"
      fill="none"
    >
      <path
        d="M 10 54 C 32 54, 32 10, 54 10"
        stroke="currentColor"
        strokeWidth="2"
        strokeLinecap="round"
      />
      <circle cx="10" cy="54" r="3.5" fill="currentColor" />
      <circle cx="54" cy="10" r="3.5" fill="currentColor" />
    </svg>
  )
}

// Minimal icon toggle matching the main site's header (ml-4 baked in for the
// desktop cluster; the mobile-left instance cancels it with -ml-4).
function ThemeToggle() {
  const { resolvedTheme, setTheme } = useTheme()
  // Client-only mount gate without a set-state-in-effect (server snapshot is
  // false, client snapshot true) — matches the calculator's toggle convention.
  const mounted = useSyncExternalStore(
    () => () => {},
    () => true,
    () => false,
  )

  if (!mounted) return <div className="ml-4 h-8 w-8" />

  return (
    <button
      onClick={() => setTheme(resolvedTheme === "dark" ? "light" : "dark")}
      className="ml-4 h-8 w-8 inline-flex items-center justify-center rounded-full text-muted-foreground hover:text-foreground transition-colors duration-300"
      aria-label="Toggle theme"
    >
      {resolvedTheme === "dark" ? (
        <Sun className="h-[15px] w-[15px]" strokeWidth={1.5} />
      ) : (
        <Moon className="h-[15px] w-[15px]" strokeWidth={1.5} />
      )}
    </button>
  )
}

function MobileMenu() {
  const [open, setOpen] = useState(false)

  // Close on Escape.
  useEffect(() => {
    if (!open) return
    const handler = (e: KeyboardEvent) => {
      if (e.key === "Escape") setOpen(false)
    }
    window.addEventListener("keydown", handler)
    return () => window.removeEventListener("keydown", handler)
  }, [open])

  return (
    <>
      <button
        type="button"
        aria-label={open ? "Close menu" : "Open menu"}
        aria-expanded={open}
        aria-controls="mobile-nav-panel"
        onClick={() => setOpen((prev) => !prev)}
        className="md:hidden inline-flex h-8 w-8 items-center justify-center rounded-full text-muted-foreground transition-colors duration-300 hover:text-foreground appearance-none border-0 bg-transparent focus:outline-none focus-visible:outline-1 focus-visible:outline-offset-2 focus-visible:outline-foreground/30"
      >
        <span className="relative block h-[9px] w-[18px]">
          <span
            className={`absolute left-0 right-0 h-px bg-current transition-all duration-300 ease-out ${
              open ? "top-1/2 -translate-y-1/2 rotate-45" : "top-0"
            }`}
          />
          <span
            className={`absolute left-0 right-0 h-px bg-current transition-all duration-300 ease-out ${
              open ? "top-1/2 -translate-y-1/2 -rotate-45" : "bottom-0"
            }`}
          />
        </span>
      </button>

      <div
        id="mobile-nav-panel"
        aria-hidden={!open}
        className={`md:hidden absolute left-0 right-0 top-full z-40 border-b border-border/50 bg-background/95 backdrop-blur-xl transition-all duration-200 ease-out ${
          open
            ? "opacity-100 translate-y-0 pointer-events-auto"
            : "pointer-events-none -translate-y-2 opacity-0"
        }`}
      >
        <nav className="mx-auto flex max-w-7xl flex-col px-5 py-1 sm:px-8 lg:px-10">
          {NAV_LINKS.map((link) => (
            <a
              key={link.href}
              href={link.href}
              tabIndex={open ? 0 : -1}
              className="border-b border-border/30 py-4 text-[17px] tracking-tight text-foreground/80 transition-colors duration-200 hover:text-foreground last:border-b-0"
              style={{ fontFamily: "var(--font-display)", fontWeight: 350 }}
            >
              {link.label}
            </a>
          ))}
        </nav>
      </div>
    </>
  )
}

export default function SiteHeader() {
  return (
    <header className="relative z-50 w-full border-b border-border/50 backdrop-blur-xl supports-[backdrop-filter]:bg-background/80">
      <div className="mx-auto relative flex h-14 max-w-7xl items-center px-5 sm:px-8 lg:px-10">
        {/* Mobile: theme toggle on the left (cancels its built-in ml-4 to sit flush) */}
        <div className="md:hidden -ml-4">
          <ThemeToggle />
        </div>

        {/* Mobile: icon centered as the masthead anchor */}
        <a
          href="/"
          aria-label="Vinit Vyas — Home"
          className="md:hidden absolute left-1/2 top-1/2 -translate-x-1/2 -translate-y-1/2"
        >
          <IconMark className="h-5 w-5 text-foreground/70" />
        </a>

        {/* Desktop: icon + wordmark, left-aligned */}
        <div className="mr-4 hidden md:flex">
          <a className="flex items-center gap-2.5" href="/">
            <IconMark className="h-5 w-5 text-foreground/70" />
            <span
              className="text-sm tracking-[0.12em] uppercase text-foreground/70"
              style={{ fontFamily: "var(--font-display)", fontWeight: 350 }}
            >
              Vinit Vyas
            </span>
          </a>
        </div>

        {/* Right cluster */}
        <div className="ml-auto flex items-center">
          <nav className="hidden md:flex items-center gap-6 text-[13px]">
            {NAV_LINKS.map((link) => (
              <a key={link.href} href={link.href} className="nav-link">
                {link.label}
              </a>
            ))}
          </nav>
          {/* Desktop: theme toggle on the right (mobile has its own on the left) */}
          <div className="hidden md:block">
            <ThemeToggle />
          </div>
          {/* Mobile: hamburger on the right (component handles md:hidden internally) */}
          <MobileMenu />
        </div>
      </div>
    </header>
  )
}
