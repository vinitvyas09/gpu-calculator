"use client"

import { useCallback, useMemo, useSyncExternalStore } from "react"

// ---------------------------------------------------------------------------
// usePersistedState — SSR-safe localStorage-backed state via useSyncExternalStore.
//
// Mirrors the existing useSyncExternalStore mount-flag usage in
// gpu-calculator.tsx so server snapshot === `initial` and the client snapshot
// reads the stored value; React reconciles on hydration without a flash.
//
//   - subscribe: listens for cross-tab `storage` events (and same-tab writes,
//     which we re-broadcast via dispatchEvent below).
//   - getSnapshot: raw localStorage string (or null when absent / unavailable).
//   - getServerSnapshot: constant `null` ⇒ SSR markup uses `initial`.
//   - the value is JSON-parsed with a safe fallback to `initial`.
//
// Never read localStorage during render outside this hook.
// ---------------------------------------------------------------------------

function safeParse<T>(raw: string, fallback: T): T {
  try {
    return JSON.parse(raw) as T
  } catch {
    return fallback
  }
}

export function usePersistedState<T>(
  key: string,
  initial: T,
): [T, (next: T) => void] {
  const subscribe = useCallback(
    (cb: () => void) => {
      const handler = (e: StorageEvent) => {
        if (e.key === key) cb()
      }
      window.addEventListener("storage", handler) // cross-tab + same-tab sync
      return () => window.removeEventListener("storage", handler)
    },
    [key],
  )

  const getSnapshot = useCallback((): string | null => {
    try {
      return window.localStorage.getItem(key)
    } catch {
      return null
    }
  }, [key])

  // CRITICAL: server snapshot is constant ⇒ markup matches first client paint.
  const getServerSnapshot = useCallback(() => null, [])

  const raw = useSyncExternalStore(subscribe, getSnapshot, getServerSnapshot)
  const value = useMemo<T>(
    () => (raw === null ? initial : safeParse(raw, initial)),
    [raw, initial],
  )

  const set = useCallback(
    (next: T) => {
      try {
        window.localStorage.setItem(key, JSON.stringify(next))
      } catch {
        // ignore quota / unavailable storage
      }
      // Notify this tab's subscribers (the `storage` event only fires in OTHER
      // tabs natively).
      window.dispatchEvent(new StorageEvent("storage", { key }))
    },
    [key],
  )

  return [value, set]
}
