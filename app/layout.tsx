import type { Metadata } from "next"
import { ThemeProvider } from "next-themes"
import "./globals.css"

export const metadata: Metadata = {
  title: {
    default: "LLM Training GPU Calculator",
    template: "%s | LLM Training GPU Calculator",
  },
  description:
    "Estimate GPU requirements for LLM training — memory breakdown, parallelism strategy, training time, and cost.",
}

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode
}>) {
  return (
    <html lang="en" suppressHydrationWarning className="h-full" style={{ colorScheme: "light dark" }}>
      <body className="min-h-full bg-background text-foreground font-sans antialiased">
        <ThemeProvider
          attribute="class"
          defaultTheme="system"
          enableSystem
          disableTransitionOnChange
        >
          {children}
        </ThemeProvider>
      </body>
    </html>
  )
}
