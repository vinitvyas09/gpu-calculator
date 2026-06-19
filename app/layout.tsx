import type { Metadata } from "next"
import { Inter, Fraunces, JetBrains_Mono } from "next/font/google"
import { ThemeProvider } from "next-themes"
import SiteHeader from "@/components/site-header"
import "./globals.css"

const inter = Inter({
  subsets: ["latin"],
  variable: "--font-sans",
  display: "swap",
})

const fraunces = Fraunces({
  subsets: ["latin"],
  variable: "--font-display",
  display: "swap",
  axes: ["SOFT", "WONK", "opsz"],
})

const jetbrainsMono = JetBrains_Mono({
  subsets: ["latin"],
  variable: "--font-mono",
  display: "swap",
})

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
    <html
      lang="en"
      suppressHydrationWarning
      className={`${inter.variable} ${fraunces.variable} ${jetbrainsMono.variable} h-full`}
      style={{ colorScheme: "light dark" }}
    >
      <body className="min-h-full bg-background text-foreground font-sans antialiased">
        <ThemeProvider
          attribute="class"
          defaultTheme="light"
          enableSystem
          disableTransitionOnChange
        >
          <SiteHeader />
          {children}
        </ThemeProvider>
      </body>
    </html>
  )
}
