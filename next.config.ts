import type { NextConfig } from "next"

const nextConfig: NextConfig = {
  // Multi-zone: vinitvyas.com proxies /tools/gpu-calculator[/...] to this
  // deployment (page → /, assets path-preserved). The calculator itself
  // renders at / — e.g. gpu-calculator-amber.vercel.app/ — while assetPrefix
  // namespaces every _next asset under /tools/gpu-calculator so the proxied
  // page's asset URLs never collide with the blog's own /_next files.
  assetPrefix: "/tools/gpu-calculator",
  async rewrites() {
    return [
      // Serve the prefixed asset URLs on direct (vercel.app) traffic too.
      {
        source: "/tools/gpu-calculator/_next/:path*",
        destination: "/_next/:path*",
      },
    ]
  },
  experimental: {
    optimizePackageImports: ["lucide-react", "framer-motion"],
  },
  turbopack: {
    root: process.cwd(),
  },
}

export default nextConfig
