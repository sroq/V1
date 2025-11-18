/** @type {import('next').NextConfig} */
const nextConfig = {
  experimental: {
    serverActions: {
      bodySizeLimit: '2mb',
    },
  },
  // Enable OpenTelemetry instrumentation (Next.js 15+)
  instrumentation: true,
}

module.exports = nextConfig
