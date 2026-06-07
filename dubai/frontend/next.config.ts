import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  // pin the workspace root; otherwise Next picks up a stray lockfile higher up
  turbopack: { root: __dirname },
};

export default nextConfig;
