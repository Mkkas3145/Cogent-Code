import { build } from "esbuild";
import { mkdir, rm } from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";

const currentDir = path.dirname(fileURLToPath(import.meta.url));
const repoRoot = path.resolve(currentDir, "..");
const outputDir = path.join(repoRoot, "apps", "dist-release", "electron-main");

await rm(outputDir, { recursive: true, force: true });
await mkdir(outputDir, { recursive: true });

await build({
  entryPoints: [path.join(repoRoot, "apps", "desktop", "electron-main", "main.ts")],
  outfile: path.join(outputDir, "main.js"),
  bundle: true,
  format: "esm",
  platform: "node",
  target: "node20",
  sourcemap: false,
  minify: false,
  external: ["electron"],
  legalComments: "none",
});
