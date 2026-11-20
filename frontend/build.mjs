/**
 * Frontend build using the locally installed esbuild binary.
 *
 * Self-contained on purpose: this app must build and run without reaching
 * outside frontend/.
 *
 * The esbuild *binary* is invoked directly rather than through esbuild's JS API:
 * the JS API spawns a long-lived child process for the service protocol, which is
 * blocked in restricted shells, while a one-shot CLI build is not. That also keeps
 * the build a single deterministic step.
 *
 * Usage:
 *   node build.mjs
 */
import { execFileSync } from "node:child_process";
import { cpSync, existsSync, mkdirSync, statSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const here = dirname(fileURLToPath(import.meta.url));
const dist = join(here, "dist");
mkdirSync(dist, { recursive: true });

/** Resolve the platform esbuild binary shipped by the @esbuild/* dependency. */
function esbuildBinary() {
  const candidates = [
    join(here, "node_modules", "@esbuild", "win32-x64", "esbuild.exe"),
    join(here, "node_modules", "@esbuild", "darwin-arm64", "esbuild"),
    join(here, "node_modules", "@esbuild", "linux-x64", "esbuild"),
  ];
  for (const c of candidates) if (existsSync(c)) return c;
  throw new Error(
    "找不到 esbuild 可执行文件。请先在 frontend/ 目录运行：npm install --registry=https://registry.npmmirror.com",
  );
}

const args = [
  join(here, "src", "main.tsx"),
  "--bundle",
  `--outfile=${join(dist, "bundle.js")}`,
  "--platform=browser",
  "--target=es2020",
  "--format=iife",
  "--jsx=automatic",
  "--minify",
  // Must be a quoted string literal; an unquoted value is treated as an
  // identifier and silently disables minification's dead-code elimination.
  `--define:process.env.NODE_ENV="production"`,
  "--loader:.css=css",
  // Emit Chinese as-is rather than escaping every codepoint to \uXXXX. Without
  // this the bundle contains no readable text, which made "is this string
  // present" checks useless and inflated the output size.
  "--charset=utf8",
  "--log-level=warning",
];

console.log("  bundling...");
execFileSync(esbuildBinary(), args, { stdio: "inherit", cwd: here });

// Copy the HTML entry next to the bundle.
const indexSrc = join(here, "index.html");
if (existsSync(indexSrc)) cpSync(indexSrc, join(dist, "index.html"));

// Copy public/ if the project grows one later.
const publicDir = join(here, "public");
if (existsSync(publicDir) && statSync(publicDir).isDirectory()) {
  cpSync(publicDir, dist, { recursive: true });
}

const bundle = join(dist, "bundle.js");
console.log(`\n  done → dist/ (bundle.js ${(statSync(bundle).size / 1024).toFixed(1)} KB)`);
