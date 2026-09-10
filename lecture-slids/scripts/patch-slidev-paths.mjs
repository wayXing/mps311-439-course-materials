import { existsSync, readFileSync, readdirSync, writeFileSync } from 'node:fs'
import { dirname, join } from 'node:path'
import { fileURLToPath } from 'node:url'

const projectRoot = dirname(dirname(fileURLToPath(import.meta.url)))
const distDir = join(projectRoot, 'node_modules', '@slidev', 'cli', 'dist')

if (!existsSync(distDir)) {
  console.warn('[patch-slidev-paths] @slidev/cli is not installed; skipping')
  process.exit(0)
}

const original = 'baseInDev + encodeURI(toAtFS(join(clientRoot, "main.ts")))'
const patched = 'baseInDev + toAtFS(join(clientRoot, "main.ts"))'
let found = false

for (const name of readdirSync(distDir)) {
  if (!name.startsWith('shared-') || !name.endsWith('.mjs'))
    continue

  const file = join(distDir, name)
  const source = readFileSync(file, 'utf8')

  if (source.includes(patched)) {
    found = true
    continue
  }

  if (source.includes(original)) {
    writeFileSync(file, source.replace(original, patched))
    console.log(`[patch-slidev-paths] patched ${name}`)
    found = true
  }
}

if (!found) {
  console.warn('[patch-slidev-paths] target code was not found; check the installed Slidev version')
}
