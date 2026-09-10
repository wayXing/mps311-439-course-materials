import { defineConfig } from 'vite'
import { viteSingleFile } from 'vite-plugin-singlefile'

const standalone = process.env.BUILD_AIO === 'true'

function removeManualChunks(config: any) {
  const output = config.build?.rollupOptions?.output
  const outputs = Array.isArray(output) ? output : [output]

  for (const item of outputs) {
    if (item)
      delete item.manualChunks
  }
}

export default defineConfig({
  base: './',
  plugins: standalone
    ? [
        viteSingleFile({ removeViteModuleLoader: true }),
        {
          name: 'slidev-singlefile-compat',
          enforce: 'post',
          config(config) {
            removeManualChunks(config)
          },
        },
      ]
    : [],
})
