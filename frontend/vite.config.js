import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    host: '0.0.0.0',
    // Docker Desktop on Windows bind-mounts this dir from the D: drive;
    // native filesystem change events (inotify) don't propagate through
    // that bridge into the container, so chokidar's default watcher never
    // fires. Polling forces Vite to actually notice edits.
    watch: { usePolling: true, interval: 300 },
  },
})
