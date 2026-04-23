import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  build: {
    // Build output goes directly into static/ so FastAPI serves it
    outDir: '../static',
    emptyOutDir: true,
  },
  server: {
    // In dev, proxy all FastAPI endpoints to port 8000
    proxy: {
      '/scrape_and_generate': 'http://localhost:8000',
      '/save_preferences': 'http://localhost:8000',
      '/analyze_preferences': 'http://localhost:8000',
      '/posts.json': 'http://localhost:8000',
      '/articles.json': 'http://localhost:8000',
    },
  },
})

