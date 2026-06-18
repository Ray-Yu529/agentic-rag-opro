import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// 開發時把 /api 轉發到 FastAPI 後端 (server.py, :8000)
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: { "/api": "http://127.0.0.1:8000" },
  },
});
