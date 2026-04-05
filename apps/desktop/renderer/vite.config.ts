import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import path from "node:path";

export default defineConfig({
  base: "./",
  plugins: [react()],
  worker: {
    format: "es",
  },
  resolve: {
    alias: {
      "@cogent/shared-types": path.resolve(__dirname, "../../../packages/shared-types/src"),
    },
  },
  optimizeDeps: {
    exclude: [
      "@codingame/monaco-vscode-api",
      "@codingame/monaco-vscode-configuration-service-override",
      "@codingame/monaco-vscode-extensions-service-override",
      "@codingame/monaco-vscode-files-service-override",
      "@codingame/monaco-vscode-languages-service-override",
      "@codingame/monaco-vscode-textmate-service-override",
      "@codingame/monaco-vscode-theme-service-override",
      "@codingame/monaco-vscode-theme-defaults-default-extension",
      "@codingame/monaco-vscode-javascript-default-extension",
      "@codingame/monaco-vscode-json-default-extension",
      "@codingame/monaco-vscode-html-default-extension",
      "@codingame/monaco-vscode-css-default-extension",
      "@codingame/monaco-vscode-typescript-basics-default-extension",
      "@codingame/monaco-vscode-typescript-language-features-default-extension",
      "monaco-editor",
    ],
  },
  server: {
    port: 5173,
    strictPort: true,
  },
});
