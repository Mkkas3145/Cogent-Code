"use strict";

const { contextBridge, ipcRenderer } = require("electron");
const { randomUUID } = require("node:crypto");

contextBridge.exposeInMainWorld("cogent", {
  runAgentTask: async (request) => ipcRenderer.invoke("agent:run-task", request),
  runAgentTaskStream: (request, onEvent) => {
    const requestId = randomUUID();
    const eventChannel = `agent:run-task-stream:${requestId}`;
    const doneChannel = `agent:run-task-stream:${requestId}:done`;

    const completed = new Promise((resolve) => {
      const handleEvent = (_event, payload) => {
        onEvent(payload);
      };

      const handleDone = () => {
        ipcRenderer.removeListener(eventChannel, handleEvent);
        resolve();
      };

      ipcRenderer.on(eventChannel, handleEvent);
      ipcRenderer.once(doneChannel, handleDone);
      ipcRenderer.send("agent:run-task-stream", { requestId, request });
    });

    return { requestId, completed };
  },
  cancelAgentTaskStream: async (requestId) => ipcRenderer.invoke("agent:cancel-run-task-stream", requestId),
  completeBrowserAssist: async (payload) => ipcRenderer.invoke("agent:browser-assist-complete", payload),
  cancelBrowserAssist: async (requestId) => ipcRenderer.invoke("agent:browser-assist-cancel", requestId),
  buildContextSnapshot: async (request) => ipcRenderer.invoke("agent:context", request),
  buildContextUsageSnapshot: async (request) => ipcRenderer.invoke("agent:context-usage", request),
  runCommand: async (request) => ipcRenderer.invoke("command:run", request),
  getGpuStatus: async () => ipcRenderer.invoke("system:gpu-status"),
  getMemoryInfo: async () => ipcRenderer.invoke("system:memory-info"),
  getGeminiModels: async (apiKey) => ipcRenderer.invoke("system:gemini-models", apiKey),
  getOllamaModels: async () => ipcRenderer.invoke("system:ollama-models"),
  getLmStudioModels: async () => ipcRenderer.invoke("system:lmstudio-models"),
  cleanupLocalModels: async (payload) => ipcRenderer.invoke("system:cleanup-local-models", payload),
  getWorkspaceInfo: async () => ipcRenderer.invoke("system:workspace-info"),
  openFolder: async () => ipcRenderer.invoke("system:open-folder"),
  openPath: async (targetPath) => ipcRenderer.invoke("system:open-path", targetPath),
  openNewWindow: async () => ipcRenderer.invoke("system:new-window"),
  openConsoleWindow: async () => ipcRenderer.invoke("system:open-console-window"),
  getFileTree: async () => ipcRenderer.invoke("system:file-tree"),
  readFile: async (filePath) => ipcRenderer.invoke("system:read-file", filePath),
  writeFile: async (request) => ipcRenderer.invoke("system:write-file", request),
  windowControls: {
    minimize: async () => ipcRenderer.invoke("window:minimize"),
    maximizeToggle: async () => ipcRenderer.invoke("window:maximize-toggle"),
    close: async () => ipcRenderer.invoke("window:close"),
    getState: async () => ipcRenderer.invoke("window:get-state"),
  },
});
