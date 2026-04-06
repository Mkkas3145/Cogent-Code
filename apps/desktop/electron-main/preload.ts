import { contextBridge, ipcRenderer } from "electron";
import type {
  AgentStreamEvent,
  BrowserAssistResult,
  ContextSnapshot,
  ContextUsageSnapshot,
  GeminiModelSummary,
  LmStudioModelSummary,
  OllamaModelSummary,
  RunCommandRequest,
  TaskRequest,
} from "@cogent/shared-types";
import { randomUUID } from "node:crypto";

contextBridge.exposeInMainWorld("cogent", {
  runAgentTask: async (request: TaskRequest): Promise<AgentStreamEvent[]> =>
    ipcRenderer.invoke("agent:run-task", request),
  runAgentTaskStream: (
    request: TaskRequest,
    onEvent: (event: AgentStreamEvent) => void,
  ): { requestId: string; completed: Promise<void> } => {
    const requestId = randomUUID();
    const eventChannel = `agent:run-task-stream:${requestId}`;
    const doneChannel = `agent:run-task-stream:${requestId}:done`;

    const completed = new Promise<void>((resolve) => {
      const handleEvent = (_event: Electron.IpcRendererEvent, payload: AgentStreamEvent) => {
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
  cancelAgentTaskStream: async (requestId: string): Promise<{ canceled: boolean }> =>
    ipcRenderer.invoke("agent:cancel-run-task-stream", requestId),
  completeBrowserAssist: async (payload: BrowserAssistResult): Promise<{ completed: boolean }> =>
    ipcRenderer.invoke("agent:browser-assist-complete", payload),
  cancelBrowserAssist: async (requestId: string): Promise<{ canceled: boolean }> =>
    ipcRenderer.invoke("agent:browser-assist-cancel", requestId),
  buildContextSnapshot: async (request: TaskRequest): Promise<ContextSnapshot> =>
    ipcRenderer.invoke("agent:context", request),
  buildContextUsageSnapshot: async (request: TaskRequest): Promise<ContextUsageSnapshot | null> =>
    ipcRenderer.invoke("agent:context-usage", request),
  runCommand: async (request: RunCommandRequest): Promise<AgentStreamEvent[]> =>
    ipcRenderer.invoke("command:run", request),
  getGpuStatus: async (): Promise<unknown> => ipcRenderer.invoke("system:gpu-status"),
  getMemoryInfo: async (): Promise<{ totalMemoryBytes: number; freeMemoryBytes: number }> => ipcRenderer.invoke("system:memory-info"),
  getGeminiModels: async (apiKey: string): Promise<{ available: boolean; models: GeminiModelSummary[]; error?: string }> =>
    ipcRenderer.invoke("system:gemini-models", apiKey),
  getOllamaModels: async (): Promise<{ available: boolean; models: OllamaModelSummary[]; error?: string }> =>
    ipcRenderer.invoke("system:ollama-models"),
  getLmStudioModels: async (): Promise<{ available: boolean; models: LmStudioModelSummary[]; error?: string }> =>
    ipcRenderer.invoke("system:lmstudio-models"),
  cleanupLocalModels: async (payload: { provider?: "ollama" | "lmstudio" | "gemini"; modelId?: string }): Promise<{ cleaned: boolean }> =>
    ipcRenderer.invoke("system:cleanup-local-models", payload),
  getWorkspaceInfo: async (): Promise<{ rootPath: string; name: string }> => ipcRenderer.invoke("system:workspace-info"),
  openFolder: async (): Promise<{ rootPath: string; name: string } | null> => ipcRenderer.invoke("system:open-folder"),
  openPath: async (targetPath: string) => ipcRenderer.invoke("system:open-path", targetPath),
  openNewWindow: async (): Promise<{ opened: boolean }> => ipcRenderer.invoke("system:new-window"),
  openConsoleWindow: async (): Promise<{ opened: boolean }> => ipcRenderer.invoke("system:open-console-window"),
  getFileTree: async () => ipcRenderer.invoke("system:file-tree"),
  readFile: async (filePath: string) => ipcRenderer.invoke("system:read-file", filePath),
  writeFile: async (request: { filePath: string; content: string }) => ipcRenderer.invoke("system:write-file", request),
  windowControls: {
    minimize: async (): Promise<void> => ipcRenderer.invoke("window:minimize"),
    maximizeToggle: async (): Promise<{ maximized: boolean }> => ipcRenderer.invoke("window:maximize-toggle"),
    close: async (): Promise<void> => ipcRenderer.invoke("window:close"),
    getState: async (): Promise<{ maximized: boolean }> => ipcRenderer.invoke("window:get-state"),
  },
});
