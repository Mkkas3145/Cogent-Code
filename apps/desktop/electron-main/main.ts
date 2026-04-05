import { app, BrowserWindow, dialog, ipcMain } from "electron";
import { mkdir, readFile, readdir, rmdir, stat, unlink, writeFile } from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { buildContextSnapshot, buildContextUsageSnapshot, runAgentTask } from "@cogent/agent-core";
import { streamCommand } from "@cogent/command-runtime";
import type {
  AgentStreamEvent,
  BrowserAssistRequest,
  BrowserAssistResult,
  CommandChunk,
  ContextUsageSnapshot,
  OllamaModelSummary,
  RunCommandRequest,
  TaskRequest,
} from "@cogent/shared-types";

// Prefer the GPU compositor path so backdrop-filter and glass surfaces work more reliably.
app.commandLine.appendSwitch("ignore-gpu-blocklist");
app.commandLine.appendSwitch("enable-gpu-rasterization");
app.commandLine.appendSwitch("enable-zero-copy");

const isDev = !app.isPackaged;
const currentDir = path.dirname(fileURLToPath(import.meta.url));
const workspaceRoot = process.env.INIT_CWD || (app.isPackaged ? app.getPath("documents") : process.cwd());
const windowWorkspaceRoots = new Map<number, string>();
const activeAgentRuns = new Map<string, AbortController>();
const activeOllamaRequestCounts = new Map<string, number>();
const knownOllamaModels = new Set<string>();
const pendingBrowserAssistRequests = new Map<
  string,
  {
    senderId: number;
    resolve: (result: BrowserAssistResult) => void;
    reject: (error: Error) => void;
  }
>();
const preloadPath = isDev
  ? path.resolve(currentDir, "..", "..", "..", "..", "desktop", "electron-main", "preload.cjs")
  : path.join(app.getAppPath(), "apps", "desktop", "electron-main", "preload.cjs");

function delay(ms: number) {
  return new Promise<void>((resolve) => {
    setTimeout(resolve, ms);
  });
}

async function waitForPageSettled(webContents: Electron.WebContents, options?: { minWaitMs?: number; maxWaitMs?: number; stableRounds?: number }) {
  const minWaitMs = options?.minWaitMs ?? 1200;
  const maxWaitMs = options?.maxWaitMs ?? 20000;
  const stableRounds = options?.stableRounds ?? 3;
  const pollMs = 500;
  const start = Date.now();
  let stableCount = 0;
  let lastSignature = "";

  while (Date.now() - start < maxWaitMs) {
    const elapsed = Date.now() - start;
    const signature = await webContents.executeJavaScript(`
      (() => {
        const bodyText = (document.body?.innerText || '').replace(/\\s+/g, ' ').trim();
        const bodyLength = bodyText.length;
        const htmlLength = document.documentElement?.outerHTML?.length || 0;
        const readyState = document.readyState;
        const title = document.title || '';
        const linkCount = document.querySelectorAll('a[href]').length;
        return [readyState, title, bodyLength, htmlLength, linkCount].join('|');
      })();
    `);

    if (signature === lastSignature) {
      stableCount += 1;
    } else {
      stableCount = 0;
      lastSignature = signature;
    }

    if (elapsed >= minWaitMs && stableCount >= stableRounds) {
      return;
    }

    await delay(pollMs);
  }
}

async function withHiddenBrowserWindow<T>(
  url: string,
  evaluator: () => Promise<T>,
  settleOptions?: { minWaitMs?: number; maxWaitMs?: number; stableRounds?: number },
) {
  const browser = new BrowserWindow({
    show: false,
    webPreferences: {
      contextIsolation: true,
      sandbox: false,
      nodeIntegration: false,
    },
  });

  try {
    await browser.loadURL(url);
    await waitForPageSettled(browser.webContents, settleOptions);
    return await evaluator.call(browser.webContents) as T;
  } finally {
    browser.destroy();
  }
}

async function searchWeb(query: string) {
  const targetUrl = `https://duckduckgo.com/html/?q=${encodeURIComponent(query)}`;
  return withHiddenBrowserWindow(
    targetUrl,
    async function (this: Electron.WebContents) {
      return this.executeJavaScript(`
        (() => {
          const results = Array.from(document.querySelectorAll('.result'))
            .slice(0, 8)
            .map((node) => {
              const link = node.querySelector('.result__title a');
              const snippet = node.querySelector('.result__snippet');
              return {
                title: (link?.textContent || '').trim(),
                url: link?.href || '',
                snippet: (snippet?.textContent || '').trim(),
              };
            })
            .filter((result) => result.title && result.url);
          return {
            query: ${JSON.stringify(query)},
            results,
          };
        })();
      `);
    },
    { minWaitMs: 500, maxWaitMs: 5000, stableRounds: 2 },
  );
}

async function openWebpage(url: string) {
  return withHiddenBrowserWindow(
    url,
    async function (this: Electron.WebContents) {
      return this.executeJavaScript(`
        (() => {
          const links = Array.from(document.querySelectorAll('a[href]'))
            .slice(0, 20)
            .map((link) => ({
              text: (link.textContent || '').trim(),
              url: link.href || '',
            }))
            .filter((link) => link.url);

          const content = (document.body?.innerText || '')
            .replace(/\\n{3,}/g, '\\n\\n')
            .trim()
            .slice(0, 20000);

          return {
            url: ${JSON.stringify(url)},
            finalUrl: location.href,
            title: document.title || location.href,
            content,
            links,
          };
        })();
      `);
    },
    { minWaitMs: 1500, maxWaitMs: 30000, stableRounds: 4 },
  );
}

function normalizeKeyForInput(key: string) {
  if (key.length === 1) {
    return key.toUpperCase();
  }
  return key;
}

function createBrowserAutomationSession() {
  let browser: BrowserWindow | null = null;

  async function ensureBrowser() {
    if (browser && !browser.isDestroyed()) {
      return browser;
    }

    browser = new BrowserWindow({
      show: false,
      webPreferences: {
        contextIsolation: true,
        sandbox: false,
        nodeIntegration: false,
      },
    });

    return browser;
  }

  async function getElementCenter(selector: string) {
    const activeBrowser = await ensureBrowser();
    const webContents = activeBrowser.webContents;
    const rect = await webContents.executeJavaScript(`
      (() => {
        const element = document.querySelector(${JSON.stringify(selector)});
        if (!element) {
          return null;
        }
        const bounds = element.getBoundingClientRect();
        return {
          x: Math.round(bounds.left + bounds.width / 2),
          y: Math.round(bounds.top + bounds.height / 2),
        };
      })();
    `);

    if (!rect || typeof rect.x !== "number" || typeof rect.y !== "number") {
      throw new Error(`Web element not found for selector: ${selector}`);
    }

    return { browser: activeBrowser, x: rect.x, y: rect.y };
  }

  return {
    async open(url: string) {
      const activeBrowser = await ensureBrowser();
      await activeBrowser.loadURL(url);
      await waitForPageSettled(activeBrowser.webContents, { minWaitMs: 1500, maxWaitMs: 30000, stableRounds: 4 });
      return activeBrowser.webContents.executeJavaScript(`
        (() => {
          const links = Array.from(document.querySelectorAll('a[href]'))
            .slice(0, 20)
            .map((link) => ({
              text: (link.textContent || '').trim(),
              url: link.href || '',
            }))
            .filter((link) => link.url);

          const content = (document.body?.innerText || '')
            .replace(/\\n{3,}/g, '\\n\\n')
            .trim()
            .slice(0, 20000);

          return {
            url: ${JSON.stringify(url)},
            finalUrl: location.href,
            title: document.title || location.href,
            content,
            links,
          };
        })();
      `);
    },
    async click(selector: string) {
      const { browser: activeBrowser, x, y } = await getElementCenter(selector);
      activeBrowser.webContents.sendInputEvent({ type: "mouseMove", x, y, movementX: 0, movementY: 0 });
      activeBrowser.webContents.sendInputEvent({ type: "mouseDown", x, y, button: "left", clickCount: 1 });
      activeBrowser.webContents.sendInputEvent({ type: "mouseUp", x, y, button: "left", clickCount: 1 });
      await waitForPageSettled(activeBrowser.webContents, { minWaitMs: 300, maxWaitMs: 10000, stableRounds: 2 });
      return { selector, clicked: true as const };
    },
    async scroll(deltaY: number) {
      const activeBrowser = await ensureBrowser();
      await activeBrowser.webContents.executeJavaScript(`window.scrollBy({ top: ${JSON.stringify(deltaY)}, behavior: 'instant' });`);
      await waitForPageSettled(activeBrowser.webContents, { minWaitMs: 300, maxWaitMs: 10000, stableRounds: 2 });
      return { deltaY, scrolled: true as const };
    },
    async type(selector: string, text: string, clear = true) {
      const activeBrowser = await ensureBrowser();
      const webContents = activeBrowser.webContents;
      const focused = await webContents.executeJavaScript(`
        (() => {
          const element = document.querySelector(${JSON.stringify(selector)});
          if (!element) {
            return false;
          }
          if (${clear ? "true" : "false"}) {
            if ('value' in element) {
              element.value = '';
            }
            if (element.isContentEditable) {
              element.textContent = '';
            }
          }
          element.focus();
          return true;
        })();
      `);
      if (!focused) {
        throw new Error(`Web element not found for selector: ${selector}`);
      }
      webContents.insertText(text);
      await delay(150);
      return { selector, typed: true as const };
    },
    async press(key: string) {
      const activeBrowser = await ensureBrowser();
      const normalizedKey = normalizeKeyForInput(key);
      activeBrowser.webContents.sendInputEvent({ type: "keyDown", keyCode: normalizedKey });
      activeBrowser.webContents.sendInputEvent({ type: "keyUp", keyCode: normalizedKey });
      await waitForPageSettled(activeBrowser.webContents, { minWaitMs: 200, maxWaitMs: 8000, stableRounds: 2 });
      return { key, pressed: true as const };
    },
    async drag(selector: string, deltaX: number, deltaY: number) {
      const { browser: activeBrowser, x, y } = await getElementCenter(selector);
      activeBrowser.webContents.sendInputEvent({ type: "mouseMove", x, y, movementX: 0, movementY: 0 });
      activeBrowser.webContents.sendInputEvent({ type: "mouseDown", x, y, button: "left", clickCount: 1 });
      activeBrowser.webContents.sendInputEvent({
        type: "mouseMove",
        x: x + deltaX,
        y: y + deltaY,
        movementX: deltaX,
        movementY: deltaY,
        button: "left",
      });
      activeBrowser.webContents.sendInputEvent({ type: "mouseUp", x: x + deltaX, y: y + deltaY, button: "left", clickCount: 1 });
      await waitForPageSettled(activeBrowser.webContents, { minWaitMs: 300, maxWaitMs: 10000, stableRounds: 2 });
      return { selector, dragged: true as const };
    },
    async resize(width: number, height: number) {
      const activeBrowser = await ensureBrowser();
      const safeWidth = Math.max(320, Math.round(width));
      const safeHeight = Math.max(240, Math.round(height));
      activeBrowser.setContentSize(safeWidth, safeHeight);
      await waitForPageSettled(activeBrowser.webContents, { minWaitMs: 300, maxWaitMs: 10000, stableRounds: 2 });
      return { width: safeWidth, height: safeHeight, resized: true as const };
    },
    async screenshot() {
      const activeBrowser = await ensureBrowser();
      await waitForPageSettled(activeBrowser.webContents, { minWaitMs: 300, maxWaitMs: 12000, stableRounds: 2 });
      const [pageMeta, image] = await Promise.all([
        activeBrowser.webContents.executeJavaScript(`
          (() => ({
            finalUrl: location.href,
            title: document.title || location.href,
            width: window.innerWidth || document.documentElement.clientWidth || 0,
            height: window.innerHeight || document.documentElement.clientHeight || 0,
          }))();
        `) as Promise<{ finalUrl: string; title: string; width: number; height: number }>,
        activeBrowser.webContents.capturePage(),
      ]);

      return {
        finalUrl: pageMeta.finalUrl,
        title: pageMeta.title,
        mimeType: "image/png" as const,
        dataUrl: `data:image/png;base64,${image.toPNG().toString("base64")}`,
        width: pageMeta.width,
        height: pageMeta.height,
      };
    },
    dispose() {
      if (browser && !browser.isDestroyed()) {
        browser.destroy();
      }
      browser = null;
    },
  };
}

function resolveToolPath(targetPath: string, rootPath: string) {
  if (path.isAbsolute(targetPath)) {
    return path.normalize(targetPath);
  }

  return path.resolve(rootPath, targetPath);
}

async function getOllamaModels(): Promise<OllamaModelSummary[]> {
  const tagsResponse = await fetch("http://127.0.0.1:11434/api/tags");
  if (!tagsResponse.ok) {
    const body = await tagsResponse.text();
    throw new Error(`Ollama tags failed: ${tagsResponse.status} ${body}`);
  }

  const tagsPayload = (await tagsResponse.json()) as {
    models?: Array<{
      name?: string;
      model?: string;
      size?: number;
      modified_at?: string;
      details?: {
        family?: string;
        parameter_size?: string;
      };
    }>;
  };

  return Promise.all(
    (tagsPayload.models ?? []).map(async (model) => {
      let contextLength: number | null = null;

      try {
        const showResponse = await fetch("http://127.0.0.1:11434/api/show", {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify({
            model: model.model ?? model.name ?? "",
          }),
        });

        if (showResponse.ok) {
          const showPayload = (await showResponse.json()) as { model_info?: Record<string, unknown> };
          const contextLengthEntry = Object.entries(showPayload.model_info ?? {}).find(([key]) => key.endsWith(".context_length"));
          contextLength = typeof contextLengthEntry?.[1] === "number" ? contextLengthEntry[1] : null;
        }
      } catch {
        contextLength = null;
      }

      return {
        name: model.name ?? model.model ?? "unknown",
        model: model.model ?? model.name ?? "unknown",
        size: model.size ?? 0,
        modifiedAt: model.modified_at ?? "",
        parameterSize: model.details?.parameter_size,
        family: model.details?.family,
        contextLength,
      } satisfies OllamaModelSummary;
    }),
  );
}

async function unloadOllamaModel(model: string) {
  if (!model) {
    return;
  }

  try {
    await fetch("http://127.0.0.1:11434/api/generate", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        model,
        prompt: "",
        stream: false,
        keep_alive: 0,
      }),
    });
  } catch {
    // Ignore unload failures. Ollama may already be stopped.
  }
}

async function unloadIdleOllamaModels(exceptModel?: string) {
  const unloadTargets = [...knownOllamaModels].filter(
    (model) => model !== exceptModel && (activeOllamaRequestCounts.get(model) ?? 0) <= 0,
  );
  await Promise.all(unloadTargets.map((model) => unloadOllamaModel(model)));
}

function beginOllamaRequest(model: string) {
  knownOllamaModels.add(model);
  activeOllamaRequestCounts.set(model, (activeOllamaRequestCounts.get(model) ?? 0) + 1);
}

async function endOllamaRequest(model: string) {
  const nextCount = (activeOllamaRequestCounts.get(model) ?? 1) - 1;
  if (nextCount > 0) {
    activeOllamaRequestCounts.set(model, nextCount);
  } else {
    activeOllamaRequestCounts.delete(model);
  }
}

async function readDirectoryTree(rootPath: string, depth = 0): Promise<Array<{
  name: string;
  path: string;
  type: "directory" | "file";
  children?: Array<{
    name: string;
    path: string;
    type: "directory" | "file";
    children?: unknown[];
  }>;
}>> {
  if (depth > 5) {
    return [];
  }

  const entries = await readdir(rootPath, { withFileTypes: true });
  const visibleEntries = entries.filter((entry) => entry.name !== ".DS_Store");
  const sortedEntries = visibleEntries.sort((left, right) => {
    if (left.isDirectory() && !right.isDirectory()) {
      return -1;
    }
    if (!left.isDirectory() && right.isDirectory()) {
      return 1;
    }
    return left.name.localeCompare(right.name);
  });

  return Promise.all(
    sortedEntries.map(async (entry) => {
      const entryPath = path.join(rootPath, entry.name);
      if (entry.isDirectory()) {
        return {
          name: entry.name,
          path: entryPath,
          type: "directory" as const,
          children: await readDirectoryTree(entryPath, depth + 1),
        };
      }

      return {
        name: entry.name,
        path: entryPath,
        type: "file" as const,
      };
    }),
  );
}

async function createWindow(workspacePath = workspaceRoot) {
  const window = new BrowserWindow({
    width: 1480,
    height: 920,
    minWidth: 1080,
    minHeight: 760,
    backgroundColor: "#111111",
    frame: false,
    titleBarStyle: "hidden",
    webPreferences: {
      preload: preloadPath,
      contextIsolation: true,
      nodeIntegration: false,
      sandbox: false,
      webviewTag: true,
    },
  });

  if (isDev) {
    await window.loadURL("http://localhost:5173");
    windowWorkspaceRoots.set(window.id, workspacePath);
    window.on("closed", () => {
      windowWorkspaceRoots.delete(window.id);
    });
    return window;
  }

  await window.loadFile(path.join(app.getAppPath(), "apps/desktop/renderer/dist/index.html"));
  windowWorkspaceRoots.set(window.id, workspacePath);
  window.on("closed", () => {
    windowWorkspaceRoots.delete(window.id);
  });
  return window;
}

async function openAbsolutePath(targetPath: string) {
  const targetStat = await stat(targetPath);

  if (targetStat.isDirectory()) {
    return {
      type: "directory" as const,
      rootPath: targetPath,
      name: path.basename(targetPath),
      children: await readDirectoryTree(targetPath),
    };
  }

  const content = await readFile(targetPath, "utf8");
  const rootPath = path.dirname(targetPath);

  return {
    type: "file" as const,
    rootPath,
    name: path.basename(rootPath),
    path: targetPath,
    content,
    children: await readDirectoryTree(rootPath),
  };
}

app.whenReady().then(async () => {
  app.on("gpu-info-update", () => {
    // Helpful for debugging backdrop-filter / compositor behavior in local dev.
    console.log("GPU feature status:", app.getGPUFeatureStatus());
  });

  const createAgentToolHandlers = (rootPath: string, sender: Electron.WebContents) => {
    const browserSession = createBrowserAutomationSession();

    return {
    readFile: async (targetPath: string) => {
      const resolvedPath = resolveToolPath(targetPath, rootPath);
      const content = await readFile(resolvedPath, "utf8");
      return { path: resolvedPath, content };
    },
    listDir: async (targetPath: string) => {
      const resolvedPath = resolveToolPath(targetPath, rootPath);
      const entries = await readdir(resolvedPath, { withFileTypes: true });
      const visibleEntries = entries.filter((entry) => entry.name !== ".DS_Store");
      const sortedEntries = visibleEntries.sort((left, right) => {
        if (left.isDirectory() && !right.isDirectory()) {
          return -1;
        }
        if (!left.isDirectory() && right.isDirectory()) {
          return 1;
        }
        return left.name.localeCompare(right.name);
      });

      return {
        path: resolvedPath,
        entries: sortedEntries.map((entry) => ({
          name: entry.name,
          path: path.join(resolvedPath, entry.name),
          type: entry.isDirectory() ? ("directory" as const) : ("file" as const),
        })),
      };
    },
    writeFile: async (targetPath: string, content: string) => {
      const resolvedPath = resolveToolPath(targetPath, rootPath);
      let originalContent = "";
      try {
        originalContent = await readFile(resolvedPath, "utf8");
      } catch {
        originalContent = "";
      }
      await writeFile(resolvedPath, content, "utf8");
      return { path: resolvedPath, saved: true as const, originalContent };
    },
    createFile: async (targetPath: string, content: string) => {
      const resolvedPath = resolveToolPath(targetPath, rootPath);
      await mkdir(path.dirname(resolvedPath), { recursive: true });
      let originalContent = "";
      try {
        originalContent = await readFile(resolvedPath, "utf8");
      } catch {
        originalContent = "";
      }
      await writeFile(resolvedPath, content, "utf8");
      return { path: resolvedPath, saved: true as const, originalContent };
    },
    deleteFile: async (targetPath: string) => {
      const resolvedPath = resolveToolPath(targetPath, rootPath);
      await unlink(resolvedPath);
      return { path: resolvedPath, deleted: true as const };
    },
    deleteDirectory: async (targetPath: string) => {
      const resolvedPath = resolveToolPath(targetPath, rootPath);
      await rmdir(resolvedPath);
      return { path: resolvedPath, deleted: true as const };
    },
    runCommand: async function* (command: string, cwd?: string) {
      const queue: CommandChunk[] = [];
      let pendingResolve: (() => void) | null = null;
      let finished = false;
      let exitCode: number | null = null;

      void streamCommand(
        {
          command,
          cwd: cwd ? resolveToolPath(cwd, rootPath) : rootPath,
        },
        (chunk) => {
          queue.push(chunk);
          pendingResolve?.();
          pendingResolve = null;
        },
      ).then((result) => {
        exitCode = result.exitCode;
        finished = true;
        pendingResolve?.();
        pendingResolve = null;
      });

      while (!finished || queue.length > 0) {
        if (queue.length === 0) {
          await new Promise<void>((resolve) => {
            pendingResolve = resolve;
          });
          continue;
        }

        const chunk = queue.shift();
        if (chunk) {
          yield chunk;
        }
      }

      return exitCode;
    },
    searchWeb: async (query: string) => {
      return searchWeb(query);
    },
    openWebpage: async (url: string) => {
      return browserSession.open(url);
    },
    clickWebpage: async (selector: string) => {
      return browserSession.click(selector);
    },
    scrollWebpage: async (deltaY: number) => {
      return browserSession.scroll(deltaY);
    },
    typeWebpage: async (selector: string, text: string, clear?: boolean) => {
      return browserSession.type(selector, text, clear);
    },
    pressWebpageKey: async (key: string) => {
      return browserSession.press(key);
    },
    dragWebpage: async (selector: string, deltaX: number, deltaY: number) => {
      return browserSession.drag(selector, deltaX, deltaY);
    },
    resizeWebpage: async (width: number, height: number) => {
      return browserSession.resize(width, height);
    },
    screenshotWebpage: async () => {
      return browserSession.screenshot();
    },
    requestBrowserAssist: async (request: BrowserAssistRequest) => {
      return new Promise<BrowserAssistResult>((resolve, reject) => {
        pendingBrowserAssistRequests.set(request.requestId, {
          senderId: sender.id,
          resolve,
          reject,
        });
      });
    },
    dispose: () => {
      browserSession.dispose();
    },
    };
  };

  ipcMain.on("agent:run-task-stream", async (event, payload: { requestId: string; request: TaskRequest }) => {
    const { requestId, request } = payload;
    const eventChannel = `agent:run-task-stream:${requestId}`;
    const doneChannel = `agent:run-task-stream:${requestId}:done`;
    const abortController = new AbortController();
    const agentToolHandlers = createAgentToolHandlers(request.workspaceRoot || workspaceRoot, event.sender);
    activeAgentRuns.set(requestId, abortController);
    const ollamaModel = request.modelProvider === "ollama" ? request.modelId?.trim() || "llama3:latest" : null;

    if (ollamaModel) {
      beginOllamaRequest(ollamaModel);
      void unloadIdleOllamaModels(ollamaModel);
    } else {
      void unloadIdleOllamaModels();
    }

    try {
      for await (const item of runAgentTask(request, agentToolHandlers, abortController.signal)) {
        if (event.sender.isDestroyed()) {
          break;
        }
        event.sender.send(eventChannel, item);
      }
    } catch (error) {
      const message = error instanceof Error ? error.message : "Unknown streaming error";
      if (!event.sender.isDestroyed()) {
        event.sender.send(eventChannel, {
          type: "message",
          chunk: `Agent streaming error: ${message}`,
        } satisfies AgentStreamEvent);
        event.sender.send(eventChannel, {
          type: "done",
          summary: "Agent streaming failed.",
        } satisfies AgentStreamEvent);
      }
    } finally {
      for (const [assistRequestId, pending] of pendingBrowserAssistRequests) {
        if (pending.senderId === event.sender.id) {
          pending.reject(new Error("Browser assistance was canceled."));
          pendingBrowserAssistRequests.delete(assistRequestId);
        }
      }
      agentToolHandlers.dispose();
      activeAgentRuns.delete(requestId);
      if (ollamaModel) {
        await endOllamaRequest(ollamaModel);
      }
      if (!event.sender.isDestroyed()) {
        event.sender.send(doneChannel);
      }
    }
  });

  ipcMain.handle("agent:cancel-run-task-stream", async (_event, requestId: string) => {
    activeAgentRuns.get(requestId)?.abort();
    return { canceled: true as const };
  });

  ipcMain.handle("agent:browser-assist-complete", async (_event, payload: BrowserAssistResult) => {
    const pending = pendingBrowserAssistRequests.get(payload.requestId);
    if (!pending) {
      return { completed: false as const };
    }

    pending.resolve(payload);
    pendingBrowserAssistRequests.delete(payload.requestId);
    return { completed: true as const };
  });

  ipcMain.handle("agent:browser-assist-cancel", async (_event, requestId: string) => {
    const pending = pendingBrowserAssistRequests.get(requestId);
    if (!pending) {
      return { canceled: false as const };
    }

    pending.reject(new Error("Browser assistance was canceled by the user."));
    pendingBrowserAssistRequests.delete(requestId);
    return { canceled: true as const };
  });

  ipcMain.handle("agent:run-task", async (event, request: TaskRequest) => {
    const agentToolHandlers = createAgentToolHandlers(request.workspaceRoot || workspaceRoot, event.sender);
    const ollamaModel = request.modelProvider === "ollama" ? request.modelId?.trim() || "llama3:latest" : null;
    if (ollamaModel) {
      beginOllamaRequest(ollamaModel);
      void unloadIdleOllamaModels(ollamaModel);
    } else {
      void unloadIdleOllamaModels();
    }

    const events: AgentStreamEvent[] = [];
    try {
      for await (const item of runAgentTask(request, agentToolHandlers)) {
        events.push(item);
      }
    } finally {
      for (const [assistRequestId, pending] of pendingBrowserAssistRequests) {
        if (pending.senderId === event.sender.id) {
          pending.reject(new Error("Browser assistance was canceled."));
          pendingBrowserAssistRequests.delete(assistRequestId);
        }
      }
      agentToolHandlers.dispose();
      if (ollamaModel) {
        await endOllamaRequest(ollamaModel);
      }
    }
    return events;
  });

  ipcMain.handle("agent:context", async (_event, request: TaskRequest) => {
    return buildContextSnapshot(request);
  });

  ipcMain.handle("agent:context-usage", async (_event, request: TaskRequest): Promise<ContextUsageSnapshot | null> => {
    return buildContextUsageSnapshot(request);
  });

  ipcMain.handle("command:run", async (_event, request: RunCommandRequest) => {
    const events: AgentStreamEvent[] = [];
    const commandId = `command-${Date.now()}`;
    events.push({ type: "command-start", commandId, command: request.command });
    const result = await streamCommand(request, (chunk) => {
      events.push({ type: "command", commandId, chunk });
    });
    events.push({ type: "command-end", commandId, exitCode: result.exitCode });
    events.push({ type: "done", summary: "Command completed" });
    return events;
  });

  ipcMain.handle("system:gpu-status", async () => {
    return {
      featureStatus: app.getGPUFeatureStatus(),
      sandboxed: app.isPackaged,
    };
  });

  ipcMain.handle("system:ollama-models", async () => {
    try {
      const models = await getOllamaModels();
      return {
        available: models.length > 0,
        models,
      };
    } catch (error) {
      return {
        available: false,
        models: [],
        error: error instanceof Error ? error.message : "Unknown Ollama error",
      };
    }
  });

  ipcMain.handle("system:workspace-info", async (event) => {
    const target = BrowserWindow.fromWebContents(event.sender);
    const rootPath = target ? (windowWorkspaceRoots.get(target.id) ?? workspaceRoot) : workspaceRoot;
    return {
      rootPath,
      name: path.basename(rootPath),
    };
  });

  ipcMain.handle("system:open-folder", async (event) => {
    const target = BrowserWindow.fromWebContents(event.sender);
    if (!target) {
      return null;
    }

    const currentRoot = windowWorkspaceRoots.get(target.id) ?? workspaceRoot;
    const result = await dialog.showOpenDialog(target, {
      title: "Open Folder",
      defaultPath: currentRoot,
      properties: ["openDirectory"],
    });

    if (result.canceled || result.filePaths.length === 0) {
      return null;
    }

    const rootPath = result.filePaths[0];
    windowWorkspaceRoots.set(target.id, rootPath);

    return {
      rootPath,
      name: path.basename(rootPath),
    };
  });

  ipcMain.handle("system:new-window", async (event) => {
    const target = BrowserWindow.fromWebContents(event.sender);
    const currentRoot = target ? (windowWorkspaceRoots.get(target.id) ?? workspaceRoot) : workspaceRoot;
    await createWindow(currentRoot);
    return { opened: true };
  });

  ipcMain.handle("system:file-tree", async (event) => {
    const target = BrowserWindow.fromWebContents(event.sender);
    const rootPath = target ? (windowWorkspaceRoots.get(target.id) ?? workspaceRoot) : workspaceRoot;
    return {
      rootPath,
      name: path.basename(rootPath),
      children: await readDirectoryTree(rootPath),
    };
  });

  ipcMain.handle("system:read-file", async (_event, filePath: string) => {
    try {
      const content = await readFile(filePath, "utf8");
      return { path: filePath, content };
    } catch (error) {
      if ((error as NodeJS.ErrnoException).code === "ENOENT") {
        return { path: filePath, content: "", missing: true };
      }
      throw error;
    }
  });

  ipcMain.handle("system:open-path", async (event, targetPath: string) => {
    const target = BrowserWindow.fromWebContents(event.sender);
    const opened = await openAbsolutePath(targetPath);

    if (target) {
      windowWorkspaceRoots.set(target.id, opened.rootPath);
    }

    return opened;
  });

  ipcMain.handle("system:write-file", async (_event, request: { filePath: string; content: string }) => {
    await writeFile(request.filePath, request.content, "utf8");
    return { path: request.filePath, saved: true };
  });

  ipcMain.handle("window:minimize", (event) => {
    BrowserWindow.fromWebContents(event.sender)?.minimize();
  });

  ipcMain.handle("window:maximize-toggle", (event) => {
    const target = BrowserWindow.fromWebContents(event.sender);
    if (!target) {
      return { maximized: false };
    }

    if (target.isMaximized()) {
      target.unmaximize();
      return { maximized: false };
    }

    target.maximize();
    return { maximized: true };
  });

  ipcMain.handle("window:close", (event) => {
    BrowserWindow.fromWebContents(event.sender)?.close();
  });

  ipcMain.handle("window:get-state", (event) => {
    const target = BrowserWindow.fromWebContents(event.sender);
    return {
      maximized: target?.isMaximized() ?? false,
    };
  });

  const mainWindow = await createWindow();
  mainWindow.webContents.on("did-finish-load", () => {
    void mainWindow.webContents
      .executeJavaScript("typeof window.cogent !== 'undefined'")
      .then((hasBridge) => {
        console.log("Renderer bridge available:", hasBridge);
      })
      .catch((error) => {
        console.error("Failed to inspect renderer bridge:", error);
      });
  });

  app.on("activate", async () => {
    if (BrowserWindow.getAllWindows().length === 0) {
      await createWindow();
    }
  });
});

app.on("window-all-closed", () => {
  if (process.platform !== "darwin") {
    void Promise.all([...knownOllamaModels].map((model) => unloadOllamaModel(model)));
    app.quit();
  }
});
