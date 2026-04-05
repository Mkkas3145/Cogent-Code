import { app, BrowserWindow, dialog, ipcMain } from "electron";
import { execFile } from "node:child_process";
import { mkdir, readFile, readdir, rmdir, stat, unlink, writeFile } from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { promisify } from "node:util";
import { buildContextSnapshot, buildContextUsageSnapshot, runAgentTask } from "@cogent/agent-core";
import { streamCommand } from "@cogent/command-runtime";
import type {
  AgentStreamEvent,
  BrowserAssistRequest,
  BrowserAssistResult,
  CommandChunk,
  ContextUsageSnapshot,
  GeminiModelSummary,
  LmStudioModelSummary,
  ModelProvider,
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
let isRunningShutdownCleanup = false;
let didFinishShutdownCleanup = false;
let shutdownCleanupPromise: Promise<void> | null = null;
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
const execFileAsync = promisify(execFile);

function delay(ms: number) {
  return new Promise<void>((resolve) => {
    setTimeout(resolve, ms);
  });
}

function buildResolveElementScript(selector: string, options?: { focus?: boolean; clear?: boolean }) {
  return `
    (() => {
      const selector = ${JSON.stringify(selector)};
      const shouldFocus = ${options?.focus ? "true" : "false"};
      const shouldClear = ${options?.clear ? "true" : "false"};

      const normalize = (value) =>
        String(value ?? "")
          .replace(/\\s+/g, " ")
          .trim()
          .toLowerCase();

      const isVisible = (element) => {
        if (!element) return false;
        const bounds = element.getBoundingClientRect();
        if (bounds.width <= 0 || bounds.height <= 0) return false;
        const style = window.getComputedStyle(element);
        return style.display !== "none" && style.visibility !== "hidden" && style.opacity !== "0";
      };

      const getScore = (element, attributes, needle) => {
        const normalizedNeedle = normalize(needle);
        if (!normalizedNeedle) return 0;
        let best = 0;

        for (let index = 0; index < attributes.length; index += 1) {
          const attribute = attributes[index];
          const value =
            attribute === "text"
              ? normalize(element.textContent)
              : normalize(element.getAttribute?.(attribute) ?? element[attribute]);

          if (!value) continue;
          const attrPenalty = index * 7;
          if (value === normalizedNeedle) {
            best = Math.max(best, 100 - attrPenalty);
          } else if (value.includes(normalizedNeedle)) {
            best = Math.max(best, 82 - attrPenalty);
          } else if (normalizedNeedle.includes(value)) {
            best = Math.max(best, 72 - attrPenalty);
          }
        }

        if (isVisible(element)) {
          best += 5;
        }

        return best;
      };

      const pickBestMatch = (tagName, primaryAttribute, rawValue) => {
        const tagSelector = tagName && tagName !== "*" ? tagName : "*";
        const candidates = Array.from(document.querySelectorAll(tagSelector));
        const attributePriority = {
          placeholder: ["placeholder", "aria-label", "name", "id", "text"],
          "aria-label": ["aria-label", "placeholder", "name", "id", "text"],
          name: ["name", "placeholder", "aria-label", "id", "text"],
          id: ["id", "name", "aria-label", "placeholder", "text"],
          title: ["title", "aria-label", "placeholder", "text"],
          value: ["value", "placeholder", "aria-label", "text"],
          role: ["role", "aria-label", "text"],
          type: ["type", "name", "aria-label", "placeholder", "text"],
        };

        const attributes = attributePriority[primaryAttribute] ?? [primaryAttribute, "aria-label", "placeholder", "name", "id", "text"];
        let bestElement = null;
        let bestScore = 0;

        for (const element of candidates) {
          const score = getScore(element, attributes, rawValue);
          if (score > bestScore) {
            bestScore = score;
            bestElement = element;
          }
        }

        return bestScore >= 70 ? bestElement : null;
      };

      let element = null;

      try {
        element = document.querySelector(selector);
      } catch {
        element = null;
      }

      if (!isVisible(element)) {
        element = null;
      }

      if (!element) {
        const exactAttributeMatch = selector.match(/^\\s*([a-zA-Z0-9_*:-]+)?\\[(placeholder|aria-label|name|id|type|role|title|value)=['"](.+?)['"]\\]\\s*$/i);
        if (exactAttributeMatch) {
          const [, tagName = "*", attributeName, rawValue] = exactAttributeMatch;
          element = pickBestMatch(tagName, attributeName.toLowerCase(), rawValue);
        }
      }

      if (!element) {
        const embeddedAttributeMatch = selector.match(/(placeholder|aria-label|name|id|type|role|title|value)=['"](.+?)['"]/i);
        if (embeddedAttributeMatch) {
          const [, attributeName, rawValue] = embeddedAttributeMatch;
          element = pickBestMatch("*", attributeName.toLowerCase(), rawValue);
        }
      }

      if (!element || !isVisible(element)) {
        return null;
      }

      if (shouldFocus) {
        if (shouldClear) {
          if ("value" in element) {
            element.value = "";
          }
          if (element.isContentEditable) {
            element.textContent = "";
          }
        }
        element.focus();
      }

      const bounds = element.getBoundingClientRect();
      return {
        x: Math.round(bounds.left + bounds.width / 2),
        y: Math.round(bounds.top + bounds.height / 2),
      };
    })();
  `;
}

function compareGeminiModelPriority(left: GeminiModelSummary, right: GeminiModelSummary) {
  const tokenize = (value: string) =>
    value
      .toLowerCase()
      .split(/[^a-z0-9]+/)
      .filter(Boolean);

  const extractVersion = (value: string) => {
    const match = value.match(/gemini[-\s]?(\d+(?:\.\d+)?)/i);
    return match ? Number(match[1]) : 0;
  };

  const scoreVariant = (value: string) => {
    const tokens = tokenize(value);
    if (tokens.includes("pro")) return 5;
    if (tokens.includes("flash") && tokens.includes("lite")) return 3;
    if (tokens.includes("flash")) return 4;
    return 2;
  };

  const leftSource = `${left.id} ${left.name}`;
  const rightSource = `${right.id} ${right.name}`;

  const versionDiff = extractVersion(rightSource) - extractVersion(leftSource);
  if (versionDiff !== 0) {
    return versionDiff;
  }

  const variantDiff = scoreVariant(rightSource) - scoreVariant(leftSource);
  if (variantDiff !== 0) {
    return variantDiff;
  }

  return left.name.localeCompare(right.name);
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

  function getExistingBrowser() {
    if (browser && !browser.isDestroyed()) {
      return browser;
    }
    return null;
  }

  async function waitForSelector(
    webContents: Electron.WebContents,
    selector: string,
    timeoutMs = 3000,
  ) {
    const deadline = Date.now() + timeoutMs;
    while (Date.now() < deadline) {
      const rect = await webContents.executeJavaScript(buildResolveElementScript(selector));

      if (rect && typeof rect.x === "number" && typeof rect.y === "number") {
        return rect;
      }

      await delay(200);
    }

    return null;
  }

  async function getElementCenter(selector: string) {
    const activeBrowser = await ensureBrowser();
    const webContents = activeBrowser.webContents;
    const rect = await waitForSelector(webContents, selector);

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
      await activeBrowser.webContents.executeJavaScript(`window.scrollBy(0, ${JSON.stringify(deltaY)});`);
      await waitForPageSettled(activeBrowser.webContents, { minWaitMs: 300, maxWaitMs: 10000, stableRounds: 2 });
      return { deltaY, scrolled: true as const };
    },
    async type(selector: string, text: string, clear = true) {
      const activeBrowser = await ensureBrowser();
      const webContents = activeBrowser.webContents;
      let focused = false;
      const deadline = Date.now() + 3000;
      while (!focused && Date.now() < deadline) {
        focused = Boolean(
          await webContents.executeJavaScript(
            buildResolveElementScript(selector, { focus: true, clear }),
          ),
        );
        if (!focused) {
          await delay(200);
        }
      }
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
    async getCurrentPreview() {
      const activeBrowser = getExistingBrowser();
      if (!activeBrowser) {
        return null;
      }
      const [meta, image] = await Promise.all([
        activeBrowser.webContents.executeJavaScript(`
          (() => ({
            url: location.href,
            title: document.title || location.href,
            width: window.innerWidth || document.documentElement.clientWidth || 0,
            height: window.innerHeight || document.documentElement.clientHeight || 0,
          }))();
        `) as Promise<{ url: string; title: string; width: number; height: number }>,
        activeBrowser.webContents.capturePage(),
      ]);

      return {
        ...meta,
        imageDataUrl: `data:image/png;base64,${image.toPNG().toString("base64")}`,
      };
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
  const loadedContexts = await getLoadedOllamaModelContexts();
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
        contextLength: loadedContexts.get(model.model ?? model.name ?? "") ?? contextLength,
      } satisfies OllamaModelSummary;
    }),
  );
}

async function getLoadedOllamaModelContexts(): Promise<Map<string, number>> {
  try {
    const response = await fetch("http://127.0.0.1:11434/api/ps");
    if (!response.ok) {
      return new Map();
    }

    const payload = (await response.json()) as {
      models?: Array<{
        model?: string;
        name?: string;
        context_length?: number;
      }>;
    };

    return new Map(
      (payload.models ?? [])
        .map((model) => ({
          id: (model.model ?? model.name ?? "").trim(),
          contextLength: model.context_length,
        }))
        .filter((entry): entry is { id: string; contextLength: number } => entry.id.length > 0 && typeof entry.contextLength === "number")
        .map((entry) => [entry.id, entry.contextLength] as const),
    );
  } catch {
    return new Map();
  }
}

async function getLmStudioModels(): Promise<LmStudioModelSummary[]> {
  const cliPath = path.join(process.env.USERPROFILE ?? "", ".lmstudio", "bin", "lms.exe");
  const { stdout } = await execFileAsync(cliPath, ["ls", "--llm", "--json"], {
    windowsHide: true,
    timeout: 15000,
    maxBuffer: 1024 * 1024 * 8,
  });

  const modelsPayload = JSON.parse(stdout) as Array<{
    modelKey?: string;
    displayName?: string;
    publisher?: string;
    indexedModelIdentifier?: string;
    maxContextLength?: number;
  }>;

  return modelsPayload
    .filter((model) => typeof (model.modelKey ?? model.indexedModelIdentifier) === "string")
    .map((model) => ({
      id: String(model.modelKey ?? model.indexedModelIdentifier).trim(),
      name: model.displayName?.trim() || String(model.modelKey ?? model.indexedModelIdentifier ?? "LM Studio model").trim(),
      ownedBy: model.publisher,
      contextLength: typeof model.maxContextLength === "number" ? model.maxContextLength : null,
    }));
}

async function getGeminiModels(apiKey: string): Promise<GeminiModelSummary[]> {
  const trimmedKey = apiKey.trim();
  if (!trimmedKey) {
    return [];
  }

  const response = await fetch(`https://generativelanguage.googleapis.com/v1beta/models?pageSize=1000&key=${encodeURIComponent(trimmedKey)}`);

  if (!response.ok) {
    const body = await response.text();
    throw new Error(`Gemini models failed: ${response.status} ${body}`);
  }

  const payload = (await response.json()) as {
    models?: Array<{
      name?: string;
      displayName?: string;
      description?: string;
      inputTokenLimit?: number;
      outputTokenLimit?: number;
      supportedGenerationMethods?: string[];
    }>;
  };

  const rawModels = payload.models ?? [];
  const filteredModels = rawModels
    .filter((model) => {
      const id = String(model.name ?? "").replace(/^models\//, "");
      const displayName = String(model.displayName ?? "").trim();
      const description = String(model.description ?? "").trim();
      const haystack = `${id} ${displayName} ${description}`.toLowerCase();
      return (
        /gemini/i.test(displayName) &&
        (model.supportedGenerationMethods ?? []).includes("generateContent") &&
        !/(?:embedding|embed|aqa|imagen|image generation|tts|speech)/i.test(haystack)
      );
    })
    .map((model) => {
      const id = String(model.name ?? "").replace(/^models\//, "").trim();
      const haystack = `${model.name ?? ""} ${model.displayName ?? ""} ${model.description ?? ""}`.toLowerCase();
      return {
        id,
        name: model.displayName?.trim() || id,
        description: model.description?.trim(),
        inputTokenLimit: typeof model.inputTokenLimit === "number" ? model.inputTokenLimit : null,
        outputTokenLimit: typeof model.outputTokenLimit === "number" ? model.outputTokenLimit : null,
        supportsImages: !/(?:text-only|text only)/i.test(haystack),
      } satisfies GeminiModelSummary;
    })
    .sort(compareGeminiModelPriority);

  return filteredModels;
}

async function waitForLmStudioServer(maxWaitMs = 15000) {
  const startedAt = Date.now();
  while (Date.now() - startedAt < maxWaitMs) {
    try {
      const response = await fetch("http://127.0.0.1:1234/v1/models");
      if (response.ok) {
        return;
      }
    } catch {
      // Keep waiting.
    }
    await delay(500);
  }

  throw new Error("LM Studio server did not become ready.");
}

function getDefaultLmStudioContextLength(totalMemoryBytes: number) {
  const totalMemoryGb = totalMemoryBytes / 1024 / 1024 / 1024;
  const reservedForSystemGb = Math.max(4, totalMemoryGb * 0.25);
  const usableMemoryGb = Math.max(4, totalMemoryGb - reservedForSystemGb);
  const estimatedContext = Math.floor(usableMemoryGb * 512);
  const clampedContext = Math.max(4096, Math.min(32768, estimatedContext));
  return Math.round(clampedContext / 1024) * 1024;
}

function normalizeLmStudioContextLength(value: number | undefined) {
  if (!Number.isFinite(value)) {
    return getDefaultLmStudioContextLength(os.totalmem());
  }

  return Math.max(1024, Math.round(value as number));
}

async function ensureLmStudioModelReady(model: string, contextLength?: number) {
  const normalizedModel = model.trim().split("@")[0]?.trim() ?? "";
  if (!normalizedModel) {
    return;
  }
  const normalizedContextLength = normalizeLmStudioContextLength(contextLength);

  const cliPath = path.join(process.env.USERPROFILE ?? "", ".lmstudio", "bin", "lms.exe");
  try {
    const response = await fetch("http://127.0.0.1:1234/v1/models");
    if (!response.ok) {
      throw new Error("LM Studio server is not reachable.");
    }
  } catch {
    await execFileAsync(cliPath, ["server", "start"], {
      windowsHide: true,
      timeout: 15000,
      maxBuffer: 1024 * 1024 * 2,
    });
    await waitForLmStudioServer();
  }

  await execFileAsync(
    cliPath,
    ["load", normalizedModel, "-c", String(normalizedContextLength), "-y"],
    {
      windowsHide: true,
      timeout: 120000,
      maxBuffer: 1024 * 1024 * 4,
    },
  );
}

async function getLoadedOllamaModels(): Promise<string[]> {
  try {
    const response = await fetch("http://127.0.0.1:11434/api/ps");
    if (!response.ok) {
      return [];
    }

    const payload = (await response.json()) as {
      models?: Array<{
        model?: string;
        name?: string;
      }>;
    };

    return (payload.models ?? [])
      .map((model) => (model.model ?? model.name ?? "").trim())
      .filter((model): model is string => model.length > 0);
  } catch {
    return [];
  }
}

async function unloadAllLmStudioModels() {
  const cliPath = path.join(process.env.USERPROFILE ?? "", ".lmstudio", "bin", "lms.exe");
  try {
    await execFileAsync(cliPath, ["unload", "--all"], {
      windowsHide: true,
      timeout: 30000,
      maxBuffer: 1024 * 1024 * 2,
    });
  } catch {
    // Ignore unload failures. LM Studio may not be running or may already be empty.
  }
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
  const loadedModels = await getLoadedOllamaModels();
  const unloadTargets = [...new Set([...knownOllamaModels, ...loadedModels])].filter(
    (model) => model !== exceptModel && (activeOllamaRequestCounts.get(model) ?? 0) <= 0,
  );
  await Promise.all(unloadTargets.map((model) => unloadOllamaModel(model)));
}

async function cleanupLocalModels(provider?: ModelProvider, modelId?: string) {
  if (provider === "ollama") {
    await unloadIdleOllamaModels(modelId?.trim() || undefined);
  } else {
    await unloadIdleOllamaModels();
  }

  if (provider === "lmstudio" && modelId?.trim()) {
    await unloadAllLmStudioModels();
    await ensureLmStudioModelReady(modelId, undefined);
  } else {
    await unloadAllLmStudioModels();
  }
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

function runShutdownCleanup() {
  if (shutdownCleanupPromise) {
    return shutdownCleanupPromise;
  }

  shutdownCleanupPromise = (async () => {
    const loadedOllamaModels = await getLoadedOllamaModels();
    await Promise.all([
      ...[...new Set([...knownOllamaModels, ...loadedOllamaModels])].map((model) => unloadOllamaModel(model)),
      unloadAllLmStudioModels(),
    ]);
  })()
    .catch(() => {
      // Ignore shutdown unload failures and continue quitting.
    })
    .then(() => {
      didFinishShutdownCleanup = true;
    })
    .finally(() => {
      isRunningShutdownCleanup = false;
    });

  return shutdownCleanupPromise;
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

  const createAgentToolHandlers = (rootPath: string, sender: Electron.WebContents, eventChannel?: string) => {
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
    const agentToolHandlers = createAgentToolHandlers(request.workspaceRoot || workspaceRoot, event.sender, eventChannel);
    activeAgentRuns.set(requestId, abortController);
    const ollamaModel = request.modelProvider === "ollama" ? request.modelId?.trim() || "llama3:latest" : null;

    if (ollamaModel) {
      beginOllamaRequest(ollamaModel);
      void unloadIdleOllamaModels(ollamaModel);
    } else {
      void unloadIdleOllamaModels();
    }

    if (request.modelProvider === "lmstudio" && request.modelId) {
      await unloadAllLmStudioModels();
      await ensureLmStudioModelReady(request.modelId, request.lmStudioContextLength);
    } else {
      void unloadAllLmStudioModels();
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

    if (request.modelProvider === "lmstudio" && request.modelId) {
      await unloadAllLmStudioModels();
      await ensureLmStudioModelReady(request.modelId, request.lmStudioContextLength);
    } else {
      void unloadAllLmStudioModels();
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
    if (request.modelProvider === "lmstudio" && request.modelId) {
      await unloadAllLmStudioModels();
      await ensureLmStudioModelReady(request.modelId, request.lmStudioContextLength);
    } else {
      void unloadAllLmStudioModels();
    }
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

  ipcMain.handle("system:memory-info", async () => {
    return {
      totalMemoryBytes: os.totalmem(),
      freeMemoryBytes: os.freemem(),
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

  ipcMain.handle("system:lmstudio-models", async () => {
    try {
      const models = await getLmStudioModels();
      return {
        available: models.length > 0,
        models,
      };
    } catch (error) {
      return {
        available: false,
        models: [],
        error: error instanceof Error ? error.message : "Unknown LM Studio error",
      };
    }
  });

  ipcMain.handle("system:gemini-models", async (_event, apiKey: string) => {
    try {
      const models = await getGeminiModels(apiKey);
      return {
        available: models.length > 0,
        models,
      };
    } catch (error) {
      console.error("Failed to fetch Gemini models:", error);
      return {
        available: false,
        models: [],
        error: error instanceof Error ? error.message : "Unknown Gemini error",
      };
    }
  });

  ipcMain.handle("system:cleanup-local-models", async (_event, payload: { provider?: ModelProvider; modelId?: string }) => {
    await cleanupLocalModels(payload.provider, payload.modelId);
    return { cleaned: true as const };
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
    app.quit();
  }
});

app.on("before-quit", (event) => {
  if (didFinishShutdownCleanup) {
    return;
  }

  if (isRunningShutdownCleanup) {
    event.preventDefault();
    return;
  }

  isRunningShutdownCleanup = true;
  event.preventDefault();
  BrowserWindow.getAllWindows().forEach((window) => window.hide());
  void runShutdownCleanup().finally(() => {
    app.exit(0);
  });
});
