import * as monaco from "monaco-editor";
import { loader } from "@monaco-editor/react";
import { initialize, getService, IExtensionService, IWorkbenchThemeService } from "@codingame/monaco-vscode-api";
import getConfigurationServiceOverride, {
  updateUserConfiguration,
} from "@codingame/monaco-vscode-configuration-service-override";
import getExtensionsServiceOverride from "@codingame/monaco-vscode-extensions-service-override";
import getFilesServiceOverride from "@codingame/monaco-vscode-files-service-override";
import getLanguagesServiceOverride from "@codingame/monaco-vscode-languages-service-override";
import getTextmateServiceOverride from "@codingame/monaco-vscode-textmate-service-override";
import getThemeServiceOverride from "@codingame/monaco-vscode-theme-service-override";
import "@codingame/monaco-vscode-theme-defaults-default-extension";
import { whenReady as themeDefaultsReady } from "@codingame/monaco-vscode-theme-defaults-default-extension";
import "@codingame/monaco-vscode-javascript-default-extension";
import { whenReady as javascriptExtensionReady } from "@codingame/monaco-vscode-javascript-default-extension";
import "@codingame/monaco-vscode-json-default-extension";
import { whenReady as jsonExtensionReady } from "@codingame/monaco-vscode-json-default-extension";
import "@codingame/monaco-vscode-html-default-extension";
import { whenReady as htmlExtensionReady } from "@codingame/monaco-vscode-html-default-extension";
import "@codingame/monaco-vscode-css-default-extension";
import { whenReady as cssExtensionReady } from "@codingame/monaco-vscode-css-default-extension";
import "@codingame/monaco-vscode-typescript-basics-default-extension";
import { whenReady as typescriptBasicsExtensionReady } from "@codingame/monaco-vscode-typescript-basics-default-extension";
import "@codingame/monaco-vscode-typescript-language-features-default-extension";
import { whenReady as typescriptLanguageFeaturesExtensionReady } from "@codingame/monaco-vscode-typescript-language-features-default-extension";
import editorWorker from "monaco-editor/esm/vs/editor/editor.worker?worker";
import tsWorker from "monaco-editor/esm/vs/language/typescript/ts.worker?worker";
import jsonWorker from "monaco-editor/esm/vs/language/json/json.worker?worker";
import cssWorker from "monaco-editor/esm/vs/language/css/css.worker?worker";
import htmlWorker from "monaco-editor/esm/vs/language/html/html.worker?worker";
import textMateWorker from "@codingame/monaco-vscode-textmate-service-override/worker?worker";

export const VSCODE_COLOR_THEME = "Dark 2026";

declare global {
  interface Window {
    monaco?: typeof monaco;
    MonacoEnvironment?: {
      getWorkerUrl?: (_workerId: string, label: string) => string;
      getWorker?: (_workerId: string, label: string) => Worker;
    };
  }
}

let monacoVscodeReady: Promise<void> | null = null;
let activeMonacoTheme: string | null = null;

function ensureWorkerEnvironment() {
  const globalScope = globalThis as typeof globalThis & {
    MonacoEnvironment?: unknown;
  };
  const monacoEnvironment = {
    getWorker(_workerId: string, label: string) {
      if (label === "TextMateWorker") {
        return new textMateWorker();
      }

      if (label === "typescript" || label === "javascript") {
        return new tsWorker();
      }

      if (label === "json") {
        return new jsonWorker();
      }

      if (label === "css" || label === "scss" || label === "less") {
        return new cssWorker();
      }

      if (label === "html" || label === "handlebars" || label === "razor") {
        return new htmlWorker();
      }

      return new editorWorker();
    },
  } as unknown as Window["MonacoEnvironment"];
  loader.config({ monaco });
  window.monaco = monaco;
  window.MonacoEnvironment = monacoEnvironment as never;
  globalScope.MonacoEnvironment = monacoEnvironment;
}

async function applyCurrentVsCodeTheme() {
  try {
    const extensionService = await getService(IExtensionService);
    await extensionService.whenInstalledExtensionsRegistered();

    const workbenchThemeService = await getService(IWorkbenchThemeService);
    const availableThemes = await workbenchThemeService.getColorThemes();
    const preferredTheme = availableThemes.find((theme) => theme.settingsId === VSCODE_COLOR_THEME);

    if (preferredTheme) {
      await workbenchThemeService.setColorTheme(preferredTheme.id, undefined);
    }

    const colorTheme = workbenchThemeService.getColorTheme();
    const themeName = colorTheme.id || colorTheme.settingsId || colorTheme.label;
    console.info("[monaco-vscode] Active workbench theme:", {
      id: colorTheme.id,
      label: colorTheme.label,
      settingsId: colorTheme.settingsId,
      type: colorTheme.type,
      availableThemes: availableThemes.map((theme) => ({
        id: theme.id,
        label: theme.label,
        settingsId: theme.settingsId,
      })),
    });
    if (themeName) {
      activeMonacoTheme = themeName;
      console.info("[monaco-vscode] Applying Monaco theme:", themeName);
      monaco.editor.setTheme(themeName);
    }
  } catch (error) {
    console.warn("Failed to sync VS Code theme into Monaco.", error);
    activeMonacoTheme = "vs-dark";
    monaco.editor.setTheme("vs-dark");
  }
}

export function getCurrentMonacoTheme() {
  return activeMonacoTheme;
}

export async function initializeMonacoVscode() {
  if (!monacoVscodeReady) {
    monacoVscodeReady = (async () => {
      ensureWorkerEnvironment();

      await initialize(
        {
          ...getExtensionsServiceOverride(),
          ...getFilesServiceOverride(),
          ...getLanguagesServiceOverride(),
          ...getTextmateServiceOverride(),
          ...getThemeServiceOverride(),
          ...getConfigurationServiceOverride(),
        },
        document.body,
        {
          configurationDefaults: {
            "workbench.colorTheme": VSCODE_COLOR_THEME,
          },
        },
      );

      try {
  await Promise.all([
    themeDefaultsReady(),
    javascriptExtensionReady(),
    jsonExtensionReady(),
    htmlExtensionReady(),
    cssExtensionReady(),
    typescriptBasicsExtensionReady(),
    typescriptLanguageFeaturesExtensionReady(),
  ]);
        await updateUserConfiguration(`{
          "workbench.colorTheme": "${VSCODE_COLOR_THEME}"
        }`);
        await applyCurrentVsCodeTheme();
      } catch (error) {
        console.warn("VS Code theme initialization failed. Falling back to Monaco dark theme.", error);
        activeMonacoTheme = "vs-dark";
        monaco.editor.setTheme("vs-dark");
      }
    })();
  }

  return monacoVscodeReady;
}
