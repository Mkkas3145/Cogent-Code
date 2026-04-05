import { useEffect, useLayoutEffect, useRef } from "react";
import { loader } from "@monaco-editor/react";
import type * as MonacoEditor from "monaco-editor";
import { getService, ILanguageService, IWorkbenchThemeService } from "@codingame/monaco-vscode-api";
import { getCurrentMonacoTheme } from "./monaco-vscode";

export type MonacoApi = typeof MonacoEditor;
export type MonacoStandaloneEditor = MonacoEditor.editor.IStandaloneCodeEditor;
export type MonacoConfigureContext = {
  monaco: MonacoApi;
  editor: MonacoStandaloneEditor;
  model: MonacoEditor.editor.ITextModel;
};

type MonacoCodeEditorProps = {
  path: string;
  value: string;
  language?: string;
  height?: string | number;
  className?: string;
  options?: MonacoEditor.editor.IStandaloneEditorConstructionOptions;
  onChange?: (value: string) => void;
  onMount?: (context: MonacoConfigureContext) => void;
  configureMonaco?: (monaco: MonacoApi) => void | (() => void);
  configureEditor?: (context: MonacoConfigureContext) => void | (() => void);
};

type TokenizedModel = MonacoEditor.editor.ITextModel & {
  tokenization?: {
    forceTokenization?: (lineNumber: number) => void;
  };
};

function normalizePathToUri(monaco: MonacoApi, path: string) {
  if (/^[a-z]+:\/\//i.test(path)) {
    return monaco.Uri.parse(path);
  }

  if (path.includes("?")) {
    return monaco.Uri.parse(`inmemory://${path.replace(/\\/g, "/")}`);
  }

  return monaco.Uri.file(path);
}

function isStandaloneResource(path: string) {
  return path.includes("?");
}

function getOrCreateStandaloneModel(monaco: MonacoApi, uri: MonacoEditor.Uri, value: string, language?: string) {
  const existingModel = monaco.editor.getModel(uri);
  if (existingModel) {
    return existingModel;
  }

  return monaco.editor.createModel(value, language, uri);
}

async function resolveLanguageId(
  uri: MonacoEditor.Uri,
  value: string,
  explicitLanguage?: string,
) {
  if (explicitLanguage) {
    return explicitLanguage;
  }

  const languageService = await getService(ILanguageService);
  const firstLine = value.split(/\r?\n/, 1)[0];
  return languageService.guessLanguageIdByFilepathOrFirstLine(uri, firstLine) ?? undefined;
}

function warmupTokenization(model: MonacoEditor.editor.ITextModel) {
  const tokenizedModel = model as TokenizedModel;
  if (!tokenizedModel.tokenization?.forceTokenization) {
    return () => {};
  }

  let cancelled = false;
  let timeoutId: number | null = null;
  let idleId: number | null = null;
  const lineCount = model.getLineCount();
  const step = Math.max(200, Math.min(2000, Math.ceil(lineCount / 20)));
  let nextLine = step;

  const schedule = (callback: () => void) => {
    if (cancelled) {
      return;
    }

    if (typeof window.requestIdleCallback === "function") {
      idleId = window.requestIdleCallback(() => {
        idleId = null;
        callback();
      });
      return;
    }

    timeoutId = window.setTimeout(() => {
      timeoutId = null;
      callback();
    }, 0);
  };

  const tokenizeChunk = () => {
    if (cancelled) {
      return;
    }

    tokenizedModel.tokenization?.forceTokenization?.(Math.min(nextLine, lineCount));
    if (nextLine >= lineCount) {
      return;
    }

    nextLine += step;
    schedule(tokenizeChunk);
  };

  schedule(tokenizeChunk);

  return () => {
    cancelled = true;
    if (timeoutId != null) {
      window.clearTimeout(timeoutId);
    }
    if (idleId != null && typeof window.cancelIdleCallback === "function") {
      window.cancelIdleCallback(idleId);
    }
  };
}

export function MonacoCodeEditor({
  path,
  value,
  language,
  height = "100%",
  className,
  options,
  onChange,
  onMount,
  configureMonaco,
  configureEditor,
}: MonacoCodeEditorProps) {
  const containerRef = useRef<HTMLDivElement | null>(null);
  const editorRef = useRef<MonacoStandaloneEditor | null>(null);
  const monacoRef = useRef<MonacoApi | null>(null);
  const disposablesRef = useRef<MonacoEditor.IDisposable[]>([]);
  const cancelWarmupRef = useRef<() => void>(() => {});

  useLayoutEffect(() => {
    let disposed = false;

    void loader.init().then((monacoInstance) => {
      if (disposed || !containerRef.current) {
        return;
      }

      const monaco = monacoInstance as MonacoApi;
      monacoRef.current = monaco;

      void (async () => {
        const configureMonacoCleanup = configureMonaco?.(monaco);
        const uri = normalizePathToUri(monaco, path);
        const resolvedLanguage = await resolveLanguageId(uri, value, language);
        if (disposed || !containerRef.current) {
          return;
        }
        const model = getOrCreateStandaloneModel(monaco, uri, value, resolvedLanguage);
        const initialTheme = getCurrentMonacoTheme();

        if (resolvedLanguage && model.getLanguageId() !== resolvedLanguage) {
          monaco.editor.setModelLanguage(model, resolvedLanguage);
        }

        if (initialTheme) {
          monaco.editor.setTheme(initialTheme);
        }

        const editor = monaco.editor.create(containerRef.current!, {
          automaticLayout: true,
          fontFamily: '"JetBrains Mono", "Pretendard", monospace',
          model,
          quickSuggestions: {
            other: true,
            comments: false,
            strings: false,
          },
          suggestOnTriggerCharacters: true,
          acceptSuggestionOnEnter: "on",
          wordBasedSuggestions: "off",
          tabCompletion: "on",
          ...options,
        });

        if (disposed) {
          editor.dispose();
          return;
        }

        editorRef.current = editor;
        cancelWarmupRef.current();
        cancelWarmupRef.current = warmupTokenization(model);

        void getService(IWorkbenchThemeService)
          .then((themeService) => {
            const syncTheme = () => {
              const activeTheme = themeService.getColorTheme();
              const themeName = activeTheme.id || activeTheme.settingsId || activeTheme.label;
              if (themeName) {
                monaco.editor.setTheme(themeName);
              }
            };

            syncTheme();
            const themeDisposable = themeService.onDidColorThemeChange(() => {
              syncTheme();
            });
            disposablesRef.current.push(themeDisposable);
          })
          .catch((error) => {
            console.warn("Failed to attach VS Code theme bridge.", error);
          });

        const changeDisposable = editor.onDidChangeModelContent(() => {
          onChange?.(editor.getValue());
        });

        disposablesRef.current.push(changeDisposable);

        const context = { monaco, editor, model };
        const configureEditorCleanup = configureEditor?.(context);
        onMount?.(context);

        if (configureMonacoCleanup) {
          disposablesRef.current.push({ dispose: configureMonacoCleanup });
        }
        if (configureEditorCleanup) {
          disposablesRef.current.push({ dispose: configureEditorCleanup });
        }
      })();
    });

    return () => {
      disposed = true;
      cancelWarmupRef.current();
      cancelWarmupRef.current = () => {};
      for (const disposable of disposablesRef.current) {
        disposable.dispose();
      }
      disposablesRef.current = [];
      const currentModel = editorRef.current?.getModel();
      editorRef.current?.dispose();
      editorRef.current = null;
      if (currentModel && isStandaloneResource(path)) {
        currentModel.dispose();
      }
    };
  }, []);

  useEffect(() => {
    const monaco = monacoRef.current;
    const editor = editorRef.current;
    if (!monaco || !editor) {
      return;
    }

    let cancelled = false;

    void (async () => {
      const uri = normalizePathToUri(monaco, path);
      const resolvedLanguage = await resolveLanguageId(uri, value, language);
      if (cancelled || editorRef.current !== editor) {
        return;
      }
      const model = getOrCreateStandaloneModel(monaco, uri, value, resolvedLanguage);

      if (resolvedLanguage && model.getLanguageId() !== resolvedLanguage) {
        monaco.editor.setModelLanguage(model, resolvedLanguage);
      }

      if (editor.getModel() !== model) {
        editor.setModel(model);
        cancelWarmupRef.current();
        cancelWarmupRef.current = warmupTokenization(model);
      }
    })();

    return () => {
      cancelled = true;
    };
  }, [language, path]);

  useEffect(() => {
    const editor = editorRef.current;
    if (!editor) {
      return;
    }

    const model = editor.getModel();
    if (!model) {
      return;
    }

    if (model.getValue() !== value) {
      model.pushEditOperations(
        [],
        [
          {
            range: model.getFullModelRange(),
            text: value,
          },
        ],
        () => null,
      );
    }
  }, [value]);

  useEffect(() => {
    const editor = editorRef.current;
    if (!editor || !options) {
      return;
    }

    editor.updateOptions(options);
  }, [options]);

  return <div ref={containerRef} className={className} style={{ height, width: "100%" }} />;
}
