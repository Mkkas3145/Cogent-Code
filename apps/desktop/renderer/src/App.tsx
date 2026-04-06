import { memo, useEffect, useLayoutEffect, useMemo, useRef, useState, type ChangeEvent, type DragEvent } from "react";
import { createPortal } from "react-dom";
import type {
  AgentMode,
  AgentStreamEvent,
  BrowserAssistRequest,
  BrowserAssistResult,
  ConversationTurn,
  ContextSnapshot,
  ContextUsageSnapshot,
  GeminiModelSummary,
  LmStudioModelSummary,
  ModelProvider,
  ModelTier,
  OllamaModelSummary,
  Thoroughness,
} from "@cogent/shared-types";
import { diffArrays as computeDiffArrays } from "diff";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { getIconUrlByName, getIconUrlForFilePath } from "vscode-material-icons";
import { MonacoCodeEditor, type MonacoApi, type MonacoStandaloneEditor } from "./MonacoCodeEditor";

type Locale = "en" | "ko" | "ja";
type FileTreeNode = {
  name: string;
  path: string;
  type: "directory" | "file";
  children?: FileTreeNode[];
};
type FilePreview = {
  path: string;
  content: string;
  missing?: boolean;
};
type OpenPathResult =
  | {
      type: "directory";
      rootPath: string;
      name: string;
      children: FileTreeNode[];
    }
  | {
      type: "file";
      rootPath: string;
      name: string;
      path: string;
      content: string;
      children: FileTreeNode[];
    };
type PendingModelSelectionWarning = {
  provider: "ollama" | "lmstudio";
  modelId: string;
  title: string;
  description: string;
};
type TextChunk = {
  id: number;
  start: number;
  end: number;
};

type ImageAttachment = {
  id: string;
  name: string;
  mimeType: string;
  dataUrl: string;
};

type FileAttachment = {
  id: string;
  name: string;
  mimeType: string;
  content: string;
};

type ChatMessage = {
  id: string;
  role: "user" | "assistant";
  text: string;
  attachments?: ImageAttachment[]; fileAttachments?: FileAttachment[];
  textChunks?: TextChunk[];
  streaming?: boolean;
  code?: string;
  meta?: string[];
  status?: "requesting" | "loading" | "thinking" | "writing";
  activityStatusLabel?: string;
  diffOriginal?: string;
  diffPath?: string;
  command?: {
    id: string;
    command: string;
    output: string;
    running: boolean;
    startedAt: number;
    exitCode?: number | null;
  };
};

type ChatSession = {
  id: string;
  name: string;
  titleState: "empty" | "generated";
  prompt: string;
  messages: ChatMessage[];
  context: ContextSnapshot | null;
  contextUsage: ContextUsageSnapshot | null;
};

type BrowserAssistPanelState = {
  request: BrowserAssistRequest;
  visible: boolean;
};

type BrowserAssistFloatingPosition = {
  left: number;
  top: number;
};

type RenderItem =
  | {
      type: "message";
      message: ChatMessage;
    }
  | {
      type: "tool-group";
      id: string;
      messages: ChatMessage[];
    };

function hasDiffPreview(message: ChatMessage) {
  return message.code !== undefined;
}

function isToolCardMessage(message: ChatMessage) {
  return message.role === "assistant" && (hasDiffPreview(message) || Boolean(message.command));
}

function hasRenderableMessageContent(message: ChatMessage) {
  return Boolean(
    message.text.trim() ||
      message.status ||
      message.code !== undefined ||
      message.meta?.length ||
      message.command ||
      message.attachments?.length,
  );
}

function isEphemeralAssistantPlaceholder(message: ChatMessage) {
  return (
    message.role === "assistant" &&
    !message.text.trim() &&
    !message.meta?.length &&
    message.code === undefined &&
    !message.command &&
    !message.attachments?.length &&
    !message.fileAttachments?.length
  );
}

function buildRenderItems(messages: ChatMessage[]) {
  const items: RenderItem[] = [];

  for (let index = 0; index < messages.length; index += 1) {
    const message = messages[index];
    if (!isToolCardMessage(message)) {
      items.push({ type: "message", message });
      continue;
    }

    const groupedMessages = [message];
    let cursor = index + 1;
    while (cursor < messages.length && isToolCardMessage(messages[cursor])) {
      groupedMessages.push(messages[cursor]);
      cursor += 1;
    }

    items.push({
      type: "tool-group",
      id: groupedMessages.map((entry) => entry.id).join(":"),
      messages: groupedMessages,
    });
    index = cursor - 1;
  }

  return items;
}

function findLatestToolGroupId(messages: ChatMessage[]) {
  const latestToolGroup = [...buildRenderItems(messages.filter(hasRenderableMessageContent))]
    .reverse()
    .find((item) => item.type === "tool-group");

  return latestToolGroup?.type === "tool-group" ? latestToolGroup.id : null;
}

function createSession(localeText: { emptySessionTitle: string }): ChatSession {
  return {
    id: `session-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`,
    name: localeText.emptySessionTitle,
    titleState: "empty",
    prompt: "",
    messages: [],
    context: null,
    contextUsage: null,
  };
}

function suggestSessionTitle(source: string, localeText: { emptySessionTitle: string }) {
  const normalized = source.replace(/\s+/g, " ").trim();
  if (!normalized) {
    return localeText.emptySessionTitle;
  }

  const firstSentence = normalized.split(/[.!?\n]/, 1)[0]?.trim() ?? normalized;
  const shortened = firstSentence.length > 32 ? `${firstSentence.slice(0, 32).trimEnd()}...` : firstSentence;
  return shortened || localeText.emptySessionTitle;
}

function getSessionPreview(session: ChatSession, localeText: { emptySession: string }) {
  const latestText = [...session.messages]
    .reverse()
    .map((message) => message.text.trim())
    .find((text) => text.length > 0);

  if (!latestText) {
    return localeText.emptySession;
  }

  const singleLine = latestText.replace(/\s+/g, " ").trim();
  return singleLine.length > 72 ? `${singleLine.slice(0, 72).trimEnd()}...` : singleLine;
}

function formatInteger(value: number) {
  return Math.round(value).toLocaleString();
}

function formatThousandsWithSuffix(value: number) {
  return `${Math.round(value / 1024).toLocaleString()}k`;
}

function getGeminiMenuDescription(model: GeminiModelSummary, locale: Locale) {
  const inputLimit = model.inputTokenLimit ? formatThousandsWithSuffix(model.inputTokenLimit) : null;
  const outputLimit = model.outputTokenLimit ? formatThousandsWithSuffix(model.outputTokenLimit) : null;
  const separator = " · ";

  if (locale === "ko") {
    return [inputLimit ? `입력 ${inputLimit}` : null, outputLimit ? `출력 ${outputLimit}` : null]
      .filter(Boolean)
      .join(separator) || "입력/출력 토큰 미확인";
  }
  if (locale === "ja") {
    return [inputLimit ? `入力 ${inputLimit}` : null, outputLimit ? `出力 ${outputLimit}` : null]
      .filter(Boolean)
      .join(separator) || "入力/出力トークン不明";
  }
  return [inputLimit ? `Input ${inputLimit}` : null, outputLimit ? `Output ${outputLimit}` : null]
    .filter(Boolean)
    .join(separator) || "Input/output limits unknown";
}

function getDefaultLmStudioContextLength(totalMemoryBytes?: number) {
  if (!totalMemoryBytes || totalMemoryBytes <= 0) {
    return "";
  }
  const totalMemoryGb = (totalMemoryBytes ?? 0) / 1024 / 1024 / 1024;
  const reservedForSystemGb = Math.max(4, totalMemoryGb * 0.25);
  const usableMemoryGb = Math.max(4, totalMemoryGb - reservedForSystemGb);
  const estimatedContext = Math.floor(usableMemoryGb * 512);
  const clampedContext = Math.max(4096, Math.min(32768, estimatedContext));
  return String(Math.round(clampedContext / 1024) * 1024);
}

function getDefaultOllamaContextLength(totalMemoryBytes?: number) {
  return getDefaultLmStudioContextLength(totalMemoryBytes);
}

function parseParameterBillions(rawValue: string | undefined) {
  if (!rawValue) {
    return null;
  }

  const match = rawValue.toLowerCase().match(/(\d+(?:\.\d+)?)\s*b/);
  if (!match) {
    return null;
  }

  const numericValue = Number(match[1]);
  return Number.isFinite(numericValue) ? numericValue : null;
}

function estimateModelWeightBytes(provider: "ollama" | "lmstudio", model: OllamaModelSummary | LmStudioModelSummary) {
  if ("size" in model && typeof model.size === "number" && model.size > 0) {
    return model.size;
  }

  const sourceText = provider === "ollama"
    ? [("parameterSize" in model ? model.parameterSize : undefined), model.name, "model" in model ? model.model : undefined].filter(Boolean).join(" ")
    : [model.name, "id" in model ? model.id : undefined].filter(Boolean).join(" ");
  const parameterBillions = parseParameterBillions(sourceText);
  if (!parameterBillions) {
    return null;
  }

  const normalizedSource = sourceText.toLowerCase();
  const bytesPerBillion =
    normalizedSource.includes("q2") ? 350_000_000
      : normalizedSource.includes("q3") ? 450_000_000
      : normalizedSource.includes("q4") ? 600_000_000
      : normalizedSource.includes("q5") ? 750_000_000
      : normalizedSource.includes("q6") ? 900_000_000
      : normalizedSource.includes("q8") ? 1_100_000_000
      : normalizedSource.includes("fp16") ? 2_100_000_000
      : 700_000_000;

  return Math.round(parameterBillions * bytesPerBillion);
}

function estimateKvCacheBytes(model: OllamaModelSummary | LmStudioModelSummary, requestedContextLength: number) {
  const sourceText = "parameterSize" in model
    ? [model.parameterSize, model.name, "model" in model ? model.model : undefined].filter(Boolean).join(" ")
    : [model.name, "id" in model ? model.id : undefined].filter(Boolean).join(" ");
  const parameterBillions = parseParameterBillions(sourceText);
  if (!parameterBillions || requestedContextLength <= 0) {
    return 0;
  }

  const familyFactor =
    "family" in model && typeof model.family === "string" && model.family.toLowerCase().includes("moe")
      ? 0.72
      : sourceText.toLowerCase().includes("moe")
        ? 0.72
        : 1;

  return Math.round(parameterBillions * requestedContextLength * 24_000 * familyFactor);
}

function findPreviousToolGroupId(messages: ChatMessage[], targetMessageId: string) {
  const items = buildRenderItems(messages.filter(hasRenderableMessageContent));
  const targetIndex = items.findIndex(
    (item) => item.type === "message" && item.message.id === targetMessageId,
  );

  if (targetIndex <= 0) {
    return null;
  }

  const previousItem = items[targetIndex - 1];
  return previousItem?.type === "tool-group" ? previousItem.id : null;
}

function getFileName(pathValue: string) {
  return pathValue.split(/[\\/]/).pop() ?? pathValue;
}

function CommandPreview({ command }: { command: NonNullable<ChatMessage["command"]> }) {
  const [elapsedSeconds, setElapsedSeconds] = useState(() => Math.max(0, Math.floor((Date.now() - command.startedAt) / 1000)));

  useEffect(() => {
    if (!command.running) {
      setElapsedSeconds(Math.max(0, Math.floor((Date.now() - command.startedAt) / 1000)));
      return;
    }

    const timer = window.setInterval(() => {
      setElapsedSeconds(Math.max(0, Math.floor((Date.now() - command.startedAt) / 1000)));
    }, 1000);

    return () => {
      window.clearInterval(timer);
    };
  }, [command.running, command.startedAt]);

  return (
    <div className="command-preview" aria-label="Command output">
      <div className="command-preview-header">
        <span className="command-preview-status">
          {command.running ? `${elapsedSeconds}초간 명령어 실행 중` : `명령어 실행 완료${command.exitCode != null ? ` (${command.exitCode})` : ""}`}
        </span>
        <code className="command-preview-command">{command.command}</code>
      </div>
      <pre className="command-preview-output">
        <code>{command.output || " "}</code>
      </pre>
    </div>
  );
}

function getToolGroupSummary(
  messages: ChatMessage[],
  localeText: {
    toolGroupDiffSingle: string;
    toolGroupDiffMulti: (count: number) => string;
    toolGroupCommandSingle: string;
    toolGroupCommandMulti: (count: number) => string;
    toolGroupMixed: (parts: string[]) => string;
  },
) {
  const diffCount = messages.filter(hasDiffPreview).length;
  const commandCount = messages.filter((message) => Boolean(message.command)).length;
  const totalCount = messages.length;

  if (diffCount === totalCount) {
    return totalCount === 1 ? localeText.toolGroupDiffSingle : localeText.toolGroupDiffMulti(totalCount);
  }

  if (commandCount === totalCount) {
    return totalCount === 1 ? localeText.toolGroupCommandSingle : localeText.toolGroupCommandMulti(totalCount);
  }

  const parts: string[] = [];
  if (diffCount > 0) {
    parts.push(diffCount === 1 ? localeText.toolGroupDiffSingle : localeText.toolGroupDiffMulti(diffCount));
  }
  if (commandCount > 0) {
    parts.push(commandCount === 1 ? localeText.toolGroupCommandSingle : localeText.toolGroupCommandMulti(commandCount));
  }

  return localeText.toolGroupMixed(parts);
}

function AssistantToolGroup({
  groupId,
  messages,
  collapsed,
  onToggle,
  localeText,
  selectedFilePath,
  onOpenDiffPath,
}: {
  groupId: string;
  messages: ChatMessage[];
  collapsed: boolean;
  onToggle: (groupId: string) => void;
  localeText: {
    toolGroupDiffSingle: string;
    toolGroupDiffMulti: (count: number) => string;
    toolGroupCommandSingle: string;
    toolGroupCommandMulti: (count: number) => string;
    toolGroupMixed: (parts: string[]) => string;
    toolItemDiff: string;
    toolItemCommand: string;
  };
  selectedFilePath?: string;
  onOpenDiffPath: (path: string) => void;
}) {
  const waferContentRef = useRef<HTMLDivElement | null>(null);
  const [waferHeight, setWaferHeight] = useState(0);
  const [waferReady, setWaferReady] = useState(false);
  const setWaferContentRef = (element: HTMLDivElement | null) => {
    waferContentRef.current = element;
    if (element) {
      setWaferHeight(element.scrollHeight);
    }
  };

  useLayoutEffect(() => {
    setWaferReady(false);
    const frame = window.requestAnimationFrame(() => {
      setWaferReady(true);
    });

    return () => {
      window.cancelAnimationFrame(frame);
    };
  }, [groupId]);

  useLayoutEffect(() => {
    const element = waferContentRef.current;
    if (!element) {
      return;
    }

    const updateHeight = () => {
      setWaferHeight(element.scrollHeight);
    };

    updateHeight();

    const observer = new ResizeObserver(() => {
      updateHeight();
    });
    observer.observe(element);

    return () => {
      observer.disconnect();
    };
  }, [messages, collapsed]);

  return (
    <div className="message-row is-assistant">
      <div className="tool-group-card">
        <button className={`tool-group-header ${collapsed ? "is-collapsed" : ""}`} onClick={() => onToggle(groupId)} type="button">
          <span className="tool-group-header-main">
            <span className="tool-group-title">{getToolGroupSummary(messages, localeText)}</span>
            <span className="tool-group-count">{messages.length}</span>
          </span>
          <span className={`tool-group-chevron ${collapsed ? "is-collapsed" : ""}`} aria-hidden="true">
            <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 32 20">
              <path d="M31.998 4.144a1.5 1.5 0 0 0-.498-1.033 1.5 1.5 0 0 0-2.117.117L17.674 16.311c-.85.95-2.274.956-3.133.013L2.61 3.218a1.5 1.5 0 0 0-2.12-.1 1.5 1.5 0 0 0-.1 2.12l11.932 13.106c2.023 2.222 5.585 2.206 7.588-.033L31.617 5.228a1.5 1.5 0 0 0 .381-1.084Z" />
            </svg>
          </span>
        </button>
        <div className={`tool-group-wafer ${collapsed ? "is-collapsed" : ""} ${waferReady ? "is-ready" : ""}`} style={{ height: collapsed ? 0 : waferHeight }}>
          <div ref={setWaferContentRef} className={`tool-group-body-wrap ${collapsed ? "is-collapsed" : ""}`}>
            <div className="tool-group-body">
              {messages.map((message) => (
                <div key={message.id} className="tool-group-item">
                  <div className="tool-group-item-label">
                    {message.command ? localeText.toolItemCommand : localeText.toolItemDiff}
                    {message.diffPath ? ` · ${getFileName(message.diffPath)}` : ""}
                  </div>
                  {hasDiffPreview(message) ? (
                    <ChatDiffPreview
                      original={message.diffOriginal ?? ""}
                      modified={message.code ?? ""}
                      path={message.diffPath ?? selectedFilePath ?? "chat-diff.ts"}
                      openPath={message.diffPath}
                      onOpenPath={onOpenDiffPath}
                    />
                  ) : null}
                  {message.command ? <CommandPreview command={message.command} /> : null}
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

type UnifiedDiffLine = {
  kind: "context" | "add" | "remove";
  text: string;
  originalLineNumber?: number;
  modifiedLineNumber?: number;
};

const MATERIAL_ICONS_BASE_URL = "./material-icons";

function IconGlyph({ svg, className }: { svg: string; className?: string }) {
  return <span className={className} aria-hidden="true" dangerouslySetInnerHTML={{ __html: svg }} />;
}

function getChatDiffHeight(original: string, modified: string) {
  const lineCount = Math.max(buildUnifiedDiffLines(original, modified).length, 1);
  if (lineCount <= 8) {
    return Math.max(lineCount * 22 + 18, 160);
  }
  return 420;
}

function getStableDiffKey(path: string, original: string, modified: string) {
  const source = `${path}\u0000${original}\u0000${modified}`;
  let hash = 0;

  for (let index = 0; index < source.length; index += 1) {
    hash = (hash * 31 + source.charCodeAt(index)) | 0;
  }

  return `${path}:${Math.abs(hash)}`;
}

function getPathLabel(path: string | undefined) {
  if (!path) {
    return "file";
  }

  return path.replace(/\\/g, "/").split("/").pop() ?? path;
}

function formatCustomStatusLabel(locale: Locale, label: string) {
  if (label === "Requesting" || label === "Loading" || label === "Thinking") {
    return null;
  }

  if (locale === "ko") {
    if (label.startsWith("Searching the web for ")) return "웹 검색 중";
    if (label.startsWith("Opening ")) return "브라우저 탐색 중";
    if (label.startsWith("Clicking ")) return "브라우저 클릭 중";
    if (label.startsWith("Scrolling page by ")) return "브라우저 스크롤 중";
    if (label.startsWith("Typing into ")) return "브라우저 입력 중";
    if (label.startsWith("Pressing ")) return "키 누르는 중";
    if (label.startsWith("Dragging ")) return "드래그 중";
    if (label.startsWith("Waiting for browser help: ")) return "브라우저 도움 대기 중";
    if (label === "Capturing browser screenshot") return "브라우저 캡처 중";
    if (label === "Capturing app window") return "앱 화면 캡처 중";
    if (label.startsWith("Capturing app window ")) return "앱 화면 캡처 중";
    if (label.startsWith("Starting command session ")) return "명령 세션 시작 중";
    if (label.startsWith("Sending input to command session ")) return "명령 세션 입력 중";
    if (label.startsWith("Reading command session ")) return "명령 세션 확인 중";
    if (label.startsWith("Stopping command session ")) return "명령 세션 종료 중";
    if (label.startsWith("Exploring ")) return `${getPathLabel(label.slice("Exploring ".length))} 탐색 중`;
    if (label.startsWith("Writing ")) return `${getPathLabel(label.slice("Writing ".length))} 파일 수정 중`;
    if (label.startsWith("Creating ")) return `${getPathLabel(label.slice("Creating ".length))} 파일 생성 중`;
    if (label.startsWith("Deleting ")) return `${getPathLabel(label.slice("Deleting ".length))} 삭제 중`;
  }

  if (locale === "ja") {
    if (label.startsWith("Searching the web for ")) return "Web ?索中";
    if (label.startsWith("Opening ")) return "ブラウザ?探索中";
    if (label.startsWith("Clicking ")) return "ブラウザ?クリック中";
    if (label.startsWith("Scrolling page by ")) return "ブラウザ?スクロ?ル中";
    if (label.startsWith("Typing into ")) return "ブラウザ?入力中";
    if (label.startsWith("Pressing ")) return "キー入力中";
    if (label.startsWith("Dragging ")) return "ドラッグ中";
    if (label.startsWith("Waiting for browser help: ")) return "ブラウザ操作を待機中";
    if (label === "Capturing browser screenshot") return "ブラウザをキャプチャ中";
    if (label === "Capturing app window") return "アプリ画面をキャプチャ中";
    if (label.startsWith("Capturing app window ")) return "アプリ画面をキャプチャ中";
    if (label.startsWith("Starting command session ")) return "コマンドセッション開始中";
    if (label.startsWith("Sending input to command session ")) return "コマンドセッション入力中";
    if (label.startsWith("Reading command session ")) return "コマンドセッション確認中";
    if (label.startsWith("Stopping command session ")) return "コマンドセッション終了中";
    if (label.startsWith("Exploring ")) return `${getPathLabel(label.slice("Exploring ".length))} を探索中`;
    if (label.startsWith("Writing ")) return `${getPathLabel(label.slice("Writing ".length))} を編集中`;
    if (label.startsWith("Creating ")) return `${getPathLabel(label.slice("Creating ".length))} を作成中`;
    if (label.startsWith("Deleting ")) return `${getPathLabel(label.slice("Deleting ".length))} を削除中`;
  }

  return `${label}...`;
}

function getStatusLabel(
  localeText: {
    statusRequesting: string;
    statusLoading: string;
    statusThinking: string;
    statusWriting: (fileName: string) => string;
  },
  status: ChatMessage["status"],
  activityStatusLabel: string | undefined,
  path: string | undefined,
  locale: Locale,
) {
  const customLabel = activityStatusLabel ? formatCustomStatusLabel(locale, activityStatusLabel) : null;
  if (customLabel) {
    return customLabel;
  }
  const fileName = getPathLabel(path);
  if (status === "requesting") return localeText.statusRequesting;
  if (status === "loading") return localeText.statusLoading;
  if (status === "writing") return localeText.statusWriting(fileName);
  return localeText.statusThinking;
}

function buildUnifiedDiffLines(original: string, modified: string): UnifiedDiffLine[] {
  const lines: UnifiedDiffLine[] = [];
  const originalLines = original.replace(/\r\n/g, "\n").split("\n");
  const modifiedLines = modified.replace(/\r\n/g, "\n").split("\n");
  const changes = computeDiffArrays(originalLines, modifiedLines);
  let originalLineNumber = 1;
  let modifiedLineNumber = 1;

  for (const change of changes) {
    const valueLines = change.value;

    for (const line of valueLines) {
      if (change.added) {
        lines.push({ kind: "add", text: line, modifiedLineNumber });
        modifiedLineNumber += 1;
      } else if (change.removed) {
        lines.push({ kind: "remove", text: line, originalLineNumber });
        originalLineNumber += 1;
      } else {
        lines.push({
          kind: "context",
          text: line,
          originalLineNumber,
          modifiedLineNumber,
        });
        originalLineNumber += 1;
        modifiedLineNumber += 1;
      }
    }
  }

  return lines;
}

function ChatDiffPreview({
  original,
  modified,
  path,
  openPath,
  onOpenPath,
}: {
  original: string;
  modified: string;
  path: string;
  openPath?: string;
  onOpenPath?: (path: string) => void;
}) {
  const diffLines = useMemo(() => buildUnifiedDiffLines(original, modified), [original, modified]);
  const diffValue = useMemo(() => diffLines.map((line) => line.text).join("\n"), [diffLines]);
  const diffEditorKey = useMemo(() => getStableDiffKey(path, original, modified), [path, original, modified]);
  const fileName = useMemo(() => path.replace(/\\/g, "/").split("/").pop() ?? path, [path]);
  const firstChangedLineNumber = useMemo(
    () => diffLines.find((line) => line.kind !== "context")?.modifiedLineNumber ?? diffLines.findIndex((line) => line.kind !== "context") + 1,
    [diffLines],
  );
  const summary = useMemo(() => {
    let additions = 0;
    let removals = 0;

    for (const line of diffLines) {
      if (line.kind === "add") additions += 1;
      if (line.kind === "remove") removals += 1;
    }

    return { additions, removals };
  }, [diffLines]);
  const displayLineNumbers = useMemo(
    () =>
      diffLines.map((line) => {
        if (line.kind === "remove") {
          return line.originalLineNumber;
        }

        return line.modifiedLineNumber ?? line.originalLineNumber;
      }),
    [diffLines],
  );

  return (
    <div className="chat-diff" aria-label="Code diff preview">
      <button className="chat-diff-header" type="button" onClick={() => openPath && onOpenPath?.(openPath)}>
        <div className="chat-diff-title">
          <span className="chat-diff-file">{fileName}</span>
          <span className="chat-diff-stats">
            <span className="chat-diff-stat add">+{summary.additions}</span>
            <span className="chat-diff-stat remove">-{summary.removals}</span>
          </span>
        </div>
      </button>
      <MonacoCodeEditor
        key={diffEditorKey}
        path={`${path}?chat-diff=${encodeURIComponent(diffEditorKey)}`}
        language={undefined}
        value={diffValue}
        height={`${getChatDiffHeight(original, modified)}px`}
        configureEditor={({ monaco, editor, model }) => {
          handleEditorMount(editor, monaco);
          editor.updateOptions({ "semanticHighlighting.enabled": true } as never);
          const decorations = diffLines.flatMap((line, index) => {
            if (line.kind === "context") {
              return [];
            }

            return [
              {
                range: new monaco.Range(index + 1, 1, index + 1, 1),
                options: {
                  isWholeLine: true,
                  className: line.kind === "add" ? "chat-diff-line-add" : "chat-diff-line-remove",
                  lineNumberClassName:
                    line.kind === "add" ? "chat-diff-line-number-add" : "chat-diff-line-number-remove",
                  glyphMarginClassName:
                    line.kind === "add" ? "chat-diff-glyph-add" : "chat-diff-glyph-remove",
                  linesDecorationsClassName:
                    line.kind === "add" ? "chat-diff-gutter-add" : "chat-diff-gutter-remove",
                  firstLineDecorationClassName:
                    line.kind === "add" ? "chat-diff-gutter-add" : "chat-diff-gutter-remove",
                },
              },
            ];
          });

          const collection = editor.createDecorationsCollection(decorations);
          if (firstChangedLineNumber > 0) {
            editor.revealLineInCenter(firstChangedLineNumber);
          }
          return () => collection.clear();
        }}
        options={{
          readOnly: true,
          minimap: {
            enabled: true,
            side: "right",
            size: "proportional",
            showSlider: "always",
            renderCharacters: false,
          },
          lineNumbers: (lineNumber) => `${displayLineNumbers[lineNumber - 1] ?? ""}`,
          glyphMargin: true,
          folding: false,
          scrollBeyondLastLine: true,
          overviewRulerLanes: 0,
          wordWrap: "off",
          lineNumbersMinChars: 3,
          lineDecorationsWidth: 10,
          renderLineHighlight: "none",
          smoothScrolling: true,
          guides: {
            indentation: false,
          },
          scrollbar: {
            vertical: "hidden",
            horizontal: "hidden",
            verticalScrollbarSize: 0,
            horizontalScrollbarSize: 0,
            alwaysConsumeMouseWheel: false,
          },
        }}
      />
    </div>
  );
}

function getModeLabel(mode: AgentMode | "auto", localeText: { auto: string; backend: string; frontend: string }) {
  if (mode === "auto") {
    return localeText.auto;
  }
  return mode === "backend" ? localeText.backend : localeText.frontend;
}

type MarkdownNode = {
  type: string;
  value?: string;
  children?: MarkdownNode[];
  data?: {
    hName?: string;
    hProperties?: Record<string, unknown>;
    hChildren?: MarkdownNode[];
  };
  position?: {
    start?: { offset?: number };
    end?: { offset?: number };
  };
};

function getMarkdownChunkClass(chunkId: number, readyChunkId: number) {
  return chunkId > readyChunkId ? 'markdown-chunk-pending' : 'markdown-chunk-new';
}

function createAnimatedTextNode(value: string, chunkId: number, readyChunkId: number, chunkKey: string): MarkdownNode {
  return {
    type: 'animated-text',
    data: {
      hName: 'span',
      hProperties: {
        className: [getMarkdownChunkClass(chunkId, readyChunkId)],
        'data-chunk-id': String(chunkId),
        'data-chunk-key': chunkKey,
      },
      hChildren: [{ type: 'text', value }],
    },
  };
}

function appendChunkClass(child: MarkdownNode, chunkId: number, readyChunkId: number, chunkKey: string) {
  const nextClassName = getMarkdownChunkClass(chunkId, readyChunkId);
  const existingClassName = child.data?.hProperties?.className;
  const normalizedClassName = Array.isArray(existingClassName)
    ? existingClassName
    : typeof existingClassName === 'string'
      ? [existingClassName]
      : [];

  if (normalizedClassName.includes(nextClassName)) {
    return;
  }

  child.data = {
    ...child.data,
    hProperties: {
      ...(child.data?.hProperties ?? {}),
      className: [...normalizedClassName, nextClassName],
      'data-chunk-id': String(chunkId),
      'data-chunk-key': chunkKey,
    },
  };
}

function splitTextNodeByChunks(
  child: MarkdownNode,
  chunks: TextChunk[],
  startOffset: number,
  endOffset: number,
  readyChunkId: number,
) {
  const value = child.value ?? '';
  const breakpoints = new Set([startOffset, endOffset]);

  for (const chunk of chunks) {
    if (chunk.end <= startOffset || chunk.start >= endOffset) {
      continue;
    }
    breakpoints.add(Math.max(startOffset, chunk.start));
    breakpoints.add(Math.min(endOffset, chunk.end));
  }

  const ordered = [...breakpoints].sort((a, b) => a - b);
  const segments = [];

  for (let index = 0; index < ordered.length - 1; index += 1) {
    const segmentStart = ordered[index];
    const segmentEnd = ordered[index + 1];
    if (segmentEnd <= segmentStart) continue;

    const segmentValue = value.slice(segmentStart - startOffset, segmentEnd - startOffset);
    if (!segmentValue) continue;

    const matchingChunk = chunks.find((chunk) => chunk.start < segmentEnd && chunk.end > segmentStart);
    if (matchingChunk) {
      segments.push(
        createAnimatedTextNode(
          segmentValue,
          matchingChunk.id,
          readyChunkId,
          `${matchingChunk.id}:${segmentStart}-${segmentEnd}`,
        ),
      );
    } else {
      segments.push({ ...child, value: segmentValue });
    }
  }

  return segments;
}

function annotateMarkdownChunks(children: MarkdownNode[] | undefined, chunks: TextChunk[], readyChunkId: number) {
  if (!children?.length || chunks.length === 0) return;

  for (let index = 0; index < children.length; index += 1) {
    const child = children[index];
    const startOffset = child.position?.start?.offset;
    const endOffset = child.position?.end?.offset;

    if (typeof startOffset !== 'number' || typeof endOffset !== 'number') {
      if (child.children?.length) annotateMarkdownChunks(child.children, chunks, readyChunkId);
      continue;
    }

    const relevantChunks = chunks.filter((chunk) => chunk.end > startOffset && chunk.start < endOffset);
    if (relevantChunks.length === 0) continue;

    if (child.type === 'text') {
      const replacement = splitTextNodeByChunks(child, relevantChunks, startOffset, endOffset, readyChunkId);
      children.splice(index, 1, ...replacement);
      index += replacement.length - 1;
      continue;
    }

    const fullyContainedChunk = relevantChunks.find((chunk) => chunk.start <= startOffset && chunk.end >= endOffset);
    if (fullyContainedChunk) {
      appendChunkClass(
        child,
        fullyContainedChunk.id,
        readyChunkId,
        `${fullyContainedChunk.id}:${startOffset}-${endOffset}`,
      );
    }

    if (child.children?.length) {
      annotateMarkdownChunks(child.children, relevantChunks, readyChunkId);
    }
  }
}

function createMarkdownChunkPlugin(chunks: TextChunk[], readyChunkId: number) {
  return () => (tree: MarkdownNode) => {
    annotateMarkdownChunks(tree.children, chunks, readyChunkId);
  };
}

const AnimatedMarkdown = memo(function AnimatedMarkdown({
  text,
  chunks = [],
  animate = false,
  onOpenLink,
}: {
  text: string;
  chunks?: TextChunk[];
  animate?: boolean;
  onOpenLink?: (url: string, label?: string) => void;
}) {
  const initialChunkCutoffRef = useRef<number | null>(null);
  const containerRef = useRef<HTMLDivElement | null>(null);
  const animatedChunkKeysRef = useRef<Set<string>>(new Set());

  if (initialChunkCutoffRef.current === null) {
    initialChunkCutoffRef.current = animate ? 0 : chunks.reduce((max, chunk) => Math.max(max, chunk.id), 0);
  }

  const activeChunks = useMemo(
    () => chunks.filter((chunk) => chunk.id > (initialChunkCutoffRef.current ?? 0)),
    [chunks],
  );
  const latestChunkId = activeChunks.reduce((max, chunk) => Math.max(max, chunk.id), 0);
  const [readyChunkId, setReadyChunkId] = useState(() => (animate ? 0 : latestChunkId));
  const previousLatestChunkIdRef = useRef(latestChunkId);
  const effectiveReadyChunkId =
    animate && previousLatestChunkIdRef.current === 0 && latestChunkId > 0 ? latestChunkId : readyChunkId;

  useLayoutEffect(() => {
    if (!animate) {
      setReadyChunkId(latestChunkId);
      previousLatestChunkIdRef.current = latestChunkId;
      return;
    }

    if (latestChunkId === 0 || latestChunkId <= readyChunkId) {
      previousLatestChunkIdRef.current = latestChunkId;
      return;
    }

    const previousLatestChunkId = previousLatestChunkIdRef.current;
    previousLatestChunkIdRef.current = latestChunkId;

    if (previousLatestChunkId === 0) {
      setReadyChunkId(latestChunkId);
      return;
    }

    const frame = window.requestAnimationFrame(() => {
      setReadyChunkId(latestChunkId);
    });

    return () => {
      window.cancelAnimationFrame(frame);
    };
  }, [animate, latestChunkId, readyChunkId]);

  const remarkPlugins = useMemo(
    () =>
      activeChunks.length > 0 ? [remarkGfm, createMarkdownChunkPlugin(activeChunks, effectiveReadyChunkId)] : [remarkGfm],
    [activeChunks, effectiveReadyChunkId],
  );

  useLayoutEffect(() => {
    if (!containerRef.current) {
      return;
    }

    const targets = containerRef.current.querySelectorAll<HTMLElement>("[data-chunk-key]");
    for (const target of targets) {
      const chunkKey = target.dataset.chunkKey;
      const chunkId = Number(target.dataset.chunkId);
      if (!chunkKey || !Number.isFinite(chunkId)) {
        continue;
      }

      if (chunkId <= (initialChunkCutoffRef.current ?? 0)) {
        animatedChunkKeysRef.current.add(chunkKey);
      }
    }
  }, []);

  useLayoutEffect(() => {
    if (!animate || !containerRef.current) {
      return;
    }

    const targets = containerRef.current.querySelectorAll<HTMLElement>("[data-chunk-key]");
    for (const target of targets) {
      const chunkKey = target.dataset.chunkKey;
      if (!chunkKey || animatedChunkKeysRef.current.has(chunkKey)) {
        continue;
      }

      animatedChunkKeysRef.current.add(chunkKey);
      const animation = target.animate(
        [
          { opacity: 0, transform: "translateY(6px)", filter: "blur(6px)" },
          { opacity: 1, transform: "translateY(0)", filter: "blur(0)" },
        ],
        {
          duration: 500,
          easing: "ease",
        },
      );
      animation.onfinish = () => {
        target.style.opacity = "";
        target.style.transform = "";
        target.style.filter = "";
        target.style.willChange = "";
      };
    }
  }, [animate, activeChunks]);

  return (
    <div ref={containerRef} className="markdown-body">
      <ReactMarkdown
        remarkPlugins={remarkPlugins}
        components={{
          a: ({ href, children, ...props }) => (
            <a
              {...props}
              href={href}
              onClick={(event) => {
                if (!href) {
                  return;
                }
                event.preventDefault();
                onOpenLink?.(href, typeof children === "string" ? children : undefined);
              }}
            >
              {children}
            </a>
          ),
        }}
      >
        {text}
      </ReactMarkdown>
    </div>
  );
});

type OpenMenu = "attach" | "files" | "mode" | "model" | "thoroughness" | null;
const GEMINI_API_KEY_STORAGE_KEY = "cogent.geminiApiKey";
const MODEL_PROVIDER_STORAGE_KEY = "cogent.modelProvider";
const GEMINI_MODEL_STORAGE_KEY = "cogent.geminiModel";
const OLLAMA_MODEL_STORAGE_KEY = "cogent.ollamaModel";
const OLLAMA_CONTEXT_LENGTH_STORAGE_KEY = "cogent.ollamaContextLength";
const LMSTUDIO_MODEL_STORAGE_KEY = "cogent.lmstudioModel";
const LMSTUDIO_CONTEXT_LENGTH_STORAGE_KEY = "cogent.lmstudioContextLength";
const AUTO_COLLAPSE_TOOL_GROUPS_STORAGE_KEY = "cogent.autoCollapseToolGroups";
const COMMAND_REVIEW_MODE_STORAGE_KEY = "cogent.commandReviewMode";
const CHAT_SESSIONS_STORAGE_KEY = "cogent.chatSessions";
const ACTIVE_SESSION_STORAGE_KEY = "cogent.activeSessionId";
const GLOBAL_SYSTEM_PROMPT_STORAGE_KEY = "cogent.globalSystemPrompt";
const MODEL_INPUT_TOKEN_LIMITS: Record<ModelTier, number> = {
  "flash-lite": 1_048_576,
  flash: 1_048_576,
  pro: 1_048_576,
};

const COPY = {
  en: {
    title: "Cogent Code",
    subtitle: "AI coding workspace with quiet tool use and chat-first flow.",
    systemLabel: "System",
    systemText: "Search-based reasoning, hidden command execution, and automatic context compression are active.",
    you: "You",
    agent: "Cogent",
    trace: "Trace",
    start: "Start a conversation to begin.",
    emptyHeroTitle: "Start With Cogent",
    emptyHeroDescription: "Ask for edits, debugging, file exploration, or command execution to get moving.",
    workspace: "Workspace",
    files: "Files",
    workspaceHint: "Hover to browse files without pinning the panel open.",
    send: "Send",
    inputPlaceholder: "Ask the agent what you want to build, fix, or explore.",
    thoroughnessLight: "Light",
    thoroughnessBalanced: "Balanced",
    thoroughnessDeep: "Deep",
    auto: "Auto",
    backend: "Backend",
    frontend: "Frontend",
    lockMode: "Lock mode selection",
    modelTitle: "Model",
    modelDescription: "Pick how fast or deep the agent should respond.",
    modelGeminiSection: "Gemini",
    modelOllamaSection: "Local Ollama",
    modelLmStudioSection: "Local LM Studio",
    modelOllamaDescription: "Detected local models available through Ollama.",
    modeTitle: "Mode",
    modeDescription: "Match the agent to the kind of code you are shaping.",
    thoroughnessTitle: "Thoroughness",
    thoroughnessDescription: "Control how lightweight or careful the vibe coding flow should feel.",
    modelFlashLiteDescription: "Fastest loop for quick iteration.",
    modelFlashDescription: "Balanced speed and quality.",
    modelProDescription: "More reasoning for heavier tasks.",
    autoDescription: "Let the agent choose between backend and frontend flow.",
    backendDescription: "For APIs, services, data flow, and server logic.",
    frontendDescription: "For UI work across HTML, CSS, JavaScript, and React.",
    thoroughnessLightDescription: "Move fast with lighter checking.",
    thoroughnessBalancedDescription: "Keep speed and care in balance.",
    thoroughnessDeepDescription: "Inspect more before making changes.",
    saved: "Saved",
    saving: "Saving...",
    modified: "Modified",
    openFolderTitle: "Open Folder",
    openFolderDescription: "Open an existing project folder and jump straight into the workspace.",
    newWindowTitle: "New Window",
    newWindowDescription: "Start another workspace in a separate window.",
    openConsoleTitle: "Open Command Prompt in New Window",
    openConsoleDescription: "Open a new Command Prompt window rooted at the current workspace folder.",
    settingsTitle: "Settings",
    settingsDescription: "Adjust app behavior and connected services.",
    settingsClose: "Close settings",
    settingsApiKeyLabel: "Gemini API Key",
    settingsApiKeyDescription: "Enter the API key used to call Gemini models.",
    geminiApiKeyRequiredMessage: "Add your Gemini API key in Settings to use cloud models.",
    settingsOllamaLabel: "Local Ollama",
    settingsOllamaAvailable: "Detected and ready to use.",
    settingsOllamaUnavailable: "Not detected on this computer.",
    settingsOllamaContextLengthLabel: "Ollama context length",
    settingsOllamaContextLengthDescription: "Lower this to reduce memory usage. Example: 8192",
    settingsLmStudioContextLengthLabel: "LM Studio context length",
    settingsLmStudioContextLengthDescription: "Lower this to reduce memory usage. Example: 8192",
    settingsAutomaticPlaceholder: "Calculates the maximum by considering available system headroom.",
    settingsCommandReviewLabel: "Command review mode",
    settingsCommandReviewDescription: "Show a confirmation dialog before running any shell command.",
    commandReviewTitle: "Run this command?",
    commandReviewApprove: "Run",
    settingsAutoCollapseLabel: "Auto-collapse tool cards",
    settingsAutoCollapseDescription: "Collapse diff and command groups automatically when they appear.",
    settingsGlobalSystemPromptLabel: "Global system prompt",
    settingsGlobalSystemPromptDescription: "Always prepend this instruction set to every agent request.",
    settingsResetAppLabel: "Reset app",
    settingsResetAppDescription: "Clear sessions and local settings, then return the app to its initial state.",
    settingsResetAppAction: "Reset app",
    modelWarningTitle: "This model may fail to load",
    modelWarningDescription: "The current local model settings look heavy for this system. It may fail to load or become unstable.",
    modelWarningConfirm: "Select anyway",
    modelWarningMenu: "May fail to load",
    resetWarningTitle: "Reset this app?",
    resetWarningDescription: "This clears saved sessions and local app settings. Project files and the open workspace stay untouched.",
    resetWarningConfirm: "Reset",
    cancel: "Cancel",
    save: "Save",
    delete: "Delete",
    deleteMessage: "Delete message",
    sessions: "Sessions",
    newSession: "New Session",
    emptySessionTitle: "Empty Session",
    emptySession: "No messages yet.",
    contextUsage: "Context Window",
    contextUsageTitle: (percent: number) => `Context Window: ${percent}%`,
    contextState: "State",
    contextSnippets: "Snippets",
    contextEstimate: "tokens used",
    toolGroupDiffSingle: "File edit",
    toolGroupDiffMulti: (count: number) => `${count} file edits`,
    toolGroupCommandSingle: "Command run",
    toolGroupCommandMulti: (count: number) => `${count} commands`,
    toolGroupMixed: (parts: string[]) => parts.join(" · "),
    toolItemDiff: "File edit",
    toolItemCommand: "Command",
    fileReadSingle: (fileName: string) => `Cogent read ${fileName}.`,
    fileReadRange: (fileName: string, startLine: number, endLine: number) =>
      `Cogent read ${fileName} from line ${startLine} to line ${endLine}.`,
    fileDelete: (fileName: string) => `Cogent deleted ${fileName}.`,
    fileMove: (fromName: string, toName: string) => `Cogent moved ${fromName} → ${toName}.`,
    directoryDelete: (directoryName: string) => `Cogent deleted the ${directoryName} folder.`,
    statusRequesting: "Requesting",
    statusLoading: "Loading",
    statusThinking: "Thinking",
    statusWriting: (fileName: string) => `Editing ${fileName}`,
  },
  ko: {
    title: "Cogent Code",
    subtitle: "조용한 도구 실행과 채팅 중심 흐름을 갖춘 AI 코딩 워크스페이스.",
    systemLabel: "시스템",
    systemText: "검색 기반 추론, 숨겨진 명령 실행, 자동 컨텍스트 압축이 활성화되어 있습니다.",
    you: "나",
    agent: "Cogent",
    trace: "기록",
    start: "대화를 시작해 보세요.",
    emptyHeroTitle: "Cogent로 시작하기",
    emptyHeroDescription: "코드 수정, 디버깅, 파일 탐색, 명령 실행 같은 작업을 요청해 보세요.",
    workspace: "워크스페이스",
    files: "파일",
    workspaceHint: "패널을 고정하지 않고 호버만으로 파일을 탐색할 수 있습니다.",
    send: "전송",
    inputPlaceholder: "만들 것, 고칠 것, 살펴볼 것을 에이전트에게 요청해 보세요.",
    thoroughnessLight: "빠르게",
    thoroughnessBalanced: "균형",
    thoroughnessDeep: "깊게",
    auto: "자동",
    backend: "백엔드",
    frontend: "프론트엔드",
    lockMode: "모드 선택 고정",
    modelTitle: "모델",
    modelDescription: "에이전트의 응답 속도와 추론 깊이를 고릅니다.",
    modelGeminiSection: "Gemini",
    modelOllamaSection: "로컬 Ollama",
    modelLmStudioSection: "로컬 LM Studio",
    modelOllamaDescription: "이 컴퓨터에서 감지된 로컬 모델입니다.",
    modeTitle: "모드",
    modeDescription: "현재 다루는 코드 성격에 맞게 에이전트를 맞춥니다.",
    thoroughnessTitle: "깊이",
    thoroughnessDescription: "바이브 코딩 흐름을 얼마나 가볍게 혹은 신중하게 가져갈지 정합니다.",
    modelFlashLiteDescription: "빠른 반복에 맞는 가장 빠른 응답.",
    modelFlashDescription: "속도와 품질의 균형.",
    modelProDescription: "무거운 작업을 위한 더 깊은 추론.",
    autoDescription: "에이전트가 백엔드와 프론트엔드 흐름 중 적절한 쪽을 고릅니다.",
    backendDescription: "API, 서비스, 데이터 흐름, 서버 로직에 적합합니다.",
    frontendDescription: "HTML, CSS, JavaScript, React 기반 UI 작업에 적합합니다.",
    thoroughnessLightDescription: "검증을 가볍게 하고 빠르게 진행합니다.",
    thoroughnessBalancedDescription: "속도와 신중함을 균형 있게 가져갑니다.",
    thoroughnessDeepDescription: "수정 전에 더 많이 확인합니다.",
    saved: "저장됨",
    saving: "저장 중...",
    modified: "수정됨",
    openFolderTitle: "폴더 열기",
    openFolderDescription: "기존 프로젝트 폴더를 열고 바로 작업을 이어갑니다.",
    newWindowTitle: "새 창",
    newWindowDescription: "별도의 창에서 다른 워크스페이스를 시작합니다.",
    openConsoleTitle: "새 명령 프롬프트 창 열기",
    openConsoleDescription: "현재 워크스페이스 폴더 기준으로 새 명령 프롬프트 창을 엽니다.",
    settingsTitle: "설정",
    settingsDescription: "앱 동작과 연결된 서비스를 조정합니다.",
    settingsClose: "설정 닫기",
    settingsApiKeyLabel: "Gemini API Key",
    settingsApiKeyDescription: "Gemini 모델 호출에 사용할 API Key를 입력합니다.",
    geminiApiKeyRequiredMessage: "클라우드 모델을 쓰려면 설정에서 Gemini API Key를 입력해 주세요.",
    settingsOllamaLabel: "로컬 Ollama",
    settingsOllamaAvailable: "감지되었고 바로 사용할 수 있습니다.",
    settingsOllamaUnavailable: "이 컴퓨터에서 감지되지 않았습니다.",
    settingsOllamaContextLengthLabel: "Ollama 컨텍스트 길이",
    settingsOllamaContextLengthDescription: "메모리를 줄이려면 이 값을 낮추세요. 예: 8192",
    settingsLmStudioContextLengthLabel: "LM Studio 컨텍스트 길이",
    settingsLmStudioContextLengthDescription: "메모리를 줄이려면 이 값을 낮추세요. 예: 8192",
    settingsAutomaticPlaceholder: "시스템의 여유분을 고려하여 최대값을 계산합니다.",
    settingsCommandReviewLabel: "명령어 검토 모드",
    settingsCommandReviewDescription: "쉘 명령어를 실행하기 전에 확인 창을 표시합니다.",
    commandReviewTitle: "이 명령어를 실행할까요?",
    commandReviewApprove: "실행",
    settingsAutoCollapseLabel: "도구 카드 자동 접기",
    settingsAutoCollapseDescription: "diff와 명령어 그룹이 생길 때 자동으로 접습니다.",
    settingsGlobalSystemPromptLabel: "전역 시스템 프롬프트",
    settingsGlobalSystemPromptDescription: "모든 에이전트 요청 앞에 항상 붙일 지침입니다.",
    settingsResetAppLabel: "앱 초기화",
    settingsResetAppDescription: "저장된 세션과 로컬 설정을 지우고 앱 상태를 처음처럼 되돌립니다.",
    settingsResetAppAction: "앱 초기화",
    modelWarningTitle: "이 모델은 로드에 실패할 수 있습니다",
    modelWarningDescription: "현재 로컬 모델 설정이 이 시스템 사양에 비해 무거워 보입니다. 로드에 실패하거나 불안정할 수 있습니다.",
    modelWarningConfirm: "그래도 선택",
    modelWarningMenu: "로드 실패 가능성",
    resetWarningTitle: "앱을 초기화할까요?",
    resetWarningDescription: "저장된 세션과 로컬 설정이 삭제됩니다. 프로젝트 파일과 현재 워크스페이스는 건드리지 않습니다.",
    resetWarningConfirm: "초기화",
    cancel: "취소",
    save: "저장",
    delete: "삭제",
    deleteMessage: "메시지 삭제",
    sessions: "세션",
    newSession: "새 세션",
    emptySessionTitle: "빈 세션",
    emptySession: "아직 메시지가 없습니다.",
    contextUsage: "컨텍스트 창",
    contextUsageTitle: (percent: number) => `컨텍스트 창: ${percent}%`,
    contextState: "상태",
    contextSnippets: "스니펫",
    contextEstimate: " 개의 토큰 사용",
    toolGroupDiffSingle: "파일 수정",
    toolGroupDiffMulti: (count: number) => `파일 수정 ${count}개`,
    toolGroupCommandSingle: "명령어 실행",
    toolGroupCommandMulti: (count: number) => `명령어 실행 ${count}개`,
    toolGroupMixed: (parts: string[]) => parts.join(" · "),
    toolItemDiff: "파일 수정",
    toolItemCommand: "명령어",
    fileReadSingle: (fileName: string) => `Cogent 에이전트가 ${fileName} 파일을 읽었습니다.`,
    fileReadRange: (fileName: string, startLine: number, endLine: number) =>
      `Cogent 에이전트가 ${fileName} 파일을 ${startLine}번 줄에서 ${endLine}번 줄까지 읽었습니다.`,
    fileDelete: (fileName: string) => `Cogent 에이전트가 ${fileName} 파일을 삭제했습니다.`,
    fileMove: (fromName: string, toName: string) => `Cogent 에이전트가 ${fromName}을(를) ${toName}으로 이동했습니다.`,
    directoryDelete: (directoryName: string) => `Cogent 에이전트가 ${directoryName} 폴더를 삭제했습니다.`,
    statusRequesting: "요청 중",
    statusLoading: "로드 중",
    statusThinking: "생각 중",
    statusWriting: (fileName: string) => `${fileName} 파일 수정 중`,
  },
  ja: {
    title: "Cogent Code",
    subtitle: "?かなツ?ル?行とチャット中心の流れを備えた AI コ?ディングワ?クスペ?ス。",
    systemLabel: "システム",
    systemText: "?索ベ?スの推論、?れたコマンド?行、自動コンテキスト?縮が有?です。",
    you: "あなた",
    agent: "Cogent",
    trace: "記?",
    start: "?話を始めましょう。",
    emptyHeroTitle: "Cogent を始める",
    emptyHeroDescription: "コ?ド編集、デバッグ、ファイル探索、コマンド?行などを?んで始めましょう。",
    workspace: "ワ?クスペ?ス",
    files: "ファイル",
    workspaceHint: "パネルを固定しなくても、ホバ?だけでファイルを?照できます。",
    send: "送信",
    inputPlaceholder: "作りたいもの、直したいもの、調べたいものをエ?ジェントに依?してください。",
    thoroughnessLight: "?く",
    thoroughnessBalanced: "バランス",
    thoroughnessDeep: "深く",
    auto: "自動",
    backend: "バックエンド",
    frontend: "フロントエンド",
    lockMode: "モ?ド選?を固定",
    modelTitle: "モデル",
    modelDescription: "エ?ジェントの?答速度と推論の深さを選びます。",
    modelGeminiSection: "Gemini",
    modelOllamaSection: "ロ?カル Ollama",
    modelLmStudioSection: "ローカル LM Studio",
    modelOllamaDescription: "この端末で?出されたロ?カルモデルです。",
    modeTitle: "モ?ド",
    modeDescription: "扱っているコ?ドの種類に合わせてエ?ジェントを調整します。",
    thoroughnessTitle: "丁寧さ",
    thoroughnessDescription: "バイブコ?ディングの流れをどれだけ?く、または?重にするかを調整します。",
    modelFlashLiteDescription: "素早い反復のための最速ル?プ。",
    modelFlashDescription: "速度と品質のバランス。",
    modelProDescription: "重い作業向けのより深い推論。",
    autoDescription: "エ?ジェントがバックエンドとフロントエンドの流れを自動で選びます。",
    backendDescription: "API、サ?ビス、デ?タフロ?、サ?バ?ロジック向けです。",
    frontendDescription: "HTML、CSS、JavaScript、React を使う UI 作業向けです。",
    thoroughnessLightDescription: "確認を?めにして素早く進めます。",
    thoroughnessBalancedDescription: "速度と?重さのバランスを取ります。",
    thoroughnessDeepDescription: "?更前により多くを確認します。",
    saved: "保存?み",
    saving: "保存中...",
    modified: "未保存の?更",
    openFolderTitle: "フォルダ?を開く",
    openFolderDescription: "?存のプロジェクトフォルダ?を開いて、すぐに作業を?けます。",
    newWindowTitle: "新しいウィンドウ",
    newWindowDescription: "別のウィンドウで別のワ?クスペ?スを開始します。",
    openConsoleTitle: "新しいコマンドプロンプトを開く",
    openConsoleDescription: "現在のワ?クスペ?スを基準に新しいコマンドプロンプトを開きます。",
    settingsTitle: "設定",
    settingsDescription: "アプリの動作と接?サ?ビスを調整します。",
    settingsClose: "設定を閉じる",
    settingsApiKeyLabel: "Gemini API Key",
    settingsApiKeyDescription: "Gemini モデルの呼び出しに使う API Key を入力します。",
    geminiApiKeyRequiredMessage: "クラウドモデルを使うには設定で Gemini API Key を入力してください。",
    settingsOllamaLabel: "ロ?カル Ollama",
    settingsOllamaAvailable: "?出?みで、そのまま使えます。",
    settingsOllamaUnavailable: "この端末では?出されませんでした。",
    settingsOllamaContextLengthLabel: "Ollama コンテキスト長",
    settingsOllamaContextLengthDescription: "メモリ使用量を減らすにはこの値を下げます。例: 8192",
    settingsLmStudioContextLengthLabel: "LM Studio コンテキスト長",
    settingsLmStudioContextLengthDescription: "メモリ使用量を減らすにはこの値を下げます。例: 8192",
    settingsAutomaticPlaceholder: "システムの余裕分を考慮して最大値を計算します。",
    settingsCommandReviewLabel: "コマンドレビューモード",
    settingsCommandReviewDescription: "シェルコマンドを実行する前に確認ダイアログを表示します。",
    commandReviewTitle: "このコマンドを実行しますか?",
    commandReviewApprove: "実行",
    settingsAutoCollapseLabel: "ツ?ルカ?ドを自動で折りたたむ",
    settingsAutoCollapseDescription: "diff とコマンドのグル?プを生成時に自動で折りたたみます。",
    settingsGlobalSystemPromptLabel: "グロ?バルシステムプロンプト",
    settingsGlobalSystemPromptDescription: "すべてのエ?ジェントリクエストの先頭に常に追加する指示です。",
    settingsResetAppLabel: "アプリを初期化",
    settingsResetAppDescription: "保存?みセッションとロ?カル設定を消去し、アプリ?態を初期?態に?します。",
    settingsResetAppAction: "アプリを初期化",
    modelWarningTitle: "このモデルはロードに失敗する可能性があります",
    modelWarningDescription: "現在のローカルモデル設定はこのシステムに対して重すぎる可能性があります。ロード失敗や不安定化の恐れがあります。",
    modelWarningConfirm: "それでも選択",
    modelWarningMenu: "ロード失敗の可能性",
    resetWarningTitle: "アプリを初期化しますか?",
    resetWarningDescription: "保存?みセッションとロ?カル設定が削除されます。プロジェクトファイルと現在のワ?クスペ?スは?更しません。",
    resetWarningConfirm: "初期化",
    cancel: "キャンセル",
    save: "保存",
    delete: "削除",
    deleteMessage: "メッセージを削除",
    sessions: "セッション",
    newSession: "新しいセッション",
    emptySessionTitle: "空のセッション",
    emptySession: "まだメッセ?ジがありません。",
    contextUsage: "コンテキストウィンドウ",
    contextUsageTitle: (percent: number) => `コンテキストウィンドウ: ${percent}%`,
    contextState: "?態",
    contextSnippets: "スニペット",
    contextEstimate: "ト?クン使用",
    toolGroupDiffSingle: "ファイル修正",
    toolGroupDiffMulti: (count: number) => `ファイル修正 ${count}件`,
    toolGroupCommandSingle: "コマンド?行",
    toolGroupCommandMulti: (count: number) => `コマンド?行 ${count}件`,
    toolGroupMixed: (parts: string[]) => parts.join(" · "),
    toolItemDiff: "ファイル修正",
    toolItemCommand: "コマンド",
    fileReadSingle: (fileName: string) => `Cogent が ${fileName} を?みました。`,
    fileReadRange: (fileName: string, startLine: number, endLine: number) =>
      `Cogent が ${fileName} を ${startLine} 行目から ${endLine} 行目まで?みました。`,
    fileDelete: (fileName: string) => `Cogent が ${fileName} を削除しました。`,
    fileMove: (fromName: string, toName: string) => `Cogent が ${fromName} を ${toName} に移動しました。`,
    directoryDelete: (directoryName: string) => `Cogent が ${directoryName} フォルダを削除しました。`,
    statusRequesting: "リクエスト中",
    statusLoading: "ロ?ド中",
    statusThinking: "思考中",
    statusWriting: (fileName: string) => `${fileName} を編集中`,
  },
} as const;

function detectLocale(): Locale {
  if (typeof navigator === "undefined") {
    return "ko";
  }

  const language = navigator.language.toLowerCase();
  if (language.startsWith("ja")) {
    return "ja";
  }
  if (language.startsWith("ko")) {
    return "ko";
  }
  return "en";
}

declare global {
  interface Window {
    cogent: {
      runAgentTask: (request: {
        prompt: string;
        currentPromptImages?: ImageAttachment[];
        currentPromptFiles?: FileAttachment[];
        ollamaContextLength?: number;
        lmStudioContextLength?: number;
        activeFile?: string;
        selectedText?: string;
        openFiles: string[];
        explicitMode?: Exclude<AgentMode, "auto">;
        modelTier: ModelTier;
        modelProvider?: ModelProvider;
        modelId?: string;
        liveApply: boolean;
        currentCode: string;
        apiKey?: string;
        conversation?: ConversationTurn[];
        workspaceRoot?: string;
      }) => Promise<AgentStreamEvent[]>;
      runAgentTaskStream: (
        request: {
          prompt: string;
          currentPromptImages?: ImageAttachment[];
          currentPromptFiles?: FileAttachment[];
          ollamaContextLength?: number;
          lmStudioContextLength?: number;
          activeFile?: string;
          selectedText?: string;
          openFiles: string[];
          explicitMode?: Exclude<AgentMode, "auto">;
          modelTier: ModelTier;
          modelProvider?: ModelProvider;
          modelId?: string;
          liveApply: boolean;
          currentCode: string;
          apiKey?: string;
          conversation?: ConversationTurn[];
          workspaceRoot?: string;
        },
        onEvent: (event: AgentStreamEvent) => void,
      ) => { requestId: string; completed: Promise<void> };
      cancelAgentTaskStream: (requestId: string) => Promise<{ canceled: boolean }>;
      completeBrowserAssist: (payload: BrowserAssistResult) => Promise<{ completed: boolean }>;
      cancelBrowserAssist: (requestId: string) => Promise<{ canceled: boolean }>;
      approveCommandReview: (reviewId: string) => Promise<{ approved: boolean }>;
      cancelCommandReview: (reviewId: string) => Promise<{ canceled: boolean }>;
      buildContextSnapshot: (request: {
        prompt: string;
        ollamaContextLength?: number;
        lmStudioContextLength?: number;
        activeFile?: string;
        selectedText?: string;
        openFiles: string[];
        explicitMode?: Exclude<AgentMode, "auto">;
        modelTier: ModelTier;
        modelProvider?: ModelProvider;
        modelId?: string;
        liveApply: boolean;
        currentCode: string;
        workspaceRoot?: string;
      }) => Promise<ContextSnapshot>;
      buildContextUsageSnapshot: (request: {
        prompt: string;
        ollamaContextLength?: number;
        lmStudioContextLength?: number;
        activeFile?: string;
        selectedText?: string;
        openFiles: string[];
        explicitMode?: Exclude<AgentMode, "auto">;
        modelTier: ModelTier;
        modelProvider?: ModelProvider;
        modelId?: string;
        liveApply: boolean;
        currentCode: string;
        apiKey?: string;
        conversation?: ConversationTurn[];
        workspaceRoot?: string;
      }) => Promise<ContextUsageSnapshot | null>;
      runCommand: (request: { command: string; cwd?: string }) => Promise<AgentStreamEvent[]>;
      getMemoryInfo: () => Promise<{ totalMemoryBytes: number; freeMemoryBytes: number }>;
      getGeminiModels: (apiKey: string) => Promise<{ available: boolean; models: GeminiModelSummary[]; error?: string }>;
      getOllamaModels: () => Promise<{ available: boolean; models: OllamaModelSummary[]; error?: string }>;
      getLmStudioModels: () => Promise<{ available: boolean; models: LmStudioModelSummary[]; error?: string }>;
      cleanupLocalModels: (payload: { provider?: ModelProvider; modelId?: string }) => Promise<{ cleaned: boolean }>;
      getWorkspaceInfo: () => Promise<{ rootPath: string; name: string }>;
      openFolder: () => Promise<{ rootPath: string; name: string } | null>;
      openPath: (targetPath: string) => Promise<OpenPathResult>;
      openNewWindow: () => Promise<{ opened: boolean }>;
      openConsoleWindow: () => Promise<{ opened: boolean }>;
      getFileTree: () => Promise<{ rootPath: string; name: string; children: FileTreeNode[] }>;
      readFile: (filePath: string) => Promise<FilePreview>;
      writeFile: (request: { filePath: string; content: string }) => Promise<{ path: string; saved: boolean }>;
      windowControls: {
        minimize: () => Promise<void>;
        maximizeToggle: () => Promise<{ maximized: boolean }>;
        close: () => Promise<void>;
        getState: () => Promise<{ maximized: boolean }>;
      };
    };
  }

  namespace JSX {
    interface IntrinsicElements {
      webview: React.DetailedHTMLProps<React.HTMLAttributes<HTMLElement>, HTMLElement> & {
        src?: string;
        partition?: string;
        allowpopups?: boolean | "true" | "false";
      };
    }
  }
}

function getCogent() {
  return window.cogent ?? null;
}

function isChatMessage(value: unknown): value is ChatMessage {
  if (!value || typeof value !== "object") {
    return false;
  }

  const message = value as Partial<ChatMessage>;
  const attachmentsValid =
    message.attachments === undefined ||
    (Array.isArray(message.attachments) &&
      message.attachments.every(
        (attachment) =>
          attachment &&
          typeof attachment.id === "string" &&
          typeof attachment.name === "string" &&
          typeof attachment.mimeType === "string" &&
          typeof attachment.dataUrl === "string",
      ));
  const fileAttachmentsValid =
    message.fileAttachments === undefined ||
    (Array.isArray(message.fileAttachments) &&
      message.fileAttachments.every(
        (attachment) =>
          attachment &&
          typeof attachment.id === "string" &&
          typeof attachment.name === "string" &&
          typeof attachment.mimeType === "string" &&
          typeof attachment.content === "string",
      ));

  return typeof message.id === "string" && (message.role === "user" || message.role === "assistant") && attachmentsValid && fileAttachmentsValid;
}

function isChatSession(value: unknown): value is ChatSession {
  if (!value || typeof value !== "object") {
    return false;
  }

  const session = value as Partial<ChatSession>;
  return (
    typeof session.id === "string" &&
    typeof session.name === "string" &&
    (session.titleState === "empty" || session.titleState === "generated") &&
    typeof session.prompt === "string" &&
    Array.isArray(session.messages) &&
    session.messages.every(isChatMessage)
  );
}

function handleEditorMount(_editor: unknown, monaco: MonacoApi) {
  monaco.editor.remeasureFonts();
}

function toConversationTurn(message: ChatMessage): ConversationTurn {
  return {
    role: message.role,
    text: [message.text, message.code].filter(Boolean).join("\n\n"),
    images:
      message.role === "user" && message.attachments?.length
        ? message.attachments.map((attachment) => ({
            id: attachment.id,
            name: attachment.name,
            mimeType: attachment.mimeType,
            dataUrl: attachment.dataUrl,
          }))
        : undefined,
    files:
      message.role === "user" && message.fileAttachments?.length
        ? message.fileAttachments.map((attachment) => ({
            id: attachment.id,
            name: attachment.name,
            mimeType: attachment.mimeType,
            content: attachment.content,
          }))
        : undefined,
  };
}

function renderImageAttachments(attachments: ImageAttachment[], className: string) {
  return (
    <div className={className}>
      {attachments.map((attachment) => (
        <figure key={attachment.id} className="image-attachment-chip">
          <img src={attachment.dataUrl} alt={attachment.name} className="image-attachment-preview" />
          <figcaption title={attachment.name}>{attachment.name}</figcaption>
        </figure>
      ))}
    </div>
  );
}

function renderFileAttachments(attachments: FileAttachment[], className: string) {
  return (
    <div className={className}>
      {attachments.map((attachment) => (
        <div key={attachment.id} className="file-attachment-chip" title={attachment.name}>
          <img
            className="file-attachment-type-image"
            src={getIconUrlForFilePath(attachment.name, MATERIAL_ICONS_BASE_URL)}
            alt=""
            aria-hidden="true"
          />
          <span className="file-attachment-name">{attachment.name}</span>
        </div>
      ))}
    </div>
  );
}

export function App() {
  const minFileRailWidth = 260;
  const maxFileRailWidth = 720;
  const [code, setCode] = useState("");
  const [mode, setMode] = useState<AgentMode | "auto">("auto");
  const [modeLocked, setModeLocked] = useState(false);
  const [modelTier, setModelTier] = useState<ModelTier>("flash");
  const [modelProvider, setModelProvider] = useState<ModelProvider>("gemini");
  const [selectedGeminiModel, setSelectedGeminiModel] = useState("");
  const [geminiModels, setGeminiModels] = useState<GeminiModelSummary[]>([]);
  const [selectedOllamaModel, setSelectedOllamaModel] = useState("");
  const [ollamaContextLength, setOllamaContextLength] = useState("");
  const [ollamaContextLengthDraft, setOllamaContextLengthDraft] = useState("");
  const [selectedLmStudioModel, setSelectedLmStudioModel] = useState("");
  const [ollamaModels, setOllamaModels] = useState<OllamaModelSummary[]>([]);
  const [isOllamaAvailable, setIsOllamaAvailable] = useState(false);
  const [lmStudioModels, setLmStudioModels] = useState<LmStudioModelSummary[]>([]);
  const [isLmStudioAvailable, setIsLmStudioAvailable] = useState(false);
  const [thoroughness, setThoroughness] = useState<Thoroughness>("balanced");
  const [openMenu, setOpenMenu] = useState<OpenMenu>(null);
  const [closingMenu, setClosingMenu] = useState<OpenMenu>(null);
  const [menuVisible, setMenuVisible] = useState(false);
  const [isSettingsOpen, setIsSettingsOpen] = useState(false);
  const [isSettingsVisible, setIsSettingsVisible] = useState(false);
  const [isResetWarningOpen, setIsResetWarningOpen] = useState(false);
  const [isResetWarningVisible, setIsResetWarningVisible] = useState(false);
  const [modelSelectionWarning, setModelSelectionWarning] = useState<PendingModelSelectionWarning | null>(null);
  const [isModelSelectionWarningVisible, setIsModelSelectionWarningVisible] = useState(false);
  const [autoCollapseToolGroups, setAutoCollapseToolGroups] = useState(true);
  const [autoCollapseToolGroupsDraft, setAutoCollapseToolGroupsDraft] = useState(true);
  const [commandReviewMode, setCommandReviewMode] = useState(true);
  const [commandReviewModeDraft, setCommandReviewModeDraft] = useState(true);
  const [pendingCommandReview, setPendingCommandReview] = useState<{ reviewId: string; command: string; cwd?: string } | null>(null);
  const [isCommandReviewVisible, setIsCommandReviewVisible] = useState(false);
  const [globalSystemPrompt, setGlobalSystemPrompt] = useState("");
  const [globalSystemPromptDraft, setGlobalSystemPromptDraft] = useState("");
  const [lmStudioContextLength, setLmStudioContextLength] = useState("");
  const [lmStudioContextLengthDraft, setLmStudioContextLengthDraft] = useState("");
  const [draftPromptImagesBySession, setDraftPromptImagesBySession] = useState<Record<string, ImageAttachment[]>>({});
  const [draftPromptFilesBySession, setDraftPromptFilesBySession] = useState<Record<string, FileAttachment[]>>({});
  const [isContextUsageTooltipOpen, setIsContextUsageTooltipOpen] = useState(false);
  const [isContextUsageTooltipVisible, setIsContextUsageTooltipVisible] = useState(false);
  const [contextUsageTooltipPosition, setContextUsageTooltipPosition] = useState<{ left: number; top: number } | null>(null);
  const [systemTotalMemoryBytes, setSystemTotalMemoryBytes] = useState<number | undefined>(undefined);
  const [systemFreeMemoryBytes, setSystemFreeMemoryBytes] = useState<number | undefined>(undefined);
  const [locale] = useState<Locale>(detectLocale);
  const [menuPosition, setMenuPosition] = useState<{ left: number; top: number; minWidth: number; maxHeight: number } | null>(null);
  const [detectedMode, setDetectedMode] = useState<AgentMode | null>(null);
  const [windowMaximized, setWindowMaximized] = useState(false);
  const [workspaceName, setWorkspaceName] = useState("");
  const [workspaceRoot, setWorkspaceRoot] = useState("");
  const [fileRailWidth, setFileRailWidth] = useState(280);
  const [isResizingFileRail, setIsResizingFileRail] = useState(false);
  const [isFileRailDismissed, setIsFileRailDismissed] = useState(false);
  const [canRestoreFileRail, setCanRestoreFileRail] = useState(true);
  const [fileTree, setFileTree] = useState<FileTreeNode[]>([]);
  const [expandedPaths, setExpandedPaths] = useState<Record<string, boolean>>({});
  const [selectedFile, setSelectedFile] = useState<FilePreview | null>(null);
  const [isEditorOpen, setIsEditorOpen] = useState(false);
  const [shouldAutoOpenEditor, setShouldAutoOpenEditor] = useState(false);
  const [saveState, setSaveState] = useState<"idle" | "dirty" | "saving" | "saved">("idle");
  const [geminiApiKey, setGeminiApiKey] = useState("");
  const [geminiApiKeyDraft, setGeminiApiKeyDraft] = useState("");
  const text = COPY[locale];
  const [sessions, setSessions] = useState<ChatSession[]>(() => {
    if (typeof window === "undefined") {
      return [createSession(text)];
    }

    try {
      const raw = window.localStorage.getItem(CHAT_SESSIONS_STORAGE_KEY);
      if (!raw) {
        return [createSession(text)];
      }

      const parsed = JSON.parse(raw) as unknown;
      if (!Array.isArray(parsed)) {
        return [createSession(text)];
      }

      const restoredSessions = parsed.filter(isChatSession).map((session) => ({
        ...session,
        messages: session.messages.map((message) => ({
          ...message,
          streaming: false,
        })),
      }));
      return restoredSessions.length > 0 ? restoredSessions : [createSession(text)];
    } catch {
      return [createSession(text)];
    }
  });
  const [activeSessionId, setActiveSessionId] = useState(() => {
    if (typeof window === "undefined") {
      return "";
    }

    return window.localStorage.getItem(ACTIVE_SESSION_STORAGE_KEY) ?? "";
  });
  const textareaRef = useRef<HTMLTextAreaElement | null>(null);
  const composerFileInputRef = useRef<HTMLInputElement | null>(null);
  const globalSystemPromptTextareaRef = useRef<HTMLTextAreaElement | null>(null);
  const composerWrapRef = useRef<HTMLDivElement | null>(null);
  const contextUsageButtonRef = useRef<HTMLButtonElement | null>(null);
  const contextUsageTooltipRef = useRef<HTMLDivElement | null>(null);
  const contextUsageTooltipHideTimeoutRef = useRef<number | null>(null);
  const monacoRef = useRef<MonacoApi | null>(null);
  const editorInstanceRef = useRef<MonacoStandaloneEditor | null>(null);
  const browserAssistWebviewRef = useRef<any>(null);
  const pendingBrowserAssistOpenRef = useRef<string | null>(null);
  const browserAssistLastActivityRef = useRef(0);
  const browserAssistLastActivityTypeRef = useRef("");
  const browserAssistBaselineSignatureRef = useRef<string | null>(null);
  const browserAssistStableRoundsRef = useRef(0);
  const browserAssistArmedRef = useRef(false);
  const browserAssistArmStableRoundsRef = useRef(0);
  const browserAssistDragStateRef = useRef<{
    pointerId: number;
    offsetX: number;
    offsetY: number;
  } | null>(null);
  const sharedTypesLibLoadedRef = useRef(false);
  const monacoTypescriptConfiguredRef = useRef(false);
  const shouldFollowChatRef = useRef(true);
  const pendingScrollBehaviorRef = useRef<ScrollBehavior | null>(null);
  const didRestoreInitialSessionScrollRef = useRef(false);
  const [composerBottomSpace, setComposerBottomSpace] = useState(140);
  const [isComposerDragActive, setIsComposerDragActive] = useState(false);
  const [isAgentRunning, setIsAgentRunning] = useState(false);
  const [browserAssistPanel, setBrowserAssistPanel] = useState<BrowserAssistPanelState | null>(null);
  const [isBrowserAssistFloating, setIsBrowserAssistFloating] = useState(false);
  const [browserAssistFloatingPosition, setBrowserAssistFloatingPosition] = useState<BrowserAssistFloatingPosition>({
    left: 0,
    top: 0,
  });
  const [browserAssistFloatingDisplayPosition, setBrowserAssistFloatingDisplayPosition] = useState<BrowserAssistFloatingPosition>({
    left: 0,
    top: 0,
  });
  const [isBrowserAssistDragging, setIsBrowserAssistDragging] = useState(false);
  const [editingMessageId, setEditingMessageId] = useState<string | null>(null);
  const [editingMessageText, setEditingMessageText] = useState("");
  const [collapsedToolGroups, setCollapsedToolGroups] = useState<Record<string, boolean>>({});
  const activeStreamRequestIdRef = useRef<string | null>(null);
  const activeStreamCompletedRef = useRef<Promise<void> | null>(null);
  const manualFileSelectionVersionRef = useRef(0);
  const didInitializeCollapsedToolGroupsRef = useRef(false);
  const didInitializeLocalModelCleanupRef = useRef(false);
  const editingTextareaRef = useRef<HTMLTextAreaElement | null>(null);
  const triggerRefs = useRef<Record<Exclude<OpenMenu, null>, HTMLButtonElement | null>>({
    attach: null,
    files: null,
    model: null,
    mode: null,
    thoroughness: null,
  });
  const activeSession = sessions.find((session) => session.id === activeSessionId) ?? sessions[0];
  const prompt = activeSession?.prompt ?? "";
  const promptImages = activeSession?.id ? (draftPromptImagesBySession[activeSession.id] ?? []) : [];
  const promptFiles = activeSession?.id ? (draftPromptFilesBySession[activeSession.id] ?? []) : [];
  const messages = activeSession?.messages ?? [];
  const renderableMessages = useMemo(() => messages.filter(hasRenderableMessageContent), [messages]);
  const renderItems = useMemo(() => buildRenderItems(renderableMessages), [renderableMessages]);
  const context = activeSession?.context ?? null;
  const contextUsage = activeSession?.contextUsage ?? null;
  const selectedGeminiModelSummary = geminiModels.find((model) => model.id === selectedGeminiModel);
  const selectedOllamaModelSummary = ollamaModels.find(
    (model) => model.model === selectedOllamaModel || model.name === selectedOllamaModel,
  );
  const selectedLmStudioModelSummary = lmStudioModels.find((model) => model.id === selectedLmStudioModel);
  const activeModelLabel =
    modelProvider === "gemini"
      ? (selectedGeminiModelSummary?.name ?? (modelTier === "pro" ? "Pro" : modelTier === "flash" ? "Flash" : "Flash Lite"))
      : modelProvider === "ollama"
      ? (selectedOllamaModelSummary?.name ?? selectedOllamaModel ?? "Ollama")
      : modelProvider === "lmstudio"
        ? (selectedLmStudioModelSummary?.name ?? selectedLmStudioModel ?? "LM Studio")
        : modelTier === "pro"
          ? "Pro"
        : modelTier === "flash"
          ? "Flash"
          : "Flash Lite";
  const canSend = prompt.trim().length > 0 || promptImages.length > 0 || promptFiles.length > 0;
  const currentInputTokenLimit =
    modelProvider === "gemini"
      ? (selectedGeminiModelSummary?.inputTokenLimit || MODEL_INPUT_TOKEN_LIMITS[modelTier])
      : modelProvider === "ollama"
      ? (Number(ollamaContextLength) || Number(getDefaultOllamaContextLength(systemTotalMemoryBytes)) || selectedOllamaModelSummary?.contextLength || 131_072)
      : modelProvider === "lmstudio"
        ? (Number(lmStudioContextLength) || Number(getDefaultLmStudioContextLength(systemTotalMemoryBytes)) || selectedLmStudioModelSummary?.contextLength || 32_768)
        : MODEL_INPUT_TOKEN_LIMITS[modelTier];
  const contextUsageDisplay = contextUsage
    ? {
        ...contextUsage,
        inputTokenLimit: currentInputTokenLimit,
        usagePercent: Math.max(0, Math.min(100, Math.round((contextUsage.usedTokens / currentInputTokenLimit) * 100))),
      }
    : {
    model: activeModelLabel,
    usedTokens: 0,
    inputTokenLimit: currentInputTokenLimit,
    usagePercent: 0,
    compressionState: context?.retrieval.compressionState ?? "healthy",
    snippetCount: context?.retrieval.snippets.length ?? 0,
  };
  const contextCircleCenter = 16;
  const contextCircleRadius = 8;
  const contextCircleCircumference = 2 * Math.PI * contextCircleRadius;
  const contextCircleOffset =
    contextCircleCircumference - (contextUsageDisplay.usagePercent / 100) * contextCircleCircumference;

  function updateSessionById(sessionId: string, updater: (session: ChatSession) => ChatSession) {
    setSessions((current) => current.map((session) => (session.id === sessionId ? updater(session) : session)));
  }

  function updateActiveSession(updater: (session: ChatSession) => ChatSession) {
    const currentSessionId = activeSession?.id;
    if (!currentSessionId) {
      return;
    }

    updateSessionById(currentSessionId, updater);
  }

  function updateDraftImagesForSession(sessionId: string, updater: (images: ImageAttachment[]) => ImageAttachment[]) {
    setDraftPromptImagesBySession((current) => {
      const nextImages = updater(current[sessionId] ?? []);
      if (nextImages.length === 0) {
        const { [sessionId]: _removed, ...rest } = current;
        return rest;
      }

      return {
        ...current,
        [sessionId]: nextImages,
      };
    });
  }

  function updateDraftFilesForSession(sessionId: string, updater: (files: FileAttachment[]) => FileAttachment[]) {
    setDraftPromptFilesBySession((current) => {
      const nextFiles = updater(current[sessionId] ?? []);
      if (nextFiles.length === 0) {
        const { [sessionId]: _removed, ...rest } = current;
        return rest;
      }

      return {
        ...current,
        [sessionId]: nextFiles,
      };
    });
  }

  const canAttachPromptImages = modelProvider === "gemini";

  function buildTaskRequest(submittedPrompt: string, conversation: ConversationTurn[]): {
    prompt: string;
    globalSystemPrompt?: string;
    ollamaContextLength?: number;
    lmStudioContextLength?: number;
    currentPromptImages?: ImageAttachment[];
    currentPromptFiles?: FileAttachment[];
    activeFile?: string;
    openFiles: string[];
    explicitMode?: Exclude<AgentMode, "auto">;
    thoroughness?: Thoroughness;
    modelTier: ModelTier;
    modelProvider?: ModelProvider;
    modelId?: string;
    liveApply: boolean;
    currentCode: string;
    apiKey?: string;
    conversation: ConversationTurn[];
    workspaceRoot?: string;
  } {
    const activeFilePath = selectedFile?.path;

    return {
      prompt: submittedPrompt,
      globalSystemPrompt: globalSystemPrompt.trim() || undefined,
      ollamaContextLength: modelProvider === "ollama" ? Number(ollamaContextLength) || undefined : undefined,
      lmStudioContextLength: modelProvider === "lmstudio" ? Number(lmStudioContextLength) || undefined : undefined,
      currentPromptImages: promptImages.length > 0 ? promptImages : undefined,
      currentPromptFiles: promptFiles.length > 0 ? promptFiles : undefined,
      activeFile: activeFilePath,
      openFiles: activeFilePath ? [activeFilePath] : [],
      explicitMode: modeLocked && mode !== "auto" ? mode : undefined,
      thoroughness,
      modelTier,
      modelProvider,
      modelId:
        modelProvider === "gemini"
          ? selectedGeminiModel || undefined
          : modelProvider === "ollama"
          ? selectedOllamaModel || undefined
          : modelProvider === "lmstudio"
            ? selectedLmStudioModel || undefined
            : undefined,
      liveApply: true,
      currentCode: code,
      apiKey: modelProvider === "gemini" ? geminiApiKey : undefined,
      conversation,
      workspaceRoot,
    };
  }

  async function readPromptImages(files: File[]) {
    const nextImages = await Promise.all(
      files
        .filter((file) => file.type.startsWith("image/"))
        .map(
          (file) =>
            new Promise<ImageAttachment>((resolve, reject) => {
              const reader = new FileReader();
              reader.onload = () => {
                if (typeof reader.result !== "string") {
                  reject(new Error("Failed to read image."));
                  return;
                }

                resolve({
                  id: `${Date.now()}-${Math.random().toString(36).slice(2, 8)}`,
                  name: file.name,
                  mimeType: file.type || "image/png",
                  dataUrl: reader.result,
                });
              };
              reader.onerror = () => reject(reader.error ?? new Error("Failed to read image."));
              reader.readAsDataURL(file);
            }),
        ),
    );

    return nextImages;
  }

  function isImageFile(file: File) {
    return file.type.startsWith("image/");
  }

  function isTextLikeFile(file: File) {
    if (file.type.startsWith("text/")) {
      return true;
    }

    return /\.(?:txt|md|mdx|json|ya?ml|xml|html?|css|scss|less|js|jsx|ts|tsx|mjs|cjs|py|rb|php|java|kt|swift|go|rs|c|cc|cpp|h|hpp|cs|sh|ps1|sql|toml|ini|cfg|conf|env|gitignore)$/i.test(
      file.name,
    );
  }

  async function readPromptFiles(files: File[]) {
    const supportedFiles = files.filter((file) => !isImageFile(file) && isTextLikeFile(file));
    const nextFiles = await Promise.all(
      supportedFiles.map(async (file) => ({
        id: `${Date.now()}-${Math.random().toString(36).slice(2, 8)}`,
        name: file.name,
        mimeType: file.type || "text/plain",
        content: await file.text(),
      })),
    );

    return nextFiles.filter((file) => file.content.trim().length > 0);
  }

  async function handlePromptFilesSelection(files: File[]) {
    const currentSessionId = activeSession?.id;
    if (!currentSessionId || files.length === 0) {
      return;
    }

    const existingImageNames = new Set(promptImages.map((image) => image.name.toLowerCase()));
    const existingFileNames = new Set(promptFiles.map((file) => file.name.toLowerCase()));
    const incomingImageFiles = files.filter((file) => isImageFile(file) && !existingImageNames.has(file.name.toLowerCase()));
    const incomingTextFiles = files.filter((file) => !isImageFile(file) && !existingFileNames.has(file.name.toLowerCase()));

    const nextImages = canAttachPromptImages ? await readPromptImages(incomingImageFiles) : [];
    const nextFiles = await readPromptFiles(incomingTextFiles);

    if (nextImages.length > 0) {
      updateDraftImagesForSession(currentSessionId, (current) => [...current, ...nextImages]);
    }

    if (nextFiles.length > 0) {
      updateDraftFilesForSession(currentSessionId, (current) => [...current, ...nextFiles]);
    }
  }

  function handleComposerDragOver(event: DragEvent<HTMLDivElement>) {
    event.preventDefault();
    event.dataTransfer.dropEffect = "copy";
    setIsComposerDragActive(true);
  }

  function handleComposerDragLeave(event: DragEvent<HTMLDivElement>) {
    if (event.currentTarget.contains(event.relatedTarget as Node | null)) {
      return;
    }

    setIsComposerDragActive(false);
  }

  async function handleComposerDrop(event: DragEvent<HTMLDivElement>) {
    event.preventDefault();
    setIsComposerDragActive(false);

    const files = Array.from(event.dataTransfer.files ?? []);
    if (files.length === 0) {
      return;
    }

    await handlePromptFilesSelection(files);
  }

  function handleRemovePromptImage(imageId: string) {
    const currentSessionId = activeSession?.id;
    if (!currentSessionId) {
      return;
    }

    updateDraftImagesForSession(currentSessionId, (current) => current.filter((image) => image.id !== imageId));
  }

  function handleRemovePromptFile(fileId: string) {
    const currentSessionId = activeSession?.id;
    if (!currentSessionId) {
      return;
    }

    updateDraftFilesForSession(currentSessionId, (current) => current.filter((file) => file.id !== fileId));
  }

  function handleOpenAttachPicker() {
    closeMenu();
    composerFileInputRef.current?.click();
  }

  function handleAssistantLink(url: string) {
    if (url === "cogent://settings") {
      handleOpenSettings();
      return;
    }

    handleOpenBrowserPanelLink(url);
  }

  function preventMouseFocus(event: React.MouseEvent<HTMLElement> | React.PointerEvent<HTMLElement>) {
    event.preventDefault();
  }

  async function handleComposerFileInputChange(event: ChangeEvent<HTMLInputElement>) {
    const files = Array.from(event.target.files ?? []);
    if (files.length > 0) {
      await handlePromptFilesSelection(files);
    }

    event.target.value = "";
  }

  function handleStartMessageEdit(message: ChatMessage) {
    if (message.role !== "user") {
      return;
    }

    setEditingMessageId(message.id);
    setEditingMessageText(message.text);
  }

  function appendGeminiApiKeyFallbackMessage(sessionId: string) {
    updateSessionById(sessionId, (session) => ({
      ...session,
      messages: [
        ...session.messages,
        {
          id: `${Date.now()}-assistant-api-key`,
          role: "assistant",
          text: text.geminiApiKeyRequiredMessage,
        },
      ],
    }));
    shouldFollowChatRef.current = true;
    pendingScrollBehaviorRef.current = "smooth";
  }

  function handleCancelMessageEdit() {
    setEditingMessageId(null);
    setEditingMessageText("");
  }

  async function handleDeleteMessageEdit() {
    const currentSessionId = activeSession?.id;
    const targetMessageId = editingMessageId;
    if (!currentSessionId || !targetMessageId) {
      return;
    }

    if (isAgentRunning) {
      await cancelCurrentAgentRun();
    }

    const targetIndex = messages.findIndex((message) => message.id === targetMessageId);
    if (targetIndex === -1) {
      return;
    }

    const nextMessages = messages.slice(0, targetIndex);

    updateSessionById(currentSessionId, (session) => ({
      ...session,
      messages: nextMessages,
    }));

    handleCancelMessageEdit();
    shouldFollowChatRef.current = true;
    pendingScrollBehaviorRef.current = "auto";
  }

  function handleToggleToolGroup(groupId: string) {
    setCollapsedToolGroups((current) => ({
      ...current,
      [groupId]: !current[groupId],
    }));
  }

  async function cancelCurrentAgentRun() {
    const cogent = getCogent();
    const requestId = activeStreamRequestIdRef.current;
    const completed = activeStreamCompletedRef.current;
    if (!cogent || !requestId || !completed) {
      return;
    }

    await cogent.cancelAgentTaskStream(requestId);
    await completed;
  }

  async function handleSaveMessageEdit() {
    const currentSessionId = activeSession?.id;
    const targetMessageId = editingMessageId;
    const cogent = getCogent();
    if (!currentSessionId || !targetMessageId || !cogent) {
      return;
    }

    if (isAgentRunning) {
      await cancelCurrentAgentRun();
    }

    const targetIndex = messages.findIndex((message) => message.id === targetMessageId);
    if (targetIndex === -1) {
      return;
    }

    const submittedPrompt = editingMessageText.trim();
    if (!submittedPrompt) {
      return;
    }

    if (modelProvider === "gemini" && !geminiApiKey.trim()) {
      const baseMessages = messages.slice(0, targetIndex);
      const editedUserMessage: ChatMessage = {
        ...messages[targetIndex],
        text: submittedPrompt,
      };

      updateSessionById(currentSessionId, (session) => ({
        ...session,
        messages: [...baseMessages, editedUserMessage],
      }));
      handleCancelMessageEdit();
      appendGeminiApiKeyFallbackMessage(currentSessionId);
      return;
    }

    const baseMessages = messages.slice(0, targetIndex);
    const editedUserMessage: ChatMessage = {
      ...messages[targetIndex],
      text: submittedPrompt,
    };
    const requestModelProvider = modelProvider;
    const assistantId = `${Date.now()}-assistant`;
    const activeFilePath = selectedFile?.path;
    const diffOriginal = selectedFile?.content;
    const diffPath = activeFilePath;
    const agentRunFileSelectionVersion = manualFileSelectionVersionRef.current;
    let activeTextMessageId = assistantId;
    let streamedMessage = "";
    let streamedTextChunks: TextChunk[] = [];
    let streamedCode = "";
    let lastWrittenPath: string | null = null;
    let currentStatusPath = diffPath;

    updateSessionById(currentSessionId, (session) => ({
      ...session,
      messages: [
        ...baseMessages,
        editedUserMessage,
        {
          id: assistantId,
          role: "assistant",
          text: "",
          textChunks: [],
          streaming: true,
          status: "requesting",
          diffOriginal,
          diffPath,
        },
      ],
    }));

    handleCancelMessageEdit();

    shouldFollowChatRef.current = true;
    pendingScrollBehaviorRef.current = "smooth";
    setIsAgentRunning(true);
    updateDraftImagesForSession(currentSessionId, () => []);
    updateDraftFilesForSession(currentSessionId, () => []);
    const stream = cogent.runAgentTaskStream(
      buildTaskRequest(
        submittedPrompt,
        [...baseMessages, editedUserMessage].map(toConversationTurn) satisfies ConversationTurn[],
      ),
      (event) => {
        if (event.type === "mode") {
          setDetectedMode(event.decision.mode);
        }

        if (event.type === "file-write-start") {
          currentStatusPath = event.path;
          updateSessionById(currentSessionId, (session) => ({
            ...session,
            messages: session.messages.map((message) =>
              message.id === activeTextMessageId
                ? { ...message, status: "writing", diffPath: event.path }
                : message,
            ),
          }));
          return;
        }

        if (event.type === "file-read") {
          const fileName = getFileName(event.path);
          const metaLine =
            event.startLine <= 1
              ? text.fileReadSingle(fileName)
              : text.fileReadRange(fileName, event.startLine, event.endLine);
          updateSessionById(currentSessionId, (session) => ({
            ...session,
            messages: session.messages.map((message) =>
              message.id === activeTextMessageId
                ? {
                    ...message,
                    meta: [...(message.meta ?? []), metaLine],
                    status: message.status !== "writing" ? "thinking" : message.status,
                  }
                : message,
            ),
          }));
          return;
        }

        if (event.type === "command-start") {
          const nextAssistantId = `${Date.now()}-assistant-${Math.random().toString(36).slice(2, 8)}`;
          updateSessionById(currentSessionId, (session) => ({
            ...session,
            messages: [
              ...session.messages.map((message) =>
                message.id === activeTextMessageId && isEphemeralAssistantPlaceholder(message)
                  ? {
                      ...message,
                      id: `${Date.now()}-command-${event.commandId}`,
                      status: undefined,
                      activityStatusLabel: undefined,
                      command: {
                        id: event.commandId,
                        command: event.command,
                        output: "",
                        running: true,
                        startedAt: Date.now(),
                      },
                    }
                  : message,
              ),
              {
                id: nextAssistantId,
                role: "assistant",
                text: "",
                diffPath: currentStatusPath,
              },
            ],
          }));
          activeTextMessageId = nextAssistantId;
          return;
        }

        if (event.type === "command") {
          updateSessionById(currentSessionId, (session) => ({
            ...session,
            messages: session.messages.map((message) =>
              message.command?.id === event.commandId
                ? {
                    ...message,
                    command: {
                      ...message.command,
                      output: `${message.command.output}${event.chunk.text}`,
                    },
                  }
                : message,
            ),
          }));
          return;
        }

        if (event.type === "command-end") {
          updateSessionById(currentSessionId, (session) => ({
            ...session,
            messages: session.messages.map((message) =>
              message.command?.id === event.commandId
                ? {
                    ...message,
                    command: {
                      ...message.command,
                      running: false,
                      exitCode: event.exitCode,
                    },
                  }
                : message,
            ),
          }));
          return;
        }

        if (event.type === "command-review-request") {
          if (autoCollapseToolGroups) {
            const toolGroupToCollapse = findLatestToolGroupId(
              sessions.find((s) => s.id === currentSessionId)?.messages ?? [],
            );
            if (toolGroupToCollapse) {
              setCollapsedToolGroups((current) => ({ ...current, [toolGroupToCollapse]: true }));
            }
          }
          if (commandReviewMode) {
            setPendingCommandReview({ reviewId: event.reviewId, command: event.command, cwd: event.cwd });
          } else {
            void getCogent()?.approveCommandReview(event.reviewId);
          }
          return;
        }

        if (event.type === "browser-assist-request") {
          pendingBrowserAssistOpenRef.current = event.request.requestId;
          browserAssistLastActivityRef.current = 0;
          browserAssistBaselineSignatureRef.current = null;
          browserAssistStableRoundsRef.current = 0;
          browserAssistArmedRef.current = false;
          browserAssistArmStableRoundsRef.current = 0;
          setBrowserAssistPanel({
            request: event.request,
            visible: false,
          });
          return;
        }

        if (event.type === "done") {
          // no-op
        }

        if (event.type === "mode" || event.type === "retrieval" || event.type === "status") {
          updateSessionById(currentSessionId, (session) => ({
            ...session,
            messages: session.messages.map((message) =>
                message.id === activeTextMessageId && message.status !== "writing"
                  ? {
                      ...message,
                      status:
                        event.type === "status"
                          ? event.label === "Loading" && requestModelProvider === "ollama"
                            ? "loading"
                            : "thinking"
                          : message.status ?? "thinking",
                      activityStatusLabel:
                        event.type === "status"
                          ? event.label === "Thinking"
                            ? undefined
                            : event.label
                          : message.activityStatusLabel,
                    }
                  : message,
            ),
          }));
          return;
        }

        if (event.type === "message") {
          const chunkStart = streamedMessage.length;
          streamedMessage += event.chunk;
          streamedTextChunks = [
            ...streamedTextChunks,
            {
              id: streamedTextChunks.length + 1,
              start: chunkStart,
              end: streamedMessage.length,
            },
          ];
          let toolGroupToCollapse: string | null = null;
          updateSessionById(currentSessionId, (session) => {
            const nextMessages = session.messages.map((message) =>
              message.id === activeTextMessageId
                ? {
                    ...message,
                    text: streamedMessage,
                    textChunks: streamedTextChunks,
                    streaming: true,
                    status: undefined,
                    activityStatusLabel: undefined,
                  }
                : message,
            );

            if (autoCollapseToolGroups) {
              toolGroupToCollapse = findPreviousToolGroupId(nextMessages, activeTextMessageId);
            }

            return {
              ...session,
              messages: nextMessages,
            };
          });
          if (toolGroupToCollapse) {
            setCollapsedToolGroups((current) => ({ ...current, [toolGroupToCollapse as string]: true }));
          }
          return;
        }

        if (event.type === "code") {
          streamedCode += event.chunk;
          updateSessionById(currentSessionId, (session) => ({
            ...session,
            messages: session.messages.map((message) =>
              message.id === activeTextMessageId
                ? {
                    ...message,
                    status: "writing",
                    activityStatusLabel: undefined,
                    diffPath: currentStatusPath ?? message.diffPath,
                  }
                : message,
            ),
          }));
          return;
        }

        if (event.type === "done") {
          updateSessionById(currentSessionId, (session) => ({
            ...session,
            messages: session.messages.map((message) =>
              message.role === "assistant" && message.status !== undefined
                ? {
                    ...message,
                    streaming: false,
                    status: undefined,
                    activityStatusLabel: undefined,
                  }
                : message,
            ),
          }));
          return;
        }

        if (event.type === "file-write") {
          lastWrittenPath = event.path;
          setShouldAutoOpenEditor(false);
          setIsEditorOpen(false);
          streamedCode = event.content;
          setCode(event.content);
          if (manualFileSelectionVersionRef.current === agentRunFileSelectionVersion) {
            setSelectedFile({ path: event.path, content: event.content });
          }
          setSaveState("saved");
          setIsFileRailDismissed(true);
          setCanRestoreFileRail(false);
          const nextAssistantId = `${Date.now()}-assistant-${Math.random().toString(36).slice(2, 8)}`;
          updateSessionById(currentSessionId, (session) => ({
            ...session,
            messages: [
              ...session.messages.map((message) =>
                message.role === "assistant" && message.status !== undefined
                  ? ({ ...message, streaming: false, status: undefined } satisfies ChatMessage)
                  : message,
              ),
              {
                id: `${Date.now()}-diff-${Math.random().toString(36).slice(2, 8)}`,
                role: "assistant",
                text: "",
                code: event.content,
                diffOriginal: event.originalContent,
                diffPath: event.path,
              },
              {
                id: nextAssistantId,
                role: "assistant",
                text: "",
                diffPath: event.path,
              },
            ],
          }));
          activeTextMessageId = nextAssistantId;
          currentStatusPath = event.path;
          streamedMessage = "";
          void cogent.getFileTree().then((tree) => {
            setFileTree(tree.children ?? []);
            setWorkspaceRoot(tree.rootPath);
            setWorkspaceName(tree.name);
            setExpandedPaths((current) => ({
              ...current,
              [tree.rootPath]: true,
            }));
          });
          return;
        }

        if (event.type === "file-delete" || event.type === "directory-delete" || event.type === "file-move") {
          if (event.type === "file-delete" && selectedFile?.path === event.path) {
            setIsEditorOpen(false);
            setShouldAutoOpenEditor(false);
            setSelectedFile(null);
            setCode("");
            setSaveState("idle");
          }

          if (event.type === "file-move" && selectedFile?.path === event.fromPath) {
            setSelectedFile((current) => (current ? { ...current, path: event.toPath } : current));
          }

          const metaLine =
            event.type === "directory-delete"
              ? text.directoryDelete(getFileName(event.path))
              : event.type === "file-move"
                ? text.fileMove(getFileName(event.fromPath), getFileName(event.toPath))
                : text.fileDelete(getFileName(event.path));
          updateSessionById(currentSessionId, (session) => ({
            ...session,
            messages: session.messages.map((message) =>
              message.id === activeTextMessageId
                ? {
                    ...message,
                    status: "thinking",
                    meta: metaLine ? [...(message.meta ?? []), metaLine] : message.meta,
                  }
                : message,
            ),
          }));

          void cogent.getFileTree().then((tree) => {
            setFileTree(tree.children ?? []);
            setWorkspaceRoot(tree.rootPath);
            setWorkspaceName(tree.name);
            setExpandedPaths((current) => ({
              ...current,
              [tree.rootPath]: true,
            }));
          });
          return;
        }

      },
    );

    activeStreamRequestIdRef.current = stream.requestId;
    activeStreamCompletedRef.current = stream.completed;

    try {
      await stream.completed;
    } finally {
      if (activeStreamRequestIdRef.current === stream.requestId) {
        activeStreamRequestIdRef.current = null;
      }
      if (activeStreamCompletedRef.current === stream.completed) {
        activeStreamCompletedRef.current = null;
      }
      setIsAgentRunning(false);
    }

    updateSessionById(currentSessionId, (session) => ({
      ...session,
      messages: session.messages
        .map((message) => {
          if (message.role !== "assistant" || message.code !== undefined || message.status === undefined) {
            return message;
          }

          return {
            ...message,
            status: undefined,
          };
        })
        .filter(
          (message) =>
            !(
              message.role === "assistant" &&
              !message.text.trim() &&
              message.code === undefined &&
              !message.command &&
              !message.status &&
              !(message.meta?.length)
            ),
        ),
    }));

    if (lastWrittenPath) {
      try {
        const reloadedFile = await cogent.readFile(lastWrittenPath);
        if (manualFileSelectionVersionRef.current === agentRunFileSelectionVersion) {
          setSelectedFile(reloadedFile);
          setCode(reloadedFile.content);
        }
      } catch {
        // Keep optimistic editor state if disk refresh fails.
      }
    }

    await refreshContext();
  }

  function createAndSelectSession() {
    const nextSession = createSession(text);
    setSessions((current) => [...current, nextSession]);
    setActiveSessionId(nextSession.id);
    return nextSession;
  }

  function createReplacementSession() {
    const nextSession = createSession(text);
    setSessions([nextSession]);
    setActiveSessionId(nextSession.id);
    return nextSession;
  }

  useEffect(() => {
    if (!activeSession && sessions.length > 0) {
      setActiveSessionId(sessions[0].id);
    }
  }, [activeSession, sessions]);

  useEffect(() => {
    if (didInitializeCollapsedToolGroupsRef.current) {
      return;
    }

    const nextCollapsedState = Object.fromEntries(
      sessions.flatMap((session) =>
        buildRenderItems(session.messages.filter(hasRenderableMessageContent))
          .filter((item): item is Extract<RenderItem, { type: "tool-group" }> => item.type === "tool-group")
          .map((item) => [item.id, true] as const),
      ),
    );

    didInitializeCollapsedToolGroupsRef.current = true;
    setCollapsedToolGroups(nextCollapsedState);
  }, [sessions]);

  useEffect(() => {
    if (typeof window === "undefined") {
      return;
    }

    window.localStorage.setItem(CHAT_SESSIONS_STORAGE_KEY, JSON.stringify(sessions));
  }, [sessions]);

  useEffect(() => {
    if (typeof window === "undefined") {
      return;
    }

    if (activeSessionId) {
      window.localStorage.setItem(ACTIVE_SESSION_STORAGE_KEY, activeSessionId);
    } else {
      window.localStorage.removeItem(ACTIVE_SESSION_STORAGE_KEY);
    }
  }, [activeSessionId]);

  useEffect(() => {
    if (!getCogent()) {
      return;
    }

    const timer = window.setTimeout(() => {
      void refreshContext();
    }, 180);

    return () => {
      window.clearTimeout(timer);
    };
  }, [
    mode,
    modeLocked,
    modelTier,
    modelProvider,
    selectedOllamaModel,
    ollamaContextLength,
    geminiApiKey,
    prompt,
    messages,
    selectedFile?.path,
    code,
    workspaceRoot,
  ]);

  useEffect(() => {
    const cogent = getCogent();
    if (!cogent) {
      return;
    }

    void cogent.windowControls.getState().then((state) => {
      setWindowMaximized(state.maximized);
    });

    void cogent.getWorkspaceInfo().then((workspace) => {
      if (workspace?.name) {
        setWorkspaceName(workspace.name);
      }
      if (workspace?.rootPath) {
        setWorkspaceRoot(workspace.rootPath);
      }
    });

    void cogent.getFileTree().then((tree) => {
      setFileTree(tree.children ?? []);
      setExpandedPaths({ [tree.rootPath]: true });
      if (tree?.name) {
        setWorkspaceName(tree.name);
      }
      if (tree?.rootPath) {
        setWorkspaceRoot(tree.rootPath);
      }
    });
  }, []);

  useEffect(() => {
    const clearInitialFocus = () => {
      const activeElement = document.activeElement;
      if (activeElement instanceof HTMLElement && activeElement !== document.body) {
        activeElement.blur();
      }
    };

    clearInitialFocus();
    const frame = window.requestAnimationFrame(() => {
      clearInitialFocus();
      window.setTimeout(clearInitialFocus, 0);
    });

    return () => {
      window.cancelAnimationFrame(frame);
    };
  }, []);

  useLayoutEffect(() => {
    const textarea = globalSystemPromptTextareaRef.current;
    if (!textarea) {
      return;
    }

    const computed = window.getComputedStyle(textarea);
    const lineHeight = Number.parseFloat(computed.lineHeight) || 21;
    const verticalPadding =
      Number.parseFloat(computed.paddingTop || "0") + Number.parseFloat(computed.paddingBottom || "0");
    const minHeight = lineHeight * 2 + verticalPadding;
    const maxHeight = lineHeight * 5 + verticalPadding;

    textarea.style.height = `${minHeight}px`;
    textarea.style.height = `${Math.min(textarea.scrollHeight, maxHeight)}px`;
    textarea.style.overflowY = textarea.scrollHeight > maxHeight ? "auto" : "hidden";
  }, [globalSystemPromptDraft, isSettingsOpen]);

  useLayoutEffect(() => {
    const textarea = textareaRef.current;
    if (!textarea) {
      return;
    }

    const computed = window.getComputedStyle(textarea);
    const lineHeight = Number.parseFloat(computed.lineHeight) || 21;
    const verticalPadding =
      Number.parseFloat(computed.paddingTop || "0") + Number.parseFloat(computed.paddingBottom || "0");
    const minHeight = lineHeight * 2 + verticalPadding;
    const maxHeight = lineHeight * 10 + verticalPadding;

    textarea.style.height = `${minHeight}px`;
    textarea.style.height = `${Math.min(textarea.scrollHeight, maxHeight)}px`;
    textarea.style.overflowY = textarea.scrollHeight > maxHeight ? "auto" : "hidden";
  }, [prompt]);

  useLayoutEffect(() => {
    const textarea = editingTextareaRef.current;
    if (!textarea) {
      return;
    }

    textarea.focus();
    textarea.setSelectionRange(textarea.value.length, textarea.value.length);

    const computed = window.getComputedStyle(textarea);
    const lineHeight = Number.parseFloat(computed.lineHeight) || 21;
    const verticalPadding =
      Number.parseFloat(computed.paddingTop || "0") + Number.parseFloat(computed.paddingBottom || "0");
    const minHeight = lineHeight * 2 + verticalPadding;
    const maxHeight = lineHeight * 10 + verticalPadding;

    textarea.style.height = `${minHeight}px`;
    textarea.style.height = `${Math.min(textarea.scrollHeight, maxHeight)}px`;
    textarea.style.overflowY = textarea.scrollHeight > maxHeight ? "auto" : "hidden";
  }, [editingMessageId, editingMessageText]);

  useLayoutEffect(() => {
    const composerWrap = composerWrapRef.current;
    if (!composerWrap) {
      return;
    }

    const updateComposerBottomSpace = () => {
      const composerHeight = composerWrap.getBoundingClientRect().height;
      setComposerBottomSpace(Math.max(Math.ceil(composerHeight + 44), Math.ceil(window.innerHeight * 0.5)));
    };

    updateComposerBottomSpace();

    const resizeObserver = new ResizeObserver(() => {
      updateComposerBottomSpace();
    });

    resizeObserver.observe(composerWrap);
    window.addEventListener("resize", updateComposerBottomSpace);

    return () => {
      resizeObserver.disconnect();
      window.removeEventListener("resize", updateComposerBottomSpace);
    };
  }, []);

  useEffect(() => {
    const getDistanceFromBottom = () => {
      const scrollingElement = document.scrollingElement ?? document.documentElement;
      return Math.max(scrollingElement.scrollHeight - (window.innerHeight + window.scrollY), 0);
    };

    const updateShouldFollow = () => {
      shouldFollowChatRef.current = getDistanceFromBottom() <= 80;
    };

    updateShouldFollow();
    window.addEventListener("scroll", updateShouldFollow, { passive: true });
    window.addEventListener("resize", updateShouldFollow);

    return () => {
      window.removeEventListener("scroll", updateShouldFollow);
      window.removeEventListener("resize", updateShouldFollow);
    };
  }, []);

  useLayoutEffect(() => {
    if (didRestoreInitialSessionScrollRef.current || messages.length === 0) {
      return;
    }

    didRestoreInitialSessionScrollRef.current = true;
    shouldFollowChatRef.current = true;
    pendingScrollBehaviorRef.current = "auto";
  }, [messages.length]);

  useLayoutEffect(() => {
    const behavior = pendingScrollBehaviorRef.current ?? "smooth";
    const shouldScroll = shouldFollowChatRef.current || pendingScrollBehaviorRef.current !== null;

    if (!shouldScroll) {
      pendingScrollBehaviorRef.current = null;
      return;
    }

    const frame = window.requestAnimationFrame(() => {
      const scrollingElement = document.scrollingElement ?? document.documentElement;
      window.scrollTo({
        top: scrollingElement.scrollHeight - window.innerHeight,
        behavior,
      });
      pendingScrollBehaviorRef.current = null;
      shouldFollowChatRef.current = true;
    });

    return () => {
      window.cancelAnimationFrame(frame);
    };
  }, [messages, composerBottomSpace]);

  useLayoutEffect(() => {
    const activeMenu = openMenu ?? closingMenu;
    if (!activeMenu) {
      setMenuPosition(null);
      return;
    }

    const updateMenuPosition = () => {
      const trigger = triggerRefs.current[activeMenu];
      if (!trigger) {
        return;
      }

      const rect = trigger.getBoundingClientRect();
      const estimatedMenuWidth = 220;
      const viewportPadding = 12;
      const titlebarHeight = 40;
      const verticalOffset = 10;
      const left = Math.min(
        Math.max(viewportPadding, rect.left),
        window.innerWidth - estimatedMenuWidth - viewportPadding,
      );
      const top = Math.max(titlebarHeight + viewportPadding, activeMenu === "files" ? rect.bottom + 10 : rect.top);
      const availableHeight =
        activeMenu === "files"
          ? window.innerHeight - rect.bottom - viewportPadding - verticalOffset
          : rect.top - (titlebarHeight + viewportPadding) - verticalOffset;

      setMenuPosition({
        left,
        top,
        minWidth: rect.width,
        maxHeight: Math.max(180, Math.floor(availableHeight)),
      });
    };

    updateMenuPosition();
    window.addEventListener("resize", updateMenuPosition);
    window.addEventListener("scroll", updateMenuPosition, true);

    return () => {
      window.removeEventListener("resize", updateMenuPosition);
      window.removeEventListener("scroll", updateMenuPosition, true);
    };
  }, [openMenu, closingMenu]);

  useLayoutEffect(() => {
    if (!isContextUsageTooltipOpen) {
      setIsContextUsageTooltipVisible(false);
      if (contextUsageTooltipHideTimeoutRef.current !== null) {
        window.clearTimeout(contextUsageTooltipHideTimeoutRef.current);
      }
      contextUsageTooltipHideTimeoutRef.current = window.setTimeout(() => {
        setContextUsageTooltipPosition(null);
        contextUsageTooltipHideTimeoutRef.current = null;
      }, 180);
      return;
    }

    if (contextUsageTooltipHideTimeoutRef.current !== null) {
      window.clearTimeout(contextUsageTooltipHideTimeoutRef.current);
      contextUsageTooltipHideTimeoutRef.current = null;
    }

    const updateTooltipPosition = () => {
      const trigger = contextUsageButtonRef.current;
      if (!trigger) {
        return;
      }

      const rect = trigger.getBoundingClientRect();
      const viewportPadding = 12;
      const tooltipWidth = contextUsageTooltipRef.current?.offsetWidth ?? 200;
      const tooltipHeight = contextUsageTooltipRef.current?.offsetHeight ?? 44;
      const gap = 10;
      const left = Math.min(
        window.innerWidth - tooltipWidth - viewportPadding,
        Math.max(viewportPadding, rect.right - tooltipWidth),
      );
      const top = Math.max(viewportPadding, rect.top - tooltipHeight - gap);

      setContextUsageTooltipPosition({
        left,
        top,
      });
    };

    updateTooltipPosition();
    const frame = window.requestAnimationFrame(() => {
      updateTooltipPosition();
      setIsContextUsageTooltipVisible(true);
    });
    window.addEventListener("resize", updateTooltipPosition);
    window.addEventListener("scroll", updateTooltipPosition, true);

    return () => {
      window.cancelAnimationFrame(frame);
      window.removeEventListener("resize", updateTooltipPosition);
      window.removeEventListener("scroll", updateTooltipPosition, true);
    };
  }, [isContextUsageTooltipOpen]);

  useEffect(() => {
    return () => {
      if (contextUsageTooltipHideTimeoutRef.current !== null) {
        window.clearTimeout(contextUsageTooltipHideTimeoutRef.current);
      }
    };
  }, []);

  useEffect(() => {
    if (openMenu) {
      setMenuVisible(false);
      if (document.activeElement === textareaRef.current) {
        textareaRef.current?.blur();
      }
      const frame = window.requestAnimationFrame(() => {
        setMenuVisible(true);
      });
      return () => window.cancelAnimationFrame(frame);
    }

    setMenuVisible(false);
    return undefined;
  }, [openMenu]);

  useEffect(() => {
    if (typeof window === "undefined") {
      return;
    }

    const savedKey = window.localStorage.getItem(GEMINI_API_KEY_STORAGE_KEY) ?? "";
    const savedProvider = window.localStorage.getItem(MODEL_PROVIDER_STORAGE_KEY);
    const savedGeminiModel = window.localStorage.getItem(GEMINI_MODEL_STORAGE_KEY) ?? "";
    const savedOllamaModel = window.localStorage.getItem(OLLAMA_MODEL_STORAGE_KEY) ?? "";
    const savedOllamaContextLength = window.localStorage.getItem(OLLAMA_CONTEXT_LENGTH_STORAGE_KEY) ?? "";
    const savedLmStudioModel = window.localStorage.getItem(LMSTUDIO_MODEL_STORAGE_KEY) ?? "";
    const savedLmStudioContextLength = window.localStorage.getItem(LMSTUDIO_CONTEXT_LENGTH_STORAGE_KEY);
    const rawAutoCollapseToolGroups = window.localStorage.getItem(AUTO_COLLAPSE_TOOL_GROUPS_STORAGE_KEY);
    const rawCommandReviewMode = window.localStorage.getItem(COMMAND_REVIEW_MODE_STORAGE_KEY);
    const savedGlobalSystemPrompt = window.localStorage.getItem(GLOBAL_SYSTEM_PROMPT_STORAGE_KEY) ?? "";
    const savedAutoCollapseToolGroups = rawAutoCollapseToolGroups === null ? true : rawAutoCollapseToolGroups === "true";
    const savedCommandReviewMode = rawCommandReviewMode === null ? true : rawCommandReviewMode === "true";
    setGeminiApiKey(savedKey);
    setGeminiApiKeyDraft(savedKey);
    setGlobalSystemPrompt(savedGlobalSystemPrompt);
    setGlobalSystemPromptDraft(savedGlobalSystemPrompt);
    setOllamaContextLength(savedOllamaContextLength);
    setOllamaContextLengthDraft(savedOllamaContextLength);
    setLmStudioContextLength(savedLmStudioContextLength ?? "");
    setLmStudioContextLengthDraft(savedLmStudioContextLength ?? "");
    setAutoCollapseToolGroups(savedAutoCollapseToolGroups);
    setAutoCollapseToolGroupsDraft(savedAutoCollapseToolGroups);
    setCommandReviewMode(savedCommandReviewMode);
    setCommandReviewModeDraft(savedCommandReviewMode);
    if (savedProvider === "gemini" || savedProvider === "ollama" || savedProvider === "lmstudio") {
      setModelProvider(savedProvider);
    }
    setSelectedGeminiModel(savedGeminiModel);
    setSelectedOllamaModel(savedOllamaModel);
    setSelectedLmStudioModel(savedLmStudioModel);
  }, []);

  useEffect(() => {
    const cogent = getCogent();
    if (!cogent || typeof window === "undefined") {
      return;
    }

    let cancelled = false;
    void cogent.getMemoryInfo().then((info) => {
      if (cancelled) {
        return;
      }

      setSystemTotalMemoryBytes(info.totalMemoryBytes);
      setSystemFreeMemoryBytes(info.freeMemoryBytes);
    });

    return () => {
      cancelled = true;
    };
  }, []);

  useEffect(() => {
    const cogent = getCogent();
    if (!cogent) {
      return;
    }

    let cancelled = false;
    void cogent.getOllamaModels().then((result) => {
      if (cancelled) {
        return;
      }

      setIsOllamaAvailable(result.available);
      setOllamaModels(result.models);

      if (result.models.length === 0) {
        if (modelProvider === "ollama") {
          setModelProvider("gemini");
        }
        setSelectedOllamaModel("");
        return;
      }

      const hasCurrentSelection = result.models.some(
        (model) => model.model === selectedOllamaModel || model.name === selectedOllamaModel,
      );

      if (!hasCurrentSelection) {
        setSelectedOllamaModel(result.models[0].model);
      }
    });

    return () => {
      cancelled = true;
    };
  }, []);

  useEffect(() => {
    const cogent = getCogent();
    if (!cogent) {
      return;
    }

    let cancelled = false;
    void cogent.getLmStudioModels().then((result) => {
      if (cancelled) {
        return;
      }

      setIsLmStudioAvailable(result.available);
      setLmStudioModels(result.models);

      if (result.models.length === 0) {
        if (modelProvider === "lmstudio") {
          setModelProvider("gemini");
        }
        setSelectedLmStudioModel("");
        return;
      }

      const hasCurrentSelection = result.models.some((model) => model.id === selectedLmStudioModel);
      if (!hasCurrentSelection) {
        setSelectedLmStudioModel(result.models[0].id);
      }
    });

    return () => {
      cancelled = true;
    };
  }, []);

  useEffect(() => {
    const cogent = getCogent();
    if (!cogent || !geminiApiKey.trim()) {
      setGeminiModels([]);
      return;
    }

    let cancelled = false;
    void cogent.getGeminiModels(geminiApiKey).then((result) => {
      if (cancelled) {
        return;
      }

      if (result.error) {
        console.error("[Gemini models] Failed to fetch model list:", result.error);
      }

      setGeminiModels(result.models);
      if (result.models.length === 0) {
        setSelectedGeminiModel("");
        return;
      }
      const hasCurrentSelection = result.models.some(
        (model) => model.id === selectedGeminiModel,
      );
      if (!hasCurrentSelection) {
        setSelectedGeminiModel(result.models[0].id);
      }
    });

    return () => {
      cancelled = true;
    };
  }, [geminiApiKey]);

  useEffect(() => {
    if (isSettingsOpen) {
      setIsSettingsVisible(false);
      const frame = window.requestAnimationFrame(() => {
        setIsSettingsVisible(true);
      });

      return () => window.cancelAnimationFrame(frame);
    }

    setIsSettingsVisible(false);
    return undefined;
  }, [isSettingsOpen]);

  useEffect(() => {
    if (isResetWarningOpen) {
      setIsResetWarningVisible(false);
      const frame = window.requestAnimationFrame(() => {
        setIsResetWarningVisible(true);
      });

      return () => window.cancelAnimationFrame(frame);
    }

    setIsResetWarningVisible(false);
    return undefined;
  }, [isResetWarningOpen]);

  useEffect(() => {
    if (modelSelectionWarning) {
      setIsModelSelectionWarningVisible(false);
      const frame = window.requestAnimationFrame(() => {
        setIsModelSelectionWarningVisible(true);
      });

      return () => window.cancelAnimationFrame(frame);
    }

    setIsModelSelectionWarningVisible(false);
    return undefined;
  }, [modelSelectionWarning]);

  useEffect(() => {
    if (typeof window === "undefined") {
      return;
    }

    window.localStorage.setItem(MODEL_PROVIDER_STORAGE_KEY, modelProvider);
  }, [modelProvider]);

  useEffect(() => {
    if (typeof window === "undefined") {
      return;
    }

    if (selectedGeminiModel) {
      window.localStorage.setItem(GEMINI_MODEL_STORAGE_KEY, selectedGeminiModel);
    } else {
      window.localStorage.removeItem(GEMINI_MODEL_STORAGE_KEY);
    }
  }, [selectedGeminiModel]);

  useEffect(() => {
    if (typeof window === "undefined") {
      return;
    }

    if (selectedOllamaModel) {
      window.localStorage.setItem(OLLAMA_MODEL_STORAGE_KEY, selectedOllamaModel);
    } else {
      window.localStorage.removeItem(OLLAMA_MODEL_STORAGE_KEY);
    }
  }, [selectedOllamaModel]);

  useEffect(() => {
    if (typeof window === "undefined") {
      return;
    }

    if (ollamaContextLength) {
      window.localStorage.setItem(OLLAMA_CONTEXT_LENGTH_STORAGE_KEY, ollamaContextLength);
    } else {
      window.localStorage.removeItem(OLLAMA_CONTEXT_LENGTH_STORAGE_KEY);
    }
  }, [ollamaContextLength]);

  useEffect(() => {
    const cogent = getCogent();
    if (!cogent) {
      return;
    }

    if (!didInitializeLocalModelCleanupRef.current) {
      didInitializeLocalModelCleanupRef.current = true;
      return;
    }

    const modelId =
      modelProvider === "ollama"
        ? selectedOllamaModel || undefined
        : modelProvider === "lmstudio"
          ? selectedLmStudioModel || undefined
          : undefined;

    void cogent.cleanupLocalModels({
      provider: modelProvider,
      modelId,
    });
  }, [modelProvider, selectedOllamaModel, selectedLmStudioModel]);

  useEffect(() => {
    if (typeof window === "undefined") {
      return;
    }

    if (selectedLmStudioModel) {
      window.localStorage.setItem(LMSTUDIO_MODEL_STORAGE_KEY, selectedLmStudioModel);
    } else {
      window.localStorage.removeItem(LMSTUDIO_MODEL_STORAGE_KEY);
    }
  }, [selectedLmStudioModel]);

  useEffect(() => {
    if (typeof window === "undefined") {
      return;
    }

    window.localStorage.setItem(LMSTUDIO_CONTEXT_LENGTH_STORAGE_KEY, lmStudioContextLength);
  }, [lmStudioContextLength]);

  useEffect(() => {
    if (typeof window === "undefined") {
      return;
    }

    window.localStorage.setItem(AUTO_COLLAPSE_TOOL_GROUPS_STORAGE_KEY, String(autoCollapseToolGroups));
  }, [autoCollapseToolGroups]);

  useEffect(() => {
    if (typeof window === "undefined") return;
    window.localStorage.setItem(COMMAND_REVIEW_MODE_STORAGE_KEY, String(commandReviewMode));
  }, [commandReviewMode]);

  useEffect(() => {
    if (pendingCommandReview) {
      setIsCommandReviewVisible(false);
      const frame = window.requestAnimationFrame(() => setIsCommandReviewVisible(true));
      return () => window.cancelAnimationFrame(frame);
    }
    setIsCommandReviewVisible(false);
    return undefined;
  }, [pendingCommandReview]);

  useEffect(() => {
    if (!activeSession?.id) {
      return;
    }

    updateSessionById(activeSession.id, (session) => ({
      ...session,
      contextUsage: null,
    }));
  }, [activeSession?.id, modelTier, modelProvider, selectedGeminiModel, selectedOllamaModel, ollamaContextLength, selectedLmStudioModel, lmStudioContextLength]);

  useEffect(() => {
    if (!isResizingFileRail) {
      return;
    }

    const handlePointerMove = (event: PointerEvent) => {
      const nextWidth = Math.min(maxFileRailWidth, Math.max(minFileRailWidth, event.clientX - 18));
      setFileRailWidth(nextWidth);
    };

    const handlePointerUp = () => {
      setIsResizingFileRail(false);
    };

    window.addEventListener("pointermove", handlePointerMove);
    window.addEventListener("pointerup", handlePointerUp);

    return () => {
      window.removeEventListener("pointermove", handlePointerMove);
      window.removeEventListener("pointerup", handlePointerUp);
    };
  }, [isResizingFileRail]);

  useEffect(() => {
    const monaco = monacoRef.current;
    const cogent = getCogent();
    if (!monaco || !cogent || !fileTree.length) {
      return;
    }

    const textExtensions = [".ts", ".tsx", ".js", ".jsx", ".json", ".css", ".html", ".md", ".mjs", ".cjs"];

    function flattenFiles(nodes: FileTreeNode[]): string[] {
      return nodes.flatMap((node) => {
        if (node.type === "directory") {
          return flattenFiles(node.children ?? []);
        }

        const normalizedPath = node.path.toLowerCase();
        return textExtensions.some((extension) => normalizedPath.endsWith(extension)) ? [node.path] : [];
      });
    }

    let cancelled = false;

    void (async () => {
      if (!monacoTypescriptConfiguredRef.current) {
        const typescriptLanguage = (monaco.languages as typeof monaco.languages & {
          typescript: {
            typescriptDefaults: {
              setEagerModelSync: (value: boolean) => void;
              setCompilerOptions: (options: Record<string, unknown>) => void;
              setDiagnosticsOptions: (options: Record<string, unknown>) => void;
              addExtraLib: (content: string, filePath?: string) => void;
            };
            javascriptDefaults: {
              setEagerModelSync: (value: boolean) => void;
              setCompilerOptions: (options: Record<string, unknown>) => void;
              setDiagnosticsOptions: (options: Record<string, unknown>) => void;
            };
          };
        }).typescript;

        const compilerOptions = {
          allowJs: true,
          allowNonTsExtensions: true,
          module: 99,
          moduleResolution: 2,
          target: 99,
          jsx: 2,
          esModuleInterop: true,
          allowSyntheticDefaultImports: true,
          resolveJsonModule: true,
          isolatedModules: true,
          strict: true,
          noEmit: true,
        };

        const diagnosticsOptions = {
          noSemanticValidation: false,
          noSyntaxValidation: false,
          noSuggestionDiagnostics: false,
          onlyVisible: false,
        };

        typescriptLanguage.typescriptDefaults.setEagerModelSync(true);
        typescriptLanguage.javascriptDefaults.setEagerModelSync(true);
        typescriptLanguage.typescriptDefaults.setCompilerOptions(compilerOptions);
        typescriptLanguage.javascriptDefaults.setCompilerOptions(compilerOptions);
        typescriptLanguage.typescriptDefaults.setDiagnosticsOptions(diagnosticsOptions);
        typescriptLanguage.javascriptDefaults.setDiagnosticsOptions(diagnosticsOptions);
        monacoTypescriptConfiguredRef.current = true;
      }

      const filePaths = flattenFiles(fileTree);

      if (!sharedTypesLibLoadedRef.current) {
        const sharedTypesPath = filePaths.find((filePath) => filePath.endsWith("packages\\shared-types\\src\\index.ts"));
        if (sharedTypesPath) {
          try {
            const preview = await cogent.readFile(sharedTypesPath);
            const typescriptDefaults = (monaco.languages as typeof monaco.languages & {
              typescript: { typescriptDefaults: { addExtraLib: (content: string, filePath?: string) => void } };
            }).typescript.typescriptDefaults;
            typescriptDefaults.addExtraLib(
              `declare module "@cogent/shared-types" {\n${preview.content}\n}`,
              "file:///cogent-shared-types.d.ts",
            );
            sharedTypesLibLoadedRef.current = true;
          } catch {
            // Ignore declaration sync failures.
          }
        }
      }

    })();

    return () => {
      cancelled = true;
    };
  }, [fileTree, workspaceRoot]);

  useEffect(() => {
    if (!selectedFile) {
      setSaveState("idle");
      return;
    }

    setSaveState(code === selectedFile.content ? "saved" : "dirty");
  }, [code, selectedFile]);

  useEffect(() => {
    if (!selectedFile) {
      setIsEditorOpen(false);
      return;
    }

    if (!shouldAutoOpenEditor) {
      return;
    }

    const frame = window.requestAnimationFrame(() => {
      setIsEditorOpen(true);
      setShouldAutoOpenEditor(false);
    });

    return () => {
      window.cancelAnimationFrame(frame);
    };
  }, [selectedFile, shouldAutoOpenEditor]);

  useEffect(() => {
    const handleKeyDown = (event: KeyboardEvent) => {
      const isSave = (event.ctrlKey || event.metaKey) && event.key.toLowerCase() === "s";
      if (!isSave || !selectedFile) {
        return;
      }

      event.preventDefault();
      void handleSaveFile();
    };

    window.addEventListener("keydown", handleKeyDown);
    return () => {
      window.removeEventListener("keydown", handleKeyDown);
    };
  }, [selectedFile, code]);

  async function refreshContext() {
    const cogent = getCogent();
    if (!cogent) {
      return;
    }

    if (modelProvider === "ollama") {
      const ollamaResult = await cogent.getOllamaModels();
      setIsOllamaAvailable(ollamaResult.available);
      setOllamaModels(ollamaResult.models);
    }

    const conversation = messages.map(toConversationTurn) satisfies ConversationTurn[];
    const taskRequest = buildTaskRequest(prompt, conversation);
    const snapshot = await cogent.buildContextSnapshot(taskRequest);
    const usageSnapshot = await cogent.buildContextUsageSnapshot(taskRequest);
    updateActiveSession((session) => ({ ...session, context: snapshot, contextUsage: usageSnapshot }));
  }

  async function handleRunAgent() {
    const cogent = getCogent();
    const currentSessionId = activeSession?.id;
    if (!cogent || !canSend || !currentSessionId) {
      return;
    }

    if (modelProvider === "gemini" && !geminiApiKey.trim()) {
      const submittedPrompt = prompt.trim();
      const submittedImages = promptImages;
      const submittedFiles = promptFiles;
      const nextSessionTitle = suggestSessionTitle(submittedPrompt, text);

      updateSessionById(currentSessionId, (session) => ({
        ...session,
        name: session.titleState === "empty" ? nextSessionTitle : session.name,
        titleState: session.titleState === "empty" ? "generated" : session.titleState,
        prompt: "",
        messages: [
          ...session.messages,
          {
            id: `${Date.now()}-user`,
            role: "user",
            text: submittedPrompt,
            attachments: submittedImages.length > 0 ? submittedImages : undefined,
            fileAttachments: submittedFiles.length > 0 ? submittedFiles : undefined,
          },
        ],
      }));
      updateDraftImagesForSession(currentSessionId, () => []);
      updateDraftFilesForSession(currentSessionId, () => []);
      appendGeminiApiKeyFallbackMessage(currentSessionId);
      return;
    }

    if (isAgentRunning) {
      await cancelCurrentAgentRun();
    }

    const submittedPrompt = prompt.trim();
    const submittedImages = promptImages;
    const submittedFiles = promptFiles;
    const activeFilePath = selectedFile?.path;
    const conversation = messages.map(toConversationTurn) satisfies ConversationTurn[];
    const requestModelProvider = modelProvider;
    const assistantId = `${Date.now()}-assistant`;
    const diffOriginal = selectedFile?.content;
    const diffPath = activeFilePath;
    const agentRunFileSelectionVersion = manualFileSelectionVersionRef.current;
    let activeTextMessageId = assistantId;
    const nextSessionTitle = suggestSessionTitle(submittedPrompt, text);

    updateSessionById(currentSessionId, (session) => ({
      ...session,
      name: session.titleState === "empty" ? nextSessionTitle : session.name,
      titleState: session.titleState === "empty" ? "generated" : session.titleState,
      prompt: "",
      messages: [
        ...session.messages,
        {
          id: `${Date.now()}-user`,
          role: "user",
          text: submittedPrompt,
          attachments: submittedImages.length > 0 ? submittedImages : undefined,
          fileAttachments: submittedFiles.length > 0 ? submittedFiles : undefined,
        },
        {
          id: assistantId,
          role: "assistant",
          text: "",
          textChunks: [],
          streaming: true,
          status: "requesting",
          diffOriginal,
          diffPath,
        },
      ],
    }));

    let streamedCode = "";
    let streamedMessage = "";
    let streamedTextChunks: TextChunk[] = [];
    let lastWrittenPath: string | null = null;
    let currentStatusPath = diffPath;

    shouldFollowChatRef.current = true;
    pendingScrollBehaviorRef.current = "smooth";
    setIsAgentRunning(true);
    updateDraftImagesForSession(currentSessionId, () => []);
    updateDraftFilesForSession(currentSessionId, () => []);
    const stream = cogent.runAgentTaskStream(
      buildTaskRequest(submittedPrompt, conversation),
      (event) => {
        if (event.type === "mode") {
          setDetectedMode(event.decision.mode);
        }

        if (event.type === "file-write-start") {
          currentStatusPath = event.path;
          updateSessionById(currentSessionId, (session) => ({
            ...session,
            messages: session.messages.map((message) =>
              message.id === activeTextMessageId
                ? { ...message, status: "writing", diffPath: event.path }
                : message,
            ),
          }));
          return;
        }

        if (event.type === "file-read") {
          const fileName = getFileName(event.path);
          const metaLine =
            event.startLine <= 1
              ? text.fileReadSingle(fileName)
              : text.fileReadRange(fileName, event.startLine, event.endLine);
          updateSessionById(currentSessionId, (session) => ({
            ...session,
            messages: session.messages.map((message) =>
              message.id === activeTextMessageId
                ? {
                    ...message,
                    meta: [...(message.meta ?? []), metaLine],
                    status: message.status !== "writing" ? "thinking" : message.status,
                  }
                : message,
            ),
          }));
          return;
        }

        if (event.type === "command-start") {
          const nextAssistantId = `${Date.now()}-assistant-${Math.random().toString(36).slice(2, 8)}`;
          updateSessionById(currentSessionId, (session) => ({
            ...session,
            messages: [
              ...session.messages.map((message) =>
                message.id === activeTextMessageId && isEphemeralAssistantPlaceholder(message)
                  ? {
                      ...message,
                      id: `${Date.now()}-command-${event.commandId}`,
                      status: undefined,
                      activityStatusLabel: undefined,
                      command: {
                        id: event.commandId,
                        command: event.command,
                        output: "",
                        running: true,
                        startedAt: Date.now(),
                      },
                    }
                  : message,
              ),
              {
                id: nextAssistantId,
                role: "assistant",
                text: "",
                diffPath: currentStatusPath,
              },
            ],
          }));
          activeTextMessageId = nextAssistantId;
          return;
        }

        if (event.type === "command") {
          updateSessionById(currentSessionId, (session) => ({
            ...session,
            messages: session.messages.map((message) =>
              message.command?.id === event.commandId
                ? {
                    ...message,
                    command: {
                      ...message.command,
                      output: `${message.command.output}${event.chunk.text}`,
                    },
                  }
                : message,
            ),
          }));
          return;
        }

        if (event.type === "command-end") {
          updateSessionById(currentSessionId, (session) => ({
            ...session,
            messages: session.messages.map((message) =>
              message.command?.id === event.commandId
                ? {
                    ...message,
                    command: {
                      ...message.command,
                      running: false,
                      exitCode: event.exitCode,
                    },
                  }
                : message,
            ),
          }));
          return;
        }

        if (event.type === "command-review-request") {
          if (autoCollapseToolGroups) {
            const toolGroupToCollapse = findLatestToolGroupId(
              sessions.find((s) => s.id === currentSessionId)?.messages ?? [],
            );
            if (toolGroupToCollapse) {
              setCollapsedToolGroups((current) => ({ ...current, [toolGroupToCollapse]: true }));
            }
          }
          if (commandReviewMode) {
            setPendingCommandReview({ reviewId: event.reviewId, command: event.command, cwd: event.cwd });
          } else {
            void getCogent()?.approveCommandReview(event.reviewId);
          }
          return;
        }

        if (event.type === "browser-assist-request") {
          pendingBrowserAssistOpenRef.current = event.request.requestId;
          browserAssistLastActivityRef.current = 0;
          browserAssistBaselineSignatureRef.current = null;
          browserAssistStableRoundsRef.current = 0;
          browserAssistArmedRef.current = false;
          browserAssistArmStableRoundsRef.current = 0;
          setBrowserAssistPanel({
            request: event.request,
            visible: false,
          });
          return;
        }

        if (event.type === "done") {
          // no-op
        }

        if (event.type === "mode" || event.type === "retrieval" || event.type === "status") {
          updateSessionById(currentSessionId, (session) => ({
            ...session,
            messages: session.messages.map((message) =>
                message.id === activeTextMessageId && message.status !== "writing"
                  ? {
                      ...message,
                      status:
                        event.type === "status"
                          ? event.label === "Loading" && requestModelProvider === "ollama"
                            ? "loading"
                            : "thinking"
                          : message.status ?? "thinking",
                      activityStatusLabel:
                        event.type === "status"
                          ? event.label === "Thinking"
                            ? undefined
                            : event.label
                          : message.activityStatusLabel,
                    }
                  : message,
            ),
          }));
          return;
        }

        if (event.type === "message") {
          const chunkStart = streamedMessage.length;
          streamedMessage += event.chunk;
          streamedTextChunks = [
            ...streamedTextChunks,
            {
              id: streamedTextChunks.length + 1,
              start: chunkStart,
              end: streamedMessage.length,
            },
          ];
          let toolGroupToCollapse: string | null = null;
          updateSessionById(currentSessionId, (session) => {
            const nextMessages = session.messages.map((message) =>
              message.id === activeTextMessageId
                ? {
                    ...message,
                    text: streamedMessage,
                    textChunks: streamedTextChunks,
                    streaming: true,
                    status: undefined,
                    activityStatusLabel: undefined,
                  }
                : message,
            );

            if (autoCollapseToolGroups) {
              toolGroupToCollapse = findPreviousToolGroupId(nextMessages, activeTextMessageId);
            }

            return {
              ...session,
              messages: nextMessages,
            };
          });
          if (toolGroupToCollapse) {
            setCollapsedToolGroups((current) => ({ ...current, [toolGroupToCollapse as string]: true }));
          }
          return;
        }

        if (event.type === "code") {
          streamedCode += event.chunk;
          updateSessionById(currentSessionId, (session) => ({
            ...session,
            messages: session.messages.map((message) =>
              message.id === activeTextMessageId
                ? {
                    ...message,
                    status: "writing",
                    activityStatusLabel: undefined,
                    diffPath: currentStatusPath ?? message.diffPath,
                  }
                : message,
            ),
          }));
          return;
        }

        if (event.type === "done") {
          updateSessionById(currentSessionId, (session) => ({
            ...session,
            messages: session.messages.map((message) =>
              message.role === "assistant" && message.status !== undefined
                ? {
                    ...message,
                    streaming: false,
                    status: undefined,
                    activityStatusLabel: undefined,
                    text: message.id === activeTextMessageId && message.text.trim() ? message.text : message.text,
                  }
                : message,
            ),
          }));
          return;
        }

        if (event.type === "file-write") {
          lastWrittenPath = event.path;
          setShouldAutoOpenEditor(false);
          setIsEditorOpen(false);
          streamedCode = event.content;
          setCode(event.content);
          if (manualFileSelectionVersionRef.current === agentRunFileSelectionVersion) {
            setSelectedFile({ path: event.path, content: event.content });
          }
          setSaveState("saved");
          setIsFileRailDismissed(true);
          setCanRestoreFileRail(false);
          const nextAssistantId = `${Date.now()}-assistant-${Math.random().toString(36).slice(2, 8)}`;
          updateSessionById(currentSessionId, (session) => ({
            ...session,
            messages: [
              ...session.messages.map((message) =>
                message.role === "assistant" && message.status !== undefined
                  ? ({ ...message, status: undefined } satisfies ChatMessage)
                  : message,
              ),
              {
                id: `${Date.now()}-diff-${Math.random().toString(36).slice(2, 8)}`,
                role: "assistant",
                text: "",
                code: event.content,
                diffOriginal: event.originalContent,
                diffPath: event.path,
              },
              {
                id: nextAssistantId,
                role: "assistant",
                text: "",
                diffPath: event.path,
              },
            ],
          }));
          activeTextMessageId = nextAssistantId;
          currentStatusPath = event.path;
          streamedMessage = "";
          void cogent.getFileTree().then((tree) => {
            setFileTree(tree.children ?? []);
            setWorkspaceRoot(tree.rootPath);
            setWorkspaceName(tree.name);
            setExpandedPaths((current) => ({
              ...current,
              [tree.rootPath]: true,
            }));
          });
        }

        if (event.type === "file-delete" || event.type === "directory-delete" || event.type === "file-move") {
          if (event.type === "file-delete" && selectedFile?.path === event.path) {
            setIsEditorOpen(false);
            setShouldAutoOpenEditor(false);
            setSelectedFile(null);
            setCode("");
            setSaveState("idle");
          }

          if (event.type === "file-move" && selectedFile?.path === event.fromPath) {
            setSelectedFile((current) => (current ? { ...current, path: event.toPath } : current));
          }

          const metaLine =
            event.type === "directory-delete"
              ? text.directoryDelete(getFileName(event.path))
              : event.type === "file-move"
                ? text.fileMove(getFileName(event.fromPath), getFileName(event.toPath))
                : text.fileDelete(getFileName(event.path));
          updateSessionById(currentSessionId, (session) => ({
            ...session,
            messages: session.messages.map((message) =>
              message.id === activeTextMessageId
                ? {
                    ...message,
                    status: "thinking",
                    meta: metaLine ? [...(message.meta ?? []), metaLine] : message.meta,
                  }
                : message,
            ),
          }));

          void cogent.getFileTree().then((tree) => {
            setFileTree(tree.children ?? []);
            setWorkspaceRoot(tree.rootPath);
            setWorkspaceName(tree.name);
            setExpandedPaths((current) => ({
              ...current,
              [tree.rootPath]: true,
            }));
          });
          return;
        }
      },
    );
    activeStreamRequestIdRef.current = stream.requestId;
    activeStreamCompletedRef.current = stream.completed;

    try {
      await stream.completed;
    } finally {
      if (activeStreamRequestIdRef.current === stream.requestId) {
        activeStreamRequestIdRef.current = null;
      }
      if (activeStreamCompletedRef.current === stream.completed) {
        activeStreamCompletedRef.current = null;
      }
      setIsAgentRunning(false);
    }

    updateSessionById(currentSessionId, (session) => ({
      ...session,
      messages: session.messages
        .map((message) => {
          if (message.role !== "assistant" || message.code !== undefined || message.status === undefined) {
            return message;
          }

          return {
            ...message,
            status: undefined,
          };
        })
        .filter(
          (message) =>
            !(
              message.role === "assistant" &&
              !message.text.trim() &&
              message.code === undefined &&
              !message.command &&
              !message.status &&
              !(message.meta?.length)
            ),
        ),
    }));

    if (lastWrittenPath) {
      try {
        const reloadedFile = await cogent.readFile(lastWrittenPath);
        if (manualFileSelectionVersionRef.current === agentRunFileSelectionVersion) {
          setSelectedFile(reloadedFile);
          setCode(reloadedFile.content);
        }
      } catch {
        // Keep optimistic editor state if disk refresh fails.
      }
    }

    await refreshContext();
  }

  async function handleCancelAgentRun() {
    await cancelCurrentAgentRun();
  }

  function handlePromptKeyDown(event: React.KeyboardEvent<HTMLTextAreaElement>) {
    if (event.key !== "Enter" || event.nativeEvent.isComposing) {
      return;
    }

    if (event.shiftKey) {
      return;
    }

    event.preventDefault();
    if (canSend) {
      void handleRunAgent();
    }
  }

  function handleEditMessageKeyDown(event: React.KeyboardEvent<HTMLTextAreaElement>) {
    if (event.key !== "Enter" || event.nativeEvent.isComposing) {
      return;
    }

    if (event.shiftKey) {
      return;
    }

    event.preventDefault();
    handleSaveMessageEdit();
  }

  function handleCreateSession() {
    createAndSelectSession();
  }

  function handleSelectSession(sessionId: string) {
    didRestoreInitialSessionScrollRef.current = false;
    shouldFollowChatRef.current = true;
    pendingScrollBehaviorRef.current = "auto";
    setActiveSessionId(sessionId);
  }

  function handleDeleteSession(sessionId: string) {
    updateDraftImagesForSession(sessionId, () => []);
    updateDraftFilesForSession(sessionId, () => []);
    if (sessions.length <= 1) {
      createReplacementSession();
      return;
    }

    const remaining = sessions.filter((session) => session.id !== sessionId);
    setSessions(remaining);

    if (activeSessionId === sessionId) {
      didRestoreInitialSessionScrollRef.current = false;
      shouldFollowChatRef.current = true;
      pendingScrollBehaviorRef.current = "auto";
      setActiveSessionId(remaining[0].id);
    }
  }

  function closeMenu() {
    if (!openMenu) {
      return;
    }

    setClosingMenu(openMenu);
    setOpenMenu(null);
    window.setTimeout(() => {
      setClosingMenu((current) => (current === openMenu ? null : current));
    }, 100);
  }

  function blockTitlebarDrag(event: React.MouseEvent | React.PointerEvent) {
    event.preventDefault();
    event.stopPropagation();
  }

  async function handleWindowAction(action: "minimize" | "maximize" | "close") {
    const cogent = getCogent();
    if (!cogent) {
      return;
    }

    if (action === "minimize") {
      await cogent.windowControls.minimize();
      return;
    }

    if (action === "maximize") {
      const state = await cogent.windowControls.maximizeToggle();
      setWindowMaximized(state.maximized);
      return;
    }

    await cogent.windowControls.close();
  }

  async function handleOpenFolder() {
    const cogent = getCogent();
    if (!cogent) {
      return;
    }

    const workspace = await cogent.openFolder();
    if (workspace?.name) {
      setWorkspaceName(workspace.name);
      setWorkspaceRoot(workspace.rootPath);
      const tree = await cogent.getFileTree();
      setFileTree(tree.children ?? []);
      setExpandedPaths({ [tree.rootPath]: true });
    }
    closeMenu();
  }

  async function handleOpenNewWindow() {
    const cogent = getCogent();
    if (!cogent) {
      return;
    }

    await cogent.openNewWindow();
    closeMenu();
  }

  async function handleOpenConsoleWindow() {
    const cogent = getCogent();
    if (!cogent) {
      return;
    }

    await cogent.openConsoleWindow();
    closeMenu();
  }

  function handleOpenSettings() {
    closeMenu();
    setGeminiApiKeyDraft(geminiApiKey);
    setGlobalSystemPromptDraft(globalSystemPrompt);
    setOllamaContextLengthDraft(ollamaContextLength);
    setLmStudioContextLengthDraft(lmStudioContextLength);
    setAutoCollapseToolGroupsDraft(autoCollapseToolGroups);
    setCommandReviewModeDraft(commandReviewMode);
    setIsSettingsOpen(true);
  }

  function handleCloseSettings() {
    setIsSettingsVisible(false);
    window.setTimeout(() => {
      setIsSettingsOpen(false);
    }, 200);
  }

  function handleOpenResetWarning() {
    setIsResetWarningOpen(true);
  }

  function handleCloseResetWarning() {
    setIsResetWarningVisible(false);
    window.setTimeout(() => {
      setIsResetWarningOpen(false);
    }, 200);
  }

  function handleCloseCommandReview() {
    setIsCommandReviewVisible(false);
    window.setTimeout(() => {
      setPendingCommandReview(null);
    }, 200);
  }

  function handleCloseModelSelectionWarning() {
    setIsModelSelectionWarningVisible(false);
    window.setTimeout(() => {
      setModelSelectionWarning(null);
    }, 200);
  }

  function applyLocalModelSelection(provider: "ollama" | "lmstudio", modelId: string) {
    setModelProvider(provider);
    if (provider === "ollama") {
      setSelectedOllamaModel(modelId);
    } else {
      setSelectedLmStudioModel(modelId);
    }
    closeMenu();
  }

  function getLocalModelWarningState(
    provider: "ollama" | "lmstudio",
    model: OllamaModelSummary | LmStudioModelSummary,
  ) {
    const automaticContextLength = Number(getDefaultLmStudioContextLength(systemTotalMemoryBytes));
    const requestedContextLength =
      provider === "ollama"
        ? Number(ollamaContextLength) || automaticContextLength
        : Number(lmStudioContextLength) || automaticContextLength;
    const freeMemoryBytes = systemFreeMemoryBytes ?? 0;
    const usableMemoryBytes = freeMemoryBytes > 0 ? Math.max(2 * 1024 * 1024 * 1024, freeMemoryBytes * 0.8) : 0;
    const weightBytes = estimateModelWeightBytes(provider, model);
    const kvCacheBytes = estimateKvCacheBytes(model, requestedContextLength);
    const runtimeOverheadBytes = 1_500_000_000;
    const fragmentationBytes = weightBytes !== null ? Math.round(weightBytes * 0.12) : 0;
    const backendPeakBytes = weightBytes !== null ? Math.round(weightBytes * 0.08) : 0;
    const estimatedRequiredBytes =
      (weightBytes ?? 0) +
      kvCacheBytes +
      runtimeOverheadBytes +
      fragmentationBytes +
      backendPeakBytes;
    const isContextTooLarge = automaticContextLength > 0 && requestedContextLength > automaticContextLength * 1.1;
    const exceedsUsableMemory = usableMemoryBytes > 0 && estimatedRequiredBytes > usableMemoryBytes;
    const isUnknownHeavyModel =
      weightBytes === null &&
      requestedContextLength >= 8192 &&
      ((provider === "lmstudio" && /(?:30|32|34|35|40|70)b/i.test(`${model.name} ${"id" in model ? model.id : ""}`)) ||
        (provider === "ollama" && /(?:30|32|34|35|40|70)b/i.test(`${model.name} ${"model" in model ? model.model : ""}`)));
    return isContextTooLarge || exceedsUsableMemory || isUnknownHeavyModel;
  }

  function maybeWarnForLocalModelSelection(
    provider: "ollama" | "lmstudio",
    modelId: string,
    model: OllamaModelSummary | LmStudioModelSummary,
  ) {
    if (getLocalModelWarningState(provider, model)) {
      setModelSelectionWarning({
        provider,
        modelId,
        title: text.modelWarningTitle,
        description: text.modelWarningDescription,
      });
      closeMenu();
      return;
    }

    applyLocalModelSelection(provider, modelId);
  }

  function normalizeLmStudioContextLengthInput(value: string) {
    const normalized = value.trim();
    if (!normalized) {
      return "";
    }

    const numericValue = Number(normalized);
    if (!Number.isFinite(numericValue)) {
      return "";
    }

    return String(Math.max(1024, Math.round(numericValue)));
  }

  function normalizeOllamaContextLengthInput(value: string) {
    const normalized = value.trim();
    if (!normalized) {
      return "";
    }

    const numericValue = Number(normalized);
    if (!Number.isFinite(numericValue)) {
      return "";
    }

    return String(Math.max(1024, Math.round(numericValue)));
  }

  async function handleConfirmResetApp() {
    if (isAgentRunning) {
      await cancelCurrentAgentRun();
    }

    const cogent = getCogent();
    const browserRequestId = browserAssistPanel?.request.requestId;
    if (cogent && browserRequestId) {
      try {
        await cogent.cancelBrowserAssist(browserRequestId);
      } catch {}
    }

    if (typeof window !== "undefined") {
      window.localStorage.removeItem(GEMINI_API_KEY_STORAGE_KEY);
      window.localStorage.removeItem(MODEL_PROVIDER_STORAGE_KEY);
      window.localStorage.removeItem(GEMINI_MODEL_STORAGE_KEY);
      window.localStorage.removeItem(OLLAMA_MODEL_STORAGE_KEY);
      window.localStorage.removeItem(OLLAMA_CONTEXT_LENGTH_STORAGE_KEY);
      window.localStorage.removeItem(LMSTUDIO_MODEL_STORAGE_KEY);
      window.localStorage.removeItem(LMSTUDIO_CONTEXT_LENGTH_STORAGE_KEY);
      window.localStorage.removeItem(AUTO_COLLAPSE_TOOL_GROUPS_STORAGE_KEY);
      window.localStorage.removeItem(CHAT_SESSIONS_STORAGE_KEY);
      window.localStorage.removeItem(ACTIVE_SESSION_STORAGE_KEY);
      window.localStorage.removeItem(GLOBAL_SYSTEM_PROMPT_STORAGE_KEY);
    }

    const freshSession = createSession(text);
    didInitializeCollapsedToolGroupsRef.current = false;
    pendingBrowserAssistOpenRef.current = null;
    browserAssistLastActivityRef.current = 0;
    browserAssistLastActivityTypeRef.current = "";
    browserAssistBaselineSignatureRef.current = null;
    browserAssistStableRoundsRef.current = 0;
    browserAssistArmedRef.current = false;
    browserAssistArmStableRoundsRef.current = 0;
    activeStreamRequestIdRef.current = null;
    activeStreamCompletedRef.current = null;
    manualFileSelectionVersionRef.current += 1;

    setSessions([freshSession]);
    setDraftPromptImagesBySession({});
    setDraftPromptFilesBySession({});
    setActiveSessionId(freshSession.id);
    setGeminiApiKey("");
    setGeminiApiKeyDraft("");
    setGlobalSystemPrompt("");
    setGlobalSystemPromptDraft("");
    setModelProvider("gemini");
    setSelectedGeminiModel("");
    setSelectedOllamaModel("");
    setOllamaContextLength("");
    setOllamaContextLengthDraft("");
    setSelectedLmStudioModel("");
    setLmStudioContextLength("");
    setLmStudioContextLengthDraft("");
    setAutoCollapseToolGroups(true);
    setAutoCollapseToolGroupsDraft(true);
    setMode("auto");
    setModeLocked(false);
    setModelTier("flash");
    setThoroughness("balanced");
    setCollapsedToolGroups({});
    setBrowserAssistPanel(null);
    setEditingMessageId(null);
    setEditingMessageText("");
    setSelectedFile(null);
    setCode("");
    setShouldAutoOpenEditor(false);
    setIsEditorOpen(false);
    setSaveState("idle");
    setIsAgentRunning(false);
    setShouldAutoOpenEditor(false);

    handleCloseResetWarning();
    handleCloseSettings();
  }

  function handleSaveSettings() {
    const normalizedKey = geminiApiKeyDraft.trim();
    const normalizedGlobalSystemPrompt = globalSystemPromptDraft.trim();
    const normalizedOllamaContextLength = normalizeOllamaContextLengthInput(ollamaContextLengthDraft);
    const normalizedLmStudioContextLength = normalizeLmStudioContextLengthInput(lmStudioContextLengthDraft);
    setGeminiApiKey(normalizedKey);
    setGlobalSystemPrompt(normalizedGlobalSystemPrompt);
    setOllamaContextLength(normalizedOllamaContextLength);
    setOllamaContextLengthDraft(normalizedOllamaContextLength);
    setLmStudioContextLength(normalizedLmStudioContextLength);
    setLmStudioContextLengthDraft(normalizedLmStudioContextLength);
    setAutoCollapseToolGroups(autoCollapseToolGroupsDraft);
    setCommandReviewMode(commandReviewModeDraft);
    if (typeof window !== "undefined") {
      if (normalizedKey) {
        window.localStorage.setItem(GEMINI_API_KEY_STORAGE_KEY, normalizedKey);
      } else {
        window.localStorage.removeItem(GEMINI_API_KEY_STORAGE_KEY);
      }
      if (normalizedGlobalSystemPrompt) {
        window.localStorage.setItem(GLOBAL_SYSTEM_PROMPT_STORAGE_KEY, normalizedGlobalSystemPrompt);
      } else {
        window.localStorage.removeItem(GLOBAL_SYSTEM_PROMPT_STORAGE_KEY);
      }
      if (normalizedOllamaContextLength) {
        window.localStorage.setItem(OLLAMA_CONTEXT_LENGTH_STORAGE_KEY, normalizedOllamaContextLength);
      } else {
        window.localStorage.removeItem(OLLAMA_CONTEXT_LENGTH_STORAGE_KEY);
      }
      if (normalizedLmStudioContextLength) {
        window.localStorage.setItem(LMSTUDIO_CONTEXT_LENGTH_STORAGE_KEY, normalizedLmStudioContextLength);
      } else {
        window.localStorage.removeItem(LMSTUDIO_CONTEXT_LENGTH_STORAGE_KEY);
      }
    }
    handleCloseSettings();
  }

  async function handleFileSelect(node: FileTreeNode) {
    const cogent = getCogent();
    if (!cogent || node.type !== "file") {
      return;
    }

    const preview = await cogent.readFile(node.path);
    if (preview.missing) {
      const tree = await cogent.getFileTree();
      setFileTree(tree.children ?? []);
      setWorkspaceRoot(tree.rootPath);
      setWorkspaceName(tree.name);
      setExpandedPaths((current) => ({
        ...current,
        [tree.rootPath]: true,
      }));
      if (selectedFile?.path === node.path) {
        setSelectedFile({ path: node.path, content: "", missing: true });
        setCode("");
      }
      manualFileSelectionVersionRef.current += 1;
      setShouldAutoOpenEditor(true);
      setSelectedFile({ path: node.path, content: "", missing: true });
      setCode("");
      setIsFileRailDismissed(true);
      setCanRestoreFileRail(true);
      return;
    }
    manualFileSelectionVersionRef.current += 1;
    setShouldAutoOpenEditor(true);
    setSelectedFile(preview);
    setCode(preview.content);
    setIsFileRailDismissed(true);
    setCanRestoreFileRail(true);
  }

  async function handleOpenDiffPath(path: string) {
    const cogent = getCogent();
    if (!cogent) {
      return;
    }

    const preview = await cogent.readFile(path);
    if (preview.missing) {
      const tree = await cogent.getFileTree();
      setFileTree(tree.children ?? []);
      setWorkspaceRoot(tree.rootPath);
      setWorkspaceName(tree.name);
      setExpandedPaths((current) => ({
        ...current,
        [tree.rootPath]: true,
      }));
      if (selectedFile?.path === path) {
        setSelectedFile({ path, content: "", missing: true });
        setCode("");
      }
      manualFileSelectionVersionRef.current += 1;
      setShouldAutoOpenEditor(true);
      setSelectedFile({ path, content: "", missing: true });
      setCode("");
      setIsFileRailDismissed(true);
      setCanRestoreFileRail(true);
      return;
    }
    manualFileSelectionVersionRef.current += 1;
    setShouldAutoOpenEditor(true);
    setSelectedFile(preview);
    setCode(preview.content);
    setIsFileRailDismissed(true);
    setCanRestoreFileRail(true);
  }

  function handleCloseEditor() {
    setIsEditorOpen(false);
    setShouldAutoOpenEditor(false);
    window.setTimeout(() => {
      setSelectedFile(null);
    }, 200);
  }

  function closeBrowserAssistPanel() {
    pendingBrowserAssistOpenRef.current = null;
    setIsBrowserAssistFloating(false);
    setBrowserAssistPanel((current) => (current ? { ...current, visible: false } : current));
    window.setTimeout(() => {
      setBrowserAssistPanel((current) => (current && !current.visible ? null : current));
    }, 220);
  }

  function getBrowserAssistFloatingBounds() {
    const width = 420;
    const height = 280;
    const margin = 20;
    const titlebarHeight = 40;
    const minLeft = margin + 56;
    const maxLeft = Math.max(minLeft, window.innerWidth - width - margin - 56);
    const minTop = titlebarHeight + margin;
    const maxTop = Math.max(minTop, window.innerHeight - height - margin - 140);
    return { width, height, minLeft, maxLeft, minTop, maxTop };
  }

  function resetBrowserAssistFloatingPosition() {
    const bounds = getBrowserAssistFloatingBounds();
    const nextPosition = {
      left: bounds.maxLeft,
      top: Math.min(bounds.maxTop, bounds.minTop + 24),
    };
    setBrowserAssistFloatingPosition(nextPosition);
    setBrowserAssistFloatingDisplayPosition(nextPosition);
  }

  function handleToggleBrowserAssistFloating() {
    setIsBrowserAssistFloating((current) => {
      const next = !current;
      if (next) {
        resetBrowserAssistFloatingPosition();
      }
      return next;
    });
  }

  function handleBrowserAssistFloatingPointerDown(event: React.PointerEvent<HTMLDivElement>) {
    if (!isBrowserAssistFloating) {
      return;
    }

    setIsBrowserAssistDragging(true);
    browserAssistDragStateRef.current = {
      pointerId: event.pointerId,
      offsetX: event.clientX - browserAssistFloatingDisplayPosition.left,
      offsetY: event.clientY - browserAssistFloatingDisplayPosition.top,
    };
    event.currentTarget.setPointerCapture(event.pointerId);
  }

  function handleBrowserAssistFloatingPointerMove(event: React.PointerEvent<HTMLDivElement>) {
    const dragState = browserAssistDragStateRef.current;
    if (!dragState || dragState.pointerId !== event.pointerId || !isBrowserAssistFloating) {
      return;
    }

    const bounds = getBrowserAssistFloatingBounds();
    const nextLeft = Math.min(bounds.maxLeft, Math.max(bounds.minLeft, event.clientX - dragState.offsetX));
    const nextTop = Math.min(bounds.maxTop, Math.max(bounds.minTop, event.clientY - dragState.offsetY));
    const nextPosition = { left: nextLeft, top: nextTop };
    setBrowserAssistFloatingPosition(nextPosition);
    setBrowserAssistFloatingDisplayPosition(nextPosition);
  }

  function handleBrowserAssistFloatingPointerEnd(event: React.PointerEvent<HTMLDivElement>) {
    const dragState = browserAssistDragStateRef.current;
    if (!dragState || dragState.pointerId !== event.pointerId) {
      return;
    }

    browserAssistDragStateRef.current = null;
    setIsBrowserAssistDragging(false);
    if (event.currentTarget.hasPointerCapture(event.pointerId)) {
      event.currentTarget.releasePointerCapture(event.pointerId);
    }
  }

  function handleOpenBrowserPanelLink(url: string, label?: string) {
    const requestId = `browser-link-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`;
    pendingBrowserAssistOpenRef.current = requestId;
    browserAssistLastActivityRef.current = 0;
    browserAssistLastActivityTypeRef.current = "";
    browserAssistBaselineSignatureRef.current = null;
    browserAssistStableRoundsRef.current = 0;
    browserAssistArmedRef.current = false;
    browserAssistArmStableRoundsRef.current = 0;
    setBrowserAssistPanel({
      request: {
        requestId,
        url,
        title: label?.trim() || "링크 열기",
        description: "본문 링크를 브라우저 패널에서 열었습니다.",
        helpNeeded: "필요하면 이 페이지를 확인하고 계속 진행하세요.",
        steps: [],
      },
      visible: false,
    });
  }

  async function handleCancelBrowserAssist() {
    const cogent = getCogent();
    const requestId = browserAssistPanel?.request.requestId;
    if (cogent && requestId) {
      await cogent.cancelBrowserAssist(requestId);
    }
    closeBrowserAssistPanel();
  }

  useEffect(() => {
    if (!browserAssistPanel || browserAssistPanel.visible) {
      return;
    }

    if (pendingBrowserAssistOpenRef.current !== browserAssistPanel.request.requestId) {
      return;
    }

    const frame = window.requestAnimationFrame(() => {
      setBrowserAssistPanel((current) =>
        current && current.request.requestId === browserAssistPanel.request.requestId
          ? { ...current, visible: true }
          : current,
      );
      pendingBrowserAssistOpenRef.current = null;
    });

    return () => {
      window.cancelAnimationFrame(frame);
    };
  }, [browserAssistPanel]);

  useEffect(() => {
    if (browserAssistPanel) {
      resetBrowserAssistFloatingPosition();
    }
  }, [browserAssistPanel?.request.requestId]);

  useEffect(() => {
    if (!isBrowserAssistFloating || isBrowserAssistDragging) {
      return;
    }

    let frame = 0;
    const animate = () => {
      setBrowserAssistFloatingDisplayPosition((current) => {
        const nextLeft = current.left + (browserAssistFloatingPosition.left - current.left) * 0.18;
        const nextTop = current.top + (browserAssistFloatingPosition.top - current.top) * 0.18;
        const settled =
          Math.abs(nextLeft - browserAssistFloatingPosition.left) < 0.5 &&
          Math.abs(nextTop - browserAssistFloatingPosition.top) < 0.5;

        if (settled) {
          return browserAssistFloatingPosition;
        }

        frame = window.requestAnimationFrame(animate);
        return { left: nextLeft, top: nextTop };
      });
    };

    frame = window.requestAnimationFrame(animate);
    return () => {
      window.cancelAnimationFrame(frame);
    };
  }, [browserAssistFloatingPosition, isBrowserAssistFloating, isBrowserAssistDragging]);

  useEffect(() => {
    if (!browserAssistPanel) {
      return;
    }

    const webview = browserAssistWebviewRef.current;
    const cogent = getCogent();
    if (!webview || !cogent) {
      return;
    }

    let disposed = false;

    const injectActivityTracker = async () => {
      try {
        await webview.executeJavaScript(`
          (() => {
            if (window.__cogentBrowserAssistInstalled) {
              return true;
            }
            window.__cogentBrowserAssistInstalled = true;
            window.__cogentBrowserAssistLastActivity = 0;
            window.__cogentBrowserAssistActivityType = "";
            const mark = (type) => {
              window.__cogentBrowserAssistLastActivity = Date.now();
              window.__cogentBrowserAssistActivityType = type;
            };
            document.addEventListener("click", () => mark("click"), true);
            document.addEventListener("input", () => mark("input"), true);
            document.addEventListener("change", () => mark("change"), true);
            document.addEventListener("keydown", (event) => {
              if (event.key === "Enter" || event.key === "Tab" || event.key === " ") {
                mark("keydown");
              }
            }, true);
            window.addEventListener("scroll", () => mark("scroll"), { passive: true });
            return true;
          })();
        `);
      } catch {}
    };

    const handleDomReady = () => {
      void injectActivityTracker();
    };

    webview.addEventListener("dom-ready", handleDomReady);
    webview.addEventListener("did-navigate", handleDomReady as EventListener);
    webview.addEventListener("did-navigate-in-page", handleDomReady as EventListener);
    void injectActivityTracker();

    const poll = window.setInterval(() => {
      void (async () => {
        try {
          const snapshot = (await webview.executeJavaScript(`
            (() => ({
              activityAt: window.__cogentBrowserAssistLastActivity || 0,
              activityType: window.__cogentBrowserAssistActivityType || "interaction",
              finalUrl: location.href,
              title: document.title || location.href,
              content: (document.body?.innerText || "").replace(/\\n{3,}/g, "\\n\\n").trim().slice(0, 12000),
              links: Array.from(document.querySelectorAll("a[href]")).slice(0, 20).map((link) => ({
                text: (link.textContent || "").trim(),
                url: link.href || "",
              })).filter((link) => link.url),
            }))();
          `)) as Omit<BrowserAssistResult, "requestId"> & { activityAt: number };

          const signaturePayload = {
            finalUrl: snapshot.finalUrl,
            title: snapshot.title,
            contentLength: snapshot.content.length,
            linksLength: snapshot.links.length,
          };
          const signature = JSON.stringify(signaturePayload);

          if (!browserAssistBaselineSignatureRef.current) {
            browserAssistBaselineSignatureRef.current = signature;
            browserAssistStableRoundsRef.current = 0;
            browserAssistArmedRef.current = false;
            browserAssistArmStableRoundsRef.current = 0;
            return;
          }

          const hadUserActivity =
            Boolean(snapshot.activityAt) && snapshot.activityAt > browserAssistLastActivityRef.current;
          const pageChangedAutomatically = signature !== browserAssistBaselineSignatureRef.current;

          if (hadUserActivity) {
            browserAssistLastActivityRef.current = snapshot.activityAt;
            browserAssistLastActivityTypeRef.current = snapshot.activityType;
          }

          if (!browserAssistArmedRef.current) {
            if (!pageChangedAutomatically) {
              browserAssistArmStableRoundsRef.current += 1;
            } else {
              browserAssistArmStableRoundsRef.current = 0;
            }

            if (browserAssistArmStableRoundsRef.current >= 2) {
              browserAssistArmedRef.current = true;
            }

            return;
          }

          const baselinePayload = JSON.parse(browserAssistBaselineSignatureRef.current) as {
            finalUrl: string;
            title: string;
            contentLength: number;
            linksLength: number;
          };
          const urlChanged = snapshot.finalUrl !== browserAssistPanel.request.url;
          const titleChanged = snapshot.title !== browserAssistPanel.request.title;
          const contentChangedSubstantially =
            Math.abs(snapshot.content.length - baselinePayload.contentLength) >= 120;
          const linksChanged = snapshot.links.length !== baselinePayload.linksLength;
          const meaningfulPageChange = urlChanged || titleChanged || contentChangedSubstantially || linksChanged;
          const strongCompletionSignal =
            urlChanged && (titleChanged || contentChangedSubstantially || linksChanged) && pageChangedAutomatically;

          const interactionCanComplete =
            browserAssistLastActivityTypeRef.current !== "input" &&
            browserAssistLastActivityTypeRef.current !== "change";

          if (strongCompletionSignal && interactionCanComplete && (hadUserActivity || meaningfulPageChange)) {
            browserAssistStableRoundsRef.current += 1;
          } else {
            browserAssistStableRoundsRef.current = 0;
          }

          if (browserAssistStableRoundsRef.current < 3) {
            return;
          }

          await cogent.completeBrowserAssist({
            requestId: browserAssistPanel.request.requestId,
            finalUrl: snapshot.finalUrl,
            title: snapshot.title,
            content: snapshot.content,
            links: snapshot.links,
            activityType: snapshot.activityType,
          });

          if (!disposed) {
            closeBrowserAssistPanel();
          }
        } catch {}
      })();
    }, 900);

    return () => {
      disposed = true;
      window.clearInterval(poll);
      webview.removeEventListener("dom-ready", handleDomReady);
      webview.removeEventListener("did-navigate", handleDomReady as EventListener);
      webview.removeEventListener("did-navigate-in-page", handleDomReady as EventListener);
    };
  }, [browserAssistPanel]);

  async function handleSaveFile() {
    const cogent = getCogent();
    if (!cogent || !selectedFile || saveState === "saving") {
      return;
    }

    setSaveState("saving");
    await cogent.writeFile({
      filePath: selectedFile.path,
      content: code,
    });

    setSelectedFile((current) => (current ? { ...current, content: code } : current));
    setSaveState("saved");
  }

  function handleFileRailResizeStart(event: React.PointerEvent<HTMLDivElement>) {
    event.preventDefault();
    event.stopPropagation();
    setIsResizingFileRail(true);
  }

  function toggleDirectory(pathValue: string) {
    setExpandedPaths((current) => ({
      ...current,
      [pathValue]: !current[pathValue],
    }));
  }

  function renderFileTree(nodes: FileTreeNode[], depth = 0): React.ReactNode {
    return nodes.map((node) => {
      const isDirectory = node.type === "directory";
      const isExpanded = isDirectory ? expandedPaths[node.path] ?? false : false;

      return (
        <div key={node.path} className="tree-node">
          <button
            className={`tree-item ${isDirectory ? "is-directory" : "is-file"}`}
            style={{ paddingLeft: `${12 + depth * 14}px` }}
            onClick={() => {
              if (isDirectory) {
                toggleDirectory(node.path);
                return;
              }

              void handleFileSelect(node);
            }}
          >
            <span className="tree-item-chevron">
              {isDirectory ? (
                <span className={`tree-icon tree-chevron ${isExpanded ? "expanded" : ""}`} aria-hidden="true">
                  <svg xmlns="http://www.w3.org/2000/svg" xmlSpace="preserve" viewBox="0 0 32 20">
                    <path d="M31.998 4.144a1.5 1.5 0 0 0-.498-1.033 1.5 1.5 0 0 0-2.117.117L17.674 16.311c-.85.95-2.274.956-3.133.013L2.61 3.218a1.5 1.5 0 0 0-2.12-.1 1.5 1.5 0 0 0-.1 2.12l11.932 13.106c2.023 2.222 5.585 2.206 7.588-.033L31.617 5.228a1.5 1.5 0 0 0 .381-1.084Z" />
                  </svg>
                </span>
              ) : null}
            </span>
            <img
              className="tree-type-image"
              src={
                isDirectory
                  ? getIconUrlByName(isExpanded ? "folder-open" : "folder", MATERIAL_ICONS_BASE_URL)
                  : getIconUrlForFilePath(node.path, MATERIAL_ICONS_BASE_URL)
              }
              alt=""
              aria-hidden="true"
            />
            <span className="tree-item-label">{node.name}</span>
          </button>
          {isDirectory && isExpanded && node.children?.length ? (
            <div className="tree-children">{renderFileTree(node.children, depth + 1)}</div>
          ) : null}
        </div>
      );
    });
  }

  function renderMenu() {
    const visibleMenu = openMenu ?? closingMenu;
    if (!visibleMenu || !menuPosition) {
      return null;
    }

    let content = null;

    if (visibleMenu === "model") {
      content = (
        <>
          <div className="dropdown-copy">
            <strong>{text.modelTitle}</strong>
            <p>{text.modelDescription}</p>
          </div>
          {geminiModels.length > 0 ? (
            <>
              <div className="menu-section-label">{text.modelGeminiSection}</div>
              {geminiModels.map((model) => (
                <button
                  key={model.id}
                  className={modelProvider === "gemini" && selectedGeminiModel === model.id ? "menu-item active" : "menu-item"}
                  onClick={() => {
                    setModelProvider("gemini");
                    setSelectedGeminiModel(model.id);
                    closeMenu();
                  }}
                >
                  <span className="menu-item-title-row">
                    <span className="menu-item-title">{model.name}</span>
                    <span className="menu-item-cloud-icon" aria-hidden="true">
                      <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 32 24"><path d="M15.496 0C10 .013 5.334 3.822 3.994 9.088A7.996 7.996 0 0 0 0 16c0 4.37 3.455 8 7.75 8H24.5c4.124 0 7.5-3.376 7.5-7.5 0-3.138-2.012-5.832-4.889-6.924C25.952 4.083 21.2.002 15.5 0h-.002Zm.004 3h.004a8.981 8.981 0 0 1 8.92 7.86l.13 1.025 1.005.242A4.488 4.488 0 0 1 29 16.5c0 2.503-1.997 4.5-4.5 4.5H7.75C5.141 21 3 18.81 3 16c.001-1.972 1.101-3.731 2.768-4.537a4.816 4.816 0 0 1 2.275-.086c1.246.262 2.631.94 3.516 4.035a1.5 1.5 0 0 0 1.853 1.03 1.5 1.5 0 0 0 1.03-1.854c-1.116-3.906-3.63-5.693-5.782-6.147a7.402 7.402 0 0 0-1.33-.152A8.98 8.98 0 0 1 15.5 3Z"/></svg>
                    </span>
                  </span>
                  <span className="menu-item-description">
                    {getGeminiMenuDescription(model, locale)}
                  </span>
                </button>
              ))}
            </>
          ) : null}
          {isOllamaAvailable && ollamaModels.length > 0 ? (
            <>
              <div className="menu-section-label">{text.modelOllamaSection}</div>
              {ollamaModels.map((model) => (
                (() => {
                  const hasWarning = getLocalModelWarningState("ollama", model);
                  return (
                    <button
                      key={model.model}
                      className={modelProvider === "ollama" && selectedOllamaModel === model.model ? "menu-item active" : "menu-item"}
                      onClick={() => {
                        maybeWarnForLocalModelSelection("ollama", model.model, model);
                      }}
                    >
                      <span className="menu-item-title-row">
                        <span className="menu-item-title">{model.name}</span>
                        {hasWarning ? (
                          <span className="menu-item-warning-icon" aria-hidden="true">
                            <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 32 28">
                              <path d="M16 4.879c-1.299.002-2.597.621-3.328 1.857L.588 27.162C-.866 29.62.989 32.87 3.844 32.87h24.398c2.854 0 4.705-3.252 3.246-5.705L19.334 6.73C18.6 5.494 17.299 4.878 16 4.879zm0 8.494a1.5 1.5 0 0 1 1.5 1.5v6.496a1.5 1.5 0 0 1-3 0v-6.496a1.5 1.5 0 0 1 1.5-1.5zm0 11.496a1.5 1.5 0 1 1 0 3 1.5 1.5 0 0 1 0-3z" transform="translate(0 -4.87)" />
                            </svg>
                          </span>
                        ) : null}
                      </span>
                      <span className={`menu-item-description ${hasWarning ? "is-warning" : ""}`}>
                        {[
                          [model.parameterSize, model.contextLength ? `${formatThousandsWithSuffix(model.contextLength)}` : null]
                            .filter(Boolean)
                            .join(" · "),
                          hasWarning ? text.modelWarningMenu : null,
                        ]
                          .filter(Boolean)
                          .join(" · ")}
                      </span>
                    </button>
                  );
                })()
              ))}
            </>
          ) : null}
          {isLmStudioAvailable && lmStudioModels.length > 0 ? (
            <>
              <div className="menu-section-label">{text.modelLmStudioSection}</div>
              {lmStudioModels.map((model) => (
                (() => {
                  const hasWarning = getLocalModelWarningState("lmstudio", model);
                  return (
                    <button
                      key={model.id}
                      className={modelProvider === "lmstudio" && selectedLmStudioModel === model.id ? "menu-item active" : "menu-item"}
                      onClick={() => {
                        maybeWarnForLocalModelSelection("lmstudio", model.id, model);
                      }}
                    >
                      <span className="menu-item-title-row">
                        <span className="menu-item-title">{model.name}</span>
                        {hasWarning ? (
                          <span className="menu-item-warning-icon" aria-hidden="true">
                            <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 32 28">
                              <path d="M16 4.879c-1.299.002-2.597.621-3.328 1.857L.588 27.162C-.866 29.62.989 32.87 3.844 32.87h24.398c2.854 0 4.705-3.252 3.246-5.705L19.334 6.73C18.6 5.494 17.299 4.878 16 4.879zm0 8.494a1.5 1.5 0 0 1 1.5 1.5v6.496a1.5 1.5 0 0 1-3 0v-6.496a1.5 1.5 0 0 1 1.5-1.5zm0 11.496a1.5 1.5 0 1 1 0 3 1.5 1.5 0 0 1 0-3z" transform="translate(0 -4.87)" />
                            </svg>
                          </span>
                        ) : null}
                      </span>
                      <span className={`menu-item-description ${hasWarning ? "is-warning" : ""}`}>
                        {[model.ownedBy ?? "LM Studio", hasWarning ? text.modelWarningMenu : null]
                          .filter(Boolean)
                          .join(" · ")}
                      </span>
                    </button>
                  );
                })()
              ))}
            </>
          ) : null}
        </>
      );
    }

    if (visibleMenu === "attach") {
      content = (
        <>
          <button
            className="menu-item file-menu-item"
            onClick={handleOpenAttachPicker}
          >
            <span className="menu-item-title">사진 및 파일 첨부</span>
            <span className="menu-item-description">이미지와 텍스트 파일을 현재 요청에 함께 보냅니다.</span>
          </button>
        </>
      );
    }

    if (visibleMenu === "files") {
      content = (
        <>
          <button
            className="menu-item file-menu-item"
            onClick={() => void handleOpenFolder()}
          >
            <span className="menu-item-title">{text.openFolderTitle}</span>
            <span className="menu-item-description">{text.openFolderDescription}</span>
          </button>
          <button
            className="menu-item file-menu-item"
            onClick={() => void handleOpenNewWindow()}
          >
            <span className="menu-item-title">{text.newWindowTitle}</span>
            <span className="menu-item-description">{text.newWindowDescription}</span>
          </button>
          <button
            className="menu-item file-menu-item"
            onClick={() => void handleOpenConsoleWindow()}
          >
            <span className="menu-item-title">{text.openConsoleTitle}</span>
            <span className="menu-item-description">{text.openConsoleDescription}</span>
          </button>
          <button
            className="menu-item file-menu-item"
            onClick={handleOpenSettings}
          >
            <span className="menu-item-title">{text.settingsTitle}</span>
            <span className="menu-item-description">{text.settingsDescription}</span>
          </button>
        </>
      );
    }

    if (visibleMenu === "mode") {
      content = (
        <>
          <div className="dropdown-copy">
            <strong>{text.modeTitle}</strong>
            <p>{text.modeDescription}</p>
          </div>
          <button
            className={mode === "auto" ? "menu-item active" : "menu-item"}
            onClick={() => {
              setMode("auto");
              closeMenu();
            }}
          >
            <span className="menu-item-title">{text.auto}</span>
            <span className="menu-item-description">{text.autoDescription}</span>
          </button>
          <button
            className={mode === "backend" ? "menu-item active" : "menu-item"}
            onClick={() => {
              setMode("backend");
              closeMenu();
            }}
          >
            <span className="menu-item-title">{text.backend}</span>
            <span className="menu-item-description">{text.backendDescription}</span>
          </button>
          <button
            className={mode !== "auto" && mode !== "backend" ? "menu-item active" : "menu-item"}
            onClick={() => {
              setMode("frontend");
              closeMenu();
            }}
          >
            <span className="menu-item-title">{text.frontend}</span>
            <span className="menu-item-description">{text.frontendDescription}</span>
          </button>
        </>
      );
    }

    if (visibleMenu === "thoroughness") {
      content = (
        <>
          <div className="dropdown-copy">
            <strong>{text.thoroughnessTitle}</strong>
            <p>{text.thoroughnessDescription}</p>
          </div>
          <button
            className={thoroughness === "light" ? "menu-item active" : "menu-item"}
            onClick={() => {
              setThoroughness("light");
              closeMenu();
            }}
          >
            <span className="menu-item-title">{text.thoroughnessLight}</span>
            <span className="menu-item-description">{text.thoroughnessLightDescription}</span>
          </button>
          <button
            className={thoroughness === "balanced" ? "menu-item active" : "menu-item"}
            onClick={() => {
              setThoroughness("balanced");
              closeMenu();
            }}
          >
            <span className="menu-item-title">{text.thoroughnessBalanced}</span>
            <span className="menu-item-description">{text.thoroughnessBalancedDescription}</span>
          </button>
          <button
            className={thoroughness === "deep" ? "menu-item active" : "menu-item"}
            onClick={() => {
              setThoroughness("deep");
              closeMenu();
            }}
          >
            <span className="menu-item-title">{text.thoroughnessDeep}</span>
            <span className="menu-item-description">{text.thoroughnessDeepDescription}</span>
          </button>
        </>
      );
    }

    return createPortal(
      <>
        <button className="dropdown-backdrop" aria-label="Close dropdown" onClick={closeMenu} />
        <div
          className={`dropdown-menu dropdown-menu-floating ${
            visibleMenu === "files" ? "dropdown-menu-down" : "dropdown-menu-up"
          } ${visibleMenu === "files" ? "dropdown-menu-files" : ""} ${openMenu ? (menuVisible ? "is-open" : "is-entering") : "is-closing"}`}
          style={{
            left: `${menuPosition.left}px`,
            top: `${menuPosition.top}px`,
            minWidth: `${menuPosition.minWidth}px`,
            maxHeight: `${menuPosition.maxHeight}px`,
          }}
        >
          {content}
        </div>
      </>,
      document.body,
    );
  }

  const contextUsageTooltip =
    isContextUsageTooltipVisible && contextUsageTooltipPosition
      ? createPortal(
          <div
            ref={contextUsageTooltipRef}
            className={`context-usage-tooltip ${isContextUsageTooltipVisible ? "is-open" : ""}`}
            style={{ left: contextUsageTooltipPosition.left, top: contextUsageTooltipPosition.top }}
          >
            <div className="context-usage-tooltip-title">{text.contextUsageTitle(contextUsageDisplay.usagePercent)}</div>
            <div className="context-usage-tooltip-body">
              {`${formatThousandsWithSuffix(contextUsageDisplay.usedTokens)}/${formatThousandsWithSuffix(contextUsageDisplay.inputTokenLimit)}${text.contextEstimate}`}
            </div>
          </div>,
          document.body,
        )
      : null;

  function renderSettingsModal() {
    if (!isSettingsOpen) {
      return null;
    }

    return createPortal(
      <>
        <button
          className={`settings-backdrop ${isSettingsVisible ? "is-open" : "is-closing"}`}
          aria-label={text.settingsClose}
        />
        <section className={`settings-modal ${isSettingsVisible ? "is-open" : "is-closing"}`} aria-label={text.settingsTitle}>
          <div className="settings-header">
            <div className="settings-title-group">
              <strong>{text.settingsTitle}</strong>
              <p>{text.settingsDescription}</p>
            </div>
            <button className="settings-close" onClick={handleCloseSettings} aria-label={text.settingsClose}>
              <svg xmlns="http://www.w3.org/2000/svg" xmlSpace="preserve" viewBox="0 0 32 32">
                <path d="M5.746 4.246a1.5 1.5 0 0 0-1.06.44 1.5 1.5 0 0 0 0 2.12L13.879 16l-9.193 9.193a1.5 1.5 0 0 0 0 2.121 1.5 1.5 0 0 0 2.12 0L16 18.121l9.193 9.193a1.5 1.5 0 0 0 2.121 0 1.5 1.5 0 0 0 0-2.12L18.121 16l9.193-9.193a1.5 1.5 0 0 0 0-2.121 1.5 1.5 0 0 0-1.06-.44 1.5 1.5 0 0 0-1.06.44L16 13.879 6.807 4.686a1.5 1.5 0 0 0-1.06-.44Z" />
              </svg>
            </button>
          </div>
          <div className="settings-body">
            <label className="settings-field">
              <span className="settings-label">{text.settingsApiKeyLabel}</span>
              <span className="settings-description">{text.settingsApiKeyDescription}</span>
              <input
                type="password"
                value={geminiApiKeyDraft}
                onChange={(event) => setGeminiApiKeyDraft(event.target.value)}
                placeholder="AIza..."
                autoComplete="off"
              />
            </label>
            <label className="settings-field">
              <span className="settings-label">{text.settingsOllamaContextLengthLabel}</span>
              <span className="settings-description">{text.settingsOllamaContextLengthDescription}</span>
              <input
                type="text"
                inputMode="numeric"
                value={ollamaContextLengthDraft}
                onChange={(event) => setOllamaContextLengthDraft(event.target.value)}
                placeholder={text.settingsAutomaticPlaceholder}
                autoComplete="off"
              />
            </label>
            <label className="settings-field">
              <span className="settings-label">{text.settingsLmStudioContextLengthLabel}</span>
              <span className="settings-description">{text.settingsLmStudioContextLengthDescription}</span>
              <input
                type="text"
                inputMode="numeric"
                value={lmStudioContextLengthDraft}
                onChange={(event) => setLmStudioContextLengthDraft(event.target.value)}
                placeholder={text.settingsAutomaticPlaceholder}
                autoComplete="off"
              />
            </label>
            <label className="settings-field">
              <span className="settings-label">{text.settingsGlobalSystemPromptLabel}</span>
              <span className="settings-description">{text.settingsGlobalSystemPromptDescription}</span>
              <textarea
                ref={globalSystemPromptTextareaRef}
                value={globalSystemPromptDraft}
                onChange={(event) => setGlobalSystemPromptDraft(event.target.value)}
                rows={2}
                placeholder={text.settingsGlobalSystemPromptLabel}
              />
            </label>
            <label className="settings-toggle">
              <span className="settings-toggle-copy">
                <span className="settings-label">{text.settingsAutoCollapseLabel}</span>
                <span className="settings-description">{text.settingsAutoCollapseDescription}</span>
              </span>
              <button
                type="button"
                className={`settings-switch ${autoCollapseToolGroupsDraft ? "is-on" : ""}`}
                onClick={() => setAutoCollapseToolGroupsDraft((current) => !current)}
                aria-pressed={autoCollapseToolGroupsDraft}
              >
                <span className="settings-switch-knob" />
              </button>
            </label>
            <label className="settings-toggle">
              <span className="settings-toggle-copy">
                <span className="settings-label">{text.settingsCommandReviewLabel}</span>
                <span className="settings-description">{text.settingsCommandReviewDescription}</span>
              </span>
              <button
                type="button"
                className={`settings-switch ${commandReviewModeDraft ? "is-on" : ""}`}
                onClick={() => setCommandReviewModeDraft((current) => !current)}
                aria-pressed={commandReviewModeDraft}
              >
                <span className="settings-switch-knob" />
              </button>
            </label>
            <div className="settings-toggle settings-danger-row">
              <span className="settings-toggle-copy">
                <span className="settings-label">{text.settingsResetAppLabel}</span>
                <span className="settings-description">{text.settingsResetAppDescription}</span>
              </span>
              <button type="button" className="settings-danger-button" onClick={handleOpenResetWarning}>
                {text.settingsResetAppAction}
              </button>
            </div>
          </div>
          <div className="settings-actions">
            <button className="settings-secondary" onClick={handleCloseSettings}>
              {text.cancel}
            </button>
            <button className="settings-primary" onClick={handleSaveSettings}>
              {text.save}
            </button>
          </div>
        </section>
        {isResetWarningOpen ? (
          <>
            <button
              className={`settings-backdrop settings-warning-backdrop ${isResetWarningVisible ? "is-open" : "is-closing"}`}
              aria-label={text.cancel}
            />
            <section
              className={`settings-modal settings-warning-modal ${isResetWarningVisible ? "is-open" : "is-closing"}`}
              aria-label={text.resetWarningTitle}
            >
              <div className="settings-header">
                <div className="settings-title-group">
                  <strong>{text.resetWarningTitle}</strong>
                  <p>{text.resetWarningDescription}</p>
                </div>
                <button className="settings-close" onClick={handleCloseResetWarning} aria-label={text.cancel}>
                  <svg xmlns="http://www.w3.org/2000/svg" xmlSpace="preserve" viewBox="0 0 32 32">
                    <path d="M5.746 4.246a1.5 1.5 0 0 0-1.06.44 1.5 1.5 0 0 0 0 2.12L13.879 16l-9.193 9.193a1.5 1.5 0 0 0 0 2.121 1.5 1.5 0 0 0 2.12 0L16 18.121l9.193 9.193a1.5 1.5 0 0 0 2.121 0 1.5 1.5 0 0 0 0-2.12L18.121 16l9.193-9.193a1.5 1.5 0 0 0 0-2.121 1.5 1.5 0 0 0-1.06-.44 1.5 1.5 0 0 0-1.06.44L16 13.879 6.807 4.686a1.5 1.5 0 0 0-1.06-.44Z" />
                  </svg>
                </button>
              </div>
              <div className="settings-actions settings-warning-actions">
                <button className="settings-secondary" onClick={handleCloseResetWarning}>
                  {text.cancel}
                </button>
                <button className="settings-danger-button" onClick={() => void handleConfirmResetApp()}>
                  {text.settingsResetAppAction}
                </button>
              </div>
            </section>
          </>
        ) : null}
      </>,
      document.body,
    );
  }

  function renderModelSelectionWarning() {
    if (!modelSelectionWarning) {
      return null;
    }

    return createPortal(
      <>
        <button
          className={`settings-backdrop settings-warning-backdrop ${isModelSelectionWarningVisible ? "is-open" : "is-closing"}`}
          aria-label={text.cancel}
        />
        <section
          className={`settings-modal settings-warning-modal settings-model-warning-modal ${isModelSelectionWarningVisible ? "is-open" : "is-closing"}`}
          aria-label={modelSelectionWarning.title}
        >
          <div className="settings-header settings-model-warning-header">
            <div className="settings-title-group settings-model-warning-title-group">
              <div className="settings-model-warning-copy">
                <div className="settings-model-warning-title-line">
                  <strong>{modelSelectionWarning.title}</strong>
                </div>
                <p>{modelSelectionWarning.description}</p>
              </div>
            </div>
            <button className="settings-close" onClick={handleCloseModelSelectionWarning} aria-label={text.cancel}>
              <svg xmlns="http://www.w3.org/2000/svg" xmlSpace="preserve" viewBox="0 0 32 32">
                <path d="M5.746 4.246a1.5 1.5 0 0 0-1.06.44 1.5 1.5 0 0 0 0 2.12L13.879 16l-9.193 9.193a1.5 1.5 0 0 0 0 2.121 1.5 1.5 0 0 0 2.12 0L16 18.121l9.193 9.193a1.5 1.5 0 0 0 2.121 0 1.5 1.5 0 0 0 0-2.12L18.121 16l9.193-9.193a1.5 1.5 0 0 0 0-2.121 1.5 1.5 0 0 0-1.06-.44 1.5 1.5 0 0 0-1.06.44L16 13.879 6.807 4.686a1.5 1.5 0 0 0-1.06-.44Z" />
              </svg>
            </button>
          </div>
          <div className="settings-actions settings-warning-actions">
            <button className="settings-secondary" onClick={handleCloseModelSelectionWarning}>
              {text.cancel}
            </button>
            <button
              className="settings-danger-button"
              onClick={() => {
                applyLocalModelSelection(modelSelectionWarning.provider, modelSelectionWarning.modelId);
                handleCloseModelSelectionWarning();
              }}
            >
              {text.modelWarningConfirm}
            </button>
          </div>
        </section>
      </>,
      document.body,
    );
  }

  return (
    <div className="app-shell">
      <aside
        className={`file-hover-zone ${isResizingFileRail ? "is-resizing" : ""} ${isFileRailDismissed ? "is-dismissed" : ""}`}
        aria-label={text.files}
        onMouseEnter={() => {
          if (isFileRailDismissed && canRestoreFileRail) {
            setIsFileRailDismissed(false)
          }
        }}
        onMouseLeave={() => {
          if (isFileRailDismissed && canRestoreFileRail) {
            setCanRestoreFileRail(true)
          }
        }}
      >
        <div className="file-rail-handle">
          <span>{text.files}</span>
        </div>
        <div className="file-rail-panel" style={{ width: `${fileRailWidth}px` }}>
          <div className="file-rail-content">
            <div className="file-tree" role="tree">
              {renderFileTree(fileTree)}
            </div>
          </div>
          <div
            className="file-rail-resizer"
            onPointerDown={handleFileRailResizeStart}
            role="separator"
            aria-orientation="vertical"
            aria-label="Resize file panel"
          />
        </div>
      </aside>

      <aside className="session-hover-zone" aria-label={text.sessions}>
        <div className="file-rail-handle session-rail-handle">
          <span>{text.sessions}</span>
        </div>
        <div className="file-rail-panel session-rail-panel">
          <div className="file-rail-content">
            <div className="session-list" role="list">
              {sessions.map((session) => (
                <div key={session.id} className={`session-item ${session.id === activeSession?.id ? "active" : ""}`}>
                  <button className="session-select" onClick={() => handleSelectSession(session.id)}>
                    <span className="session-name">{session.name}</span>
                    <span className="session-meta">{getSessionPreview(session, text)}</span>
                  </button>
                  <button
                    className="session-delete"
                    onClick={() => handleDeleteSession(session.id)}
                    aria-label={`Delete ${session.name}`}
                  >
                    <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 32 32"><path d="M5.746 4.246a1.5 1.5 0 0 0-1.06.44 1.5 1.5 0 0 0 0 2.12L13.879 16l-9.193 9.193a1.5 1.5 0 0 0 0 2.121 1.5 1.5 0 0 0 2.12 0L16 18.121l9.193 9.193a1.5 1.5 0 0 0 2.121 0 1.5 1.5 0 0 0 0-2.12L18.121 16l9.193-9.193a1.5 1.5 0 0 0 0-2.121 1.5 1.5 0 0 0-1.06-.44 1.5 1.5 0 0 0-1.06.44L16 13.879 6.807 4.686a1.5 1.5 0 0 0-1.06-.44Z" /></svg>
                  </button>
                </div>
              ))}
            </div>
            <button className="session-create" onClick={handleCreateSession}>
              {text.newSession}
            </button>
          </div>
        </div>
      </aside>

      <main className={`chat-shell ${renderableMessages.length === 0 ? "is-empty" : ""}`} style={{ paddingBottom: `${composerBottomSpace}px` }}>
        <header className="titlebar">
          <div className="titlebar-drag">
            <div className="titlebar-brand">
              <span className="titlebar-logo" aria-hidden="true">
                <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 32 28">
                  <path d="M11.7 0a1.5 1.5 0 0 0-1.06.47l-9.08 9.6a5.74 5.74 0 0 0 0 7.87l9.08 9.6a1.5 1.5 0 0 0 2.13.05 1.5 1.5 0 0 0 .05-2.12l-9.08-9.6a2.7 2.7 0 0 1 0-3.74l9.08-9.6a1.5 1.5 0 0 0-.05-2.12 1.5 1.5 0 0 0-1.08-.4Zm8.6 0a1.5 1.5 0 0 0-1.07.41 1.5 1.5 0 0 0-.05 2.12l9.08 9.6c1 1.06 1 2.68 0 3.74l-9.08 9.6a1.5 1.5 0 0 0 .05 2.12 1.5 1.5 0 0 0 2.13-.06l9.08-9.6a5.74 5.74 0 0 0 0-7.87L21.36.46A1.5 1.5 0 0 0 20.3 0Z" />
                </svg>
              </span>
              <span className="titlebar-title">{text.title}</span>
              <button
                ref={(node) => {
                  triggerRefs.current.files = node;
                }}
                className={`titlebar-select ${openMenu === "files" ? "open" : ""}`}
                onClick={() => setOpenMenu((current) => (current === "files" ? null : "files"))}
                aria-label="Open file options"
              >
                <span className="titlebar-select-label">
                  {workspaceName || text.files}
                </span>
                <span className="custom-select-arrow" aria-hidden="true">
                  <svg xmlns="http://www.w3.org/2000/svg" xmlSpace="preserve" viewBox="0 0 32 20"><path d="M31.998 4.144a1.5 1.5 0 0 0-.498-1.033 1.5 1.5 0 0 0-2.117.117L17.674 16.311c-.85.95-2.274.956-3.133.013L2.61 3.218a1.5 1.5 0 0 0-2.12-.1 1.5 1.5 0 0 0-.1 2.12l11.932 13.106c2.023 2.222 5.585 2.206 7.588-.033L31.617 5.228a1.5 1.5 0 0 0 .381-1.084Z" /></svg>
                </span>
              </button>
            </div>
            <div className="titlebar-actions">
              <button className="titlebar-button" aria-label="Minimize window" onMouseDown={blockTitlebarDrag} onPointerDown={blockTitlebarDrag} onClick={() => void handleWindowAction("minimize")}>
                <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 32 3"><path d="M5.5 0c-2 0-2 3 0 3h21c2 0 2-3 0-3z" /></svg>
              </button>
              <button className="titlebar-button" aria-label="Toggle maximize window" onMouseDown={blockTitlebarDrag} onPointerDown={blockTitlebarDrag} onClick={() => void handleWindowAction("maximize")}>
                {windowMaximized ? (
                  <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 32 30"><path d="M11.5 0a1.5 1.5 0 1 0 0 3h15C27.898 3 29 4.102 29 5.5v15a1.5 1.5 0 0 0 3 0v-15C32 2.48 29.52 0 26.5 0zM4.854 5C2.192 5 0 7.03 0 9.494v16.012C0 27.97 2.192 30 4.854 30h17.292C24.808 30 27 27.97 27 25.506V9.494C27 7.03 24.808 5 22.146 5zm0 2.887h17.292c.977 0 1.737.703 1.737 1.607v16.012c0 .904-.76 1.607-1.737 1.607H4.854c-.977 0-1.737-.703-1.737-1.607V9.494c0-.904.76-1.607 1.737-1.607z" /></svg>
                ) : (
                  <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 32 30"><path d="M5.5 0C2.48 0 0 2.48 0 5.5v19C0 27.52 2.48 30 5.5 30h21c3.02 0 5.5-2.48 5.5-5.5v-19C32 2.48 29.52 0 26.5 0Zm0 3h21C27.898 3 29 4.102 29 5.5v19c0 1.398-1.102 2.5-2.5 2.5h-21A2.478 2.478 0 0 1 3 24.5v-19C3 4.102 4.102 3 5.5 3Z" /></svg>
                )}
              </button>
              <button className="titlebar-button close" aria-label="Close window" onMouseDown={blockTitlebarDrag} onPointerDown={blockTitlebarDrag} onClick={() => void handleWindowAction("close")}>
                <svg xmlns="http://www.w3.org/2000/svg" xmlSpace="preserve" viewBox="0 0 32 32"><path d="M5.746 4.246a1.5 1.5 0 0 0-1.06.44 1.5 1.5 0 0 0 0 2.12L13.879 16l-9.193 9.193a1.5 1.5 0 0 0 0 2.121 1.5 1.5 0 0 0 2.12 0L16 18.121l9.193 9.193a1.5 1.5 0 0 0 2.121 0 1.5 1.5 0 0 0 0-2.12L18.121 16l9.193-9.193a1.5 1.5 0 0 0 0-2.121 1.5 1.5 0 0 0-1.06-.44 1.5 1.5 0 0 0-1.06.44L16 13.879 6.807 4.686a1.5 1.5 0 0 0-1.06-.44Z" /></svg>
              </button>
            </div>
          </div>
        </header>

        <section className="chat-stage">
          <div className={`chat-column ${renderableMessages.length === 0 ? "is-empty" : ""}`}>
            {renderableMessages.length === 0 ? (
              <div className="chat-empty-state" aria-hidden="true">
                <div className="chat-empty-icon">
                  <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 28 32"><path d="M15.94 0a1.5 1.5 0 0 0-1.105.558l-14.505 18c-.787.98-.09 2.436 1.168 2.438h11.328l-1.314 9.291c-.221 1.538 1.757 2.356 2.687 1.11l13.501-18c.74-.987.037-2.396-1.198-2.398h-9.86l.854-9.366A1.5 1.5 0 0 0 15.94 0z" /></svg>
                </div>
                <strong>{text.emptyHeroTitle}</strong>
                <p>{text.emptyHeroDescription}</p>
              </div>
            ) : null}
            {renderItems.map((item) =>
              item.type === "tool-group" ? (
                <AssistantToolGroup
                  key={item.id}
                  groupId={item.id}
                  messages={item.messages}
                  collapsed={Boolean(collapsedToolGroups[item.id])}
                  onToggle={handleToggleToolGroup}
                  localeText={text}
                  selectedFilePath={selectedFile?.path}
                  onOpenDiffPath={handleOpenDiffPath}
                />
              ) : (
                <div key={item.message.id} className={`message-row ${item.message.role === "user" ? "is-user" : "is-assistant"}`}>
                  {item.message.role === "user" && editingMessageId === item.message.id ? (
                    <div className="message-card user-card user-card-editing">
                      <textarea ref={editingTextareaRef} className="message-edit-textarea" value={editingMessageText} onChange={(event) => setEditingMessageText(event.target.value)} onKeyDown={handleEditMessageKeyDown} rows={2} />
                      <div className="message-edit-actions">
                        <button className="message-edit-delete" onClick={() => void handleDeleteMessageEdit()} aria-label={text.deleteMessage} title={text.delete} type="button">
                          <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 30 32"><path d="M11.5 0C9.032 0 7 2.032 7 4.5V6H1.5c-2 0-2 3 0 3H3v17.5C3 29.52 5.48 32 8.5 32h13c3.02 0 5.5-2.48 5.5-5.5V9h1.5c2 0 2-3 0-3H23V4.5C23 2.032 20.968 0 18.5 0Zm0 3h7c.846 0 1.5.654 1.5 1.5V6H10V4.5c0-.846.654-1.5 1.5-1.5ZM6 9h18v17.5c0 1.398-1.102 2.5-2.5 2.5h-13A2.478 2.478 0 0 1 6 26.5Zm5.5 5c-.83 0-1.5.67-1.5 1.5v7c0 2 3 2 3 0v-7c0-.83-.67-1.5-1.5-1.5Zm7 0c-.83 0-1.5.67-1.5 1.5v7c0 2 3 2 3 0v-7c0-.83-.67-1.5-1.5-1.5Z" /></svg>
                        </button>
                        <button className="message-edit-cancel" onClick={handleCancelMessageEdit} aria-label={text.cancel} type="button">
                          <svg xmlns="http://www.w3.org/2000/svg" xmlSpace="preserve" viewBox="0 0 32 32"><path d="M5.746 4.246a1.5 1.5 0 0 0-1.06.44 1.5 1.5 0 0 0 0 2.12L13.879 16l-9.193 9.193a1.5 1.5 0 0 0 0 2.121 1.5 1.5 0 0 0 2.12 0L16 18.121l9.193 9.193a1.5 1.5 0 0 0 2.121 0 1.5 1.5 0 0 0 0-2.12L18.121 16l9.193-9.193a1.5 1.5 0 0 0 0-2.121 1.5 1.5 0 0 0-1.06-.44 1.5 1.5 0 0 0-1.06.44L16 13.879 6.807 4.686a1.5 1.5 0 0 0-1.06-.44Z" /></svg>
                        </button>
                        <button className="message-edit-save" onClick={handleSaveMessageEdit} aria-label={text.send} type="button">
                          <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 32 32"><path d="M16 2.17c-1.4 0-2.82.54-3.88 1.61L.44 15.57a1.5 1.5 0 0 0 0 2.12 1.5 1.5 0 0 0 2.12-.01L14.25 5.9c.08-.09.16-.16.25-.22v23.94a1.5 1.5 0 0 0 1.5 1.5 1.5 1.5 0 0 0 1.5-1.5V5.68c.09.06.17.13.25.22l11.69 11.78a1.5 1.5 0 0 0 2.12 0 1.5 1.5 0 0 0 0-2.11L19.88 3.78A5.45 5.45 0 0 0 16 2.18z" transform="translate(0 .88)" /></svg>
                        </button>
                      </div>
                    </div>
                  ) : (
                    <div className={`message-card ${item.message.role === "user" ? "user-card" : "assistant-card"} ${hasDiffPreview(item.message) ? "has-diff" : ""} ${item.message.status ? "is-status" : ""} ${item.message.role === "user" ? "is-editable" : ""}`} onClick={() => handleStartMessageEdit(item.message)}>
                      {item.message.role === "assistant" && item.message.meta?.length ? (
                        <div className="message-meta">
                          {item.message.meta.map((metaLine, index) => (
                            <div key={`${item.message.id}-meta-${index}`} className="message-meta-line">
                              <span className="message-meta-icon" aria-hidden="true">
                                <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 32 25"><path d="M30.297.021a1.503 1.503 0 0 0-.998.573L12.803 21.916c-.594.768-1.66.727-2.192-.086L2.77 9.85a1.503 1.503 0 0 0-2.08-.434 1.503 1.503 0 0 0-.434 2.08l7.84 11.98c1.613 2.465 5.281 2.61 7.084.28L31.676 2.432a1.503 1.503 0 0 0-.27-2.108 1.503 1.503 0 0 0-1.11-.303Z" transform="matrix(1.00086 0 0 .98256 -.01 -.01)" /></svg>
                              </span>
                              <p>{metaLine}</p>
                            </div>
                          ))}
                        </div>
                      ) : null}
                      {hasDiffPreview(item.message) ? <ChatDiffPreview original={item.message.diffOriginal ?? ""} modified={item.message.code ?? ""} path={item.message.diffPath ?? selectedFile?.path ?? "chat-diff.ts"} openPath={item.message.diffPath} onOpenPath={handleOpenDiffPath} /> : null}
                      {item.message.command ? <CommandPreview command={item.message.command} /> : null}
                      {item.message.role === "user" && item.message.attachments?.length ? renderImageAttachments(item.message.attachments, "message-attachments") : null}
                      {item.message.role === "user" && item.message.fileAttachments?.length ? renderFileAttachments(item.message.fileAttachments, "message-attachments") : null}
                      {item.message.text || item.message.status ? (
                        item.message.status ? (
                          <p className="status-text">{getStatusLabel(text, item.message.status, item.message.activityStatusLabel, item.message.diffPath, locale)}</p>
                        ) : item.message.role === "assistant" ? (
                          <AnimatedMarkdown text={item.message.text} chunks={item.message.textChunks} animate={Boolean(item.message.streaming)} onOpenLink={handleAssistantLink} />
                        ) : (
                          <p>{item.message.text}</p>
                        )
                      ) : null}
                    </div>
                  )}
                </div>
              ),
            )}
          </div>
        </section>

        {browserAssistPanel ? (
          <section
            className={`editor-stage browser-stage ${browserAssistPanel.visible ? "is-open" : "is-closing"} ${isBrowserAssistFloating ? "is-floating" : ""} ${isBrowserAssistDragging ? "is-dragging" : ""}`}
            aria-label="Browser assistance"
            style={
              isBrowserAssistFloating
                ? {
                    left: `${browserAssistFloatingDisplayPosition.left}px`,
                    top: `${browserAssistFloatingDisplayPosition.top}px`,
                    width: `${getBrowserAssistFloatingBounds().width}px`,
                    height: `${getBrowserAssistFloatingBounds().height}px`,
                  }
                : undefined
            }
          >
            <div className="editor-shell browser-shell">
              <div
                className={`editor-toolbar browser-toolbar ${isBrowserAssistFloating ? "is-draggable" : ""}`}
                onPointerDown={handleBrowserAssistFloatingPointerDown}
                onPointerMove={handleBrowserAssistFloatingPointerMove}
                onPointerUp={handleBrowserAssistFloatingPointerEnd}
                onPointerCancel={handleBrowserAssistFloatingPointerEnd}
              >
                <div className="editor-meta browser-meta">
                  <span className="editor-name">{browserAssistPanel.request.title}</span>
                  <span className="editor-path">{browserAssistPanel.request.url}</span>
                </div>
                <div className="editor-actions">
                  <button
                    className="editor-close"
                    onClick={handleToggleBrowserAssistFloating}
                    aria-label={isBrowserAssistFloating ? "Expand browser help" : "Float browser help"}
                  >
                    {isBrowserAssistFloating ? (
                      <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 32 30"><path d="M5.5 0C2.48 0 0 2.48 0 5.5v19C0 27.52 2.48 30 5.5 30h21c3.02 0 5.5-2.48 5.5-5.5v-19C32 2.48 29.52 0 26.5 0Zm0 3h21C27.898 3 29 4.102 29 5.5v19c0 1.398-1.102 2.5-2.5 2.5h-21A2.478 2.478 0 0 1 3 24.5v-19C3 4.102 4.102 3 5.5 3Z" /></svg>
                    ) : (
                      <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 32 30"><path d="M11.5 0a1.5 1.5 0 1 0 0 3h15C27.898 3 29 4.102 29 5.5v15a1.5 1.5 0 0 0 3 0v-15C32 2.48 29.52 0 26.5 0zM4.854 5C2.192 5 0 7.03 0 9.494v16.012C0 27.97 2.192 30 4.854 30h17.292C24.808 30 27 27.97 27 25.506V9.494C27 7.03 24.808 5 22.146 5zm0 2.887h17.292c.977 0 1.737.703 1.737 1.607v16.012c0 .904-.76 1.607-1.737 1.607H4.854c-.977 0-1.737-.703-1.737-1.607V9.494c0-.904.76-1.607 1.737-1.607z" /></svg>
                    )}
                  </button>
                  <button className="editor-close" onClick={closeBrowserAssistPanel} aria-label="Close browser help">
                    <svg xmlns="http://www.w3.org/2000/svg" xmlSpace="preserve" viewBox="0 0 32 32"><path d="M5.746 4.246a1.5 1.5 0 0 0-1.06.44 1.5 1.5 0 0 0 0 2.12L13.879 16l-9.193 9.193a1.5 1.5 0 0 0 0 2.121 1.5 1.5 0 0 0 2.12 0L16 18.121l9.193 9.193a1.5 1.5 0 0 0 2.121 0 1.5 1.5 0 0 0 0-2.12L18.121 16l9.193-9.193a1.5 1.5 0 0 0 0-2.121 1.5 1.5 0 0 0-1.06-.44 1.5 1.5 0 0 0-1.06.44L16 13.879 6.807 4.686a1.5 1.5 0 0 0-1.06-.44Z" /></svg>
                  </button>
                </div>
              </div>
              <div className={`browser-assist-copy ${isBrowserAssistFloating ? "is-floating" : ""}`}>
                <div className="browser-assist-summary">
                  <strong>{browserAssistPanel.request.helpNeeded}</strong>
                  <span className="browser-assist-hint">호버해서 자세히 보기</span>
                </div>
                <div className="browser-assist-details">
                  <p>{browserAssistPanel.request.description}</p>
                  <ol className="browser-assist-steps">
                    {browserAssistPanel.request.steps.map((step, index) => (
                      <li key={`${browserAssistPanel.request.requestId}-step-${index}`}>{step}</li>
                    ))}
                  </ol>
                </div>
              </div>
              <div className="browser-assist-surface">
                <webview
                  ref={browserAssistWebviewRef}
                  className="browser-assist-webview"
                  src={browserAssistPanel.request.url}
                  partition={`persist:cogent-browser-assist-${activeSession?.id ?? "default"}`}
                  allowpopups={true}
                />
              </div>
            </div>
          </section>
        ) : null}

        {selectedFile ? (
          <section className={`editor-stage ${isEditorOpen ? "is-open" : "is-closing"}`} aria-label="Code editor">
            <div className="editor-shell">
              <div className="editor-toolbar">
                <div className="editor-meta">
                  <span className="editor-name">{selectedFile.path.split("\\").pop() ?? "Untitled"}</span>
                  <span className="editor-path">{selectedFile.path}</span>
                </div>
                <div className="editor-actions">
                  {saveState !== "idle" ? <span className="editor-status">{saveState === "saving" ? text.saving : saveState === "dirty" ? text.modified : saveState === "saved" ? text.saved : ""}</span> : null}
                  <button className={`editor-save ${saveState === "dirty" ? "ready" : ""}`} onClick={() => void handleSaveFile()} aria-label="Save file" disabled={saveState === "saving" || saveState === "saved"}>
                    Save
                  </button>
                  <button className="editor-close" onClick={handleCloseEditor} aria-label="Close editor">
                    <svg xmlns="http://www.w3.org/2000/svg" xmlSpace="preserve" viewBox="0 0 32 32"><path d="M5.746 4.246a1.5 1.5 0 0 0-1.06.44 1.5 1.5 0 0 0 0 2.12L13.879 16l-9.193 9.193a1.5 1.5 0 0 0 0 2.121 1.5 1.5 0 0 0 2.12 0L16 18.121l9.193 9.193a1.5 1.5 0 0 0 2.121 0 1.5 1.5 0 0 0 0-2.12L18.121 16l9.193-9.193a1.5 1.5 0 0 0 0-2.121 1.5 1.5 0 0 0-1.06-.44 1.5 1.5 0 0 0-1.06.44L16 13.879 6.807 4.686a1.5 1.5 0 0 0-1.06-.44Z" /></svg>
                  </button>
                </div>
              </div>
              <div className="editor-surface">
                <MonacoCodeEditor
                  onMount={({ editor, monaco }) => {
                    editorInstanceRef.current = editor;
                    monacoRef.current = monaco;
                    editor.updateOptions({ "semanticHighlighting.enabled": true } as never);
                  }}
                  path={selectedFile.path}
                  value={code}
                  onChange={(value) => setCode(value)}
                  options={{
                    minimap: { enabled: true, side: "right", size: "proportional", showSlider: "mouseover" },
                    fontSize: 14,
                    roundedSelection: true,
                    scrollbar: { verticalScrollbarSize: 10, horizontalScrollbarSize: 10 },
                    smoothScrolling: true,
                  }}
                />
              </div>
            </div>
          </section>
        ) : null}

        <div ref={composerWrapRef} className="composer-wrap">
          <div className={`composer ${isComposerDragActive ? "is-drag-active" : ""} ${openMenu ? "is-menu-open" : ""}`} onDragOver={handleComposerDragOver} onDragLeave={handleComposerDragLeave} onDrop={(event) => void handleComposerDrop(event)}>
            <input
              ref={composerFileInputRef}
              className="composer-file-input"
              type="file"
              multiple
              onChange={(event) => void handleComposerFileInputChange(event)}
            />
            {promptImages.length > 0 || promptFiles.length > 0 ? (
              <div className="composer-attachments">
                {promptImages.map((image) => (
                  <div key={image.id} className="composer-attachment">
                    <img src={image.dataUrl} alt={image.name} className="composer-attachment-preview" />
                    <span className="composer-attachment-name" title={image.name}>{image.name}</span>
                    <button type="button" className="composer-attachment-remove" onClick={() => handleRemovePromptImage(image.id)} aria-label="Remove image">
                      <svg xmlns="http://www.w3.org/2000/svg" xmlSpace="preserve" viewBox="0 0 32 32"><path d="M5.746 4.246a1.5 1.5 0 0 0-1.06.44 1.5 1.5 0 0 0 0 2.12L13.879 16l-9.193 9.193a1.5 1.5 0 0 0 0 2.121 1.5 1.5 0 0 0 2.12 0L16 18.121l9.193 9.193a1.5 1.5 0 0 0 2.121 0 1.5 1.5 0 0 0 0-2.12L18.121 16l9.193-9.193a1.5 1.5 0 0 0 0-2.121 1.5 1.5 0 0 0-1.06-.44 1.5 1.5 0 0 0-1.06.44L16 13.879 6.807 4.686a1.5 1.5 0 0 0-1.06-.44Z" /></svg>
                    </button>
                  </div>
                ))}
                {promptFiles.map((file) => (
                  <div key={file.id} className="composer-file-attachment">
                    <img
                      className="file-attachment-type-image"
                      src={getIconUrlForFilePath(file.name, MATERIAL_ICONS_BASE_URL)}
                      alt=""
                      aria-hidden="true"
                    />
                    <span className="composer-file-attachment-name" title={file.name}>{file.name}</span>
                    <button type="button" className="composer-attachment-remove" onClick={() => handleRemovePromptFile(file.id)} aria-label="Remove file">
                      <svg xmlns="http://www.w3.org/2000/svg" xmlSpace="preserve" viewBox="0 0 32 32"><path d="M5.746 4.246a1.5 1.5 0 0 0-1.06.44 1.5 1.5 0 0 0 0 2.12L13.879 16l-9.193 9.193a1.5 1.5 0 0 0 0 2.121 1.5 1.5 0 0 0 2.12 0L16 18.121l9.193 9.193a1.5 1.5 0 0 0 2.121 0 1.5 1.5 0 0 0 0-2.12L18.121 16l9.193-9.193a1.5 1.5 0 0 0 0-2.121 1.5 1.5 0 0 0-1.06-.44 1.5 1.5 0 0 0-1.06.44L16 13.879 6.807 4.686a1.5 1.5 0 0 0-1.06-.44Z" /></svg>
                    </button>
                  </div>
                ))}
              </div>
            ) : null}
            <textarea ref={textareaRef} value={prompt} onChange={(event) => updateActiveSession((session) => ({ ...session, prompt: event.target.value }))} onKeyDown={handlePromptKeyDown} rows={2} placeholder={text.inputPlaceholder} />
            <div className="composer-actions">
              <div className="dropdown-group">
                <div className="dropdown-shell">
                  <button
                    ref={(node) => { triggerRefs.current.attach = node; }}
                    className={`attach-button ${openMenu === "attach" ? "open" : ""}`}
                    onMouseDown={preventMouseFocus}
                    onPointerDown={preventMouseFocus}
                    onClick={() => setOpenMenu((current) => (current === "attach" ? null : "attach"))}
                    aria-label="사진 및 파일 첨부"
                  >
                    <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 32 32">
                      <path d="M16 0a1.5 1.5 0 0 0-1.5 1.5v13h-13A1.5 1.5 0 0 0 0 16a1.5 1.5 0 0 0 1.5 1.5h13v13A1.5 1.5 0 0 0 16 32a1.5 1.5 0 0 0 1.5-1.5v-13h13A1.5 1.5 0 0 0 32 16a1.5 1.5 0 0 0-1.5-1.5h-13v-13A1.5 1.5 0 0 0 16 0Z" />
                    </svg>
                  </button>
                </div>
                <div className="dropdown-shell"><button ref={(node) => { triggerRefs.current.model = node; }} className={`custom-select ${openMenu === "model" ? "open" : ""}`} onMouseDown={preventMouseFocus} onPointerDown={preventMouseFocus} onClick={() => setOpenMenu((current) => (current === "model" ? null : "model"))} aria-label="Open model options"><span className="custom-select-main"><span className="custom-select-icon" aria-hidden="true"><svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 32 28"><path d="M16 0a2.5 2.5 0 0 0-2.5 2.5 2.5 2.5 0 0 0 1 1.992V6h-5C6.48 6 4 8.48 4 11.5v11C4 25.52 6.48 28 9.5 28h13c3.02 0 5.5-2.48 5.5-5.5v-11C28 8.48 25.52 6 22.5 6h-5V4.492a2.5 2.5 0 0 0 1-1.992A2.5 2.5 0 0 0 16 0ZM9.5 9h13c1.398 0 2.5 1.102 2.5 2.5v11c0 1.398-1.102 2.5-2.5 2.5h-13A2.478 2.478 0 0 1 7 22.5v-11C7 10.102 8.102 9 9.5 9Zm-8 3A1.5 1.5 0 0 0 0 13.5v6a1.5 1.5 0 0 0 3 0v-6A1.5 1.5 0 0 0 1.5 12Zm29 0a1.5 1.5 0 0 0-1.5 1.5v6a1.5 1.5 0 0 0 3 0v-6a1.5 1.5 0 0 0-1.5-1.5ZM12 15a2 2 0 1 0 0 4 2 2 0 0 0 0-4zm8 0a2 2 0 1 0 0 4 2 2 0 0 0 0-4z" /></svg></span><span className="custom-select-label">{activeModelLabel}</span></span><span className="custom-select-arrow" aria-hidden="true"><svg xmlns="http://www.w3.org/2000/svg" xmlSpace="preserve" viewBox="0 0 32 20"><path d="M31.998 4.144a1.5 1.5 0 0 0-.498-1.033 1.5 1.5 0 0 0-2.117.117L17.674 16.311c-.85.95-2.274.956-3.133.013L2.61 3.218a1.5 1.5 0 0 0-2.12-.1 1.5 1.5 0 0 0-.1 2.12l11.932 13.106c2.023 2.222 5.585 2.206 7.588-.033L31.617 5.228a1.5 1.5 0 0 0 .381-1.084Z" /></svg></span></button></div>
                <div className="dropdown-shell"><button ref={(node) => { triggerRefs.current.mode = node; }} className={`custom-select ${openMenu === "mode" ? "open" : ""}`} onMouseDown={preventMouseFocus} onPointerDown={preventMouseFocus} onClick={() => setOpenMenu((current) => (current === "mode" ? null : "mode"))} aria-label="Open mode options"><span className="custom-select-main"><span className="custom-select-icon mode-icon" aria-hidden="true"><svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 32 32"><path d="M10.49 0C9.14 0 7.8.28 6.56.8a1.5 1.5 0 0 0-.53 2.4l4.7 5.12-2.41 2.42L3.2 6.03a1.5 1.5 0 0 0-2.4.53 10.5 10.5 0 0 0-.8 3.93 1.5 1.5 0 0 0 0 .01C0 16.28 4.72 21 10.5 21a1.5 1.5 0 0 0 .01 0c.98 0 1.91-.3 2.85-.57L23.9 30.98a3.52 3.52 0 0 0 4.95 0l2.12-2.12a3.52 3.52 0 0 0 0-4.95L20.43 13.36c.27-.94.56-1.87.57-2.85a1.5 1.5 0 0 0 0-.01C21 4.72 16.28 0 10.5 0a1.5 1.5 0 0 0-.01 0zm.02 3a7.47 7.47 0 0 1 6.93 10.3 1.5 1.5 0 0 0 .32 1.63l11.1 11.1c.2.2.2.5 0 .7l-2.12 2.13c-.21.2-.5.2-.7 0l-11.1-11.1a1.5 1.5 0 0 0-1.64-.32A7.47 7.47 0 0 1 3 10.5c0-.17.09-.33.1-.5l4.25 3.9a1.5 1.5 0 0 0 2.07-.04l4.46-4.45a1.5 1.5 0 0 0 .04-2.07l-3.9-4.25c.16-.01.32-.1.49-.1z" /></svg></span><span className="custom-select-label">{mode === "auto" ? text.auto : getModeLabel(mode, text)}</span></span><span className="custom-select-arrow" aria-hidden="true"><svg xmlns="http://www.w3.org/2000/svg" xmlSpace="preserve" viewBox="0 0 32 20"><path d="M31.998 4.144a1.5 1.5 0 0 0-.498-1.033 1.5 1.5 0 0 0-2.117.117L17.674 16.311c-.85.95-2.274.956-3.133.013L2.61 3.218a1.5 1.5 0 0 0-2.12-.1 1.5 1.5 0 0 0-.1 2.12l11.932 13.106c2.023 2.222 5.585 2.206 7.588-.033L31.617 5.228a1.5 1.5 0 0 0 .381-1.084Z" /></svg></span></button></div>
                <div className="dropdown-shell"><button ref={(node) => { triggerRefs.current.thoroughness = node; }} className={`custom-select ${openMenu === "thoroughness" ? "open" : ""}`} onMouseDown={preventMouseFocus} onPointerDown={preventMouseFocus} onClick={() => setOpenMenu((current) => (current === "thoroughness" ? null : "thoroughness"))} aria-label="Open thoroughness options"><span className="custom-select-main"><span className="custom-select-icon" aria-hidden="true"><svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 32"><path d="M12 0C5.39 0 0 5.39 0 12v.004c.012 4.568 2.689 8.608 6.723 10.615V25.5a1.5 1.5 0 0 0 1.5 1.5h7.554a1.5 1.5 0 0 0 1.5-1.5v-2.88c4.034-2.008 6.711-6.048 6.723-10.616V12c0-6.61-5.39-12-12-12Zm0 3a8.977 8.977 0 0 1 9 8.998 8.989 8.989 0 0 1-5.762 8.371 1.5 1.5 0 0 0-.96 1.4V24H9.722v-2.23a1.5 1.5 0 0 0-.961-1.4A8.989 8.989 0 0 1 3 12v-.004A8.977 8.977 0 0 1 12 3ZM9.5 29a1.5 1.5 0 0 0 0 3h5a1.5 1.5 0 0 0 0-3Z" /></svg></span><span className="custom-select-label">{thoroughness === "light" ? text.thoroughnessLight : thoroughness === "deep" ? text.thoroughnessDeep : text.thoroughnessBalanced}</span></span><span className="custom-select-arrow" aria-hidden="true"><svg xmlns="http://www.w3.org/2000/svg" xmlSpace="preserve" viewBox="0 0 32 20"><path d="M31.998 4.144a1.5 1.5 0 0 0-.498-1.033 1.5 1.5 0 0 0-2.117.117L17.674 16.311c-.85.95-2.274.956-3.133.013L2.61 3.218a1.5 1.5 0 0 0-2.12-.1 1.5 1.5 0 0 0-.1 2.12l11.932 13.106c2.023 2.222 5.585 2.206 7.588-.033L31.617 5.228a1.5 1.5 0 0 0 .381-1.084Z" /></svg></span></button></div>
              </div>
              <div className="context-usage-shell" aria-label={`${text.contextUsage} ${contextUsageDisplay.usagePercent}%`} onMouseEnter={() => setIsContextUsageTooltipOpen(true)} onMouseLeave={() => setIsContextUsageTooltipOpen(false)}>
                <button ref={contextUsageButtonRef} className="context-usage-button" type="button" onMouseDown={preventMouseFocus} onPointerDown={preventMouseFocus} onFocus={() => setIsContextUsageTooltipOpen(true)} onBlur={() => setIsContextUsageTooltipOpen(false)}>
                  <svg className="context-usage-ring" viewBox="0 0 32 32" aria-hidden="true">
                    <circle className="context-usage-track" cx={contextCircleCenter} cy={contextCircleCenter} r={contextCircleRadius} />
                    <circle className="context-usage-progress" cx={contextCircleCenter} cy={contextCircleCenter} r={contextCircleRadius} strokeDasharray={contextCircleCircumference} strokeDashoffset={contextCircleOffset} />
                  </svg>
                </button>
              </div>
              <button className={`send-button ${canSend || isAgentRunning ? "ready" : ""} ${isAgentRunning ? "is-cancel" : ""}`} onClick={() => void (isAgentRunning ? handleCancelAgentRun() : handleRunAgent())} aria-label={isAgentRunning ? "Cancel response" : text.send} disabled={!isAgentRunning && !canSend}>
                {isAgentRunning ? (<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 32 32"><path d="M5.5 0C2.48 0 0 2.48 0 5.5v21C0 29.52 2.48 32 5.5 32h21c3.02 0 5.5-2.48 5.5-5.5v-21C32 2.48 29.52 0 26.5 0Z" /></svg>) : (<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 32 32"><path d="M16 2.17c-1.4 0-2.82.54-3.88 1.61L.44 15.57a1.5 1.5 0 0 0 0 2.12 1.5 1.5 0 0 0 2.12-.01L14.25 5.9c.08-.09.16-.16.25-.22v23.94a1.5 1.5 0 0 0 1.5 1.5 1.5 1.5 0 0 0 1.5-1.5V5.68c.09.06.17.13.25.22l11.69 11.78a1.5 1.5 0 0 0 2.12 0 1.5 1.5 0 0 0 0-2.11L19.88 3.78A5.45 5.45 0 0 0 16 2.18z" transform="translate(0 .88)" /></svg>)}
              </button>
            </div>
          </div>
        </div>
      </main>
      {renderMenu()}
      {renderSettingsModal()}
      {renderModelSelectionWarning()}
      {pendingCommandReview
        ? createPortal(
            <>
              <button
                className={`settings-backdrop settings-warning-backdrop ${isCommandReviewVisible ? "is-open" : "is-closing"}`}
                aria-label={text.cancel}
                onClick={() => {
                  void getCogent()?.cancelCommandReview(pendingCommandReview.reviewId);
                  handleCloseCommandReview();
                }}
              />
              <section
                className={`settings-modal settings-warning-modal ${isCommandReviewVisible ? "is-open" : "is-closing"}`}
                aria-label={text.commandReviewTitle}
              >
                <div className="settings-header">
                  <div className="settings-title-group">
                    <strong>{text.commandReviewTitle}</strong>
                    <p style={{ fontFamily: "monospace", wordBreak: "break-all", marginTop: 4 }}>
                      {pendingCommandReview.cwd ? `${pendingCommandReview.cwd} $ ` : "$ "}
                      {pendingCommandReview.command}
                    </p>
                  </div>
                  <button
                    className="settings-close"
                    aria-label={text.cancel}
                    onClick={() => {
                      void getCogent()?.cancelCommandReview(pendingCommandReview.reviewId);
                      handleCloseCommandReview();
                    }}
                  >
                    <svg width="16" height="16" viewBox="0 0 16 16" fill="none">
                      <path d="M12 4L4 12M4 4l8 8" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" />
                    </svg>
                  </button>
                </div>
                <div className="settings-actions settings-warning-actions">
                  <button
                    className="settings-secondary"
                    onClick={() => {
                      void getCogent()?.cancelCommandReview(pendingCommandReview.reviewId);
                      handleCloseCommandReview();
                    }}
                  >
                    {text.cancel}
                  </button>
                  <button
                    className="settings-primary"
                    onClick={() => {
                      void getCogent()?.approveCommandReview(pendingCommandReview.reviewId);
                      handleCloseCommandReview();
                    }}
                  >
                    {text.commandReviewApprove}
                  </button>
                </div>
              </section>
            </>,
            document.body,
          )
        : null}
      {contextUsageTooltip}
    </div>
  );
}

