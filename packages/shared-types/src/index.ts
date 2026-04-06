export type AgentMode = "auto" | "backend" | "frontend";
export type Thoroughness = "light" | "balanced" | "deep";

export type ModelTier = "flash-lite" | "flash" | "pro";
export type ModelProvider = "gemini" | "ollama" | "lmstudio";

export type GeminiModelSummary = {
  id: string;
  name: string;
  description?: string;
  inputTokenLimit?: number | null;
  outputTokenLimit?: number | null;
  supportsImages?: boolean;
};

export type OllamaModelSummary = {
  name: string;
  model: string;
  size: number;
  modifiedAt: string;
  parameterSize?: string;
  family?: string;
  contextLength?: number | null;
};

export type LmStudioModelSummary = {
  id: string;
  name: string;
  ownedBy?: string;
  contextLength?: number | null;
};

export type ModeDecision = {
  mode: AgentMode;
  source: "user" | "auto";
  confidence: number;
  reasons: string[];
};

export type ConversationTurn = {
  role: "user" | "assistant";
  text: string;
  images?: Array<{
    id: string;
    name: string;
    mimeType: string;
    dataUrl: string;
  }>;
  files?: Array<{
    id: string;
    name: string;
    mimeType: string;
    content: string;
  }>;
};

export type TaskRequest = {
  prompt: string;
  globalSystemPrompt?: string;
  ollamaContextLength?: number;
  lmStudioContextLength?: number;
  currentPromptImages?: Array<{
    id: string;
    name: string;
    mimeType: string;
    dataUrl: string;
  }>;
  currentPromptFiles?: Array<{
    id: string;
    name: string;
    mimeType: string;
    content: string;
  }>;
  activeFile?: string;
  selectedText?: string;
  openFiles: string[];
  explicitMode?: Exclude<AgentMode, "auto">;
  thoroughness?: Thoroughness;
  modelTier: ModelTier;
  modelProvider?: ModelProvider;
  modelId?: string;
  liveApply: boolean;
  currentCode: string;
  apiKey?: string;
  conversation?: ConversationTurn[];
  workspaceRoot?: string;
};

export type RetrievalSnippet = {
  path: string;
  content: string;
  score: number;
  reason: string;
};

export type RetrievalBundle = {
  snippets: RetrievalSnippet[];
  memorySummary: string;
  compressionState: "healthy" | "watch" | "compressing";
};

export type CommandChunk = {
  stream: "stdout" | "stderr" | "system";
  text: string;
};

export type BrowserAssistRequest = {
  requestId: string;
  url: string;
  title: string;
  description: string;
  helpNeeded: string;
  steps: string[];
};

export type BrowserAssistResult = {
  requestId: string;
  finalUrl: string;
  title: string;
  content: string;
  links: Array<{ text: string; url: string }>;
  activityType: string;
};

export type AgentStreamEvent =
  | { type: "mode"; decision: ModeDecision }
  | { type: "status"; label: string }
  | { type: "browser-assist-request"; request: BrowserAssistRequest }
  | { type: "retrieval"; bundle: RetrievalBundle }
  | { type: "message"; chunk: string }
  | { type: "code"; chunk: string }
  | { type: "file-read"; path: string; startLine: number; endLine: number }
  | { type: "file-write-start"; path: string }
  | { type: "file-write"; path: string; content: string; originalContent: string }
  | { type: "file-delete"; path: string }
  | { type: "file-move"; fromPath: string; toPath: string }
  | { type: "directory-delete"; path: string }
  | { type: "command-review-request"; reviewId: string; command: string; cwd?: string }
  | { type: "command-start"; commandId: string; command: string }
  | { type: "command"; commandId: string; chunk: CommandChunk }
  | { type: "command-end"; commandId: string; exitCode: number | null }
  | { type: "done"; summary: string };

export type RunCommandRequest = {
  command: string;
  cwd?: string;
};

export type ContextSnapshot = {
  modeDecision: ModeDecision;
  retrieval: RetrievalBundle;
  modelTier: ModelTier;
  modelProvider: ModelProvider;
  model: string;
};

export type ContextUsageSnapshot = {
  model: string;
  usedTokens: number;
  inputTokenLimit: number;
  usagePercent: number;
  compressionState: RetrievalBundle["compressionState"];
  snippetCount: number;
};
