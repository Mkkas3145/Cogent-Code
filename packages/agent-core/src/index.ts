import type {
  AgentMode,
  AgentStreamEvent,
  BrowserAssistRequest,
  BrowserAssistResult,
  CommandChunk,
  ContextSnapshot,
  ContextUsageSnapshot,
  ConversationTurn,
  ModelProvider,
  ModeDecision,
  RetrievalBundle,
  TaskRequest,
} from "@cogent/shared-types";
import { totalmem } from "node:os";

type ToolHandlers = {
  readFile: (targetPath: string) => Promise<{ path: string; content: string }>;
  listDir: (
    targetPath: string,
  ) => Promise<{ path: string; entries: Array<{ name: string; path: string; type: "directory" | "file" }> }>;
  writeFile: (targetPath: string, content: string) => Promise<{ path: string; saved: true; originalContent: string }>;
  createFile: (targetPath: string, content: string) => Promise<{ path: string; saved: true; originalContent: string }>;
  deleteFile: (targetPath: string) => Promise<{ path: string; deleted: true }>;
  deleteDirectory: (targetPath: string) => Promise<{ path: string; deleted: true }>;
  runCommand: (
    command: string,
    cwd?: string,
  ) => AsyncGenerator<CommandChunk, number | null, void>;
  searchWeb: (
    query: string,
  ) => Promise<{ query: string; results: Array<{ title: string; url: string; snippet: string }> }>;
  openWebpage: (
    url: string,
  ) => Promise<{ url: string; finalUrl: string; title: string; content: string; links: Array<{ text: string; url: string }> }>;
  clickWebpage: (
    selector: string,
  ) => Promise<{ selector: string; clicked: true }>;
  scrollWebpage: (
    deltaY: number,
  ) => Promise<{ deltaY: number; scrolled: true }>;
  typeWebpage: (
    selector: string,
    text: string,
    clear?: boolean,
  ) => Promise<{ selector: string; typed: true }>;
  pressWebpageKey: (
    key: string,
  ) => Promise<{ key: string; pressed: true }>;
  dragWebpage: (
    selector: string,
    deltaX: number,
    deltaY: number,
  ) => Promise<{ selector: string; dragged: true }>;
  resizeWebpage: (
    width: number,
    height: number,
  ) => Promise<{ width: number; height: number; resized: true }>;
  screenshotWebpage: () => Promise<{
    finalUrl: string;
    title: string;
    mimeType: "image/png";
    dataUrl: string;
    width: number;
    height: number;
  }>;
  requestBrowserAssist: (request: BrowserAssistRequest) => Promise<BrowserAssistResult>;
};

type GeminiPart =
  | { text?: string }
  | {
      inlineData?: {
        mimeType?: string;
        data?: string;
      };
    }
  | {
      functionCall?: {
        name?: string;
        args?: Record<string, unknown>;
      };
    }
  | {
      functionResponse?: {
        name?: string;
        response?: {
          result?: unknown;
          error?: string;
        };
      };
    };

type GeminiContent = {
  role: "user" | "model";
  parts: GeminiPart[];
};

type GeminiCandidate = {
  content?: {
    parts?: GeminiPart[];
  };
};

type OllamaToolCall = {
  function?: {
    name?: string;
    arguments?: Record<string, unknown> | string;
  };
};

type OllamaMessage = {
  role: "system" | "user" | "assistant" | "tool";
  content: string;
  tool_calls?: OllamaToolCall[];
  tool_name?: string;
};

type OllamaChatResponse = {
  model?: string;
  message?: {
    role?: string;
    content?: string;
    tool_calls?: OllamaToolCall[];
  };
  done?: boolean;
  prompt_eval_count?: number;
};

type OpenAiToolCall = {
  id?: string;
  type?: "function";
  function?: {
    name?: string;
    arguments?: string;
  };
};

type OpenAiChatMessage = {
  role: "system" | "user" | "assistant" | "tool";
  content: string | Array<{ type: "text"; text: string }>;
  tool_calls?: OpenAiToolCall[];
  tool_call_id?: string;
};

type OpenAiChatCompletionResponse = {
  choices?: Array<{
    message?: {
      role?: string;
      content?: string | null;
      tool_calls?: OpenAiToolCall[];
    };
    delta?: {
      content?: string | null;
      tool_calls?: Array<{
        index?: number;
        id?: string;
        type?: "function";
        function?: {
          name?: string;
          arguments?: string;
        };
      }>;
    };
    finish_reason?: string | null;
  }>;
  usage?: {
    prompt_tokens?: number;
    completion_tokens?: number;
    total_tokens?: number;
  };
};

const FILE_TOOLS = [
  {
    name: "read_file",
    description:
      "Read the full UTF-8 text content of a file. Use this before editing or when the user references a specific file path.",
    parameters: {
      type: "OBJECT",
      properties: {
        path: {
          type: "STRING",
          description: "Absolute or workspace-relative file path to read.",
        },
      },
      required: ["path"],
    },
  },
  {
    name: "list_dir",
    description:
      "List files and folders inside a directory. Use this to explore when the user mentions a folder or asks about files inside it.",
    parameters: {
      type: "OBJECT",
      properties: {
        path: {
          type: "STRING",
          description: "Absolute or workspace-relative directory path to inspect.",
        },
      },
      required: ["path"],
    },
  },
  {
    name: "write_file",
    description:
      "Write the complete updated UTF-8 file contents to disk. Use only after reading the target file and preparing the full replacement content.",
    parameters: {
      type: "OBJECT",
      properties: {
        path: {
          type: "STRING",
          description: "Absolute or workspace-relative file path to overwrite.",
        },
        content: {
          type: "STRING",
          description: "Complete updated file contents.",
        },
      },
      required: ["path", "content"],
    },
  },
  {
    name: "create_file",
    description:
      "Create a new UTF-8 file, creating missing parent folders when needed. Use this when the file does not already exist.",
    parameters: {
      type: "OBJECT",
      properties: {
        path: {
          type: "STRING",
          description: "Absolute or workspace-relative file path to create.",
        },
        content: {
          type: "STRING",
          description: "Initial file contents.",
        },
      },
      required: ["path", "content"],
    },
  },
  {
    name: "delete_file",
    description: "Delete an existing file from disk. Use this only for real file removal.",
    parameters: {
      type: "OBJECT",
      properties: {
        path: {
          type: "STRING",
          description: "Absolute or workspace-relative file path to delete.",
        },
      },
      required: ["path"],
    },
  },
  {
    name: "delete_directory",
    description: "Delete an existing empty directory from disk. Use this only when the folder is empty.",
    parameters: {
      type: "OBJECT",
      properties: {
        path: {
          type: "STRING",
          description: "Absolute or workspace-relative directory path to delete.",
        },
      },
      required: ["path"],
    },
  },
  {
    name: "run_command",
    description:
      "Run a shell command in the workspace when inspection, build, search, or verification requires it. Stream output while it runs.",
    parameters: {
      type: "OBJECT",
      properties: {
        command: {
          type: "STRING",
          description: "Shell command to execute.",
        },
        cwd: {
          type: "STRING",
          description: "Optional working directory. Omit to use the current workspace root.",
        },
      },
      required: ["command"],
    },
  },
  {
    name: "web_search",
    description:
      "Search the public web for current information and return a short list of relevant results with titles, links, and snippets.",
    parameters: {
      type: "OBJECT",
      properties: {
        query: {
          type: "STRING",
          description: "Search query to look up on the web.",
        },
      },
      required: ["query"],
    },
  },
  {
    name: "open_webpage",
    description:
      "Open a web page in a real browser context, wait for client-side content to render, and return the visible text and key links.",
    parameters: {
      type: "OBJECT",
      properties: {
        url: {
          type: "STRING",
          description: "Absolute http or https URL to open.",
        },
      },
      required: ["url"],
    },
  },
  {
    name: "web_click",
    description: "Click an element on the currently open web page using a CSS selector.",
    parameters: {
      type: "OBJECT",
      properties: {
        selector: {
          type: "STRING",
          description: "CSS selector for the element to click.",
        },
      },
      required: ["selector"],
    },
  },
  {
    name: "web_scroll",
    description: "Scroll the currently open web page vertically by a number of pixels.",
    parameters: {
      type: "OBJECT",
      properties: {
        deltaY: {
          type: "NUMBER",
          description: "Vertical scroll amount in pixels. Positive scrolls down, negative scrolls up.",
        },
      },
      required: ["deltaY"],
    },
  },
  {
    name: "web_type",
    description: "Focus an input element on the currently open web page and type text into it.",
    parameters: {
      type: "OBJECT",
      properties: {
        selector: {
          type: "STRING",
          description: "CSS selector for the target input, textarea, or editable element.",
        },
        text: {
          type: "STRING",
          description: "Text to type.",
        },
        clear: {
          type: "BOOLEAN",
          description: "Whether to clear the field before typing. Defaults to true.",
        },
      },
      required: ["selector", "text"],
    },
  },
  {
    name: "web_press",
    description: "Press a keyboard key on the currently open web page, such as Enter, Tab, or ArrowDown.",
    parameters: {
      type: "OBJECT",
      properties: {
        key: {
          type: "STRING",
          description: "Key name to press.",
        },
      },
      required: ["key"],
    },
  },
  {
    name: "web_drag",
    description: "Drag from the center of an element on the currently open web page by a pixel offset.",
    parameters: {
      type: "OBJECT",
      properties: {
        selector: {
          type: "STRING",
          description: "CSS selector for the element to drag from.",
        },
        deltaX: {
          type: "NUMBER",
          description: "Horizontal drag offset in pixels.",
        },
        deltaY: {
          type: "NUMBER",
          description: "Vertical drag offset in pixels.",
        },
      },
      required: ["selector", "deltaX", "deltaY"],
    },
  },
  {
    name: "web_screenshot",
    description: "Capture the currently open web page as a PNG screenshot for visual inspection.",
    parameters: {
      type: "OBJECT",
      properties: {},
      required: [],
    },
  },
  {
    name: "web_resize",
    description: "Resize the currently open browser viewport to a specific width and height for responsive inspection.",
    parameters: {
      type: "OBJECT",
      properties: {
        width: {
          type: "NUMBER",
          description: "Viewport width in pixels.",
        },
        height: {
          type: "NUMBER",
          description: "Viewport height in pixels.",
        },
      },
      required: ["width", "height"],
    },
  },
  {
    name: "request_browser_assistance",
    description:
      "Ask the user to help in a live browser when human interaction is required, such as login, captcha, consent, or a visually guided step.",
    parameters: {
      type: "OBJECT",
      properties: {
        url: {
          type: "STRING",
          description: "Absolute http or https URL to open for the user.",
        },
        title: {
          type: "STRING",
          description: "Short header title explaining the browser task.",
        },
        description: {
          type: "STRING",
          description: "Explain why the user's help is needed right now.",
        },
        helpNeeded: {
          type: "STRING",
          description: "Specific outcome the user should reach before the agent continues.",
        },
        steps: {
          type: "ARRAY",
          items: {
            type: "STRING",
          },
          description: "Flat checklist of detailed browser steps for the user.",
        },
      },
      required: ["url", "title", "description", "helpNeeded", "steps"],
    },
  },
  {
    name: "finish_turn",
    description:
      "Explicitly finish the current turn only when you are truly done or clearly blocked. Always include a short user-facing plain-language message.",
    parameters: {
      type: "OBJECT",
      properties: {
        message: {
          type: "STRING",
          description: "Short plain-language body text to show the user before the turn ends.",
        },
        summary: {
          type: "STRING",
          description: "Short internal completion summary.",
        },
      },
      required: ["message"],
    },
  },
];

const OLLAMA_TOOL_DEFINITIONS = FILE_TOOLS.map((tool) => ({
  type: "function",
  function: {
    name: tool.name,
    description: tool.description,
    parameters: {
      type: "object",
      properties: Object.fromEntries(
        Object.entries(tool.parameters.properties).map(([key, value]) => [
          key,
          {
            type: String((value as { type?: string }).type ?? "STRING").toLowerCase(),
            description: (value as { description?: string }).description,
          },
        ]),
      ),
      required: tool.parameters.required,
    },
  },
}));

const OPENAI_TOOL_DEFINITIONS = FILE_TOOLS.map((tool) => ({
  type: "function",
  function: {
    name: tool.name,
    description: tool.description,
    parameters: {
      type: "object",
      properties: Object.fromEntries(
        Object.entries(tool.parameters.properties).map(([key, value]) => [
          key,
          {
            type: String((value as { type?: string }).type ?? "STRING").toLowerCase(),
            description: (value as { description?: string }).description,
          },
        ]),
      ),
      required: tool.parameters.required,
      additionalProperties: false,
    },
  },
}));

function isFilesystemMutationCommand(command: string) {
  const normalized = command.trim().toLowerCase();
  return [
    /\bmkdir\b/,
    /\bmd\b/,
    /\brmdir\b/,
    /\brd\b/,
    /\brm\b/,
    /\bdel\b/,
    /\berase\b/,
    /\bmv\b/,
    /\bmove\b/,
    /\bren\b/,
    /\brename\b/,
    /\bcopy\b/,
    /\bcp\b/,
    /\btouch\b/,
    /\bnew-item\b/,
    /\bremove-item\b/,
    /\bmove-item\b/,
    /\bcopy-item\b/,
    /\brename-item\b/,
    /\bset-content\b/,
    /\badd-content\b/,
    /\bout-file\b/,
  ].some((pattern) => pattern.test(normalized));
}

function getDefaultLocalContextLength(totalMemoryBytes: number) {
  const totalMemoryGb = totalMemoryBytes / 1024 / 1024 / 1024;
  const reservedForSystemGb = Math.max(4, totalMemoryGb * 0.25);
  const usableMemoryGb = Math.max(4, totalMemoryGb - reservedForSystemGb);
  const estimatedContext = Math.floor(usableMemoryGb * 512);
  const clampedContext = Math.max(4096, Math.min(32768, estimatedContext));
  return Math.round(clampedContext / 1024) * 1024;
}

function classifyMode(request: TaskRequest): ModeDecision {
  if (request.explicitMode) {
    return {
      mode: request.explicitMode,
      source: "user",
      confidence: 1,
      reasons: ["User-selected mode"],
    };
  }

  return {
    mode: "auto",
    source: "auto",
    confidence: 1,
    reasons: ["Mode selection deferred to the model"],
  };
}

function buildRetrievalBundle(request: TaskRequest, decision: ModeDecision): RetrievalBundle {
  const snippets = [];

  if (request.activeFile && request.currentCode.trim()) {
    snippets.push({
      path: request.activeFile,
      content: request.currentCode.slice(0, 2400),
      score: 0.98,
      reason: "Active file buffer",
    });
  }

  if (request.selectedText?.trim()) {
    snippets.push({
      path: request.activeFile ?? "selection://active",
      content: request.selectedText.slice(0, 1200),
      score: 0.92,
      reason: "Current selection",
    });
  }

  if (request.openFiles.length > 0) {
    snippets.push({
      path: "session://open-files",
      content: request.openFiles.join("\n"),
      score: 0.6,
      reason: "Currently open files",
    });
  }

  return {
    snippets,
    memorySummary: `Mode=${decision.mode}; workspace=${request.workspaceRoot ?? "unknown"}; conversationTurns=${request.conversation?.length ?? 0}`,
    compressionState: "healthy",
  };
}

function modelForTier(modelTier: TaskRequest["modelTier"]): string {
  switch (modelTier) {
    case "pro":
      return "gemini-3.1-pro-preview";
    case "flash":
      return "gemini-3-flash-preview";
    default:
      return "gemini-3.1-flash-lite-preview";
  }
}

function buildGenerateContentPayload(
  request: TaskRequest,
  decision: ModeDecision,
  retrieval: RetrievalBundle,
  contents: GeminiContent[],
  includeTools: boolean,
) {
  return {
    system_instruction: {
      parts: [{ text: buildSystemInstruction(request, decision, retrieval) }],
    },
    ...(includeTools ? { tools: [{ functionDeclarations: FILE_TOOLS }] } : {}),
    ...(includeTools
      ? {
          toolConfig: {
            functionCallingConfig: {
              mode: "AUTO",
              allowedFunctionNames: FILE_TOOLS.map((tool) => tool.name),
            },
          },
        }
      : {}),
    contents,
  };
}

function buildSystemInstruction(request: TaskRequest, decision: ModeDecision, retrieval: RetrievalBundle): string {
  const instructions = [
    "You are Cogent, an AI coding agent inside a desktop coding workspace.",
    "Behave like a tool-using coding agent, not a one-shot assistant.",
    "When you need file contents, directory structure, or to save a change, call the provided tools yourself instead of asking for permission.",
    "Use run_command only for shell-native tasks such as build, test, search, git, package-manager, or environment inspection.",
    "Do not use run_command for normal file editing, file creation, file deletion, or empty-folder deletion when a dedicated file tool exists.",
    "Prefer read_file, list_dir, write_file, create_file, delete_file, and delete_directory over shell commands whenever they can accomplish the task.",
    "Use web_search to find current information on the public web and open_webpage when you need the actual rendered contents of a page, including client-side content.",
    "After opening a page, you may use web_click, web_scroll, web_type, web_press, and web_drag to interact with the page when needed.",
    "If a browser task requires a real person, such as login, captcha, consent, or visually guided confirmation, use request_browser_assistance with a clear title, reason, helpNeeded summary, and detailed ordered steps.",
    "Browser assistance results are only candidate observations. You must decide yourself whether the browser task is truly complete from the returned page snapshot.",
    "Only treat a browser-assisted task as complete when you are highly confident from the actual screen contents that the requested goal was really reached. If there is any real doubt, do not say it is done yet.",
    "When deciding whether a browser-assisted task is complete, analyze the actual returned browser page content, title, URL, and visible clues on the screen. Do not decide based only on the fact that the panel closed or that some interaction happened.",
    "If the browser snapshot is not sufficient yet, do not pretend the task is done. Explain what is still missing and request browser assistance again with updated guidance.",
    "Once the browser assistance panel closes, the user can no longer see or interact with that browser state. If you still need the user to inspect or do something there, you must ask for browser assistance again.",
    "Do not pretend that a file was read or written unless you actually called the tool and received the result.",
    "Never claim that work is finished, fixed, saved, created, or deleted unless the relevant tool actually succeeded.",
    "If a command exits non-zero or any tool returns an error, do not present the task as complete. Explain that it failed and what remains blocked.",
    "Before claiming that a file or folder changed state, rely on the tool result, and if there is any doubt, verify by reading the file again or listing the directory again.",
    "If a verification step fails or is impossible, say that clearly instead of implying success.",
    "For any meaningful edit, creation, deletion, command run, or browser task, perform an explicit verification step before saying it worked whenever verification is reasonably possible.",
    "Do not stop at the first successful tool call. If the user asked for a change, verify the resulting state and only then describe it as done.",
    "Do not end the task at an ambiguous halfway point. Continue until the requested work is actually completed, clearly blocked, or explicitly handed back to the user for a specific reason.",
    "A successful tool call is not the finish line. Finishing requires the underlying user request to be satisfied, not merely a partial action to have happened.",
    "Before you finish a turn, do a final internal checklist: what did the user ask for, what actions succeeded, what still needs verification, and is anything still unfinished or uncertain.",
    "If anything important remains uncertain, incomplete, unverified, or possibly broken, do not end with a done tone. Keep working or explain the exact blocker.",
    "When you edited code or changed files, prefer to verify by re-reading the changed file, listing the affected directory, or running an appropriate validation command when practical.",
    "When you ran a command for build, test, install, search, or verification, inspect the result and decide what to do next instead of stopping immediately after the command completes.",
    "If a tool fails because of a wrong path, missing file, missing directory, or similar mistake, treat that as a cue to recover and keep going rather than silently ending the attempt.",
    "During longer work, leave short conversational progress updates in the assistant body at meaningful milestones instead of staying silent until the very end.",
    "If you performed several tool actions in a row, summarize the current state in plain language before moving on or finishing.",
    "After several consecutive tool calls, pause and write a brief body update before continuing so the user is not left with a silent tool-only sequence.",
    "Empty folders may be removed with delete_directory. Do not use it on non-empty folders.",
    "You may chat casually between edits. Natural conversation is allowed even in the middle of code work.",
    "Use tools in multiple steps when needed: inspect, reason, edit, verify mentally, then respond.",
    "When making a real code change, use write_file, create_file, delete_file, or delete_directory instead of shelling out or pasting code into the chat.",
    "Do not paste large code blocks or full files in your conversational reply unless the user explicitly asks to see the code.",
    "After tool work, explain clearly what you changed or what you found in plain language.",
    "You must decide when the turn is actually complete. Do not rely on the runtime to assume completion from silence.",
    "Only finish a turn by calling finish_turn with a short user-facing message once you are truly done or clearly blocked.",
    "Do not call finish_turn at an ambiguous midpoint, after only partial work, or before verification that should reasonably happen.",
    "Always leave at least one conversational body line before finishing a turn, even when the task is blocked, handed off to the user, or waiting for browser assistance.",
    "Do not end a turn with only tool output, status changes, or UI actions. Leave a short plain-language message that explains the current state and what comes next.",
    "This applies to every turn without exception: never finish with zero assistant body text.",
    decision.mode === "auto"
      ? "Preferred mode: auto. Decide the best workflow yourself from the user's request and tool results."
      : `Preferred mode: ${decision.mode}.`,
    `Live apply requested: ${request.liveApply ? "yes" : "no"}.`,
    request.activeFile ? `Active file: ${request.activeFile}` : "No active file is open.",
    request.workspaceRoot ? `Current workspace root: ${request.workspaceRoot}` : "No workspace root reported.",
    request.openFiles.length > 0 ? `Open files:\n${request.openFiles.join("\n")}` : "No open files reported.",
    retrieval.snippets.length > 0
      ? `Relevant context:\n${retrieval.snippets
          .map((snippet) => `Path: ${snippet.path}\nReason: ${snippet.reason}\nContent:\n${snippet.content}`)
          .join("\n\n")}`
      : "No code snippets were attached.",
  ];

  const globalSystemPrompt = request.globalSystemPrompt?.trim();
  if (globalSystemPrompt) {
    instructions.push(`Additional global system prompt:\n${globalSystemPrompt}`);
  }

  return instructions.join("\n\n");
}

function buildContents(
  history: ConversationTurn[],
  prompt: string,
  promptImages?: Array<{ mimeType: string; dataUrl: string }>,
  promptFiles?: Array<{ name: string; mimeType: string; content: string }>,
): GeminiContent[] {
  const imagePartsFor = (
    images: Array<{ mimeType: string; dataUrl: string }> | undefined,
  ): GeminiPart[] =>
    (images ?? []).flatMap((image) => {
      const match = image.dataUrl.match(/^data:([^;]+);base64,(.+)$/);
      if (!match) {
        return [];
      }

      return [
        {
          inlineData: {
            mimeType: image.mimeType || match[1],
            data: match[2],
          },
        },
      ];
    });

  const filePartsFor = (
    files: Array<{ name: string; mimeType: string; content: string }> | undefined,
  ): GeminiPart[] =>
    (files ?? [])
      .filter((file) => file.content.trim().length > 0)
      .map((file) => ({
        text: `Attached file: ${file.name}\nMIME type: ${file.mimeType}\n\n\`\`\`\n${file.content}\n\`\`\``,
      }));

  return [
    ...history
      .filter((turn) => turn.text.trim().length > 0 || (turn.images?.length ?? 0) > 0 || (turn.files?.length ?? 0) > 0)
      .map((turn) => ({
        role: (turn.role === "assistant" ? "model" : "user") as "model" | "user",
        parts: [
          { text: turn.text },
          ...imagePartsFor(turn.role === "user" ? turn.images : undefined),
          ...filePartsFor(turn.role === "user" ? turn.files : undefined),
        ],
      })),
    {
      role: "user",
      parts: [{ text: prompt }, ...imagePartsFor(promptImages), ...filePartsFor(promptFiles)],
    },
  ];
}

function splitResponseParts(text: string): { message: string; code: string } {
  const fenceStart = text.indexOf("```");

  if (fenceStart === -1) {
    const normalizedMessage = text
      .replace(/^[\uFFFD?]+\s*/u, "")
      .replace(/\r\n/g, "\n")
      .replace(/\n{3,}/g, "\n\n")
      .trim();

    return {
      message: normalizedMessage,
      code: "",
    };
  }

  const beforeFence = text.slice(0, fenceStart);
  const afterFence = text.slice(fenceStart + 3);
  const languageLineEnd = afterFence.indexOf("\n");
  const codeStart = languageLineEnd === -1 ? afterFence.length : languageLineEnd + 1;
  const afterLanguage = afterFence.slice(codeStart);
  const fenceEnd = afterLanguage.indexOf("```");
  const codeSection = fenceEnd === -1 ? afterLanguage : afterLanguage.slice(0, fenceEnd);
  const afterCode = fenceEnd === -1 ? "" : afterLanguage.slice(fenceEnd + 3);

  const normalizedMessage = `${beforeFence}\n${afterCode}`
    .replace(/^[\uFFFD?]+\s*/u, "")
    .replace(/\r\n/g, "\n")
    .replace(/\n{3,}/g, "\n\n")
    .trim();

  return {
    message: normalizedMessage,
    code: codeSection.trim(),
  };
}

function getTextFromParts(parts: GeminiPart[]): string {
  return parts
    .map((part) => ("text" in part ? part.text ?? "" : ""))
    .join("")
    .replace(/\r\n/g, "\n");
}

function getFunctionCalls(parts: GeminiPart[]): Array<{ name: string; args: Record<string, unknown> }> {
  return parts.flatMap((part) => {
    if (!("functionCall" in part) || !part.functionCall?.name) {
      return [];
    }

    return [
      {
        name: part.functionCall.name,
        args: (part.functionCall.args as Record<string, unknown> | undefined) ?? {},
      },
    ];
  });
}

async function requestGemini(
  request: TaskRequest,
  decision: ModeDecision,
  retrieval: RetrievalBundle,
  contents: GeminiContent[],
  signal?: AbortSignal,
): Promise<GeminiCandidate> {
  const response = await fetch(
    `https://generativelanguage.googleapis.com/v1beta/models/${getEffectiveModelName(request)}:generateContent`,
    {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "x-goog-api-key": request.apiKey!.trim(),
      },
      signal,
      body: JSON.stringify(buildGenerateContentPayload(request, decision, retrieval, contents, true)),
    },
  );

  if (!response.ok) {
    const body = await response.text();
    throw new Error(`Gemini request failed: ${response.status} ${body}`);
  }

  const parsed = (await response.json()) as {
    candidates?: GeminiCandidate[];
  };

  return parsed.candidates?.[0] ?? {};
}

async function sleep(ms: number, signal?: AbortSignal) {
  await new Promise<void>((resolve, reject) => {
    const timeout = setTimeout(() => {
      signal?.removeEventListener("abort", handleAbort);
      resolve();
    }, ms);

    const handleAbort = () => {
      clearTimeout(timeout);
      signal?.removeEventListener("abort", handleAbort);
      reject(new DOMException("Aborted", "AbortError"));
    };

    if (signal) {
      signal.addEventListener("abort", handleAbort, { once: true });
    }
  });
}

async function requestGeminiWithRetry(
  request: TaskRequest,
  decision: ModeDecision,
  retrieval: RetrievalBundle,
  contents: GeminiContent[],
  signal?: AbortSignal,
): Promise<GeminiCandidate> {
  const maxAttempts = 3;
  let lastError: unknown;

  for (let attempt = 1; attempt <= maxAttempts; attempt += 1) {
    try {
      return await requestGemini(request, decision, retrieval, contents, signal);
    } catch (error) {
      lastError = error;
      if (signal?.aborted || (error instanceof Error && error.name === "AbortError") || attempt >= maxAttempts) {
        throw error;
      }

      await sleep(700 * attempt, signal);
    }
  }

  throw lastError instanceof Error ? lastError : new Error("Gemini request failed.");
}

async function requestOllamaChat(
  request: TaskRequest,
  messages: OllamaMessage[],
  includeTools: boolean,
  signal?: AbortSignal,
): Promise<OllamaChatResponse> {
  const response = await fetch("http://127.0.0.1:11434/api/chat", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    signal,
    body: JSON.stringify({
      model: getEffectiveModelName(request),
      stream: false,
      messages,
      options: (request.ollamaContextLength || getDefaultLocalContextLength(totalmem()))
        ? {
            num_ctx: request.ollamaContextLength || getDefaultLocalContextLength(totalmem()),
          }
        : undefined,
      ...(includeTools ? { tools: OLLAMA_TOOL_DEFINITIONS } : {}),
    }),
  });

  if (!response.ok) {
    const body = await response.text();
    throw new Error(`Ollama chat failed: ${response.status} ${body}`);
  }

  return (await response.json()) as OllamaChatResponse;
}

function buildOpenAiMessages(
  request: TaskRequest,
  decision: ModeDecision,
  retrieval: RetrievalBundle,
  prompt: string,
  history: ConversationTurn[],
): OpenAiChatMessage[] {
  const fileTextFor = (files: ConversationTurn["files"] | TaskRequest["currentPromptFiles"]) =>
    (files ?? [])
      .filter((file) => file.content.trim().length > 0)
      .map((file) => `Attached file: ${file.name}\nMIME type: ${file.mimeType}\n\n\`\`\`\n${file.content}\n\`\`\``)
      .join("\n\n");

  const imageTextFor = (images: ConversationTurn["images"] | TaskRequest["currentPromptImages"]) =>
    (images ?? [])
      .map((image) => `Attached image: ${image.name} (${image.mimeType})`)
      .join("\n");

  return [
    {
      role: "system",
      content: buildSystemInstruction(request, decision, retrieval),
    },
    ...history
      .filter((turn) => turn.text.trim().length > 0 || (turn.files?.length ?? 0) > 0 || (turn.images?.length ?? 0) > 0)
      .map((turn) => ({
        role: turn.role,
        content: [turn.text, imageTextFor(turn.images), fileTextFor(turn.files)].filter(Boolean).join("\n\n"),
      })),
    {
      role: "user",
      content: [prompt, imageTextFor(request.currentPromptImages), fileTextFor(request.currentPromptFiles)].filter(Boolean).join("\n\n"),
    },
  ];
}

async function requestLmStudioChat(
  request: TaskRequest,
  messages: OpenAiChatMessage[],
  includeTools: boolean,
  signal?: AbortSignal,
): Promise<OpenAiChatCompletionResponse> {
  const response = await fetch("http://127.0.0.1:1234/v1/chat/completions", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    signal,
    body: JSON.stringify({
      model: getEffectiveModelName(request),
      stream: false,
      messages,
      ...(includeTools ? { tools: OPENAI_TOOL_DEFINITIONS, tool_choice: "auto" } : {}),
    }),
  });

  if (!response.ok) {
    const body = await response.text();
    throw new Error(`LM Studio chat failed: ${response.status} ${body}`);
  }

  return (await response.json()) as OpenAiChatCompletionResponse;
}

function parseOpenAiToolCalls(
  response: OpenAiChatCompletionResponse,
): Array<{ id?: string; name: string; args: Record<string, unknown> }> {
  return (response.choices?.[0]?.message?.tool_calls ?? []).flatMap((toolCall) => {
    const name = toolCall.function?.name;
    if (!name) {
      return [];
    }

    const rawArguments = toolCall.function?.arguments;
    if (typeof rawArguments !== "string" || !rawArguments.trim()) {
      return [{ id: toolCall.id, name, args: {} }];
    }

    try {
      return [{ id: toolCall.id, name, args: JSON.parse(rawArguments) as Record<string, unknown> }];
    } catch {
      return [{ id: toolCall.id, name, args: {} }];
    }
  });
}

async function* requestGeminiStream(
  request: TaskRequest,
  decision: ModeDecision,
  retrieval: RetrievalBundle,
  contents: GeminiContent[],
  includeTools = false,
  signal?: AbortSignal,
): AsyncGenerator<{ type: "text"; delta: string } | { type: "candidate"; candidate: GeminiCandidate }> {
  const response = await fetch(
    `https://generativelanguage.googleapis.com/v1beta/models/${getEffectiveModelName(request)}:streamGenerateContent?alt=sse`,
    {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "x-goog-api-key": request.apiKey!.trim(),
      },
      signal,
      body: JSON.stringify(buildGenerateContentPayload(request, decision, retrieval, contents, includeTools)),
    },
  );

  if (!response.ok) {
    const body = await response.text();
    throw new Error(`Gemini stream request failed: ${response.status} ${body}`);
  }

  if (!response.body) {
    const fallbackCandidate = await requestGeminiWithRetry(request, decision, retrieval, contents, signal);
    const fallbackText = getTextFromParts(fallbackCandidate.content?.parts ?? []);
    if (fallbackText) {
      yield { type: "text", delta: fallbackText };
    }
    yield { type: "candidate", candidate: fallbackCandidate };
    return;
  }

  const decoder = new TextDecoder();
  const reader = response.body.getReader();
  let buffer = "";
  let pendingData = "";
  let streamedText = "";
  let latestCandidate: GeminiCandidate = {};

  const flushEvent = async function* () {
    const raw = pendingData.trim();
    pendingData = "";

    if (!raw || raw === "[DONE]") {
      return;
    }

    const parsed = JSON.parse(raw) as { candidates?: GeminiCandidate[] };
    const candidate = parsed.candidates?.[0];

    if (!candidate) {
      return;
    }

    latestCandidate = candidate;
    const nextText = getTextFromParts(candidate.content?.parts ?? []);
    if (nextText.startsWith(streamedText)) {
      const delta = nextText.slice(streamedText.length);
      if (delta) {
        streamedText = nextText;
        yield { type: "text" as const, delta };
      }
      return;
    }

    if (nextText && nextText !== streamedText) {
      yield { type: "text" as const, delta: nextText };
      streamedText += nextText;
    }
  };

  while (true) {
    const { done, value } = await reader.read();
    if (done) {
      break;
    }

    buffer += decoder.decode(value, { stream: true });
    const events = buffer.split(/\r?\n\r?\n/);
    buffer = events.pop() ?? "";

    for (const eventBlock of events) {
      const lines = eventBlock.split(/\r?\n/);
      for (const line of lines) {
        if (line.startsWith("data:")) {
          pendingData += `${line.slice(5).trim()}\n`;
        }
      }
      yield* flushEvent();
    }
  }

  if (buffer.trim()) {
    const lines = buffer.split(/\r?\n/);
    for (const line of lines) {
      if (line.startsWith("data:")) {
        pendingData += `${line.slice(5).trim()}\n`;
      }
    }
    yield* flushEvent();
  }

  yield { type: "candidate", candidate: latestCandidate };
}

async function* requestGeminiStreamWithRetry(
  request: TaskRequest,
  decision: ModeDecision,
  retrieval: RetrievalBundle,
  contents: GeminiContent[],
  includeTools = false,
  signal?: AbortSignal,
): AsyncGenerator<{ type: "text"; delta: string } | { type: "candidate"; candidate: GeminiCandidate }> {
  const maxAttempts = 3;
  let lastError: unknown;

  for (let attempt = 1; attempt <= maxAttempts; attempt += 1) {
    try {
      yield* requestGeminiStream(request, decision, retrieval, contents, includeTools, signal);
      return;
    } catch (error) {
      lastError = error;
      if (signal?.aborted || (error instanceof Error && error.name === "AbortError") || attempt >= maxAttempts) {
        throw error;
      }

      await sleep(700 * attempt, signal);
    }
  }

  throw lastError instanceof Error ? lastError : new Error("Gemini stream request failed.");
}

async function* requestOllamaStream(
  request: TaskRequest,
  messages: OllamaMessage[],
  includeTools: boolean,
  signal?: AbortSignal,
): AsyncGenerator<{ type: "text"; delta: string } | { type: "response"; response: OllamaChatResponse }> {
  const response = await fetch("http://127.0.0.1:11434/api/chat", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    signal,
    body: JSON.stringify({
      model: getEffectiveModelName(request),
      stream: true,
      messages,
      options: (request.ollamaContextLength || getDefaultLocalContextLength(totalmem()))
        ? {
            num_ctx: request.ollamaContextLength || getDefaultLocalContextLength(totalmem()),
          }
        : undefined,
      ...(includeTools ? { tools: OLLAMA_TOOL_DEFINITIONS } : {}),
    }),
  });

  if (!response.ok) {
    const body = await response.text();
    throw new Error(`Ollama stream failed: ${response.status} ${body}`);
  }

  if (!response.body) {
    const fallback = await requestOllamaChat(request, messages, includeTools, signal);
    if (fallback.message?.content) {
      yield { type: "text", delta: fallback.message.content };
    }
    yield { type: "response", response: fallback };
    return;
  }

  const decoder = new TextDecoder();
  const reader = response.body.getReader();
  let buffer = "";
  let latestResponse: OllamaChatResponse = {};

  while (true) {
    const { done, value } = await reader.read();
    if (done) {
      break;
    }

    buffer += decoder.decode(value, { stream: true });
    const lines = buffer.split(/\r?\n/);
    buffer = lines.pop() ?? "";

    for (const line of lines) {
      const trimmed = line.trim();
      if (!trimmed) {
        continue;
      }

      const parsed = JSON.parse(trimmed) as OllamaChatResponse;
      latestResponse = parsed;
      const delta = parsed.message?.content ?? "";
      if (delta) {
        yield { type: "text", delta };
      }
    }
  }

  if (buffer.trim()) {
    latestResponse = JSON.parse(buffer.trim()) as OllamaChatResponse;
    const delta = latestResponse.message?.content ?? "";
    if (delta) {
      yield { type: "text", delta };
    }
  }

  yield { type: "response", response: latestResponse };
}

async function* executeToolCall(
  call: { name: string; args: Record<string, unknown> },
  handlers: ToolHandlers,
): AsyncGenerator<AgentStreamEvent, GeminiPart> {
  const pathArg = typeof call.args.path === "string" ? call.args.path : "";

  if (call.name === "read_file") {
    yield { type: "status", label: `Exploring ${pathArg}` };
    const result = await handlers.readFile(pathArg);
    const lineCount = result.content.length === 0 ? 1 : result.content.split(/\r?\n/).length;
    yield { type: "file-read", path: result.path, startLine: 1, endLine: lineCount };
    return {
      functionResponse: {
        name: call.name,
        response: {
          result,
        },
      },
    };
  }

  if (call.name === "list_dir") {
    yield { type: "status", label: `Exploring ${pathArg}` };
    const result = await handlers.listDir(pathArg);
    return {
      functionResponse: {
        name: call.name,
        response: {
          result,
        },
      },
    };
  }

  if (call.name === "write_file") {
    const contentArg = typeof call.args.content === "string" ? call.args.content : "";
    yield { type: "file-write-start", path: pathArg };
    yield { type: "status", label: `Writing ${pathArg}` };
    const result = await handlers.writeFile(pathArg, contentArg);
    yield { type: "file-write", path: pathArg, content: contentArg, originalContent: result.originalContent };
    return {
      functionResponse: {
        name: call.name,
        response: {
          result,
        },
      },
    };
  }

  if (call.name === "create_file") {
    const contentArg = typeof call.args.content === "string" ? call.args.content : "";
    yield { type: "file-write-start", path: pathArg };
    yield { type: "status", label: `Creating ${pathArg}` };
    const result = await handlers.createFile(pathArg, contentArg);
    yield { type: "file-write", path: pathArg, content: contentArg, originalContent: result.originalContent };
    return {
      functionResponse: {
        name: call.name,
        response: {
          result,
        },
      },
    };
  }

  if (call.name === "delete_file") {
    yield { type: "status", label: `Deleting ${pathArg}` };
    const result = await handlers.deleteFile(pathArg);
    yield { type: "file-delete", path: result.path };
    return {
      functionResponse: {
        name: call.name,
        response: {
          result,
        },
      },
    };
  }

  if (call.name === "delete_directory") {
    yield { type: "status", label: `Deleting ${pathArg}` };
    const result = await handlers.deleteDirectory(pathArg);
    yield { type: "directory-delete", path: result.path };
    return {
      functionResponse: {
        name: call.name,
        response: {
          result,
        },
      },
    };
  }

  if (call.name === "run_command") {
    const commandArg = typeof call.args.command === "string" ? call.args.command : "";
    const cwdArg = typeof call.args.cwd === "string" ? call.args.cwd : undefined;
    if (isFilesystemMutationCommand(commandArg)) {
      return {
        functionResponse: {
          name: call.name,
          response: {
            error:
              "run_command cannot be used for file or folder creation, editing, moving, renaming, copying, or deletion. Use create_file, write_file, delete_file, or delete_directory instead.",
          },
        },
      };
    }
    const commandId = `command-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`;
    yield { type: "command-start", commandId, command: commandArg };
    const iterator = handlers.runCommand(commandArg, cwdArg)[Symbol.asyncIterator]();
    let exitCode: number | null = null;

    while (true) {
      const next = await iterator.next();
      if (next.done) {
        exitCode = next.value ?? null;
        break;
      }

      yield { type: "command", commandId, chunk: next.value };
    }

    yield { type: "command-end", commandId, exitCode };
    return {
      functionResponse: {
        name: call.name,
        response: {
          ...(exitCode && exitCode !== 0
            ? {
                error: `Command failed with exit code ${exitCode}. Inspect the streamed command output and fix the issue before continuing.`,
              }
            : {
                result: {
                  command: commandArg,
                  cwd: cwdArg,
                  exitCode,
                },
              }),
        },
      },
    };
  }

  if (call.name === "web_search") {
    const queryArg = typeof call.args.query === "string" ? call.args.query : "";
    yield { type: "status", label: `Searching the web for ${queryArg}` };
    const result = await handlers.searchWeb(queryArg);
    return {
      functionResponse: {
        name: call.name,
        response: {
          result,
        },
      },
    };
  }

  if (call.name === "open_webpage") {
    const urlArg = typeof call.args.url === "string" ? call.args.url : "";
    yield { type: "status", label: `Opening ${urlArg}` };
    const result = await handlers.openWebpage(urlArg);
    return {
      functionResponse: {
        name: call.name,
        response: {
          result,
        },
      },
    };
  }

  if (call.name === "web_click") {
    const selectorArg = typeof call.args.selector === "string" ? call.args.selector : "";
    yield { type: "status", label: `Clicking ${selectorArg}` };
    const result = await handlers.clickWebpage(selectorArg);
    return {
      functionResponse: {
        name: call.name,
        response: {
          result,
        },
      },
    };
  }

  if (call.name === "web_scroll") {
    const deltaYArg = typeof call.args.deltaY === "number" ? call.args.deltaY : 0;
    yield { type: "status", label: `Scrolling page by ${deltaYArg}px` };
    const result = await handlers.scrollWebpage(deltaYArg);
    return {
      functionResponse: {
        name: call.name,
        response: {
          result,
        },
      },
    };
  }

  if (call.name === "web_type") {
    const selectorArg = typeof call.args.selector === "string" ? call.args.selector : "";
    const textArg = typeof call.args.text === "string" ? call.args.text : "";
    const clearArg = typeof call.args.clear === "boolean" ? call.args.clear : true;
    yield { type: "status", label: `Typing into ${selectorArg}` };
    const result = await handlers.typeWebpage(selectorArg, textArg, clearArg);
    return {
      functionResponse: {
        name: call.name,
        response: {
          result,
        },
      },
    };
  }

  if (call.name === "web_press") {
    const keyArg = typeof call.args.key === "string" ? call.args.key : "";
    yield { type: "status", label: `Pressing ${keyArg}` };
    const result = await handlers.pressWebpageKey(keyArg);
    return {
      functionResponse: {
        name: call.name,
        response: {
          result,
        },
      },
    };
  }

  if (call.name === "web_drag") {
    const selectorArg = typeof call.args.selector === "string" ? call.args.selector : "";
    const deltaXArg = typeof call.args.deltaX === "number" ? call.args.deltaX : 0;
    const deltaYArg = typeof call.args.deltaY === "number" ? call.args.deltaY : 0;
    yield { type: "status", label: `Dragging ${selectorArg}` };
    const result = await handlers.dragWebpage(selectorArg, deltaXArg, deltaYArg);
    return {
      functionResponse: {
        name: call.name,
        response: {
          result,
        },
      },
    };
  }

  if (call.name === "web_screenshot") {
    yield { type: "status", label: "Capturing browser screenshot" };
    const result = await handlers.screenshotWebpage();
    return {
      functionResponse: {
        name: call.name,
        response: {
          result,
        },
      },
    };
  }

  if (call.name === "web_resize") {
    const widthArg = typeof call.args.width === "number" ? call.args.width : 0;
    const heightArg = typeof call.args.height === "number" ? call.args.height : 0;
    yield { type: "status", label: `Resizing browser to ${widthArg}x${heightArg}` };
    const result = await handlers.resizeWebpage(widthArg, heightArg);
    return {
      functionResponse: {
        name: call.name,
        response: {
          result,
        },
      },
    };
  }

  if (call.name === "request_browser_assistance") {
    const urlArg = typeof call.args.url === "string" ? call.args.url : "";
    const titleArg = typeof call.args.title === "string" ? call.args.title : "Browser help needed";
    const descriptionArg = typeof call.args.description === "string" ? call.args.description : "";
    const helpNeededArg = typeof call.args.helpNeeded === "string" ? call.args.helpNeeded : "";
    const stepsArg = Array.isArray(call.args.steps) ? call.args.steps.filter((step): step is string => typeof step === "string") : [];
    const requestId = `browser-assist-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`;
    const assistRequest: BrowserAssistRequest = {
      requestId,
      url: urlArg,
      title: titleArg,
      description: descriptionArg,
      helpNeeded: helpNeededArg,
      steps: stepsArg,
    };
    yield { type: "status", label: `Waiting for browser help: ${titleArg}` };
    yield { type: "browser-assist-request", request: assistRequest };
    const result = await handlers.requestBrowserAssist(assistRequest);
    return {
      functionResponse: {
        name: call.name,
        response: {
          result,
        },
      },
    };
  }

  return {
    functionResponse: {
      name: call.name,
      response: {
        error: `Unknown tool: ${call.name}`,
      },
    },
  };
}

function parseOllamaToolCalls(message: OllamaChatResponse["message"]): Array<{ name: string; args: Record<string, unknown> }> {
  return (message?.tool_calls ?? []).flatMap((toolCall) => {
    const name = toolCall.function?.name;
    if (!name) {
      return [];
    }

    const rawArguments = toolCall.function?.arguments;
    if (rawArguments && typeof rawArguments === "object" && !Array.isArray(rawArguments)) {
      return [{ name, args: rawArguments }];
    }

    if (typeof rawArguments === "string") {
      try {
        return [{ name, args: JSON.parse(rawArguments) as Record<string, unknown> }];
      } catch {
        return [{ name, args: {} }];
      }
    }

    return [{ name, args: {} }];
  });
}

function getFinishTurnPayload(args: Record<string, unknown>) {
  const message = typeof args.message === "string" ? args.message.trim() : "";
  const summary = typeof args.summary === "string" ? args.summary.trim() : "";
  return { message, summary };
}

async function waitUntilAborted(signal?: AbortSignal) {
  if (!signal) {
    await new Promise<never>(() => {});
    return;
  }

  if (signal.aborted) {
    return;
  }

  await new Promise<void>((resolve) => {
    const handleAbort = () => {
      signal.removeEventListener("abort", handleAbort);
      resolve();
    };

    signal.addEventListener("abort", handleAbort, { once: true });
  });
}

function buildOllamaMessages(
  request: TaskRequest,
  decision: ModeDecision,
  retrieval: RetrievalBundle,
  prompt: string,
  history: ConversationTurn[],
): OllamaMessage[] {
  const fileTextFor = (files: ConversationTurn["files"] | TaskRequest["currentPromptFiles"]) =>
    (files ?? [])
      .filter((file) => file.content.trim().length > 0)
      .map((file) => `Attached file: ${file.name}\nMIME type: ${file.mimeType}\n\n\`\`\`\n${file.content}\n\`\`\``)
      .join("\n\n");

  return [
    {
      role: "system",
      content: buildSystemInstruction(request, decision, retrieval),
    },
    ...history
      .filter((turn) => turn.text.trim().length > 0 || (turn.files?.length ?? 0) > 0)
      .map((turn) => ({
        role: turn.role,
        content: [turn.text, fileTextFor(turn.files)].filter(Boolean).join("\n\n"),
      })),
    {
      role: "user",
      content: [prompt, fileTextFor(request.currentPromptFiles)].filter(Boolean).join("\n\n"),
    },
  ];
}

function getModelProvider(request: TaskRequest): ModelProvider {
  return request.modelProvider ?? "gemini";
}

async function* runGeminiAgentLoop(
  request: TaskRequest,
  decision: ModeDecision,
  retrieval: RetrievalBundle,
  handlers: ToolHandlers,
  signal?: AbortSignal,
): AsyncGenerator<AgentStreamEvent> {
  const contents = buildContents(
    request.conversation ?? [],
    request.prompt,
    request.currentPromptImages,
    request.currentPromptFiles,
  );
  let currentContents = [...contents];
  let turn = 0;

  while (true) {
    if (signal?.aborted) {
      yield { type: "done", summary: "Agent request canceled." };
      return;
    }

    yield { type: "status", label: turn === 0 ? "Loading" : "Thinking" };
    const candidate = await requestGeminiWithRetry(request, decision, retrieval, currentContents, signal);
    const parts = candidate.content?.parts ?? [];
    const functionCalls = getFunctionCalls(parts);

    const finishCall = functionCalls.find((call) => call.name === "finish_turn");
    if (finishCall) {
      const payload = getFinishTurnPayload(finishCall.args);
      if (payload.message) {
        yield { type: "message", chunk: payload.message };
      }
      yield { type: "done", summary: payload.summary || "Agent turn completed." };
      return;
    }

    for (const call of functionCalls) {
      if ((call.name === "write_file" || call.name === "create_file") && typeof call.args.path === "string" && call.args.path.trim()) {
        yield { type: "file-write-start", path: call.args.path };
      }
    }

    if (functionCalls.length === 0) {
      let streamedAggregatedText = "";
      let emittedMessageLength = 0;
      let emittedCodeLength = 0;
      let streamedCandidate: GeminiCandidate = candidate;

      for await (const chunk of requestGeminiStreamWithRetry(request, decision, retrieval, currentContents, false, signal)) {
        if (signal?.aborted) {
          yield { type: "done", summary: "Agent request canceled." };
          return;
        }

        if (chunk.type === "text") {
          streamedAggregatedText += chunk.delta;
          const streamedParts = splitResponseParts(streamedAggregatedText);

          if (streamedParts.message.length > emittedMessageLength) {
            const nextMessageChunk = streamedParts.message.slice(emittedMessageLength);
            emittedMessageLength = streamedParts.message.length;
            yield { type: "message", chunk: nextMessageChunk };
          }

          if (streamedParts.code.length > emittedCodeLength) {
            const nextCodeChunk = streamedParts.code.slice(emittedCodeLength);
            emittedCodeLength = streamedParts.code.length;
            yield { type: "code", chunk: nextCodeChunk };
          }
        } else {
          streamedCandidate = chunk.candidate;
        }
      }

      const aggregatedText = getTextFromParts(streamedCandidate.content?.parts ?? parts);
      const finalParts = splitResponseParts(aggregatedText);

      if (finalParts.message.length > emittedMessageLength) {
        yield { type: "message", chunk: finalParts.message.slice(emittedMessageLength) };
      }

      if (finalParts.code.length > emittedCodeLength) {
        yield { type: "code", chunk: finalParts.code.slice(emittedCodeLength) };
      }

      currentContents.push({
        role: "model",
        parts: streamedCandidate.content?.parts ?? parts,
      });
      await waitUntilAborted(signal);
      yield { type: "done", summary: "Agent request canceled." };
      return;
    }

    currentContents.push({
      role: "model",
      parts,
    });

    const responseParts: GeminiPart[] = [];
    for (const call of functionCalls.filter((entry) => entry.name !== "finish_turn")) {
      const execution = executeToolCall(call, handlers);
      const iterator = execution[Symbol.asyncIterator]();
      while (true) {
        const next = await iterator.next();
        if (next.done) {
          responseParts.push(next.value);
          break;
        }
        yield next.value;
      }
    }

    currentContents.push({
      role: "user",
      parts: responseParts,
    });

    turn += 1;
  }
}

async function* runOllamaAgentLoop(
  request: TaskRequest,
  decision: ModeDecision,
  retrieval: RetrievalBundle,
  handlers: ToolHandlers,
  signal?: AbortSignal,
): AsyncGenerator<AgentStreamEvent> {
  const messages = buildOllamaMessages(request, decision, retrieval, request.prompt, request.conversation ?? []);
  let turn = 0;

  while (true) {
    if (signal?.aborted) {
      yield { type: "done", summary: "Agent request canceled." };
      return;
    }

    yield { type: "status", label: turn === 0 ? "Requesting" : "Thinking" };
    let latestResponse: OllamaChatResponse = {};

    for await (const chunk of requestOllamaStream(request, messages, true, signal)) {
      if (signal?.aborted) {
        yield { type: "done", summary: "Agent request canceled." };
        return;
      }

      if (chunk.type === "text") {
        yield { type: "message", chunk: chunk.delta };
      } else {
        latestResponse = chunk.response;
      }
    }

    const toolCalls = parseOllamaToolCalls(latestResponse.message);
    const finishCall = toolCalls.find((call) => call.name === "finish_turn");
    if (finishCall) {
      const payload = getFinishTurnPayload(finishCall.args);
      if (payload.message) {
        yield { type: "message", chunk: payload.message };
      }
      yield { type: "done", summary: payload.summary || "Agent turn completed." };
      return;
    }

    for (const call of toolCalls) {
      if ((call.name === "write_file" || call.name === "create_file") && typeof call.args.path === "string" && call.args.path.trim()) {
        yield { type: "file-write-start", path: call.args.path };
      }
    }

    if (toolCalls.length === 0) {
      messages.push({
        role: "assistant",
        content: latestResponse.message?.content ?? "",
      });
      await waitUntilAborted(signal);
      yield { type: "done", summary: "Agent request canceled." };
      return;
    }

    messages.push({
      role: "assistant",
      content: latestResponse.message?.content ?? "",
      tool_calls: latestResponse.message?.tool_calls,
    });

    for (const call of toolCalls.filter((entry) => entry.name !== "finish_turn")) {
      const execution = executeToolCall(call, handlers);
      const iterator = execution[Symbol.asyncIterator]();
      while (true) {
        const next = await iterator.next();
        if (next.done) {
          const toolResponse =
            "functionResponse" in next.value ? next.value.functionResponse?.response : undefined;
          messages.push({
            role: "tool",
            tool_name: call.name,
            content: JSON.stringify(toolResponse ?? {}),
          });
          break;
        }

        yield next.value;
      }
    }

    turn += 1;
  }
}

async function* runLmStudioAgentLoop(
  request: TaskRequest,
  decision: ModeDecision,
  retrieval: RetrievalBundle,
  handlers: ToolHandlers,
  signal?: AbortSignal,
): AsyncGenerator<AgentStreamEvent> {
  const messages = buildOpenAiMessages(request, decision, retrieval, request.prompt, request.conversation ?? []);
  let turn = 0;

  while (true) {
    if (signal?.aborted) {
      yield { type: "done", summary: "Agent request canceled." };
      return;
    }

    yield { type: "status", label: turn === 0 ? "Loading" : "Thinking" };
    const response = await requestLmStudioChat(request, messages, true, signal);
    const assistantMessage = response.choices?.[0]?.message;
    const assistantContent = typeof assistantMessage?.content === "string" ? assistantMessage.content : "";
    const toolCalls = parseOpenAiToolCalls(response);
    const finishCall = toolCalls.find((call) => call.name === "finish_turn");

    if (finishCall) {
      const payload = getFinishTurnPayload(finishCall.args);
      if (payload.message) {
        yield { type: "message", chunk: payload.message };
      } else if (assistantContent.trim()) {
        yield { type: "message", chunk: assistantContent };
      }
      yield { type: "done", summary: payload.summary || "Agent turn completed." };
      return;
    }

    for (const call of toolCalls) {
      if ((call.name === "write_file" || call.name === "create_file") && typeof call.args.path === "string" && call.args.path.trim()) {
        yield { type: "file-write-start", path: call.args.path };
      }
    }

    if (toolCalls.length === 0) {
      if (assistantContent) {
        yield { type: "message", chunk: assistantContent };
      }
      await waitUntilAborted(signal);
      yield { type: "done", summary: "Agent request canceled." };
      return;
    }

    messages.push({
      role: "assistant",
      content: assistantContent,
      tool_calls: assistantMessage?.tool_calls,
    });

    for (const call of toolCalls.filter((entry) => entry.name !== "finish_turn")) {
      const execution = executeToolCall(call, handlers);
      const iterator = execution[Symbol.asyncIterator]();
      while (true) {
        const next = await iterator.next();
        if (next.done) {
          const toolResponse =
            "functionResponse" in next.value ? next.value.functionResponse?.response : undefined;
          messages.push({
            role: "tool",
            tool_call_id: call.id,
            content: JSON.stringify(toolResponse ?? {}),
          });
          break;
        }

        yield next.value;
      }
    }

    turn += 1;
  }
}

export async function* runAgentTask(
  request: TaskRequest,
  handlers: ToolHandlers,
  signal?: AbortSignal,
): AsyncGenerator<AgentStreamEvent> {
  const decision = classifyMode(request);
  yield { type: "mode", decision };

  const retrieval = buildRetrievalBundle(request, decision);
  yield { type: "retrieval", bundle: retrieval };

  if (getModelProvider(request) === "gemini" && !request.apiKey?.trim()) {
    yield {
      type: "message",
      chunk: "Gemini API Key is not configured yet. Add it in settings and the model will connect right away.",
    };
    yield { type: "done", summary: "Missing Gemini API key." };
    return;
  }

  try {
    if (getModelProvider(request) === "ollama") {
      yield* runOllamaAgentLoop(request, decision, retrieval, handlers, signal);
    } else if (getModelProvider(request) === "lmstudio") {
      yield* runLmStudioAgentLoop(request, decision, retrieval, handlers, signal);
    } else {
      yield* runGeminiAgentLoop(request, decision, retrieval, handlers, signal);
    }
  } catch (error) {
    if (signal?.aborted || (error instanceof Error && error.name === "AbortError")) {
      yield { type: "done", summary: "Agent request canceled." };
      return;
    }

    const providerLabel =
      getModelProvider(request) === "ollama"
        ? "Ollama"
        : getModelProvider(request) === "lmstudio"
          ? "LM Studio"
          : "Gemini";
    const message = error instanceof Error ? error.message : `Unknown ${providerLabel} error`;
    yield { type: "message", chunk: `${providerLabel} request failed: ${message}` };
    yield { type: "done", summary: `${providerLabel} request failed.` };
  }
}

function getEffectiveModelName(request: TaskRequest): string {
  if (getModelProvider(request) === "gemini") {
    return request.modelId?.trim() || modelForTier(request.modelTier);
  }

  if (getModelProvider(request) === "ollama") {
    return request.modelId?.trim() || "llama3:latest";
  }

  if (getModelProvider(request) === "lmstudio") {
    return request.modelId?.trim() || "local-model";
  }
  return modelForTier(request.modelTier);
}

export function buildContextSnapshot(request: TaskRequest): ContextSnapshot {
  const modeDecision = classifyMode(request);
  return {
    modeDecision,
    retrieval: buildRetrievalBundle(request, modeDecision),
    modelTier: request.modelTier,
    modelProvider: getModelProvider(request),
    model: getEffectiveModelName(request),
  };
}

export async function buildContextUsageSnapshot(request: TaskRequest): Promise<ContextUsageSnapshot | null> {
  if (getModelProvider(request) === "ollama") {
    const model = getEffectiveModelName(request);
    const modeDecision = classifyMode(request);
    const retrieval = buildRetrievalBundle(request, modeDecision);
    const messages = buildOllamaMessages(request, modeDecision, retrieval, request.prompt, request.conversation ?? []);
    let runningContextLength: number | null = null;

    try {
      const runningResponse = await fetch("http://127.0.0.1:11434/api/ps");
      if (runningResponse.ok) {
        const runningPayload = (await runningResponse.json()) as {
          models?: Array<{
            model?: string;
            name?: string;
            context_length?: number;
          }>;
        };
        const matchingModel = (runningPayload.models ?? []).find((entry) => (entry.model ?? entry.name ?? "") === model);
        runningContextLength = typeof matchingModel?.context_length === "number" ? matchingModel.context_length : null;
      }
    } catch {
      runningContextLength = null;
    }

    const showResponse = await fetch("http://127.0.0.1:11434/api/show", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ model }),
    });

    if (!showResponse.ok) {
      const body = await showResponse.text();
      throw new Error(`Ollama show failed: ${showResponse.status} ${body}`);
    }

    const showPayload = (await showResponse.json()) as { model_info?: Record<string, unknown> };
    const contextLengthEntry = Object.entries(showPayload.model_info ?? {}).find(([key]) => key.endsWith(".context_length"));
    const inputTokenLimit =
      request.ollamaContextLength && request.ollamaContextLength > 0
        ? request.ollamaContextLength
        : runningContextLength && runningContextLength > 0
          ? runningContextLength
        : typeof contextLengthEntry?.[1] === "number"
          ? contextLengthEntry[1]
          : 131_072;
    const countResponse = await fetch("http://127.0.0.1:11434/api/chat", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        model,
        stream: false,
        messages,
        options: {
          num_predict: 0,
        },
      }),
    });

    if (!countResponse.ok) {
      const body = await countResponse.text();
      throw new Error(`Ollama chat count failed: ${countResponse.status} ${body}`);
    }

    const countPayload = (await countResponse.json()) as OllamaChatResponse;
    const usedTokens = countPayload.prompt_eval_count ?? 0;

    return {
      model,
      usedTokens,
      inputTokenLimit,
      usagePercent: Math.max(0, Math.min(100, Math.round((usedTokens / inputTokenLimit) * 100))),
      compressionState: retrieval.compressionState,
      snippetCount: retrieval.snippets.length,
    };
  }

  if (getModelProvider(request) === "lmstudio") {
    const model = getEffectiveModelName(request);
    const modeDecision = classifyMode(request);
    const retrieval = buildRetrievalBundle(request, modeDecision);
    const messages = buildOpenAiMessages(request, modeDecision, retrieval, request.prompt, request.conversation ?? []);
    const countResponse = await fetch("http://127.0.0.1:1234/v1/chat/completions", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        model,
        stream: false,
        max_tokens: 1,
        messages,
      }),
    });

    if (!countResponse.ok) {
      const body = await countResponse.text();
      throw new Error(`LM Studio chat count failed: ${countResponse.status} ${body}`);
    }

    const countPayload = (await countResponse.json()) as OpenAiChatCompletionResponse;
    const usedTokens = countPayload.usage?.prompt_tokens ?? 0;
    const inputTokenLimit = 32_768;

    return {
      model,
      usedTokens,
      inputTokenLimit,
      usagePercent: Math.max(0, Math.min(100, Math.round((usedTokens / inputTokenLimit) * 100))),
      compressionState: retrieval.compressionState,
      snippetCount: retrieval.snippets.length,
    };
  }

  if (!request.apiKey?.trim()) {
    return null;
  }

  const modeDecision = classifyMode(request);
  const retrieval = buildRetrievalBundle(request, modeDecision);
  const contents = buildContents(
    request.conversation ?? [],
    request.prompt,
    request.currentPromptImages,
    request.currentPromptFiles,
  );
  const model = getEffectiveModelName(request);
  const apiKey = request.apiKey.trim();

  const [countResponse, modelResponse] = await Promise.all([
    fetch(`https://generativelanguage.googleapis.com/v1beta/models/${model}:countTokens`, {
      method: "POST",
        headers: {
          "Content-Type": "application/json",
          "x-goog-api-key": apiKey,
        },
        body: JSON.stringify({
          generateContentRequest: {
            model: `models/${model}`,
            ...buildGenerateContentPayload(request, modeDecision, retrieval, contents, true),
          },
        }),
      }),
    fetch(`https://generativelanguage.googleapis.com/v1beta/models/${model}`, {
      method: "GET",
      headers: {
        "x-goog-api-key": apiKey,
      },
    }),
  ]);

  if (!countResponse.ok) {
    const body = await countResponse.text();
    throw new Error(`Gemini countTokens failed: ${countResponse.status} ${body}`);
  }

  if (!modelResponse.ok) {
    const body = await modelResponse.text();
    throw new Error(`Gemini model lookup failed: ${modelResponse.status} ${body}`);
  }

  const countPayload = (await countResponse.json()) as { totalTokens?: number };
  const modelPayload = (await modelResponse.json()) as { name?: string; inputTokenLimit?: number };
  const usedTokens = countPayload.totalTokens ?? 0;
  const inputTokenLimit = modelPayload.inputTokenLimit ?? 1;

  return {
    model: modelPayload.name ?? model,
    usedTokens,
    inputTokenLimit,
    usagePercent: Math.max(0, Math.min(100, Math.round((usedTokens / inputTokenLimit) * 100))),
    compressionState: retrieval.compressionState,
    snippetCount: retrieval.snippets.length,
  };
}
