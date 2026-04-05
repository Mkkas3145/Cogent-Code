# AI Editor Architecture

## Overview

This document describes the architecture of a cross-platform AI coding workspace built with Electron, React, Monaco, and a local runtime. The product targets macOS and Windows and is designed around a chat-first workflow that can open into file editing when needed.

The editor supports:

- User-selectable coding modes:
  - Auto
  - Backend
  - Frontend
- Automatic mode recognition when `Auto` is selected
- Real-time code generation:
  - streamed AI output can be applied into the working buffer immediately
- Real-time reasoning:
  - search-based retrieval pipeline for long-context continuity
- Local command execution:
  - run `cmd` and shell commands
  - stream stdout/stderr back into the reasoning loop
- Model selection:
  - Gemini `Flash Lite`
  - Gemini `Flash`
  - Gemini `Pro`
- Settings popup for `Gemini API Key` management
- Automatic context compression:
  - proactive context summarization without interrupting the user

This architecture is designed to feel like a flexible AI coding workspace first, not a rigid one-shot generator.

## Product Goals

- Keep the chat flow natural enough that the user can talk casually, ask for code changes, and continue the same task without mode friction.
- Separate frontend and backend reasoning behavior to improve answer quality.
- Maintain long-running task continuity without requiring the user to manually manage context.
- Allow the AI agent to read code, search code, execute tools, and continue reasoning from live outputs.
- Make every high-risk action reviewable while keeping low-friction edits fast.
- Support future expansion into multi-agent orchestration and deeper repository intelligence.

## Core Stack

- Desktop shell: Electron
- UI: React + TypeScript
- Styling: custom dark desktop UI with custom titlebar, modal system, and slide panels
- Editor: Monaco Editor
- Agent orchestration runtime: Node.js
- High-performance utilities: Rust
- Initial model provider: Gemini
- IPC boundary: Electron preload bridge
- Local storage:
  - local UI state in renderer storage
  - room for SQLite or IndexedDB-backed state later

## High-Level System

```text
Electron Main
  -> window lifecycle
  -> secure IPC
  -> local process manager
  -> filesystem permission layer

React Renderer
  -> chat-first workspace
  -> custom titlebar
  -> workspace folder menu
  -> hover file panel
  -> Monaco editor overlay
  -> settings modal
  -> mode and model selection

Agent Runtime (Node)
  -> mode classifier
  -> retrieval engine
  -> planning engine
  -> tool router
  -> streaming patch applier
  -> verification controller
  -> context compression manager
  -> Gemini provider adapter

Native Services (Rust)
  -> repository indexer
  -> symbol search
  -> file watcher
  -> safe command runner
  -> diff and patch utility
```

## User-Facing Modes

### 1. Auto Mode

Used when:

- the user wants the agent to decide how to approach the task
- the prompt mixes conversation, diagnosis, and edits
- the active file or repository context should steer the reasoning path

Behavior priorities:

- infer backend vs frontend emphasis from prompt and repo signals
- stay conversational unless code work is clearly needed
- avoid exposing internal pipeline noise unless it helps the user

### 2. Backend Mode

Used for:

- APIs
- services
- business logic
- database logic
- migrations
- auth
- background jobs
- CLI/server code

Behavior priorities:

- contract correctness
- type and schema awareness
- side effect tracing
- test execution
- command-driven debugging

### 3. Frontend Mode

Used for:

- HTML/CSS/JavaScript
- React components
- state and interaction logic
- UI layout
- styling systems
- editor panels and app shell UI

Behavior priorities:

- component and DOM understanding
- state flow tracing
- style impact tracking
- visual and interaction verification

## Main UX Surfaces

- Custom titlebar
- Workspace folder menu
- Hover file panel
- Monaco editor overlay
- Chat timeline
- Composer with mode and model controls
- Settings popup

The visible UI should stay lightweight. Detailed retrieval and tool state can exist internally, but should not flood the chat unless the agent chooses to explain it.

## Architecture Layers

### 1. Electron Main Layer

Responsibilities:

- create windows
- manage custom titlebar window controls
- expose secure APIs through preload
- own command execution authority
- own filesystem mutation authority
- own process lifecycle

Security defaults:

- `contextIsolation: true`
- `nodeIntegration: false`
- renderer cannot directly access filesystem or shell
- command execution goes through validated IPC calls
- dangerous paths can be blocked or require explicit confirmation

### 2. Renderer Layer

Responsibilities:

- render chat UI
- open workspace folders and file trees
- open files in Monaco
- show save state and file actions
- stream model responses into chat and editor surfaces
- host lightweight settings and workspace state

Recommended state domains:

- workspace state
- file tree state
- editor session state
- chat state
- model state
- mode state
- command execution state
- compression state

### 3. Agent Runtime Layer

This is the main orchestration engine and should initially be implemented in Node.js.

Responsibilities:

- classify task mode
- gather context
- retrieve relevant files and summaries
- route calls to Gemini
- stream conversational responses
- stream code edits
- run commands
- consume command output
- verify edits
- compress context in the background

### 4. Rust Native Services

Rust should be added where latency and repository size matter.

Priority Rust modules:

- repository indexing
- fast search
- symbol graph extraction
- file change watcher
- safe command execution wrapper
- patch merge and conflict helper

Keep the first version operational even if Rust services are temporarily unavailable. Node-based fallback behavior should exist for development.

## Agent Pipeline

The system should not use one fixed prompt path. It should dynamically compose a pipeline based on mode, task shape, repository state, and command output.

### Standard Task Flow

```text
User request
  -> mode selection or auto-detection
  -> retrieval strategy selection
  -> context build
  -> planning
  -> model streaming response
  -> real-time patch application
  -> live verification
  -> command execution if needed
  -> command output fed back into reasoning
  -> follow-up patching
  -> final verification
  -> background compression
```

### Backend Pipeline

```text
Prompt
  -> entry point detection
  -> service/schema/test retrieval
  -> dependency graph expansion
  -> plan generation
  -> streamed code patch
  -> typecheck and tests
  -> command output analysis
  -> follow-up patch if needed
  -> finalize
```

Backend-specific retrieval priority:

- route handlers
- services
- repository layer
- schemas and DTOs
- tests
- environment and config

Backend-specific verification priority:

- typecheck
- unit tests
- integration tests
- API contract compatibility
- migration safety

### Frontend Pipeline

```text
Prompt
  -> UI entry detection
  -> component tree retrieval
  -> state flow tracing
  -> style impact scan
  -> plan generation
  -> streamed code patch
  -> render and interaction verification
  -> command output analysis
  -> follow-up patch if needed
  -> finalize
```

Frontend-specific retrieval priority:

- page entry files
- active components
- styling sources
- hooks or stores
- shared UI primitives
- route definitions

Frontend-specific verification priority:

- build success
- lint and typecheck
- visual render status
- interaction event integrity
- accessibility checks where available

### Flexible Frontend Reasoning

The visible mode is just `frontend`, but the retrieval and prompting layer can still internally adapt for:

- plain DOM logic
- event listeners
- direct CSS and layout changes
- React component boundaries
- hooks and state management
- JSX and styling conventions

## Real-Time Code Writing

The product should support streamed application of model output into the active code buffer.

There are two patch modes:

### 1. Ghost Draft Mode

- stream the generated code into a shadow buffer
- visually show incoming edits
- let the user accept or stop midway

### 2. Live Apply Mode

- apply streamed edits directly into the file buffer
- maintain an undo boundary per generation session
- show a live patch overlay

Recommended default:

- use Ghost Draft Mode for broad or risky edits
- use Live Apply Mode for small focused edits or explicitly enabled sessions

Implementation requirements:

- token stream must be converted into structured patch chunks
- preserve cursor and scroll position when possible
- allow cancellation without corrupting the file
- continuously validate syntax while streaming

## Real-Time Reasoning with Retrieval

Long context should be handled through retrieval, not by keeping the entire repository in the prompt.

### Retrieval Engine Responsibilities

- index the repository
- track active files, recent files, and edited files
- rank relevant files per request
- fetch symbol-level snippets
- merge raw code with summaries
- maintain session memory

### Retrieval Sources

- active file
- selected code
- open files
- recently edited files
- symbol references
- test files
- terminal output
- git diff
- project config
- previous task summaries

### Retrieval Modes

- narrow retrieval:
  - use for focused edits
- broad retrieval:
  - use for architecture or debugging tasks
- recovery retrieval:
  - use after command/test failures

### Retrieval Ranking Signals

- keyword match
- symbol match
- import relationship
- path proximity
- recency
- prior task usage
- mode compatibility

## Command Execution and Reasoning Loop

The agent must be able to run shell and `cmd` commands, receive streamed output, and continue reasoning based on the result.

Supported scenarios:

- run tests
- run builds
- inspect logs
- start local servers
- run formatters and linters
- query package tools
- inspect filesystem state

Execution flow:

```text
Agent decides command
  -> command validator
  -> process launch
  -> stdout/stderr streaming
  -> output parser
  -> issue extraction
  -> retrieval refresh
  -> reasoning continuation
  -> patch or next command
```

Requirements:

- stream stdout/stderr in near real time
- support cancellation
- support long-running processes
- support structured issue extraction from logs
- command results become retrieval inputs

Recommended command safety layers:

- working directory restrictions
- allowlist and risk scoring
- destructive command warning
- environment isolation options

## Model Routing and Gemini Integration

The initial implementation should be Gemini-based, but the architecture must allow additional providers later.

### Model Selector

User-selectable controls:

- `Flash Lite`
- `Flash`
- `Pro`

Internal routing inputs:

- task mode
- edit size
- repository size
- current context pressure
- whether command reasoning is active

Provider abstraction:

```text
ModelProvider
  -> streamText()
  -> streamPatch()
  -> summarize()
  -> compressContext()
  -> classifyMode()
```

Gemini-first implementation notes:

- use Gemini for planning, generation, summarization, and compression
- keep prompt templates separated by mode where useful
- keep tool-use state outside the model prompt where possible
- support resumable streaming sessions
- let the model decide whether to answer conversationally, produce code, or do both

## Automatic Context Compression

The system should compress context proactively before the model context window becomes a problem. This should happen without requiring manual user cleanup.

### Compression Triggers

- token usage crosses a threshold
- multiple command outputs accumulate
- long multi-step task history
- many file summaries in working memory
- mode change after a long session

### Compression Strategy

Compress in layers:

1. Keep critical raw artifacts:
   - active selection
   - current diff
   - latest terminal failures
   - current plan
2. Summarize old chat turns into task memory
3. Summarize previously read files into symbol-level memory
4. Drop low-value conversational text
5. Preserve unresolved questions and assumptions

### Compression Types

- conversation compression
- code context compression
- terminal output compression
- verification history compression

### Compression Safety Rules

- never compress away the active diff
- never compress away unresolved errors
- never compress away explicit user constraints
- preserve user mode and model choices

The user should experience this as continuity, not as a visible memory reset.

## Recommended Additional Features

### 1. Intent-Aware Edit Scope

Before applying code, classify edit scope:

- line-level
- function-level
- file-level
- multi-file
- workspace-wide

Use this to choose:

- live apply or ghost draft
- strictness of verification
- whether approval is required

### 2. Confidence Scoring

Each run should produce a confidence score based on:

- retrieval quality
- mode match confidence
- command/test results
- patch coherence
- unresolved diagnostics

Use low confidence to trigger:

- wider retrieval
- stronger verification
- a second planning pass
- fallback from live apply to draft mode

### 3. Recovery Loop

When verification fails:

- parse the failure
- retrieve related files
- ask the model for a repair patch
- retry within bounded limits

This is especially important for backend tasks and command-driven debugging.

### 4. Task Memory

Store lightweight project memory:

- coding conventions
- common commands
- preferred frameworks
- ignored paths
- repo-specific safety rules

This memory should be retrieval-backed, not permanently stuffed into every prompt.

### 5. Patch Review Intelligence

For each patch, generate:

- summary of intent
- impacted files
- risk level
- verification status

This makes real-time editing safer without slowing down flow.

### 6. Conversational Agent Flow

The agent must not be hardwired into one coding-only response shape. It should be able to:

- answer normal chat naturally
- continue a coding task after casual conversation
- explain what it changed in plain language
- switch from discussion to editing without forcing a separate mode change

The visible chat should remain conversational even when internal retrieval, planning, or command execution is happening.

### 7. Checkpoints and Rewind

Create lightweight automatic checkpoints:

- before live apply
- before command-driven fixes
- before multi-file edits

Allow one-click rollback to a checkpoint.

### 8. Search-First Long Session Mode

For sessions that run a long time, degrade prompt size and rely more on:

- compressed memory
- code search
- symbol retrieval
- latest command outputs

This keeps cost and latency manageable.

## Suggested Internal Modules

```text
/apps/desktop
  /electron-main
  /renderer

/packages/shared-types
/packages/agent-core
/packages/mode-classifier
/packages/retrieval-engine
/packages/context-memory
/packages/model-gemini
/packages/streaming-patch
/packages/verification-engine
/packages/command-runtime
/packages/frontend-mode
/packages/backend-mode
/packages/rust-indexer
/packages/rust-command-guard
```

## Key Type Shapes

```ts
export type AgentMode =
  | "auto"
  | "backend"
  | "frontend";

export type ModelTier =
  | "flash-lite"
  | "flash"
  | "pro";

export type TaskRequest = {
  prompt: string;
  activeFile?: string;
  selectedText?: string;
  openFiles: string[];
  explicitMode?: Exclude<AgentMode, "auto">;
  modelTier: ModelTier;
  liveApply: boolean;
};

export type ModeDecision = {
  mode: Exclude<AgentMode, "auto">;
  source: "user" | "auto";
  confidence: number;
  reasons: string[];
};

export type RetrievalBundle = {
  files: string[];
  snippets: Array<{
    path: string;
    content: string;
    score: number;
    reason: string;
  }>;
  memorySummary?: string;
  terminalSummary?: string;
};

export type VerificationResult = {
  passed: boolean;
  steps: Array<{
    name: string;
    passed: boolean;
    summary: string;
  }>;
  issues: string[];
};
```

## Current Implementation Notes

The current app already includes:

- a custom titlebar with folder menu and window controls
- a hover-based file panel with resizable width
- a Monaco editor overlay for opened files
- local file read and save support
- a settings popup with `Gemini API Key` storage
- chat-style user and assistant message rendering
- user-selectable `Auto`, `Backend`, `Frontend` modes
- user-selectable `Flash Lite`, `Flash`, `Pro` model labels

The biggest remaining gap is the agent runtime. The UI is prepared for a real Gemini-backed conversational coding flow, but the runtime should avoid hardcoded pseudo-intent behavior and instead let the model decide how to respond, code, or continue the conversation.

## MVP Scope

Phase 1 should include:

- Electron shell
- React workspace
- Monaco integration
- backend mode
- frontend mode
- Gemini model adapter
- retrieval-based context builder
- streamed editor patching
- command execution with live output
- automatic context compression
- diff preview and rollback

Phase 2 should include:

- stronger visual verification
- richer repository graphing
- checkpoint manager
- confidence-driven retries
- persistent project memory
- more providers beyond Gemini

## Final Recommendation

Build the first version as a search-first, streaming-first workspace with strict mode-aware retrieval and verification. The biggest differentiator should be that the agent does not just chat about code. It should:

- understand whether it is doing backend or frontend work
- stream code into the editor in real time
- run commands and reason from results
- preserve long task continuity through retrieval and hidden compression
- stay conversational enough that the user can interrupt, redirect, and continue naturally

That combination will produce a more trustworthy and more useful coding editor than a generic chat panel attached to Monaco.
