import { spawn } from "node:child_process";
import type { CommandChunk, RunCommandRequest } from "@cogent/shared-types";

export async function streamCommand(
  request: RunCommandRequest,
  onChunk: (chunk: CommandChunk) => void,
): Promise<{ exitCode: number | null }> {
  const isWindows = process.platform === "win32";
  const command = isWindows
    ? `[Console]::InputEncoding = [System.Text.UTF8Encoding]::new($false); [Console]::OutputEncoding = [System.Text.UTF8Encoding]::new($false); $OutputEncoding = [System.Text.UTF8Encoding]::new($false); chcp 65001 > $null; ${request.command}`
    : request.command;
  const child = isWindows
    ? spawn(
        "powershell.exe",
        ["-NoProfile", "-NonInteractive", "-ExecutionPolicy", "Bypass", "-Command", command],
        {
          cwd: request.cwd,
        },
      )
    : spawn("/bin/sh", ["-lc", command], {
        cwd: request.cwd,
      });

  child.stdout.setEncoding("utf8");
  child.stderr.setEncoding("utf8");

  onChunk({ stream: "system", text: `Running: ${request.command}\n` });

  child.stdout.on("data", (chunk) => {
    onChunk({ stream: "stdout", text: chunk });
  });

  child.stderr.on("data", (chunk) => {
    onChunk({ stream: "stderr", text: chunk });
  });

  const exitCode = await new Promise<number | null>((resolve, reject) => {
    child.once("error", reject);
    child.once("close", (code) => resolve(code));
  });

  return { exitCode };
}
