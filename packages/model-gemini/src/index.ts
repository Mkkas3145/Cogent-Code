export type GeminiConfig = {
  apiKey?: string;
  model: string;
};

export function createGeminiAdapter(config: GeminiConfig) {
  return {
    provider: "gemini",
    model: config.model,
    async *streamText(prompt: string): AsyncGenerator<string> {
      yield `Gemini placeholder response for: ${prompt}`;
    },
  };
}
