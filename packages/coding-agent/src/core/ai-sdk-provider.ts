import type {
	JSONSchema7,
	LanguageModelV2,
	LanguageModelV2CallOptions,
	LanguageModelV2FinishReason,
	LanguageModelV2Prompt,
	LanguageModelV2StreamPart,
	SharedV2ProviderOptions,
} from "@ai-sdk/provider";
import {
	type Api,
	type AssistantMessage,
	type AssistantMessageEventStream,
	type Context,
	calculateCost,
	createAssistantMessageEventStream,
	type ImageContent,
	type Model,
	type OpenAICompletionsCompat,
	type SimpleStreamOptions,
	type StopReason,
	type TextContent,
	type ThinkingBudgets,
	type ThinkingContent,
	type ThinkingLevel,
	type Tool,
	type ToolCall,
	type ToolResultMessage,
} from "@mariozechner/pi-ai";
import type { Provider as AiSdkProvider } from "ai";

export const AI_SDK_INLINE_AUTH_TOKEN = "__PI_AI_SDK_INLINE_AUTH__";

export interface AiSdkProviderFactoryOptions {
	apiKey?: string;
	baseUrl?: string;
	headers?: Record<string, string>;
}

export type AiSdkProviderFactory = (options: AiSdkProviderFactoryOptions) => AiSdkProvider;

export interface AiSdkModelConfig {
	id: string;
	name?: string;
	reasoning?: boolean;
	input?: ("text" | "image")[];
	cost?: Partial<Model<Api>["cost"]>;
	contextWindow?: number;
	maxTokens?: number;
	headers?: Record<string, string>;
	sourceProvider?: string;
	sourceModelId?: string;
	compat?: OpenAICompletionsCompat;
}

export interface AiSdkProviderOptionsContext {
	model: Model<Api>;
	provider: string;
	modelId: string;
	reasoning?: ThinkingLevel;
	thinkingBudgets?: ThinkingBudgets;
}

export type AiSdkProviderOptionsFactory = (
	context: AiSdkProviderOptionsContext,
) => SharedV2ProviderOptions | undefined | Promise<SharedV2ProviderOptions | undefined>;

export interface AiSdkProviderConfig {
	provider: AiSdkProvider | AiSdkProviderFactory;
	discoverModels?: boolean;
	models?: Array<string | AiSdkModelConfig>;
	providerOptions?: SharedV2ProviderOptions | AiSdkProviderOptionsFactory;
}

interface GatewayModelEntry {
	id: string;
	name: string;
	specification?: {
		provider?: string;
		modelId?: string;
	};
	pricing?: {
		input: string;
		output: string;
		cachedInputTokens?: string;
		cacheCreationInputTokens?: string;
	} | null;
}

type DiscoveringProvider = AiSdkProvider & {
	getAvailableModels?: () => Promise<{ models: GatewayModelEntry[] }>;
};

type PendingToolCall = ToolCall & {
	streamId?: string;
	partialJson?: string;
	ended: boolean;
};

type StreamBlock = TextContent | ThinkingContent | PendingToolCall;

function isAiSdkProviderInstance(value: AiSdkProviderConfig["provider"]): value is AiSdkProvider {
	return (
		(typeof value === "function" || typeof value === "object") &&
		value !== null &&
		("languageModel" in value || "chatModel" in value || "completionModel" in value)
	);
}

function isAiSdkProviderFactory(value: AiSdkProviderConfig["provider"]): value is AiSdkProviderFactory {
	return typeof value === "function" && !isAiSdkProviderInstance(value);
}

export function resolveAiSdkProvider(
	config: AiSdkProviderConfig,
	options: AiSdkProviderFactoryOptions = {},
): AiSdkProvider {
	return isAiSdkProviderFactory(config.provider) ? config.provider(options) : config.provider;
}

export async function discoverAiSdkModels(
	config: AiSdkProviderConfig,
	options: AiSdkProviderFactoryOptions = {},
): Promise<AiSdkModelConfig[]> {
	const provider = resolveAiSdkProvider(config, options) as DiscoveringProvider;
	if (!provider.getAvailableModels) {
		return [];
	}

	const result = await provider.getAvailableModels();
	return result.models.map((model) => ({
		id: model.id,
		name: model.name,
		sourceProvider: model.specification?.provider,
		sourceModelId: model.specification?.modelId,
		cost: model.pricing
			? {
					input: parseMillionTokenPrice(model.pricing.input),
					output: parseMillionTokenPrice(model.pricing.output),
					cacheRead: parseMillionTokenPrice(model.pricing.cachedInputTokens),
					cacheWrite: parseMillionTokenPrice(model.pricing.cacheCreationInputTokens),
				}
			: undefined,
	}));
}

function parseMillionTokenPrice(value: string | undefined): number | undefined {
	if (!value) return undefined;
	const parsed = Number.parseFloat(value);
	if (!Number.isFinite(parsed)) return undefined;
	return parsed * 1_000_000;
}

function mergeProviderOptions(
	...options: Array<SharedV2ProviderOptions | undefined>
): SharedV2ProviderOptions | undefined {
	const merged = options.reduce<SharedV2ProviderOptions>((result, value) => {
		if (!value) return result;
		for (const [provider, providerOptions] of Object.entries(value)) {
			result[provider] = {
				...(result[provider] ?? {}),
				...(providerOptions ?? {}),
			};
		}
		return result;
	}, {});
	return Object.keys(merged).length > 0 ? merged : undefined;
}

function inferReasoningProviderKey(languageModelProvider: string, modelId: string): string {
	if (languageModelProvider === "gateway" && modelId.includes("/")) {
		return modelId.slice(0, modelId.indexOf("/"));
	}
	return languageModelProvider;
}

function getThinkingBudget(level: ThinkingLevel, budgets?: ThinkingBudgets): number {
	const defaults: Record<ThinkingLevel, number> = {
		minimal: 1024,
		low: 4096,
		medium: 10240,
		high: 20480,
		xhigh: 32768,
	};
	const configuredBudget = level === "xhigh" ? budgets?.high : budgets?.[level];
	return configuredBudget ?? defaults[level];
}

function inferReasoningProviderOptions(
	languageModelProvider: string,
	modelId: string,
	reasoning: ThinkingLevel | undefined,
	thinkingBudgets?: ThinkingBudgets,
): SharedV2ProviderOptions | undefined {
	if (!reasoning) return undefined;

	const providerKey = inferReasoningProviderKey(languageModelProvider, modelId);
	if (providerKey === "anthropic") {
		return {
			anthropic: {
				thinking: {
					type: "enabled",
					budgetTokens: getThinkingBudget(reasoning, thinkingBudgets),
				},
			},
		};
	}

	if (providerKey === "openai") {
		const effort = reasoning === "xhigh" ? "high" : reasoning;
		return {
			openai: {
				reasoningEffort: effort,
			},
		};
	}

	return undefined;
}

function mapStopReason(reason: LanguageModelV2FinishReason): StopReason {
	switch (reason) {
		case "length":
			return "length";
		case "tool-calls":
			return "toolUse";
		case "stop":
		case "content-filter":
		case "other":
		case "unknown":
			return "stop";
		case "error":
			return "error";
	}
}

function extractToolResultOutput(message: ToolResultMessage) {
	const hasImages = message.content.some((item: TextContent | ImageContent) => item.type === "image");
	if (hasImages) {
		return {
			type: "content" as const,
			value: message.content.map((item: TextContent | ImageContent) =>
				item.type === "text"
					? { type: "text" as const, text: item.text }
					: { type: "media" as const, data: item.data, mediaType: item.mimeType },
			),
		};
	}

	const text = message.content
		.filter((item: TextContent | ImageContent): item is TextContent => item.type === "text")
		.map((item: TextContent) => item.text)
		.join("\n");

	return message.isError ? { type: "error-text" as const, value: text } : { type: "text" as const, value: text };
}

function toPrompt(context: Context): LanguageModelV2Prompt {
	const prompt: LanguageModelV2Prompt = [];

	if (context.systemPrompt) {
		prompt.push({
			role: "system",
			content: context.systemPrompt,
		});
	}

	for (let index = 0; index < context.messages.length; index++) {
		const message = context.messages[index];

		if (message.role === "user") {
			if (typeof message.content === "string") {
				prompt.push({
					role: "user",
					content: [{ type: "text", text: message.content }],
				});
				continue;
			}

			prompt.push({
				role: "user",
				content: message.content.map((item: TextContent | ImageContent) =>
					item.type === "text"
						? { type: "text" as const, text: item.text }
						: { type: "file" as const, data: item.data, mediaType: item.mimeType },
				),
			});
			continue;
		}

		if (message.role === "assistant") {
			prompt.push({
				role: "assistant",
				content: message.content.map((item: TextContent | ThinkingContent | ToolCall) => {
					if (item.type === "text") {
						return { type: "text" as const, text: item.text };
					}
					if (item.type === "thinking") {
						return { type: "reasoning" as const, text: item.thinking };
					}
					return {
						type: "tool-call" as const,
						toolCallId: item.id,
						toolName: item.name,
						input: item.arguments,
					};
				}),
			});
			continue;
		}

		const toolResults = [extractToolResultOutput(message)];
		let nextIndex = index + 1;
		while (nextIndex < context.messages.length && context.messages[nextIndex]?.role === "toolResult") {
			toolResults.push(extractToolResultOutput(context.messages[nextIndex] as ToolResultMessage));
			nextIndex++;
		}

		const groupedMessages = context.messages.slice(index, nextIndex) as ToolResultMessage[];
		prompt.push({
			role: "tool",
			content: groupedMessages.map((toolMessage, toolIndex) => ({
				type: "tool-result" as const,
				toolCallId: toolMessage.toolCallId,
				toolName: toolMessage.toolName,
				output: toolResults[toolIndex],
			})),
		});
		index = nextIndex - 1;
	}

	return prompt;
}

function toTools(tools: Tool[] | undefined) {
	if (!tools || tools.length === 0) return undefined;
	return tools.map((tool) => ({
		type: "function" as const,
		name: tool.name,
		description: tool.description,
		inputSchema: tool.parameters as JSONSchema7,
	}));
}

function getStreamBlockIndex(blocks: StreamBlock[], streamId: string): number {
	return blocks.findIndex((block) => "streamId" in block && block.streamId === streamId);
}

function ensureStartPushed(stream: AssistantMessageEventStream, output: AssistantMessage, state: { started: boolean }) {
	if (state.started) return;
	state.started = true;
	stream.push({ type: "start", partial: output });
}

function closePendingToolCalls(stream: AssistantMessageEventStream, output: AssistantMessage, blocks: StreamBlock[]) {
	for (const [index, block] of blocks.entries()) {
		if (block.type !== "toolCall" || block.ended) continue;
		block.ended = true;
		try {
			block.arguments = block.partialJson ? JSON.parse(block.partialJson) : {};
		} catch {
			block.arguments = {};
		}
		delete block.partialJson;
		delete block.streamId;
		stream.push({ type: "toolcall_end", contentIndex: index, toolCall: block, partial: output });
	}
}

export function createAiSdkStreamSimple(options: {
	resolveProvider: (options: AiSdkProviderFactoryOptions) => AiSdkProvider;
	providerOptions?: SharedV2ProviderOptions | AiSdkProviderOptionsFactory;
}) {
	return function streamAiSdkProvider(
		model: Model<Api>,
		context: Context,
		streamOptions?: SimpleStreamOptions,
	): AssistantMessageEventStream {
		const stream = createAssistantMessageEventStream();

		(async () => {
			const output: AssistantMessage = {
				role: "assistant",
				content: [],
				api: model.api,
				provider: model.provider,
				model: model.id,
				usage: {
					input: 0,
					output: 0,
					cacheRead: 0,
					cacheWrite: 0,
					totalTokens: 0,
					cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 },
				},
				stopReason: "stop",
				timestamp: Date.now(),
			};

			const blocks = output.content as StreamBlock[];
			const state = { started: false };

			try {
				const provider = options.resolveProvider({
					apiKey:
						streamOptions?.apiKey && streamOptions.apiKey !== AI_SDK_INLINE_AUTH_TOKEN
							? streamOptions.apiKey
							: undefined,
					baseUrl: model.baseUrl,
					headers: model.headers,
				});
				const languageModel = provider.languageModel(model.id) as LanguageModelV2;
				const reasoning = model.reasoning ? streamOptions?.reasoning : undefined;
				const dynamicProviderOptions =
					typeof options.providerOptions === "function"
						? await options.providerOptions({
								model,
								provider: languageModel.provider,
								modelId: languageModel.modelId,
								reasoning,
								thinkingBudgets: streamOptions?.thinkingBudgets,
							})
						: options.providerOptions;

				const requestOptions: LanguageModelV2CallOptions = {
					prompt: toPrompt(context),
					maxOutputTokens: streamOptions?.maxTokens,
					temperature: streamOptions?.temperature,
					abortSignal: streamOptions?.signal,
					tools: toTools(context.tools),
					toolChoice: context.tools?.length ? { type: "auto" } : undefined,
					providerOptions: mergeProviderOptions(
						inferReasoningProviderOptions(
							languageModel.provider,
							languageModel.modelId,
							reasoning,
							streamOptions?.thinkingBudgets,
						),
						dynamicProviderOptions,
					),
				};

				const payload = (await streamOptions?.onPayload?.(requestOptions, model)) ?? requestOptions;
				const response = await languageModel.doStream(payload as LanguageModelV2CallOptions);

				for await (const event of response.stream as AsyncIterable<LanguageModelV2StreamPart>) {
					switch (event.type) {
						case "stream-start":
							ensureStartPushed(stream, output, state);
							break;
						case "text-start":
							ensureStartPushed(stream, output, state);
							blocks.push({ type: "text", text: "", streamId: event.id } as TextContent & { streamId: string });
							stream.push({ type: "text_start", contentIndex: blocks.length - 1, partial: output });
							break;
						case "text-delta": {
							const index = getStreamBlockIndex(blocks, event.id);
							const block = index >= 0 ? blocks[index] : undefined;
							if (!block || block.type !== "text") break;
							block.text += event.delta;
							stream.push({ type: "text_delta", contentIndex: index, delta: event.delta, partial: output });
							break;
						}
						case "text-end": {
							const index = getStreamBlockIndex(blocks, event.id);
							const block = index >= 0 ? blocks[index] : undefined;
							if (!block || block.type !== "text") break;
							delete (block as TextContent & { streamId?: string }).streamId;
							stream.push({ type: "text_end", contentIndex: index, content: block.text, partial: output });
							break;
						}
						case "reasoning-start":
							ensureStartPushed(stream, output, state);
							blocks.push({
								type: "thinking",
								thinking: "",
								streamId: event.id,
							} as ThinkingContent & { streamId: string });
							stream.push({ type: "thinking_start", contentIndex: blocks.length - 1, partial: output });
							break;
						case "reasoning-delta": {
							const index = getStreamBlockIndex(blocks, event.id);
							const block = index >= 0 ? blocks[index] : undefined;
							if (!block || block.type !== "thinking") break;
							block.thinking += event.delta;
							stream.push({ type: "thinking_delta", contentIndex: index, delta: event.delta, partial: output });
							break;
						}
						case "reasoning-end": {
							const index = getStreamBlockIndex(blocks, event.id);
							const block = index >= 0 ? blocks[index] : undefined;
							if (!block || block.type !== "thinking") break;
							delete (block as ThinkingContent & { streamId?: string }).streamId;
							stream.push({
								type: "thinking_end",
								contentIndex: index,
								content: block.thinking,
								partial: output,
							});
							break;
						}
						case "tool-input-start":
							ensureStartPushed(stream, output, state);
							blocks.push({
								type: "toolCall",
								id: event.id,
								name: event.toolName,
								arguments: {},
								streamId: event.id,
								partialJson: "",
								ended: false,
							});
							stream.push({ type: "toolcall_start", contentIndex: blocks.length - 1, partial: output });
							break;
						case "tool-input-delta": {
							const index = getStreamBlockIndex(blocks, event.id);
							const block = index >= 0 ? blocks[index] : undefined;
							if (!block || block.type !== "toolCall") break;
							block.partialJson += event.delta;
							try {
								block.arguments = block.partialJson ? JSON.parse(block.partialJson) : {};
							} catch {
								block.arguments = {};
							}
							stream.push({ type: "toolcall_delta", contentIndex: index, delta: event.delta, partial: output });
							break;
						}
						case "tool-call": {
							let index = getStreamBlockIndex(blocks, event.toolCallId);
							if (index === -1) {
								ensureStartPushed(stream, output, state);
								blocks.push({
									type: "toolCall",
									id: event.toolCallId,
									name: event.toolName,
									arguments: {},
									streamId: event.toolCallId,
									partialJson: event.input,
									ended: false,
								});
								index = blocks.length - 1;
								stream.push({ type: "toolcall_start", contentIndex: index, partial: output });
							}

							const block = blocks[index];
							if (!block || block.type !== "toolCall") break;
							block.name = event.toolName;
							block.partialJson = event.input;
							try {
								block.arguments = event.input ? JSON.parse(event.input) : {};
							} catch {
								block.arguments = {};
							}
							if (!block.ended) {
								block.ended = true;
								delete block.partialJson;
								delete block.streamId;
								stream.push({ type: "toolcall_end", contentIndex: index, toolCall: block, partial: output });
							}
							break;
						}
						case "finish":
							output.stopReason = mapStopReason(event.finishReason);
							output.usage.input = event.usage.inputTokens ?? 0;
							output.usage.output = event.usage.outputTokens ?? 0;
							output.usage.cacheRead = event.usage.cachedInputTokens ?? 0;
							output.usage.cacheWrite = 0;
							output.usage.totalTokens =
								event.usage.totalTokens ??
								output.usage.input + output.usage.output + output.usage.cacheRead + output.usage.cacheWrite;
							calculateCost(model, output.usage);
							break;
						case "error":
							throw event.error;
						case "tool-result":
						case "tool-input-end":
						case "file":
						case "raw":
						case "response-metadata":
						case "source":
							break;
					}
				}

				ensureStartPushed(stream, output, state);
				closePendingToolCalls(stream, output, blocks);
				stream.push({ type: "done", reason: output.stopReason as "stop" | "length" | "toolUse", message: output });
				stream.end();
			} catch (error) {
				closePendingToolCalls(stream, output, blocks);
				output.stopReason = streamOptions?.signal?.aborted ? "aborted" : "error";
				output.errorMessage = error instanceof Error ? error.message : JSON.stringify(error);
				stream.push({ type: "error", reason: output.stopReason, error: output });
				stream.end();
			}
		})();

		return stream;
	};
}

export function createAiSdkApiProvider(options: {
	resolveProvider: (options: AiSdkProviderFactoryOptions) => AiSdkProvider;
	providerOptions?: SharedV2ProviderOptions | AiSdkProviderOptionsFactory;
}) {
	const streamSimple = createAiSdkStreamSimple(options);
	return {
		stream: streamSimple,
		streamSimple,
	};
}
