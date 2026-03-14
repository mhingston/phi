/**
 * Model registry - manages built-in and custom models, provides API key resolution.
 */

import {
	type Api,
	type AssistantMessageEventStream,
	type Context,
	getModels,
	getProviders,
	type KnownProvider,
	type Model,
	type OAuthProviderInterface,
	type OpenAICompletionsCompat,
	type OpenAIResponsesCompat,
	registerApiProvider,
	resetApiProviders,
	type SimpleStreamOptions,
} from "@mariozechner/pi-ai";
import { registerOAuthProvider, resetOAuthProviders } from "@mariozechner/pi-ai/oauth";
import { type Static, Type } from "@sinclair/typebox";
import AjvModule from "ajv";
import { existsSync, readFileSync } from "fs";
import { join } from "path";
import { getAgentDir } from "../config.js";
import {
	AI_SDK_INLINE_AUTH_TOKEN,
	type AiSdkModelConfig,
	type AiSdkProviderConfig,
	type AiSdkProviderFactoryOptions,
	createAiSdkApiProvider,
	discoverAiSdkModels,
	resolveAiSdkProvider,
} from "./ai-sdk-provider.js";
import type { AuthStorage } from "./auth-storage.js";
import { clearConfigValueCache, resolveConfigValue, resolveHeaders } from "./resolve-config-value.js";

const Ajv = (AjvModule as any).default || AjvModule;
const ajv = new Ajv();

// Schema for OpenRouter routing preferences
const OpenRouterRoutingSchema = Type.Object({
	only: Type.Optional(Type.Array(Type.String())),
	order: Type.Optional(Type.Array(Type.String())),
});

// Schema for Vercel AI Gateway routing preferences
const VercelGatewayRoutingSchema = Type.Object({
	only: Type.Optional(Type.Array(Type.String())),
	order: Type.Optional(Type.Array(Type.String())),
});

// Schema for OpenAI compatibility settings
const OpenAICompletionsCompatSchema = Type.Object({
	supportsStore: Type.Optional(Type.Boolean()),
	supportsDeveloperRole: Type.Optional(Type.Boolean()),
	supportsReasoningEffort: Type.Optional(Type.Boolean()),
	supportsUsageInStreaming: Type.Optional(Type.Boolean()),
	maxTokensField: Type.Optional(Type.Union([Type.Literal("max_completion_tokens"), Type.Literal("max_tokens")])),
	requiresToolResultName: Type.Optional(Type.Boolean()),
	requiresAssistantAfterToolResult: Type.Optional(Type.Boolean()),
	requiresThinkingAsText: Type.Optional(Type.Boolean()),
	requiresMistralToolIds: Type.Optional(Type.Boolean()),
	thinkingFormat: Type.Optional(Type.Union([Type.Literal("openai"), Type.Literal("zai"), Type.Literal("qwen")])),
	openRouterRouting: Type.Optional(OpenRouterRoutingSchema),
	vercelGatewayRouting: Type.Optional(VercelGatewayRoutingSchema),
});

const OpenAIResponsesCompatSchema = Type.Object({
	// Reserved for future use
});

const OpenAICompatSchema = Type.Union([OpenAICompletionsCompatSchema, OpenAIResponsesCompatSchema]);

// Schema for custom model definition
// Most fields are optional with sensible defaults for local models (Ollama, LM Studio, etc.)
const ModelDefinitionSchema = Type.Object({
	id: Type.String({ minLength: 1 }),
	name: Type.Optional(Type.String({ minLength: 1 })),
	api: Type.Optional(Type.String({ minLength: 1 })),
	baseUrl: Type.Optional(Type.String({ minLength: 1 })),
	reasoning: Type.Optional(Type.Boolean()),
	input: Type.Optional(Type.Array(Type.Union([Type.Literal("text"), Type.Literal("image")]))),
	cost: Type.Optional(
		Type.Object({
			input: Type.Number(),
			output: Type.Number(),
			cacheRead: Type.Number(),
			cacheWrite: Type.Number(),
		}),
	),
	contextWindow: Type.Optional(Type.Number()),
	maxTokens: Type.Optional(Type.Number()),
	headers: Type.Optional(Type.Record(Type.String(), Type.String())),
	compat: Type.Optional(OpenAICompatSchema),
});

// Schema for per-model overrides (all fields optional, merged with built-in model)
const ModelOverrideSchema = Type.Object({
	name: Type.Optional(Type.String({ minLength: 1 })),
	reasoning: Type.Optional(Type.Boolean()),
	input: Type.Optional(Type.Array(Type.Union([Type.Literal("text"), Type.Literal("image")]))),
	cost: Type.Optional(
		Type.Object({
			input: Type.Optional(Type.Number()),
			output: Type.Optional(Type.Number()),
			cacheRead: Type.Optional(Type.Number()),
			cacheWrite: Type.Optional(Type.Number()),
		}),
	),
	contextWindow: Type.Optional(Type.Number()),
	maxTokens: Type.Optional(Type.Number()),
	headers: Type.Optional(Type.Record(Type.String(), Type.String())),
	compat: Type.Optional(OpenAICompatSchema),
});

type ModelOverride = Static<typeof ModelOverrideSchema>;

const ProviderConfigSchema = Type.Object({
	baseUrl: Type.Optional(Type.String({ minLength: 1 })),
	apiKey: Type.Optional(Type.String({ minLength: 1 })),
	api: Type.Optional(Type.String({ minLength: 1 })),
	headers: Type.Optional(Type.Record(Type.String(), Type.String())),
	authHeader: Type.Optional(Type.Boolean()),
	models: Type.Optional(Type.Array(ModelDefinitionSchema)),
	modelOverrides: Type.Optional(Type.Record(Type.String(), ModelOverrideSchema)),
});

const ModelsConfigSchema = Type.Object({
	providers: Type.Record(Type.String(), ProviderConfigSchema),
});

ajv.addSchema(ModelsConfigSchema, "ModelsConfig");

type ModelsConfig = Static<typeof ModelsConfigSchema>;

/** Provider override config (baseUrl, headers, apiKey) without custom models */
interface ProviderOverride {
	baseUrl?: string;
	headers?: Record<string, string>;
	apiKey?: string;
}

/** Result of loading custom models from models.json */
interface CustomModelsResult {
	models: Model<Api>[];
	/** Providers with baseUrl/headers/apiKey overrides for built-in models */
	overrides: Map<string, ProviderOverride>;
	/** Per-model overrides: provider -> modelId -> override */
	modelOverrides: Map<string, Map<string, ModelOverride>>;
	error: string | undefined;
}

function emptyCustomModelsResult(error?: string): CustomModelsResult {
	return { models: [], overrides: new Map(), modelOverrides: new Map(), error };
}

function mergeCompat(
	baseCompat: Model<Api>["compat"],
	overrideCompat: ModelOverride["compat"],
): Model<Api>["compat"] | undefined {
	if (!overrideCompat) return baseCompat;

	const base = baseCompat as OpenAICompletionsCompat | OpenAIResponsesCompat | undefined;
	const override = overrideCompat as OpenAICompletionsCompat | OpenAIResponsesCompat;
	const merged = { ...base, ...override } as OpenAICompletionsCompat | OpenAIResponsesCompat;

	const baseCompletions = base as OpenAICompletionsCompat | undefined;
	const overrideCompletions = override as OpenAICompletionsCompat;
	const mergedCompletions = merged as OpenAICompletionsCompat;

	if (baseCompletions?.openRouterRouting || overrideCompletions.openRouterRouting) {
		mergedCompletions.openRouterRouting = {
			...baseCompletions?.openRouterRouting,
			...overrideCompletions.openRouterRouting,
		};
	}

	if (baseCompletions?.vercelGatewayRouting || overrideCompletions.vercelGatewayRouting) {
		mergedCompletions.vercelGatewayRouting = {
			...baseCompletions?.vercelGatewayRouting,
			...overrideCompletions.vercelGatewayRouting,
		};
	}

	return merged as Model<Api>["compat"];
}

/**
 * Deep merge a model override into a model.
 * Handles nested objects (cost, compat) by merging rather than replacing.
 */
function applyModelOverride(model: Model<Api>, override: ModelOverride): Model<Api> {
	const result = { ...model };

	// Simple field overrides
	if (override.name !== undefined) result.name = override.name;
	if (override.reasoning !== undefined) result.reasoning = override.reasoning;
	if (override.input !== undefined) result.input = override.input as ("text" | "image")[];
	if (override.contextWindow !== undefined) result.contextWindow = override.contextWindow;
	if (override.maxTokens !== undefined) result.maxTokens = override.maxTokens;

	// Merge cost (partial override)
	if (override.cost) {
		result.cost = {
			input: override.cost.input ?? model.cost.input,
			output: override.cost.output ?? model.cost.output,
			cacheRead: override.cost.cacheRead ?? model.cost.cacheRead,
			cacheWrite: override.cost.cacheWrite ?? model.cost.cacheWrite,
		};
	}

	// Merge headers
	if (override.headers) {
		const resolvedHeaders = resolveHeaders(override.headers);
		result.headers = resolvedHeaders ? { ...model.headers, ...resolvedHeaders } : model.headers;
	}

	// Deep merge compat
	result.compat = mergeCompat(model.compat, override.compat);

	return result;
}

/** Clear the config value command cache. Exported for testing. */
export const clearApiKeyCache = clearConfigValueCache;

/**
 * Model registry - loads and manages models, resolves API keys via AuthStorage.
 */
export class ModelRegistry {
	private models: Model<Api>[] = [];
	private customProviderApiKeys: Map<string, string> = new Map();
	private registeredProviders: Map<string, ProviderConfigInput> = new Map();
	private inlineAuthProviders: Set<string> = new Set();
	private loadError: string | undefined = undefined;

	constructor(
		readonly authStorage: AuthStorage,
		private modelsJsonPath: string | undefined = join(getAgentDir(), "models.json"),
	) {
		// Set up fallback resolver for custom provider API keys
		this.authStorage.setFallbackResolver((provider) => {
			const keyConfig = this.customProviderApiKeys.get(provider);
			if (keyConfig) {
				return resolveConfigValue(keyConfig);
			}
			return undefined;
		});

		// Load models
		this.loadModels();
	}

	/**
	 * Reload models from disk (built-in + custom from models.json).
	 */
	refresh(): void {
		this.customProviderApiKeys.clear();
		this.inlineAuthProviders.clear();
		this.loadError = undefined;

		// Ensure dynamic API/OAuth registrations are rebuilt from current provider state.
		resetApiProviders();
		resetOAuthProviders();

		this.loadModels();

		for (const [providerName, config] of this.registeredProviders.entries()) {
			this.applyProviderConfig(providerName, config);
		}
	}

	/**
	 * Get any error from loading models.json (undefined if no error).
	 */
	getError(): string | undefined {
		return this.loadError;
	}

	private loadModels(): void {
		// Load custom models and overrides from models.json
		const {
			models: customModels,
			overrides,
			modelOverrides,
			error,
		} = this.modelsJsonPath ? this.loadCustomModels(this.modelsJsonPath) : emptyCustomModelsResult();

		if (error) {
			this.loadError = error;
			// Keep built-in models even if custom models failed to load
		}

		const builtInModels = this.loadBuiltInModels(overrides, modelOverrides);
		let combined = this.mergeCustomModels(builtInModels, customModels);

		// Let OAuth providers modify their models (e.g., update baseUrl)
		for (const oauthProvider of this.authStorage.getOAuthProviders()) {
			const cred = this.authStorage.get(oauthProvider.id);
			if (cred?.type === "oauth" && oauthProvider.modifyModels) {
				combined = oauthProvider.modifyModels(combined, cred);
			}
		}

		this.models = combined;
	}

	/** Load built-in models and apply provider/model overrides */
	private loadBuiltInModels(
		overrides: Map<string, ProviderOverride>,
		modelOverrides: Map<string, Map<string, ModelOverride>>,
	): Model<Api>[] {
		return getProviders().flatMap((provider: string) => {
			const models = getModels(provider as KnownProvider) as Model<Api>[];
			const providerOverride = overrides.get(provider);
			const perModelOverrides = modelOverrides.get(provider);

			return models.map((m) => {
				let model = m;

				// Apply provider-level baseUrl/headers override
				if (providerOverride) {
					const resolvedHeaders = resolveHeaders(providerOverride.headers);
					model = {
						...model,
						baseUrl: providerOverride.baseUrl ?? model.baseUrl,
						headers: resolvedHeaders ? { ...model.headers, ...resolvedHeaders } : model.headers,
					};
				}

				// Apply per-model override
				const modelOverride = perModelOverrides?.get(m.id);
				if (modelOverride) {
					model = applyModelOverride(model, modelOverride);
				}

				return model;
			});
		});
	}

	/** Merge custom models into built-in list by provider+id (custom wins on conflicts). */
	private mergeCustomModels(builtInModels: Model<Api>[], customModels: Model<Api>[]): Model<Api>[] {
		const merged = [...builtInModels];
		for (const customModel of customModels) {
			const existingIndex = merged.findIndex((m) => m.provider === customModel.provider && m.id === customModel.id);
			if (existingIndex >= 0) {
				merged[existingIndex] = customModel;
			} else {
				merged.push(customModel);
			}
		}
		return merged;
	}

	private loadCustomModels(modelsJsonPath: string): CustomModelsResult {
		if (!existsSync(modelsJsonPath)) {
			return emptyCustomModelsResult();
		}

		try {
			const content = readFileSync(modelsJsonPath, "utf-8");
			const config: ModelsConfig = JSON.parse(content);

			// Validate schema
			const validate = ajv.getSchema("ModelsConfig")!;
			if (!validate(config)) {
				const errors =
					validate.errors?.map((e: any) => `  - ${e.instancePath || "root"}: ${e.message}`).join("\n") ||
					"Unknown schema error";
				return emptyCustomModelsResult(`Invalid models.json schema:\n${errors}\n\nFile: ${modelsJsonPath}`);
			}

			// Additional validation
			this.validateConfig(config);

			const overrides = new Map<string, ProviderOverride>();
			const modelOverrides = new Map<string, Map<string, ModelOverride>>();

			for (const [providerName, providerConfig] of Object.entries(config.providers)) {
				// Apply provider-level baseUrl/headers/apiKey override to built-in models when configured.
				if (providerConfig.baseUrl || providerConfig.headers || providerConfig.apiKey) {
					overrides.set(providerName, {
						baseUrl: providerConfig.baseUrl,
						headers: providerConfig.headers,
						apiKey: providerConfig.apiKey,
					});
				}

				// Store API key for fallback resolver.
				if (providerConfig.apiKey) {
					this.customProviderApiKeys.set(providerName, providerConfig.apiKey);
				}

				if (providerConfig.modelOverrides) {
					modelOverrides.set(providerName, new Map(Object.entries(providerConfig.modelOverrides)));
				}
			}

			return { models: this.parseModels(config), overrides, modelOverrides, error: undefined };
		} catch (error) {
			if (error instanceof SyntaxError) {
				return emptyCustomModelsResult(`Failed to parse models.json: ${error.message}\n\nFile: ${modelsJsonPath}`);
			}
			return emptyCustomModelsResult(
				`Failed to load models.json: ${error instanceof Error ? error.message : error}\n\nFile: ${modelsJsonPath}`,
			);
		}
	}

	private validateConfig(config: ModelsConfig): void {
		for (const [providerName, providerConfig] of Object.entries(config.providers)) {
			const hasProviderApi = !!providerConfig.api;
			const models = providerConfig.models ?? [];
			const hasModelOverrides =
				providerConfig.modelOverrides && Object.keys(providerConfig.modelOverrides).length > 0;

			if (models.length === 0) {
				// Override-only config: needs baseUrl OR modelOverrides (or both)
				if (!providerConfig.baseUrl && !hasModelOverrides) {
					throw new Error(`Provider ${providerName}: must specify "baseUrl", "modelOverrides", or "models".`);
				}
			} else {
				// Custom models are merged into provider models and require endpoint + auth.
				if (!providerConfig.baseUrl) {
					throw new Error(`Provider ${providerName}: "baseUrl" is required when defining custom models.`);
				}
				if (!providerConfig.apiKey) {
					throw new Error(`Provider ${providerName}: "apiKey" is required when defining custom models.`);
				}
			}

			for (const modelDef of models) {
				const hasModelApi = !!modelDef.api;

				if (!hasProviderApi && !hasModelApi) {
					throw new Error(
						`Provider ${providerName}, model ${modelDef.id}: no "api" specified. Set at provider or model level.`,
					);
				}

				if (!modelDef.id) throw new Error(`Provider ${providerName}: model missing "id"`);
				// Validate contextWindow/maxTokens only if provided (they have defaults)
				if (modelDef.contextWindow !== undefined && modelDef.contextWindow <= 0)
					throw new Error(`Provider ${providerName}, model ${modelDef.id}: invalid contextWindow`);
				if (modelDef.maxTokens !== undefined && modelDef.maxTokens <= 0)
					throw new Error(`Provider ${providerName}, model ${modelDef.id}: invalid maxTokens`);
			}
		}
	}

	private parseModels(config: ModelsConfig): Model<Api>[] {
		const models: Model<Api>[] = [];

		for (const [providerName, providerConfig] of Object.entries(config.providers)) {
			const modelDefs = providerConfig.models ?? [];
			if (modelDefs.length === 0) continue; // Override-only, no custom models

			// Store API key config for fallback resolver
			if (providerConfig.apiKey) {
				this.customProviderApiKeys.set(providerName, providerConfig.apiKey);
			}

			for (const modelDef of modelDefs) {
				const api = modelDef.api || providerConfig.api;
				if (!api) continue;

				// Merge headers: provider headers are base, model headers override
				// Resolve env vars and shell commands in header values
				const providerHeaders = resolveHeaders(providerConfig.headers);
				const modelHeaders = resolveHeaders(modelDef.headers);
				let headers = providerHeaders || modelHeaders ? { ...providerHeaders, ...modelHeaders } : undefined;

				// If authHeader is true, add Authorization header with resolved API key
				if (providerConfig.authHeader && providerConfig.apiKey) {
					const resolvedKey = resolveConfigValue(providerConfig.apiKey);
					if (resolvedKey) {
						headers = { ...headers, Authorization: `Bearer ${resolvedKey}` };
					}
				}

				// Provider baseUrl is required when custom models are defined.
				// Individual models can override it with modelDef.baseUrl.
				const defaultCost = { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 };
				models.push({
					id: modelDef.id,
					name: modelDef.name ?? modelDef.id,
					api: api as Api,
					provider: providerName,
					baseUrl: modelDef.baseUrl ?? providerConfig.baseUrl!,
					reasoning: modelDef.reasoning ?? false,
					input: (modelDef.input ?? ["text"]) as ("text" | "image")[],
					cost: modelDef.cost ?? defaultCost,
					contextWindow: modelDef.contextWindow ?? 128000,
					maxTokens: modelDef.maxTokens ?? 16384,
					headers,
					compat: modelDef.compat,
				} as Model<Api>);
			}
		}

		return models;
	}

	private resolveAiSdkFactoryOptions(config: ProviderConfigInput, apiKey?: string): AiSdkProviderFactoryOptions {
		return {
			apiKey: apiKey && apiKey !== AI_SDK_INLINE_AUTH_TOKEN ? apiKey : undefined,
			baseUrl: config.baseUrl,
			headers: resolveHeaders(config.headers),
		};
	}

	private getCatalogModel(
		providerName: string,
		modelId: string,
		sourceProvider?: string,
		sourceModelId?: string,
	): Model<Api> | undefined {
		const direct = this.models.find((model) => model.provider === providerName && model.id === modelId);
		if (direct) return direct;

		if (sourceProvider && sourceModelId) {
			const sourceMatch = this.models.find(
				(model) => model.provider === sourceProvider && model.id === sourceModelId,
			);
			if (sourceMatch) return sourceMatch;
		}

		return this.models.find((model) => model.id === modelId);
	}

	private buildAiSdkModel(
		providerName: string,
		api: Api,
		config: ProviderConfigInput,
		modelConfig: AiSdkModelConfig,
	): Model<Api> {
		const sourceModel = this.getCatalogModel(
			providerName,
			modelConfig.id,
			modelConfig.sourceProvider,
			modelConfig.sourceModelId,
		);
		const providerHeaders = resolveHeaders(config.headers);
		const modelHeaders = resolveHeaders(modelConfig.headers);
		const defaultCost = {
			input: 0,
			output: 0,
			cacheRead: 0,
			cacheWrite: 0,
		};

		return {
			id: modelConfig.id,
			name: modelConfig.name ?? sourceModel?.name ?? modelConfig.id,
			api,
			provider: providerName,
			baseUrl: config.baseUrl ?? sourceModel?.baseUrl ?? "",
			reasoning: modelConfig.reasoning ?? sourceModel?.reasoning ?? false,
			input: modelConfig.input ?? sourceModel?.input ?? ["text"],
			cost: {
				input: modelConfig.cost?.input ?? sourceModel?.cost.input ?? defaultCost.input,
				output: modelConfig.cost?.output ?? sourceModel?.cost.output ?? defaultCost.output,
				cacheRead: modelConfig.cost?.cacheRead ?? sourceModel?.cost.cacheRead ?? defaultCost.cacheRead,
				cacheWrite: modelConfig.cost?.cacheWrite ?? sourceModel?.cost.cacheWrite ?? defaultCost.cacheWrite,
			},
			contextWindow: modelConfig.contextWindow ?? sourceModel?.contextWindow ?? 128000,
			maxTokens: modelConfig.maxTokens ?? sourceModel?.maxTokens ?? 16384,
			headers:
				providerHeaders || modelHeaders || sourceModel?.headers
					? { ...sourceModel?.headers, ...providerHeaders, ...modelHeaders }
					: undefined,
			compat: modelConfig.compat ?? sourceModel?.compat,
		};
	}

	private async buildAiSdkModels(providerName: string, api: Api, config: ProviderConfigInput): Promise<Model<Api>[]> {
		const sdkConfig = config.sdk;
		if (!sdkConfig) return [];

		let modelConfigs: AiSdkModelConfig[] = [];
		if (sdkConfig.models && sdkConfig.models.length > 0) {
			const resolvedApiKey = config.apiKey ? resolveConfigValue(config.apiKey) : undefined;
			const provider = resolveAiSdkProvider(sdkConfig, this.resolveAiSdkFactoryOptions(config, resolvedApiKey));
			modelConfigs = sdkConfig.models.map((entry) => {
				if (typeof entry !== "string") return entry;
				try {
					const languageModel = provider.languageModel(entry) as import("@ai-sdk/provider").LanguageModelV2;
					return {
						id: entry,
						name: entry,
						sourceProvider: languageModel.provider,
						sourceModelId: languageModel.modelId,
					};
				} catch {
					return { id: entry, name: entry };
				}
			});
		} else if (sdkConfig.discoverModels !== false) {
			const resolvedApiKey = config.apiKey ? resolveConfigValue(config.apiKey) : undefined;
			modelConfigs = await discoverAiSdkModels(sdkConfig, this.resolveAiSdkFactoryOptions(config, resolvedApiKey));
		}

		if (modelConfigs.length === 0) {
			const existingProviderModels = this.models.filter((model) => model.provider === providerName);
			if (existingProviderModels.length > 0) {
				modelConfigs = existingProviderModels.map((model) => ({
					id: model.id,
					name: model.name,
					reasoning: model.reasoning,
					input: model.input,
					cost: model.cost,
					contextWindow: model.contextWindow,
					maxTokens: model.maxTokens,
					headers: model.headers,
					compat: model.compat as Model<Api>["compat"] & AiSdkModelConfig["compat"],
					sourceProvider: model.provider,
					sourceModelId: model.id,
				}));
			}
		}

		return modelConfigs.map((modelConfig) => this.buildAiSdkModel(providerName, api, config, modelConfig));
	}

	/**
	 * Get all models (built-in + custom).
	 * If models.json had errors, returns only built-in models.
	 */
	getAll(): Model<Api>[] {
		return this.models;
	}

	hasConfiguredAuthForProvider(provider: string): boolean {
		return this.inlineAuthProviders.has(provider) || this.authStorage.hasAuth(provider);
	}

	/**
	 * Get only models that have auth configured.
	 * This is a fast check that doesn't refresh OAuth tokens.
	 */
	getAvailable(): Model<Api>[] {
		return this.models.filter((m) => this.hasConfiguredAuthForProvider(m.provider));
	}

	/**
	 * Find a model by provider and ID.
	 */
	find(provider: string, modelId: string): Model<Api> | undefined {
		return this.models.find((m) => m.provider === provider && m.id === modelId);
	}

	/**
	 * Get API key for a model.
	 */
	async getApiKey(model: Model<Api>): Promise<string | undefined> {
		if (this.inlineAuthProviders.has(model.provider)) {
			return AI_SDK_INLINE_AUTH_TOKEN;
		}
		return this.authStorage.getApiKey(model.provider);
	}

	/**
	 * Get API key for a provider.
	 */
	async getApiKeyForProvider(provider: string): Promise<string | undefined> {
		if (this.inlineAuthProviders.has(provider)) {
			return AI_SDK_INLINE_AUTH_TOKEN;
		}
		return this.authStorage.getApiKey(provider);
	}

	/**
	 * Check if a model is using OAuth credentials (subscription).
	 */
	isUsingOAuth(model: Model<Api>): boolean {
		const cred = this.authStorage.get(model.provider);
		return cred?.type === "oauth";
	}

	/**
	 * Register a provider dynamically (from extensions).
	 *
	 * If provider has models: replaces all existing models for this provider.
	 * If provider has only baseUrl/headers: overrides existing models' URLs.
	 * If provider has oauth: registers OAuth provider for /login support.
	 */
	registerProvider(providerName: string, config: ProviderConfigInput): Promise<void> {
		this.registeredProviders.set(providerName, config);
		return this.applyProviderConfig(providerName, config);
	}

	/**
	 * Unregister a previously registered provider.
	 *
	 * Removes the provider from the registry and reloads models from disk so that
	 * built-in models overridden by this provider are restored to their original state.
	 * Also resets dynamic OAuth and API stream registrations before reapplying
	 * remaining dynamic providers.
	 * Has no effect if the provider was never registered.
	 */
	unregisterProvider(providerName: string): void {
		if (!this.registeredProviders.has(providerName)) return;
		this.registeredProviders.delete(providerName);
		this.customProviderApiKeys.delete(providerName);
		this.refresh();
	}

	private async applyProviderConfig(providerName: string, config: ProviderConfigInput): Promise<void> {
		// Register OAuth provider if provided
		if (config.oauth) {
			// Ensure the OAuth provider ID matches the provider name
			const oauthProvider: OAuthProviderInterface = {
				...config.oauth,
				id: providerName,
			};
			registerOAuthProvider(oauthProvider);
		}

		if (config.streamSimple) {
			if (!config.api) {
				throw new Error(`Provider ${providerName}: "api" is required when registering streamSimple.`);
			}
			const streamSimple = config.streamSimple;
			registerApiProvider(
				{
					api: config.api,
					stream: (model: Model<Api>, context: Context, options?: SimpleStreamOptions) =>
						streamSimple(model, context, options as SimpleStreamOptions),
					streamSimple,
				},
				`provider:${providerName}`,
			);
		}

		if (config.sdk) {
			const api = `ai-sdk:${providerName}` as Api;
			registerApiProvider(
				{
					api,
					...createAiSdkApiProvider({
						resolveProvider: (providerOptions) =>
							resolveAiSdkProvider(config.sdk!, {
								...this.resolveAiSdkFactoryOptions(config, providerOptions.apiKey),
								baseUrl: providerOptions.baseUrl ?? config.baseUrl,
								headers: resolveHeaders(config.headers)
									? { ...resolveHeaders(config.headers), ...providerOptions.headers }
									: providerOptions.headers,
							}),
						providerOptions: config.sdk.providerOptions,
					}),
				},
				`provider:${providerName}`,
			);

			if (!config.apiKey && !config.oauth) {
				this.inlineAuthProviders.add(providerName);
			}

			const explicitSdkModels = config.models?.length
				? config.models.map((modelDef) => ({
						id: modelDef.id,
						name: modelDef.name,
						reasoning: modelDef.reasoning,
						input: modelDef.input,
						cost: modelDef.cost,
						contextWindow: modelDef.contextWindow,
						maxTokens: modelDef.maxTokens,
						headers: modelDef.headers,
						compat: modelDef.compat as AiSdkModelConfig["compat"],
					}))
				: undefined;
			const resolvedSdkModels: Model<Api>[] = explicitSdkModels
				? explicitSdkModels.map((model) => this.buildAiSdkModel(providerName, api, config, model))
				: await this.buildAiSdkModels(providerName, api, config);

			this.models = this.models.filter((model) => model.provider !== providerName);
			if (resolvedSdkModels.length === 0) {
				throw new Error(
					`Provider ${providerName}: could not infer models from the AI SDK provider. Add sdk.models or register the provider under an existing built-in provider name.`,
				);
			}
			this.models.push(...resolvedSdkModels);
			return;
		}

		// Store API key for auth resolution
		if (config.apiKey) {
			this.customProviderApiKeys.set(providerName, config.apiKey);
		}

		if (config.models && config.models.length > 0) {
			// Full replacement: remove existing models for this provider
			this.models = this.models.filter((m) => m.provider !== providerName);

			// Validate required fields
			if (!config.baseUrl) {
				throw new Error(`Provider ${providerName}: "baseUrl" is required when defining models.`);
			}
			if (!config.apiKey && !config.oauth) {
				throw new Error(`Provider ${providerName}: "apiKey" or "oauth" is required when defining models.`);
			}

			// Parse and add new models
			for (const modelDef of config.models) {
				const api = modelDef.api || config.api;
				if (!api) {
					throw new Error(`Provider ${providerName}, model ${modelDef.id}: no "api" specified.`);
				}

				// Merge headers
				const providerHeaders = resolveHeaders(config.headers);
				const modelHeaders = resolveHeaders(modelDef.headers);
				let headers = providerHeaders || modelHeaders ? { ...providerHeaders, ...modelHeaders } : undefined;

				// If authHeader is true, add Authorization header
				if (config.authHeader && config.apiKey) {
					const resolvedKey = resolveConfigValue(config.apiKey);
					if (resolvedKey) {
						headers = { ...headers, Authorization: `Bearer ${resolvedKey}` };
					}
				}

				this.models.push({
					id: modelDef.id,
					name: modelDef.name,
					api: api as Api,
					provider: providerName,
					baseUrl: config.baseUrl,
					reasoning: modelDef.reasoning,
					input: modelDef.input as ("text" | "image")[],
					cost: modelDef.cost,
					contextWindow: modelDef.contextWindow,
					maxTokens: modelDef.maxTokens,
					headers,
					compat: modelDef.compat,
				} as Model<Api>);
			}

			// Apply OAuth modifyModels if credentials exist (e.g., to update baseUrl)
			if (config.oauth?.modifyModels) {
				const cred = this.authStorage.get(providerName);
				if (cred?.type === "oauth") {
					this.models = config.oauth.modifyModels(this.models, cred);
				}
			}
		} else if (config.baseUrl) {
			// Override-only: update baseUrl/headers for existing models
			const resolvedHeaders = resolveHeaders(config.headers);
			this.models = this.models.map((m) => {
				if (m.provider !== providerName) return m;
				return {
					...m,
					baseUrl: config.baseUrl ?? m.baseUrl,
					headers: resolvedHeaders ? { ...m.headers, ...resolvedHeaders } : m.headers,
				};
			});
		}
	}
}

/**
 * Input type for registerProvider API.
 */
export interface ProviderConfigInput {
	baseUrl?: string;
	apiKey?: string;
	sdk?: AiSdkProviderConfig;
	api?: Api;
	streamSimple?: (model: Model<Api>, context: Context, options?: SimpleStreamOptions) => AssistantMessageEventStream;
	headers?: Record<string, string>;
	authHeader?: boolean;
	/** OAuth provider for /login support */
	oauth?: Omit<OAuthProviderInterface, "id">;
	models?: Array<{
		id: string;
		name: string;
		api?: Api;
		baseUrl?: string;
		reasoning: boolean;
		input: ("text" | "image")[];
		cost: { input: number; output: number; cacheRead: number; cacheWrite: number };
		contextWindow: number;
		maxTokens: number;
		headers?: Record<string, string>;
		compat?: Model<Api>["compat"];
	}>;
}
