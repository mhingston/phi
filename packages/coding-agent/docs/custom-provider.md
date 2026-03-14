# Custom Providers

Extensions can register custom model providers via `pi.registerProvider()`. This enables:

- **Proxies** - Route requests through corporate proxies or API gateways
- **Custom endpoints** - Use self-hosted or private model deployments
- **OAuth/SSO** - Add authentication flows for enterprise providers
- **AI SDK providers** - Bring your own Vercel AI SDK provider and let pi infer the transport
- **Custom APIs** - Implement streaming for non-standard LLM APIs when AI SDK is not enough

The preferred path is now `sdk`. If your provider already has an AI SDK provider package, pi can use it directly and handle streaming automatically. You only need the legacy `api` and `streamSimple` fields for providers with no suitable AI SDK integration or for heavily custom protocols.

## Example Extensions

See these complete provider examples:

- [`examples/extensions/custom-provider-anthropic/`](../examples/extensions/custom-provider-anthropic/)
- [`examples/extensions/custom-provider-gitlab-duo/`](../examples/extensions/custom-provider-gitlab-duo/)
- [`examples/extensions/custom-provider-qwen-cli/`](../examples/extensions/custom-provider-qwen-cli/)

## Table of Contents

- [Example Extensions](#example-extensions)
- [Quick Reference](#quick-reference)
- [AI SDK Providers](#ai-sdk-providers)
- [Override Existing Provider](#override-existing-provider)
- [Register New Provider](#register-new-provider)
- [Unregister Provider](#unregister-provider)
- [OAuth Support](#oauth-support)
- [Custom Streaming API](#custom-streaming-api)
- [Testing Your Implementation](#testing-your-implementation)
- [Config Reference](#config-reference)
- [Model Definition Reference](#model-definition-reference)

## Quick Reference

```typescript
import type { ExtensionAPI } from "@mariozechner/pi-coding-agent";
import { createOpenAI } from "@ai-sdk/openai";

export default function (pi: ExtensionAPI) {
  // Override an existing provider with an AI SDK provider factory.
  // pi still resolves apiKey/baseUrl/headers for you.
  void pi.registerProvider("openai", {
    apiKey: "OPENAI_API_KEY",
    sdk: {
      provider: ({ apiKey, baseUrl, headers }) =>
        createOpenAI({ apiKey, baseURL: baseUrl, headers })
    }
  });

  // Register a brand-new provider backed by AI SDK
  void pi.registerProvider("my-provider", {
    apiKey: "MY_API_KEY",
    sdk: {
      provider: ({ apiKey, baseUrl, headers }) =>
        createOpenAI({ apiKey, baseURL: baseUrl, headers }),
      models: [
        {
          id: "my-model",
          name: "My Model",
          reasoning: false,
          input: ["text", "image"],
          cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
          contextWindow: 128000,
          maxTokens: 4096
        }
      ]
    }
  });
}
```

## AI SDK Providers

Bring your own AI SDK provider package in the extension’s `package.json`, for example `@ai-sdk/openai`, `@ai-sdk/anthropic`, or another provider package that returns an AI SDK `Provider`.

```typescript
import type { ExtensionAPI } from "@mariozechner/pi-coding-agent";
import { createAnthropic } from "@ai-sdk/anthropic";

export default function (pi: ExtensionAPI) {
  void pi.registerProvider("anthropic", {
    apiKey: "ANTHROPIC_API_KEY",
    baseUrl: "ANTHROPIC_BASE_URL",
    headers: {
      "X-Proxy-Token": "CORP_PROXY_TOKEN"
    },
    sdk: {
      provider: ({ apiKey, baseUrl, headers }) =>
        createAnthropic({
          apiKey,
          baseURL: baseUrl,
          headers
        })
    }
  });
}
```

How model inference works:

- If you override an existing pi provider name such as `openai` or `anthropic` and do not provide `sdk.models`, pi keeps that provider’s built-in model catalog.
- If the AI SDK provider exposes model discovery, pi can fetch models dynamically with `sdk.discoverModels: true`.
- Otherwise, provide `sdk.models` explicitly.

For providers that manage auth internally, you can also pass a fully constructed AI SDK provider instance instead of a factory:

```typescript
import { createOpenAI } from "@ai-sdk/openai";

export default function (pi: ExtensionAPI) {
  const provider = createOpenAI({ apiKey: process.env.OPENAI_API_KEY! });

  void pi.registerProvider("openai", {
    sdk: { provider }
  });
}
```

That works, but pi can no longer resolve credentials or endpoint settings for you. Using a factory is the better default.

## Override Existing Provider

The simplest use case: redirect an existing provider through a proxy.

```typescript
import { createOpenAI } from "@ai-sdk/openai";
import { createAnthropic } from "@ai-sdk/anthropic";

// All Anthropic requests now go through your proxy
void pi.registerProvider("anthropic", {
  apiKey: "ANTHROPIC_API_KEY",
  baseUrl: "https://proxy.example.com",
  sdk: {
    provider: ({ apiKey, baseUrl, headers }) =>
      createAnthropic({ apiKey, baseURL: baseUrl, headers })
  }
});

// Add custom headers to OpenAI requests
void pi.registerProvider("openai", {
  apiKey: "OPENAI_API_KEY",
  headers: {
    "X-Custom-Header": "value"
  },
  sdk: {
    provider: ({ apiKey, baseUrl, headers }) =>
      createOpenAI({ apiKey, baseURL: baseUrl, headers })
  }
});
```

When you override a known provider and omit `sdk.models`, pi preserves the existing model catalog and swaps in your AI SDK provider transport.

## Register New Provider

To add a completely new provider, either specify `sdk.models`, or enable discovery if the provider supports it.

```typescript
import { createOpenAI } from "@ai-sdk/openai";

void pi.registerProvider("my-llm", {
  apiKey: "MY_LLM_API_KEY",
  baseUrl: "https://api.my-llm.com/v1",
  sdk: {
    provider: ({ apiKey, baseUrl, headers }) =>
      createOpenAI({ apiKey, baseURL: baseUrl, headers }),
    models: [
      {
        id: "my-llm-large",
        name: "My LLM Large",
        reasoning: true,
        input: ["text", "image"],
        cost: {
          input: 3.0,
          output: 15.0,
          cacheRead: 0.3,
          cacheWrite: 3.75
        },
        contextWindow: 200000,
        maxTokens: 16384
      }
    ]
  }
});
```

When `sdk.models` is provided, it **replaces** all existing models for that provider.

### Discovery-capable providers

Some AI SDK providers can list models. For those, enable discovery:

```typescript
import { createGateway } from "@ai-sdk/gateway";

void pi.registerProvider("vercel-ai-gateway", {
  apiKey: "AI_GATEWAY_API_KEY",
  sdk: {
    provider: ({ apiKey }) => createGateway({ apiKey }),
    discoverModels: true
  }
});
```

Today this is provider-specific, not universal. If discovery is not available, provide `sdk.models`.

## Unregister Provider

Use `pi.unregisterProvider(name)` to remove a provider that was previously registered via `pi.registerProvider(name, ...)`:

```typescript
// Register
void pi.registerProvider("my-llm", {
  baseUrl: "https://api.my-llm.com/v1",
  apiKey: "MY_LLM_API_KEY",
  api: "openai-completions",
  models: [
    {
      id: "my-llm-large",
      name: "My LLM Large",
      reasoning: true,
      input: ["text", "image"],
      cost: { input: 3.0, output: 15.0, cacheRead: 0.3, cacheWrite: 3.75 },
      contextWindow: 200000,
      maxTokens: 16384
    }
  ]
});

// Later, remove it
pi.unregisterProvider("my-llm");
```

Unregistering removes that provider's dynamic models, API key fallback, OAuth provider registration, and custom stream handler registrations. Any built-in models or provider behavior that were overridden are restored.

Calls made after the initial extension load phase are applied immediately, so no `/reload` is required.

### API Types

The `api` field determines which streaming implementation is used:

| API | Use for |
|-----|---------|
| `anthropic-messages` | Anthropic Claude API and compatibles |
| `openai-completions` | OpenAI Chat Completions API and compatibles |
| `openai-responses` | OpenAI Responses API |
| `azure-openai-responses` | Azure OpenAI Responses API |
| `openai-codex-responses` | OpenAI Codex Responses API |
| `mistral-conversations` | Mistral SDK Conversations/Chat streaming |
| `google-generative-ai` | Google Generative AI API |
| `google-gemini-cli` | Google Cloud Code Assist API |
| `google-vertex` | Google Vertex AI API |
| `bedrock-converse-stream` | Amazon Bedrock Converse API |

Most OpenAI-compatible providers work with `openai-completions`. Use `compat` for quirks:

```typescript
models: [{
  id: "custom-model",
  // ...
  compat: {
    supportsDeveloperRole: false,      // use "system" instead of "developer"
    supportsReasoningEffort: true,
    reasoningEffortMap: {              // map pi-ai levels to provider values
      minimal: "default",
      low: "default",
      medium: "default",
      high: "default",
      xhigh: "default"
    },
      maxTokensField: "max_tokens",      // instead of "max_completion_tokens"
      requiresToolResultName: true,      // tool results need name field
      thinkingFormat: "qwen"             // uses enable_thinking: true
    }
  }]
```

> Migration note: Mistral moved from `openai-completions` to `mistral-conversations`.
> Use `mistral-conversations` for native Mistral models.
> If you intentionally route Mistral-compatible/custom endpoints through `openai-completions`, set `compat` flags explicitly as needed.

### Auth Header

If your provider expects `Authorization: Bearer <key>` but doesn't use a standard API, set `authHeader: true`:

```typescript
pi.registerProvider("custom-api", {
  baseUrl: "https://api.example.com",
  apiKey: "MY_API_KEY",
  authHeader: true,  // adds Authorization: Bearer header
  api: "openai-completions",
  models: [...]
});
```

## OAuth Support

Add OAuth/SSO authentication that integrates with `/login`:

```typescript
import type { OAuthCredentials, OAuthLoginCallbacks } from "@mariozechner/pi-ai";

pi.registerProvider("corporate-ai", {
  baseUrl: "https://ai.corp.com/v1",
  api: "openai-responses",
  models: [...],
  oauth: {
    name: "Corporate AI (SSO)",

    async login(callbacks: OAuthLoginCallbacks): Promise<OAuthCredentials> {
      // Option 1: Browser-based OAuth
      callbacks.onAuth({ url: "https://sso.corp.com/authorize?..." });

      // Option 2: Device code flow
      callbacks.onDeviceCode({
        userCode: "ABCD-1234",
        verificationUri: "https://sso.corp.com/device"
      });

      // Option 3: Prompt for token/code
      const code = await callbacks.onPrompt({ message: "Enter SSO code:" });

      // Exchange for tokens (your implementation)
      const tokens = await exchangeCodeForTokens(code);

      return {
        refresh: tokens.refreshToken,
        access: tokens.accessToken,
        expires: Date.now() + tokens.expiresIn * 1000
      };
    },

    async refreshToken(credentials: OAuthCredentials): Promise<OAuthCredentials> {
      const tokens = await refreshAccessToken(credentials.refresh);
      return {
        refresh: tokens.refreshToken ?? credentials.refresh,
        access: tokens.accessToken,
        expires: Date.now() + tokens.expiresIn * 1000
      };
    },

    getApiKey(credentials: OAuthCredentials): string {
      return credentials.access;
    },

    // Optional: modify models based on user's subscription
    modifyModels(models, credentials) {
      const region = decodeRegionFromToken(credentials.access);
      return models.map(m => ({
        ...m,
        baseUrl: `https://${region}.ai.corp.com/v1`
      }));
    }
  }
});
```

After registration, users can authenticate via `/login corporate-ai`.

### OAuthLoginCallbacks

The `callbacks` object provides three ways to authenticate:

```typescript
interface OAuthLoginCallbacks {
  // Open URL in browser (for OAuth redirects)
  onAuth(params: { url: string }): void;

  // Show device code (for device authorization flow)
  onDeviceCode(params: { userCode: string; verificationUri: string }): void;

  // Prompt user for input (for manual token entry)
  onPrompt(params: { message: string }): Promise<string>;
}
```

### OAuthCredentials

Credentials are persisted in `~/.pi/agent/auth.json`:

```typescript
interface OAuthCredentials {
  refresh: string;   // Refresh token (for refreshToken())
  access: string;    // Access token (returned by getApiKey())
  expires: number;   // Expiration timestamp in milliseconds
}
```

## Custom Streaming API

For providers with non-standard APIs, implement `streamSimple`. This is the escape hatch when an AI SDK provider is not available or cannot express the behavior you need. Study the existing provider implementations before writing your own:

**Reference implementations:**
- [anthropic.ts](https://github.com/badlogic/pi-mono/blob/main/packages/ai/src/providers/anthropic.ts) - Anthropic Messages API
- [mistral.ts](https://github.com/badlogic/pi-mono/blob/main/packages/ai/src/providers/mistral.ts) - Mistral Conversations API
- [openai-completions.ts](https://github.com/badlogic/pi-mono/blob/main/packages/ai/src/providers/openai-completions.ts) - OpenAI Chat Completions
- [openai-responses.ts](https://github.com/badlogic/pi-mono/blob/main/packages/ai/src/providers/openai-responses.ts) - OpenAI Responses API
- [google.ts](https://github.com/badlogic/pi-mono/blob/main/packages/ai/src/providers/google.ts) - Google Generative AI
- [amazon-bedrock.ts](https://github.com/badlogic/pi-mono/blob/main/packages/ai/src/providers/amazon-bedrock.ts) - AWS Bedrock

### Stream Pattern

All providers follow the same pattern:

```typescript
import {
  type AssistantMessage,
  type AssistantMessageEventStream,
  type Context,
  type Model,
  type SimpleStreamOptions,
  calculateCost,
  createAssistantMessageEventStream,
} from "@mariozechner/pi-ai";

function streamMyProvider(
  model: Model<any>,
  context: Context,
  options?: SimpleStreamOptions
): AssistantMessageEventStream {
  const stream = createAssistantMessageEventStream();

  (async () => {
    // Initialize output message
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

    try {
      // Push start event
      stream.push({ type: "start", partial: output });

      // Make API request and process response...
      // Push content events as they arrive...

      // Push done event
      stream.push({
        type: "done",
        reason: output.stopReason as "stop" | "length" | "toolUse",
        message: output
      });
      stream.end();
    } catch (error) {
      output.stopReason = options?.signal?.aborted ? "aborted" : "error";
      output.errorMessage = error instanceof Error ? error.message : String(error);
      stream.push({ type: "error", reason: output.stopReason, error: output });
      stream.end();
    }
  })();

  return stream;
}
```

### Event Types

Push events via `stream.push()` in this order:

1. `{ type: "start", partial: output }` - Stream started

2. Content events (repeatable, track `contentIndex` for each block):
   - `{ type: "text_start", contentIndex, partial }` - Text block started
   - `{ type: "text_delta", contentIndex, delta, partial }` - Text chunk
   - `{ type: "text_end", contentIndex, content, partial }` - Text block ended
   - `{ type: "thinking_start", contentIndex, partial }` - Thinking started
   - `{ type: "thinking_delta", contentIndex, delta, partial }` - Thinking chunk
   - `{ type: "thinking_end", contentIndex, content, partial }` - Thinking ended
   - `{ type: "toolcall_start", contentIndex, partial }` - Tool call started
   - `{ type: "toolcall_delta", contentIndex, delta, partial }` - Tool call JSON chunk
   - `{ type: "toolcall_end", contentIndex, toolCall, partial }` - Tool call ended

3. `{ type: "done", reason, message }` or `{ type: "error", reason, error }` - Stream ended

The `partial` field in each event contains the current `AssistantMessage` state. Update `output.content` as you receive data, then include `output` as the `partial`.

### Content Blocks

Add content blocks to `output.content` as they arrive:

```typescript
// Text block
output.content.push({ type: "text", text: "" });
stream.push({ type: "text_start", contentIndex: output.content.length - 1, partial: output });

// As text arrives
const block = output.content[contentIndex];
if (block.type === "text") {
  block.text += delta;
  stream.push({ type: "text_delta", contentIndex, delta, partial: output });
}

// When block completes
stream.push({ type: "text_end", contentIndex, content: block.text, partial: output });
```

### Tool Calls

Tool calls require accumulating JSON and parsing:

```typescript
// Start tool call
output.content.push({
  type: "toolCall",
  id: toolCallId,
  name: toolName,
  arguments: {}
});
stream.push({ type: "toolcall_start", contentIndex: output.content.length - 1, partial: output });

// Accumulate JSON
let partialJson = "";
partialJson += jsonDelta;
try {
  block.arguments = JSON.parse(partialJson);
} catch {}
stream.push({ type: "toolcall_delta", contentIndex, delta: jsonDelta, partial: output });

// Complete
stream.push({
  type: "toolcall_end",
  contentIndex,
  toolCall: { type: "toolCall", id, name, arguments: block.arguments },
  partial: output
});
```

### Usage and Cost

Update usage from API response and calculate cost:

```typescript
output.usage.input = response.usage.input_tokens;
output.usage.output = response.usage.output_tokens;
output.usage.cacheRead = response.usage.cache_read_tokens ?? 0;
output.usage.cacheWrite = response.usage.cache_write_tokens ?? 0;
output.usage.totalTokens = output.usage.input + output.usage.output +
                           output.usage.cacheRead + output.usage.cacheWrite;
calculateCost(model, output.usage);
```

### Registration

Register your stream function:

```typescript
pi.registerProvider("my-provider", {
  baseUrl: "https://api.example.com",
  apiKey: "MY_API_KEY",
  api: "my-custom-api",
  models: [...],
  streamSimple: streamMyProvider
});
```

## Testing Your Implementation

Test your provider against the same test suites used by built-in providers. Copy and adapt these test files from [packages/ai/test/](https://github.com/badlogic/pi-mono/tree/main/packages/ai/test):

| Test | Purpose |
|------|---------|
| `stream.test.ts` | Basic streaming, text output |
| `tokens.test.ts` | Token counting and usage |
| `abort.test.ts` | AbortSignal handling |
| `empty.test.ts` | Empty/minimal responses |
| `context-overflow.test.ts` | Context window limits |
| `image-limits.test.ts` | Image input handling |
| `unicode-surrogate.test.ts` | Unicode edge cases |
| `tool-call-without-result.test.ts` | Tool call edge cases |
| `image-tool-result.test.ts` | Images in tool results |
| `total-tokens.test.ts` | Total token calculation |
| `cross-provider-handoff.test.ts` | Context handoff between providers |

Run tests with your provider/model pairs to verify compatibility.

## Config Reference

```typescript
interface ProviderConfig {
  /** API endpoint URL. Required when defining models. */
  baseUrl?: string;

  /** API key or environment variable name. Required when defining models (unless oauth). */
  apiKey?: string;

  /** AI SDK provider config. Preferred over low-level api/streamSimple. */
  sdk?: {
    provider: Provider | ((options: {
      apiKey?: string;
      baseUrl?: string;
      headers?: Record<string, string>;
    }) => Provider);
    discoverModels?: boolean;
    models?: Array<string | AiSdkModelConfig>;
    providerOptions?: SharedV2ProviderOptions | ((context) => SharedV2ProviderOptions | Promise<SharedV2ProviderOptions>);
  };

  /** API type for streaming. Required at provider or model level when defining models. */
  api?: Api;

  /** Custom streaming implementation for non-standard APIs. */
  streamSimple?: (
    model: Model<Api>,
    context: Context,
    options?: SimpleStreamOptions
  ) => AssistantMessageEventStream;

  /** Custom headers to include in requests. Values can be env var names. */
  headers?: Record<string, string>;

  /** If true, adds Authorization: Bearer header with the resolved API key. */
  authHeader?: boolean;

  /** Models to register. If provided, replaces all existing models for this provider. */
  models?: ProviderModelConfig[];

  /** OAuth provider for /login support. */
  oauth?: {
    name: string;
    login(callbacks: OAuthLoginCallbacks): Promise<OAuthCredentials>;
    refreshToken(credentials: OAuthCredentials): Promise<OAuthCredentials>;
    getApiKey(credentials: OAuthCredentials): string;
    modifyModels?(models: Model<Api>[], credentials: OAuthCredentials): Model<Api>[];
  };
}
```

## Model Definition Reference

```typescript
interface ProviderModelConfig {
  /** Model ID (e.g., "claude-sonnet-4-20250514"). */
  id: string;

  /** Display name (e.g., "Claude 4 Sonnet"). */
  name: string;

  /** API type override for this specific model. */
  api?: Api;

  /** Whether the model supports extended thinking. */
  reasoning: boolean;

  /** Supported input types. */
  input: ("text" | "image")[];

  /** Cost per million tokens (for usage tracking). */
  cost: {
    input: number;
    output: number;
    cacheRead: number;
    cacheWrite: number;
  };

  /** Maximum context window size in tokens. */
  contextWindow: number;

  /** Maximum output tokens. */
  maxTokens: number;

  /** Custom headers for this specific model. */
  headers?: Record<string, string>;

  /** OpenAI compatibility settings for openai-completions API. */
  compat?: {
    supportsStore?: boolean;
    supportsDeveloperRole?: boolean;
    supportsReasoningEffort?: boolean;
    reasoningEffortMap?: Partial<Record<"minimal" | "low" | "medium" | "high" | "xhigh", string>>;
    supportsUsageInStreaming?: boolean;
    maxTokensField?: "max_completion_tokens" | "max_tokens";
    requiresToolResultName?: boolean;
    requiresAssistantAfterToolResult?: boolean;
    requiresThinkingAsText?: boolean;
    thinkingFormat?: "openai" | "zai" | "qwen";
  };
}
```

When both `sdk` and legacy provider fields are present, `sdk` controls transport inference and model discovery. The legacy fields remain useful for auth, headers, OAuth, or as a fallback for models that still need explicit metadata.
