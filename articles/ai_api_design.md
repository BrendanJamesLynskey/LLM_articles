# AI API Design Patterns

*April 2026*

## 1. Introduction

The design of APIs for large language models has converged on a set of patterns that are now industry standard. The chat completions API, pioneered by OpenAI and adopted by virtually every LLM provider, defines how applications interact with language models: through messages with roles, streaming responses via server-sent events, structured output through JSON mode, and tool use through function calling schemas. These patterns are not arbitrary conventions—they reflect the deep structure of how language models work and how applications need to use them.

Understanding these API design patterns is essential for three audiences: application developers who consume LLM APIs and need to use them effectively, platform engineers who build LLM serving infrastructure and need to design compatible APIs, and product designers who define the capabilities that APIs expose. The patterns have implications for latency, cost, reliability, and developer experience that go beyond the surface-level documentation.

This report provides a comprehensive technical examination of AI API design patterns. It covers the core chat completions pattern, streaming, structured output, function calling, multimodal inputs, batch processing, rate limiting, error handling, SDK design, and operational concerns like versioning, caching, and observability. The intended audience is engineers who build or consume LLM APIs at production scale.

## 2. The Chat Completions API Pattern

### 2.1 The Messages Array

The foundational pattern of modern LLM APIs is the chat completions endpoint, which accepts an array of messages, each with a role and content:

```json
{
  "model": "gpt-4o",
  "messages": [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What is the capital of France?"},
    {"role": "assistant", "content": "The capital of France is Paris."},
    {"role": "user", "content": "What about Germany?"}
  ]
}
```

The messages array represents the conversation history. The model processes the entire array as its context and generates the next assistant message. This pattern has several important design properties:

**Statelessness.** The API is stateless—each request contains the full conversation history. The server does not maintain session state between requests. This simplifies server architecture (no session affinity, no state management) and gives the client full control over conversation management.

**Role-based structure.** The roles (system, user, assistant, tool) provide a lightweight structure that maps to the model's training:
- `system`: Instructions from the application developer. Highest priority in the instruction hierarchy.
- `user`: Input from the end user.
- `assistant`: Previous model outputs. Including these in the messages array provides the model with its own conversation history.
- `tool`: Output from tool calls (function calling results).

**Conversation management is the client's responsibility.** The client decides which messages to include, which to trim (for context window management), and how to format them. This is intentional—different applications have different conversation management needs, and the API should not impose a specific strategy.

### 2.2 The Response Format

The response includes the generated message, usage statistics, and metadata:

```json
{
  "id": "chatcmpl-abc123",
  "object": "chat.completion",
  "created": 1713254400,
  "model": "gpt-4o",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "The capital of Germany is Berlin."
      },
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 53,
    "completion_tokens": 8,
    "total_tokens": 61
  }
}
```

Key design decisions in the response format:

**The choices array.** Supports generating multiple completions per request (n > 1). Each choice has an index and a finish_reason. In practice, most applications use n=1, but the array structure supports use cases like best-of-N sampling and A/B testing.

**finish_reason.** Indicates why the model stopped generating: `stop` (natural completion or stop sequence), `length` (max_tokens reached), `tool_calls` (model wants to invoke a tool), or `content_filter` (output blocked by safety filters). This field is critical for application logic—the application must handle each reason differently.

**Usage statistics.** Token counts for billing and monitoring. The separation of prompt and completion tokens reflects the different pricing tiers for input and output tokens.

### 2.3 Parameters and Controls

The API exposes several parameters that control generation:

**temperature.** Controls randomness. 0 for deterministic output, 0.7-1.0 for creative tasks, >1.0 for high randomness. This is the most frequently tuned parameter.

**max_tokens / max_completion_tokens.** Limits the output length. Essential for cost control and for preventing runaway generation. Setting this too low truncates useful responses; setting it too high wastes tokens on unnecessary output.

**top_p.** Nucleus sampling—considers only the top p probability mass of tokens. An alternative to temperature for controlling diversity. Most practitioners use either temperature or top_p, not both.

**stop.** A list of strings that, if generated, terminate the completion. Useful for structured output (stop at the end of a JSON object) or conversational formatting (stop at the user's turn marker).

**frequency_penalty and presence_penalty.** Penalize repeated tokens, reducing repetition. Useful for creative tasks but can degrade quality for technical tasks where repetition is appropriate.

**logprobs.** Request log-probabilities for each generated token. Essential for confidence estimation, routing, and debugging.

## 3. Streaming Responses

### 3.1 Why Streaming Matters

Without streaming, the user must wait for the entire response to be generated before seeing any output. For a response of 500 tokens at 50 tokens per second, this is a 10-second delay—unacceptable for interactive applications. Streaming sends tokens to the client as they are generated, reducing the perceived latency to the time-to-first-token (TTFT), typically 100-500ms.

### 3.2 Server-Sent Events (SSE)

The standard streaming protocol for LLM APIs is Server-Sent Events (SSE), a simple HTTP-based protocol for server-to-client streaming:

```
POST /v1/chat/completions
Content-Type: application/json
{"model": "gpt-4o", "messages": [...], "stream": true}

HTTP/1.1 200 OK
Content-Type: text/event-stream

data: {"id":"chatcmpl-abc","choices":[{"delta":{"role":"assistant","content":"The"}}]}

data: {"id":"chatcmpl-abc","choices":[{"delta":{"content":" capital"}}]}

data: {"id":"chatcmpl-abc","choices":[{"delta":{"content":" of"}}]}

...

data: [DONE]
```

Each SSE event contains a `delta` (incremental content) rather than the full message. The client concatenates deltas to reconstruct the full response. The stream ends with a `data: [DONE]` sentinel.

Design properties of SSE for LLM APIs:

**HTTP-based.** SSE uses standard HTTP, making it compatible with existing infrastructure (load balancers, proxies, CDNs, firewalls). No special protocol support is required beyond keeping the HTTP connection open.

**One-directional.** SSE supports only server-to-client streaming. The client cannot send messages during the stream. This matches the LLM inference pattern: the client sends a request, the server streams the response. (WebSockets would support bidirectional streaming but add complexity.)

**Reconnection.** SSE includes built-in reconnection logic: if the connection drops, the client can reconnect and potentially resume. However, LLM APIs typically do not support resumption—if the connection drops mid-stream, the request must be retried from the beginning.

**Simplicity.** SSE is text-based and human-readable, making it easy to debug, log, and inspect. This is a significant advantage over binary protocols during development.

### 3.3 Chunked Transfer Encoding

SSE is typically delivered over HTTP/1.1 with chunked transfer encoding, where the server sends the response in chunks without knowing the total content length in advance. HTTP/2 and HTTP/3 natively support streaming without chunked encoding. Most production LLM APIs support HTTP/2, which provides multiplexing (multiple concurrent streams over a single connection) and header compression.

### 3.4 Streaming for Tool Calls

Streaming becomes more complex when the model generates tool calls. The tool call (function name and arguments) must be fully received before the client can execute it, so partial tool call deltas must be buffered and assembled:

```
data: {"choices":[{"delta":{"tool_calls":[{"index":0,"id":"call_abc","function":{"name":"get_weather","arguments":""}}]}}]}
data: {"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":"{\"lo"}}]}}]}
data: {"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":"cati"}}]}}]}
data: {"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":"on\": \"Paris\"}"}}]}}]}
```

The client must concatenate the `arguments` field across multiple deltas, then parse the resulting JSON to extract the function arguments. This is error-prone—JSON parsing cannot begin until the full arguments string is assembled—and adds complexity to streaming implementations.

Anthropic's Claude API uses a slightly different streaming format with explicit event types (`content_block_start`, `content_block_delta`, `content_block_stop`) that make the stream structure more explicit and easier to parse, though the fundamental challenge of buffering tool call arguments remains.

### 3.5 Token-Level vs. Chunk-Level Streaming

Some APIs stream one token at a time (true token-level streaming), while others batch multiple tokens into chunks to reduce overhead:

**Token-level streaming.** Each SSE event contains a single token. Provides the smoothest perceived typing speed. Generates many small events (one per token), which can create overhead from HTTP framing and event processing.

**Chunk-level streaming.** Each SSE event contains 3-10 tokens. Reduces event overhead but introduces slight jitter in the perceived typing speed. Some providers use this approach to reduce server-side overhead.

The difference is usually imperceptible to users. Chunk-level streaming is more efficient for high-throughput serving because it reduces the number of system calls and HTTP frames per response.

## 4. Structured Output

### 4.1 The Problem

LLMs generate text, but applications often need structured data—JSON objects, typed fields, enumerations, database records. Extracting structured data from free-form text is unreliable: the model may produce malformed JSON, include extra text around the JSON, or use unexpected field names or types.

### 4.2 JSON Mode

The simplest structured output mechanism is JSON mode: a parameter that instructs the model to produce valid JSON:

```json
{
  "model": "gpt-4o",
  "messages": [...],
  "response_format": {"type": "json_object"}
}
```

JSON mode guarantees that the output is valid JSON but does not enforce a specific schema. The output might be `{"answer": "Paris"}` or `{"capital": "Paris", "country": "France"}` depending on the model's interpretation of the prompt. This is useful for simple cases but insufficient when the application requires a specific structure.

### 4.3 Schema-Enforced Output

A more powerful mechanism enforces a specific JSON schema:

```json
{
  "model": "gpt-4o",
  "messages": [...],
  "response_format": {
    "type": "json_schema",
    "json_schema": {
      "name": "city_info",
      "schema": {
        "type": "object",
        "properties": {
          "city": {"type": "string"},
          "country": {"type": "string"},
          "population": {"type": "integer"}
        },
        "required": ["city", "country"],
        "additionalProperties": false
      }
    }
  }
}
```

This guarantees that the output conforms to the specified schema. The enforcement is implemented through constrained decoding—at each generation step, the model's output is restricted to tokens that are consistent with the schema given the tokens generated so far. This is implemented using finite state automata or grammar-based constraints that mask invalid tokens before sampling.

The trade-offs of schema enforcement:

**Pros:** Guaranteed valid output. Eliminates the need for output parsing error handling. Enables direct deserialization into typed objects.

**Cons:** Constrained decoding can reduce quality (the best token may be masked if it violates the schema). Complex schemas can slow generation. Not all schema features are supported by all providers.

### 4.4 Provider-Specific Approaches

**OpenAI.** Supports `json_schema` in `response_format` with a subset of JSON Schema. Strict mode enforces the schema through constrained decoding.

**Anthropic.** Supports JSON mode through prompting and prefill. Tool use with explicit schemas provides structured output. Anthropic also supports prefilling the assistant's response with a JSON opening brace to encourage JSON output.

**Google (Gemini).** Supports `response_mime_type` of `application/json` with an optional `response_schema` in OpenAPI format.

**Open-source frameworks.** Outlines, Instructor, and LMQL provide schema-enforced generation for any model, using constrained decoding at the serving layer.

## 5. Function Calling / Tool Use

### 5.1 The Pattern

Function calling (also called tool use) allows the model to invoke external functions:

1. The API request includes a list of available tools with their schemas.
2. The model generates a tool call (function name + arguments) instead of a text response.
3. The client executes the function and sends the result back to the model.
4. The model generates a final text response incorporating the function result.

```json
// Request
{
  "model": "gpt-4o",
  "messages": [
    {"role": "user", "content": "What's the weather in Paris?"}
  ],
  "tools": [
    {
      "type": "function",
      "function": {
        "name": "get_weather",
        "description": "Get current weather for a location",
        "parameters": {
          "type": "object",
          "properties": {
            "location": {"type": "string", "description": "City name"}
          },
          "required": ["location"]
        }
      }
    }
  ]
}

// Response (tool call)
{
  "choices": [{
    "message": {
      "role": "assistant",
      "tool_calls": [{
        "id": "call_abc123",
        "type": "function",
        "function": {
          "name": "get_weather",
          "arguments": "{\"location\": \"Paris\"}"
        }
      }]
    },
    "finish_reason": "tool_calls"
  }]
}

// Follow-up request with tool result
{
  "messages": [
    {"role": "user", "content": "What's the weather in Paris?"},
    {"role": "assistant", "tool_calls": [...]},
    {"role": "tool", "tool_call_id": "call_abc123", "content": "{\"temp\": 18, \"condition\": \"cloudy\"}"}
  ]
}
```

### 5.2 Design Decisions in Tool Use APIs

**Tool schemas as JSON Schema.** Tools are described using JSON Schema for their parameters. This provides a standardized, well-understood format that can be validated by both the client and the server.

**Tool call IDs.** Each tool call has a unique ID that links the tool result back to the specific call. This is necessary because the model may make multiple tool calls in a single turn, and results must be matched to their corresponding calls.

**Parallel tool calls.** The model may generate multiple tool calls in a single response (e.g., "Look up the weather in Paris AND London"). The client should execute these in parallel and return all results in the next request.

**tool_choice parameter.** Controls whether the model can use tools:
- `auto`: Model decides whether to use tools or respond directly.
- `required`: Model must use at least one tool.
- `none`: Tools are not available for this request.
- `{"type": "function", "function": {"name": "..."}}`: Model must call the specified function.

### 5.3 Anthropic's Tool Use Design

Anthropic's Claude API uses a slightly different tool use design that provides additional structure:

**Content blocks.** Claude's responses contain an array of content blocks, each with a type (`text` or `tool_use`). This makes the structure of multi-step responses explicit:

```json
{
  "content": [
    {"type": "text", "text": "Let me check the weather for you."},
    {"type": "tool_use", "id": "toolu_abc", "name": "get_weather", "input": {"location": "Paris"}}
  ]
}
```

The content block approach is cleaner for responses that interleave text and tool calls, because the model can explain what it is doing between tool calls.

### 5.4 Agentic Loops

Function calling enables agentic loops: the model calls tools, observes results, reasons about them, calls more tools, and eventually produces a final response. The API pattern for this is straightforward—the client repeatedly sends requests until the model responds with text (no tool calls):

```
while True:
    response = api.chat(messages)
    if response.has_tool_calls():
        results = execute_tools(response.tool_calls)
        messages.append(response.message)
        messages.extend(tool_result_messages(results))
    else:
        return response.content
```

This pattern is the foundation of LLM-based agents. The API design decisions (stateless requests, tool call IDs, the messages array) make this loop simple to implement while giving the client full control over tool execution, error handling, and loop termination.

## 6. Vision, Audio, and File Upload APIs

### 6.1 Multimodal Input

Modern LLM APIs accept multimodal inputs—images, audio, PDFs, and other file types—in addition to text. The design pattern typically embeds multimodal content in the messages array:

**Image input (URL):**
```json
{
  "role": "user",
  "content": [
    {"type": "text", "text": "Describe this image."},
    {"type": "image_url", "image_url": {"url": "https://example.com/photo.jpg"}}
  ]
}
```

**Image input (base64):**
```json
{
  "role": "user",
  "content": [
    {"type": "text", "text": "Describe this image."},
    {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,/9j/4AAQ..."}}
  ]
}
```

**Audio input (Gemini pattern):**
```json
{
  "role": "user",
  "content": [
    {"type": "text", "text": "Transcribe this audio."},
    {"type": "audio", "audio": {"data": "base64...", "format": "wav"}}
  ]
}
```

### 6.2 Design Considerations for Multimodal APIs

**Token cost of images.** Images are converted to tokens for processing. The token count depends on the image resolution and the provider's vision encoder. A typical high-resolution image may cost 1,000-4,000 tokens, significantly affecting cost. APIs should document the token mapping clearly.

**File size limits.** Base64-encoded images in the request body can be very large (10-50 MB for high-resolution photos). APIs must set and enforce file size limits, and clients should resize images appropriately.

**URL fetching.** When images are provided as URLs, the API server must fetch them. This introduces latency (network round-trip to fetch the image), security concerns (the server may be directed to fetch from malicious URLs—SSRF vulnerability), and reliability concerns (the URL may be inaccessible). Rate limiting and URL validation are essential.

**Audio duration limits.** Audio inputs can represent minutes or hours of content, translating to enormous token counts. APIs typically limit audio input duration (e.g., 30 seconds to 10 minutes) and charge based on audio duration or transcribed tokens.

### 6.3 File Upload Patterns

For large files (PDFs, long audio, video), embedding the content in the request body is impractical. Two patterns are common:

**Pre-upload with file ID.** The client uploads the file to a separate endpoint, receives a file ID, and references the file ID in the chat request:

```json
// Step 1: Upload
POST /v1/files
Content-Type: multipart/form-data
file: @document.pdf

// Response: {"id": "file-abc123"}

// Step 2: Reference in message
{
  "messages": [{
    "role": "user",
    "content": [
      {"type": "text", "text": "Summarize this document."},
      {"type": "file", "file_id": "file-abc123"}
    ]
  }]
}
```

**Signed URL upload.** The client requests a signed upload URL, uploads the file directly to object storage, and references the storage URL in the chat request. This avoids routing large files through the API server.

## 7. Batch APIs

### 7.1 The Batch Processing Pattern

Batch APIs accept multiple requests at once and process them asynchronously, returning results when all requests are complete:

```json
// Submit batch
POST /v1/batches
{
  "requests": [
    {"custom_id": "req-1", "body": {"model": "gpt-4o", "messages": [...]}},
    {"custom_id": "req-2", "body": {"model": "gpt-4o", "messages": [...]}},
    ...
  ]
}

// Response
{"batch_id": "batch-abc123", "status": "in_progress"}

// Check status
GET /v1/batches/batch-abc123
{"batch_id": "batch-abc123", "status": "completed", "output_file_id": "file-xyz"}

// Retrieve results
GET /v1/files/file-xyz
```

### 7.2 Batch API Design Decisions

**Asynchronous processing.** Batch requests are queued and processed during off-peak times, allowing the provider to optimize utilization. Results are available hours later, not immediately.

**Cost discount.** Batch APIs typically offer 50% discounts because the provider can process requests at optimal utilization without latency constraints.

**Custom IDs.** Each request in the batch has a client-assigned ID for matching results to requests.

**Error handling.** Individual requests in a batch can fail while others succeed. The results include per-request success/failure status.

**Size limits.** Batches are limited in number of requests (typically 10,000-50,000) and total token count.

### 7.3 Use Cases

- **Evaluation and benchmarking.** Running model evaluations over thousands of test cases.
- **Data processing.** Classifying, extracting, or transforming large datasets.
- **Content generation.** Generating product descriptions, translations, or summaries for a catalog.
- **Offline analysis.** Analyzing log data, customer feedback, or documents where real-time response is not needed.

## 8. Embeddings API Patterns

### 8.1 The Embeddings Endpoint

Embeddings APIs convert text to dense vectors for similarity search, clustering, and classification:

```json
POST /v1/embeddings
{
  "model": "text-embedding-3-large",
  "input": ["The capital of France is Paris.", "Berlin is the capital of Germany."],
  "dimensions": 1024
}

// Response
{
  "data": [
    {"index": 0, "embedding": [0.0123, -0.0456, ...]},
    {"index": 1, "embedding": [0.0234, -0.0567, ...]}
  ],
  "usage": {"prompt_tokens": 15}
}
```

### 8.2 Design Considerations

**Batched input.** The `input` field accepts an array of strings, enabling efficient batch processing. The embedding model processes all inputs in a single forward pass, which is much more efficient than individual requests.

**Dimensionality control.** Some APIs (OpenAI's text-embedding-3) support a `dimensions` parameter that truncates the embedding to a specified length. Lower dimensions reduce storage and search cost at the expense of some quality.

**Normalization.** Some APIs return normalized embeddings (unit vectors), which simplifies cosine similarity to dot product. Others return unnormalized embeddings, requiring the client to normalize.

**Token limits.** Each input text has a maximum token limit (typically 512-8192 tokens). Texts exceeding the limit must be chunked or truncated by the client.

## 9. Rate Limiting and Quotas

### 9.1 Rate Limiting Dimensions

LLM APIs apply rate limits on multiple dimensions:

**Requests per minute (RPM).** The number of API calls per time window. Prevents abuse and ensures fair sharing of capacity.

**Tokens per minute (TPM).** The total tokens (input + output) processed per time window. Reflects the actual compute cost, which is proportional to tokens, not requests.

**Concurrent requests.** The maximum number of in-flight requests at any time. Prevents a single client from monopolizing serving capacity.

**Tokens per day (TPD).** Daily aggregate limit, primarily for free tiers or trial accounts.

### 9.2 Rate Limit Headers

The standard practice is to include rate limit information in HTTP response headers:

```
x-ratelimit-limit-requests: 500
x-ratelimit-remaining-requests: 487
x-ratelimit-reset-requests: 12s
x-ratelimit-limit-tokens: 200000
x-ratelimit-remaining-tokens: 185234
x-ratelimit-reset-tokens: 7s
```

These headers enable clients to implement proactive rate limiting—slowing down before hitting the limit rather than receiving 429 errors.

### 9.3 Rate Limit Design Considerations

**Sliding window vs. fixed window.** Fixed-window rate limits (e.g., 1000 RPM resetting at the top of each minute) can lead to bursts at window boundaries. Sliding-window rate limits provide smoother enforcement but are more complex to implement.

**Per-key vs. per-organization.** Rate limits can be applied per API key, per organization, or globally. Per-key limits provide isolation between different applications or teams within the same organization. Per-organization limits prevent circumvention through key proliferation.

**Tier-based limits.** Different pricing tiers have different rate limits. Free tiers have restrictive limits; paid tiers have higher limits; enterprise agreements have custom limits.

**Graceful degradation.** When a client exceeds rate limits, the API should return a clear 429 error with a `Retry-After` header indicating when the client can retry. Some APIs implement priority queuing where rate-limited requests are queued rather than rejected, providing degraded latency rather than errors.

## 10. Error Handling

### 10.1 Error Categories

LLM API errors fall into several categories:

**Client errors (4xx):**
- 400 Bad Request: Invalid request format, unsupported parameters, prompt too long.
- 401 Unauthorized: Invalid or missing API key.
- 403 Forbidden: API key does not have access to the requested resource.
- 404 Not Found: Invalid endpoint or model name.
- 422 Unprocessable Entity: Valid format but semantically invalid (e.g., conflicting parameters).
- 429 Too Many Requests: Rate limit exceeded.

**Server errors (5xx):**
- 500 Internal Server Error: Unexpected server failure.
- 502 Bad Gateway: Upstream service failure.
- 503 Service Unavailable: Temporary capacity issues.
- 529 Overloaded: Provider-specific; the service is temporarily overloaded.

### 10.2 Retry Strategies

**Exponential backoff with jitter.** The standard retry strategy for transient errors (429, 500, 502, 503):

```python
import random
import time

def retry_with_backoff(func, max_retries=5, base_delay=1.0):
    for attempt in range(max_retries):
        try:
            return func()
        except (RateLimitError, ServerError) as e:
            if attempt == max_retries - 1:
                raise
            delay = base_delay * (2 ** attempt) + random.uniform(0, 1)
            time.sleep(delay)
```

The jitter (random component) prevents synchronized retries from multiple clients, which can cause "thundering herd" problems.

**Retry-After header.** When the server provides a `Retry-After` header, use it instead of the computed backoff delay—the server knows better than the client when capacity will be available.

**Idempotency.** For non-streaming requests, retries are straightforward because the same request produces the same result (given the same seed/temperature=0). For streaming requests, retries restart the entire generation. Some providers support idempotency keys that guarantee at-most-once execution, preventing duplicate charges when a response is received but the client misses the acknowledgment.

### 10.3 Timeout Handling

LLM requests can take seconds to minutes (for long outputs or reasoning models). Client timeout settings must account for this:

- **Connection timeout:** 10-30 seconds (time to establish the connection).
- **Read timeout (non-streaming):** 60-300 seconds, depending on expected output length.
- **Read timeout (streaming):** Per-chunk timeout of 30-60 seconds. The total request may take minutes, but each chunk should arrive within this timeout.

Setting timeouts too aggressively causes unnecessary failures; setting them too loosely allows resources to be held by hung requests.

## 11. SDK Design

### 11.1 Language-Specific SDKs

LLM providers typically offer SDKs in Python and TypeScript/JavaScript, with some offering Go, Java, Ruby, and other languages. SDK design principles:

**Mirror the API.** The SDK should closely mirror the HTTP API structure, making it easy to translate between the API documentation and the SDK usage. Method names should correspond to endpoints, parameter names should match the API parameters.

**Type safety.** Use typed classes for requests and responses, enabling IDE autocompletion and compile-time error checking. TypeScript's type system and Python's dataclasses/Pydantic models are commonly used.

**Streaming support.** Provide ergonomic streaming interfaces:

```python
# Python (synchronous)
with client.chat.completions.create(
    model="gpt-4o",
    messages=[...],
    stream=True
) as stream:
    for chunk in stream:
        print(chunk.choices[0].delta.content, end="")

# Python (asynchronous)
async with client.chat.completions.create(
    model="gpt-4o",
    messages=[...],
    stream=True
) as stream:
    async for chunk in stream:
        print(chunk.choices[0].delta.content, end="")
```

```typescript
// TypeScript
const stream = await client.chat.completions.create({
  model: "gpt-4o",
  messages: [...],
  stream: true
});

for await (const chunk of stream) {
  process.stdout.write(chunk.choices[0]?.delta?.content || "");
}
```

**Automatic retries.** SDKs should implement exponential backoff with jitter for retryable errors, configurable by the caller.

**Streaming helpers.** Higher-level helpers that assemble streamed tool calls, accumulate content, and provide events for different content types simplify the most common streaming patterns.

### 11.2 OpenAI SDK Compatibility

The OpenAI Python and TypeScript SDKs have become the de facto standard for LLM API interaction. Many providers (Anthropic, Google, Mistral, Together, Groq) offer OpenAI-compatible endpoints that can be accessed using the OpenAI SDK with a different base URL:

```python
from openai import OpenAI

# Using OpenAI
client = OpenAI(api_key="sk-...")

# Using a compatible provider
client = OpenAI(
    api_key="provider-key",
    base_url="https://api.provider.com/v1"
)
```

This compatibility reduces friction for developers and enables easy switching between providers. However, provider-specific features (extended thinking, citations, prompt caching) require provider-specific SDK extensions.

## 12. Versioning and Deprecation

### 12.1 Model Versioning

LLM APIs version at two levels: the API itself and the model.

**Model versions.** Models are updated over time (quality improvements, safety patches, capability additions). Providers handle this with:

- **Dated versions:** `gpt-4o-2024-05-13` refers to a specific model snapshot. Pinning to a dated version ensures reproducibility.
- **Floating aliases:** `gpt-4o` points to the latest stable version. Using the alias means the model may change without notice.
- **Deprecation windows:** Older model versions are deprecated with advance notice (typically 3-12 months). After deprecation, requests to the old version are automatically redirected to the latest version or rejected.

**API versions.** The API format itself (endpoint paths, request/response schemas, parameter names) can change:

- **URL versioning:** `/v1/chat/completions` (most common).
- **Header versioning:** `Anthropic-Version: 2024-01-01` (Anthropic's approach).
- **Date-based versioning:** Stripe-style, where the API behavior is pinned to the version specified in the header.

### 12.2 Deprecation Best Practices

**Advance notice.** 3-6 months minimum for model deprecations; 6-12 months for API version deprecations.

**Migration guides.** Documentation of changes between versions and code examples for migrating.

**Deprecation warnings.** Response headers indicating that the requested model or API version is deprecated, alerting developers before the deprecation takes effect.

**Compatibility layers.** Where possible, maintain backward compatibility so that old requests continue to work with the new version, even if they do not take advantage of new features.

## 13. Caching Strategies

### 13.1 Provider-Side Caching

**Prompt caching.** The provider caches the KV cache from the system prompt (or any shared prefix) and reuses it across requests. This reduces prefill latency and cost for applications with a consistent system prompt.

OpenAI's prompt caching is automatic—requests with the same prefix are cached transparently, and cached tokens are billed at a discount. Anthropic's prompt caching requires explicit `cache_control` markers in the request, giving the developer control over what is cached.

**Semantic caching.** Some providers or middleware layers cache responses for semantically similar queries. If a new query is similar enough to a previously cached query, the cached response is returned without running inference. This is effective for applications with high query repetition but introduces the risk of returning stale or imprecise answers.

### 13.2 Client-Side Caching

**Exact-match caching.** Cache the response for each unique (model, messages, parameters) tuple. Simple and correct but low cache hit rate in most applications.

**Normalized caching.** Normalize the request (trim whitespace, canonicalize parameters) before computing the cache key, improving hit rate.

**TTL-based expiration.** Cache responses for a limited time, ensuring freshness for queries about current events or time-sensitive data.

**Embedding-based similarity caching.** Compute embeddings for incoming queries and check for similar cached queries above a similarity threshold. Higher hit rate but risk of returning incorrect answers for queries that are similar but not identical.

## 14. Observability

### 14.1 Request Logging

Every API request and response should be logged for debugging, monitoring, and cost tracking:

**What to log:**
- Timestamp
- Request ID (for correlation)
- Model name and version
- Input token count and output token count
- Latency (TTFT, total)
- Finish reason
- Error code (if applicable)
- Cost (computed from tokens and pricing)

**What NOT to log (by default):**
- Full prompt content (may contain sensitive user data)
- Full response content (may contain sensitive generated content)
- API keys

Content logging should be opt-in and subject to data privacy requirements (GDPR, HIPAA, etc.).

### 14.2 Cost Tracking

Per-request cost tracking enables:

- **Budget monitoring.** Alert when spending approaches budget limits.
- **Cost attribution.** Allocate costs to teams, features, or users.
- **Optimization identification.** Identify high-cost queries, models, or features that are candidates for optimization.

The cost per request is:

```
cost = input_tokens × input_price + output_tokens × output_price
     + cached_tokens × cached_price + thinking_tokens × thinking_price
```

### 14.3 Quality Monitoring

Beyond operational metrics, monitoring response quality in production is essential:

**Automated quality signals.** Format validation (is the output valid JSON?), length checks (is the output suspiciously short or long?), safety classifiers (does the output contain harmful content?).

**User feedback.** Thumbs up/down, ratings, explicit corrections. These provide ground-truth quality signals but are sparse and biased (users are more likely to give feedback on bad responses).

**LLM-as-judge sampling.** Periodically sample production responses and evaluate them with a judge model. This provides a more comprehensive quality signal than user feedback but adds cost.

## 15. API Security

### 15.1 Authentication and Authorization

**API keys.** The standard authentication mechanism. Keys should be:
- Rotatable (the client can generate new keys and revoke old ones without downtime).
- Scoped (keys can be restricted to specific models, endpoints, or IP ranges).
- Rate-limited (per-key rate limits prevent a compromised key from exhausting the organization's quota).

**OAuth 2.0.** For server-to-server integrations, OAuth provides token-based authentication with scoping and expiration.

**Per-key quotas.** Assigning spend limits or token limits to individual API keys limits the blast radius of a compromised key.

### 15.2 Input Validation

**Prompt length limits.** Enforce maximum input length to prevent denial-of-service through extremely long prompts.

**Content filtering.** Input and output content moderation to prevent abuse of the API for generating harmful content.

**File upload validation.** Validate file types, sizes, and content to prevent malicious file uploads.

### 15.3 Data Privacy

**Data retention policies.** Clearly document how long request and response data is retained, and for what purposes. Enterprise customers typically require zero-retention policies (data is not stored after the response is returned).

**Data processing agreements.** For regulated industries, formal DPAs that specify how data is handled, where it is processed, and what compliance certifications apply.

**Encryption.** TLS for data in transit (mandatory). Encryption at rest for any stored data (logs, cached responses, uploaded files).

## 16. Webhook Patterns for Async Processing

### 16.1 The Webhook Pattern

For long-running operations (batch processing, fine-tuning, large file processing), the API uses a webhook pattern:

1. The client submits a job and provides a webhook URL.
2. The server processes the job asynchronously.
3. When the job completes, the server sends a POST request to the webhook URL with the results.

```json
// Submit job
POST /v1/fine_tuning/jobs
{
  "training_file": "file-abc123",
  "model": "gpt-4o-mini",
  "webhook_url": "https://my-app.com/webhooks/fine-tuning"
}

// Webhook callback (from provider to client)
POST https://my-app.com/webhooks/fine-tuning
{
  "event": "fine_tuning.job.completed",
  "job_id": "ftjob-abc123",
  "status": "succeeded",
  "fine_tuned_model": "ft:gpt-4o-mini:my-org:custom-suffix:abc123"
}
```

### 16.2 Webhook Security

**Signature verification.** The provider signs webhook payloads with a shared secret, allowing the client to verify that the webhook came from the provider and was not tampered with. HMAC-SHA256 signatures in a header (e.g., `X-Webhook-Signature`) are the standard approach.

**Replay protection.** Include a timestamp in the webhook payload and reject webhooks older than a threshold (e.g., 5 minutes) to prevent replay attacks.

**Retry logic.** If the webhook delivery fails (client returns a non-2xx response), the provider retries with exponential backoff. The client must be idempotent—handling duplicate webhook deliveries without side effects.

## 17. Conclusion

The design patterns for AI APIs have matured rapidly, converging on conventions that balance simplicity, flexibility, and performance. The chat completions pattern, with its stateless messages array and role-based structure, provides a clean abstraction over the complexity of autoregressive language model inference. Streaming via SSE enables responsive user experiences. Structured output and function calling enable reliable integration with application logic. Rate limiting, error handling, and observability patterns address the operational realities of running LLM APIs at scale.

The most important design principles that emerge from examining these patterns are:

**Statelessness.** The API does not maintain state between requests. The client controls the conversation context by including it in each request. This simplifies scaling, reduces failure modes, and gives the client maximum flexibility.

**Transparency.** Token counts, pricing, rate limit status, and deprecation warnings are exposed to the client through response headers and metadata. This enables clients to build cost-aware, reliable applications.

**Progressive complexity.** The simplest use case (send a text prompt, get a text response) is trivial. Streaming, tool use, structured output, and multimodal inputs add complexity incrementally as needed. New developers can start simple and adopt advanced features as their applications mature.

**Compatibility.** The OpenAI API format has become a de facto standard, with most providers offering compatible endpoints. This reduces lock-in and enables easy migration between providers. Provider-specific features are exposed through extensions to the standard format rather than replacements.

For API designers, the key advice is to adopt these established patterns rather than inventing new ones. The patterns exist because they solve real problems that any LLM API will face. Deviating from them imposes a learning cost on developers and creates integration friction. Where the standard patterns are insufficient, extend them conservatively—add new parameters, content types, or response fields rather than restructuring the fundamental request/response model.

For API consumers, the key advice is to build abstractions over the API that isolate your application from provider-specific details. Use provider SDKs for convenience, but wrap them in your own interfaces that can be retargeted to a different provider. Cache aggressively, implement retries with backoff, log costs and latency, and monitor quality. The API is the interface to the model, but the reliability, cost, and quality of your application depend on how you use it.
