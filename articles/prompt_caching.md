# Prompt Caching: Reducing Redundant Computation in LLM Inference

*April 2026 • Technical Report*

## 1. Introduction

A surprising amount of LLM compute is spent re-processing text that the model has already seen. A coding assistant calls a model with the same system prompt and the same set of files in context, modifying only the user's most recent message. A document analysis tool sends the same 50-page contract to the model dozens of times with different questions. A chatbot maintains a long conversation history that grows with every turn, and on each turn the model re-reads everything from the beginning. In each case, the prefix of the prompt is identical to a prefix the model has processed before — and in each case, conventional inference re-computes the entire forward pass through the prefix, paying full price for tokens that produce the same intermediate values they did last time.

Prompt caching is the optimization that recognizes this waste and eliminates it. By saving the key-value (KV) cache produced when a prefix is first processed, and reusing that cache on subsequent requests with the same prefix, an inference system can skip the prefill phase entirely for the cached portion. The user experiences dramatically faster responses for repeated queries, and the operator pays for compute proportional to the new tokens rather than the total prompt length.

This report examines how prompt caching works, the variants that have been deployed, the cost and pricing implications, and the practical considerations for both users and providers.

## 2. The Inefficiency Prompt Caching Solves

### 2.1 Prefill Cost Dominance

In a typical LLM inference request, the work decomposes into two phases. **Prefill** processes the entire input prompt in parallel, computing the key and value vectors for every token at every layer and storing them in the KV cache. The cost of prefill is proportional to the prompt length squared (for attention) plus the prompt length times the model size (for the linear projections). For long prompts, prefill dominates the total inference cost.

**Decode** then generates the output tokens one at a time, reading from the KV cache and producing one new token per forward pass. The cost of decode is proportional to the number of output tokens, with a much smaller constant factor than prefill.

For a model like Claude or GPT-4 running with a 32,000-token system prompt and producing a 200-token response, prefill might consume 95% of the compute and decode the remaining 5%. The user perceives this as latency: the time to first token is high (a second or more), and then the subsequent tokens stream out quickly. Eliminating that first-token latency is the primary user-visible benefit of prompt caching.

### 2.2 The Repetition Pattern

The economic case for prompt caching hinges on how often prompts repeat. In production deployments, the answer is "constantly":

- **Coding agents** typically prepend a fixed 5,000-50,000 token system prompt with tool definitions, plus the contents of relevant files in the user's project. The user's question changes turn to turn, but the prefix is identical.
- **Customer support bots** use a fixed knowledge base concatenated to the system prompt. The user's question varies, but the knowledge base is the same.
- **Document Q&A** uploads a long document once and answers many questions about it. Each question re-processes the document if no caching is in place.
- **Multi-turn conversations** grow the prompt by one user message and one assistant response per turn, but the previous turns are unchanged.

A measurement Anthropic published in 2024 showed that for Claude API traffic, the median cacheable prefix was approximately 4,000 tokens and the 90th percentile exceeded 50,000 tokens. The opportunity for compute savings was substantial.

## 3. How Prompt Caching Works

### 3.1 KV Cache as the Memoization Target

The KV cache produced during prefill is exactly the data needed to skip prefill on subsequent requests. If a prompt has prefix P followed by suffix S, and the model has previously processed P, then the KV cache for P contains all the keys and values that the attention layers need for S to attend to P. As long as the model processes S as if P were already in its cache, the result is identical to running the full prompt P+S from scratch.

This is the conceptual basis of prompt caching. When a request arrives, the inference system checks whether the request's prompt prefix matches a cached KV cache. If yes, it loads the cache, runs prefill only for the new tokens at the end of the prompt, and proceeds with decode. If no, it runs full prefill and stores the resulting KV cache for future use.

### 3.2 Granularity: Page-Level vs. Token-Level

The simplest form of prompt caching uses exact prefix matching — the cached prefix must exactly match the new prompt's prefix, character-for-character. This works for cases like fixed system prompts but fails when users prepend even slightly different content.

A more sophisticated form uses page-level caching, where the KV cache is divided into fixed-size blocks (typically 16-128 tokens per block) and each block can be cached independently. This integrates naturally with PagedAttention-based inference systems like vLLM, where the KV cache is already organized into pages. A new request can match a cached prefix on a per-page basis, sharing as many leading pages as possible and only re-computing the pages that diverge.

Token-level caching takes this further by allowing prefix matching at single-token granularity. The implementation is more complex (every cached prefix corresponds to a unique trie node) but the cache hit rates are higher because near-duplicate prompts can share more of their cached state.

### 3.3 Trie-Based Cache Indexing

vLLM and SGLang (along with several other modern inference frameworks) use a trie data structure to index cached prefixes. Each node in the trie corresponds to a token, and the path from the root to a node represents a sequence of tokens whose KV cache is currently in memory. When a new request arrives, the system walks the trie from the root, following the request's tokens until it finds the deepest node whose cache is still valid. The cache contents along this path are reused; the remaining tokens are processed in a fresh prefill.

This structure handles arbitrary branching: two requests sharing a prefix of 10,000 tokens but diverging at token 10,001 will both reuse the shared prefix and store their unique suffixes as separate trie branches. A subsequent request matching either branch will find the longest common prefix in the trie. The trie can be evicted using LRU or other policies when memory pressure forces it.

## 4. Provider Implementations

### 4.1 Anthropic Prompt Caching

Anthropic introduced prompt caching for the Claude API in 2024. Users mark portions of their prompts as cacheable using a `cache_control` field in the API request, which tells Claude to write the KV cache for that prefix to a fast-access store after the first request. Subsequent requests with the same cached prefix incur a 90% discount on the cached input tokens.

The Anthropic implementation uses explicit cache marking rather than automatic detection. The user must opt in by adding cache breakpoints, and the system caches up to 4 breakpoints per request. Cached prefixes have a default time-to-live of 5 minutes, with a 1-hour TTL option at higher per-token cost. Cache writes (the first request that populates the cache) incur a 25% premium over normal input pricing; cache reads (subsequent matching requests) cost 10% of normal.

The economic consequence is that prompt caching is profitable when the same cached prefix is reused at least twice. For typical agent workloads with hundreds of reuses per hour, the savings approach 80% of the input token cost.

### 4.2 OpenAI Prompt Caching

OpenAI shipped automatic prompt caching for GPT-4o and o1 in October 2024. Unlike Anthropic, OpenAI does not require explicit cache markers — the system automatically detects cacheable prefixes and applies a 50% discount when a cached prefix is reused. The cache TTL is approximately 5-10 minutes, with no longer-lived option.

The OpenAI approach trades off control for ease of use. Users do not need to think about cache markers, but they also have less ability to manage cache lifetimes or guarantee cache hits. The discount is smaller (50% vs. 90%) but applies automatically to any qualifying request.

### 4.3 Google Gemini

Google's Gemini API introduced explicit context caching in mid-2024. Users create a cached content object via a separate API call, receive a cache identifier, and then reference the cached content in subsequent requests. The cache persists for a configurable duration (default one hour, billed by the cached token-hour) and is reused across many requests.

The Gemini approach is the most explicit of the three: users manage cached content objects as first-class resources, deciding what to cache and how long to retain it. This works well for use cases like document Q&A, where a single document is cached once and referenced from many requests, but is cumbersome for highly dynamic prompts.

### 4.4 Open-Source Frameworks

vLLM, SGLang, and TGI all support prompt caching natively. The implementations are typically automatic (no user opt-in required) and use trie-based indexing to maximize hit rates. SGLang in particular has invested heavily in prompt caching as part of its broader RadixAttention architecture, achieving cache hit rates above 95% for some benchmark workloads.

For self-hosted deployments, the cache is managed entirely in GPU memory by default. Frameworks support spilling cold cache pages to CPU memory or disk, with the cost of slightly higher latency on cache hits that require reloading. Organizations running their own inference servers typically configure cache policies based on their workload patterns.

## 5. Cache Invalidation and Consistency

A critical correctness requirement is that the cached KV values must match what the model would compute today, not just what it computed when the cache was populated. Several factors can invalidate a cache:

- **Model version changes**: If the underlying model is updated, all cached KV values become stale. Providers typically tag caches with model version and invalidate on update.
- **Tokenization differences**: Subtle changes in tokenization can change which tokens are produced, even for identical input strings. This is rare in production but can occur during tokenizer upgrades.
- **Numerical non-determinism**: Floating point arithmetic in attention is not perfectly deterministic across hardware or even across batches on the same hardware. The differences are tiny and do not affect output quality, but they mean that a cached KV value is not bit-identical to a freshly-computed one. Most implementations accept this and assume the differences are negligible.
- **Sampling parameters**: Sampling parameters (temperature, top-p, etc.) do not affect the KV cache, so they are safe to vary across requests sharing a cached prefix.

The provider's cache infrastructure must track these factors and invalidate appropriately. Bugs in this area can produce subtle output errors that are very hard to diagnose because they manifest as intermittent quality regressions rather than outright failures.

## 6. Memory Pressure and Eviction

Cached KV values consume GPU memory — typically several megabytes per 1,000 tokens for a large model. A production inference server might have hundreds of GB of GPU memory dedicated to KV cache, of which a substantial fraction is "cold" prefix cache from previous requests.

When new requests arrive and the KV cache memory is exhausted, some cached prefixes must be evicted. The standard policy is LRU (least recently used), evicting the prefix that has gone the longest without a cache hit. More sophisticated policies consider the cost of recreating the prefix (longer prefixes are more valuable to retain) or the predicted probability of reuse (frequently-accessed prefixes are protected from eviction).

Spilling to CPU memory or NVMe storage extends cache capacity at the cost of higher cache-hit latency. A cache hit served from GPU memory has near-zero latency overhead; a hit requiring a load from CPU memory adds 1-5 milliseconds; a hit requiring a load from NVMe adds 20-100 milliseconds. For most use cases, GPU cache hits are dramatically more valuable than spilled cache hits, but the spillover capacity provides graceful degradation when GPU memory pressure is high.

## 7. Cost Implications

### 7.1 Provider Economics

For providers, prompt caching reduces compute cost on the cache-hit path. The savings are roughly proportional to the cached prefix length: skipping prefill for 10,000 tokens saves approximately the compute of generating 10,000 tokens of output, which on a large model is a substantial number of FLOPs and a meaningful fraction of GPU-seconds.

The catch is that providers must also bear the cost of storing the cached KV values, which consume GPU memory. If memory is the binding constraint on serving capacity (which it often is for very large models), prompt caching trades off cache storage against the number of concurrent users that can be served. Providers must make capacity decisions that balance these factors, which is part of why caching is offered as a discount rather than free — the discount funds the additional memory allocation.

### 7.2 User Economics

For users, prompt caching is essentially free money for any application with repeated prefixes. A coding agent that previously paid for 50,000 input tokens per turn now pays for 50,000 input tokens on the first turn and 5,000 input tokens (50,000 cached at 10% + 5,000 new) on subsequent turns. For a user with 100 turns per session, the cost reduction is approximately 80%.

The users who benefit most are those with the most predictable prompt patterns. Customer support bots, document Q&A, code assistants, and other applications with stable prefixes see massive savings. Applications with highly variable prompts — translation, summarization of unique documents, ad-hoc Q&A — see little benefit because cache hit rates are low.

## 8. Application-Level Strategies

### 8.1 Cache-Friendly Prompt Design

To maximize cache hit rates, application developers should structure prompts so that the stable parts come first and the variable parts come last. A well-designed prompt looks like:

```
<cacheable system prompt>
<cacheable tool definitions>
<cacheable knowledge base context>
<cacheable conversation history up to N-1 turns>
<variable: most recent user message>
```

Reordering this sequence to put variable content earlier breaks the cache. Sticking to the cache-friendly order is sometimes counterintuitive (developers often want to put the most important context closest to the question) but yields large cost savings.

### 8.2 Cache Warming

For applications with known prompt prefixes, it can be valuable to "warm" the cache by sending a dummy request with the prefix and discarding the response. This populates the cache before real requests arrive, ensuring that all subsequent requests are cache hits. The cost of warming is one cache write, which is amortized over many cache reads.

### 8.3 Conversation Threading

For multi-turn conversations, caching naturally accumulates as the conversation grows. Each turn extends the previous cache by the user message and assistant response, so the next turn's cache hit length grows monotonically. This is the ideal pattern for caching — the cache hit ratio approaches 100% over time, and the per-turn input cost flattens out at the cost of just the new tokens.

## 9. Advanced Topics

### 9.1 Speculative Cache Reuse

Some inference systems experiment with speculative cache reuse, where a cached KV cache from a similar but not identical prefix is reused with the assumption that the small differences will not affect quality. This is risky — the resulting outputs are subtly different from a freshly-computed forward pass — but for low-stakes applications it can extend cache hit rates beyond what exact matching allows.

### 9.2 Cross-Tenant Caching

In multi-tenant inference services, prompt caching raises a security question: if two tenants happen to send the same prompt (e.g., a popular open-source system prompt), should they share a cached KV cache? The compute savings would be substantial, but it creates a cross-tenant data flow where one tenant's request influences another tenant's latency, potentially leaking information about prompt patterns. Most providers disable cross-tenant caching and treat each tenant's cache as isolated.

### 9.3 Distributed Caching

For inference deployments spanning many GPUs or many nodes, the question arises of where to store cached prefixes. The simplest approach is per-GPU caches — each GPU stores caches for the requests it handles. This is suboptimal because requests for the same prefix may arrive at different GPUs, missing the cache. A more sophisticated approach uses a distributed cache, with prefix-to-GPU routing that ensures requests with the same prefix consistently land on the same GPU and hit the cache.

The tradeoff is routing complexity versus cache hit rate. Distributed caching can dramatically improve hit rates but requires the load balancer to be cache-aware, which complicates the inference serving architecture.

## 10. Conclusion

Prompt caching is one of the most effective inference optimizations in modern LLM serving. The savings are large (typically 50-90% on the cached portion), the implementation is well-understood, and the user experience improvement (lower time to first token) is immediately visible. Every major commercial provider offers some form of prompt caching, and every major open-source inference framework supports it natively.

The deeper lesson is that LLM inference is full of redundancy that conventional approaches do not exploit. Prompt caching is the most prominent example, but related techniques — speculative decoding, paged attention, prefix sharing across batches, KV quantization — all attack different forms of compute waste. As inference workloads grow and economics tighten, expect more such optimizations to emerge. The model architecture is increasingly fixed; the inference stack around it is where the next several generations of efficiency improvement will come from.

For application developers, the practical advice is simple: structure your prompts to maximize prefix stability, opt into caching when your provider offers explicit controls, and measure your cache hit rate to verify you are getting the savings you expect. The cost reduction is often the largest single optimization available to LLM applications, requiring no model changes and no complex engineering — just attention to prompt structure.
