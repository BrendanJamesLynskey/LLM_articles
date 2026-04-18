# Agentic Performance: End-to-End Measures for Tool-Using LLM Systems

*April 2026*

## 1. Introduction

An LLM call is a function from prompt to response. An agent is a loop
around that function, with tool invocations, memory lookups, and
orchestration code in between. Everything the inference engine does
well can be undone by a chatty tool, a slow vector database, a
congested network link, or a Python interpreter spending 40 ms per
iteration parsing JSON.

This article is the companion to *Inference Performance Measures*. That
article enumerates the per-call metrics — TTFT, TPOT, interactivity,
goodput, percentile latencies. This one asks a different question: how
do those per-call measures compose into the user-visible performance
of a multi-step agent, and what else gets in the way?

The short answer is that agent latency is rarely dominated by the
model. It is dominated by the number of round-trips, the tail of each
round-trip, and the coupling between the model and everything it talks
to. Understanding that coupling is the whole job.

## 2. The Agent Loop as a Latency Pipeline

A typical ReAct-style agent step looks like this:

```
         ┌─ LLM call (TTFT + decode of a tool-call JSON) ─┐
         │                                                │
 user ──►│  tokenize → prefill → decode → parse           │──► tool
         │                                                │     │
         └────────────────────────────────────────────────┘     ▼
                                                        execute tool
                                                             │
         ┌─ LLM call (observation → next action) ────────┐    │
         │                                                │   │
         └────────────────────────────────────────────────┘◄──┘
                              ...
                          final answer
```

Each "⇒" in the pipeline is a latency contribution. For a multi-step
agent, the user-visible end-to-end time is approximately:

```
T_agent  ≈  Σ_steps (  T_LLM_call
                     + T_tool_roundtrip
                     + T_orchestration
                     + T_memory_retrieval
                    )
```

The interesting terms are not `T_LLM_call` — those are well-studied
and tracked — but the three that come after. Each of them is a
coupling point to a different subsystem, with its own performance
envelope and tail behaviour.

## 3. What Gets Amplified in a Multi-Step Agent

### 3.1 Tail Amplification

The single most important property of agent performance is that tails
compound. If a single model call has a p99 latency of 2 s and a typical
agent trajectory is 8 steps, then the probability that *at least one*
step lands in its own p99 is:

```
P(any step ≥ p99)  =  1 − (0.99)^8  ≈  7.7%
```

So roughly 1 agent run in 13 will experience at least one step
operating at its own p99. For a p99.9 latency event on an 8-step run,
about 1 in 125 runs will hit one. The end-to-end latency distribution
is not the per-step distribution — it is the *sum of draws* from the
per-step distribution, and sums are dominated by the worst draw when
the tail is heavy.

The practical consequence: **tighten p99.9 on the model call, not
p50.** Cutting median TPOT in half improves the happy path. Cutting
p99.9 TTFT is what makes a 10-step agent feel reliable.

### 3.2 Round-Trip Count Beats Per-Round-Trip Speed

A tool-calling agent that needs 6 round-trips on a 50-ms link will
spend 300 ms on network alone, regardless of how fast the model
generates tokens. Two classes of optimization attack this directly:

- **Parallel tool calls.** The model emits N tool calls that are
  independent, and the orchestrator dispatches them concurrently
  rather than sequentially. OpenAI's parallel function calls and
  Anthropic's parallel tool use both exist to collapse
  otherwise-sequential round-trips into one. The latency saving is
  roughly `(N − 1) × T_tool`.
- **Speculative tool execution.** While the model is still generating
  the current tool-call JSON, the orchestrator begins executing the
  tool it is *most likely* calling. If correct, the tool result is
  ready the moment parsing completes. This is cache-warming applied
  to tool side-effects, and is safe only for idempotent reads.

Streaming tool-call arguments (most modern serving stacks support
this) also lets the orchestrator begin tool execution before the
complete argument JSON has been generated — useful when the tool itself
streams input, such as a database query that can start scanning as
soon as the table name is known.

### 3.3 Reasoning Models Shift the Bottleneck Back to Decode

Reasoning models (o1, o3, DeepSeek-R1, Claude with extended thinking)
produce long chains of hidden thought tokens before any tool call or
user-visible answer. The per-step decode cost can rise by 10–100×
relative to a standard model. For these systems the bottleneck moves
*back* into the model call, and agentic latency becomes sensitive to
decode throughput in a way that simple tool-using agents are not.

## 4. Coupling to the CPU

Between every GPU inference call and every tool invocation, CPU code
runs. It is easy to overlook because no single line is expensive. In
aggregate it is often the second-largest contributor to per-step
latency after the model itself.

The CPU is responsible for:

- **Tokenization.** Client-side BPE tokenization of a 4K-token prompt
  typically takes 10–40 ms on a single core for most implementations.
  Tokenizer libraries written in Rust (Hugging Face's `tokenizers`,
  OpenAI's `tiktoken`) are substantially faster than Python-native
  ones.
- **Prompt assembly.** Stitching together system prompt, tool
  definitions, conversation history, retrieved memory, and
  tool-result observations. For a 20K-token prompt this is often a
  series of string concatenations and JSON serializations totalling a
  few milliseconds — but it happens on every step, so an 8-step agent
  pays it 8 times.
- **Response parsing.** Extracting tool-call JSON from the model's
  streamed output. Most stacks do this line-by-line with a
  state-machine parser; a minority block until the full response
  arrives before parsing, adding full E2E latency.
- **Orchestration framework overhead.** LangChain, LlamaIndex,
  AutoGen, and their peers add Python overhead for each
  call-to-tool transition. This is normally 5–30 ms per step and is
  invisible until you profile it.
- **Serialization across the process boundary.** If the orchestrator
  and the tool run in different processes (or containers), every tool
  call pays a pickle/JSON/gRPC encode-decode round-trip.

A fast agent keeps the CPU path tight. Tactics:

- Cache tokenized prefixes (system prompts, tool schemas) so the
  tokenizer is not re-run on static content each step.
- Pre-compile templates and pre-render parts of the prompt that don't
  change across steps.
- Use streaming parsers that yield tool calls as soon as they are
  complete, not after the whole response.
- Pin the orchestrator to the same host as the inference endpoint to
  eliminate one network hop, if latency matters more than horizontal
  scalability.

## 5. Coupling to the Network

Agents run on top of a network far more than they run on top of a
model. Each tool call, each RAG retrieval, each MCP request, each
outbound HTTP call is a round-trip.

### 5.1 The Physics of Round-Trips

Network latency is a floor, not an average. Intra-datacenter: 0.1–1 ms
one-way. Across a region: 5–20 ms. Across continents: 70–150 ms. These
numbers cannot be optimized below the speed of light, and they compose
linearly with round-trip count.

A 10-step agent making one inter-regional API call per step pays
roughly `10 × 2 × 100 ms = 2 seconds` on network alone, and another
few seconds in TCP/TLS handshake overhead if connections are not
pooled. This is why production agent deployments:

- Pin the agent loop to the same region as its most-called tools.
- Use connection-pooled, long-lived HTTP/2 or gRPC clients.
- Co-locate the MCP server with the orchestrator when latency is
  critical (and centralize it when manageability is critical — a
  classic trade-off discussed in the MCP Gateways article).

### 5.2 Tail Behaviour of External APIs

A public search API or a third-party LLM endpoint has a latency
distribution that is entirely outside the agent operator's control.
Defensive tactics:

- **Hedged requests.** Fire the same idempotent request to two
  endpoints (or the same endpoint at staggered times) and take the
  first response. This collapses the p99 of two draws into something
  close to the p50 of one. The cost is roughly 2× in request volume
  for the hedged fraction.
- **Per-tool deadlines with fallback.** Cap each tool call at a
  per-tool timeout, and have the agent proceed with "tool failed"
  rather than blocking the entire run when a slow tool is in its own
  tail. This keeps end-to-end p99 bounded even when one tool
  misbehaves.
- **Parallel fan-out when the agent's logic allows.** If two searches
  must both be consulted, issue them concurrently.

### 5.3 Streaming Is Not Free Across Long Links

A streaming response over a 100-ms RTT link pays TCP ack overhead on
every chunk. For short responses the streaming benefit disappears and
the user experiences effectively non-streaming behaviour. For agents
whose first observation is a tool-call JSON, the whole structured
response must arrive anyway before the orchestrator can act, so
streaming provides a TTFT benefit only up to the point the JSON is
structurally complete.

## 6. Coupling to Storage

Agents read from three different kinds of storage, each with different
latency characteristics and each capable of dominating a step.

### 6.1 Vector Databases and RAG

An HNSW index over a few million vectors returns a top-k result in 1–5
ms on a modern vector database running locally, 5–20 ms over the
network, and up to 100 ms on cold indexes or very large collections.
Flat brute-force search is slower but deterministic; quantized
approximate indexes are faster but introduce recall loss that can
change agent outcomes.

The hidden cost of RAG is not the search itself but the re-embedding
of the query. Embedding a query through a 7B embedding model takes
20–60 ms on a GPU and several hundred ms on CPU. Batch those embeddings
when possible; cache them for repeated queries.

### 6.2 KV Cache and Prompt Caching

Prompt caching (discussed in its own article) turns a recurring prefix
into an O(1) prefill operation. For an agent with a long system prompt
and many short steps, prompt caching can cut per-step TTFT by an order
of magnitude. The caveat is that the cache lives on the serving
backend, not on the client, and its hit rate depends on how the
backend shards requests. A load balancer that pins a session to a
single backend preserves cache hits; one that spreads them across
replicas destroys them.

### 6.3 Agent Memory

Long-term memory systems (Zep, LangGraph checkpoints, custom
vector-store-backed memories) are read on most steps. Read latency
varies from 1 ms for local SQLite-backed memory to 50 ms for a remote
document store with a vector search in front of it.

Storage latency compounds with step count exactly like network
latency. A 12-step agent that reads memory once per step on a 30-ms
memory store pays 360 ms just on memory. Paths to reduce this:

- Read memory once per run, not per step, if the memory content is
  stable across the run.
- Pre-fetch memory in parallel with the model call, not sequentially
  after it.
- Keep short-term context in process memory, reserving the external
  store for cross-session or persistent state.

### 6.4 Disk, Swap, and KV-Cache Offload

When the KV cache spills from HBM to CPU DRAM or to NVMe (as in
offload engines such as FlexGen or DeepSpeed-Inference), per-token
decode latency can rise from tens of milliseconds to hundreds. For an
agent this is fatal — each step lands squarely in the slow regime.
Offload is reasonable for batch workloads but not for interactive
agents.

## 7. The Interactivity Measures That Matter for Agents

Single-call interactivity does not translate directly to agent
interactivity. The measures that matter at the agent level are:

| Measure | Definition | Typical target |
|---|---|---|
| Time to first user-visible output | Wall-clock from user message to the first token rendered to the user (not to a tool) | < 1 s for conversational agents, < 2 s with reasoning |
| Time to final answer | End-to-end wall-clock for the complete agent run | < 10 s for chat agents, task-dependent elsewhere |
| Steps to completion | Number of tool/LLM round-trips before the agent emits a final answer | Minimize; correlates linearly with latency |
| Token budget per task | Total input + output tokens consumed across all steps | Cost proxy; correlates with task complexity |
| Task success rate at deadline | Fraction of tasks completing successfully within a wall-clock bound | The goodput of agents |
| Interactive stall probability | P(any single step exceeds user-visible stall threshold) | < 1% for voice, < 5% for chat |

The last row is the agent analogue of stream jitter: not how long the
run takes on average but how often the user sees the agent
mysteriously pause.

### 7.1 Task Goodput

The honest agent-level metric is **task goodput**: successful tasks
per unit time per dollar, or per unit GPU, where "successful" requires
both correctness and meeting a wall-clock deadline. This is the metric
that ties together model quality, inference performance, tool quality,
and the CPU/network/storage coupling terms above. A system that
executes brilliant but slow agents has poor task goodput; so does one
that executes fast but wrong agents.

## 8. Critical-Path Analysis

Because agent latency is a sum over a pipeline, the standard tool for
optimizing it is critical-path analysis. For each step, measure:

- `T_LLM` — time in the model call (TTFT + decode)
- `T_CPU` — time in orchestration, parsing, tokenization
- `T_net` — time in network for tool invocations
- `T_storage` — time in memory/RAG/vector reads
- `T_wait` — time the step is blocked on an external event

Sum across steps and rank. Whichever term dominates is the one worth
optimizing first; the others can wait. This sounds obvious but is
routinely violated — agent platforms tend to invest heavily in model
latency while CPU parsing, serial tool dispatch, or an undersized
vector DB are the actual bottleneck.

Good agent observability stacks (LangSmith, Langfuse, Arize Phoenix,
OpenTelemetry with LLM-aware spans) break the per-step wall clock
into these components automatically. The critical path is rarely the
one the developer thinks it is before measurement.

## 9. Putting It Together: A Worked Example

An 8-step ReAct agent. Per step:

| Component | p50 | p99 |
|---|---|---|
| LLM call (TTFT + decode of 200 tokens) | 400 ms | 1.5 s |
| Tool call (HTTP round-trip + server work) | 60 ms | 400 ms |
| Orchestration CPU | 25 ms | 80 ms |
| Memory read (RAG) | 30 ms | 150 ms |
| Per-step total | 515 ms | 2.13 s |

End-to-end p50 ≈ 8 × 515 ms ≈ 4.1 s. End-to-end p99 is *not* 8 × 2.13
s; the right calculation treats each step's latency as an independent
draw, and the sum's p99 lies between the sum of p50s and the sum of
p99s — typically closer to the former but dragged toward the latter by
tail coincidences. Simulation or measurement is the only honest way to
get it.

The dominant term is clearly `T_LLM`. Cutting it in half (via
speculative decoding or a smaller model) moves the p50 from 4.1 s to
≈ 2.5 s. Cutting the tool-call p99 from 400 ms to 100 ms — by
co-locating the tool or adding a hedge — has a smaller effect on p50
but a large effect on the p99 tail, and therefore on how reliable the
agent feels.

## 10. Summary

The performance of an agent is a composition of model-call performance,
CPU orchestration overhead, network round-trips, and storage reads.
Each of these has its own latency distribution, and the multi-step
nature of agents amplifies tails far more than it amplifies means. The
consequences for anyone designing, deploying, or benchmarking agentic
systems:

1. Measure per-step latency broken down by component — not just the
   model call.
2. Care about p99 and p99.9 on the per-step distribution, because the
   agent rolls the dice once per step.
3. Collapse round-trips wherever the agent's logic allows — parallel
   tool calls, speculative tool execution, streaming tool arguments.
4. Co-locate the agent with its most-called tools and its
   most-accessed storage; the speed of light is a real floor.
5. Treat CPU orchestration as a first-class performance surface, not
   a zero-cost layer between "the interesting parts."
6. Report task goodput — successful tasks per deadline per dollar —
   as the headline agent metric. It is the only one that captures
   everything at once.

A fast inference engine is a necessary condition for a fast agent; it
is nowhere near sufficient. The coupling points around the model are
where most of the latency lives and where most of the optimization
opportunity lives too.
