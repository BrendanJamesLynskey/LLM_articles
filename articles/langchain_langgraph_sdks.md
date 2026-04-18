# LangChain and LangGraph SDKs

*April 2026*

## 1. Introduction

"LangChain SDK" is shorthand for what is actually a constellation of packages: a stable core, dozens of integration packages, a separate graph-orchestration stack, a server SDK for the LangGraph Platform, a CLI, and the LangSmith tracing client. Each exists in both Python and TypeScript with broadly equivalent surfaces, a handful of deliberate platform differences (Pydantic vs Zod, asyncio vs promises, decorators vs higher-order functions), and independent release cadences.

This article maps the full SDK surface as it stands in April 2026: which package to install for which job, how Python and TypeScript differ, how the LangGraph client SDK relates to the LangGraph Platform, and how LangSmith fits alongside everything. It complements the conceptual articles in this collection by focusing on the packaging, not the concepts.

## 2. Python: `langchain-*` Packages

### 2.1 The Core Stack

```
langchain-core              # Runnables, messages, prompt templates, tool abstractions
langchain                   # Legacy chain helpers; still useful for RAG helpers
langchain-community         # Long-tail integrations that haven't earned their own package
langchain-text-splitters    # Chunkers
```

`langchain-core` is the load-bearing package. It defines `Runnable`, `BaseChatModel`, `BaseRetriever`, `BaseTool`, `PromptTemplate`, output parsers, and the message types. A minimal LangChain app that uses a single model provider and a single vector store typically imports from `langchain-core`, the provider package, the vector store package, and nothing else.

### 2.2 Model Provider Packages

Each major provider has its own package. Pin these in your `pyproject.toml`.

| Package | Typical exports |
|---|---|
| `langchain-openai` | `ChatOpenAI`, `OpenAIEmbeddings`, `AzureChatOpenAI` |
| `langchain-anthropic` | `ChatAnthropic` |
| `langchain-google-genai` | `ChatGoogleGenerativeAI` (public Gemini API) |
| `langchain-google-vertexai` | `ChatVertexAI` (enterprise Gemini via Vertex) |
| `langchain-aws` | `ChatBedrock`, `BedrockEmbeddings` |
| `langchain-cohere` | `ChatCohere`, `CohereEmbeddings` |
| `langchain-mistralai` | `ChatMistralAI` |
| `langchain-xai` | `ChatXAI` |
| `langchain-together` | `ChatTogether` |
| `langchain-groq` | `ChatGroq` |
| `langchain-fireworks` | `ChatFireworks` |
| `langchain-ollama` | `ChatOllama` — local models |

All of them implement `BaseChatModel`, so once you have written your pipeline against the base class, swapping providers is one import away.

### 2.3 Vector Store Packages

```
langchain-chroma            # Chroma, local-first vector DB
langchain-pinecone          # Pinecone managed vector search
langchain-qdrant            # Qdrant open-source/managed
langchain-weaviate          # Weaviate
langchain-milvus            # Milvus / Zilliz
langchain-postgres          # pgvector
langchain-mongodb           # MongoDB Atlas Vector Search
langchain-redis             # Redis vector
langchain-elasticsearch     # Elasticsearch
langchain-opensearch        # OpenSearch
```

All expose a `VectorStore` interface with `add_texts`, `similarity_search`, `as_retriever`, and an ingestion API. As with chat models, they are interchangeable at the interface level.

### 2.4 Other Integration Packages

Loaders, retrievers, and specialty integrations live in dedicated packages where volume justifies it (`langchain-unstructured` for rich document parsing, `langchain-tavily` for search, `langchain-exa` for neural search, `langchain-huggingface` for local embeddings/inference) or in `langchain-community` otherwise.

## 3. Python: LangGraph Packages

```
langgraph                    # The graph library
langgraph-checkpoint-*       # Persistence backends (sqlite, postgres, ...)
langgraph-sdk                # Client for LangGraph Server / Platform
langgraph-cli                # `langgraph dev`, `langgraph build`, `langgraph up`
langgraph-prebuilt           # Prebuilt graphs (ReAct, tool-node, etc.)
deepagents                   # Deep Agents preset (see companion article)
langmem                      # Long-term memory helpers
```

### 3.1 `langgraph`

The core library: `StateGraph`, `Send`, `Command`, `interrupt`, `add_messages`, `ToolNode`, checkpointing interfaces. No platform dependencies; works standalone with or without LangChain.

### 3.2 Checkpoint backends

Separate packages so you don't pay for a Postgres driver if you only need SQLite:

- `langgraph-checkpoint-sqlite` → `SqliteSaver` and `AsyncSqliteSaver`.
- `langgraph-checkpoint-postgres` → `PostgresSaver` and `AsyncPostgresSaver`.

`MemorySaver` is in the base `langgraph` package for notebooks and tests.

### 3.3 `langgraph-sdk` (client)

The HTTP client for LangGraph Server / LangGraph Platform. You install this in applications that consume a deployed graph — a web backend, a mobile API server, a Slack bot — *not* in the graph itself. It handles thread creation, streaming, HITL resume, assistants, and store operations.

```python
from langgraph_sdk import get_client

client = get_client(url="https://my-graph.example.com", api_key="...")
thread = await client.threads.create()
async for event in client.runs.stream(
    thread["thread_id"],
    assistant_id="agent",
    input={"messages": [{"role": "user", "content": "hi"}]},
    stream_mode="messages-tuple",
):
    print(event)
```

### 3.4 `langgraph-cli`

The developer workhorse for LangGraph Server. Usage:

```bash
pip install "langgraph-cli[inmem]"

# Start a local server with hot reload for development
langgraph dev

# Build a Docker image for production deployment
langgraph build -t my-agent

# Start a local server using your built image
langgraph up
```

`langgraph dev` runs an in-memory LangGraph Server on your laptop that understands the same config and exposes the same API as the hosted platform. It includes Studio — a browser-based visual debugger for graphs — which is enabled automatically.

The CLI reads a `langgraph.json` manifest:

```json
{
  "dependencies": ["."],
  "graphs": {
    "agent": "./src/my_agent/graph.py:graph",
    "research": "./src/my_agent/research.py:research_graph"
  },
  "env": ".env"
}
```

## 4. Python: LangSmith Client

```
langsmith                    # The tracing + eval client
```

LangSmith is a separate product and SDK. It integrates with both LangChain and LangGraph as a callback handler, but it can also be used standalone for any Python code via `traceable`:

```python
from langsmith import traceable

@traceable(name="fetch_and_summarise")
def pipeline(url: str) -> str:
    ...
```

Traces are captured when `LANGSMITH_TRACING=true` and `LANGSMITH_API_KEY=...` are set. No code changes required beyond the decorator for non-LangChain code.

LangSmith also ships evaluation primitives (`evaluate`, `Client.create_dataset`, `Client.list_examples`) that are useful for regression testing agents, but those live outside the scope of this article.

## 5. TypeScript: `@langchain/*` Packages

The TS side mirrors the Python side with scoped packages under `@langchain/*` and `@langchain/langgraph`.

### 5.1 Core

```
@langchain/core
@langchain/community
@langchain/textsplitters
```

`@langchain/core` is the TS equivalent of `langchain-core`: Runnable, messages, prompts, parsers, tool abstractions.

### 5.2 Providers

| Package | Exports |
|---|---|
| `@langchain/openai` | `ChatOpenAI`, `OpenAIEmbeddings` |
| `@langchain/anthropic` | `ChatAnthropic` |
| `@langchain/google-genai` | `ChatGoogleGenerativeAI` |
| `@langchain/google-vertexai` | `ChatVertexAI` |
| `@langchain/aws` | `ChatBedrockConverse` |
| `@langchain/mistralai` | `ChatMistralAI` |
| `@langchain/cohere` | `ChatCohere` |
| `@langchain/groq` | `ChatGroq` |
| `@langchain/ollama` | `ChatOllama` |

All implement `BaseChatModel` in the TS sense (same interface shape as Python, ported idiomatically).

### 5.3 Vector stores

```
@langchain/pinecone
@langchain/qdrant
@langchain/weaviate
@langchain/mongodb
@langchain/redis
@langchain/postgres       # pgvector
@langchain/community      # Chroma, Milvus, Supabase, Upstash, ...
```

### 5.4 LangGraph for TypeScript

```
@langchain/langgraph                    # Core graph library
@langchain/langgraph-checkpoint         # Base checkpointer interfaces
@langchain/langgraph-checkpoint-sqlite
@langchain/langgraph-checkpoint-postgres
@langchain/langgraph-sdk                # Client SDK for LangGraph Server
```

The TS graph library has the same mental model as Python — `StateGraph`, `Annotation`, `Command`, `Send`, `interrupt`, prebuilt `createReactAgent` — with idiomatic TS differences noted below.

### 5.5 LangSmith

```
langsmith                   # Same name, different registry
```

The `traceable` wrapper in TS is a higher-order function rather than a decorator:

```typescript
import { traceable } from "langsmith/traceable";

const pipeline = traceable(
  async (url: string) => { /* ... */ },
  { name: "fetch_and_summarise" },
);
```

## 6. Python vs TypeScript: Where They Differ

The two SDKs are 95% interchangeable at the mental-model level. The remaining 5% is worth understanding.

### 6.1 State Declarations

Python uses `TypedDict` + `Annotated` for reducers. TS uses `Annotation.Root({...})`:

```typescript
import { Annotation, messagesStateReducer } from "@langchain/langgraph";

const StateAnnotation = Annotation.Root({
  messages: Annotation<BaseMessage[]>({
    reducer: messagesStateReducer,
    default: () => [],
  }),
  plan: Annotation<string>(),
  currentStep: Annotation<number>(),
});
```

### 6.2 Schema Validation

Python: Pydantic models. TS: Zod schemas.

```typescript
import { z } from "zod";

const Person = z.object({ name: z.string(), age: z.number() });
const extractor = model.withStructuredOutput(Person);
```

Tool argument schemas in TS are Zod-first; in Python they are Pydantic-first (with TypedDict also supported).

### 6.3 Concurrency

Python uses asyncio. TS uses promises and async iterators. The call sites differ:

```python
async for chunk in app.astream(input, config):
    ...
```

```typescript
for await (const chunk of app.stream(input, config)) {
  ...
}
```

### 6.4 Tool Definitions

Python: `@tool` decorator.

```python
@tool
def get_weather(city: str) -> str:
    """Weather for a city."""
    ...
```

TS: `tool(fn, schema)` factory.

```typescript
import { tool } from "@langchain/core/tools";
import { z } from "zod";

const getWeather = tool(
  async ({ city }: { city: string }) => `${city}: 18°C`,
  {
    name: "get_weather",
    description: "Weather for a city.",
    schema: z.object({ city: z.string() }),
  },
);
```

### 6.5 Feature Parity

As of April 2026, TypeScript LangGraph typically trails Python by 2-6 weeks on new features. Deep Agents ships Python-only at the preset level — you can build the same architecture in TS from primitives, but the `deepagents` package is Python. Most core features (checkpointing, HITL, `Send`, `Command`, subgraphs, streaming modes, prebuilt ReAct, Store API) are at parity.

## 7. The LangGraph Platform

### 7.1 What It Is

LangGraph Platform is a managed runtime for LangGraph graphs. You push a graph (via `langgraph build` + deploy, or via a connected Git repository) and the platform:

- Hosts the graph behind an HTTPS API.
- Provides a managed Postgres checkpointer and store.
- Exposes thread and run APIs.
- Handles SSE streaming.
- Integrates with LangSmith for tracing.
- Handles scaling and availability.

It is comparable to hosted runtimes like Cloudflare Workers or Vercel, but specialised for long-running, stateful, potentially-human-in-the-loop agent graphs.

### 7.2 The Server API Shape

Whether hosted or self-hosted, the LangGraph Server exposes a consistent REST + SSE API:

| Endpoint | Purpose |
|---|---|
| `POST /threads` | Create a thread |
| `GET /threads/{id}/state` | Current thread state |
| `POST /threads/{id}/runs` | Start a run (optionally streaming) |
| `POST /threads/{id}/runs/{run_id}/cancel` | Cancel a run |
| `POST /threads/{id}/history` | Full checkpoint history |
| `POST /threads/{id}/runs/wait` | Wait for completion (blocking) |
| `POST /threads/{id}/runs/stream` | Stream events (SSE) |
| `POST /runs` / `POST /runs/stream` | Stateless runs (no thread) |
| `GET /assistants` / `POST /assistants` | Assistants (saved graph configurations) |
| `GET /store/items` / `POST /store/items` | Long-term memory store |

Clients talk to this API via `langgraph-sdk` (Python) or `@langchain/langgraph-sdk` (TS). You only need these client SDKs when calling the server *from another process*. Inside a graph, the graph calls itself directly.

### 7.3 Self-Hosted LangGraph Server

The same server binary that powers the managed platform runs standalone on your own infrastructure. A typical Dockerfile produced by `langgraph build` gives you a container that expects a Postgres connection string and exposes port 8000 with the full server API. Deploy on Kubernetes, ECS, Cloud Run, or a VM.

### 7.4 Assistants

An Assistant is a named, versioned configuration of a graph — specific model, specific tools, specific system prompt — that clients can reference by ID. Assistants decouple "what graph" from "what configuration", which matters in practice because the same graph is often deployed many times with different prompts/models for different use cases.

## 8. The Local Development Flow

The intended local development loop with LangGraph Platform in scope:

1. Write your graph as Python/TS code in a repo.
2. Declare it in `langgraph.json`.
3. `langgraph dev` — starts a local LangGraph Server (via `langgraph-cli`) with hot reload and LangGraph Studio. You can hit the real REST API, stream events, inspect state, and step through threads in the Studio UI.
4. Iterate until the graph behaves.
5. `langgraph build -t my-agent` — produces a Docker image.
6. Deploy to the managed platform, or push the image to your own infrastructure.

This flow is the same whether you are building a Deep Agent or a minimal ReAct agent.

## 9. Versioning and Stability

### 9.1 Core Stability

`langchain-core` and `langgraph` are conservatively versioned. Breaking changes are rare, announced ahead of time, and typically come with a deprecation period.

### 9.2 Integration Churn

Integration packages (`langchain-openai`, vector stores, etc.) move faster. Provider SDKs introduce new parameters, deprecate old ones, and occasionally change defaults. Pin minor versions in production dependencies and bump on a cadence with regression tests.

### 9.3 Python Pinning Example

```toml
[project]
dependencies = [
    "langchain-core>=0.3,<0.4",
    "langchain-openai>=0.3,<0.4",
    "langchain-anthropic>=0.3,<0.4",
    "langgraph>=0.2,<0.3",
    "langgraph-checkpoint-postgres>=2.0,<3.0",
    "langsmith>=0.2,<0.3",
]
```

### 9.4 TypeScript Pinning Example

```json
{
  "dependencies": {
    "@langchain/core": "^0.3",
    "@langchain/openai": "^0.5",
    "@langchain/anthropic": "^0.3",
    "@langchain/langgraph": "^0.3",
    "@langchain/langgraph-checkpoint-postgres": "^0.1",
    "langsmith": "^0.3"
  }
}
```

Caret ranges are appropriate for non-major versions; bump majors deliberately.

## 10. Choosing What to Install

Three common profiles:

**RAG / single-chain app (Python):**
```
langchain-core
langchain-openai          # or your provider
langchain-pinecone        # or your vector store
langchain-text-splitters
langsmith
```

**Agent with a loop (Python):**
```
langchain-core
langchain-anthropic       # or your provider
langgraph
langgraph-checkpoint-postgres
langsmith
```

**Deep Agent (Python):**
```
deepagents                # brings langgraph + langchain as deps
langchain-anthropic       # if you override the default model
langchain-tavily          # if using Tavily for web search
langsmith
```

**TypeScript web app calling a deployed graph:**
```
@langchain/langgraph-sdk  # client only
langsmith
```

## 11. Environment Variables

Canonical env vars used across the SDKs:

| Variable | Purpose |
|---|---|
| `OPENAI_API_KEY` / `ANTHROPIC_API_KEY` / ... | Provider keys |
| `LANGSMITH_TRACING` | `true` to enable tracing |
| `LANGSMITH_API_KEY` | LangSmith API key |
| `LANGSMITH_PROJECT` | Project name for trace grouping |
| `LANGSMITH_ENDPOINT` | For self-hosted LangSmith |
| `LANGGRAPH_API_URL` | Default URL for `langgraph-sdk` |
| `LANGGRAPH_AUTH_TYPE`, `LANGGRAPH_AUTH_JWT_SECRET` | Self-hosted auth configuration |

Keep them in `.env` files for local development and in your secrets manager for production.

## 12. Interop with Non-LangChain Code

Three clean interop points:

### 12.1 Raw Provider SDKs

You can interleave a direct `anthropic.messages.create(...)` or `openai.chat.completions.create(...)` call with LangChain code — `traceable` will capture it in the LangSmith trace, and the return type is just JSON/dict that you can pass into subsequent LangChain code.

### 12.2 MCP Servers

LangChain has first-class MCP client support via `langchain-mcp-adapters`. Connect to any MCP server and its tools become `BaseTool` instances usable in any LangChain/LangGraph agent.

### 12.3 FastAPI / Next.js

LangGraph graphs plug into HTTP frameworks without special integration. For a FastAPI endpoint that streams tokens:

```python
from fastapi import FastAPI
from fastapi.responses import StreamingResponse

app = FastAPI()

@app.post("/chat")
async def chat(body: dict):
    async def gen():
        async for tok, _ in graph.astream(
            {"messages": body["messages"]},
            {"configurable": {"thread_id": body["thread_id"]}},
            stream_mode="messages",
        ):
            yield tok.content
    return StreamingResponse(gen(), media_type="text/plain")
```

The Next.js equivalent uses `@langchain/langgraph` inside a route handler and returns a `ReadableStream`.

## 13. Common Packaging Pitfalls

- **Importing from `langchain` when you mean `langchain-core`.** The `langchain` package re-exports some types, but depending on it pulls in the full legacy surface. Import from `langchain-core` for anything new.
- **Mixing `langchain-community` with dedicated provider packages.** If `langchain-openai` is installed, import `ChatOpenAI` from there, not from `langchain-community`.
- **Forgetting to install the checkpoint backend.** `langgraph` itself contains only `MemorySaver`. Production needs `langgraph-checkpoint-postgres` or equivalent as a separate install.
- **Outdated types after upgrading.** The TS packages occasionally ship type changes that require a `tsc --noEmit` pass to surface. Do this immediately after a bump, not in production.

## 14. Conclusion

The LangChain + LangGraph SDK surface in April 2026 is large but well-factored. The discipline required is to:

1. Pick the smallest set of packages that meet your needs — `langchain-core` + one provider + one vector store + `langsmith` is enough for a huge class of applications.
2. Add LangGraph when your agent grows a loop.
3. Add `deepagents` when the agent grows a long horizon.
4. Add `langgraph-sdk` only in client applications that consume a deployed server.
5. Pin integration packages; let `-core` float within a minor.

TypeScript parity is high for the core features that matter for web deployments; Python leads on ecosystem breadth, prebuilts, and Deep Agents presets. Choose the language your application lives in; you can build roughly the same thing on either side.

## 15. Further Reading

- **LangChain docs index** — `python.langchain.com/docs/`
- **LangChain.js docs** — `js.langchain.com/docs/`
- **LangGraph docs** — `langchain-ai.github.io/langgraph/`
- **LangGraph.js docs** — `langchain-ai.github.io/langgraphjs/`
- **LangSmith docs** — `docs.smith.langchain.com/`
- **Companion articles**: LangChain overview, LangGraph overview, LangGraph Deep Agents, Python and TypeScript tutorials.
