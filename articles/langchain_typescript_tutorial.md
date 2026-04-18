# TypeScript Tutorial: LangChain.js and LangGraph.js

*April 2026*

## 1. Introduction

This tutorial is the TypeScript counterpart to the Python RAG-agent tutorial in this collection. It builds the same shape of application — a retrieval-augmented research agent with tools, persistence, human-in-the-loop, and streaming — in idiomatic TypeScript using `@langchain/core`, `@langchain/anthropic`, `@langchain/langgraph`, and the surrounding ecosystem.

Where the Python tutorial uses Pydantic, decorators, and asyncio, this tutorial uses Zod, higher-order tool factories, and async iterators. The mental model is identical; the surface syntax differs. If you've read the Python tutorial, you'll recognise every step here.

It assumes Node.js 20+ (or Bun), TypeScript 5.x, and basic familiarity with async/await in JavaScript.

## 2. Setup

### 2.1 Project Init

```bash
mkdir rag-agent-ts && cd rag-agent-ts
npm init -y
npm install typescript tsx @types/node -D
npx tsc --init --target ES2022 --module NodeNext --moduleResolution NodeNext \
  --strict --esModuleInterop --skipLibCheck

# Core
npm install @langchain/core

# Model and embedding providers
npm install @langchain/anthropic @langchain/openai

# Vector store (local, persistent)
npm install @langchain/community chromadb

# Graph + checkpointer
npm install @langchain/langgraph @langchain/langgraph-checkpoint-sqlite

# Search
npm install @langchain/tavily

# Tracing
npm install langsmith

# Runtime deps
npm install zod dotenv
```

Add to `package.json`:

```json
{
  "type": "module",
  "scripts": {
    "dev": "tsx --env-file=.env",
    "build": "tsc"
  }
}
```

### 2.2 Environment

Create `.env`:

```
ANTHROPIC_API_KEY=sk-ant-...
OPENAI_API_KEY=sk-...
TAVILY_API_KEY=tvly-...

LANGSMITH_TRACING=true
LANGSMITH_API_KEY=lsv2_...
LANGSMITH_PROJECT=rag-agent-ts
```

Each script starts by ensuring env is loaded. If you use `--env-file=.env` with tsx/Node 20.6+, nothing else is required; otherwise use `dotenv/config` as the first import.

### 2.3 Sample Corpus

Same as the Python tutorial: create `docs/` with a few `.txt` files of real content, one topic per file.

## 3. Step One: A Plain Chain

```typescript
// src/step1_plain.ts
import { ChatAnthropic } from "@langchain/anthropic";
import { ChatPromptTemplate } from "@langchain/core/prompts";
import { StringOutputParser } from "@langchain/core/output_parsers";

const prompt = ChatPromptTemplate.fromMessages([
  ["system", "You are a precise technical writer. Answer in under three sentences."],
  ["human", "{question}"],
]);
const model = new ChatAnthropic({ model: "claude-sonnet-4-5", temperature: 0 });
const parser = new StringOutputParser();

const chain = prompt.pipe(model).pipe(parser);

console.log(await chain.invoke({ question: "What is PagedAttention?" }));
```

Run `npm run dev src/step1_plain.ts`. The chain composes with `.pipe()` — the JS equivalent of Python's `|` operator, since JS doesn't overload operators.

## 4. Step Two: Add Retrieval

### 4.1 Ingest

```typescript
// src/ingest.ts
import { readdirSync, readFileSync } from "node:fs";
import { join } from "node:path";
import { Chroma } from "@langchain/community/vectorstores/chroma";
import { OpenAIEmbeddings } from "@langchain/openai";
import { RecursiveCharacterTextSplitter } from "@langchain/textsplitters";
import { Document } from "@langchain/core/documents";

const docs: Document[] = readdirSync("docs")
  .filter((f) => f.endsWith(".txt"))
  .map((f) => new Document({
    pageContent: readFileSync(join("docs", f), "utf8"),
    metadata: { source: f },
  }));

const splitter = new RecursiveCharacterTextSplitter({
  chunkSize: 600,
  chunkOverlap: 100,
});
const chunks = await splitter.splitDocuments(docs);

await Chroma.fromDocuments(
  chunks,
  new OpenAIEmbeddings({ model: "text-embedding-3-small" }),
  { collectionName: "rag-agent", url: "http://localhost:8000" },
);

console.log(`Ingested ${chunks.length} chunks.`);
```

(For the local tutorial you can run Chroma with `docker run -p 8000:8000 chromadb/chroma` or swap to `MemoryVectorStore` for a no-dependency alternative.)

### 4.2 RAG Chain

```typescript
// src/step2_rag.ts
import { ChatAnthropic } from "@langchain/anthropic";
import { Chroma } from "@langchain/community/vectorstores/chroma";
import { OpenAIEmbeddings } from "@langchain/openai";
import { ChatPromptTemplate } from "@langchain/core/prompts";
import { StringOutputParser } from "@langchain/core/output_parsers";
import { RunnablePassthrough, RunnableSequence } from "@langchain/core/runnables";
import type { Document } from "@langchain/core/documents";

const store = new Chroma(
  new OpenAIEmbeddings({ model: "text-embedding-3-small" }),
  { collectionName: "rag-agent", url: "http://localhost:8000" },
);
const retriever = store.asRetriever({ k: 4 });

const formatDocs = (docs: Document[]) =>
  docs.map((d) => `[${d.metadata.source}]\n${d.pageContent}`).join("\n\n---\n\n");

const ragPrompt = ChatPromptTemplate.fromTemplate(`
You are a technical assistant. Answer the question using only the context below.
If the context is insufficient, say so and suggest what's missing.
Cite sources inline as [filename].

<context>
{context}
</context>

Question: {question}
`.trim());

const model = new ChatAnthropic({ model: "claude-sonnet-4-5", temperature: 0 });

const rag = RunnableSequence.from([
  {
    context: retriever.pipe(formatDocs),
    question: new RunnablePassthrough(),
  },
  ragPrompt,
  model,
  new StringOutputParser(),
]);

console.log(await rag.invoke("What is PagedAttention?"));
```

**Note.** `RunnableSequence.from([...])` is TS's idiomatic way to wire together a multi-input parallel step (the `{ context, question }` object) followed by sequential stages. The object literal is interpreted as a `RunnableMap` that runs its branches in parallel.

## 5. Step Three: Tools

```typescript
// src/tools.ts
import { tool } from "@langchain/core/tools";
import { z } from "zod";
import { Chroma } from "@langchain/community/vectorstores/chroma";
import { OpenAIEmbeddings } from "@langchain/openai";
import { TavilySearch } from "@langchain/tavily";

const store = new Chroma(
  new OpenAIEmbeddings({ model: "text-embedding-3-small" }),
  { collectionName: "rag-agent", url: "http://localhost:8000" },
);

export const searchCorpus = tool(
  async ({ query }: { query: string }) => {
    const docs = await store.similaritySearch(query, 4);
    if (docs.length === 0) return "No relevant documents found.";
    return docs.map((d) => `[${d.metadata.source}]\n${d.pageContent}`).join("\n\n---\n\n");
  },
  {
    name: "search_corpus",
    description: "Search the local knowledge corpus. Returns up to 4 chunks with source labels.",
    schema: z.object({ query: z.string() }),
  },
);

const tavily = new TavilySearch({ maxResults: 4 });

export const searchWeb = tool(
  async ({ query }: { query: string }) => {
    const hits = await tavily.invoke(query);
    return hits.map((h: any) => `[${h.url}]\n${h.content}`).join("\n\n");
  },
  {
    name: "search_web",
    description: "Search the web. Use only when the corpus is insufficient.",
    schema: z.object({ query: z.string() }),
  },
);

export const deleteDocument = tool(
  async ({ source }: { source: string }) => `Deleted ${source}.`,
  {
    name: "delete_document",
    description: "Permanently delete a document from the corpus. Irreversible — require human approval.",
    schema: z.object({ source: z.string() }),
  },
);

export const TOOLS = [searchCorpus, searchWeb, deleteDocument];
```

The TS `tool(...)` factory is the direct equivalent of Python's `@tool` decorator. The Zod schema replaces Pydantic for argument validation. The function body is the same shape as its Python counterpart.

## 6. Step Four: The LangGraph Agent

```typescript
// src/agent.ts
import { ChatAnthropic } from "@langchain/anthropic";
import { SystemMessage, ToolMessage, type BaseMessage } from "@langchain/core/messages";
import { StateGraph, Annotation, END, START, messagesStateReducer } from "@langchain/langgraph";
import { TOOLS } from "./tools.js";

const SYSTEM = new SystemMessage(`
You are a research assistant. Prefer search_corpus first; only use search_web
if the corpus lacks relevant information. Cite sources with [name] or [url].
delete_document is destructive — call it only when the user has explicitly
asked to delete a named document.
`.trim());

const toolsByName = Object.fromEntries(TOOLS.map((t) => [t.name, t]));

const model = new ChatAnthropic({ model: "claude-sonnet-4-5", temperature: 0 })
  .bindTools(TOOLS);

export const AgentState = Annotation.Root({
  messages: Annotation<BaseMessage[]>({
    reducer: messagesStateReducer,
    default: () => [],
  }),
});

async function callModel(state: typeof AgentState.State) {
  const reply = await model.invoke([SYSTEM, ...state.messages]);
  return { messages: [reply] };
}

async function runTools(state: typeof AgentState.State) {
  const last = state.messages[state.messages.length - 1] as any;
  const out: ToolMessage[] = [];
  for (const call of last.tool_calls ?? []) {
    try {
      const result = await toolsByName[call.name].invoke(call.args);
      out.push(new ToolMessage({ content: String(result), tool_call_id: call.id }));
    } catch (err) {
      out.push(new ToolMessage({ content: `ERROR: ${err}`, tool_call_id: call.id }));
    }
  }
  return { messages: out };
}

function route(state: typeof AgentState.State): "tools" | typeof END {
  const last = state.messages[state.messages.length - 1] as any;
  return last.tool_calls?.length ? "tools" : END;
}

export const graph = new StateGraph(AgentState)
  .addNode("agent", callModel)
  .addNode("tools", runTools)
  .addEdge(START, "agent")
  .addConditionalEdges("agent", route, ["tools", END])
  .addEdge("tools", "agent");

export const app = graph.compile();
```

Drive it:

```typescript
// src/step4_run.ts
import { HumanMessage } from "@langchain/core/messages";
import { app } from "./agent.js";

const stream = await app.stream(
  { messages: [new HumanMessage("Explain PagedAttention.")] },
  { streamMode: "updates" },
);

for await (const chunk of stream) {
  console.log(chunk);
}
```

**Equivalent to Python in every respect.** `Annotation.Root` replaces `TypedDict`, builder-chain `.addNode(...).addEdge(...)` replaces `graph.add_node(...)`, and `typeof AgentState.State` gives you the derived type for state-shaped function parameters.

## 7. Step Five: Persistence and Threads

```typescript
// src/persistentAgent.ts
import { SqliteSaver } from "@langchain/langgraph-checkpoint-sqlite";
import { graph } from "./agent.js";

const checkpointer = SqliteSaver.fromConnString("agent.sqlite");

export const app = graph.compile({ checkpointer });
```

Multi-turn:

```typescript
// src/step5_run.ts
import { HumanMessage } from "@langchain/core/messages";
import { app } from "./persistentAgent.js";

const config = { configurable: { thread_id: "user-42" } };

await app.invoke({ messages: [new HumanMessage("What is FlashAttention?")] }, config);
await app.invoke(
  { messages: [new HumanMessage("How does that compare to PagedAttention?")] },
  config,
);

const state = await app.getState(config);
console.log(state.values.messages.length, "messages so far");
```

## 8. Step Six: Human-in-the-Loop

```typescript
// src/hitlAgent.ts
import { ChatAnthropic } from "@langchain/anthropic";
import { SystemMessage, ToolMessage, type BaseMessage } from "@langchain/core/messages";
import {
  StateGraph, Annotation, END, START, messagesStateReducer, interrupt,
} from "@langchain/langgraph";
import { SqliteSaver } from "@langchain/langgraph-checkpoint-sqlite";
import { TOOLS } from "./tools.js";

const DESTRUCTIVE = new Set(["delete_document"]);
const toolsByName = Object.fromEntries(TOOLS.map((t) => [t.name, t]));
const model = new ChatAnthropic({ model: "claude-sonnet-4-5", temperature: 0 }).bindTools(TOOLS);
const SYSTEM = new SystemMessage("You are a research assistant. ...");

const State = Annotation.Root({
  messages: Annotation<BaseMessage[]>({ reducer: messagesStateReducer, default: () => [] }),
});

async function callModel(state: typeof State.State) {
  return { messages: [await model.invoke([SYSTEM, ...state.messages])] };
}

async function approveDestructive(state: typeof State.State) {
  const last = state.messages[state.messages.length - 1] as any;
  const destructive = (last.tool_calls ?? []).filter((c: any) => DESTRUCTIVE.has(c.name));
  if (destructive.length === 0) return {};
  const decision = interrupt({
    prompt: "Approve these destructive tool calls?",
    calls: destructive,
  }) as string;
  if (decision !== "approve") {
    return {
      messages: destructive.map(
        (c: any) => new ToolMessage({ content: "User refused.", tool_call_id: c.id }),
      ),
    };
  }
  return {};
}

async function runTools(state: typeof State.State) {
  const last = state.messages[state.messages.length - 1] as any;
  const answered = new Set(
    state.messages.filter((m: any) => m._getType?.() === "tool").map((m: any) => m.tool_call_id),
  );
  const out: ToolMessage[] = [];
  for (const call of last.tool_calls ?? []) {
    if (answered.has(call.id)) continue;
    try {
      const result = await toolsByName[call.name].invoke(call.args);
      out.push(new ToolMessage({ content: String(result), tool_call_id: call.id }));
    } catch (err) {
      out.push(new ToolMessage({ content: `ERROR: ${err}`, tool_call_id: call.id }));
    }
  }
  return { messages: out };
}

function route(state: typeof State.State): "approve" | typeof END {
  const last = state.messages[state.messages.length - 1] as any;
  return last.tool_calls?.length ? "approve" : END;
}

const g = new StateGraph(State)
  .addNode("agent", callModel)
  .addNode("approve", approveDestructive)
  .addNode("tools", runTools)
  .addEdge(START, "agent")
  .addConditionalEdges("agent", route, ["approve", END])
  .addEdge("approve", "tools")
  .addEdge("tools", "agent");

export const app = g.compile({ checkpointer: SqliteSaver.fromConnString("agent.sqlite") });
```

Drive it:

```typescript
// src/step6_run.ts
import { HumanMessage } from "@langchain/core/messages";
import { Command } from "@langchain/langgraph";
import { app } from "./hitlAgent.js";

const config = { configurable: { thread_id: "hitl-demo" } };

await app.invoke(
  { messages: [new HumanMessage("Please delete paged_attention.txt.")] },
  config,
);

const state = await app.getState(config);
if (state.next?.includes("approve")) {
  const last = state.values.messages[state.values.messages.length - 1] as any;
  console.log("Awaiting approval for:", last.tool_calls);
  await app.invoke(new Command({ resume: "approve" }), config);
}
const final = await app.getState(config);
console.log(final.values.messages.at(-1)?.content);
```

**Note.** TS's `interrupt()` looks exactly like Python's; `Command({ resume: "approve" })` is the JS instantiation of the same class. The pattern is identical down to field names.

## 9. Step Seven: Streaming for UIs

```typescript
// src/step7_stream.ts
import { HumanMessage } from "@langchain/core/messages";
import { app } from "./hitlAgent.js";

const config = { configurable: { thread_id: "stream-demo" } };

const events = app.streamEvents(
  { messages: [new HumanMessage("What is speculative decoding?")] },
  { ...config, version: "v2" },
);

for await (const event of events) {
  if (event.event === "on_chat_model_stream") {
    const content = (event.data as any).chunk?.content;
    if (content) process.stdout.write(content);
  } else if (event.event === "on_tool_start") {
    process.stdout.write(`\n[tool: ${event.name}(${JSON.stringify(event.data.input)})]\n`);
  } else if (event.event === "on_tool_end") {
    process.stdout.write(`[tool ${event.name} done]\n`);
  }
}
```

For a Next.js route handler that streams SSE:

```typescript
// app/api/chat/route.ts
import { HumanMessage } from "@langchain/core/messages";
import { app } from "@/server/hitlAgent";

export async function POST(req: Request) {
  const { thread_id, messages } = await req.json();
  const config = { configurable: { thread_id } };
  const encoder = new TextEncoder();

  const stream = new ReadableStream({
    async start(controller) {
      for await (const [tok] of app.stream(
        { messages: messages.map((m: any) => new HumanMessage(m.content)) },
        { ...config, streamMode: "messages" },
      )) {
        controller.enqueue(encoder.encode(`data: ${JSON.stringify({ content: tok.content })}\n\n`));
      }
      controller.close();
    },
  });
  return new Response(stream, { headers: { "Content-Type": "text/event-stream" } });
}
```

## 10. Step Eight: Graceful Failures

```typescript
import { ChatAnthropic } from "@langchain/anthropic";
import { ChatOpenAI } from "@langchain/openai";

const primary = new ChatAnthropic({ model: "claude-sonnet-4-5", temperature: 0 });
const backup  = new ChatOpenAI({ model: "gpt-4o", temperature: 0 });

export const model = primary
  .withRetry({ stopAfterAttempt: 3 })
  .withFallbacks({ fallbacks: [backup] });
```

Bind this `model` with tools before handing it to the graph.

## 11. Step Nine: Tracing with Tags

```typescript
const config = {
  configurable: { thread_id },
  tags: ["prod", "rag-agent"],
  metadata: { user_id: user.id, org_id: org.id, feature_flag: "rag-v2" },
};

await app.invoke(input, config);
```

Identical to Python. LangSmith filters by tag, groups by metadata.

## 12. Step Ten: Production Layout

```
rag-agent-ts/
  src/
    graph.ts           # StateGraph + compile
    tools/
      corpus.ts
      web.ts
      admin.ts
    prompts/
      system.md
      rag.md
    config.ts
    ingest.ts
  tests/
    graph.test.ts
  langgraph.json
  package.json
  tsconfig.json
```

`langgraph.json` for the LangGraph CLI:

```json
{
  "node_version": "20",
  "dependencies": ["."],
  "graphs": { "agent": "./src/graph.ts:app" },
  "env": ".env"
}
```

`langgraph dev` detects the TS project, runs `tsx` under the hood, boots a local LangGraph Server on port 2024, and Studio opens in your browser.

## 13. Step Eleven: Testing

For tool-calling flows use a fake model that returns scripted AI messages:

```typescript
// tests/graph.test.ts
import { test, expect } from "vitest";
import { FakeListChatModel } from "@langchain/core/utils/testing";
import { HumanMessage } from "@langchain/core/messages";
import { buildGraph } from "../src/graph.js";

test("simple turn without tools", async () => {
  const model = new FakeListChatModel({ responses: ["It's complicated."] });
  const app = buildGraph(model).compile();
  const out = await app.invoke({ messages: [new HumanMessage("hi")] });
  expect(out.messages.at(-1)?.content).toBe("It's complicated.");
});
```

(`buildGraph` should be a factory in `graph.ts` that accepts a model so tests can inject the fake. Export both `buildGraph` and a default `app` for production use.)

## 14. TS-Specific Gotchas

**Annotations vs type literals.** `typeof AgentState.State` is the *derived* state type; always use that, not a hand-rolled interface. Keeping the annotation the single source of truth avoids drift.

**ESM vs CommonJS.** LangChain.js is ESM-first. Set `"type": "module"` in `package.json`. Use `.js` in relative imports even when the file is `.ts` — that is the TypeScript ESM convention, not a typo.

**Tool arg types.** Zod's `z.infer<typeof schema>` gives you the TS type for tool args. If you type the argument of your tool function directly (as in `src/tools.ts`), make sure it matches the schema — TS won't check this for you automatically.

**Message type guards.** `BaseMessage` union doesn't carry `tool_calls` as a typed field. Either narrow with `AIMessage` imports and `instanceof`, or use the `any` cast shown in the examples. In production code, the instanceof approach is preferable.

## 15. Parity With Python

| Feature | Python | TypeScript |
|---|---|---|
| Chat model | `ChatAnthropic(...).bind_tools(tools)` | `new ChatAnthropic(...).bindTools(tools)` |
| Tool definition | `@tool` decorator | `tool(fn, { name, description, schema })` |
| Argument schema | Pydantic | Zod |
| State | `TypedDict` + `Annotated` | `Annotation.Root({...})` |
| Pipe operator | `a \| b` | `a.pipe(b)` |
| Stream | `async for chunk in app.astream(...)` | `for await (const c of app.stream(...))` |
| Structured events | `app.astream_events(...)` | `app.streamEvents(...)` |
| Interrupt | `interrupt(payload)` | `interrupt(payload)` |
| Resume | `Command(resume=...)` | `new Command({ resume: ... })` |
| Checkpoint | `SqliteSaver.from_conn_string(...)` | `SqliteSaver.fromConnString(...)` |
| Prebuilt ReAct | `create_react_agent(...)` | `createReactAgent(...)` |

The two SDKs are the same application framework expressed in two languages; migrating between them is mechanical.

## 16. When to Choose TS Over Python

- You already have a Node/TS backend and want to keep one runtime.
- You're deploying to edge runtimes (Cloudflare Workers, Vercel Edge, Deno Deploy) that are Node-compatible but not Python-compatible.
- You want a single codebase shared between your agent and your Next.js/SvelteKit/Remix frontend.
- You prefer Zod's type-narrowing ergonomics to Pydantic's.

- Python leads for ecosystem breadth (data science, scientific computing), for Deep Agents (the `deepagents` preset is Python-only today), and for early access to new LangGraph features.

Neither is wrong. Pick the language your surrounding stack is in.

## 17. Conclusion

The TS path is the same path:

1. LCEL first — compose with `.pipe()`.
2. Switch to LangGraph when the flow cycles.
3. `SqliteSaver` → `PostgresSaver` when you go to production.
4. `interrupt()` for HITL on destructive actions.
5. `streamEvents({ version: "v2" })` for token + tool streaming.
6. `withRetry().withFallbacks()` for reliability.
7. `tags` + `metadata` on every invoke for traceability.

Deploy via LangGraph Platform or self-host with `langgraph build`/`up`. If your task horizon grows, port the Deep Agents pattern from the Python version — the pieces (todo-list tool, virtual FS tools, sub-agent dispatch) are all expressible in TS using primitives from `@langchain/langgraph`, even without a packaged preset.

## 18. Further Reading

- **LangChain.js docs** — `js.langchain.com/docs/`
- **LangGraph.js docs** — `langchain-ai.github.io/langgraphjs/`
- **LangSmith JS** — `docs.smith.langchain.com/`
- **Companion articles**: LangChain overview, LangGraph overview, LangGraph Deep Agents, SDKs, Python tutorial.
