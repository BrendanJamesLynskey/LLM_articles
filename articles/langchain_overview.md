# LangChain: Chains, Composability, and LCEL

*April 2026*

## 1. Introduction

LangChain began in late 2022 as a thin Python wrapper for calling OpenAI models, chaining prompts together, and plugging in a handful of external tools. In the three-and-a-bit years since, it has expanded into the most widely used LLM application framework, a sprawling integration layer that covers nearly every commercial and open-source model provider, vector store, retriever, document loader, and evaluation harness. The framework's scope is simultaneously its greatest strength and its most common criticism: LangChain makes it trivially easy to wire up a prototype, and equally easy to end up with a tower of abstractions that obscures what the underlying model call actually looks like.

This article is a focused technical tour of the parts of LangChain that matter for production engineering: the Runnable interface and LCEL (LangChain Expression Language), the unified chat-model abstraction, prompt templates, output parsers, tools, retrievers, and the package layout that separates stable interfaces from community integrations. It deliberately does not try to cover every module — the integration surface is measured in hundreds of packages — but instead concentrates on the core that every LangChain user meets on day one.

LangGraph, Deep Agents, LangSmith, and the LangGraph Platform are covered in companion articles. This one is strictly about the classic LangChain stack: prompts, models, chains, and the LCEL composition model that binds them.

## 2. What LangChain Is For

### 2.1 The Wrapper Problem

Every LLM provider ships a slightly different SDK. Anthropic's messages API, OpenAI's chat completions and responses APIs, Google's Vertex and Gemini APIs, Mistral's chat endpoint, xAI's Grok API, Cohere's, and the OSS world's OpenAI-compatible `llama.cpp`, `vLLM`, `TGI`, and `Ollama` all present subtly different request shapes, streaming semantics, tool-call formats, and error surfaces.

Writing production code directly against a provider SDK locks you to that provider. Writing against an internal abstraction forces every team to reinvent the same adapter layer. LangChain's first job is to collapse this zoo into a single interface — `BaseChatModel` — with consistent methods (`invoke`, `stream`, `batch`, `ainvoke`, `astream`) and consistent return types.

### 2.2 The Composition Problem

Most non-trivial LLM applications are not single model calls. They are pipelines: fetch documents → rerank → format into a prompt → call the model → parse structured output → hand off to a tool → feed the result back to the model. LangChain's second job is to make these pipelines composable, observable, and swappable.

LCEL is the answer. Every LangChain primitive implements a common `Runnable` protocol. Runnables compose with the pipe operator (`|`) into larger Runnables, which themselves support streaming, async, batching, retries, fallbacks, and automatic tracing.

### 2.3 The Integration Problem

A production RAG system touches a chunker, an embedding model, a vector store, a reranker, a chat model, a prompt template, an output parser, and a tracing backend. Writing all of these from scratch is a multi-week effort. LangChain's community packages provide pre-built integrations for most of them, so the initial wiring is often a dozen lines of code rather than a few hundred.

The tradeoff is that the integration layer moves fast. Breaking changes in a third-party vector store or model SDK occasionally require users to pin package versions or patch adapter code. This is the cost of riding on a large shared ecosystem, and it is why LangChain split into narrowly scoped sub-packages in 2024.

## 3. Package Layout

In April 2026 LangChain ships as a constellation of small packages rather than a single monolith. Understanding which package each symbol lives in is the difference between a clean `pyproject.toml` and a cascade of conflicting transitive dependencies.

| Package | Role | Typical imports |
|---|---|---|
| `langchain-core` | Runnables, base interfaces, messages, prompt templates, output parsers. No integrations. | `BaseChatModel`, `Runnable`, `PromptTemplate`, `StrOutputParser` |
| `langchain` | High-level convenience: legacy chain classes, agent types, retrievers. | `create_retrieval_chain`, `Agent`, `RetrievalQA` |
| `langchain-community` | Third-party integrations that do not warrant a dedicated package. | Older loaders, older vector stores |
| `langchain-openai` | OpenAI-specific chat models, embeddings, image generation. | `ChatOpenAI`, `OpenAIEmbeddings` |
| `langchain-anthropic` | Claude chat models, Claude tool use. | `ChatAnthropic` |
| `langchain-google-genai` / `langchain-google-vertexai` | Gemini APIs. | `ChatGoogleGenerativeAI` |
| `langchain-aws` | Bedrock, SageMaker. | `ChatBedrock` |
| `langchain-chroma` / `langchain-pinecone` / `langchain-qdrant` / ... | Vector store integrations. | `Chroma`, `PineconeVectorStore` |
| `langchain-text-splitters` | Chunkers for documents and code. | `RecursiveCharacterTextSplitter` |
| `langgraph`, `langgraph-sdk`, `langgraph-cli`, `deepagents` | Separate graph/agent stack. Covered in companion articles. | — |
| `langsmith` | Tracing and evaluation client. Standalone. | `Client`, `traceable` |

**Rule of thumb**: if your application is using only a small number of integrations, import them directly from their dedicated package. Avoid depending on `langchain-community` unless you need a loader or store that hasn't earned its own package yet.

## 4. The Runnable and LCEL

LCEL is not a DSL in the traditional sense. It is a set of classes that implement the `Runnable` protocol and overload the `|` operator to compose. Every LangChain object you are likely to touch is a Runnable.

### 4.1 The Runnable Protocol

```python
from langchain_core.runnables import Runnable

class Runnable(Generic[Input, Output]):
    def invoke(self, input: Input, config: RunnableConfig | None = None) -> Output: ...
    def stream(self, input: Input, config: RunnableConfig | None = None) -> Iterator[Output]: ...
    def batch(self, inputs: list[Input], config: RunnableConfig | None = None) -> list[Output]: ...
    async def ainvoke(self, input: Input, config: RunnableConfig | None = None) -> Output: ...
    async def astream(self, input: Input, config: RunnableConfig | None = None) -> AsyncIterator[Output]: ...
    async def abatch(self, inputs: list[Input], config: RunnableConfig | None = None) -> list[Output]: ...
```

Implementing `invoke` is enough to get reasonable default implementations of the rest. Chat models, prompt templates, retrievers, tools, and output parsers all implement this interface, which is why they compose uniformly.

### 4.2 The Pipe Operator

`a | b` constructs a `RunnableSequence` whose `invoke(x)` is `b.invoke(a.invoke(x))`. The pipe is left-to-right data flow — the output of the left-hand Runnable becomes the input of the right-hand Runnable.

```python
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a concise technical writer."),
    ("human", "Summarize this paragraph in one sentence:\n\n{paragraph}"),
])
model = ChatOpenAI(model="gpt-4o", temperature=0)
parser = StrOutputParser()

chain = prompt | model | parser

chain.invoke({"paragraph": "LangChain is ..."})
# -> "LangChain is an LLM application framework..."
```

### 4.3 RunnableParallel

Parallel branches are expressed with a dict or with `RunnableParallel`. Every value in the dict runs concurrently on the same input.

```python
from langchain_core.runnables import RunnableParallel

analysis = RunnableParallel(
    summary   = summary_prompt   | model | parser,
    sentiment = sentiment_prompt | model | parser,
    keywords  = keywords_prompt  | model | parser,
)
# analysis.invoke({"text": "..."}) -> {"summary": "...", "sentiment": "...", "keywords": "..."}
```

### 4.4 RunnablePassthrough and RunnableLambda

`RunnablePassthrough` forwards its input unchanged. `RunnableLambda` wraps an arbitrary Python function as a Runnable. Together they let you splice plain functions into an LCEL pipeline without breaking the pipe-operator flow.

```python
from langchain_core.runnables import RunnablePassthrough, RunnableLambda

def fetch_docs(question: str) -> list[str]:
    return retriever.invoke(question)

rag = (
    {"context": RunnableLambda(fetch_docs), "question": RunnablePassthrough()}
    | rag_prompt
    | model
    | parser
)
```

### 4.5 Config, Tags, and Callbacks

Every `invoke` / `stream` accepts a `RunnableConfig` with `tags`, `metadata`, `callbacks`, `run_name`, `max_concurrency`, and `recursion_limit`. Tags and metadata propagate through the entire chain and show up in LangSmith traces, making them invaluable for production triage.

```python
result = chain.invoke(
    {"paragraph": text},
    config={"tags": ["prod", "summariser"], "metadata": {"user_id": user.id}},
)
```

### 4.6 Streaming

`.stream()` and `.astream()` return an iterator of partial outputs. For simple LCEL pipelines, streaming propagates cleanly: the model emits tokens, the output parser passes them through, and the consumer sees them incrementally.

```python
for chunk in chain.stream({"paragraph": text}):
    print(chunk, end="", flush=True)
```

Streaming works even across parallel branches. The richer `astream_events` API surfaces structured events (`on_llm_start`, `on_llm_stream`, `on_tool_end`, ...) for every step in the chain, which is the usual way to implement UI streaming that highlights tool calls, retrieved documents, and intermediate thoughts.

### 4.7 with_retry, with_fallbacks, with_config

Every Runnable has a few small builder methods that return a new wrapped Runnable:

- `runnable.with_retry(stop_after_attempt=3)` — wrap with tenacity-style retries.
- `runnable.with_fallbacks([backup_model])` — if the primary fails, try the fallbacks.
- `runnable.with_config(tags=[...])` — attach default config.
- `runnable.with_types(input_type=..., output_type=...)` — refine schemas for tracing.

These are a large part of what makes LCEL production-worthy: reliability patterns are a one-line transformation of an existing chain rather than a rewrite.

## 5. Chat Models

`BaseChatModel` is the uniform front door for every provider.

### 5.1 Messages

Messages are Pydantic models: `SystemMessage`, `HumanMessage`, `AIMessage`, `ToolMessage`, `FunctionMessage` (legacy). Each message has a `content`, optional `name`, and — for `AIMessage` — `tool_calls` and `usage_metadata`. The `content` can be a string or a list of content blocks (text, image_url, tool_use, tool_result), which is how LangChain supports multimodal and thinking blocks consistently across providers.

```python
from langchain_core.messages import SystemMessage, HumanMessage

response = model.invoke([
    SystemMessage(content="You are a helpful assistant."),
    HumanMessage(content="What is the capital of Portugal?"),
])
# response is an AIMessage
```

### 5.2 Structured Output

`model.with_structured_output(Schema)` returns a Runnable that guarantees the output validates against `Schema`. The implementation uses provider-native JSON modes when available (OpenAI's `response_format={"type": "json_schema"}`, Anthropic tool use) and falls back to a parser that retries on validation failure.

```python
from pydantic import BaseModel

class Person(BaseModel):
    name: str
    age: int

extractor = model.with_structured_output(Person)
extractor.invoke("Alice is 32 and lives in Lisbon.")
# -> Person(name="Alice", age=32)
```

### 5.3 Tool Binding

`model.bind_tools([tool_a, tool_b])` attaches a list of tools to a model, producing a new model whose `invoke` will return `AIMessage.tool_calls` when the model decides to call one.

```python
from langchain_core.tools import tool

@tool
def get_weather(city: str) -> str:
    """Return the current weather in a named city."""
    ...

agent_model = model.bind_tools([get_weather])
reply = agent_model.invoke("What's the weather in Porto?")
# reply.tool_calls == [{"name": "get_weather", "args": {"city": "Porto"}, "id": "..."}]
```

Tool-calling loops (call model → execute tools → feed results back → call model again) are usually built with LangGraph rather than pure LCEL, because the loop is a cyclic graph, not a linear chain. See the LangGraph overview for the full pattern.

## 6. Prompt Templates

### 6.1 ChatPromptTemplate

`ChatPromptTemplate.from_messages` takes a list of `(role, template_string)` tuples and returns a Runnable that, given a dict of variables, yields a `ChatPromptValue` — a sequence of messages that a chat model can consume.

```python
prompt = ChatPromptTemplate.from_messages([
    ("system", "You translate English to {language}."),
    ("human", "{text}"),
])

prompt.invoke({"language": "French", "text": "Good morning."})
# -> ChatPromptValue(messages=[SystemMessage(...), HumanMessage(...)])
```

### 6.2 MessagesPlaceholder

`MessagesPlaceholder("history")` slots a list of messages into the prompt at a designated spot. This is how you feed a conversation's prior turns into the next call.

```python
from langchain_core.prompts import MessagesPlaceholder

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant."),
    MessagesPlaceholder("history"),
    ("human", "{input}"),
])
```

### 6.3 Few-Shot Prompting

`FewShotChatMessagePromptTemplate` composes a fixed or dynamically retrieved set of examples into the prompt. Example selectors (length-based, semantic-similarity-based) choose examples from a pool at run time, and are themselves Runnables.

## 7. Output Parsers

Output parsers convert an `AIMessage` into a structured value. Common built-ins:

- `StrOutputParser` — `AIMessage.content` as a plain string.
- `JsonOutputParser` — parse JSON, optionally against a Pydantic schema.
- `PydanticOutputParser` — parse and validate against a Pydantic model.
- `CommaSeparatedListOutputParser`, `XMLOutputParser`, `YamlOutputParser`.

For most production cases, prefer `model.with_structured_output(schema)` over manual parsers — it uses provider-native structured output where supported and has fewer failure modes than prose-based parsing. Output parsers remain useful when the provider lacks a structured mode, when you need streaming JSON (`JsonOutputParser` streams incremental dicts as tokens arrive), or when the output format is an exotic one that the provider cannot enforce.

## 8. Retrievers and RAG

### 8.1 The Retriever Interface

`BaseRetriever` has a single method `invoke(query: str) -> list[Document]`. Vector stores, BM25 indexes, hybrid search engines, ensembles, parent-document retrievers, multi-query retrievers, contextual-compression retrievers, and self-query retrievers all implement this interface.

```python
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

vstore = Chroma(collection_name="docs", embedding_function=OpenAIEmbeddings())
retriever = vstore.as_retriever(search_kwargs={"k": 8})
docs = retriever.invoke("how does paged attention work?")
```

### 8.2 A Minimal RAG Chain

```python
from langchain_core.runnables import RunnablePassthrough

def format_docs(docs):
    return "\n\n".join(d.page_content for d in docs)

rag_prompt = ChatPromptTemplate.from_template("""
Answer the question from the context. If you don't know, say so.

<context>
{context}
</context>

Question: {question}
""")

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | rag_prompt
    | model
    | StrOutputParser()
)

rag_chain.invoke("What is PagedAttention?")
```

This seven-line chain is a complete RAG system. Replacing the retriever with a hybrid or reranker-augmented one is a single substitution; swapping the model is one import away. That is the payoff of the Runnable interface.

### 8.3 Document Loaders and Splitters

`langchain-community` and the dedicated loader packages provide loaders for PDFs, websites, Notion, Slack, Google Drive, S3, and roughly a hundred other sources. Each returns a list of `Document` objects with `page_content` and `metadata`.

`langchain-text-splitters` provides chunkers: `RecursiveCharacterTextSplitter`, `TokenTextSplitter`, `MarkdownHeaderTextSplitter`, language-aware splitters for Python, JavaScript, and several others. The recursive character splitter is the default for most text.

## 9. Tools

### 9.1 The @tool Decorator

```python
from langchain_core.tools import tool

@tool
def search_arxiv(query: str, max_results: int = 5) -> str:
    """Search arxiv.org and return titles + abstracts."""
    ...
```

The decorator introspects the function signature and docstring to generate a JSON-schema description that the model can consume. Type hints determine the parameter schema; the docstring becomes the tool description.

### 9.2 StructuredTool and BaseTool

For tools with complex argument schemas, use `StructuredTool.from_function(func, args_schema=MyPydanticModel)`. For stateful tools, subclass `BaseTool` and implement `_run` and `_arun`.

### 9.3 Toolkits

A toolkit is a collection of related tools (an SQL toolkit, a GitHub toolkit, a filesystem toolkit). Toolkits are the integration-layer equivalent of an MCP server — but they run in-process rather than over a protocol, and they are Python-specific.

## 10. Memory (Legacy) and History

The legacy `ConversationBufferMemory` / `ConversationSummaryMemory` classes still exist in `langchain` but are deprecated for new code. The current idiom is to store chat history explicitly (in a database, a Redis list, a `RunnableWithMessageHistory` wrapper, or — for agentic workflows — in LangGraph's checkpointer) and inject it into the prompt via `MessagesPlaceholder`.

```python
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import RedisChatMessageHistory

def get_history(session_id: str):
    return RedisChatMessageHistory(session_id, url="redis://localhost:6379")

chain_with_history = RunnableWithMessageHistory(
    chain,
    get_history,
    input_messages_key="input",
    history_messages_key="history",
)

chain_with_history.invoke(
    {"input": "Hi there"},
    config={"configurable": {"session_id": "user-42"}},
)
```

For any agent more complex than a single-turn Q&A, move history management into LangGraph — its checkpointer handles it natively and also gives you human-in-the-loop and time-travel for free.

## 11. Callbacks and Tracing

LangChain's callback system is how tracing, streaming, metrics, and debugging hook into a chain without modifying its code. A `BaseCallbackHandler` has methods like `on_llm_start`, `on_llm_new_token`, `on_tool_end`, `on_chain_error`. LangSmith's tracing client is a callback handler; so are most third-party observability integrations.

Enable tracing with two environment variables:

```bash
export LANGSMITH_TRACING=true
export LANGSMITH_API_KEY=lsv2_...
export LANGSMITH_PROJECT=my-project
```

From that point on every LCEL chain run is captured in LangSmith with a hierarchical trace, token counts, latency, and the full inputs/outputs of every step.

## 12. LCEL vs Legacy Chain Classes

LangChain's pre-LCEL history is full of specific chain classes: `LLMChain`, `SimpleSequentialChain`, `SequentialChain`, `MapReduceChain`, `RetrievalQA`, `ConversationalRetrievalChain`, and many more. These classes still exist and are imported by a large amount of legacy code, but they are deprecated in favour of LCEL equivalents.

| Legacy class | LCEL equivalent |
|---|---|
| `LLMChain(prompt, llm)` | `prompt \| llm \| StrOutputParser()` |
| `SimpleSequentialChain([a, b])` | `a \| b` |
| `RetrievalQA.from_chain_type(llm, retriever=r)` | The LCEL RAG chain in §8.2 |
| `ConversationalRetrievalChain` | `RunnableWithMessageHistory` wrapping a RAG chain |

Greenfield code should use LCEL everywhere. The legacy classes are retained for backward compatibility but receive no new features.

## 13. Observability and Production Patterns

### 13.1 Trace Everything

Enable LangSmith tracing from day one. Agent bugs are notoriously hard to reproduce without a full trace; enabling it later means you have no data for the incidents that already happened.

### 13.2 Tag and Metadata Every Call

Attach `tags` and `metadata` at every `invoke` call site. In production you will want to filter traces by user, environment, feature flag, experiment, and cohort. Doing this retroactively is painful; doing it at call time is one line of code.

### 13.3 Wrap with Retries and Fallbacks at the Model Boundary

```python
primary = ChatAnthropic(model="claude-sonnet-4-5")
backup  = ChatOpenAI(model="gpt-4o")

model = primary.with_retry(stop_after_attempt=3).with_fallbacks([backup])
```

This pattern handles transient rate-limit errors and hard provider outages with zero application code.

### 13.4 Lock Down Package Versions

Because `langchain-community` and dedicated integration packages move quickly, pin minor versions in `pyproject.toml` and upgrade on a cadence with regression tests. `langchain-core` is far more stable and can usually be allowed to float within a minor version.

### 13.5 Prefer `.bind_tools` + LangGraph Over Agent Classes

The legacy `AgentExecutor` and the assorted `create_*_agent` helpers still exist, but any non-trivial agent should be built with LangGraph. LCEL gives you the building blocks (models, prompts, tools, parsers); LangGraph gives you the loop. Mixing them is the idiomatic 2026 pattern.

## 14. Strengths and Criticisms

### 14.1 Strengths

- **Breadth of integrations** — if it has an SDK, there is probably a LangChain adapter for it.
- **Stable core abstractions** — Runnable, BaseChatModel, and BaseRetriever have been load-bearing interfaces since 2023 and have not had a major breaking change.
- **Observability for free** — LangSmith integration is a two-env-var change.
- **Ecosystem gravity** — documentation, blog posts, Stack Overflow answers, and hiring pool are all larger than any competing framework.

### 14.2 Criticisms

- **Abstraction sprawl** — the library exposes many ways to do the same thing; choosing between them is its own skill.
- **Package fragmentation** — resolving the right combination of `langchain`, `langchain-core`, `langchain-community`, and per-provider packages is an occasional source of dependency conflicts.
- **Hidden prompt templates** — some high-level helpers (legacy `RetrievalQA`, some agent factories) embed opaque prompts. Production systems should own the prompts explicitly.
- **Trace overhead** — LangSmith tracing adds latency and a dependency. It is configurable, but non-trivial throughput systems need to test the overhead.

## 15. When to Use LangChain vs Alternatives

**Use LangChain when:** you want a broad integration layer, you plan to use LangGraph for agent loops, LangSmith for observability, and you value ecosystem maturity over minimalism.

**Consider alternatives when:** you have extreme latency requirements (the direct provider SDK is always the fastest), you only use one model provider and don't need the abstraction, or your use case is well-served by a narrower framework (Pydantic AI for type-safe single-shot extraction, Haystack for search-centric pipelines, LlamaIndex for RAG-heavy workloads).

In 2026 the most common production pattern is a thin LangChain layer for prompts/models/retrievers plus a LangGraph graph for the agent loop, traced end-to-end with LangSmith. The rest of the framework is optional and should be adopted only when it earns its place.

## 16. Conclusion

LangChain's value in 2026 is the same as it was in 2023 — a shared abstraction over a fast-moving ecosystem — but the centre of gravity has shifted. The legacy chain classes are deprecated; LCEL is the composition model; agent loops live in LangGraph; tracing lives in LangSmith. The core stays small: a `BaseChatModel`, a `Runnable`, a `BaseRetriever`, a `BaseTool`, and the pipe operator that binds them.

Start with that core. Add integrations when you need them. Move agent logic into LangGraph as soon as it has a loop. Trace everything. That is roughly the entire playbook.

## 17. Further Reading

- **LangChain conceptual guide** — `python.langchain.com/docs/concepts/`
- **LCEL reference** — `python.langchain.com/docs/concepts/lcel/`
- **Runnable API** — `python.langchain.com/api_reference/core/runnables.html`
- **Integration catalogue** — `python.langchain.com/docs/integrations/`
- **Companion articles in this collection**: LangGraph overview, LangGraph Deep Agents, LangChain/LangGraph SDKs, and the Python and TypeScript tutorials.
