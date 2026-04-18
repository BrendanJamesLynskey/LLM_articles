# Python Tutorial: Building a RAG Agent with LangChain and LangGraph

*April 2026*

## 1. Introduction

This tutorial is an end-to-end worked example of a LangChain + LangGraph application in Python. The goal is a practical RAG-augmented research agent: it answers user questions about a corpus of documents, searches the web when the corpus is insufficient, and gracefully hands off to a human when it encounters a destructive operation. Along the way we touch every concept that matters for production: LCEL composition, chat models with tools, retrievers, LangGraph state and checkpointing, human-in-the-loop, streaming, and tracing.

The tutorial is deliberately incremental. Each section produces a working artifact and builds on the previous one. By the end you have a production-shaped agent of roughly 150 lines of code.

It assumes Python 3.11+, basic familiarity with async/await, and that you have API keys for Anthropic (or OpenAI) and, optionally, Tavily for web search.

## 2. Setup

### 2.1 Environment

```bash
mkdir rag-agent && cd rag-agent
python -m venv .venv && source .venv/bin/activate

pip install \
    langchain-core \
    langchain-anthropic \
    langchain-openai \
    langchain-chroma \
    langchain-text-splitters \
    langchain-tavily \
    langgraph \
    langgraph-checkpoint-sqlite \
    langsmith
```

Create a `.env`:

```
ANTHROPIC_API_KEY=sk-ant-...
OPENAI_API_KEY=sk-...        # for embeddings
TAVILY_API_KEY=tvly-...      # optional, for web search

LANGSMITH_TRACING=true
LANGSMITH_API_KEY=lsv2_...
LANGSMITH_PROJECT=rag-agent-tutorial
```

Load it at the top of each script:

```python
from dotenv import load_dotenv
load_dotenv()
```

### 2.2 Sample Corpus

For the tutorial, we use a handful of text files as the knowledge base. Create `docs/`:

```
docs/
  paged_attention.txt
  flash_attention.txt
  speculative_decoding.txt
```

Fill each with a paragraph or two of real content — for example, copy from the matching articles in this collection. The corpus content does not matter for the mechanics; any domain will do.

## 3. Step One: A Plain Chain

Start with the simplest possible pipeline: read a question, stuff it into a prompt, call a model, parse the output. This is LCEL at its most minimal.

```python
# step1_plain.py
from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a precise technical writer. Answer in under three sentences."),
    ("human", "{question}"),
])
model = ChatAnthropic(model="claude-sonnet-4-5", temperature=0)
chain = prompt | model | StrOutputParser()

print(chain.invoke({"question": "What is PagedAttention?"}))
```

Run it. You should see a short answer. Check LangSmith — you'll see a trace with three spans (prompt, model, parser) plus the full input and output.

**What we've built.** A Runnable that composes three Runnables. The `|` operator threads the data through in order. `StrOutputParser` plucks `.content` off the `AIMessage` the model returned.

## 4. Step Two: Add Retrieval

Upgrade the chain to answer questions from the corpus. We ingest the documents, build a Chroma vector store, and compose a RAG chain.

### 4.1 Ingest

```python
# ingest.py
from pathlib import Path
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

load_dotenv()

def load_corpus(path: Path) -> list[Document]:
    docs = []
    for file in path.glob("*.txt"):
        docs.append(Document(
            page_content=file.read_text(),
            metadata={"source": file.name},
        ))
    return docs

splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=100)
chunks = splitter.split_documents(load_corpus(Path("docs")))

Chroma.from_documents(
    documents=chunks,
    embedding=OpenAIEmbeddings(model="text-embedding-3-small"),
    persist_directory="./chroma",
)
print(f"Ingested {len(chunks)} chunks.")
```

Run it once. The persisted index goes into `./chroma/`.

### 4.2 RAG Chain

```python
# step2_rag.py
from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import OpenAIEmbeddings

load_dotenv()

store = Chroma(
    persist_directory="./chroma",
    embedding_function=OpenAIEmbeddings(model="text-embedding-3-small"),
)
retriever = store.as_retriever(search_kwargs={"k": 4})

def format_docs(docs):
    return "\n\n---\n\n".join(f"[{d.metadata['source']}]\n{d.page_content}" for d in docs)

rag_prompt = ChatPromptTemplate.from_template("""
You are a technical assistant. Answer the question using only the context below.
If the context is insufficient, say so and suggest what's missing.
Cite sources inline as [filename].

<context>
{context}
</context>

Question: {question}
""")

model = ChatAnthropic(model="claude-sonnet-4-5", temperature=0)

rag = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | rag_prompt
    | model
    | StrOutputParser()
)

print(rag.invoke("What is PagedAttention and why does it help?"))
```

**What we've built.** An LCEL chain where the `{context}` slot is filled by running the retriever on the question and formatting the returned documents; the `{question}` slot is the original string. Both fill the prompt; the model answers; the parser extracts the string. The LangSmith trace will show the retrieval step with its hit list, including document metadata.

## 5. Step Three: Add Tools

RAG works when the corpus is enough. We want a fallback: if the retrieved context is insufficient, the agent should be able to search the web. That means moving from a pure chain to a loop — from LCEL to LangGraph.

First, define the tools.

```python
# tools.py
from langchain_core.tools import tool
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_tavily import TavilySearch

_store = Chroma(
    persist_directory="./chroma",
    embedding_function=OpenAIEmbeddings(model="text-embedding-3-small"),
)

@tool
def search_corpus(query: str) -> str:
    """Search the local knowledge corpus. Returns up to 4 chunks with source labels."""
    docs = _store.similarity_search(query, k=4)
    if not docs:
        return "No relevant documents found."
    return "\n\n---\n\n".join(
        f"[{d.metadata.get('source', '?')}]\n{d.page_content}" for d in docs
    )

_tavily = TavilySearch(max_results=4)

@tool
def search_web(query: str) -> str:
    """Search the web. Use only when the corpus is insufficient."""
    hits = _tavily.invoke(query)
    return "\n\n".join(f"[{h['url']}]\n{h['content']}" for h in hits)

@tool
def delete_document(source: str) -> str:
    """Permanently delete a document from the corpus. Irreversible — require human approval."""
    # In a real system this would actually delete. Here we just simulate.
    return f"Deleted {source}."
```

`delete_document` is deliberately destructive so we can demonstrate human-in-the-loop later.

## 6. Step Four: The LangGraph Agent

Now the loop. Build a minimal ReAct-style graph: one agent node, one tool node, one conditional edge.

```python
# agent.py
from typing import Annotated, TypedDict
from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import SystemMessage, ToolMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from tools import search_corpus, search_web, delete_document

load_dotenv()

TOOLS = [search_corpus, search_web, delete_document]
tools_by_name = {t.name: t for t in TOOLS}

SYSTEM = SystemMessage(content="""
You are a research assistant. Prefer search_corpus first; only use search_web
if the corpus lacks relevant information. Cite sources with [name] or [url].
delete_document is destructive — call it only when the user has explicitly
asked to delete a named document.
""".strip())

model = ChatAnthropic(model="claude-sonnet-4-5", temperature=0).bind_tools(TOOLS)

class AgentState(TypedDict):
    messages: Annotated[list, add_messages]

def call_model(state: AgentState) -> dict:
    reply = model.invoke([SYSTEM, *state["messages"]])
    return {"messages": [reply]}

def run_tools(state: AgentState) -> dict:
    out = []
    for call in state["messages"][-1].tool_calls:
        try:
            result = tools_by_name[call["name"]].invoke(call["args"])
        except Exception as e:
            result = f"ERROR: {e}"
        out.append(ToolMessage(content=str(result), tool_call_id=call["id"]))
    return {"messages": out}

def route(state: AgentState) -> str:
    return "tools" if state["messages"][-1].tool_calls else END

graph = StateGraph(AgentState)
graph.add_node("agent", call_model)
graph.add_node("tools", run_tools)
graph.add_edge(START, "agent")
graph.add_conditional_edges("agent", route, ["tools", END])
graph.add_edge("tools", "agent")

app = graph.compile()
```

Invoke it:

```python
# step4_run.py
from langchain_core.messages import HumanMessage
from agent import app

for chunk in app.stream(
    {"messages": [HumanMessage(content="Explain PagedAttention.")]},
    stream_mode="updates",
):
    print(chunk)
```

**What we've built.** A graph with two nodes (`agent`, `tools`) and three edges (`START -> agent`, `agent -> tools` or `END`, `tools -> agent`). The agent decides whether to call tools; if it does, `run_tools` executes them and feeds the `ToolMessage`s back; the cycle continues until the model produces an answer without tool calls.

In LangSmith you'll see a trace with nested spans — each iteration of the loop is its own pair of spans, with the full tool call inputs and outputs visible.

## 7. Step Five: Persistence and Threads

Add a checkpointer. Now every run is durable — you can interrupt the process and resume from the last checkpoint, and you can have multi-turn conversations that persist across process restarts.

```python
# persistent_agent.py
from langgraph.checkpoint.sqlite import SqliteSaver
from agent import graph

checkpointer = SqliteSaver.from_conn_string("agent.sqlite")
app = graph.compile(checkpointer=checkpointer)
```

Now multi-turn:

```python
# step5_run.py
from langchain_core.messages import HumanMessage
from persistent_agent import app

config = {"configurable": {"thread_id": "user-42"}}

app.invoke({"messages": [HumanMessage("What is FlashAttention?")]}, config)
app.invoke({"messages": [HumanMessage("How does that compare to PagedAttention?")]}, config)
```

The second call sees the first call's full history. Kill the process, restart, invoke again — history is intact.

Inspect the state:

```python
state = app.get_state(config)
print(len(state.values["messages"]), "messages so far")
for snap in app.get_state_history(config):
    print(snap.config["configurable"]["checkpoint_id"], snap.next)
```

## 8. Step Six: Human-in-the-Loop

`delete_document` is destructive. We want the agent to pause before executing it and wait for a human's approval. Two pieces:

1. Compile the graph with `interrupt_before=["tools"]` — but that pauses before *every* tool call, which is too coarse.
2. Better: inspect the pending tool calls inside the agent's response and only pause when a destructive one is present.

The cleanest pattern uses dynamic `interrupt`:

```python
# hitl_agent.py
from typing import Annotated, TypedDict
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import SystemMessage, ToolMessage
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.types import interrupt
from tools import search_corpus, search_web, delete_document

TOOLS = [search_corpus, search_web, delete_document]
tools_by_name = {t.name: t for t in TOOLS}
DESTRUCTIVE = {"delete_document"}

model = ChatAnthropic(model="claude-sonnet-4-5", temperature=0).bind_tools(TOOLS)
SYSTEM = SystemMessage(content="You are a research assistant. ...")

class AgentState(TypedDict):
    messages: Annotated[list, add_messages]

def call_model(state):
    return {"messages": [model.invoke([SYSTEM, *state["messages"]])]}

def approve_destructive(state):
    pending = state["messages"][-1].tool_calls
    destructive = [c for c in pending if c["name"] in DESTRUCTIVE]
    if not destructive:
        return {}  # no-op; fall through
    decision = interrupt({
        "prompt": "Approve these destructive tool calls?",
        "calls": destructive,
    })
    if decision != "approve":
        # Synthesize tool messages that tell the model the user refused.
        out = [ToolMessage(content="User refused.", tool_call_id=c["id"]) for c in destructive]
        return {"messages": out}
    return {}

def run_tools(state):
    out = []
    last = state["messages"][-1]
    already_answered = {m.tool_call_id for m in state["messages"]
                        if isinstance(m, ToolMessage)}
    for call in last.tool_calls:
        if call["id"] in already_answered:
            continue
        try:
            result = tools_by_name[call["name"]].invoke(call["args"])
        except Exception as e:
            result = f"ERROR: {e}"
        out.append(ToolMessage(content=str(result), tool_call_id=call["id"]))
    return {"messages": out}

def route(state):
    last = state["messages"][-1]
    return "approve" if getattr(last, "tool_calls", None) else END

graph = StateGraph(AgentState)
graph.add_node("agent", call_model)
graph.add_node("approve", approve_destructive)
graph.add_node("tools", run_tools)
graph.add_edge(START, "agent")
graph.add_conditional_edges("agent", route, ["approve", END])
graph.add_edge("approve", "tools")
graph.add_edge("tools", "agent")

checkpointer = SqliteSaver.from_conn_string("agent.sqlite")
app = graph.compile(checkpointer=checkpointer)
```

Drive it:

```python
# step6_run.py
from langchain_core.messages import HumanMessage
from langgraph.types import Command
from hitl_agent import app

config = {"configurable": {"thread_id": "hitl-demo"}}

# First turn — the model may decide to call delete_document.
result = app.invoke(
    {"messages": [HumanMessage("Please delete the paged_attention document.")]},
    config,
)
state = app.get_state(config)
if state.next == ("approve",):
    print("Awaiting approval for:", state.values["messages"][-1].tool_calls)
    # Approve and resume:
    app.invoke(Command(resume="approve"), config)
# Now the destructive tool ran; the agent continues.
print(app.get_state(config).values["messages"][-1].content)
```

**What we've built.** A graph with an `approve` node that pauses via `interrupt()` when the model has requested a destructive tool call, and resumes once a decision is passed in via `Command(resume=...)`. State is persisted across the pause — even if the process dies while waiting for approval, the next invocation picks up exactly where it left off.

## 9. Step Seven: Streaming for UIs

Most of the time you want the user to see tokens appear as they are produced and tool call activity as it happens. `astream_events` is the right API.

```python
# step7_stream.py
import asyncio
from langchain_core.messages import HumanMessage
from hitl_agent import app

async def main():
    config = {"configurable": {"thread_id": "stream-demo"}}
    async for event in app.astream_events(
        {"messages": [HumanMessage("What is speculative decoding? Use the corpus.")]},
        config,
        version="v2",
    ):
        kind = event["event"]
        if kind == "on_chat_model_stream":
            chunk = event["data"]["chunk"]
            if chunk.content:
                print(chunk.content, end="", flush=True)
        elif kind == "on_tool_start":
            print(f"\n[tool: {event['name']}({event['data']['input']})]", flush=True)
        elif kind == "on_tool_end":
            print(f"[tool {event['name']} done]", flush=True)

asyncio.run(main())
```

**What we've built.** Token-level streaming of the final answer, plus structured events for tool calls. In a web UI you would forward these as SSE events; in a CLI, you print them as above.

## 10. Step Eight: Graceful Failures

Wrap the model with retries and a fallback to a cheaper model, and wrap risky tool calls in try/except that return structured error messages (already done in §6's `run_tools`).

```python
# wiring.py
from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI

primary = ChatAnthropic(model="claude-sonnet-4-5", temperature=0)
backup  = ChatOpenAI(model="gpt-4o", temperature=0)

model = primary.with_retry(stop_after_attempt=3).with_fallbacks([backup])
```

Now a transient Anthropic 529 or a hard outage triggers the backup automatically. No application code changes.

## 11. Step Nine: Tracing and Tagging

Every `invoke` call site should attach context for post-hoc analysis:

```python
config = {
    "configurable": {"thread_id": thread_id},
    "tags": ["prod", "rag-agent"],
    "metadata": {
        "user_id": user.id,
        "org_id": org.id,
        "feature_flag": "rag-v2",
    },
}
```

In LangSmith you can now filter by tag, group by metadata field, and correlate traces with business context. Make this config object the single source of truth in your HTTP layer — one place to tweak when tracing policy changes.

## 12. Step Ten: Layout for Production

A production structure that scales:

```
my_agent/
  __init__.py
  graph.py           # StateGraph definition + compile(checkpointer)
  tools/
    __init__.py
    corpus.py
    web.py
    admin.py         # destructive tools, flagged for HITL
  prompts/
    system.md
    rag.md
  config.py          # env loading, model selection, retry policy
  ingest.py          # corpus ingestion job
tests/
  test_graph.py      # unit tests with FakeListChatModel
  test_tools.py
langgraph.json       # LangGraph CLI manifest
pyproject.toml
```

`langgraph.json`:

```json
{
  "dependencies": ["."],
  "graphs": {"agent": "./my_agent/graph.py:app"},
  "env": ".env"
}
```

`langgraph dev` now boots a local LangGraph Server with hot reload and Studio, hitting the same graph at `http://localhost:2024`.

## 13. Step Eleven: Testing

Mock the model with a recorded sequence of responses:

```python
# test_graph.py
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.messages import AIMessage, HumanMessage
from my_agent.graph import build_graph   # factory that accepts a model

def test_single_turn_no_tools():
    model = FakeListChatModel(responses=["It's complicated."])
    app = build_graph(model=model).compile()
    out = app.invoke({"messages": [HumanMessage("hi")]})
    assert out["messages"][-1].content == "It's complicated."
```

For tool-calling flows, use a recorded-response model or a deterministic stub that returns crafted `AIMessage.tool_calls`.

## 14. What We Built

At this point you have:

- An LCEL RAG chain (Step 2) — for simple Q&A.
- A LangGraph ReAct agent with tools (Step 4) — for multi-step reasoning.
- Persistence and threads (Step 5) — for multi-turn and durability.
- Human-in-the-loop on destructive tools (Step 6) — for safety.
- Streaming for UIs (Step 7) — for UX.
- Retries and fallbacks (Step 8) — for reliability.
- Tracing with tags and metadata (Step 9) — for observability.
- A production-shaped repository layout (Step 10–11).

That's the core toolkit. From here, likely next steps are:

- Promote the graph to Deep Agents if the tasks grow long-horizon (research reports, coding).
- Deploy via LangGraph Platform or self-hosted LangGraph Server.
- Add evaluation datasets in LangSmith for regression testing.
- Add `langgraph.store` cross-thread memory for user preferences.

## 15. Conclusion

The tutorial's structure is the one-true-path for building a LangChain + LangGraph application in Python:

1. LCEL first — chain the pieces.
2. Move to LangGraph when the flow cycles or branches.
3. Checkpoint everything.
4. Add HITL where destructive actions happen.
5. Stream by default.
6. Trace everything with tags.
7. Pin your dependencies; own your prompts.

Each step is a few dozen lines of code. None of it is magic. The framework's value is that the building blocks are uniform and durable — not that any single block is uniquely clever.

## 16. Further Reading

- **LangChain Python docs** — `python.langchain.com/docs/`
- **LangGraph Python docs** — `langchain-ai.github.io/langgraph/`
- **LangSmith docs** — `docs.smith.langchain.com/`
- **Companion articles**: LangChain overview, LangGraph overview, LangGraph Deep Agents, SDKs, TypeScript tutorial.
