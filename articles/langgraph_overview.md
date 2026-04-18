# LangGraph: Stateful Agent Graphs

*April 2026*

## 1. Introduction

Agent loops are not chains. A chain is a directed acyclic pipeline — data flows from input to output through a fixed sequence of steps. An agent, by contrast, is cyclic: the model decides what to do next, tools execute, their results flow back to the model, and the cycle continues until some termination condition is met. Branching, looping, retries, sub-agent dispatch, and human approval steps all demand a state machine, not a pipeline.

LangGraph is LangChain's answer to this mismatch. It is a small, opinionated library for building cyclic, stateful, durable LLM workflows as explicit graphs. A graph has a typed state, a set of nodes that each take state and return a state update, and edges — including conditional and parallel edges — that describe the control flow between nodes. A compiled graph is itself a Runnable that supports `invoke`, `stream`, `batch`, and async variants, so it composes cleanly with LCEL-built prompts, retrievers, and models.

In April 2026 LangGraph is the default substrate for production agent systems built with LangChain, and it is also used standalone without the rest of LangChain. This article is a detailed tour of the library: the graph model, state reducers, checkpointing and persistence, streaming, human-in-the-loop, time travel, subgraphs, and the prebuilt agent helpers. The Deep Agents pattern is covered in a companion article; the SDK and platform details are in another.

## 2. Why a Graph

### 2.1 The ReAct Loop, Explicitly

A vanilla ReAct agent is a two-node graph:

```
START -> agent
agent -> tools  (if agent produced tool calls)
agent -> END    (if agent produced a final answer)
tools -> agent  (feed results back)
```

That is one conditional edge, one unconditional edge, two nodes, and a shared state that accumulates `messages`. Written as an explicit graph it looks like a state machine. Written as a procedural loop in pure Python it looks like a ten-line while loop. Both work — but when you need to add a human-approval checkpoint between `tools` and `agent`, or branch into a planner on first turn, or add a reflection node that runs every third iteration, the graph formulation degrades gracefully and the while-loop formulation collapses into spaghetti.

### 2.2 What You Get For Free

Once your agent is a LangGraph graph, a number of otherwise-expensive capabilities become one-liners:

- **Durable execution.** A checkpointer saves state after every node. A crashed or interrupted run resumes from the last checkpoint.
- **Human-in-the-loop.** Breakpoints pause execution at any node; approval arrives via an API call; execution resumes exactly where it paused.
- **Time travel.** Inspect state at any prior step, branch a new run from an old checkpoint, replay with different tools.
- **Streaming.** Token streams, node-by-node state updates, or structured events — all emitted as the graph runs.
- **Parallel fan-out.** The `Send` API dispatches multiple state copies to the same node for concurrent execution.
- **Subgraphs.** Compose a small graph as a single node in a larger graph.

These are the capabilities that make LangGraph worth the upfront cost of the graph abstraction.

## 3. The Mental Model

```
┌─────────────────────────────────────────────────────────────┐
│                   STATE (TypedDict / Pydantic)              │
│   messages: Annotated[list, add_messages]                   │
│   plan: str                                                 │
│   current_step: int                                         │
│   scratchpad: dict                                          │
└─────────────────────────────────────────────────────────────┘
         ▲                      ▲                      ▲
         │                      │                      │
    ┌────┴────┐            ┌────┴────┐            ┌────┴────┐
    │ planner │ ──────────▶│  agent  │ ──────────▶│  tools  │
    └─────────┘            └─────────┘            └─────────┘
                                 ▲                      │
                                 └──────────────────────┘
                                  (results feed back)
```

- The **state** is the single source of truth. Nodes never see each other directly; they see and update state.
- A **node** is a function: `node(state) -> partial_state_update`. The graph framework merges the update into the state according to the state's reducer functions.
- An **edge** moves execution from one node to the next. Unconditional edges are fixed; conditional edges are a function of state that returns the name of the next node (or END).

Everything else — checkpointing, streaming, HITL — is layered on top of this core.

## 4. State and Reducers

### 4.1 Declaring State

State is a TypedDict or Pydantic model. Fields can be annotated with a reducer function that says how two values should be combined when nodes return updates for the same field.

```python
from typing import Annotated, TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages

class AgentState(TypedDict):
    messages: Annotated[list, add_messages]   # append-and-dedup
    scratchpad: dict                          # last-write-wins (default)
    budget_remaining_usd: float
```

`add_messages` is the canonical message-list reducer: it appends new messages, deduplicates by ID, and handles removal messages. Without a reducer, the default behaviour is "last write wins" — later updates overwrite earlier ones.

### 4.2 Custom Reducers

A reducer is any `(old, new) -> merged` function.

```python
def merge_counters(old: dict, new: dict) -> dict:
    return {**old, **{k: old.get(k, 0) + v for k, v in new.items()}}

class State(TypedDict):
    counters: Annotated[dict, merge_counters]
```

Parallel nodes that update the same field concurrently go through the reducer, which is what makes fan-out safe.

### 4.3 Partial Updates

A node returns *only* the fields it wants to update:

```python
def planner(state: AgentState) -> dict:
    plan = make_plan(state["messages"])
    return {"scratchpad": {"plan": plan}}
```

The framework merges that partial update into the full state. Returning `{}` is a valid no-op.

## 5. Building a Graph

### 5.1 A Minimal ReAct Agent

```python
from typing import Annotated, TypedDict
from langchain_core.messages import ToolMessage
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages

@tool
def get_weather(city: str) -> str:
    """Current weather for a city."""
    return f"{city}: 18°C, clear"

tools = [get_weather]
model = ChatOpenAI(model="gpt-4o").bind_tools(tools)
tools_by_name = {t.name: t for t in tools}

class AgentState(TypedDict):
    messages: Annotated[list, add_messages]

def call_model(state: AgentState) -> dict:
    return {"messages": [model.invoke(state["messages"])]}

def run_tools(state: AgentState) -> dict:
    out = []
    for call in state["messages"][-1].tool_calls:
        result = tools_by_name[call["name"]].invoke(call["args"])
        out.append(ToolMessage(content=str(result), tool_call_id=call["id"]))
    return {"messages": out}

def should_continue(state: AgentState) -> str:
    return "run_tools" if state["messages"][-1].tool_calls else END

graph = StateGraph(AgentState)
graph.add_node("agent", call_model)
graph.add_node("run_tools", run_tools)
graph.add_edge(START, "agent")
graph.add_conditional_edges("agent", should_continue, ["run_tools", END])
graph.add_edge("run_tools", "agent")

app = graph.compile()
```

`app` is a Runnable. You can `app.invoke({"messages": [HumanMessage("What's the weather in Lisbon?")]})` and it will loop until the model stops calling tools.

### 5.2 The Prebuilt ReAct Agent

LangGraph ships a prebuilt equivalent of the above:

```python
from langgraph.prebuilt import create_react_agent

app = create_react_agent(model, tools)
```

For simple tool-loops this one-liner is fine. For anything more complex — planning, reflection, sub-agents, custom state fields — build the graph explicitly.

### 5.3 Conditional Edges

`add_conditional_edges(src, router_fn, mapping)` lets the graph branch. The router function returns a string (or list of strings, for fan-out) naming the next node(s).

```python
def route(state: AgentState) -> str:
    if state["budget_remaining_usd"] < 0.01:
        return "budget_exhausted"
    if state["messages"][-1].tool_calls:
        return "run_tools"
    return END

graph.add_conditional_edges("agent", route, ["run_tools", "budget_exhausted", END])
```

### 5.4 Parallel Edges with Send

`Send` dispatches multiple independent payloads to the same target node for concurrent execution.

```python
from langgraph.types import Send

def fan_out(state: AgentState) -> list[Send]:
    return [Send("research", {"query": q}) for q in state["subquestions"]]

graph.add_conditional_edges("planner", fan_out, ["research"])
```

Each `research` invocation runs concurrently, with its own slice of state. Reducers merge their results back into the parent state.

## 6. Checkpointing and Persistence

### 6.1 Checkpointers

A checkpointer saves the full state after every node execution. LangGraph ships several:

| Checkpointer | Backing store | Use case |
|---|---|---|
| `MemorySaver` | In-process dict | Unit tests, notebooks |
| `SqliteSaver` | Local SQLite DB | Single-machine dev + small apps |
| `PostgresSaver` | Postgres | Production |
| Platform | LangGraph Platform managed store | Hosted production |

```python
from langgraph.checkpoint.postgres import PostgresSaver

checkpointer = PostgresSaver.from_conn_string("postgresql://...")
checkpointer.setup()

app = graph.compile(checkpointer=checkpointer)
```

### 6.2 Threads

A **thread** is the unit of conversation / agent run. Every invocation specifies a `thread_id` in its config; the checkpointer persists all state under that thread.

```python
config = {"configurable": {"thread_id": "user-42-session-7"}}

app.invoke({"messages": [HumanMessage("hi")]},    config)
app.invoke({"messages": [HumanMessage("again")]}, config)  # continues the thread
```

Re-invoking a thread resumes from its last checkpoint. The second call above sees the full history from the first.

### 6.3 Cross-Thread (Long-Term) Memory

Thread checkpoints are scoped to a single conversation. For long-term memory that spans conversations — user preferences, facts about the user, prior resolutions — use the `Store` API:

```python
from langgraph.store.postgres import PostgresStore

store = PostgresStore.from_conn_string("postgresql://...")
app = graph.compile(checkpointer=checkpointer, store=store)

def remember_preference(state, config, *, store):
    store.put(
        namespace=("user", config["configurable"]["user_id"]),
        key="favourite_language",
        value={"lang": "Portuguese"},
    )
```

The store is a key-value store with namespaces and optional semantic search via vector embeddings. This is the right home for anything that must survive past the current conversation.

## 7. Human-in-the-Loop

### 7.1 Static Breakpoints

Compile with `interrupt_before=["node"]` or `interrupt_after=["node"]`. When the graph reaches that node it stops, returns control to the caller, and its state is saved. The caller can inspect state, edit it, and then resume.

```python
app = graph.compile(checkpointer=checkpointer, interrupt_before=["run_tools"])

# First pass: runs up to run_tools and stops.
app.invoke({"messages": [...]}, config)

# Approve by resuming with no state update, or edit by updating state first.
app.invoke(None, config)
```

### 7.2 Dynamic Interrupts

`interrupt()` pauses from inside a node, with a payload that surfaces to the caller.

```python
from langgraph.types import interrupt, Command

def confirm_action(state):
    decision = interrupt({"action": state["proposed_action"]})
    return {"approved": decision == "yes"}
```

The caller resumes with `Command(resume="yes")`, which replays the node with `interrupt()` now returning `"yes"`.

This is the canonical pattern for approvals, clarifying questions, and tool-use consent.

## 8. Time Travel

Every checkpoint is addressable. You can list the full history, read the state at any step, and branch a new run from any point:

```python
history = list(app.get_state_history(config))
past = history[3]               # state after step 3

# Continue with edited state from that point:
new_config = app.update_state(past.config, {"plan": "new plan"})
app.invoke(None, new_config)
```

Time travel is primarily a debugging and experimentation tool — it lets you rewind an agent after an error, tweak its plan, and continue. It is also the foundation for A/B branches from a shared prefix checkpoint.

## 9. Streaming

### 9.1 Stream Modes

`app.stream(input, config, stream_mode=...)` chooses what is emitted:

| Mode | Emits |
|---|---|
| `"values"` | Full state after each node |
| `"updates"` | Node-by-node state updates (default for many use cases) |
| `"messages"` | Token-level message chunks (LLM streaming) |
| `"custom"` | Arbitrary events written from nodes via `get_stream_writer()` |
| `"debug"` | Everything — node start/end, checkpoints, writes |

You can pass multiple modes as a list; the stream yields `(mode, payload)` tuples.

```python
for mode, chunk in app.stream(input, config, stream_mode=["messages", "updates"]):
    if mode == "messages":
        tok, meta = chunk
        print(tok.content, end="", flush=True)
    elif mode == "updates":
        print("\n[update]", chunk)
```

### 9.2 astream_events

`app.astream_events(...)` yields structured events (`on_chain_start`, `on_llm_stream`, `on_tool_end`, ...) across the whole graph. It is the richest streaming interface and is the one typically used by front-ends that want to render tool calls, thinking blocks, retrieved documents, and token streams as distinct UI elements.

## 10. Subgraphs

A compiled graph is itself a Runnable. Compile one and use it as a node in a bigger graph:

```python
research_app = research_graph.compile()

parent = StateGraph(ParentState)
parent.add_node("research", research_app)
parent.add_node("writer", writer_node)
parent.add_edge("research", "writer")
```

Subgraphs can share state fields with the parent (any field in the parent state whose key matches a subgraph state key is passed through) or declare a distinct state schema (in which case you pass an input-mapping function).

Subgraphs are the natural mechanism for **multi-agent architectures**: one supervisor graph dispatches to several sub-agent graphs, each with its own state, tools, and prompt.

## 11. Prebuilt Helpers

LangGraph ships a growing set of `langgraph.prebuilt.*` helpers:

- `create_react_agent(model, tools, state_schema=..., prompt=...)` — ReAct tool-use agent.
- `ToolNode(tools)` — the tool-executor node, with automatic parallel tool calls.
- `tools_condition(state)` — the `END` vs `tools` router.

They are useful starting points and are worth reading as reference implementations.

## 12. Multi-Agent Patterns

Three patterns dominate production multi-agent systems built on LangGraph:

### 12.1 Supervisor

A supervisor node receives state, decides which sub-agent is best placed to act, and dispatches via `Command(goto="sub_agent_x", update={...})`. Sub-agents run, return to the supervisor, and the loop continues.

### 12.2 Network (Any-to-Any)

Every agent can hand off to every other agent. Each agent node is bound to tools plus a set of "handoff" tools that emit `Command` updates routing to the chosen peer.

### 12.3 Planner → Workers

A planner decomposes the task into subtasks, emits them via `Send`, workers run in parallel, results fan in via a reducer. This is the same shape as the Deep Agents architecture — see the companion article.

## 13. Commands: Combining State and Routing

`Command` lets a node both update state and choose the next node in a single return value:

```python
from langgraph.types import Command

def supervisor(state):
    choice = pick_agent(state)
    return Command(
        update={"handoff_reason": "needs_coding"},
        goto=choice,   # "coder" | "reviewer" | END
    )
```

Without `Command`, routing is purely a function of state (via conditional edges). With `Command`, the routing decision can carry context that is explicit rather than implicit in state fields.

## 14. Error Handling

Exceptions raised inside a node propagate to the caller. Because the checkpointer has already saved state up to the previous node, re-invoking the same thread will re-execute only the failed node — the prior work is not lost.

Common patterns:

- Wrap model calls with `with_retry` from LCEL.
- Wrap tool calls in try/except inside the tool function and return a structured error message; let the agent decide whether to retry.
- For whole-graph retries (e.g., the model returns malformed tool calls persistently), use a `retries` field in state and route to a repair node on too many failures.

## 15. Testing

LangGraph graphs are deterministic given the same inputs and the same model outputs, so the usual agent-testing techniques apply:

- **Replay from checkpoints.** Unit tests can seed a known state and invoke a single node directly (`graph.nodes["planner"].invoke(state)`).
- **Mock the model.** Use `FakeListChatModel` or a recorded replay model to make graph behaviour deterministic in CI.
- **Trajectory assertions.** Assert that the sequence of node names visited matches an expected trajectory for a given input.

The explicit graph structure is the reason trajectory assertions are tractable — you are checking which nodes were entered, not parsing prose.

## 16. Observability

LangGraph has native LangSmith integration. Set `LANGSMITH_TRACING=true` and every graph run, including state at each step, is captured.

For custom observability, callbacks work as they do in LCEL. Additionally, `astream_events` gives you a programmatic event stream you can forward to any logging backend.

## 17. Performance

LangGraph is a thin coordination layer; the bulk of runtime is in model calls and tool execution. Two specific factors matter:

- **Checkpointer latency.** Every node writes a checkpoint. On Postgres this is a single insert; on an async Postgres pool it is typically sub-millisecond. For very high-throughput graphs with many small nodes, consider collapsing nodes or using a faster checkpointer.
- **Parallel fan-out.** Parallel nodes run concurrently via the async event loop. Pure-Python nodes with the GIL are still serialised; use `asyncio` and async node functions for real concurrency.

Streaming has negligible overhead; it is cooperative and does not add wire traffic beyond what the model SDK already produces.

## 18. Deployment Options

LangGraph graphs can be deployed in three ways:

1. **Embedded in your own server.** `app.invoke` inside FastAPI, Django, Flask, or a worker queue. Use any checkpointer; bring your own observability.
2. **LangGraph Platform (managed).** Push a graph to the platform; it exposes it as a hosted API with checkpointing, threads, HITL endpoints, streaming, and LangSmith tracing pre-wired. The SDK article covers the developer flow.
3. **Self-hosted LangGraph Server.** The same server that backs the platform, run on your own Kubernetes. Same APIs as the managed version.

For most production systems, either the platform or the self-hosted server is worth adopting once you have more than a couple of graphs — the HITL and thread APIs are non-trivial to build yourself.

## 19. Strengths and Limitations

### 19.1 Strengths

- **Durable execution is first-class.** Checkpointing, threads, and HITL are the core, not afterthoughts.
- **Explicit control flow.** The graph is inspectable, testable, and diffable in code review.
- **Streams tokens and structured events.** Good UX is easy.
- **Composes with LCEL.** Runnables go inside nodes; compiled graphs are Runnables.

### 19.2 Limitations

- **Verbose for trivial agents.** A 20-line while loop is simpler than a two-node graph if you truly have only two nodes and no need for persistence.
- **Learning curve.** Reducers, `Send`, `Command`, subgraphs, interrupts — it takes time to internalise the whole surface.
- **Python-first.** TypeScript LangGraph (`@langchain/langgraph`) is a first-class citizen but trails the Python version by a few weeks on new features.

## 20. Conclusion

LangGraph is the right tool for any agent that has a loop, needs to pause for humans, needs to survive restarts, or needs to branch. It is overkill for single-turn extraction or plain Q&A RAG — LCEL is plenty for those. The intellectually honest rule is: as soon as your agent has a cycle or a step that might need to pause, move it into LangGraph.

Once inside LangGraph, the high-leverage features are checkpointing, threads, and human-in-the-loop. Streaming is table stakes. Subgraphs and `Send` are what let you scale from one agent to a multi-agent system without rewriting.

The Deep Agents pattern, built on top of LangGraph, is the current state of the art for long-horizon autonomous agents and is covered in a companion article.

## 21. Further Reading

- **LangGraph docs** — `langchain-ai.github.io/langgraph/`
- **Conceptual guides** — `langchain-ai.github.io/langgraph/concepts/`
- **Prebuilt agents reference** — `langchain-ai.github.io/langgraph/reference/prebuilt/`
- **Companion articles**: LangChain overview, LangGraph Deep Agents, LangChain/LangGraph SDKs, Python and TypeScript tutorials.
