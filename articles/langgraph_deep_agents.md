# LangGraph Deep Agents

*April 2026*

## 1. Introduction

A conventional ReAct agent — call model, execute tool, feed result back, repeat — works well for tasks that can be completed in a handful of tool calls. It falls apart on long-horizon work: research reports, coding projects, multi-file refactors, multi-step analyses. The agent forgets earlier findings once they scroll out of the context window, the plan that implicitly lives in the prompt drifts, and the whole run collapses into a tool-call grind that produces nothing coherent.

LangGraph Deep Agents is a named architecture — and a small library, `deepagents`, that implements it — designed to make long-horizon agents actually finish. The pattern is drawn directly from systems like Claude Code, Anthropic's and OpenAI's research agents, and Manus: a planner that thinks about the task, sub-agents spawned for sub-tasks, and a virtual filesystem used as a scratchpad to carry state across turns. The implementation is a LangGraph graph with a standard set of built-in tools, pluggable sub-agents, and persistence baked in.

This article is a technical walkthrough of the Deep Agents architecture, the `deepagents` package, how to customise it, and how to use it in production. It assumes you are familiar with LangGraph (see the companion article).

## 2. The Long-Horizon Problem

### 2.1 Why ReAct Alone Breaks

A ReAct agent holds its entire working memory in the message history. For a five-tool-call task, that is fine. For a fifty-tool-call task, three things go wrong:

- **Context bloat.** Old tool outputs consume tokens that should be available for new reasoning. Truncating them loses information.
- **No persistent plan.** The plan, if it exists, is implicit in the system prompt or in the first assistant message. The model has no affordance for updating it.
- **No separation of concerns.** A single model instance is simultaneously planning, executing, summarising, and writing. It does all of them at a mediocre level.

### 2.2 What Long-Horizon Agents Do Differently

Empirical study of the most-effective long-horizon agents — Claude Code, Deep Research products, Manus, agentic coding benchmarks — reveals four recurring patterns:

1. **An explicit plan (a todo list) that the agent maintains, revises, and consults.**
2. **A virtual filesystem used as durable scratchpad — the agent writes findings to files, reads them back later, and keeps the context window for active reasoning.**
3. **Sub-agents spawned for clearly scoped sub-tasks, each with its own isolated context window.**
4. **A detailed system prompt that teaches the agent *when and how* to use the above.**

Deep Agents packages these four patterns into a LangGraph graph plus a small toolset. The cost is a few hundred extra tokens of system prompt and five or six always-available tools; the benefit is that the agent stays coherent over runs that would otherwise diverge.

## 3. The `deepagents` Package

### 3.1 What You Get

`pip install deepagents` gives you a single top-level constructor:

```python
from deepagents import create_deep_agent

agent = create_deep_agent(
    tools=[search_web, fetch_url],
    instructions="You are a research analyst...",
    subagents=[...],       # optional
    model=...,             # optional, defaults to Claude Sonnet
    builtin_tools=["write_todos", "ls", "read_file", "write_file", "edit_file"],
).compile(checkpointer=checkpointer)
```

`create_deep_agent` returns a LangGraph graph pre-wired with:

- A **planner/executor node** running a capable model (Claude Sonnet 4.5 by default) with a Deep-Agent system prompt layered on top of your instructions.
- A **ToolNode** executing your tools plus the built-in filesystem and todo tools.
- A **task-delegation** mechanism for invoking registered sub-agents as tools.
- Default LangGraph state extended with `files` (a virtual filesystem) and `todos` (a task list).

### 3.2 The Built-in State

The default Deep Agent state extends the standard agent state with two fields:

```python
class DeepAgentState(AgentState):
    files: Annotated[dict[str, str], file_reducer]   # path -> content
    todos: Annotated[list[Todo], todo_reducer]        # plan items
```

`files` is the virtual filesystem — an in-memory dict of filename to content — persisted by the checkpointer like any other state. `todos` is the plan. Both are visible to all nodes and to the model (through the built-in tools).

## 4. The Four Pillars

### 4.1 Planning with `write_todos`

The agent is given a `write_todos` tool that replaces its plan:

```
write_todos(todos: list[{content: str, status: "pending"|"in_progress"|"completed"}])
```

The system prompt instructs the model to call `write_todos` at the start of the task, mark items `in_progress` as it works on them, and mark them `completed` as it finishes. The plan is part of state; it survives checkpoint/resume; and because it is displayed in the prompt on every turn, the model stays anchored to the overall goal even when the message history grows.

This is a direct port of the Claude Code "todo list" behaviour and is one of the highest-leverage features in the architecture. Agents that use an explicit plan outperform agents that don't, in the same benchmarks, with the same models and tools.

### 4.2 Virtual Filesystem: `ls`, `read_file`, `write_file`, `edit_file`

The agent has four filesystem tools backed by the `files` state field:

- `ls()` — list all files.
- `read_file(path, offset?, limit?)` — read a file (optionally a slice).
- `write_file(path, content)` — overwrite a file.
- `edit_file(path, old_string, new_string)` — exact-string edit in place.

The files are not on the host filesystem; they live in state. This has two consequences:

- **The "filesystem" is sandboxed.** The agent cannot touch host files unless you explicitly expose a real-filesystem tool.
- **The "filesystem" is persisted.** Every file is in state; state is checkpointed; a resumed thread has all its files.

The system prompt teaches the model to use the filesystem the way a human uses a scratchpad: write research notes as you go, read them back when composing the final answer, keep the message history for active reasoning.

### 4.3 Sub-Agents

Sub-agents are small, scoped agents that the main agent can invoke as tools. Each sub-agent has:

- A **name** (used in the tool name).
- A **description** (used as the tool description, which is what the model sees when deciding whether to dispatch).
- A **prompt** (the sub-agent's instructions — typically much more focused than the main agent's).
- An optional **tools** list (subset of the parent's tools; defaults to all).
- An optional **model** override.

```python
subagents = [
    {
        "name": "research",
        "description": "Research a specific topic and return a summary with citations.",
        "prompt": "You are a research agent. Search the web, read pages, and return a concise summary with URLs. Do not speculate.",
        "tools": ["search_web", "fetch_url"],
    },
    {
        "name": "critic",
        "description": "Review a draft for accuracy and suggest improvements.",
        "prompt": "You are a critic. Read the draft at the given path, identify factual errors, and return a list of concrete suggestions.",
        "tools": ["read_file"],
    },
]
```

The main agent sees these as tools named `task.research(...)` and `task.critic(...)`. Dispatching one spawns a fresh sub-graph with its own context window, its own message history, access to the shared filesystem and todos, but running the sub-agent's prompt instead of the parent's.

This is crucial: sub-agents **do not** inherit the parent's message history. They start clean, do their scoped work, return a summary, and are discarded. The parent's context window is preserved.

### 4.4 The System Prompt

The Deep Agent system prompt, layered on top of your custom instructions, is the glue that makes the four pillars cohere. It tells the model:

- Start every non-trivial task by drafting a todo list with `write_todos`.
- Write findings, research notes, and drafts to the filesystem, not into the chat.
- Read files back when composing the final answer.
- Dispatch sub-agents for anything that needs its own clean context.
- Update the todo list as status changes.

The prompt is substantial — a few hundred tokens — but it is the difference between a tool-using model and an agent that actually finishes a long task.

## 5. Minimal Example

```python
from deepagents import create_deep_agent
from langchain_core.tools import tool
from tavily import TavilyClient

tavily = TavilyClient()

@tool
def search_web(query: str) -> list[dict]:
    """Search the web and return a list of {title, url, content} results."""
    return tavily.search(query, max_results=5)["results"]

@tool
def fetch_url(url: str) -> str:
    """Fetch a URL and return its cleaned text content."""
    import requests, trafilatura
    html = requests.get(url, timeout=20).text
    return trafilatura.extract(html) or ""

instructions = """
You are a research assistant. Given a question, produce a well-sourced
research report. Write intermediate notes to files under /notes/ and
write the final report to /report.md. Cite your sources with URLs.
"""

agent = create_deep_agent(
    tools=[search_web, fetch_url],
    instructions=instructions,
).compile()

result = agent.invoke({
    "messages": [{"role": "user", "content": "Write a 1000-word report on photonic AI accelerators."}]
})

print(result["files"]["/report.md"])
```

Running this produces a sequence of:
1. `write_todos` to establish the plan.
2. Multiple `search_web` calls.
3. `write_file` calls to save intermediate research notes.
4. More searches and notes as subtopics are explored.
5. A final `write_file` to `/report.md` synthesising the notes.
6. The final assistant message summarising what was done.

The state at the end contains the full plan, every note file, and the final report, all addressable and persisted.

## 6. Anatomy of a Deep Agent Graph

Compiled, a Deep Agent is a LangGraph graph with roughly this shape:

```
      START
        │
        ▼
     ┌──────┐
     │agent │◀──────────────────┐
     └──┬───┘                   │
        │                       │
 tool_calls?─┐                  │
        │   │no                 │
        │   └──▶ END            │
        │yes                    │
        ▼                       │
     ┌──────┐                   │
     │tools │ (built-in + user) │
     └──┬───┘                   │
        │                       │
        └───────────────────────┘
```

Plus, if sub-agents are registered, each one is a subgraph dispatched from the `tools` node via the `task.<name>` tool. The subgraph has the same structure — it's a Deep Agent itself — with the sub-agent's prompt and tool subset.

Because it is a normal LangGraph graph, you get everything in the LangGraph overview: checkpointing, threads, HITL breakpoints, time travel, streaming, LangSmith tracing.

## 7. Customisation

### 7.1 Model Choice

`create_deep_agent(model=...)` accepts any LangChain chat model. The default is Claude Sonnet 4.5 — chosen because Deep Agents depend on strong tool-use, instruction-following, and reasoning, all of which Anthropic models handle well. GPT-4.1 and Gemini 2.5 Pro also work; smaller models generally struggle with the long system prompt and multi-tool coordination.

Per-sub-agent model overrides let you use cheap models for simple sub-tasks:

```python
{"name": "summariser", "description": "...", "prompt": "...", "model": "gpt-4o-mini"}
```

### 7.2 Custom State

`create_deep_agent(state_schema=MyState)` accepts a state schema that extends `DeepAgentState`. Add domain-specific fields (current user, organisation, budget remaining) and they are persisted, streamed, and visible to every node.

### 7.3 Middleware

Deep Agents supports LangGraph middleware — functions that wrap node execution to add logging, metrics, redaction, policy checks, or cost tracking. Middleware is the right home for cross-cutting concerns that should apply uniformly to every node.

### 7.4 Custom Built-ins

You can disable any of the built-in tools (pass `builtin_tools=[]` to remove all of them) or subset them. An agent with just `write_todos` and your own tools is already a large upgrade over vanilla ReAct.

You can also replace the filesystem backend. The default is in-state, but the package supports swapping it for a local filesystem, an object store, or a hosted sandbox — useful when the agent must produce files the host system can use directly (e.g. for coding agents).

## 8. The Coding Agent Pattern

A Deep Agent specialised for coding replaces the virtual filesystem with a real one (sandboxed) and adds:

- `run_shell(cmd)` — execute a shell command in the sandbox.
- `run_tests(pattern?)` — run the test suite.
- `grep`, `glob`, `find` equivalents.

Plus sub-agents such as:

- `plan` — high-level planner that never writes code, only plans.
- `implement` — writes code, calls tests, iterates.
- `review` — reads the diff and critiques.

This is essentially the architecture of Claude Code, mapped into the Deep Agents shape. The library does not ship a "coding agent" preset — the expectation is that you configure one yourself from the primitives — but it is the most common specialisation seen in the wild.

## 9. The Research Agent Pattern

The original motivating example. Specialisation:

- Tools: web search, fetch URL, PDF reader.
- Sub-agents: `research` (scoped topical research), `fact_check` (verify claims), `write` (draft sections).
- Output contract: a final Markdown report in `/report.md` with inline citations.

This pattern is directly comparable to Anthropic's Research product, OpenAI's Deep Research, and Perplexity's offering. Open-source implementations using Deep Agents produce reports of comparable quality when given a strong model and good search tools.

## 10. Interaction with LangGraph Platform

Deep Agents compile to a normal LangGraph graph, so they deploy on the LangGraph Platform or a self-hosted LangGraph Server without modification. On the platform you additionally get:

- A hosted thread store — one conversation per thread, with the full history including files and todos.
- HITL endpoints for approvals.
- SSE streaming of node updates, messages, and custom events.
- LangSmith tracing integrated into the UI.

For production Deep Agents these platform features are substantial; without them you are reimplementing the thread API, the HITL wiring, and the streaming protocol yourself.

## 11. Observability and Debugging

The most common failure modes of Deep Agents are:

- **Plan drift.** The model writes a plan at the start, then ignores it. Mitigation: strengthen the system prompt's "always update the todo list" instruction; surface the plan in the UI so operators notice when it goes stale.
- **Filesystem thrash.** The model rewrites the same file repeatedly without converging. Mitigation: add a write-counter to state and route to a diagnostic node when it exceeds a threshold.
- **Sub-agent context loss.** A sub-agent is asked to do something that depends on context the parent has but didn't pass. Mitigation: make the parent pass all relevant context explicitly in the sub-agent invocation argument.
- **Runaway costs.** A long task, a cheap planning pass, an expensive model underneath; easy to rack up costs unnoticed. Mitigation: track `total_cost_usd` in state, add a budget check in a middleware.

All four are visible in LangSmith traces. Enable tracing from day one.

## 12. Performance Considerations

Deep Agents are not the fastest tool-use pattern. The long system prompt, the extra tools, and the sub-agent spawning all add overhead relative to a vanilla ReAct agent. Typical numbers:

- **System prompt overhead**: ~500-800 tokens per model call. Use prompt caching; on Anthropic with caching this becomes effectively free after the first call.
- **Sub-agent spawn cost**: one additional model call to start the sub-agent plus its own context.
- **Filesystem tool roundtrips**: each read/write is a tool call (2 round-trips to the model). Not free, but a good investment relative to stuffing content into the chat.

For tasks where a plain ReAct loop would finish in five tool calls, Deep Agents is strictly worse. For tasks that a plain ReAct loop would fail on entirely, Deep Agents is strictly better. Choose accordingly.

## 13. Alternatives and Related Patterns

- **Plain LangGraph + ToolNode.** Lower overhead; you provide the planning prompt yourself if you need one. Use for medium-horizon tasks.
- **CrewAI.** Similar multi-agent shape, role-based metaphor. Less opinionated about planning; no virtual filesystem.
- **AutoGen / AG2.** Conversation-based multi-agent; less focus on long-horizon plans and filesystems.
- **OpenAI Agents SDK with handoffs.** Similar sub-agent dispatch but without a planning/filesystem pattern built in.
- **Custom LangGraph from scratch.** The right choice once your agent outgrows the Deep Agents defaults. Deep Agents is a preset; LangGraph is the substrate.

## 14. Limitations

- **Single main loop.** Deep Agents has one primary planner/executor. Truly parallel multi-agent topologies (peer-to-peer, networked) are better built from LangGraph primitives directly.
- **Opinionated system prompt.** The default prompt assumes research-and-writing or coding shapes. For task shapes that differ substantially (e.g., live operations triage), the prompt needs rewriting.
- **Model-dependence.** Weak models struggle. Sonnet-class or better is required for reliable behaviour.
- **Cost.** Plans + sub-agents + filesystem I/O multiply the number of model calls. Prompt caching is essentially mandatory.

## 15. Conclusion

Deep Agents is a named, packaged version of the architecture that powers the strongest long-horizon agents in production as of April 2026. The formula — an explicit plan, a virtual filesystem, scoped sub-agents, and a system prompt that knits them together — is reproducible, well-understood, and available as a one-call constructor.

For long-horizon tasks (research reports, multi-file coding, deep analysis), start with `create_deep_agent` and customise from there. For short tool-use loops, a prebuilt ReAct agent or a hand-built LangGraph graph is simpler and cheaper.

The ceiling of Deep Agents is set by the underlying model plus the quality of the custom tools and sub-agents you register. The architecture removes the long-horizon failure modes; it does not replace the model's underlying capability.

## 16. Further Reading

- **`deepagents` package** — `pypi.org/project/deepagents/`
- **Blog post introducing Deep Agents** — `blog.langchain.com/deep-agents/`
- **Companion articles**: LangGraph overview, LangChain/LangGraph SDKs, Python tutorial, TypeScript tutorial.
- **Related collection articles**: Agent Orchestration Frameworks, Multi-Agent LLM Systems, Memory Systems for LLM Agents.
