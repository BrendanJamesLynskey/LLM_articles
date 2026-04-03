# Agent Orchestration Frameworks

*April 2026*

## 1. Introduction

Building an LLM agent from scratch requires implementing a substantial amount of infrastructure: the model interaction loop, tool execution, state management, error handling, conversation memory, streaming, observability, and human-in-the-loop controls. Every team that builds an agent independently reimplements these same patterns, often with subtle bugs and missing edge cases.

Agent orchestration frameworks provide this infrastructure as reusable libraries and platforms. They encode best practices for agent construction, provide abstractions that simplify common patterns, and offer observability tools that make agent behavior inspectable and debuggable. The tradeoff is familiar from software frameworks generally: you gain speed and reliability by adopting the framework's patterns, but you accept its constraints and abstractions.

The ecosystem of agent frameworks has matured significantly since the early days of LangChain in 2022. The current landscape includes general-purpose frameworks (LangGraph, CrewAI, AutoGen), provider-specific SDKs (Anthropic Agent SDK, OpenAI Agents SDK), and specialized tools (DSPy for optimization, Pydantic AI for type-safe agents). Each makes different tradeoffs between flexibility, simplicity, and capability.

This report provides a comprehensive technical examination of the major agent orchestration frameworks available in 2026. It covers their architectures, design philosophies, strengths, and limitations, with practical guidance on choosing between them. The intended audience is engineers evaluating frameworks for building agentic applications.

## 2. Why Frameworks

### 2.1 The Boilerplate Problem

A minimal agent—one that takes user input, calls tools, and iterates—requires surprisingly little code. But a production agent requires substantially more:

- **Retry logic** for failed API calls and tool executions
- **Streaming support** for real-time token delivery
- **State management** for multi-turn conversations and long-running tasks
- **Error handling** for every failure mode (model errors, tool errors, timeout, rate limits)
- **Input/output validation** to catch formatting errors before they propagate
- **Logging and tracing** for debugging and monitoring
- **Human-in-the-loop** controls for approval workflows
- **Token counting** and context window management
- **Conversation memory** for multi-session persistence
- **Rate limit handling** with automatic backoff
- **Multi-model support** for using different models for different tasks
- **Cost tracking** for monitoring and budgeting

Implementing all of this from scratch is a multi-week engineering effort. Frameworks provide most of it out of the box.

### 2.2 The Pattern Problem

Agent architectures—ReAct, plan-and-execute, reflection, multi-agent—are well-understood patterns, but implementing them correctly requires getting many details right. A ReAct loop seems simple, but handling tool call parsing failures, managing the growing context, detecting stuck loops, and implementing graceful termination adds significant complexity. Frameworks encode these patterns with their edge cases handled.

### 2.3 The Observability Problem

Agent behavior is non-deterministic and often surprising. The same input can lead to different tool call sequences on different runs. Debugging a failed agent run requires seeing the full trace: every model call, every tool call, every decision point, and the state at each point. Building this observability from scratch is a significant investment. Frameworks either provide it directly or integrate with observability platforms.

### 2.4 The Evaluation Problem

Evaluating agent performance requires running agents on test cases, capturing their behavior, and assessing the results. This requires infrastructure for batch execution, result comparison, metrics computation, and regression detection. Several frameworks provide evaluation capabilities or integrate with evaluation platforms.

## 3. LangGraph

### 3.1 Overview

LangGraph, developed by LangChain, is the most widely used framework for building stateful, multi-step agents. It models agent workflows as directed graphs (state machines) where nodes are processing steps and edges are transitions between steps. This explicit graph structure makes the agent's control flow visible, debuggable, and modifiable.

### 3.2 Core Concepts

**State.** Every LangGraph application has a state object that is passed through the graph. The state accumulates information as nodes process it—conversation messages, intermediate results, tool outputs, decisions. The state type is defined using TypedDict or Pydantic models, providing type safety.

```python
from typing import Annotated, TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages

class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    plan: str
    current_step: int
    results: list[str]
```

**Nodes.** Functions that process the state and return updates. A node might call the LLM, execute a tool, evaluate a condition, or perform any other processing.

```python
def call_model(state: AgentState) -> dict:
    response = model.invoke(state["messages"])
    return {"messages": [response]}

def execute_tool(state: AgentState) -> dict:
    tool_call = state["messages"][-1].tool_calls[0]
    result = tools[tool_call["name"]].invoke(tool_call["args"])
    return {"messages": [ToolMessage(content=result, tool_call_id=tool_call["id"])]}
```

**Edges.** Define transitions between nodes. Edges can be unconditional (always follow this path) or conditional (follow different paths based on state).

```python
def should_continue(state: AgentState) -> str:
    last_message = state["messages"][-1]
    if last_message.tool_calls:
        return "execute_tool"
    return END

graph = StateGraph(AgentState)
graph.add_node("agent", call_model)
graph.add_node("execute_tool", execute_tool)
graph.add_edge(START, "agent")
graph.add_conditional_edges("agent", should_continue)
graph.add_edge("execute_tool", "agent")
agent = graph.compile()
```

### 3.3 Checkpointing

LangGraph's checkpointing system automatically saves the state after each node execution. This enables:

**Resumability.** A long-running agent can be interrupted and resumed from the last checkpoint. This is essential for agents that run for minutes or hours.

**Human-in-the-loop.** The agent can pause at designated breakpoints and wait for human input. The state is saved, and the agent resumes when the human responds.

**Time travel.** Developers can inspect the state at any point in the execution history, enabling detailed debugging.

**Branching.** Multiple executions can branch from the same checkpoint, enabling A/B testing or parallel exploration.

### 3.4 Subgraphs

LangGraph supports composing graphs from subgraphs—smaller graphs that are used as nodes in a larger graph. This enables modular agent design where each sub-agent is a self-contained graph.

### 3.5 Strengths

- Explicit control flow makes agent behavior predictable and debuggable
- Checkpointing provides built-in persistence and human-in-the-loop
- Flexible enough to implement any agent architecture
- Strong ecosystem integration (LangSmith for observability, LangChain for tools)
- Active development and large community

### 3.6 Limitations

- Steep learning curve—the graph abstraction is powerful but complex
- Verbose for simple agents (significant boilerplate for a basic tool loop)
- State management can become complex for large, multi-branch graphs
- Tight coupling with the LangChain ecosystem

## 4. CrewAI

### 4.1 Overview

CrewAI takes a role-based approach to multi-agent systems. Agents are defined by their role, goal, and backstory, and they collaborate as a "crew" to complete tasks. The framework is designed to make multi-agent systems accessible to developers who think in terms of team roles rather than graph topologies.

### 4.2 Core Concepts

**Agent.** A role-playing AI entity with a specific role, goal, backstory, and set of tools.

```python
from crewai import Agent

researcher = Agent(
    role="Research Analyst",
    goal="Find comprehensive information on the given topic",
    backstory="You are an experienced research analyst with expertise "
              "in finding and synthesizing information from multiple sources.",
    tools=[web_search, document_reader],
    llm="claude-sonnet-4-20250514",
    verbose=True
)

writer = Agent(
    role="Technical Writer",
    goal="Write clear, comprehensive technical content",
    backstory="You are a skilled technical writer who can transform "
              "research findings into engaging, well-structured content.",
    tools=[text_editor],
    llm="claude-sonnet-4-20250514"
)
```

**Task.** A specific piece of work assigned to an agent.

```python
from crewai import Task

research_task = Task(
    description="Research the current state of {topic}",
    expected_output="A detailed research report with key findings",
    agent=researcher
)

writing_task = Task(
    description="Write a technical article based on the research",
    expected_output="A well-structured technical article",
    agent=writer,
    context=[research_task]  # Writing depends on research
)
```

**Crew.** A team of agents working together on tasks.

```python
from crewai import Crew, Process

crew = Crew(
    agents=[researcher, writer],
    tasks=[research_task, writing_task],
    process=Process.sequential,  # or Process.hierarchical
    verbose=True
)

result = crew.kickoff(inputs={"topic": "reasoning models"})
```

### 4.3 Process Types

**Sequential.** Tasks are executed in order, with each agent completing its task before the next begins. Simple and predictable.

**Hierarchical.** A manager agent coordinates the crew, assigning tasks to agents and evaluating results. The manager can reassign tasks, request revisions, and determine when the overall objective is met.

### 4.4 Strengths

- Intuitive role-based metaphor that is easy to understand
- Quick to set up for common multi-agent patterns
- Good documentation and examples
- Lower learning curve than graph-based frameworks

### 4.5 Limitations

- Less flexible than graph-based approaches for complex workflows
- Role-based abstraction may not fit all use cases
- Limited control over inter-agent communication
- Less fine-grained control over execution flow

## 5. AutoGen / AG2

### 5.1 Overview

AutoGen, originally developed by Microsoft Research and later continued as the AG2 project, introduced the concept of "conversable agents"—agents that interact through structured conversations. The framework's central insight is that multi-agent collaboration can be modeled as a series of conversations between agents.

### 5.2 Core Concepts

**ConversableAgent.** The base class for all agents. Each agent can send and receive messages, generate responses (using an LLM or custom logic), and invoke tools.

```python
from autogen import ConversableAgent

assistant = ConversableAgent(
    name="assistant",
    system_message="You are a helpful AI assistant.",
    llm_config={"model": "gpt-4o"},
)

user_proxy = ConversableAgent(
    name="user_proxy",
    human_input_mode="TERMINATE",  # Ask human for input at termination
    code_execution_config={"work_dir": "coding"},
)
```

**Two-agent chat.** The simplest interaction pattern: two agents have a conversation until a termination condition is met.

```python
user_proxy.initiate_chat(
    assistant,
    message="Write a Python function to find prime numbers up to N."
)
```

**Group chat.** Multiple agents participate in a conversation, with a GroupChatManager deciding which agent should speak next.

```python
from autogen import GroupChat, GroupChatManager

group_chat = GroupChat(
    agents=[researcher, coder, reviewer],
    messages=[],
    max_round=20,
    speaker_selection_method="auto"  # Manager selects next speaker
)

manager = GroupChatManager(
    groupchat=group_chat,
    llm_config={"model": "gpt-4o"}
)

user_proxy.initiate_chat(manager, message="Build a web scraper for...")
```

### 5.3 Code Execution

AutoGen has strong built-in support for code execution. When an agent generates code, the code can be automatically executed in a sandboxed environment, and the results are fed back into the conversation. This enables iterative coding where the agent writes code, observes the output, and refines the code based on errors.

### 5.4 Nested Conversations

AutoGen supports nested conversations where a single "turn" in an outer conversation triggers an entire inner conversation between other agents. This enables hierarchical agent architectures where high-level decisions trigger detailed sub-conversations.

### 5.5 Strengths

- Flexible conversation topologies
- Strong code execution integration
- Human-in-the-loop as a native pattern
- Nested conversations for complex workflows
- Good for research and experimentation

### 5.6 Limitations

- Complex API with many configuration options
- The conversation-based paradigm does not fit all use cases
- Transition from Microsoft Research to AG2 created some ecosystem fragmentation
- Documentation can be inconsistent across versions

## 6. Semantic Kernel

### 6.1 Overview

Semantic Kernel, developed by Microsoft, is an open-source framework for integrating LLMs into applications. It takes a plugin-based approach where capabilities are organized as "plugins" containing "functions" that can be composed into workflows.

### 6.2 Core Concepts

**Kernel.** The central orchestrator that manages plugins, models, and execution.

**Plugins.** Collections of functions that provide capabilities to the agent. Plugins can contain both "semantic functions" (LLM prompts) and "native functions" (code).

**Planners.** Components that automatically generate execution plans from user goals. The planner examines available plugins and creates a sequence of function calls to achieve the goal.

**Memory.** Built-in support for semantic memory using vector stores.

### 6.3 Strengths

- Strong .NET/C# support (also available in Python and Java)
- Enterprise-focused design with good security and compliance features
- Natural integration with Azure services
- Plugin system encourages modular, reusable components

### 6.4 Limitations

- Historically .NET-first; Python support is solid but lags
- More complex than necessary for simple agent applications
- Enterprise focus can mean more boilerplate for simple use cases
- Smaller community than LangChain/LangGraph

## 7. Anthropic Agent SDK

### 7.1 Overview

The Anthropic Agent SDK (also called Claude Agent SDK) is a Python framework for building agents powered by Claude models. It provides an opinionated but flexible structure for building agents with tool use, extended thinking, and multi-agent architectures.

### 7.2 Design Philosophy

The SDK emphasizes:

**Simplicity.** A basic agent can be created in a few lines of code. Complexity is additive—you start simple and add capabilities as needed.

**Native Claude features.** Deep integration with Claude's capabilities: extended thinking, tool use, multi-turn conversations, and prompt caching.

**Type safety.** Python type hints and Pydantic models for tool definitions, state, and agent configuration.

**Composability.** Agents can be composed into multi-agent systems through handoff mechanisms.

### 7.3 Core Concepts

```python
from claude_agent_sdk import Agent, tool

@tool
def search_web(query: str) -> str:
    """Search the web for information."""
    return web_search(query)

@tool
def read_file(path: str) -> str:
    """Read the contents of a file."""
    return open(path).read()

agent = Agent(
    model="claude-sonnet-4-20250514",
    tools=[search_web, read_file],
    system_prompt="You are a helpful research assistant.",
    max_turns=20
)

result = agent.run("What are the latest developments in quantum computing?")
```

### 7.4 Handoffs

The SDK supports agent handoffs—one agent transferring control to another. This enables building multi-agent systems where different agents handle different aspects of a task.

```python
coding_agent = Agent(
    model="claude-sonnet-4-20250514",
    tools=[code_editor, run_tests],
    system_prompt="You are a coding specialist."
)

review_agent = Agent(
    model="claude-sonnet-4-20250514",
    tools=[read_file, code_analysis],
    system_prompt="You are a code review specialist."
)

orchestrator = Agent(
    model="claude-sonnet-4-20250514",
    handoffs=[coding_agent, review_agent],
    system_prompt="Route coding tasks to the coding agent and "
                  "review tasks to the review agent."
)
```

### 7.5 Strengths

- Clean, minimal API
- Deep Claude integration (extended thinking, prompt caching)
- Good default behaviors with minimal configuration
- Type-safe tool definitions

### 7.6 Limitations

- Claude-specific (does not support other model providers)
- Newer framework with a smaller ecosystem
- Fewer built-in patterns than more mature frameworks

## 8. OpenAI Agents SDK

### 8.1 Overview

The OpenAI Agents SDK provides a framework for building agents powered by OpenAI models. It includes built-in support for tool use, agent handoffs, guardrails, and tracing.

### 8.2 Core Concepts

**Agent.** An LLM configured with instructions, tools, and optional handoff targets.

```python
from openai_agents import Agent, Runner

agent = Agent(
    name="research_assistant",
    instructions="You are a helpful research assistant.",
    model="gpt-4o",
    tools=[web_search, calculator]
)

result = Runner.run_sync(agent, "What is the GDP of France?")
```

**Handoffs.** Agents can hand off conversations to other agents, similar to the Anthropic SDK.

**Guardrails.** Input and output validation that runs before and after the agent's response. Guardrails can block inappropriate inputs, validate outputs, and enforce policies.

```python
from openai_agents import Guardrail

content_filter = Guardrail(
    name="content_filter",
    instructions="Block requests for harmful content.",
    model="gpt-4o-mini"
)

agent = Agent(
    name="assistant",
    guardrails=[content_filter],
    ...
)
```

**Tracing.** Built-in tracing that records every agent step for debugging and monitoring.

### 8.3 Strengths

- Simple, clean API
- Built-in guardrails for safety
- Native tracing for observability
- Good integration with OpenAI's model ecosystem

### 8.4 Limitations

- OpenAI-specific
- Relatively new with a developing ecosystem
- Fewer advanced features than LangGraph

## 9. smolagents

### 9.1 Overview

smolagents, developed by Hugging Face, is a lightweight framework for building agents that emphasizes simplicity and code-based tool execution. Its distinctive feature is that agents can write and execute Python code as their primary action mechanism, rather than generating structured tool calls.

### 9.2 Code-Based Actions

Instead of generating JSON tool calls, smolagents agents generate Python code that calls tools directly:

```python
from smolagents import CodeAgent, tool, HfApiModel

@tool
def get_weather(city: str) -> str:
    """Get current weather for a city."""
    return weather_api.get(city)

agent = CodeAgent(
    tools=[get_weather],
    model=HfApiModel("meta-llama/Llama-3.3-70B-Instruct")
)

agent.run("What's the weather in Paris and London?")
```

The agent might generate:
```python
paris_weather = get_weather("Paris")
london_weather = get_weather("London")
print(f"Paris: {paris_weather}\nLondon: {london_weather}")
```

This code-based approach enables natural parallel tool calls, complex control flow, and the full expressiveness of Python without requiring the framework to support these features explicitly.

### 9.3 Strengths

- Extremely lightweight and easy to learn
- Code-based actions are natural and flexible
- Good integration with Hugging Face ecosystem
- Works well with open-source models
- Multi-model support out of the box

### 9.4 Limitations

- Code execution introduces security concerns
- Less structured than JSON-based tool calling
- Fewer built-in patterns for complex workflows
- Smaller ecosystem of extensions

## 10. Pydantic AI

### 10.1 Overview

Pydantic AI is a framework that brings Pydantic's type validation philosophy to LLM agent development. Every input, output, tool parameter, and state value is validated through Pydantic models, providing strong type safety throughout the agent lifecycle.

### 10.2 Core Concepts

```python
from pydantic_ai import Agent
from pydantic import BaseModel

class ResearchResult(BaseModel):
    topic: str
    key_findings: list[str]
    sources: list[str]
    confidence: float

agent = Agent(
    model="claude-sonnet-4-20250514",
    result_type=ResearchResult,  # Structured, validated output
    system_prompt="You are a research analyst."
)

result = agent.run_sync("Research the impact of reasoning models on AI development")
# result.data is a validated ResearchResult instance
```

### 10.3 Dependency Injection

Pydantic AI uses dependency injection to provide tools and context to agents. Dependencies are type-checked and injected at runtime.

```python
from pydantic_ai import Agent, RunContext

class DatabaseDeps:
    def __init__(self, connection_string: str):
        self.db = connect(connection_string)

agent = Agent(
    model="gpt-4o",
    deps_type=DatabaseDeps
)

@agent.tool
def query_database(ctx: RunContext[DatabaseDeps], sql: str) -> str:
    """Execute a SQL query."""
    return ctx.deps.db.execute(sql)
```

### 10.4 Strengths

- Strong type safety throughout
- Structured output validation
- Clean dependency injection
- Multi-model support
- Pythonic API design

### 10.5 Limitations

- Focused on single-agent patterns (multi-agent requires additional orchestration)
- Newer framework with a smaller community
- Less built-in support for complex workflows

## 11. DSPy

### 11.1 Overview

DSPy (Declarative Self-improving Language Programs) takes a fundamentally different approach to agent development. Rather than manually writing prompts and tool definitions, DSPy allows developers to declare what the agent should do, and the framework optimizes the prompts and few-shot examples automatically.

### 11.2 Core Concepts

**Signatures.** Declarative specifications of input-output behavior.

```python
import dspy

class AnswerQuestion(dspy.Signature):
    """Answer a factual question with a sourced answer."""
    question: str = dspy.InputField()
    answer: str = dspy.OutputField()
    sources: list[str] = dspy.OutputField()
```

**Modules.** Composable processing steps that implement signatures.

```python
class ResearchAgent(dspy.Module):
    def __init__(self):
        self.search = dspy.Retrieve(k=5)
        self.answer = dspy.ChainOfThought(AnswerQuestion)
    
    def forward(self, question):
        context = self.search(question)
        return self.answer(question=question, context=context)
```

**Optimizers.** Automatically improve the module's prompts and few-shot examples based on training data.

```python
optimizer = dspy.MIPROv2(metric=answer_accuracy)
optimized_agent = optimizer.compile(
    ResearchAgent(),
    trainset=training_examples
)
```

### 11.3 The Optimization Approach

DSPy's key innovation is treating prompt engineering as an optimization problem. Given a dataset of (input, expected output) pairs and a metric, DSPy searches for the best combination of prompts, few-shot examples, and module configurations. This can significantly improve agent performance without manual prompt tuning.

### 11.4 Strengths

- Automatic prompt optimization
- Modular, composable design
- Reduces manual prompt engineering
- Can significantly improve performance on specific tasks

### 11.5 Limitations

- Requires training data for optimization
- Learning curve for the declarative paradigm
- Less intuitive for developers used to imperative programming
- May not suit tasks where the optimal prompt is context-dependent

## 12. Framework Comparison

### 12.1 Comparison Table

| Feature | LangGraph | CrewAI | AutoGen/AG2 | Semantic Kernel | Anthropic SDK | OpenAI SDK | smolagents | Pydantic AI | DSPy |
|---|---|---|---|---|---|---|---|---|---|
| **Primary Pattern** | Graph/state machine | Role-based crews | Conversations | Plugin-based | Tool loop + handoff | Tool loop + handoff | Code execution | Type-safe tools | Declarative optimization |
| **Multi-Agent** | Yes (subgraphs) | Yes (core feature) | Yes (core feature) | Yes | Yes (handoffs) | Yes (handoffs) | Limited | Limited | Composable modules |
| **Multi-Model** | Yes | Yes | Yes | Yes | Claude only | OpenAI only | Yes | Yes | Yes |
| **Checkpointing** | Built-in | Limited | Limited | Limited | Limited | Limited | No | No | No |
| **Human-in-Loop** | Built-in | Limited | Built-in | Limited | Manual | Manual | No | No | No |
| **Observability** | LangSmith | Limited built-in | Built-in logging | Azure integration | Basic | Built-in tracing | Basic | Logfire | Basic |
| **Learning Curve** | High | Low-Medium | Medium-High | Medium | Low | Low | Low | Low | High |
| **Maturity** | High | Medium | Medium | High | Low-Medium | Low-Medium | Medium | Low-Medium | Medium |
| **Best For** | Complex, stateful agents | Multi-agent teams | Research, code agents | Enterprise/.NET | Claude-native apps | OpenAI-native apps | Quick prototyping | Type-safe apps | Optimized pipelines |

### 12.2 Decision Framework

**Choose LangGraph when:**
- You need complex, stateful workflows with conditional branching
- Checkpointing and human-in-the-loop are requirements
- You want maximum flexibility and control over the agent architecture
- You are building a production system that must be reliable and debuggable

**Choose CrewAI when:**
- You want to build multi-agent systems quickly
- The role-based metaphor fits your use case
- You do not need fine-grained control over agent interactions
- You want a lower learning curve than LangGraph

**Choose AutoGen/AG2 when:**
- Conversation-based agent interaction fits your use case
- You need strong code execution integration
- You want flexible multi-agent topologies
- You are doing research on multi-agent systems

**Choose Semantic Kernel when:**
- You are building in a .NET/C# environment
- Enterprise integration (Azure, Office 365) is important
- You want a plugin-based architecture with good modularity

**Choose Anthropic Agent SDK when:**
- You are building exclusively with Claude models
- You want the simplest possible setup for a Claude-powered agent
- Deep integration with Claude features (extended thinking, caching) matters

**Choose OpenAI Agents SDK when:**
- You are building exclusively with OpenAI models
- Built-in guardrails and tracing are important
- You want a simple setup for an OpenAI-powered agent

**Choose smolagents when:**
- You want the simplest possible framework
- Code-based actions are natural for your use case
- You are working with Hugging Face models
- You need quick prototyping

**Choose Pydantic AI when:**
- Type safety and structured outputs are priorities
- You want clean dependency injection
- You are building single-agent applications with well-defined inputs and outputs

**Choose DSPy when:**
- You have training data for optimization
- You want to minimize manual prompt engineering
- Your use case benefits from automated prompt tuning
- You are comfortable with the declarative programming paradigm

## 13. When to Use a Framework vs. Build Custom

### 13.1 Use a Framework When

- You want to build quickly and are comfortable with the framework's abstractions
- Your use case fits one of the standard patterns (ReAct, multi-agent, pipeline)
- You need observability, checkpointing, or human-in-the-loop
- You have a team and want consistent patterns across projects
- You are not sure which architecture will work best and want to experiment

### 13.2 Build Custom When

- Your use case has unique requirements that frameworks cannot accommodate
- You need maximum control over every aspect of the agent's behavior
- The framework's overhead (dependencies, abstractions, performance) is unacceptable
- Your agent is simple enough that a framework adds more complexity than it removes
- You are building a framework yourself (for a product or platform)

### 13.3 The Thin Wrapper Approach

A middle ground is the thin wrapper: a small amount of custom code around direct API calls. This provides the control of custom development with some of the convenience of a framework.

```python
class SimpleAgent:
    def __init__(self, model, tools, system_prompt, max_turns=10):
        self.model = model
        self.tools = {t.name: t for t in tools}
        self.system_prompt = system_prompt
        self.max_turns = max_turns
    
    def run(self, user_message):
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_message}
        ]
        
        for turn in range(self.max_turns):
            response = self.model.chat(messages, tools=list(self.tools.values()))
            messages.append(response.message)
            
            if not response.tool_calls:
                return response.content
            
            for call in response.tool_calls:
                result = self.tools[call.name].execute(call.args)
                messages.append({
                    "role": "tool",
                    "tool_call_id": call.id,
                    "content": str(result)
                })
        
        return messages[-1].content
```

This is approximately 30 lines of code and handles the basic agent loop. For many applications, this is sufficient. When it is not, the specific gaps (checkpointing, streaming, multi-agent) identify exactly what framework features are needed.

## 14. Lock-In Risks

### 14.1 Framework Lock-In

Adopting a framework means accepting its abstractions, patterns, and dependencies. Switching frameworks later requires rewriting significant portions of the application. This lock-in risk is important to consider when choosing a framework.

**Mitigation**: Choose frameworks with clear abstraction boundaries. Isolate framework-specific code from business logic. Use adapter patterns so that the core agent logic does not depend on framework-specific types.

### 14.2 Provider Lock-In

Provider-specific SDKs (Anthropic Agent SDK, OpenAI Agents SDK) bind you to a single model provider. If you later want to switch providers or use multiple providers, the SDK's assumptions may not translate.

**Mitigation**: Use multi-model frameworks (LangGraph, Pydantic AI, smolagents) if provider flexibility is important. Or build a thin abstraction layer over the provider SDK that allows swapping the underlying model.

### 14.3 Abstraction Lock-In

Frameworks impose specific abstractions (graphs, conversations, crews) that shape how you think about your agent. If the abstraction does not fit your use case, you fight the framework rather than benefit from it.

**Mitigation**: Evaluate whether the framework's core abstraction fits your mental model of the problem before committing. Build a proof of concept with the framework before adopting it for production.

## 15. Observability

### 15.1 Why Observability Matters

Agent behavior is non-deterministic. The same input can lead to different tool call sequences, different reasoning paths, and different outputs on different runs. Without observability, debugging failed runs is guesswork. With observability, every decision point is visible, every tool call is logged, and the full trajectory from input to output can be inspected.

### 15.2 LangSmith

LangSmith, developed by LangChain, is the most widely used observability platform for LLM agents. It provides:

- **Trace visualization**: A hierarchical view of every step in an agent run—LLM calls, tool calls, chain executions, with timing and token counts.
- **Run comparison**: Side-by-side comparison of different runs to identify what changed.
- **Evaluation**: Run agents on test datasets and compute metrics.
- **Monitoring**: Real-time dashboards of agent performance, cost, and error rates.
- **Feedback**: Capture human feedback on agent outputs for quality tracking.

LangSmith works natively with LangChain and LangGraph but also supports other frameworks through its tracing API.

### 15.3 Arize Phoenix

Phoenix, developed by Arize AI, provides open-source observability for LLM applications:

- **Trace visualization**: Similar to LangSmith but open-source and self-hostable.
- **Span analysis**: Detailed analysis of individual spans (LLM calls, tool calls) with timing and cost.
- **Evaluation**: Built-in evaluators for relevance, faithfulness, and other quality metrics.
- **OpenTelemetry integration**: Uses the OpenTelemetry standard for tracing, enabling integration with existing observability infrastructure.

### 15.4 Braintrust

Braintrust focuses on evaluation and experimentation for LLM applications:

- **Evaluation framework**: Define evaluation criteria and run agents against test datasets.
- **Experiment tracking**: Compare different agent configurations and prompt versions.
- **Online evaluation**: Monitor production agent performance in real-time.
- **Logging**: Comprehensive logging of all LLM interactions.

### 15.5 Provider-Native Observability

Both OpenAI and Anthropic provide some observability through their dashboards:

- **OpenAI**: Token usage tracking, rate limit monitoring, content filter logging
- **Anthropic**: Usage tracking, prompt caching metrics, conversation logging (through the Console)

These are useful for basic monitoring but lack the depth needed for debugging complex agent behavior.

### 15.6 Building Observability

For teams building their own observability, key components include:

**Structured logging.** Log every model call, tool call, and state transition with structured data (JSON) that can be queried and analyzed.

**Trace IDs.** Assign a unique trace ID to each agent run and propagate it through all steps. This enables correlating all events in a single run.

**Timing.** Record the duration of every step to identify bottlenecks.

**Token counting.** Track input and output tokens for every model call to monitor costs.

**Error classification.** Categorize errors (model error, tool error, timeout, rate limit) to identify patterns.

**Dashboards.** Build dashboards that show key metrics: success rate, average latency, cost per run, error rate by type.

## 16. Production Patterns

### 16.1 Deployment Architecture

Production agent deployments typically include:

- **API layer**: Receives user requests and returns agent responses
- **Agent execution**: Runs the agent logic (framework, model calls, tool execution)
- **Tool servers**: External services that implement tools (databases, APIs, file systems)
- **State store**: Persists conversation history, checkpoints, and memory
- **Observability stack**: Logging, tracing, monitoring, alerting
- **Queue/worker**: For long-running or asynchronous agent tasks

### 16.2 Reliability Patterns

**Circuit breakers.** If a tool or model consistently fails, stop calling it temporarily to avoid cascading failures.

**Timeouts.** Set timeouts on every external call (model API, tool execution) to prevent indefinite hanging.

**Rate limiting.** Limit the rate of model calls and tool calls to stay within provider limits and control costs.

**Fallback models.** If the primary model is unavailable, fall back to an alternative model rather than failing entirely.

**Graceful degradation.** If specific capabilities are unavailable (a tool is down, a model is overloaded), provide a degraded but functional experience rather than a complete failure.

### 16.3 Cost Control

**Token budgets.** Set maximum token budgets per request and per session. Stop the agent when the budget is exhausted.

**Model routing.** Use cheaper models for simple tasks and expensive models only for complex tasks.

**Caching.** Cache tool results and model responses where appropriate.

**Alerting.** Alert when cost per request exceeds a threshold, indicating a possible runaway agent loop.

### 16.4 Security

**Input sanitization.** Validate and sanitize user input before passing it to the agent.

**Tool sandboxing.** Execute tools in sandboxed environments with limited permissions.

**Output filtering.** Filter agent output for sensitive information before returning it to the user.

**Authentication and authorization.** Ensure that tool access is scoped to the user's permissions.

**Audit logging.** Log all agent actions for security review and compliance.

## 17. Conclusion

The agent framework ecosystem in 2026 offers a rich set of options for building LLM-powered agents. LangGraph provides the most flexible and mature platform for complex, stateful agents. CrewAI offers the most accessible path to multi-agent systems. Provider SDKs from Anthropic and OpenAI offer the simplest setup for their respective models. DSPy provides a unique optimization-driven approach. And lightweight options like smolagents and Pydantic AI serve developers who want minimal abstractions.

The choice of framework depends on the specific requirements of the application: the complexity of the workflow, the need for multi-model or multi-agent support, the importance of observability and debugging, the team's familiarity with different paradigms, and the acceptable level of vendor lock-in.

For practitioners, the most important advice is to resist the temptation to over-engineer. Start with the simplest approach that could work—often a thin wrapper around direct API calls. Adopt a framework when the specific features it provides (checkpointing, multi-agent coordination, observability integration) are genuinely needed, not as a default choice. Evaluate frameworks with a proof of concept before committing to them for production. And invest in observability from the beginning—the ability to see what your agent is doing is worth more than any architectural pattern.
