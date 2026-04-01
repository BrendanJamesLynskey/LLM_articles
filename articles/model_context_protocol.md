# Model Context Protocol (MCP): A Universal Interface for LLM Tool Integration

*April 2026 • Technical Report*

## 1. Introduction

Large language models are increasingly deployed not as isolated text generators but as agents that interact with external systems — databases, APIs, file systems, code interpreters, and enterprise tools. Each integration has historically required bespoke glue code: custom function definitions, provider-specific tool-calling formats, and ad hoc serialization schemes. The result is an N×M integration problem in which N LLM applications must each implement connectors for M external tools, leading to duplicated effort, inconsistent behavior, and fragile deployments.

The **Model Context Protocol (MCP)**, introduced by Anthropic in late 2024 and rapidly adopted across the industry through 2025–2026, addresses this problem by defining a universal, open standard for connecting LLM applications to external data sources and tools. MCP provides a structured client-server protocol — analogous to what USB did for peripheral hardware or what the Language Server Protocol (LSP) did for IDE tooling — that allows any compliant LLM host to discover, invoke, and compose tools exposed by any compliant MCP server, without bespoke integration code on either side.

This report provides a detailed technical examination of MCP: its architecture, protocol mechanics, transport layers, security model, ecosystem, and practical deployment considerations.

## 2. The Problem MCP Solves

### 2.1 The N×M Integration Problem

Before MCP, connecting an LLM-powered application to external capabilities required writing tool-specific integration code for each combination of host application and external service. A coding assistant that needs access to GitHub, a database, and a documentation search engine must implement three separate integrations, each with its own schema, authentication, error handling, and serialization logic. A different coding assistant wanting the same three capabilities must re-implement all of them from scratch. This scales poorly: N applications accessing M tools require up to N×M custom integrations.

### 2.2 Context Injection Was Fragile

Before MCP, context injection into LLM prompts was largely manual. Developers wrote scripts to fetch relevant documents, database rows, or API responses and stuffed them into system prompts or user messages. This approach suffered from several problems: context was often stale by the time the model processed it, there was no standardized way to indicate the provenance or freshness of injected data, and the model had no mechanism to request additional context mid-conversation when it realized it needed more information.

### 2.3 Tool Discovery Was Static

Traditional function-calling approaches require tools to be defined at prompt construction time. The set of available tools is fixed for the duration of a conversation, and the model has no way to discover new capabilities that might be relevant to an evolving task. MCP's dynamic tool discovery mechanism allows the model to query available tools at runtime, enabling adaptive workflows that respond to the actual needs of the task.

## 3. Architecture

### 3.1 The Three Roles

MCP defines three distinct roles in its architecture:

**MCP Hosts** are the LLM-facing applications that initiate connections and mediate between the model and external services. Examples include Claude Desktop, IDE extensions (such as Claude Code, Cursor, Windsurf, or Cline), and custom AI agents. The host is responsible for managing the lifecycle of MCP connections, routing tool calls from the model to the appropriate server, and returning results back to the model.

**MCP Clients** are protocol-level connectors that maintain a one-to-one connection with a specific MCP server. Each client handles the JSON-RPC communication, capability negotiation, and transport management for a single server connection. A host application typically manages multiple clients simultaneously — one per connected server.

**MCP Servers** are lightweight programs that expose specific capabilities through the standardized MCP interface. A server might wrap a database, a web API, a local file system, or any other data source or tool. Servers are designed to be small, focused, and composable — a single server typically exposes a coherent set of related capabilities rather than attempting to be a universal connector.

### 3.2 The Capability Primitives

MCP defines six core primitives that servers can expose and clients can consume:

**Tools** are executable functions that the model can invoke to perform actions or retrieve computed results. Each tool has a name, a human-readable description, and a JSON Schema defining its input parameters. Tools are model-controlled — the LLM decides when and how to call them based on the user's request and the tool's description. Examples include `query_database`, `create_github_issue`, or `run_python_code`.

**Resources** are data sources that provide context to the model. Unlike tools, resources are typically application-controlled — the host application decides which resources to attach to a conversation based on user selections or heuristics. Resources are identified by URIs (e.g., `file:///path/to/document.md` or `postgres://server/db/table`) and can be static (read once) or dynamic (updated on change via subscriptions). Resources are analogous to opening a file in an IDE — they provide contextual information that informs the model's responses.

**Prompts** are reusable prompt templates that servers can expose for specific workflows. A database server might expose a `debug_query` prompt template that structures a natural-language request into an effective debugging workflow. Prompts are user-controlled — they appear as options in the host's UI (e.g., slash commands) and are selected explicitly by the user.

**Sampling** allows servers to request LLM completions through the client, enabling agentic behaviors where a tool can itself use the model to reason about intermediate steps. This creates a bidirectional flow: the model calls a tool, which calls back to the model for additional reasoning, which may call further tools. Sampling requests go through the host, which maintains control over model access and can enforce approval policies.

**Roots** declare the filesystem or URI scopes that a server is expected to operate within. A server handling a Git repository might declare `file:///home/user/project` as a root, informing the client of its operational boundary. Roots are informational — they help clients understand server scope but are not enforced as a hard security boundary by the protocol itself.

**Logging** provides a structured mechanism for servers to emit log messages at various severity levels (debug, info, warning, error) back to the client. This enables observability and debugging of MCP server behavior without requiring separate logging infrastructure.

### 3.3 Capability Negotiation

When a client connects to a server, they perform a capability negotiation handshake. The client sends an `initialize` request specifying the protocol version it supports and the client capabilities it offers (such as sampling support or root declarations). The server responds with its own capabilities, indicating which primitives it supports (tools, resources, prompts, etc.). This negotiation ensures that both sides have a clear understanding of what the connection supports before any tool calls or resource reads occur.

## 4. Protocol Mechanics

### 4.1 JSON-RPC 2.0 Foundation

MCP is built on JSON-RPC 2.0, the same lightweight remote procedure call protocol used by LSP. Every message is a JSON object with a `jsonrpc: "2.0"` field and falls into one of three categories:

- **Requests** include an `id`, a `method` string, and optional `params`. They expect a response.
- **Responses** include the matching `id` and either a `result` or an `error` object.
- **Notifications** include a `method` and optional `params` but no `id`. They are fire-and-forget with no response expected.

This structure provides a familiar, well-understood foundation with clear semantics for request-response patterns, error handling, and asynchronous notifications.

### 4.2 The Connection Lifecycle

A typical MCP session proceeds through these phases:

1. **Transport establishment**: The client starts the server process (for stdio transport) or connects to its HTTP endpoint (for HTTP+SSE or Streamable HTTP transport).
2. **Initialization**: The client sends `initialize` with its protocol version and capabilities. The server responds with its capabilities and instructions. The client sends `initialized` as a notification to confirm.
3. **Operation**: The client and server exchange requests and notifications. The client may call `tools/list` to discover available tools, `tools/call` to invoke them, `resources/list` and `resources/read` to access data, or `prompts/list` and `prompts/get` to retrieve prompt templates.
4. **Shutdown**: Either side may terminate the connection. The client sends a `close` notification, or the transport is terminated directly.

### 4.3 Tool Invocation Flow

When the LLM decides to call a tool, the following sequence occurs:

1. The host application receives the model's tool-call request (in the model's native function-calling format).
2. The host identifies which MCP server exposes the requested tool.
3. The host's MCP client sends a `tools/call` JSON-RPC request to that server, including the tool name and arguments.
4. The server executes the tool logic (querying a database, calling an API, reading files, etc.).
5. The server returns a `tools/call` response containing the result as structured content (text, images, or embedded resources).
6. The host formats the result and feeds it back to the model as a tool-call response in the model's native format.
7. The model incorporates the result into its reasoning and continues generating.

### 4.4 Dynamic Tool Updates

MCP supports dynamic tool registration. A server can notify the client that its tool list has changed via a `notifications/tools/list_changed` notification. The client can then re-fetch the tool list with `tools/list`. This enables servers whose capabilities evolve at runtime — for example, a database server that exposes query tools only after a successful connection, or a plugin system that loads new tools on demand.

## 5. Transport Layers

### 5.1 stdio (Standard Input/Output)

The simplest transport: the client launches the server as a child process and communicates via stdin/stdout. Each JSON-RPC message is written as a single line. This transport is ideal for local tool servers, requires no network configuration, and inherits the security context of the host process. It is the default transport for most desktop MCP integrations (Claude Desktop, IDE extensions).

### 5.2 HTTP with Server-Sent Events (SSE)

For remote servers, the original MCP specification defined an HTTP+SSE transport. The client connects to an SSE endpoint to receive server-initiated messages and sends requests via HTTP POST to a separate endpoint. This transport supports network-accessible servers but requires maintaining a persistent SSE connection.

### 5.3 Streamable HTTP

Introduced in the 2025-03-26 protocol revision, Streamable HTTP is the current recommended transport for remote servers. It unifies the request and streaming channels into a single HTTP endpoint. Clients send JSON-RPC requests via POST; the server can respond with a single JSON response (Content-Type: application/json) or upgrade to an SSE stream (Content-Type: text/event-stream) for long-running operations or server-initiated notifications. This is more firewall-friendly, simpler to deploy behind load balancers, and supports stateless server architectures. An optional `Mcp-Session-Id` header enables session affinity for stateful servers.

### 5.4 Transport Selection Guidance

| Scenario | Recommended Transport | Rationale |
|---|---|---|
| Local tools (file access, local DB) | stdio | Simplest, no network exposure, inherits OS permissions |
| Remote APIs, cloud services | Streamable HTTP | Network-accessible, supports load balancing, stateless option |
| Legacy deployments | HTTP+SSE | Still supported but being superseded by Streamable HTTP |
| High-security environments | stdio + SSH tunnel | Keeps server local, uses SSH for remote access |

## 6. Security Model

### 6.1 Threat Landscape

MCP expands the attack surface of LLM applications in important ways. A compromised or malicious MCP server could inject misleading context into the model's reasoning, exfiltrate sensitive data through tool arguments, or perform unauthorized actions on backend systems. Prompt injection attacks become more potent when the model has access to tools that can read files, query databases, or modify external state.

### 6.2 Consent and Human-in-the-Loop

The MCP specification mandates that hosts implement user consent flows for sensitive operations. Before a tool executes an action with side effects (writing to a database, sending a message, modifying files), the host should present the action to the user for approval. This human-in-the-loop requirement is a fundamental security principle of MCP — the protocol is designed to keep humans informed and in control of consequential actions.

### 6.3 Principle of Least Privilege

MCP servers should request and be granted only the minimum permissions necessary for their function. A server that reads documentation should not have write access to the file system. A database query server should use read-only database credentials. The `roots` primitive supports this by declaring the expected operational scope, though enforcement is left to the host and the server's deployment configuration.

### 6.4 Input Validation and Sanitization

Servers must validate all inputs from clients, and clients must validate all outputs from servers. Tool arguments should be checked against their declared JSON Schema before execution. Tool results should be sanitized before injection into model prompts to prevent indirect prompt injection — a scenario where a malicious document read by a tool contains instructions that manipulate the model's behavior.

### 6.5 Authentication and Authorization

For remote MCP servers, the specification recommends OAuth 2.1 for authentication. The 2025-03-26 protocol revision introduced a formal authorization framework where servers can declare their authentication requirements during connection setup, and clients handle the OAuth flow (including PKCE for public clients). This enables enterprise deployments where MCP servers are protected by identity providers and role-based access controls.

### 6.6 Tool Poisoning and Server Trust

A key security concern is **tool poisoning**: a malicious MCP server that advertises benign-looking tools whose descriptions contain hidden instructions for the model. For example, a tool described as "search documentation" might include hidden text in its description instructing the model to exfiltrate the contents of other tool calls. Mitigations include: reviewing tool descriptions before approval, using allowlists of trusted servers, and host-level filtering of tool descriptions for suspicious content.

## 7. Ecosystem and Adoption

### 7.1 Reference Implementations and SDKs

Anthropic and the open-source community have produced official MCP SDKs in multiple languages:

- **TypeScript SDK** (`@modelcontextprotocol/sdk`): The reference implementation, with full support for all primitives and transports.
- **Python SDK** (`mcp`): Comprehensive Python support using asyncio, with both client and server implementations.
- **Java/Kotlin SDK**: For JVM-based server implementations, commonly used in enterprise environments.
- **C# SDK**: For .NET ecosystem integration.
- **Go, Rust, Swift SDKs**: Community-maintained implementations covering additional language ecosystems.

### 7.2 Pre-Built Servers

A rich ecosystem of pre-built MCP servers has emerged, covering common integration scenarios:

- **File system**: Read, write, search, and manage local files within declared root directories.
- **GitHub/GitLab**: Repository management, issue tracking, pull request operations, code search.
- **Databases**: PostgreSQL, MySQL, SQLite, MongoDB — schema inspection, query execution, data exploration.
- **Web**: Fetch and parse web pages, search engines, Brave Search integration.
- **Productivity**: Slack, Google Drive, Notion, Linear, Jira — read and write to collaboration tools.
- **Development**: Docker management, Kubernetes operations, CI/CD pipeline interaction.
- **Memory and knowledge**: Persistent memory stores, vector databases, knowledge graph access.

### 7.3 Host Application Support

MCP has been adopted by a wide range of LLM host applications:

- **Claude Desktop and Claude Code**: Anthropic's own applications were the first MCP hosts, supporting both local and remote servers.
- **IDE extensions**: Cursor, Windsurf, Cline, Continue, and Zed have integrated MCP support, allowing developers to connect their AI coding assistants to project-specific tools.
- **Agent frameworks**: LangChain, CrewAI, AutoGen, and other agent frameworks support MCP as a tool integration mechanism, enabling agents to use any MCP server as a tool source.
- **Enterprise platforms**: Amazon Bedrock, Cloudflare Workers AI, and other cloud platforms have announced or shipped MCP support, enabling server-side MCP integration.

### 7.4 The Registry and Discovery

The MCP ecosystem includes server registries and discovery mechanisms. Community-maintained registries (such as the official MCP servers repository on GitHub, Smithery, Glama, and others) catalog available servers with descriptions, configuration examples, and compatibility information. Some host applications integrate registry browsing directly into their UI, allowing users to discover and install MCP servers without manual configuration.

## 8. Building an MCP Server

### 8.1 Server Structure (Python Example)

A minimal MCP server in Python illustrates the simplicity of the interface:

```python
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("weather-server")

@mcp.tool()
async def get_weather(city: str) -> str:
    """Get the current weather for a city."""
    # In production, call a real weather API
    return f"Weather in {city}: 22°C, partly cloudy"

@mcp.resource("config://settings")
async def get_settings() -> str:
    """Return server configuration as a resource."""
    return "timezone: UTC\nunits: metric"

@mcp.prompt()
async def weather_report(city: str) -> str:
    """Generate a structured weather report prompt."""
    return f"Please provide a detailed weather analysis for {city}, including temperature, humidity, wind, and forecast."

if __name__ == "__main__":
    mcp.run(transport="stdio")
```

The `FastMCP` class handles all protocol mechanics — JSON-RPC serialization, capability negotiation, transport management — allowing the developer to focus on the tool logic. Type hints on function parameters are automatically converted to JSON Schema for tool definitions.

### 8.2 Server Configuration in Hosts

Host applications typically configure MCP servers through a JSON configuration file. For Claude Desktop, this is `claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "weather": {
      "command": "python",
      "args": ["weather_server.py"],
      "env": {
        "API_KEY": "..."
      }
    },
    "database": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-postgres", "postgresql://localhost/mydb"]
    }
  }
}
```

Each entry specifies the command to launch the server process, any arguments, and environment variables. The host manages the server lifecycle — starting it when needed, restarting on crash, and shutting it down on exit.

### 8.3 Remote Server Configuration

For remote MCP servers using Streamable HTTP transport:

```json
{
  "mcpServers": {
    "remote-analytics": {
      "type": "streamable-http",
      "url": "https://analytics.example.com/mcp",
      "headers": {
        "Authorization": "Bearer <token>"
      }
    }
  }
}
```

## 9. MCP and Agentic Workflows

### 9.1 Composing Tools into Workflows

MCP's real power emerges when models compose multiple tools into multi-step workflows. A developer asks their AI assistant to "find and fix the performance regression in the latest PR." The model might:

1. Call a GitHub MCP tool to list recent PRs and identify the relevant one.
2. Call the GitHub tool again to fetch the diff.
3. Call a file system tool to read the affected source files for full context.
4. Call a database tool to query performance benchmarks from CI.
5. Reason about the cause and generate a fix.
6. Call the file system tool to write the corrected code.
7. Call the GitHub tool to create a new PR with the fix.

Each step uses a different MCP tool (potentially from different servers), but the model orchestrates them seamlessly because they all speak the same protocol.

### 9.2 Sampling and Recursive Agents

The sampling primitive enables a powerful pattern: an MCP server can itself request LLM completions through the host. This allows building "inner agents" — tool servers that use AI reasoning as part of their execution. For example, a code review server might use sampling to have the model analyze each changed file, then aggregate the results into a structured review. The host maintains control by mediating all sampling requests and enforcing rate limits or approval policies.

### 9.3 Multi-Agent Coordination

In multi-agent architectures, MCP provides a natural integration layer. Each agent can be an MCP host with its own set of connected servers, or agents can expose their capabilities as MCP servers that other agents consume. This composability enables hierarchical agent systems where a planning agent delegates to specialist agents, each equipped with domain-specific MCP tools.

## 10. Comparison with Alternatives

### 10.1 MCP vs. Native Function Calling

Most LLM providers offer native function-calling (tool-use) APIs. These define tools as JSON Schema objects in the API request. MCP does not replace this mechanism — it sits above it. Native function calling defines how the model requests tool invocations; MCP defines how the host discovers and executes those tools on the server side. MCP-connected tools are ultimately presented to the model through its native function-calling interface.

### 10.2 MCP vs. OpenAPI/REST

OpenAPI specifications describe REST APIs in a machine-readable format. An LLM could theoretically use an OpenAPI spec directly as a tool definition. However, OpenAPI was designed for developer documentation, not for LLM consumption. It lacks features MCP provides: capability negotiation, resource subscriptions, prompt templates, sampling, dynamic tool updates, and a defined lifecycle. MCP servers can wrap REST APIs while adding these LLM-specific capabilities.

### 10.3 MCP vs. LangChain Tools

LangChain and similar frameworks define their own tool abstractions. These are framework-specific — a LangChain tool cannot be used directly by a non-LangChain application. MCP provides an interoperable standard that works across frameworks. LangChain has adopted MCP as an integration mechanism, allowing LangChain agents to consume MCP tools alongside native LangChain tools.

### 10.4 MCP vs. OpenAI's Agents SDK and Function Calling

OpenAI's approach to tool integration has been through native function calling and, more recently, the Agents SDK with built-in tool types (web search, file search, code interpreter). These are tightly integrated with OpenAI's platform but are not interoperable with other LLM providers. MCP's provider-agnostic design means the same MCP servers work with Claude, open-source models, and (with appropriate host implementations) any LLM that supports tool use.

## 11. Performance Considerations

### 11.1 Latency Budget

Each MCP tool call adds latency to the model's response: the JSON-RPC round-trip, the tool's execution time, and the serialization overhead. For stdio transport, the round-trip overhead is minimal (sub-millisecond). For remote Streamable HTTP servers, network latency dominates. Tool execution time varies enormously — a local file read completes in microseconds, while a database query or API call may take seconds.

### 11.2 Context Window Pressure

Every tool result consumes tokens in the model's context window. A database query that returns thousands of rows, or a file read that includes an entire large source file, can exhaust context capacity and degrade model performance. Well-designed MCP servers should return concise, relevant results — paginated where appropriate — and servers should document the expected result size in tool descriptions so the model can plan accordingly.

### 11.3 Parallel Tool Calls

Many LLM APIs support parallel tool calling, where the model requests multiple tool invocations in a single generation step. MCP hosts should execute these calls concurrently (across different servers or as concurrent requests to the same server) to minimize total latency. The MCP protocol itself is stateless at the request level, so concurrent calls are safe as long as the server implementation handles concurrent execution correctly.

## 12. Practical Deployment Patterns

### 12.1 Local Development Setup

For individual developers, the typical pattern is a set of stdio-based MCP servers configured in their IDE or Claude Desktop. These servers run locally, have access to the local file system and development databases, and require no network infrastructure. Configuration is a JSON file listing the servers and their launch commands.

### 12.2 Team and Enterprise Deployment

For teams, remote MCP servers deployed as microservices provide shared access to common tools. A team might run MCP servers for their issue tracker, documentation system, and staging database, secured with OAuth and accessible to all team members' AI assistants. This avoids every developer needing to configure and run the same servers locally.

### 12.3 Gateway and Proxy Patterns

An MCP gateway can aggregate multiple backend MCP servers behind a single endpoint, handling authentication, rate limiting, logging, and access control centrally. This simplifies client configuration (one connection instead of many) and provides a central point for security policy enforcement. Several open-source MCP gateway projects have emerged to support this pattern.

## 13. Limitations and Open Challenges

### 13.1 Statelessness vs. Statefulness

The Streamable HTTP transport supports both stateless and stateful server architectures, but the choice involves trade-offs. Stateless servers are easier to scale and deploy but cannot maintain conversation-scoped context. Stateful servers (using session IDs) can maintain state across requests but require session affinity in load-balanced deployments.

### 13.2 Error Handling and Reliability

MCP inherits JSON-RPC's error handling (error codes and messages), but the specification provides limited guidance on retry semantics, timeout behavior, and graceful degradation. In practice, different hosts handle server failures differently — some retry automatically, some surface errors to the model, and some present them to the user. Standardizing failure modes remains an area for protocol evolution.

### 13.3 Versioning and Compatibility

MCP uses a date-based versioning scheme (e.g., `2025-03-26`). The initialization handshake negotiates the protocol version, but the ecosystem's rapid evolution means servers and clients may support different feature sets. Capability negotiation mitigates this, but developers must still handle version mismatches gracefully.

### 13.4 Security Maturity

While MCP defines security principles and the OAuth 2.1 framework, the security model is still maturing. Tool poisoning, indirect prompt injection through tool results, and overly permissive server configurations remain practical risks. The community is actively developing best practices, auditing tools, and security-focused MCP proxies to address these challenges.

## 14. Future Directions

The MCP specification continues to evolve. Active areas of development include:

- **Richer media types**: Support for streaming audio, video, and interactive content in tool results, enabling multimodal tool interactions.
- **Standardized authentication profiles**: Pre-defined OAuth configurations for common identity providers, reducing the setup burden for enterprise deployments.
- **Formal verification of tool descriptions**: Mechanisms to cryptographically sign and verify tool descriptions, mitigating tool poisoning attacks.
- **Inter-server communication**: Allowing MCP servers to discover and invoke each other, enabling server-side composition without routing through the host.
- **Observability standards**: Standardized tracing and metrics for MCP interactions, integrating with OpenTelemetry and other observability frameworks.

## 15. Conclusion

The Model Context Protocol represents a fundamental shift in how LLM applications integrate with external systems. By defining a universal, open standard for tool discovery, invocation, and context provision, MCP eliminates the N×M integration problem that has plagued the LLM ecosystem. Its architecture — separating hosts, clients, and servers with well-defined primitives and transport layers — enables a composable ecosystem where tools built once work everywhere.

MCP's impact extends beyond mere convenience. It enables a new class of agentic workflows where models dynamically discover and compose tools to solve complex, multi-step problems. The security model, while still maturing, provides a principled framework for keeping humans in control of consequential actions. And the rapidly growing ecosystem of SDKs, pre-built servers, and host applications demonstrates that MCP has achieved the critical mass needed to become a lasting standard.

For practitioners building LLM-powered applications, MCP should be the default integration mechanism for external tools and data sources. For tool and service providers, exposing capabilities via an MCP server ensures compatibility with the broadest possible range of AI applications. As the protocol matures and the ecosystem grows, MCP is positioned to become the foundational infrastructure layer that connects the world's AI models to the world's data and services.

## References

- Anthropic. "Model Context Protocol Specification." modelcontextprotocol.io, 2024–2026.
- Anthropic. "Model Context Protocol SDKs." github.com/modelcontextprotocol, 2024–2026.
- MCP Working Group. "MCP Specification — 2025-03-26 Revision." modelcontextprotocol.io/specification, 2025.
- Anthropic. "Introducing the Model Context Protocol." anthropic.com/news, November 2024.
- JSON-RPC Working Group. "JSON-RPC 2.0 Specification." jsonrpc.org, 2010.
