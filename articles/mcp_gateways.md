# MCP Gateways: Centralized Proxy Architecture for the Model Context Protocol Ecosystem

*April 2026 • Technical Report*

## 1. Introduction

The Model Context Protocol (MCP), introduced by Anthropic in late 2024 and rapidly adopted throughout 2025-2026, solved the N-by-M integration problem between LLM applications and external tools by defining a universal client-server protocol. But MCP's initial design assumed a direct, point-to-point topology: each MCP client connects to each MCP server independently, managing its own authentication, configuration, and lifecycle. This works well for a developer connecting Claude Desktop to three or four local MCP servers. It does not work well when an enterprise has fifty agents accessing two hundred tool servers, each requiring distinct credentials, rate limits, audit logging, and access control policies.

MCP Gateways have emerged as the architectural answer to this scaling problem. An MCP Gateway is a centralized proxy and management layer that sits between MCP clients (LLM hosts and AI agents) and MCP servers (tool providers), analogous to how an API gateway sits between frontend applications and backend microservices. The gateway intercepts all MCP JSON-RPC traffic, applies security policies, aggregates tool catalogs from multiple servers into a unified surface, manages credentials, enforces rate limits, and provides centralized observability — all without requiring changes to individual MCP servers or clients.

By mid-2026, the MCP Gateway has become a distinct infrastructure category with over forty implementations ranging from open-source projects by Microsoft, IBM, and Docker to commercial platforms by Kong, Traefik, Cloudflare, and numerous startups. This report provides a comprehensive technical examination of MCP Gateways: their architecture, protocol-level mechanics, security models, observability capabilities, and the rapidly evolving implementation landscape.

## 2. Why MCP Gateways Are Needed

### 2.1 The Management Problem at Scale

In a direct-connection MCP topology, each AI agent or host application must independently manage connections to every MCP server it needs. For a single developer with a local setup — perhaps a GitHub server, a database server, and a filesystem server — this is manageable. Configuration lives in a JSON file, servers run as child processes, and credentials are environment variables.

At enterprise scale, this approach collapses. An organization with twenty AI agents accessing thirty MCP servers faces six hundred distinct connections, each requiring separate authentication configuration, credential distribution, version tracking, and monitoring setup. Adding a new MCP server means updating the configuration of every agent that needs access. Rotating a credential means touching every client that holds it. There is no single place to answer the question "which agents are calling which tools, how often, and with what error rates?"

### 2.2 The Security Gap

The MCP specification, particularly before the OAuth 2.1 additions in mid-2025, left many security concerns to the implementation. Authentication, authorization, audit logging, rate limiting, credential management, and prompt injection defense were not prescribed by the protocol. In a direct-connection model, every MCP server must independently implement these controls — or, more commonly, they are simply omitted. A gateway provides a single enforcement point where security policies can be applied consistently across all servers, regardless of how security-aware any individual server implementation might be.

### 2.3 The Observability Gap

When tool calls flow directly between agents and servers, observability is fragmented. Each server may log differently (or not at all), metrics are scattered across processes, and there is no unified trace connecting a user's request through the LLM's reasoning to a sequence of tool calls across multiple servers. Compliance requirements — notably the EU AI Act's high-risk system provisions taking effect in August 2026, which mandate comprehensive logging and traceability for every AI system interaction — demand centralized audit trails that a direct-connection topology cannot easily provide.

### 2.4 The Context Window Problem

As organizations connect more MCP servers, the number of tools exposed to the LLM grows. A naive aggregation of tools from all connected servers can present hundreds of tool definitions to the model, consuming a significant portion of the context window (estimates suggest 30-50% of context tokens can be wasted on tool definitions alone) and degrading the model's ability to select the right tool. Gateways can curate and filter tool catalogs, presenting only relevant tools based on user identity, task context, or progressive disclosure strategies.

## 3. Architecture Patterns

### 3.1 Centralized Reverse Proxy

The most common architecture pattern positions the gateway as a centralized reverse proxy — a single service (or clustered service) that all MCP clients connect to instead of connecting to individual servers. The gateway maintains connections to all upstream MCP servers and routes requests to the appropriate server based on the tool being invoked.

In this model, clients see a single MCP endpoint that presents an aggregated catalog of tools from all connected servers. When a client calls `tools/list`, the gateway queries all upstream servers (or returns cached results), merges the responses, applies access control filters, and returns a unified list. When a client calls `tools/call` for a specific tool, the gateway identifies which upstream server owns that tool and forwards the request.

This is the pattern implemented by Microsoft MCP Gateway, Docker MCP Gateway, IBM ContextForge, Traefik Hub, and the majority of commercial offerings.

### 3.2 Sidecar Proxy

In a sidecar pattern, each MCP server is paired with a lightweight gateway proxy that handles security, observability, and policy enforcement for that individual server. This pattern is common in Kubernetes environments where sidecars are a natural deployment unit. The Kuadrant MCP Gateway (backed by Red Hat) uses this approach, deploying Envoy-based sidecars alongside MCP servers with AuthPolicy resources controlling authentication and authorization.

The sidecar pattern provides isolation — a misconfigured or compromised proxy only affects one server — but sacrifices the unified tool aggregation that centralized gateways provide. It is often combined with a lightweight central registry for tool discovery.

### 3.3 Mesh Architecture

The mesh pattern extends the sidecar concept with inter-gateway communication, analogous to a service mesh. Each MCP server has an associated proxy, and these proxies communicate to enable cross-server tool discovery, distributed tracing, and coordinated policy enforcement. The agentgateway project (a Linux Foundation project written in Rust) implements elements of this pattern, supporting both MCP and the Agent-to-Agent (A2A) protocol for agent-to-agent communication alongside agent-to-tool interactions.

### 3.4 Micro-MCP Architecture

The Micro-MCP pattern, inspired by microservices architecture, decomposes large MCP servers into many single-purpose servers (each exposing one or a small number of tools), composed behind a lightweight gateway. Rather than a monolithic MCP server that exposes database queries, file operations, and API calls, the Micro-MCP approach deploys separate servers for each capability, with the gateway providing unified access, security isolation, and independent deployability. This pattern enables granular scaling — a heavily-used tool's server can be scaled independently — and limits the blast radius of server failures or compromises.

### 3.5 Dual-Plane Architecture

Microsoft's MCP Gateway implements a dual-plane architecture separating concerns into a data plane and a control plane. The data plane handles runtime request routing with session affinity, proxying JSON-RPC messages to the correct upstream server instance. The control plane manages the lifecycle of MCP server deployments through RESTful management APIs — deploying, updating, and deleting server instances. This separation mirrors the control plane/data plane split in Kubernetes networking and allows independent scaling and evolution of routing logic versus management logic. The deployment uses Kubernetes-native StatefulSets and headless services for session-aware stateful routing.

## 4. Protocol-Level Mechanics

### 4.1 JSON-RPC Message Interception

At the protocol level, an MCP Gateway operates as a JSON-RPC-aware proxy. Every message between client and server is a JSON-RPC 2.0 object — requests (with `id`, `method`, and `params`), responses (with `id` and `result` or `error`), and notifications (with `method` but no `id`). The gateway parses each message to understand its intent and apply policies.

When a client sends a request such as:

```json
{
  "jsonrpc": "2.0",
  "id": 1,
  "method": "tools/call",
  "params": {
    "name": "query_database",
    "arguments": {
      "sql": "SELECT * FROM users WHERE id = 42"
    }
  }
}
```

The gateway can inspect the method (`tools/call`), the tool name (`query_database`), and the arguments (`sql` parameter) to make routing, authorization, and logging decisions. Traefik Hub's MCP Gateway implementation provides a concrete example of this with a policy expression language that exposes the entire JSON-RPC body under an `mcp` namespace — `mcp.method`, `mcp.params.name`, `mcp.params.arguments.sql` — alongside JWT claims under a `jwt` namespace, enabling fine-grained policy expressions such as:

```yaml
policies:
  - match: Equals(`mcp.method`, `tools/call`) &&
           Equals(`mcp.params.name`, `get_weather`) &&
           Contains(`jwt.groups`, `weather-users`)
    action: allow
  - match: Equals(`mcp.method`, `tools/call`) &&
           Prefix(`mcp.params.name`, `admin_`) &&
           !Contains(`jwt.groups`, `admin`)
    action: deny
defaultAction: deny
```

The gateway automatically permits `initialize` and `notifications/initialized` methods, which are essential for the MCP protocol handshake and cannot be meaningfully policy-controlled.

### 4.2 List Response Filtering

Beyond request-level policy enforcement, gateways intercept list responses (`tools/list`, `resources/list`, `prompts/list`) from upstream servers and filter them based on the requesting client's identity. This is capability-aware access control: a user in the "developers" group sees developer tools, while a user in the "analysts" group sees analytics tools, even though both connect to the same gateway endpoint backed by the same upstream servers.

Traefik Hub implements this through separate `listPolicies` evaluated at discovery time:

```yaml
listPolicies:
  - match: Prefix(`mcp.params.name`, `dev_`) &&
           Contains(`jwt.groups`, `developers`)
    action: show
listDefaultAction: hide
```

This ensures the LLM's context window is only populated with tools the user is authorized to use, simultaneously improving security and reducing context bloat.

### 4.3 Transport Bridging

MCP defines three transport mechanisms, and gateways must bridge between them:

**stdio** is used for local MCP servers running as child processes. The client writes JSON-RPC messages to the server's standard input and reads responses from standard output. This is the simplest transport with zero network overhead, but it is inherently local — the server must run on the same machine as the client.

**HTTP+SSE** (deprecated as of the March 2025 specification but still widely deployed) uses HTTP POST requests for client-to-server messages and Server-Sent Events (SSE) for server-to-client streaming. This was the original remote transport but required two separate connections (one for POST, one for SSE) and had limitations around scalability and infrastructure compatibility.

**Streamable HTTP** (specified in June 2025 and now the standard remote transport) unifies communication over standard HTTP. Clients send JSON-RPC messages via HTTP POST requests to the MCP endpoint. Servers may respond with a single JSON-RPC result, or they may open an SSE stream on the response to deliver multiple messages (useful for progress notifications during long-running tool calls). Clients can also issue HTTP GET requests to establish an SSE stream for server-initiated notifications. Sessions are managed through `Mcp-Session-Id` headers assigned during initialization.

A critical gateway function is **transport translation**: converting stdio-based local servers into remotely accessible Streamable HTTP endpoints. The gateway spawns the stdio server as a child process, wraps its stdin/stdout with JSON-RPC message framing, and exposes the server over Streamable HTTP to remote clients. This allows organizations to make any existing stdio-based MCP server available to remote agents without modifying the server code. Docker MCP Gateway, for example, launches MCP servers as Docker containers, communicates with them via stdio internally, and exposes them through its gateway endpoint.

Envoy AI Gateway takes a particularly sophisticated approach to transport handling, implementing the full MCP Streamable HTTP transport specification with session management, SSE reconnection logic (using `Last-Event-ID` for resumability), and multiplexed streams. The MCP proxy is implemented as a lightweight Go server that bridges between Envoy's HTTP networking stack and the stateful JSON-RPC protocol.

### 4.4 Session Management

MCP is a stateful protocol. Unlike REST APIs where each request is independent, MCP sessions maintain context across multiple interactions. The initialization handshake establishes capabilities, and subsequent tool calls may depend on session state. This creates a routing challenge for gateways: all requests within a session must be directed to the same upstream server instance.

Microsoft's MCP Gateway solves this with session-aware stateful routing — all requests carrying a given `Mcp-Session-Id` are consistently routed to the same server instance, implemented through Kubernetes StatefulSets and headless services. Traefik Hub uses the Highest Random Weight (HRW) consistent hashing algorithm, which provides deterministic client-to-server routing based on client IP without requiring sticky cookies, with automatic failover when servers become unavailable.

The gateway must also manage session lifecycle — detecting when sessions expire or servers restart, cleaning up session state, and handling reconnection. For SSE-based transports, gateways handle reconnection semantics including `Last-Event-ID` propagation to ensure message continuity.

## 5. Tool Registry, Discovery, and Aggregation

### 5.1 Dynamic Tool Registration

Gateways maintain a live registry of available tools by periodically querying upstream servers via `tools/list` (and `resources/list`, `prompts/list`). When an MCP server starts, registers with the gateway, or updates its capabilities (signaled via a `notifications/tools/list_changed` notification), the gateway refreshes its catalog. This dynamic registry eliminates hardcoded routing logic and enables zero-downtime server updates.

Several gateways implement cached aggregation, refreshing the tool catalog on a timer (Stacklok's Virtual MCP Server, for example, uses a 60-second cache for list responses) while allowing immediate refresh on change notifications. The Gravitee MCP Gateway documentation recommends caching `tools/list`, `resources/list`, and `prompts/list` responses with TTL-based or notification-based invalidation, but explicitly warns against caching `tools/call` responses since tool invocations are side-effecting operations.

### 5.2 Namespace Management and Conflict Resolution

When aggregating tools from multiple upstream servers, name conflicts are inevitable — two servers might both expose a `create_issue` tool. Gateways resolve this through namespace prefixing, automatically converting conflicting names into disambiguated versions: `github_create_issue` and `jira_create_issue`. Stacklok's Virtual MCP Server allows administrators to customize these prefixed names according to organizational preferences.

Microsoft's MCP Gateway implements a Tool Gateway Router — an MCP server that acts as an intelligent router, directing tool execution requests to the appropriate registered tool server based on tool definitions. This indirection layer allows tool aliasing, version pinning, and policy-controlled routing.

### 5.3 Context Window Optimization

With dozens of MCP servers aggregated behind a gateway, the naive approach of presenting all tools to the LLM quickly exhausts the context window. Several strategies have emerged:

**Selective exposure**: Virtual MCP Servers (like Stacklok's implementation) allow administrators to cherry-pick exactly which tools from each upstream server to expose, eliminating irrelevant tools from the LLM's context. An engineer might receive access to GitHub's `search_code` restricted to a specific repository, particular Slack channels, and internal documentation — but not the full catalog of every connected server.

**Progressive disclosure**: The AIRIS MCP Gateway aggregates over 60 tools but claims to reduce context tokens by 97% through progressive disclosure — initially presenting only high-level tool categories and revealing specific tools on demand as the conversation narrows.

**Meta-tool abstraction**: Some gateways replace direct tool exposure with meta-tools (a "Code Mode" approach where the LLM writes code to orchestrate tools in a sandbox), reducing token usage by 50% or more compared to enumerating every tool definition.

**Profile-based filtering**: Docker MCP Gateway uses profiles to control which servers and tools are available to a given client session, activated via a `--profile` flag at connection time.

## 6. Authentication and Authorization

### 6.1 The Multi-Hop Authorization Challenge

MCP transforms traditional OAuth from a straightforward user-app-API flow into a multi-hop architecture: user to AI host to MCP client to (potentially many) MCP servers to downstream services. Each hop introduces authorization questions. Does the user have permission to use this tool? Does the agent have permission to call this server? Does the server have permission to access this downstream API on behalf of this user? A gateway provides a single enforcement point for these layered authorization decisions.

### 6.2 OAuth 2.1 and the MCP Authorization Specification

The MCP specification defines MCP servers as OAuth 2.1 resource servers and MCP clients as OAuth 2.1 clients. Servers may support delegated authorization through third-party authorization servers. The specification requires clients to implement Resource Indicators (RFC 8707) to explicitly identify the target MCP server for which a token is being requested, preventing token misuse across servers.

Gateways centralize this OAuth implementation. Rather than each MCP server independently implementing OAuth token validation, the gateway validates tokens once at the edge and either forwards authenticated requests or performs token exchange (RFC 8693) to issue narrowly-scoped downstream tokens.

### 6.3 Token Exchange and Credential Scoping

Red Hat's Kuadrant MCP Gateway implements a sophisticated three-layer authorization architecture:

**Layer 1 — Identity-based tool filtering**: An external authorization component (Authorino) validates the user's OAuth2 token, extracts permissions from Keycloak client roles, and creates a cryptographically signed JWT "wristband" containing the user's allowed tools. The MCP broker validates this wristband's signature and filters `tools/list` responses accordingly.

**Layer 2 — OAuth2 Token Exchange (RFC 8693)**: When routing a tool call to an upstream server, the gateway exchanges the client's broad access token for a narrowly-scoped token with an audience claim limited to the specific target server's hostname and reduced scopes. This prevents privilege escalation — a compromised token for one MCP server cannot be used against another.

**Layer 3 — Vault Integration**: For non-OAuth2 servers that require Personal Access Tokens or API keys, the gateway fetches credentials from HashiCorp Vault using the pattern `/v1/secret/data/{username}/{mcp-server}`, implementing a priority-based fallback: Vault credentials first, then OAuth2 token exchange.

### 6.4 Credential Vaulting

A fundamental security principle of MCP Gateways is that raw credentials never reach the client or the LLM. Peta (from Dunia Labs) positions itself as "1Password for AI Agents" — its server-side encrypted vault stores all API keys and tokens, and agents receive only scoped, time-limited gateway tokens. The gateway injects real credentials into upstream requests at execution time, so neither the MCP client nor the LLM conversation history ever contains sensitive credential material.

This is critical because tokens appearing in prompts, conversation history, debug traces, or analytics could be exploited through prompt injection attacks or model context leakage. Docker MCP Gateway similarly centralizes credentials, injecting them into containerized MCP server processes at runtime.

### 6.5 Sequence-Aware Authorization

Traditional OAuth validates individual requests in isolation, but autonomous agents can chain legitimate tool calls in unauthorized combinations. An agent with permission to read files and permission to send emails could, through a prompt injection attack, be induced to read sensitive files and email them to an attacker — each individual action appears authorized, but the sequence is not.

Gateways address this through sequence-aware authorization, evaluating not just whether an individual action is permitted but whether the accumulated sequence of operations exceeds the intended scope. The GitGuardian analysis of MCP OAuth patterns identifies this as a fundamental gap that gateway-based enforcement can address, considering user behavior, device trust, and operation sequences alongside token validity.

### 6.6 Zero Trust Architecture

Peta implements a Zero Trust MCP Gateway that validates identity, policy, and human-approval rules on every MCP request. Its architecture includes three components: Peta Core (the vault and gateway), Peta Console (the policy and audit plane), and Peta Desk (the human-in-the-loop approval interface). When an agent attempts a high-risk action — such as deleting a production database record — Peta Desk presents the request to a human reviewer who can approve, deny, or modify it, without exposing raw credentials. Kong's MCP Gateway documentation similarly describes zero-trust implementation with device posture checks, network location evaluation, and periodic trust expiration requiring renewal.

## 7. Security: Threats and Gateway-Level Defenses

### 7.1 Prompt Injection at the Gateway

Prompt injection — malicious instructions embedded in tool outputs or user inputs that hijack the LLM's behavior — is MCP's most discussed security threat. A gateway provides a natural interception point for prompt injection defense. Lasso MCP Gateway implements a plugin-based guardrail system where security plugins process requests and responses before they reach the client:

- A **Presidio plugin** detects PII (names, emails, phone numbers, SSNs) in tool responses and masks or blocks them before they reach the LLM.
- A **prompt injection detection plugin** (via Lasso's API) analyzes incoming prompts and tool outputs for injection patterns.
- A **basic plugin** provides 12 built-in patterns for token and secret masking, protecting API keys and credentials that might appear in tool responses.

By sanitizing tool outputs at the gateway layer, prompt injection attacks that operate through compromised tool responses can be neutralized before they reach the model.

### 7.2 Tool Poisoning

Tool poisoning occurs when an MCP server's tool metadata or behavior is manipulated to compromise AI agents. Unlike prompt injection targeting individual sessions, tool poisoning can affect all users relying on the compromised tool. The Solo.io analysis identifies three variants: direct tampering (modifying a tool's description to include hidden instructions), shadowing (a malicious tool mimicking a legitimate one's name), and rug pulls (a tool behaving normally during evaluation but activating malicious behavior later).

Gateways defend against tool poisoning through tool registration and approval workflows — new MCP servers must be vetted before being added to the gateway's upstream pool — and by enforcing digitally signed, version-locked tool definitions. Open Edison specifically focuses on data exfiltration prevention and execution controls at the gateway level.

### 7.3 The Triple-Gate Security Pattern

A defense-in-depth approach structures security into three distinct gates:

- **Gate 1** protects the AI client to LLM communication path: prompt injection filtering, PII detection, and input sanitization.
- **Gate 2** protects the LLM to MCP server communication path: tool authorization, parameter validation, and argument sanitization.
- **Gate 3** protects the MCP server to external API communication path: rate limiting, authentication enforcement, and response filtering.

A gateway naturally sits at Gate 2 but can extend its enforcement across all three gates, especially when the LLM host itself connects through the gateway.

### 7.4 Multi-Tenant Blast Radius

For gateways serving multiple tenants, the blast radius of a security breach must be bounded. This requires per-tenant credential isolation (each client or agent operates with scoped credentials), per-tenant session isolation (sessions cannot access another tenant's tool state), and audit trails logging every tool call with caller identity, parameters, and outcome. IBM ContextForge implements multi-tenant support with email-based user accounts, team/project segregation, role-based access control, and per-tenant visibility rules.

## 8. Observability

### 8.1 Centralized Logging

A gateway provides a single point for logging all MCP interactions across every connected server. Agentgateway logs structured fields to stdout including `mcp.method` (the MCP method being invoked), `mcp.session.id` (the session identifier), `mcp.target` (the target MCP server), `mcp.resource.type` (tool, prompt, or resource), and `mcp.resource.name` (the specific resource being accessed). These structured fields integrate with CEL (Common Expression Language) expressions for custom access logging policies.

Lasso MCP Gateway produces structured JSON audit logs with integrations for ELK (Elasticsearch, Logstash, Kibana), Prometheus, and Grafana. Every tool call, prompt execution, and resource read is logged with sufficient detail for compliance audits.

### 8.2 Distributed Tracing with OpenTelemetry

OpenTelemetry (OTel) has become the standard for MCP observability. A proposal to add native OpenTelemetry trace support to the MCP specification is under discussion, and several gateways already implement it.

The trace structure is hierarchical: a parent span representing the agent's overall request spawns child spans for each MCP tool call, which in turn spawn spans for downstream API calls made by the tool server. OpenTelemetry's W3C Trace Context propagation automatically injects trace headers into requests, allowing tool servers to continue the trace — enabling end-to-end visibility from the user's prompt through the LLM's reasoning to every tool call and downstream interaction.

Key span attributes include `tool.name`, `tool.input_size`, `tool.output_size`, HTTP status codes, and error classifications. Traefik Hub exposes OTel-compatible metrics including `mcp.client.operation.duration` histograms tagged by session ID, method name, tool name, and resource URI.

Agentgateway supports OTLP (OpenTelemetry Protocol) trace export to collectors on `localhost:4317`, with Jaeger integration for trace storage and visualization. The Nexus gateway provides complete OpenTelemetry observability across metrics, traces, and logs.

### 8.3 Metrics

Standard gateway metrics include:

- **Tool invocation count**: How often each tool is called, broken down by server and tool name. Agentgateway tracks this via the `list_calls_total` Prometheus metric.
- **Tool invocation duration**: p50, p95, and p99 latency histograms per tool, revealing which tools are slow and which are fast.
- **Error rates**: Counters with error-type labels categorizing failure modes per tool.
- **Token usage**: Input and output token counts per tool call, essential for cost tracking and context window management.
- **Session metrics**: Active session counts, session duration distributions, and session error rates.

### 8.4 Regulatory Compliance

The EU AI Act's high-risk system provisions (effective August 2026) mandate comprehensive logging and traceability for AI system interactions. MCP Gateways provide the centralized audit infrastructure needed to demonstrate compliance — every tool call is logged with timestamp, user identity, agent identity, tool name, parameters, response, and outcome. DreamFactory and similar observability-focused tools emphasize tamper-proof audit trails integrated with SIEM platforms for regulatory reporting.

## 9. Rate Limiting and Performance

### 9.1 Multi-Dimensional Rate Limiting

Gateways implement rate limiting across multiple dimensions simultaneously:

- **Per-user limits**: Preventing individual users from monopolizing shared tool infrastructure.
- **Per-tool limits**: Protecting downstream APIs that have their own rate limits (e.g., GitHub API's rate limits).
- **Per-team limits**: Ensuring fair resource allocation across organizational units.
- **Per-agent limits**: Constraining autonomous agents that might otherwise generate unlimited tool calls.

Advanced gateways implement these limits with in-memory enforcement achieving sub-millisecond overhead. Zapier MCP Gateway, for example, enforces rate limits of 80 calls per hour, 160 per day, and 300 per month, reflecting the external API constraints of the services it connects to.

### 9.2 Caching

Gateways employ layered caching to reduce load on upstream servers:

- **In-memory cache**: Fast, per-instance caching for tool lists and resource metadata.
- **Distributed cache**: Redis or Memcached for multi-instance gateway deployments.
- **TTL-based expiration**: Tool and resource lists cached with time-to-live values.
- **Notification-based invalidation**: Servers notify gateways of changes, triggering cache refresh.

The Gravitee analysis explicitly warns that `tools/call` responses must never be cached since tool invocations have side effects. Safe cache candidates are `tools/list`, `resources/list`, `resources/read` (for static data), and `prompts/list`.

### 9.3 Load Balancing and Resilience

When multiple instances of an MCP server are deployed, gateways distribute traffic using round-robin, least-connections, or consistent-hashing algorithms. Circuit breakers detect unhealthy upstream servers and temporarily route traffic away from them. Retry logic handles transient failures, with careful restrictions on retrying side-effecting operations. Kong's MCP Gateway implements health checks, circuit breakers, and session-affinity-aware load balancing.

## 10. Multi-Tenant Enterprise Deployments

### 10.1 Tenant Isolation

Enterprise gateways support multiple teams, departments, or customers sharing the same gateway infrastructure while maintaining strict isolation. IBM ContextForge implements multi-tenant support with team/project segregation, role-based access control, and per-tenant visibility rules — a team can only discover and invoke tools they have been granted access to.

Scalekit provides a "secure tool-calling layer" with user-consented delegation and per-tenant token management, ensuring that one tenant's credentials and session state are never accessible to another tenant.

### 10.2 Virtual Servers and Profiles

Stacklok's Virtual MCP Server uses Kubernetes Custom Resource Definitions (MCPGroup, MCPServer, VirtualMCPServer) to define tenant-specific views of the tool catalog. A VirtualMCPServer combines selected tools from multiple upstream MCPServers with access control policies, pre-configured defaults, and namespace prefixes, creating a curated tool surface tailored to a specific team or use case.

Docker MCP Gateway uses profiles — a simpler mechanism where an operator specifies which profile to activate via a `--profile` flag, controlling which servers and tools a given client session can access.

### 10.3 Per-Tenant Observability

Structured logging with tenant identifiers enables per-tenant dashboards, cost allocation, and compliance reporting. Gateways tag every metric and trace with the originating tenant, allowing infrastructure teams to provide each team with visibility into their own tool usage while maintaining a global operational view.

## 11. Comparison to Traditional API Gateways

### 11.1 What Is Different

MCP Gateways and API gateways share a common ancestry — both are reverse proxies that provide authentication, rate limiting, routing, and observability. But several characteristics of the MCP protocol require purpose-built infrastructure that traditional API gateways cannot provide out of the box:

**Stateful sessions**: API gateways assume stateless HTTP request-response pairs. MCP sessions are persistent and bidirectional, with context accumulated across multiple tool calls. The gateway must maintain session affinity and manage session lifecycle, which standard HTTP load balancers are not designed for.

**Tool discovery semantics**: API gateways route requests to known endpoints. MCP gateways must understand tool discovery (`tools/list`), aggregate tool catalogs from multiple servers, resolve naming conflicts, filter results based on authorization, and manage the tool registry dynamically. This level of protocol-aware aggregation is beyond what traditional API gateways were designed to handle.

**Bidirectional communication**: MCP supports server-initiated messages (notifications, sampling requests) flowing from server to client. This requires the gateway to maintain open connections for server push, manage SSE streams, and route server-initiated messages back to the correct client session.

**JSON-RPC awareness**: An API gateway proxies HTTP requests. An MCP gateway must parse JSON-RPC message bodies to understand the method being invoked, the tool being called, and the arguments being passed, in order to make routing and policy decisions. This deep protocol awareness goes beyond URL-based routing.

**Context-window awareness**: MCP gateways may need to consider token counts, compress tool outputs, or filter tool definitions based on the model's context window constraints — concerns that have no equivalent in traditional API gateway operations.

### 11.2 Can You Adapt an Existing API Gateway?

Several vendors have taken the approach of extending existing API gateway platforms with MCP support rather than building from scratch. Kong added MCP support in Gateway 3.12 with the AI MCP Proxy plugin, which translates between MCP and HTTP, allowing MCP clients to call existing REST APIs through Kong without rewriting them as MCP servers. Apache APISIX offers MCP server proxying through its plugin system. Traefik Hub added a dedicated MCP Gateway middleware.

The API7.ai analysis argues that these are pragmatic choices: "Start with a proven gateway and configure the behavior you need." However, the Solo.io analysis contends that "bolting MCP support onto a standard API gateway does not work well; the protocol needs infrastructure that understands its semantics." The truth lies somewhere between — existing gateways can handle basic MCP proxying, but deep features like session management, tool aggregation, and context-window optimization require MCP-specific logic that amounts to significant new code regardless of the starting point.

## 12. Implementation Landscape

### 12.1 Open-Source Implementations

The open-source MCP Gateway ecosystem has grown rapidly. Major implementations include:

**Microsoft MCP Gateway**: A reverse proxy and management layer for Kubernetes environments with dual-plane architecture (data plane for routing, control plane for lifecycle management), session-aware stateful routing via StatefulSets, and a Tool Gateway Router for intelligent tool-level routing.

**Docker MCP Gateway**: Part of Docker Desktop's MCP Toolkit, it orchestrates MCP servers in isolated Docker containers. Servers launch on demand, credentials are injected at runtime, and security restrictions (limited privileges, restricted network access, constrained resources) are enforced by the container runtime.

**IBM ContextForge**: An Apache 2.0-licensed gateway and registry (3,500+ GitHub stars) that federates MCP servers, A2A protocol agents, and REST/gRPC APIs behind a single endpoint. It provides multi-tenant support, centralized governance, and forward compatibility with agent-to-agent architectures.

**agentgateway** (Linux Foundation): A high-performance data plane written in Rust that supports MCP, A2A, and LLM routing through a unified OpenAI-compatible API. Features include JWT authentication, RBAC, structured logging, OpenTelemetry tracing, and a modular architecture where each component operates independently.

**Lasso MCP Gateway**: An open-source security-focused gateway with a plugin-based guardrail system. Plugins include Presidio for PII detection, prompt injection detection, and token/secret masking. Structured JSON audit logging integrates with ELK, Prometheus, and Grafana.

**Envoy AI Gateway**: Extends the Envoy proxy with a lightweight Go-based MCP proxy that handles session management, stream multiplexing, and tool aggregation, leveraging Envoy's existing networking stack for connection management, load balancing, and circuit breaking.

**Stacklok Virtual MCP Server**: A Kubernetes-native gateway using CRDs (MCPGroup, MCPServer, VirtualMCPServer) with features including tool cherry-picking, namespace conflict resolution, OAuth 2.1 authentication, and per-tool scope-based access control.

**Kuadrant MCP Gateway**: An Envoy-based gateway with Istio integration, featuring three-layer authorization (identity-based tool filtering, OAuth2 token exchange, Vault credential injection) through Kuadrant AuthPolicy resources.

**Pomerium**: An open-source gateway focused on securing MCP server access with authentication and per-tool access control policies.

**Gate22**: A control plane for governing which tools agents can use, with RBAC, audit logging, and policy management.

**MCP Mesh**: Features RBAC, an encrypted token vault, and full OpenTelemetry observability.

**Open Edison**: Focuses on data exfiltration prevention and execution controls.

**Unla**: A lightweight gateway that transforms existing HTTP services into MCP servers with zero code changes.

### 12.2 Commercial Platforms

The commercial landscape spans established infrastructure vendors and MCP-native startups:

**Kong AI Gateway** (Gateway 3.12+): The AI MCP Proxy plugin translates between MCP and HTTP, allowing MCP clients to invoke existing REST APIs as if they were MCP tools. This is valuable for enterprises with large existing API surface areas that want MCP compatibility without rewriting.

**Traefik Hub**: Provides an MCP Gateway middleware with JSON-RPC-level policy enforcement, expression-based access control, list filtering, HRW load balancing, OAuth 2.1 Resource Server support, and OpenTelemetry metrics.

**Cloudflare**: Offers MCP server portals through Cloudflare One, routing MCP portal traffic through Cloudflare Gateway for HTTP logging and DLP (Data Loss Prevention) scanning. Gateway HTTP policies with DLP profiles detect and block sensitive data sent to upstream MCP servers.

**Zapier MCP Gateway**: Converts 8,000+ app integrations into MCP-compatible endpoints, exposing 30,000+ actions with built-in authentication and rate limiting.

**Peta** (Dunia Labs): Enterprise MCP infrastructure with an encrypted vault, zero-trust gateway, granular access policies, and human-in-the-loop approvals via Peta Desk.

**MintMCP**: SOC 2 Type II certified gateway for healthcare and financial services with comprehensive audit trails, RBAC, and real-time monitoring.

**Webrix**: Enterprise infrastructure with SSO, RBAC, audit trails, and governance controls.

**Zuplo**: AI gateway with hierarchical cost controls and semantic caching.

**Unified Context Layer (UCL)**: A multi-tenant server connecting over 1,000 SaaS tools via a standardized command endpoint.

**Smithery, Composio (Rube), ToolRouter, TurboMCP, Runlayer, Golf, and others**: The long tail of commercial MCP gateway offerings, each targeting different niches from developer experience to enterprise compliance.

**AWS Amazon Bedrock AgentCore**: AWS's managed gateway for uniting MCP servers within the Bedrock agent ecosystem.

### 12.3 Convergence Trend

A clear convergence trend is visible: traditional API gateway vendors (Kong, Traefik, Envoy/Solo.io, APISIX) are adding MCP support to their existing platforms, while MCP-native projects (agentgateway, ContextForge, Lasso) are building purpose-built gateways from scratch. The two approaches will likely meet in the middle as the MCP specification matures and gateway patterns standardize.

## 13. The MCP 2026 Roadmap and Gateway Implications

### 13.1 Enterprise-Managed Auth (Q2 2026)

The MCP specification roadmap targets moving from static client secrets to full OAuth 2.1 with PKCE and SAML/OIDC integration by mid-2026. This will reduce the burden on gateways to compensate for lacking server-side auth but will also require gateways to implement more sophisticated token management and delegation flows.

### 13.2 Gateway and Proxy Patterns (2026)

The specification explicitly plans to define protocol-level behavior for intermediaries, covering authorization propagation and session semantics. This will standardize how gateways intercept, forward, and modify MCP messages, reducing the current fragmentation where each gateway implements its own proxy semantics.

### 13.3 MCP Registry (Q4 2026)

A curated, verified server directory with security audits and SLA commitments is planned for late 2026. Gateways will integrate with this registry for trusted server discovery, reducing the risk of tool poisoning from unvetted servers.

### 13.4 Agent-to-Agent Coordination (Q3 2026)

The A2A protocol support planned for Q3 2026 will enable hierarchical multi-agent architectures where orchestrator agents delegate to specialized sub-agents. Gateways like agentgateway and IBM ContextForge already support both MCP and A2A, positioning them as unified agentic infrastructure rather than tool-only proxies.

## 14. Open Challenges

### 14.1 Performance Overhead

Every request through a gateway adds latency. While in-memory policy evaluation can achieve sub-millisecond overhead for simple checks, more complex operations — token exchange, Vault lookups, PII scanning, prompt injection detection — can add meaningful latency to each tool call. For latency-sensitive agentic workflows where the LLM makes dozens of sequential tool calls, gateway overhead compounds. The challenge is maintaining comprehensive security and observability without degrading the interactive agent experience.

### 14.2 Specification Gaps

As of early 2026, the MCP specification does not formally define gateway behavior. Each gateway implements its own session forwarding, tool aggregation, and authorization propagation semantics. This fragmentation means clients may behave differently behind different gateways, and migrating between gateways is nontrivial. The planned specification work on gateway and proxy patterns should address this, but until then, gateway interoperability remains a challenge.

### 14.3 Context Window Management

The fundamental tension between comprehensive tool availability and context window efficiency remains unresolved. Progressive disclosure and meta-tool approaches reduce token usage but add complexity to the agent's decision-making. There is no standardized protocol mechanism for a gateway to communicate context budget constraints to clients or for clients to request tool subsets based on task context.

### 14.4 Testing and Debugging

Adding a gateway between client and server complicates debugging. When a tool call fails, the failure might originate in the client, the gateway's policy engine, the transport bridge, the upstream server, or a downstream API. Distributed tracing helps, but gateway-specific debugging tools are still immature. When Traefik Hub's gateway denies a request, it returns HTTP 403 with a plain text "Forbidden" body — MCP clients like Claude Desktop interpret this as a broken session requiring a restart, which is not a helpful developer experience.

### 14.5 Standardized Configuration

Each gateway uses its own configuration format — Kubernetes CRDs, YAML policy files, JSON configs, web-based consoles. There is no standard way to define "this user group can access these tools with these rate limits" that is portable across gateway implementations. The MCP roadmap mentions configuration portability as a goal, but concrete proposals are still emerging.

## 15. Conclusion

MCP Gateways represent the natural maturation of the MCP ecosystem from developer-scale experimentation to enterprise-scale production. Just as the proliferation of microservices necessitated API gateways, the proliferation of MCP servers necessitates MCP Gateways — centralized infrastructure that provides security, observability, governance, and operational efficiency for agent-to-tool communication.

The architecture is still young and rapidly evolving. Over forty implementations compete across different philosophies: purpose-built versus extended API gateways, security-first versus developer-experience-first, open-source versus commercial. The MCP specification itself is catching up, with planned standardization of gateway and proxy patterns, enterprise authentication, and configuration portability through 2026.

For organizations deploying MCP in production today, the practical guidance is clear: implement a gateway early, centralize credentials and audit logging from the start, architect for multi-tenancy even if you start with a single team, and choose a gateway whose security model matches your regulatory requirements. The cost of retrofitting gateway infrastructure after agents are already in production is substantially higher than including it from the beginning.

---

## Sources

- [Kong: What is an MCP Gateway?](https://konghq.com/blog/learning-center/what-is-a-mcp-gateway)
- [InfraCloud: The MCP Gateway — Enabling Secure and Scalable Enterprise AI Integration](https://www.infracloud.io/blogs/mcp-gateway/)
- [AIMultiple: Centralizing AI Tool Access with the MCP Gateway in 2026](https://aimultiple.com/mcp-gateway/)
- [Gravitee: MCP API Gateway Explained — Protocols, Caching, and Remote Server Integration](https://www.gravitee.io/blog/mcp-api-gateway-explained-protocols-caching-and-remote-server-integration)
- [Envoy AI Gateway: Announcing MCP Support](https://aigateway.envoyproxy.io/blog/mcp-implementation/)
- [Traefik Hub: MCP Gateway Documentation](https://doc.traefik.io/traefik-hub/mcp-gateway/mcp)
- [Docker: MCP Gateway — Secure Infrastructure for Agentic AI](https://www.docker.com/blog/docker-mcp-gateway-secure-infrastructure-for-agentic-ai/)
- [Docker MCP Gateway Documentation](https://docs.docker.com/ai/mcp-catalog-and-toolkit/mcp-gateway/)
- [Microsoft MCP Gateway (GitHub)](https://github.com/microsoft/mcp-gateway)
- [IBM ContextForge (GitHub)](https://github.com/IBM/mcp-context-forge)
- [agentgateway (GitHub)](https://github.com/agentgateway/agentgateway)
- [Lasso: Open Source MCP Security Gateway](https://www.lasso.security/resources/lasso-releases-first-open-source-security-gateway-for-mcp)
- [Stacklok: Introducing Virtual MCP Server](https://stacklok.com/blog/introducing-virtual-mcp-server-unified-gateway-for-multi-mcp-workflows/)
- [Awesome MCP Gateways (GitHub)](https://github.com/e2b-dev/awesome-mcp-gateways)
- [Solo.io: Why We Need a New Gateway for AI Agents](https://www.solo.io/blog/why-do-we-need-a-new-gateway-for-ai-agents)
- [API7.ai: AI Gateway vs MCP Gateway vs API Gateway](https://api7.ai/learning-center/api-gateway-guide/ai-gateway-vs-mcp-gateway-vs-api-gateway)
- [GitGuardian: OAuth for MCP — Emerging Enterprise Patterns for Agent Authorization](https://blog.gitguardian.com/oauth-for-mcp-emerging-enterprise-patterns-for-agent-authorization/)
- [Red Hat Developer: Advanced Authentication and Authorization for MCP Gateway](https://developers.redhat.com/articles/2025/12/12/advanced-authentication-authorization-mcp-gateway)
- [SigNoz: MCP Observability with OpenTelemetry](https://signoz.io/blog/mcp-observability-with-otel/)
- [agentgateway: MCP Observability](https://agentgateway.dev/docs/standalone/latest/mcp/mcp-observability/)
- [Peta: The Control Plane for MCP](https://peta.io/)
- [Cloudflare: MCP Server Portals](https://developers.cloudflare.com/cloudflare-one/access-controls/ai-controls/mcp-portals/)
- [MCP Specification: Transports](https://modelcontextprotocol.io/specification/2025-03-26/basic/transports)
- [MCP Specification: Authorization](https://modelcontextprotocol.io/specification/draft/basic/authorization)
- [Practical DevSecOps: MCP Security Vulnerabilities](https://www.practical-devsecops.com/mcp-security-vulnerabilities/)
- [Micro-MCP Architecture Pattern (GitHub)](https://github.com/mabualzait/MicroMCP)
- [Composio: Best MCP Gateway for Developers](https://composio.dev/content/best-mcp-gateway-for-developers)
- [TrueFoundry: 5 Best MCP Gateways in 2026](https://www.truefoundry.com/blog/best-mcp-gateways)
- [MintMCP: MCP Gateways for Rate Limiting and Access Control](https://www.mintmcp.com/blog/mcp-gateways-rate-limiting-access-control)
- [AWS: Transform Your MCP Architecture — Unite MCP Servers Through AgentCore Gateway](https://aws.amazon.com/blogs/machine-learning/transform-your-mcp-architecture-unite-mcp-servers-through-agentcore-gateway/)
- [Apigene: MCP Router — Route Tool Calls Across Multiple Servers](https://apigene.ai/blog/mcp-router)
