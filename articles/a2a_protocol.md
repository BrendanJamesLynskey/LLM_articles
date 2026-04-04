# Agent-to-Agent (A2A) Protocol: An Open Standard for Inter-Agent Communication

*April 2026 • Technical Report*

## 1. Introduction

As organizations deploy AI agents from multiple vendors and frameworks, a fundamental interoperability problem emerges: agents cannot talk to each other. A customer service agent built with LangChain cannot delegate a subtask to a procurement agent built on Salesforce's platform. A planning agent running on Google's infrastructure cannot coordinate with a specialist agent on Azure. Each integration requires bespoke glue code, creating the same N×M problem that plagued pre-USB peripherals and pre-LSP IDEs — but now at the agent layer.

The **Agent-to-Agent (A2A) Protocol**, announced by Google on April 9, 2025, and contributed to the Linux Foundation as an open-source project, addresses this problem by defining a standard protocol for inter-agent communication. A2A enables agents built by different vendors, using different frameworks, running on different infrastructure, to discover each other's capabilities, exchange messages, delegate tasks, and coordinate multi-step workflows — all without sharing memory, tools, or internal context.

As of March 2026, A2A has reached **version 1.0** with support from over 150 organizations, including every major cloud provider (Google, Microsoft, AWS), enterprise software leaders (Salesforce, SAP, ServiceNow), and the major agent framework communities (LangChain, CrewAI). The protocol is governed under the Linux Foundation with a Technical Steering Committee comprising Google, Microsoft, AWS, Cisco, Salesforce, ServiceNow, SAP, and IBM.

This report provides a detailed technical examination of A2A: its architecture, protocol mechanics, communication patterns, security model, relationship to other protocols, ecosystem, and practical deployment considerations.

## 2. The Problem A2A Solves

### 2.1 Agent Isolation

Modern enterprises are deploying AI agents across every function — customer service, sales, finance, HR, supply chain, IT operations. These agents are built on diverse platforms: Google's Agent Development Kit, Microsoft's Copilot Studio, Salesforce's Agentforce, SAP's Joule, custom implementations on LangChain or CrewAI. Each agent is capable within its domain, but they operate in isolation. A finance agent cannot ask a supply chain agent to check inventory levels. A hiring agent cannot coordinate with an onboarding agent. The result is intelligent islands that cannot collaborate.

### 2.2 The N×M Integration Problem at the Agent Layer

Before A2A, connecting agents required custom integration code for each pair of agent platforms. If an enterprise runs agents on five different platforms and wants them to collaborate, they need up to 20 custom bidirectional integrations — each with its own message format, authentication scheme, error handling, and state management. Adding a sixth platform requires ten more integrations. This scales quadratically and is unsustainable.

### 2.3 No Standard for Agent Discovery

Even if two agents could communicate, there was no standard way for one agent to discover what another agent can do. An orchestrator agent that wants to delegate a research task has no mechanism to find which agents in the enterprise are capable of research, what input formats they accept, what output they produce, or how to authenticate with them. Discovery was manual, static, and fragile.

### 2.4 No Support for Long-Running Agent Tasks

Many real-world agent tasks take minutes, hours, or even days — a deep research task, a complex procurement workflow, a multi-step hiring process. Existing RPC and API patterns are designed for request-response interactions that complete in seconds. There was no standard protocol for initiating a long-running task with an agent, receiving incremental progress updates, providing additional input mid-task, and collecting results when the task eventually completes.

### 2.5 Opacity as a Feature

Unlike tool-calling protocols where the caller specifies exactly what function to invoke and with what parameters, agent-to-agent communication must respect the autonomy and opacity of each agent. When a client agent delegates a task to a remote agent, it should not need to understand the remote agent's internal architecture, what tools it uses, what models it runs, or how it decomposes the task internally. A2A treats remote agents as opaque services — the client describes what it needs, and the remote agent decides how to accomplish it.

## 3. Architecture

### 3.1 The Three Actors

A2A defines three actors in its communication model:

**Users** are the end users — human operators or automated services — that initiate requests. Users interact with client agents through whatever interface the client provides (chat, API, voice, etc.).

**Client Agents (A2A Clients)** are agents that initiate communication on behalf of users. A client agent discovers remote agents, sends them tasks, monitors progress, and collects results. In a multi-agent system, the orchestrator or supervisor agent typically acts as the A2A client.

**Remote Agents (A2A Servers)** are agents that expose an HTTP endpoint implementing the A2A protocol. They receive tasks from clients, process them, and return results. A remote agent is opaque to the client — the client does not know or control how the remote agent accomplishes its work internally.

An agent can act as both a client and a server simultaneously. A mid-level coordinator agent might receive tasks from an orchestrator (acting as a server) and delegate subtasks to specialist agents (acting as a client).

### 3.2 Core Data Model

A2A's data model comprises five core objects:

**Agent Card** is a JSON metadata document that serves as a digital business card for an agent. It describes the agent's identity, capabilities, skills, endpoint URL, and authentication requirements. Agent Cards are the foundation of agent discovery.

**Task** is the fundamental stateful unit of work. Each task has a unique identifier and progresses through a defined lifecycle of states. Tasks are the primary abstraction for tracking work delegated to a remote agent.

**Message** is a single communication turn between a client and a remote agent. Each message has a role ("user" for client-originated, "agent" for server-originated) and contains one or more Parts. Messages are identified by a unique `messageId`.

**Part** is the smallest unit of content within a Message or Artifact. Each Part holds exactly one content type: `text` (plain textual content), `file` (binary data, either inline or by URI reference), or `data` (structured JSON). Parts include optional metadata fields for media type, filename, and custom annotations.

**Artifact** is a tangible output generated by the remote agent during task execution — documents, images, structured data, code files. Artifacts have a unique `artifactId`, a human-readable name, and one or more Parts. They support incremental streaming, allowing the remote agent to deliver partial results as they are produced.

### 3.3 Context and Conversational Threading

A2A introduces a `contextId` — a server-generated identifier that groups related Tasks into a conversational thread. When a client sends a message that initiates a new task, the server assigns a `contextId`. Subsequent tasks related to the same conversation reference this `contextId`, allowing the server to maintain conversational state across multiple task interactions. This is essential for multi-turn interactions where context from earlier exchanges informs later processing.

## 4. Agent Cards: Discovery and Capability Advertisement

### 4.1 The Agent Card Schema

The Agent Card is the entry point for all A2A interactions. Before a client can communicate with a remote agent, it must obtain the agent's card. The card is a JSON document with the following key fields:

**Identity**: `name`, `description`, `id`, `version`, and optionally `icon_url` and a `provider` object (with the provider organization's name, URL, and contact information).

**Service Endpoint**: The `url` field specifying the A2A service endpoint where the agent accepts requests.

**Capabilities**: A capabilities object declaring protocol features the agent supports:

```json
{
  "capabilities": {
    "streaming": true,
    "pushNotifications": true,
    "extendedAgentCard": true
  }
}
```

**Skills**: An array of skill objects describing what the agent can do. Each skill includes an `id`, `name`, `description`, `tags`, `examples` (sample queries), and `inputModes`/`outputModes` (MIME types the skill accepts and produces):

```json
{
  "skills": [
    {
      "id": "expense_report_analysis",
      "name": "Expense Report Analysis",
      "description": "Analyzes expense reports for policy compliance",
      "tags": ["finance", "compliance", "expenses"],
      "examples": ["Check if this expense report meets policy"],
      "inputModes": ["text/plain", "application/pdf"],
      "outputModes": ["text/plain", "application/json"]
    }
  ]
}
```

**Security**: `securitySchemes` defines available authentication mechanisms, and a `security` array declares which schemes are required for access. The security scheme definitions align with the OpenAPI specification.

### 4.2 Discovery Mechanisms

A2A defines three strategies for agent discovery:

**Well-Known URI**: Following RFC 8615, agents publish their Agent Card at `https://{domain}/.well-known/agent-card.json`. Clients can discover any agent by performing an HTTP GET to this standardized path. This is the primary discovery mechanism for publicly accessible agents.

**Curated Registries**: Intermediary services maintain collections of Agent Cards, enabling search and filtering by capability, domain, or organization. Registries are useful in enterprise environments where agents are deployed across multiple teams and need centralized cataloging.

**Direct Configuration**: For static agent relationships in controlled environments, Agent Card URLs can be hardcoded in configuration files. This is the simplest approach but does not scale to dynamic environments.

### 4.3 Extended Agent Cards

Some agent capabilities or metadata may be sensitive — pricing information, internal API details, or capability restrictions that should not be publicly discoverable. A2A supports an Extended Agent Card, available at an authenticated endpoint (`GET /extendedAgentCard`). The extended card supplements the public card with additional details that are only revealed to authenticated, authorized clients.

### 4.4 Agent Card Caching and Signing

Servers include `Cache-Control` headers with `max-age` and `ETag` values on Agent Card responses. Clients use conditional requests (`If-None-Match`, `If-Modified-Since`) to minimize unnecessary fetches.

Starting with protocol version 0.3, Agent Cards support a `signature` field for authenticity verification. The signing process uses canonicalized JSON to ensure deterministic signatures, preventing tampering with the card during transit or storage in registries.

## 5. Task Lifecycle

### 5.1 Task States

Tasks progress through a well-defined set of states:

| State | Description |
|-------|-------------|
| `submitted` | Task received by the server, not yet processing |
| `working` | Server is actively processing the task |
| `input-required` | Server needs additional input from the client to proceed |
| `completed` | Task finished successfully |
| `failed` | Task terminated due to an error |
| `canceled` | Task canceled at the client's request |
| `rejected` | Server declined to accept the task |
| `auth-required` | Additional authentication is needed to proceed |

### 5.2 State Transitions

A typical successful task follows the path: `submitted → working → completed`. However, real-world tasks are often more complex. A task may cycle through `working → input-required → working` multiple times as the remote agent gathers information through multi-turn dialogue. A task may be `rejected` immediately if the server determines it cannot handle the request. A `working` task may transition to `failed` if an unrecoverable error occurs, or to `canceled` if the client explicitly requests cancellation.

### 5.3 Multi-Turn Interactions

The `input-required` state is central to A2A's support for collaborative task completion. When a remote agent needs clarification, additional data, or a decision from the client, it transitions the task to `input-required` and includes a message explaining what it needs. The client responds by sending a follow-up message referencing the same `taskId` and `contextId`. The server resumes processing and may repeat this cycle as many times as needed before reaching a terminal state.

This pattern supports scenarios such as: a travel booking agent asking for date preferences, a financial agent requesting approval for a transaction amount, or a research agent presenting preliminary findings and asking which direction to explore further.

### 5.4 Task History

Tasks maintain a message history that records the full conversation between client and server. When retrieving a task with `tasks/get`, the client can specify a `historyLength` parameter to control how much of the conversation history is returned. This is important for long-running tasks where the full history may be large.

## 6. Communication Patterns

A2A supports three communication patterns, each suited to different operational requirements.

### 6.1 Request/Response (Synchronous)

The simplest pattern. The client sends a `message/send` request and receives a complete response. This works well for quick tasks that complete in seconds.

In **blocking mode** (the default), the server holds the HTTP connection open until the task reaches a terminal state, then returns the complete result. In **non-blocking mode** (triggered by setting `returnImmediately: true`), the server returns immediately with the task in its current state (typically `submitted` or `working`). The client then polls for completion using `tasks/get`.

```json
{
  "jsonrpc": "2.0",
  "id": 1,
  "method": "message/send",
  "params": {
    "message": {
      "role": "user",
      "parts": [{"kind": "text", "text": "Analyze Q1 sales data"}],
      "messageId": "msg-001"
    }
  }
}
```

### 6.2 Streaming (Server-Sent Events)

For tasks that produce incremental output or take longer than a few seconds, A2A supports streaming via Server-Sent Events (SSE). The client sends a `message/stream` request, and the server responds with a stream of events as the task progresses.

The stream uses a `StreamResponse` wrapper, with each event containing one of:
- `TaskStatusUpdateEvent` — state transitions with timestamps
- `TaskArtifactUpdateEvent` — artifact additions or updates, with `append` and `lastChunk` flags for incremental delivery

Streaming allows clients to display intermediate progress — showing the user that the agent is working, presenting partial results as they become available, and indicating when additional input is needed.

### 6.3 Push Notifications (Webhooks)

For long-running tasks where maintaining an open HTTP connection or SSE stream is impractical, A2A supports push notifications. The client registers a webhook URL using `tasks/pushNotificationConfig/create`, and the server sends HTTP POST requests to that URL whenever the task's state changes or new artifacts are produced.

Push notifications use the same `StreamResponse` payload format as SSE, ensuring consistency across communication patterns. Webhook delivery includes authentication credentials (Bearer tokens or mTLS) configured during registration, ensuring that notifications are sent securely.

This pattern is essential for tasks spanning hours or days — a deep research project, a multi-stage procurement workflow, or an analysis task waiting for external data that arrives on an unpredictable schedule.

## 7. Protocol Mechanics

### 7.1 JSON-RPC 2.0 Foundation

A2A is built on JSON-RPC 2.0, transported over HTTP(S). All requests are sent as HTTP POST with `application/json` content type. This is the same RPC foundation used by MCP, LSP, and Ethereum — a well-understood, lightweight protocol with clear request-response semantics.

### 7.2 JSON-RPC Methods

**Message Operations:**
- `message/send` — send a message and receive a Task or Message response
- `message/stream` — send a message and receive an SSE streaming response

**Task Management:**
- `tasks/get` — retrieve a task's current state, with optional `historyLength` parameter
- `tasks/list` — paginated task listing with filtering by `contextId`, `status`, and timestamps; cursor-based pagination via `pageToken`
- `tasks/cancel` — request cancellation of a running task
- `tasks/subscribe` — establish an SSE stream for updates on an existing task

**Push Notification Configuration:**
- `tasks/pushNotificationConfig/create` — register a webhook endpoint
- `tasks/pushNotificationConfig/get` — retrieve a webhook configuration
- `tasks/pushNotificationConfig/list` — list all webhook configurations for a task
- `tasks/pushNotificationConfig/delete` — remove a webhook endpoint

### 7.3 HTTP/REST Binding

In addition to JSON-RPC, A2A defines an alternative HTTP/REST binding that maps operations to conventional REST endpoints:

| Endpoint | Method | Operation |
|----------|--------|-----------|
| `/messages` | POST | Send message |
| `/messages/stream` | POST | Send streaming message |
| `/tasks/{taskId}` | GET | Get task |
| `/tasks` | GET | List tasks |
| `/tasks/{taskId}/cancel` | POST | Cancel task |
| `/tasks/{taskId}/subscribe` | GET | Subscribe to task (SSE) |
| `/tasks/{taskId}/pushConfigs` | POST/GET | Create/List webhooks |

### 7.4 gRPC Binding

Added in protocol version 0.3, A2A also supports gRPC as a transport. The gRPC binding maps all operations to RPCs, uses gRPC server streaming for streaming responses, and transmits service parameters via gRPC metadata. This binding is particularly relevant for high-throughput, latency-sensitive, and polyglot enterprise environments where gRPC is already the standard internal transport.

### 7.5 Service Parameters

A2A defines HTTP headers for protocol-level metadata:
- `A2A-Version` — the protocol version (e.g., "1.0")
- `A2A-Extensions` — comma-separated URIs of extensions the client supports

### 7.6 Error Handling

A2A defines structured error codes for common failure modes: `TaskNotFoundError`, `TaskNotCancelableError`, `PushNotificationNotSupportedError`, `UnsupportedOperationError`, `ContentTypeNotSupportedError`, `InvalidAgentResponseError`, `VersionNotSupportedError`, and others. These error codes provide clear, actionable feedback for client implementations.

## 8. Security Model

### 8.1 Authentication Schemes

A2A supports five authentication schemes, aligned with the OpenAPI specification:

**API Key** — key transmitted in header, query parameter, or cookie. Suitable for server-to-server communication in trusted environments.

**HTTP Authentication** — Basic or Bearer authentication. Bearer tokens (typically JWT) are the most common choice for production deployments.

**OAuth 2.0** — Authorization Code, Client Credentials, and Device Code flows. The recommended scheme for enterprise deployments where agents need fine-grained, delegated access.

**OpenID Connect** — OIDC provider discovery for identity federation across organizational boundaries.

**Mutual TLS** — Certificate-based authentication providing strong identity guarantees at the transport layer. Required for high-security environments.

### 8.2 Identity and Authorization

Credentials are transmitted at the HTTP transport layer (e.g., `Authorization: Bearer <TOKEN>`) — never embedded in JSON-RPC payloads. Agents are treated as standard enterprise applications from an authentication and authorization perspective, meaning existing identity infrastructure (IdPs, OAuth servers, certificate authorities) can be reused directly.

Authorization operates at multiple levels: skill-based access (OAuth scopes restricting which agent skills a client can invoke), resource-level gating (agents enforce backend access controls), and identity-based policies (different clients receive different capabilities based on their authenticated identity).

### 8.3 Transport Security

HTTPS is mandatory in production. TLS 1.2 or higher with strong cipher suites is required. Certificate verification against trusted certificate authorities is enforced.

### 8.4 Agent Card Signing

Agent Card signing, introduced in version 0.3, uses canonicalized JSON to produce deterministic signatures. This prevents tampering with Agent Cards during transit or when stored in third-party registries. Clients can verify that a card was issued by the claimed provider before trusting its contents.

### 8.5 Enterprise Security Recommendations

The A2A specification includes guidance for enterprise deployments: integration with API management platforms for externally exposed agents, centralized policy enforcement (authentication, rate limiting, quotas), OpenTelemetry with W3C Trace Context for distributed tracing across agent interactions, comprehensive audit logging with `taskId`, `contextId`, and correlation IDs, and awareness of regulatory compliance requirements (GDPR, CCPA, HIPAA).

## 9. A2A and MCP: Complementary Protocols

### 9.1 Different Axes of Communication

A2A and MCP (Model Context Protocol) address fundamentally different communication axes. MCP is **vertical** — it connects a single agent downward to tools, data sources, and external systems. A2A is **horizontal** — it connects agents laterally to other agents.

MCP answers: "How does an agent use a tool?" A2A answers: "How do agents talk to each other?"

### 9.2 How They Work Together

In a sophisticated multi-agent system, both protocols operate simultaneously:

1. A user requests a complex task through an orchestrator agent
2. The orchestrator uses **A2A** to discover and delegate subtasks to specialist agents
3. Each specialist agent uses **MCP** to access the tools and data sources it needs
4. Specialist agents return results to the orchestrator via **A2A**
5. The orchestrator synthesizes results and responds to the user

This separation of concerns is architecturally clean: A2A handles the coordination layer (who does what), while MCP handles the capability layer (how each agent accesses its tools).

### 9.3 Key Differences

| Aspect | MCP | A2A |
|--------|-----|-----|
| Focus | Tool and data access for one agent | Inter-agent communication |
| Direction | Vertical (agent → tools) | Horizontal (agent ↔ agent) |
| State model | Stateless tool calls | Stateful tasks with lifecycle |
| Discovery | Tool manifests within server | Agent Cards at well-known URIs |
| Long-running operations | Not designed for | First-class support |
| Agent opacity | Tools are fully specified | Remote agents are opaque |
| Governance | Linux Foundation (AAIF) | Linux Foundation (A2A Project) |

### 9.4 Organizational Convergence

Both protocols are now under Linux Foundation governance. In December 2025, the Agentic AI Foundation (AAIF) was launched by OpenAI, Anthropic, Google, Microsoft, AWS, and Block to coordinate standards across the agentic AI ecosystem. While MCP and A2A remain separate specifications with separate governance, there is increasing coordination to ensure they work well together.

## 10. Ecosystem and Adoption

### 10.1 Cloud Platform Support

All three major cloud providers have integrated A2A:

**Google Cloud** — A2A is natively supported in the Agent Development Kit (ADK), Vertex AI Agent Engine, and Cloud Run. Google provides first-party SDKs and reference implementations.

**Microsoft Azure** — Azure AI Foundry and Copilot Studio support A2A for cross-platform agent orchestration. Microsoft joined the A2A Technical Steering Committee and has demonstrated A2A integration with Copilot agents.

**Amazon Web Services** — Amazon Bedrock AgentCore natively supports the A2A protocol contract, enabling Bedrock-hosted agents to participate in A2A ecosystems.

### 10.2 Enterprise Software

Major enterprise software vendors have committed to A2A support: Salesforce (Agentforce), SAP (Joule, Business AI), ServiceNow, Workday, Intuit, and others. This means that agents built on these platforms can interoperate with agents on any other A2A-compatible platform.

### 10.3 Agent Framework Integration

The major open-source agent frameworks support A2A: LangChain/LangGraph, CrewAI, and Google ADK all provide A2A client and server implementations. This means that custom agents built with these frameworks can participate in A2A ecosystems alongside commercial platform agents.

### 10.4 SDKs and Tooling

Five official language SDKs are available: Python, Go, JavaScript, Java, and .NET. The ecosystem also includes an A2A Inspector tool for debugging agent interactions and a Protocol Technology Compatibility Kit (TCK) for validating that an agent implementation conforms to the specification.

### 10.5 Protocol Consolidation

In August 2025, IBM's Agent Communication Protocol (ACP) merged into A2A under the Linux Foundation, consolidating two competing inter-agent communication standards into one. This merger brought IBM onto the A2A Technical Steering Committee and signaled industry convergence around A2A as the standard for agent-to-agent communication.

## 11. Enterprise Use Cases

### 11.1 Multi-Vendor Agent Orchestration

A large enterprise runs customer-facing agents on Salesforce, internal IT agents on ServiceNow, finance agents on a custom LangChain stack, and HR agents on Workday. A2A enables an orchestrator agent to route user requests to the appropriate specialist agent regardless of the underlying platform. A request like "I need to onboard a new contractor, set up their laptop, and create a purchase order for their equipment" can flow through HR, IT, and procurement agents seamlessly.

### 11.2 Travel and Expense Management

A user asks an assistant to plan a business trip. The assistant (A2A client) delegates to a travel booking agent (flights and hotels), a finance agent (policy compliance and budget checking), a payment agent (transaction processing), and a currency conversion agent. Each operates as an independent A2A server. The orchestrator manages the workflow, handles the multi-turn interactions when agents need preferences or approvals, and presents a unified experience to the user.

### 11.3 Supply Chain Optimization

A sales forecasting agent detects increased demand for a product line. Using A2A, it notifies the supply chain planning agent, which coordinates with inventory management agents, logistics agents, and supplier agents to optimize stock levels and delivery schedules — all without human intervention or custom API integrations between the systems.

### 11.4 Collaborative Research

A research orchestrator agent breaks a complex investigation into parallel workstreams. It delegates literature review to one specialist agent, data analysis to another, and competitive landscape analysis to a third. Each agent works independently, with the orchestrator collecting results via A2A push notifications as they complete over hours or days. The orchestrator synthesizes the findings into a comprehensive report.

## 12. Protocol Evolution

### 12.1 Version History

| Version | Date | Key Features |
|---------|------|-------------|
| 0.1 | April 2025 | Initial specification with core concepts |
| 0.2 | May–June 2025 | Standardized authentication, stateless interaction support |
| 0.3 | July 2025 | gRPC binding, Agent Card signing, well-known path change to `agent-card.json` |
| 1.0 | March 2026 | Production-ready release, extended Agent Card in capabilities object, full SDK coverage |

### 12.2 Design Principles

Five principles have guided A2A's evolution:

1. **Embrace agentic capabilities** — agents collaborate through natural, unstructured communication without requiring shared memory or tools
2. **Build on existing standards** — HTTP, SSE, JSON-RPC, gRPC — no proprietary transports
3. **Secure by default** — enterprise-grade authentication and authorization from day one
4. **Support for long-running tasks** — from sub-second operations to multi-day workflows
5. **Modality agnostic** — text, audio, video, and file streaming are all first-class citizens

### 12.3 Future Directions

The A2A roadmap includes enhanced multi-modal streaming, richer agent capability negotiation, tighter integration with identity federations, and tooling for monitoring and debugging production multi-agent systems. The protocol's governance under the Linux Foundation ensures that its evolution reflects the needs of the broad community of adopters.

## 13. Conclusion

A2A fills a critical gap in the agentic AI stack. While MCP standardized how agents access tools, A2A standardizes how agents communicate with each other. Together, they provide a complete interoperability layer: MCP for the vertical connection between agents and their capabilities, A2A for the horizontal connection between agents.

The protocol's rapid adoption — from initial announcement to 150+ supporting organizations and version 1.0 in under a year — reflects genuine industry demand for agent interoperability. As multi-agent systems move from research prototypes to production deployments, a standard protocol for agent communication becomes infrastructure rather than convenience. A2A, built on familiar web standards, governed by the Linux Foundation, and supported by every major cloud and enterprise platform, is positioned to be that infrastructure.
