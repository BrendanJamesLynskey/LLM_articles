# OAuth 2.1 for AI Applications: Securing the Agent-Tool Interface

*April 2026 • Technical Report*

## 1. Introduction

The rapid expansion of AI agents into production environments has exposed a fundamental tension in how these systems authenticate and authorize their interactions with external services. For years, the default approach was simple: embed an API key in the agent's configuration file or environment variable, and use it for every request. This worked well enough when LLM applications were stateless prompt-response systems with a single integration point. It works poorly — and dangerously — when an autonomous agent orchestrates dozens of tools, acts on behalf of multiple users, and maintains long-running sessions that persist for hours or days.

OAuth 2.1, the forthcoming consolidation of OAuth 2.0 best practices into a single specification, provides the authorization framework that AI applications need. It offers scoped, time-limited, revocable tokens; delegated authorization where an agent acts on behalf of a user without holding the user's credentials; and a mature ecosystem of identity providers, token introspection endpoints, and security tooling. The Model Context Protocol (MCP), which has emerged as the standard interface between LLM hosts and external tools, has adopted OAuth as its authentication layer for remote server connections, making OAuth literacy essential for anyone building or deploying MCP-based systems.

This report provides a comprehensive technical examination of OAuth 2.1 in the context of AI applications. It covers the protocol fundamentals, the specific challenges that autonomous agents introduce, MCP's OAuth integration, token management strategies, scope design, delegated authorization chains, machine-to-machine authentication, JWT structure, identity provider integration, multi-tenant deployment patterns, API gateway considerations, security pitfalls, practical implementation walkthroughs, and the evolving standards landscape.

## 2. Why AI Applications Need OAuth

### 2.1 The API Key Problem

API keys are shared secrets. They grant the bearer a fixed set of permissions with no concept of who the bearer is, what they intend to do, or how long they should retain access. In a traditional web application, this is manageable: the API key lives in a server-side environment variable, is used by a single process, and is rotated on a known schedule.

AI agents break every assumption that makes API keys tolerable. An agent may be given an API key for a database service, a cloud storage API, a code repository, and a communication platform — all embedded in its configuration. If the agent's context window, log output, or tool descriptions are ever exposed (through prompt injection, debug logging, or an insecure MCP transport), all of those keys are compromised simultaneously. There is no way to limit what the agent does with the key; it has the same permissions as whoever created the key. There is no audit trail connecting the agent's actions to the user who initiated the request. And there is no mechanism for the agent to obtain elevated permissions temporarily when a task requires it, or to have its permissions reduced when it is performing a routine operation.

### 2.2 Agent Autonomy Creates Delegation Challenges

When a human user authenticates to a service, they are present to approve each action. When an AI agent authenticates to a service on behalf of a user, the user may not be present at all — the agent may be executing a multi-step workflow that was initiated hours ago. This creates a delegation problem: the agent needs sufficient authority to complete its task, but the user needs assurance that the agent will not exceed the intended scope.

OAuth was designed precisely for this delegation pattern. The authorization code flow allows a user to grant an application (in this case, an agent) specific, limited permissions to act on their behalf, without sharing their credentials. The resulting access token encodes those permissions as scopes, has a defined expiration, and can be revoked at any time. The agent never sees the user's password, and the service can distinguish between actions taken by the user directly and actions taken by the agent on the user's behalf.

### 2.3 The MCP Catalyst

The adoption of MCP as the standard protocol for connecting LLM applications to external tools has made OAuth integration an immediate practical concern rather than a theoretical best practice. MCP's specification for remote server authentication is built on OAuth 2.1, including support for dynamic client registration, PKCE, and token-based access control. Any MCP server that exposes tools over HTTP must implement OAuth or integrate with an identity provider that does. This means that the intersection of OAuth and AI is no longer a niche concern — it is the default authentication model for the emerging agent ecosystem.

## 3. OAuth 2.0/2.1 Fundamentals

### 3.1 Core Concepts

OAuth is an authorization framework, not an authentication protocol. It answers the question "what is this application allowed to do?" rather than "who is this user?" (though OpenID Connect, built on top of OAuth, adds authentication). The framework defines four roles:

**Resource Owner** — the entity that can grant access to a protected resource. In AI contexts, this is typically the human user who initiates the agent's task.

**Client** — the application requesting access on behalf of the resource owner. In AI contexts, this is the AI agent or the MCP host application.

**Authorization Server** — the server that issues access tokens after authenticating the resource owner and obtaining their consent. This might be an identity provider like Auth0, Okta, or Keycloak, or a custom authorization server.

**Resource Server** — the server hosting the protected resources. In AI contexts, this is the MCP server, API endpoint, or tool service that the agent needs to access.

### 3.2 The Authorization Code Flow

The authorization code flow is the primary flow recommended by OAuth 2.1 for applications that can interact with a user. The sequence is:

1. The client (agent/MCP host) directs the resource owner (user) to the authorization server's authorization endpoint, including the requested scopes, a redirect URI, and a state parameter for CSRF protection.
2. The authorization server authenticates the user (via login page, SSO, etc.) and presents a consent screen showing the requested permissions.
3. The user approves the request.
4. The authorization server redirects back to the client's redirect URI with an authorization code — a short-lived, single-use credential.
5. The client exchanges the authorization code for an access token (and optionally a refresh token) by calling the authorization server's token endpoint, authenticating itself in the process.
6. The client uses the access token to make requests to the resource server.

This flow ensures that the user's credentials are never exposed to the client, that the client receives only the permissions the user explicitly approved, and that the authorization code is useless without the client's own credentials (or, with PKCE, without the code verifier).

### 3.3 Client Credentials Flow

For machine-to-machine communication where no user is present — such as a backend MCP server calling another backend service — the client credentials flow is used. The client authenticates directly to the authorization server using its own credentials (client ID and client secret, or a client assertion) and receives an access token. There is no user involvement and no authorization code. The token's permissions are determined by what the authorization server has been configured to grant to that specific client.

### 3.4 Refresh Tokens

Access tokens are intentionally short-lived — typically 5 to 60 minutes. This limits the window of damage if a token is compromised. For longer-running sessions, the authorization server issues a refresh token alongside the access token. The client uses the refresh token to obtain a new access token when the current one expires, without requiring the user to re-authenticate. Refresh tokens have longer lifetimes (hours, days, or indefinite) but can be revoked and are subject to rotation policies where each use of a refresh token invalidates it and issues a new one.

### 3.5 Scopes

Scopes are strings that define the specific permissions an access token grants. They are requested by the client and approved by the user (or configured by an administrator for client credentials). Examples in an AI context might include `tools:read` (list available tools), `tools:execute` (invoke tools), `database:query` (run read-only database queries), or `files:write` (create and modify files). The authorization server may grant all requested scopes, a subset, or additional default scopes.

### 3.6 Audiences

The audience (`aud` claim in a JWT) identifies which resource server(s) the token is intended for. This prevents a token issued for Service A from being replayed against Service B. In multi-tool agent scenarios where the agent accesses multiple MCP servers, audience restrictions ensure that a token obtained for the database tool server cannot be used to access the code execution tool server.

### 3.7 What OAuth 2.1 Changes

OAuth 2.1 (formalized in draft RFC 9728 and related documents) is not a new protocol — it is a consolidation of OAuth 2.0 best practices, making mandatory what was previously recommended:

- **PKCE is required** for all authorization code grants, not just public clients.
- **The implicit flow is removed** — it is no longer a valid grant type.
- **Refresh token rotation is recommended** — each refresh token use should return a new refresh token.
- **Exact redirect URI matching is required** — no pattern matching or partial matching.
- **Bearer tokens in query parameters are prohibited** — tokens must be sent in the Authorization header or request body.

These changes are directly relevant to AI applications, where the attack surface is broader (agent context windows, tool descriptions, and log output are all potential leak vectors) and the consequences of token misuse are amplified by agent autonomy.

## 4. PKCE: Proof Key for Code Exchange

### 4.1 The Problem PKCE Solves

The authorization code flow has a vulnerability: the authorization code is transmitted via the user's browser as a query parameter in the redirect URI. If an attacker can intercept this redirect — through a malicious browser extension, a compromised redirect URI, or a man-in-the-middle on the redirect — they can exchange the authorization code for an access token. For confidential clients (those with a client secret), this is mitigated because the attacker also needs the client secret to complete the exchange. But for public clients (native apps, SPAs, and notably, dynamically registered MCP clients that may not have a pre-shared secret), the authorization code alone is sufficient to obtain a token.

### 4.2 How PKCE Works

PKCE (pronounced "pixie," defined in RFC 7636) adds a proof-of-possession mechanism to the authorization code exchange:

1. **Code Verifier Generation**: The client generates a cryptographically random string called the `code_verifier` — a high-entropy string between 43 and 128 characters.
2. **Code Challenge Computation**: The client computes a `code_challenge` by applying a SHA-256 hash to the code verifier and base64url-encoding the result. (A plain method is also defined but SHA-256 is required by OAuth 2.1.)
3. **Authorization Request**: The client includes the `code_challenge` and `code_challenge_method` (S256) in its authorization request to the authorization server.
4. **Authorization Code Issuance**: The authorization server stores the code challenge alongside the authorization code it issues.
5. **Token Exchange**: When the client exchanges the authorization code for a token, it includes the original `code_verifier`. The authorization server hashes this verifier and compares it to the stored code challenge. If they match, the exchange proceeds. If not, it is rejected.

The security property is straightforward: even if an attacker intercepts the authorization code and the code challenge (both of which are transmitted through the browser), they cannot derive the code verifier from the code challenge (SHA-256 is a one-way function). Only the legitimate client, which generated the code verifier, can complete the token exchange.

### 4.3 Why OAuth 2.1 Makes PKCE Mandatory

OAuth 2.0 required PKCE only for public clients. OAuth 2.1 requires it for all clients, including confidential ones. The rationale is defense in depth: even confidential clients can have their secrets compromised, and PKCE provides an additional layer of protection that costs virtually nothing to implement. For AI applications, where MCP clients may be dynamically registered and may not have a pre-established trust relationship with the authorization server, PKCE provides a critical safeguard.

### 4.4 Implementation Example

In Python using a standard OAuth library:

```python
import hashlib
import base64
import secrets

# Step 1: Generate code verifier
code_verifier = secrets.token_urlsafe(64)  # 64 bytes of randomness, base64url-encoded

# Step 2: Compute code challenge
code_challenge = base64.urlsafe_b64encode(
    hashlib.sha256(code_verifier.encode('ascii')).digest()
).rstrip(b'=').decode('ascii')

# Step 3: Include in authorization request
auth_url = (
    f"{authorization_endpoint}?"
    f"response_type=code&"
    f"client_id={client_id}&"
    f"redirect_uri={redirect_uri}&"
    f"scope=tools:execute+database:query&"
    f"code_challenge={code_challenge}&"
    f"code_challenge_method=S256&"
    f"state={state}"
)

# Step 5: Include code_verifier in token exchange
token_response = requests.post(token_endpoint, data={
    'grant_type': 'authorization_code',
    'code': authorization_code,
    'redirect_uri': redirect_uri,
    'client_id': client_id,
    'code_verifier': code_verifier,
})
```

The equivalent in TypeScript:

```typescript
import crypto from 'crypto';

// Step 1: Generate code verifier
const codeVerifier = crypto.randomBytes(64).toString('base64url');

// Step 2: Compute code challenge
const codeChallenge = crypto
  .createHash('sha256')
  .update(codeVerifier)
  .digest('base64url');

// Step 3: Build authorization URL
const authUrl = new URL(authorizationEndpoint);
authUrl.searchParams.set('response_type', 'code');
authUrl.searchParams.set('client_id', clientId);
authUrl.searchParams.set('redirect_uri', redirectUri);
authUrl.searchParams.set('scope', 'tools:execute database:query');
authUrl.searchParams.set('code_challenge', codeChallenge);
authUrl.searchParams.set('code_challenge_method', 'S256');
authUrl.searchParams.set('state', state);

// Step 5: Exchange code for token
const tokenResponse = await fetch(tokenEndpoint, {
  method: 'POST',
  headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
  body: new URLSearchParams({
    grant_type: 'authorization_code',
    code: authorizationCode,
    redirect_uri: redirectUri,
    client_id: clientId,
    code_verifier: codeVerifier,
  }),
});
```

## 5. MCP's OAuth Integration

### 5.1 How MCP Servers Declare Authentication Requirements

When an MCP client connects to a remote MCP server over HTTP, the server must declare its authentication requirements. This is done through the OAuth 2.0 Authorization Server Metadata endpoint, as defined in RFC 8414. The MCP client discovers the authorization server by making a GET request to the server's well-known configuration endpoint:

```
GET https://mcp-server.example.com/.well-known/oauth-authorization-server
```

The response is a JSON document containing the authorization server's metadata:

```json
{
  "issuer": "https://mcp-server.example.com",
  "authorization_endpoint": "https://mcp-server.example.com/oauth/authorize",
  "token_endpoint": "https://mcp-server.example.com/oauth/token",
  "registration_endpoint": "https://mcp-server.example.com/oauth/register",
  "scopes_supported": ["tools:list", "tools:execute", "resources:read"],
  "response_types_supported": ["code"],
  "grant_types_supported": ["authorization_code", "client_credentials", "refresh_token"],
  "code_challenge_methods_supported": ["S256"],
  "token_endpoint_auth_methods_supported": ["client_secret_post", "none"]
}
```

If the server does not provide this endpoint, the MCP client falls back to well-known defaults based on the server's base URL. The key point is that authentication discovery is automated — the MCP client does not need pre-configured knowledge of the server's OAuth endpoints.

### 5.2 The Authorization Flow for Remote MCP Servers

The full authorization flow for a remote MCP server connection proceeds as follows:

1. The MCP client attempts to connect to the remote server.
2. The server responds with a 401 Unauthorized status, indicating that authentication is required.
3. The MCP client fetches the server's OAuth metadata from the well-known endpoint.
4. If the client has not previously registered with this authorization server, it performs dynamic client registration (see next section).
5. The MCP client initiates the OAuth authorization code flow with PKCE, directing the user to the authorization server's consent page.
6. The user authenticates and approves the requested scopes.
7. The MCP client receives the authorization code and exchanges it for an access token.
8. The MCP client retries the original connection, including the access token in the Authorization header.
9. The server validates the token and establishes the MCP session.

For subsequent requests within the same session, the access token is included automatically. When the token expires, the client uses the refresh token to obtain a new one without user interaction.

### 5.3 Dynamic Client Registration

One of MCP's most important OAuth features is support for dynamic client registration, defined in RFC 7591. In traditional OAuth deployments, a client must be manually registered with the authorization server by an administrator, who assigns it a client ID and client secret. This is impractical for MCP, where users may connect to arbitrary MCP servers that they have never interacted with before.

Dynamic client registration allows the MCP client to register itself with the authorization server at runtime:

```
POST https://mcp-server.example.com/oauth/register
Content-Type: application/json

{
  "client_name": "Claude Desktop",
  "redirect_uris": ["http://localhost:8765/oauth/callback"],
  "grant_types": ["authorization_code", "refresh_token"],
  "response_types": ["code"],
  "token_endpoint_auth_method": "none"
}
```

The authorization server responds with a client ID (and optionally a client secret):

```json
{
  "client_id": "dyn_abc123xyz",
  "client_name": "Claude Desktop",
  "redirect_uris": ["http://localhost:8765/oauth/callback"],
  "grant_types": ["authorization_code", "refresh_token"],
  "token_endpoint_auth_method": "none"
}
```

The MCP client stores this registration and uses the assigned client ID for subsequent authorization requests. Dynamic registration means that users can connect to new MCP servers without any manual configuration — the MCP host handles the entire OAuth bootstrapping process transparently.

### 5.4 Security Considerations for MCP OAuth

Several security properties are important in the MCP OAuth context. The MCP specification requires PKCE for all authorization code flows, since dynamically registered clients are typically public clients (no client secret). The redirect URI must use localhost for desktop applications, preventing redirect-based attacks. The server's well-known metadata endpoint must be served over HTTPS, ensuring that the metadata itself is not tampered with. And the access token must be transmitted only in the HTTP Authorization header, never in query parameters or MCP message content.

## 6. Token Management for Agents

### 6.1 The Long-Running Agent Challenge

AI agents often execute workflows that span hours or days. A coding agent tasked with refactoring a large codebase might work autonomously for 8 hours, accessing a code repository, running tests, and updating documentation. A data pipeline agent might run continuously, processing new data as it arrives. These long-running sessions exceed the typical access token lifetime of 15-60 minutes, requiring robust token refresh mechanisms.

### 6.2 Refresh Token Rotation

OAuth 2.1 recommends refresh token rotation: each time a refresh token is used to obtain a new access token, the authorization server issues a new refresh token and invalidates the old one. This limits the window of exposure if a refresh token is compromised — the legitimate client and the attacker will eventually use the same refresh token, and the authorization server can detect the conflict and revoke all tokens in the chain.

For agents, refresh token rotation introduces a coordination challenge: if the agent crashes between receiving a new token pair and persisting the new refresh token, it may lose access entirely. Robust implementations must use atomic token persistence — storing the new refresh token before using the new access token for any requests.

### 6.3 Token Storage Security

Where tokens are stored is as important as how they are obtained. The hierarchy of storage options, from most to least secure:

**Hardware Security Modules (HSMs)** — Tokens are stored in tamper-resistant hardware. Suitable for high-security deployments but impractical for most agent workloads.

**Secret Managers (Vault, AWS Secrets Manager, GCP Secret Manager, Azure Key Vault)** — Tokens are stored in a dedicated secrets management service with access controls, audit logging, and automatic rotation support. This is the recommended approach for production agent deployments.

```python
import boto3

secrets_client = boto3.client('secretsmanager')

def store_tokens(agent_id: str, access_token: str, refresh_token: str):
    secrets_client.put_secret_value(
        SecretId=f"agent/{agent_id}/oauth_tokens",
        SecretString=json.dumps({
            "access_token": access_token,
            "refresh_token": refresh_token,
            "stored_at": datetime.utcnow().isoformat(),
        })
    )

def get_tokens(agent_id: str) -> dict:
    response = secrets_client.get_secret_value(
        SecretId=f"agent/{agent_id}/oauth_tokens"
    )
    return json.loads(response['SecretString'])
```

**Encrypted filesystem storage** — Tokens are encrypted at rest on the local filesystem. Acceptable for development and desktop agents (this is how most MCP hosts store tokens today) but lacks audit logging and centralized management.

**Environment variables** — Tokens in memory only, lost on restart. Acceptable for ephemeral agents that re-authenticate on each launch.

**Plaintext files or databases** — Unacceptable. Equivalent to the API key problem that OAuth is meant to solve.

### 6.4 Token Lifecycle Management

A well-designed agent token management system implements several patterns:

**Proactive refresh** — The agent refreshes its access token before it expires, not after a 401 response. This avoids failed requests and the retry complexity they introduce. A common approach is to refresh when 75% of the token's lifetime has elapsed.

**Retry with refresh** — If a request fails with 401 despite a valid-looking token (e.g., the token was revoked server-side), the agent attempts a single token refresh and retries the request. If the refresh also fails, the agent surfaces the authentication failure to the user.

**Graceful degradation** — If token refresh fails, the agent should not crash or silently continue with degraded capabilities. It should pause the current workflow, notify the user that re-authentication is required, and resume from the paused state once new tokens are obtained.

**Token scoping per tool** — An agent that accesses multiple MCP servers may hold multiple independent token sets. Each token set should be scoped to the minimum permissions required for that specific server, stored separately, and refreshed independently.

## 7. Scoped Permissions for AI Tools

### 7.1 Designing OAuth Scopes for Tool Capabilities

Scope design is one of the most impactful security decisions in an AI application's architecture. Too broad, and a compromised token grants access to everything. Too narrow, and the user is bombarded with consent prompts or the agent is unable to complete legitimate tasks.

The recommended approach for MCP servers is to define scopes that map directly to tool capabilities:

```
tools:list          — Enumerate available tools and their schemas
tools:execute       — Invoke any tool
tools:execute:query — Invoke only read-only query tools
tools:execute:write — Invoke tools that modify state
resources:read      — Read resource data
resources:subscribe — Subscribe to resource change notifications
prompts:list        — List available prompt templates
prompts:get         — Retrieve prompt template content
```

### 7.2 Fine-Grained vs. Coarse-Grained Scopes

Fine-grained scopes (e.g., `database:table:users:read`, `github:repo:myproject:pull_request:create`) provide precise control but create complexity: the consent screen becomes overwhelming, the agent must request a long list of scopes, and adding new tools requires defining new scopes.

Coarse-grained scopes (e.g., `database:full_access`, `github:all`) are simpler but violate the principle of least privilege. A token with `database:full_access` lets the agent drop tables when it only needed to run SELECT queries.

The pragmatic middle ground is a two-level hierarchy:

- **Resource-level scopes** define what category of resource can be accessed: `database`, `files`, `github`, `slack`.
- **Action-level qualifiers** define what operations are permitted: `:read`, `:write`, `:admin`.

This yields scopes like `database:read`, `files:write`, `github:read`, `github:write` — granular enough to enforce least privilege, coarse enough to be manageable.

### 7.3 Scope Escalation Risks

Scope escalation occurs when an agent obtains broader permissions than the user intended. This can happen in several ways:

**Transitive tool access** — An agent with `tools:execute` scope on an MCP server that exposes a "run SQL" tool effectively has full database access, even if it was not explicitly granted database scopes. The tool's capabilities implicitly expand the agent's effective permissions.

**Scope aggregation** — An agent that holds tokens for five different services may, in combination, have access patterns that no single service's scope model anticipated. For example, read access to a user database plus write access to an email service enables the agent to email every user.

**Dynamic scope inheritance** — If an MCP server adds new tools after the initial consent, the agent may gain access to those tools under existing scopes without additional user approval.

Mitigations include: requiring separate consent for each resource server, implementing tool-level authorization checks on the server side (beyond OAuth scopes), and conducting regular scope audits.

## 8. Delegated Authorization

### 8.1 The Principal Chain

When an AI agent acts on behalf of a user, there is a chain of delegation: the user delegates authority to the agent, the agent delegates to a tool, and the tool calls an API. Each link in this chain must be accountable:

```
User → Agent (MCP Host) → Tool (MCP Server) → Backend API
```

OAuth models this chain through the access token. The token was issued because the user (resource owner) granted consent to the agent (client) to access the tool server (resource server) with specific scopes. The tool server can inspect the token to determine who the original user is (via the `sub` claim), which agent is acting (via the `client_id` or `azp` claim), and what permissions were granted (via the `scope` claim).

### 8.2 Maintaining Accountability in the Chain

For the delegation chain to be auditable, several properties must hold:

**Token propagation, not credential sharing** — The agent passes its access token to the MCP server, which uses it to authorize the request. The agent never shares the user's original credentials with the tool server.

**Token exchange for downstream calls** — If the MCP server needs to call a downstream API on behalf of the user, it should use the OAuth 2.0 Token Exchange flow (RFC 8693) to obtain a new token scoped specifically for the downstream API, rather than forwarding the agent's token. This ensures that the downstream API receives a token with appropriate audience restrictions and scope limitations.

**Audit logging at each hop** — Each service in the chain logs the token's claims (subject, client ID, scopes, audience) alongside the action performed. This creates an audit trail that can reconstruct the full delegation path: "User Alice, acting through Claude Desktop (client_id: claude_desktop_abc), invoked the query_users tool (scope: database:read) on the HR database MCP server, which called the PostgreSQL API."

### 8.3 On-Behalf-Of (OBO) Flow

The On-Behalf-Of flow (a variation of token exchange) is particularly relevant for multi-tier AI architectures. When an MCP server receives a request with an access token, it can exchange that token for a new token that represents the original user but is scoped for a different downstream service:

```
POST /oauth/token
Content-Type: application/x-www-form-urlencoded

grant_type=urn:ietf:params:oauth:grant-type:token-exchange
&subject_token=<agent's_access_token>
&subject_token_type=urn:ietf:params:oauth:token-type:access_token
&requested_token_type=urn:ietf:params:oauth:token-type:access_token
&audience=https://downstream-api.example.com
&scope=data:read
```

The authorization server validates the original token, checks that the MCP server is authorized to perform token exchange, and issues a new token with the requested audience and scope. The downstream API receives a token that identifies the original user but is scoped exclusively for the downstream API's resources.

## 9. Machine-to-Machine Authentication

### 9.1 Client Credentials for Server-to-Server MCP

Not all MCP interactions involve a human user. Server-to-server MCP connections — where a backend agent framework calls an MCP server to execute tools as part of an automated pipeline — use the client credentials flow. The MCP client authenticates directly to the authorization server with its own credentials:

```python
token_response = requests.post(token_endpoint, data={
    'grant_type': 'client_credentials',
    'client_id': 'mcp-pipeline-agent',
    'client_secret': pipeline_client_secret,
    'scope': 'tools:execute resources:read',
    'audience': 'https://mcp-server.example.com',
})
```

The resulting token represents the client application itself, not any user. The MCP server authorizes requests based on the client's identity and the scopes assigned to it by the authorization server's configuration.

### 9.2 Service Accounts

Service accounts are pre-provisioned identities for automated systems. In the AI context, each agent deployment (production, staging, development) might have its own service account with different permissions:

- Production agent: `tools:execute`, `database:read`, `files:read` — read-heavy, limited write access.
- Staging agent: `tools:execute`, `database:read`, `database:write`, `files:write` — broader access for testing.
- Development agent: `admin:*` — full access for development, restricted to non-production resources.

Service accounts should be managed through the identity provider's API, not hardcoded in application configuration. This enables centralized rotation, auditing, and revocation.

### 9.3 Workload Identity

Cloud platforms provide workload identity mechanisms that eliminate the need for static client secrets entirely. Instead of configuring a client secret for an agent running on AWS, the agent uses its IAM role to obtain a short-lived assertion that the authorization server trusts:

**AWS** — The agent's EC2 instance or ECS task role provides a signed identity document. The authorization server validates this document against AWS's public keys and issues an OAuth token.

**GCP** — The agent's service account provides a Google-signed JWT. The authorization server exchanges this for an OAuth token.

**Azure** — Managed Identity provides a token from the Azure Instance Metadata Service. The authorization server validates it against Azure AD.

**Kubernetes** — Service account tokens (projected volumes) provide a signed JWT that the authorization server can validate against the cluster's OIDC discovery endpoint.

Workload identity is the most secure option for cloud-deployed agents because it eliminates static secrets entirely. The agent's identity is tied to its runtime environment, and credentials are rotated automatically.

## 10. JWT and Token Introspection

### 10.1 JWT Structure

JSON Web Tokens (JWTs, RFC 7519) are the most common format for OAuth access tokens in modern deployments. A JWT consists of three base64url-encoded segments separated by dots: header, payload, and signature.

The header specifies the signing algorithm:

```json
{
  "alg": "RS256",
  "typ": "JWT",
  "kid": "key-2026-04"
}
```

The payload contains claims — assertions about the token's subject, issuer, audience, and permissions:

```json
{
  "iss": "https://auth.example.com",
  "sub": "user_alice_123",
  "aud": "https://mcp-server.example.com",
  "client_id": "claude_desktop_abc",
  "scope": "tools:execute database:read",
  "iat": 1743580800,
  "exp": 1743584400,
  "jti": "token_unique_id_xyz"
}
```

The signature is computed over the header and payload using the specified algorithm and the authorization server's private key. The resource server validates the signature using the authorization server's public key (obtained from the JWKS endpoint).

### 10.2 Claims for AI Contexts

Standard JWT claims can be extended with custom claims that carry AI-specific context:

- `agent_type` — Identifies the type of agent (e.g., "coding_assistant", "data_pipeline", "customer_support").
- `session_id` — Links the token to a specific agent session, enabling session-level audit trails.
- `tool_restrictions` — An explicit list of tools the token is authorized to invoke, providing finer-grained control than scopes alone.
- `max_actions` — A budget limiting the number of tool invocations the token permits, preventing runaway agents.
- `delegation_chain` — An array recording the delegation path (e.g., ["user:alice", "agent:claude_desktop", "mcp:database_server"]).

Custom claims should be namespaced to avoid collisions (e.g., `https://myapp.example.com/agent_type`).

### 10.3 Token Validation

Resource servers (MCP servers) validate JWT access tokens by:

1. **Parsing** the JWT and extracting the header.
2. **Fetching the signing key** from the authorization server's JWKS (JSON Web Key Set) endpoint, caching it for performance.
3. **Verifying the signature** using the public key matching the `kid` in the header.
4. **Checking standard claims**: `exp` (not expired), `nbf` (not before), `iss` (expected issuer), `aud` (expected audience).
5. **Checking scopes**: The `scope` claim contains the permissions required for the requested operation.
6. **Checking custom claims**: Any additional authorization logic based on AI-specific claims.

This validation can be performed locally without contacting the authorization server, which is a significant performance advantage over opaque tokens that require introspection.

### 10.4 Token Introspection

For opaque (non-JWT) tokens, or when the resource server needs to check whether a JWT has been revoked, the OAuth 2.0 Token Introspection endpoint (RFC 7662) provides server-side validation:

```
POST /oauth/introspect
Content-Type: application/x-www-form-urlencoded

token=<access_token>
&token_type_hint=access_token
```

The authorization server responds with the token's current status and claims:

```json
{
  "active": true,
  "sub": "user_alice_123",
  "client_id": "claude_desktop_abc",
  "scope": "tools:execute database:read",
  "exp": 1743584400
}
```

If the token has been revoked (e.g., the user revoked the agent's access), the response is simply `{"active": false}`. Introspection adds a network round-trip per request, so it is typically used at session establishment or periodically, not on every API call.

### 10.5 Audience Restrictions

In multi-server agent architectures, audience restrictions are critical. An agent interacting with three MCP servers should obtain separate tokens for each, each with the appropriate audience:

- Token A: `aud: "https://database-mcp.example.com"`, scope: `database:read`
- Token B: `aud: "https://github-mcp.example.com"`, scope: `repos:write`
- Token C: `aud: "https://slack-mcp.example.com"`, scope: `messages:send`

If the agent accidentally sends Token A to the GitHub MCP server, it will be rejected because the audience does not match. This prevents token replay attacks across services.

## 11. Identity Providers for AI

### 11.1 Auth0

Auth0 provides a comprehensive OAuth 2.1 implementation with features directly relevant to AI applications:

- **Machine-to-Machine applications**: Pre-built support for client credentials flow with audience-scoped tokens.
- **Custom claims via Actions**: Auth0 Actions (serverless hooks) can inject AI-specific claims into tokens during the authentication pipeline.
- **Dynamic client registration**: Supported through the Management API, enabling automated MCP client registration.
- **Token exchange**: Supported for On-Behalf-Of flows in multi-tier agent architectures.

Configuration for an MCP server API in Auth0 involves creating an API resource with the MCP server's URL as the identifier, defining scopes that map to tool capabilities, and configuring the MCP host as a client application authorized to request those scopes.

### 11.2 Okta

Okta's Workforce Identity platform is particularly strong for enterprise AI deployments:

- **Inline Hooks**: Customize token claims based on the requesting agent's identity.
- **API Access Management**: Fine-grained scope definitions with default scope assignment per client.
- **Custom authorization servers**: Multiple authorization servers for different environments (production, staging, development agents).
- **OAuth for Okta APIs**: Service accounts for agent administration.

Okta's admin API also supports programmatic client registration, making it suitable for MCP's dynamic registration requirement.

### 11.3 Keycloak

Keycloak, the open-source identity provider, is popular for self-hosted AI deployments:

- **Realm-per-tenant**: Natural multi-tenant isolation for AI platforms serving multiple organizations.
- **Client Policies**: Define authorization policies that evaluate multiple attributes (agent type, time of day, source IP) beyond simple scope checks.
- **Token Exchange**: Built-in support for RFC 8693 token exchange, enabling the On-Behalf-Of patterns described in Section 8.
- **Lightweight and containerized**: Runs as a single container, suitable for development environments and edge deployments.

### 11.4 OIDC for Agent Identity

OpenID Connect (OIDC), built on top of OAuth 2.0, provides authentication (identity verification) in addition to authorization (permission grants). For AI applications, OIDC enables:

**Agent identity verification** — An agent can prove its identity to an MCP server by presenting an ID token (a JWT containing the agent's identity claims) alongside its access token. This allows the MCP server to make authorization decisions based on who the agent is, not just what permissions it has.

**Federated identity** — An agent deployed by Organization A can authenticate to an MCP server operated by Organization B using federated OIDC. Organization B trusts Organization A's identity provider, and the agent's identity claims are accepted across organizational boundaries.

**Identity propagation** — When a user authenticates to an MCP host via OIDC, the host receives an ID token containing the user's identity claims. These claims can be propagated through the agent-tool chain, ensuring that every service in the chain knows which user ultimately initiated the request.

## 12. Multi-Tenant AI Deployments

### 12.1 Per-Tenant Isolation

AI platforms that serve multiple organizations (tenants) must ensure strict isolation between tenants' data, configurations, and access tokens. OAuth provides the foundation for tenant isolation through several mechanisms:

**Tenant-specific authorization servers** — Each tenant has its own authorization server (or its own realm in Keycloak, tenant in Auth0, organization in Okta). Tokens issued for Tenant A are structurally incapable of granting access to Tenant B's resources because they are signed by different keys and have different issuers.

**Tenant-aware audience claims** — Even with a shared authorization server, the audience claim can encode tenant identity: `aud: "https://mcp.example.com/tenants/acme-corp"`. The MCP server validates that the token's tenant matches the requested resource's tenant.

**Tenant-scoped client registrations** — Each tenant's MCP clients are registered separately, with different client IDs, secrets, and authorized scopes. This prevents one tenant's agent from requesting scopes that belong to another tenant.

### 12.2 Tenant-Aware Token Scoping

Scopes can encode tenant context: `tenant:acme:database:read` vs. `tenant:globex:database:read`. This approach is explicit but creates a scope proliferation problem as the number of tenants grows. The alternative is to use a tenant claim in the token (`tenant_id: "acme"`) and let the resource server enforce tenant isolation based on the claim value rather than the scope string.

### 12.3 Data Residency Constraints

Some tenants require their data — including OAuth tokens and audit logs — to remain within specific geographic regions. This affects:

- **Authorization server deployment**: The identity provider must have endpoints in the required region.
- **Token storage**: The agent's token cache must reside in the correct region.
- **Token introspection**: Introspection requests must be routed to the regional authorization server.

Cloud identity providers generally support regional configurations. Self-hosted Keycloak instances can be deployed per-region with database replication respecting residency boundaries.

## 13. API Gateway Patterns

### 13.1 Token Validation at the Edge

API gateways (Kong, Envoy, AWS API Gateway, Azure API Management) can validate OAuth tokens before requests reach the MCP server. This offloads token validation from application code, provides a centralized enforcement point, and enables rate limiting and quota enforcement based on token claims:

```yaml
# Kong plugin configuration example
plugins:
  - name: openid-connect
    config:
      issuer: "https://auth.example.com/.well-known/openid-configuration"
      scopes_required: ["tools:execute"]
      audience_required: ["https://mcp-server.example.com"]
      consumer_claim: ["client_id"]
```

### 13.2 Rate Limiting and Quota Enforcement

Agents can be aggressive consumers of APIs — a single agent session might generate hundreds of tool calls in minutes. API gateways can enforce rate limits and quotas based on OAuth token claims:

- **Per-user rate limits**: Based on the `sub` claim, limiting how many requests any single user's agents can make.
- **Per-client rate limits**: Based on the `client_id` claim, limiting a specific agent application.
- **Per-scope rate limits**: Write operations (`tools:execute:write`) might have stricter limits than read operations (`tools:execute:read`).
- **Quota budgets**: A monthly token-based quota tracked by the gateway, with usage reported back to the billing system via the `sub` and `client_id` claims.

### 13.3 API Key to OAuth Migration

Organizations transitioning from API-key-based access to OAuth can use the API gateway as a migration bridge:

1. **Phase 1**: The gateway accepts both API keys and OAuth tokens. API keys are mapped to a synthetic OAuth token internally, with fixed scopes and no user identity.
2. **Phase 2**: New clients must use OAuth. Existing API key clients receive deprecation warnings in response headers.
3. **Phase 3**: API keys are disabled. All access is OAuth-based.

The gateway handles the mapping transparently — the MCP server only ever sees OAuth tokens, simplifying its authorization logic.

## 14. Security Pitfalls

### 14.1 Token Leakage Through Prompts

The most novel security risk in AI OAuth deployments is token leakage through the LLM's context window. If a tool response includes an access token in its output — whether in an error message, a debug log, or the tool's description — the LLM may include that token in its response to the user, store it in conversation history, or use it in subsequent tool calls in unexpected ways.

Mitigations:
- Never include tokens in tool descriptions or tool response content.
- Sanitize error messages returned to the LLM, stripping any Authorization headers or token values.
- Use structured tool responses that separate data from metadata, keeping authentication metadata outside the content that reaches the model.

### 14.2 Bearer Tokens in Tool Descriptions

Some MCP server implementations have been observed to include authentication instructions in tool descriptions: "When calling this API, include the header `Authorization: Bearer sk-abc123...`". This is catastrophic — the token is now part of the LLM's context, can be exfiltrated through prompt injection, and is visible to anyone with access to the tool listing.

The fix is architectural: authentication must be handled at the transport layer (HTTP headers managed by the MCP client), never at the prompt/tool-description layer. Tools should describe what they do, not how to authenticate.

### 14.3 Overly Broad Scopes

A common anti-pattern is requesting a blanket scope like `admin:*` or `tools:execute` when the agent only needs to read data. This violates the principle of least privilege and means that a compromised token grants full access rather than limited read-only access.

Every MCP server should define granular scopes, and every MCP client should request only the scopes it actually needs for the current task. If a long-running agent occasionally needs elevated permissions (e.g., write access for one step in a multi-step workflow), it should use incremental authorization — requesting additional scopes only when needed and releasing them afterward.

### 14.4 Implicit Flow Deprecation

OAuth 2.0's implicit flow returned access tokens directly in the URL fragment, without an authorization code exchange. This was intended for browser-based applications that could not securely store a client secret. OAuth 2.1 removes the implicit flow entirely because:

- Tokens in URL fragments are exposed to browser history, referrer headers, and log files.
- PKCE provides a better solution for public clients.
- The implicit flow lacks refresh token support, forcing clients to re-authenticate frequently.

Any AI application that was using the implicit flow (e.g., a browser-based agent frontend) must migrate to the authorization code flow with PKCE.

### 14.5 Refresh Token Theft

If an attacker obtains a refresh token, they can generate new access tokens indefinitely (or until the refresh token is rotated or revoked). Refresh tokens are long-lived by design, making them high-value targets. Protections include:

- **Refresh token rotation**: Each use invalidates the old token, so a stolen token can be used only once before the legitimate client's next refresh detects the theft.
- **Sender-constrained refresh tokens**: Binding the refresh token to the client's TLS certificate (mTLS) or DPoP key, so that the token is useless without the client's private key.
- **Short-lived refresh tokens for agents**: Limiting refresh token lifetime to the expected duration of the agent's task, rather than using indefinite refresh tokens.

### 14.6 Cross-Server Token Reuse

An agent holding tokens for multiple MCP servers might accidentally (or through prompt injection) send a token intended for Server A to Server B. If both servers accept the same audience, Server B receives valid authorization intended for Server A. Strict audience validation prevents this, but requires that each MCP server use a distinct audience identifier.

## 15. Practical Implementation

### 15.1 OAuth Setup for an MCP Server (Python)

The following example demonstrates implementing OAuth protection for an MCP server using Python with FastAPI and the Authlib library:

```python
from fastapi import FastAPI, Depends, HTTPException, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from authlib.jose import jwt, JsonWebKey
from authlib.integrations.requests_client import OAuth2Session
import httpx
import json
from functools import lru_cache

app = FastAPI()
security = HTTPBearer()

# Configuration
ISSUER = "https://auth.example.com"
AUDIENCE = "https://mcp-server.example.com"
JWKS_URI = f"{ISSUER}/.well-known/jwks.json"

# Cache JWKS keys
@lru_cache(maxsize=1)
def get_jwks():
    response = httpx.get(JWKS_URI)
    return JsonWebKey.import_key_set(response.json())

# Token validation dependency
async def validate_token(
    credentials: HTTPAuthorizationCredentials = Depends(security),
) -> dict:
    token = credentials.credentials
    try:
        jwks = get_jwks()
        claims = jwt.decode(token, jwks)
        claims.validate()

        # Validate issuer and audience
        if claims.get("iss") != ISSUER:
            raise HTTPException(status_code=401, detail="Invalid issuer")
        if claims.get("aud") != AUDIENCE:
            raise HTTPException(status_code=401, detail="Invalid audience")

        return dict(claims)
    except Exception as e:
        raise HTTPException(status_code=401, detail=f"Token validation failed: {e}")

# Scope checking helper
def require_scope(required: str):
    async def checker(claims: dict = Depends(validate_token)):
        scopes = claims.get("scope", "").split()
        if required not in scopes:
            raise HTTPException(
                status_code=403,
                detail=f"Insufficient scope: requires '{required}'",
            )
        return claims
    return checker

# OAuth metadata endpoint
@app.get("/.well-known/oauth-authorization-server")
async def oauth_metadata():
    return {
        "issuer": AUDIENCE,
        "authorization_endpoint": f"{ISSUER}/oauth/authorize",
        "token_endpoint": f"{ISSUER}/oauth/token",
        "registration_endpoint": f"{AUDIENCE}/oauth/register",
        "scopes_supported": [
            "tools:list", "tools:execute",
            "resources:read", "resources:subscribe",
        ],
        "response_types_supported": ["code"],
        "grant_types_supported": [
            "authorization_code", "client_credentials", "refresh_token",
        ],
        "code_challenge_methods_supported": ["S256"],
        "token_endpoint_auth_methods_supported": ["client_secret_post", "none"],
    }

# Dynamic client registration endpoint
registered_clients = {}

@app.post("/oauth/register")
async def register_client(request: Request):
    body = await request.json()
    import secrets
    client_id = f"dyn_{secrets.token_hex(16)}"
    registered_clients[client_id] = {
        "client_id": client_id,
        "client_name": body.get("client_name", "Unknown"),
        "redirect_uris": body.get("redirect_uris", []),
        "grant_types": body.get("grant_types", ["authorization_code"]),
        "token_endpoint_auth_method": body.get(
            "token_endpoint_auth_method", "none"
        ),
    }
    return registered_clients[client_id]

# MCP tool listing (requires tools:list scope)
@app.get("/mcp/tools/list")
async def list_tools(claims: dict = Depends(require_scope("tools:list"))):
    return {
        "tools": [
            {
                "name": "query_database",
                "description": "Execute a read-only SQL query",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "SQL query"},
                    },
                    "required": ["query"],
                },
            },
        ],
    }

# MCP tool execution (requires tools:execute scope)
@app.post("/mcp/tools/call")
async def call_tool(
    request: Request,
    claims: dict = Depends(require_scope("tools:execute")),
):
    body = await request.json()
    tool_name = body.get("name")
    arguments = body.get("arguments", {})

    # Audit log
    print(
        f"Tool call: {tool_name} by user={claims.get('sub')} "
        f"client={claims.get('client_id')} scope={claims.get('scope')}"
    )

    if tool_name == "query_database":
        # Execute query (simplified)
        result = execute_readonly_query(arguments["query"])
        return {"content": [{"type": "text", "text": json.dumps(result)}]}

    raise HTTPException(status_code=404, detail=f"Unknown tool: {tool_name}")
```

### 15.2 OAuth Setup for an MCP Server (TypeScript)

The equivalent implementation in TypeScript using Express and the jose library:

```typescript
import express from 'express';
import { createRemoteJWKSet, jwtVerify, JWTPayload } from 'jose';
import crypto from 'crypto';

const app = express();
app.use(express.json());

// Configuration
const ISSUER = 'https://auth.example.com';
const AUDIENCE = 'https://mcp-server.example.com';
const JWKS_URI = `${ISSUER}/.well-known/jwks.json`;

// Create JWKS fetcher (caches automatically)
const jwks = createRemoteJWKSet(new URL(JWKS_URI));

// Token validation middleware
interface AuthenticatedRequest extends express.Request {
  claims?: JWTPayload & { scope?: string; client_id?: string };
}

async function validateToken(
  req: AuthenticatedRequest,
  res: express.Response,
  next: express.NextFunction,
): Promise<void> {
  const authHeader = req.headers.authorization;
  if (!authHeader?.startsWith('Bearer ')) {
    res.status(401).json({ error: 'Missing bearer token' });
    return;
  }

  const token = authHeader.slice(7);
  try {
    const { payload } = await jwtVerify(token, jwks, {
      issuer: ISSUER,
      audience: AUDIENCE,
    });
    req.claims = payload as AuthenticatedRequest['claims'];
    next();
  } catch (err) {
    res.status(401).json({ error: `Token validation failed: ${err}` });
  }
}

// Scope checking middleware factory
function requireScope(required: string) {
  return (
    req: AuthenticatedRequest,
    res: express.Response,
    next: express.NextFunction,
  ) => {
    const scopes = req.claims?.scope?.split(' ') ?? [];
    if (!scopes.includes(required)) {
      res.status(403).json({
        error: `Insufficient scope: requires '${required}'`,
      });
      return;
    }
    next();
  };
}

// OAuth metadata endpoint
app.get('/.well-known/oauth-authorization-server', (_req, res) => {
  res.json({
    issuer: AUDIENCE,
    authorization_endpoint: `${ISSUER}/oauth/authorize`,
    token_endpoint: `${ISSUER}/oauth/token`,
    registration_endpoint: `${AUDIENCE}/oauth/register`,
    scopes_supported: [
      'tools:list', 'tools:execute',
      'resources:read', 'resources:subscribe',
    ],
    response_types_supported: ['code'],
    grant_types_supported: [
      'authorization_code', 'client_credentials', 'refresh_token',
    ],
    code_challenge_methods_supported: ['S256'],
    token_endpoint_auth_methods_supported: ['client_secret_post', 'none'],
  });
});

// Dynamic client registration
const registeredClients = new Map<string, object>();

app.post('/oauth/register', (req, res) => {
  const clientId = `dyn_${crypto.randomBytes(16).toString('hex')}`;
  const client = {
    client_id: clientId,
    client_name: req.body.client_name ?? 'Unknown',
    redirect_uris: req.body.redirect_uris ?? [],
    grant_types: req.body.grant_types ?? ['authorization_code'],
    token_endpoint_auth_method:
      req.body.token_endpoint_auth_method ?? 'none',
  };
  registeredClients.set(clientId, client);
  res.status(201).json(client);
});

// MCP tool listing
app.get(
  '/mcp/tools/list',
  validateToken,
  requireScope('tools:list'),
  (_req, res) => {
    res.json({
      tools: [
        {
          name: 'query_database',
          description: 'Execute a read-only SQL query',
          inputSchema: {
            type: 'object',
            properties: {
              query: { type: 'string', description: 'SQL query' },
            },
            required: ['query'],
          },
        },
      ],
    });
  },
);

// MCP tool execution
app.post(
  '/mcp/tools/call',
  validateToken,
  requireScope('tools:execute'),
  async (req: AuthenticatedRequest, res) => {
    const { name, arguments: args } = req.body;

    // Audit log
    console.log(
      `Tool call: ${name} by user=${req.claims?.sub} ` +
      `client=${req.claims?.client_id} scope=${req.claims?.scope}`,
    );

    if (name === 'query_database') {
      const result = await executeReadonlyQuery(args.query);
      res.json({
        content: [{ type: 'text', text: JSON.stringify(result) }],
      });
      return;
    }

    res.status(404).json({ error: `Unknown tool: ${name}` });
  },
);

app.listen(3000, () => console.log('MCP server running on port 3000'));
```

### 15.3 Protecting a Tool Endpoint End-to-End

Bringing together the components, here is the complete flow for protecting an MCP tool endpoint:

1. **Server startup**: The MCP server starts and configures its OAuth metadata endpoint, pointing to the identity provider's authorization and token endpoints.

2. **Client discovery**: When an MCP host connects, it fetches the `.well-known/oauth-authorization-server` metadata to discover the OAuth endpoints.

3. **Client registration**: The MCP host registers itself via the dynamic registration endpoint, obtaining a client ID.

4. **User authentication**: The MCP host redirects the user to the authorization server's consent page, requesting specific scopes (e.g., `tools:list tools:execute`). PKCE parameters are included.

5. **Token acquisition**: After user consent, the MCP host exchanges the authorization code (with PKCE verifier) for an access token and refresh token.

6. **Authenticated requests**: The MCP host includes the access token in the Authorization header of all subsequent MCP requests.

7. **Server-side validation**: The MCP server validates the JWT (signature, expiry, issuer, audience, scopes) on each request.

8. **Token refresh**: When the access token approaches expiration, the MCP host uses the refresh token to obtain a new access token transparently.

9. **Audit trail**: Every tool invocation is logged with the token's claims, creating an auditable record of who did what, when, and through which agent.

## 16. Standards Landscape

### 16.1 OAuth 2.1 (RFC 9728 Draft)

OAuth 2.1 consolidates OAuth 2.0 (RFC 6749) with its accumulated best practice documents into a single specification. The key changes relevant to AI applications have been discussed throughout this report: mandatory PKCE, removal of the implicit flow, refresh token rotation, and strict redirect URI matching. As of April 2026, the specification is in advanced draft status with broad industry adoption of its requirements even before formal finalization.

### 16.2 DPoP: Demonstrating Proof of Possession

Demonstrating Proof of Possession (DPoP, RFC 9449) addresses a fundamental limitation of bearer tokens: anyone who possesses the token can use it. DPoP binds an access token to a specific client by requiring the client to prove possession of a private key on each request.

The flow works as follows:

1. The client generates an asymmetric key pair (typically ECDSA P-256).
2. When requesting a token, the client includes a DPoP proof — a JWT signed with the private key, containing the HTTP method, URL, and a timestamp.
3. The authorization server issues a DPoP-bound access token (indicated by the `token_type: "DPoP"` response).
4. On each API request, the client includes both the access token and a fresh DPoP proof.
5. The resource server validates that the DPoP proof was signed by the same key that was used when the token was issued.

For AI agents, DPoP is particularly valuable because it prevents token theft from being useful. Even if an attacker extracts a DPoP-bound token from an agent's memory or logs, they cannot use it without the agent's private key. The private key never leaves the agent's process, and ideally is stored in a hardware security module or TPM.

### 16.3 RAR: Rich Authorization Requests

Rich Authorization Requests (RAR, RFC 9396) extend OAuth's scope mechanism with structured authorization details. Instead of simple string scopes like `database:read`, RAR allows the client to request detailed, structured permissions:

```json
{
  "type": "mcp_tool_access",
  "locations": ["https://mcp-server.example.com"],
  "tools": ["query_database", "list_tables"],
  "constraints": {
    "max_rows": 1000,
    "allowed_tables": ["users", "products"],
    "denied_operations": ["DROP", "DELETE", "UPDATE"]
  }
}
```

This is dramatically more expressive than flat scopes. The authorization server can present a detailed consent screen ("This agent wants to query the users and products tables, reading up to 1000 rows, with no write operations"), and the resource server can enforce these constraints precisely.

RAR is especially promising for AI applications because tool capabilities are inherently structured — they have input schemas, output formats, and operational constraints that cannot be captured by simple scope strings.

### 16.4 GNAP: Grant Negotiation and Authorization Protocol

The Grant Negotiation and Authorization Protocol (GNAP, RFC 9635) is a next-generation authorization protocol designed to address limitations in OAuth 2.0 that are not fully resolved by OAuth 2.1. Key GNAP features relevant to AI:

**Instance-based client identity** — GNAP clients identify themselves by their signing key rather than a pre-registered client ID. This eliminates the need for dynamic client registration, as the client's identity is self-asserted and verified cryptographically.

**Request-based interaction** — GNAP supports multiple interaction modes (redirect, user code, push notification) negotiated per-request, rather than a fixed flow. An AI agent could request a "push notification" interaction where the user approves the request on their phone, rather than requiring a browser redirect.

**Ongoing access management** — GNAP's continuation mechanism allows the client to modify its access requests over time, adding or removing permissions without starting a new authorization flow. This maps naturally to long-running agent sessions where the required permissions evolve as the task progresses.

**First-class delegation** — GNAP has built-in support for delegation chains, where each link in the chain is explicit and verifiable. This is more natural for the user-agent-tool-API chain than OAuth's token exchange approach.

GNAP is less mature than OAuth 2.1 and has limited identity provider support as of April 2026, but it represents the likely direction for authorization protocols in agent-first architectures.

## 17. Conclusion

The transition from API keys to OAuth for AI applications is not a matter of preference — it is a necessity driven by the unique security challenges that autonomous agents introduce. API keys provide no delegation model, no scope control, no audit trail, and no revocation mechanism adequate for systems that act on behalf of users across multiple services over extended periods.

OAuth 2.1 provides the complete authorization framework that AI applications need. Its authorization code flow with PKCE enables secure delegated access. Its scoping mechanism allows fine-grained permission control. Its token lifecycle (short-lived access tokens, rotatable refresh tokens) limits the blast radius of compromises. And its ecosystem of identity providers, API gateways, and security tooling means that practitioners can implement production-grade authorization without building from scratch.

MCP's adoption of OAuth as its authentication layer for remote servers has made this transition concrete and immediate. Every MCP server that exposes tools over HTTP must implement OAuth, and every MCP host must be an OAuth client. This creates a virtuous cycle: as more MCP servers adopt OAuth, the tooling and documentation improve, making it easier for the next server to follow.

The standards landscape continues to evolve. DPoP addresses token theft. RAR provides the structured authorization that tool-based AI applications need. GNAP offers a vision of authorization protocols designed from the ground up for agent architectures. Practitioners building AI systems today should implement OAuth 2.1 with PKCE as the baseline, adopt DPoP for high-security deployments, and monitor GNAP's maturation for future adoption.

The fundamental principle remains simple: an AI agent should have exactly the permissions it needs, for exactly as long as it needs them, with a complete audit trail of everything it did with those permissions. OAuth 2.1 is the framework that makes this possible.
