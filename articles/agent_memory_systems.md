# Memory Systems for LLM Agents

*April 2026*

## 1. Introduction

A language model without memory is an amnesiac savant. It can reason brilliantly about the information in its current context window, but it forgets everything the moment the conversation ends. It cannot remember what the user discussed yesterday, what it learned from a previous task, or what preferences the user has expressed over time. Each interaction begins from zero.

This limitation is not merely inconvenient. It is a fundamental barrier to building agents that can maintain long-term relationships with users, accumulate knowledge over time, learn from their mistakes, and handle tasks that span multiple sessions. A personal assistant that cannot remember the user's name, a coding agent that cannot remember the architecture decisions made last week, a research agent that cannot build on its previous findings—these are not useful agents, regardless of how capable the underlying model is.

Memory systems for LLM agents address this gap by providing mechanisms to store, retrieve, and manage information across the agent's context window boundary. These systems range from simple conversation buffers to sophisticated architectures that mimic human memory systems, with working memory, episodic memory, semantic memory, and consolidation processes.

This report provides a comprehensive technical examination of memory systems for LLM agents. It covers the taxonomy of memory types, the mechanisms for storing and retrieving memories, the practical implementations in popular frameworks, and the design considerations for building agents with effective memory. The intended audience is engineers building memory-augmented agents and researchers studying how to give LLMs persistent, useful memory.

## 2. Why Agents Need Memory Beyond Context

### 2.1 The Context Window Constraint

Modern LLMs have context windows ranging from 8,000 to over 1,000,000 tokens. This might seem sufficient for most tasks, but in practice, the context window is consumed by multiple competing demands: the system prompt (instructions, tool definitions, persona), the conversation history, tool results, retrieved documents, and the model's own reasoning. A coding agent with 20 tool definitions, a detailed system prompt, and a conversation history of a few dozen exchanges can easily fill a 128K context window.

Moreover, even models with very large context windows do not attend uniformly to all content. Research has consistently shown that information in the middle of long contexts is retrieved less reliably than information at the beginning or end—the "lost in the middle" phenomenon. Simply stuffing everything into the context and hoping the model will find the relevant bits is not a reliable strategy.

### 2.2 The Multi-Session Problem

Many agent interactions span multiple sessions. A user works with a coding agent throughout a development project, returning to the agent over days or weeks. Each session begins a new conversation, and without memory, the agent has no knowledge of previous sessions. The user must re-explain their project, their preferences, and their progress every time.

Memory systems solve this by persisting information across sessions. When the user returns, the agent retrieves relevant memories from previous sessions and includes them in its context, providing continuity.

### 2.3 The Learning Problem

Agents that cannot learn from experience repeat the same mistakes. If an agent encounters an error—using the wrong API, misunderstanding a user's preference, generating code that fails a test—it should remember this experience and avoid the mistake in future interactions. Without memory, every interaction is the agent's first interaction, and it has no accumulated wisdom.

### 2.4 The Personalization Problem

Effective agents adapt to their users. They learn the user's communication style, technical preferences, project context, and domain terminology. This personalization requires memory—the agent must observe the user's behavior over time and store observations that inform future interactions.

## 3. Taxonomy of Memory Types

### 3.1 Overview

Memory systems for LLM agents draw heavily on the taxonomy of human memory from cognitive science. The major categories are:

- **Working memory**: The information currently in the model's context window. Actively used for the current task.
- **Short-term memory**: Recent conversation history and intermediate results. Persists for the duration of a session but is pruned or summarized to manage context size.
- **Long-term memory**: Information that persists across sessions. Stored in an external database and retrieved when relevant.
- **Episodic memory**: Memories of specific events and experiences. "Last Tuesday, the user asked me to refactor the authentication module, and I encountered a bug in the session management code."
- **Semantic memory**: General knowledge and facts. "The user's project uses Python 3.11, FastAPI, and PostgreSQL."
- **Procedural memory**: Knowledge of how to perform tasks. "When the user asks for a code review, they expect me to check for security issues, performance problems, and style violations."

These categories are not mutually exclusive. A single piece of information might be stored as episodic memory when it is first encountered ("the user told me they prefer type hints") and later consolidated into semantic memory ("the user prefers type hints").

### 3.2 Working Memory

Working memory in an LLM agent is the content of the model's current context window. It includes the system prompt, the current conversation history, any retrieved documents or memories, and the model's in-progress reasoning. Working memory is the only memory the model can directly access—everything else must be loaded into working memory before the model can use it.

The capacity of working memory is limited by the context window size and the model's ability to attend to content within that window. Effective memory management requires decisions about what to keep in working memory, what to move to short-term or long-term memory, and what to discard.

### 3.3 Short-Term Memory

Short-term memory holds information from the current session that is not currently in the context window but may be needed again. Common implementations include:

- **Full conversation history**: All messages from the current session, even those that have been pruned from the context window.
- **Intermediate results**: Tool outputs, intermediate calculations, and partial results that may be referenced later.
- **Session state**: The current task, the plan, the list of completed and pending steps.

Short-term memory is typically stored in memory (RAM) and is discarded when the session ends.

### 3.4 Long-Term Memory

Long-term memory persists across sessions and is stored in a durable external store (database, file system, vector store). It includes information that the agent has learned over time and that may be relevant to future interactions. Long-term memory is the primary mechanism for agent personalization and learning.

### 3.5 Episodic Memory

Episodic memory stores specific events and experiences. Each episodic memory is a record of what happened, when it happened, and in what context. Episodic memories are useful for:

- Recalling previous interactions with a user ("Last time you asked about X, we found that Y")
- Learning from mistakes ("The last time I tried this approach, it failed because Z")
- Providing continuity across sessions ("We were working on the authentication module when we left off last time")

```
episodic_memory = {
    "timestamp": "2026-03-28T14:30:00Z",
    "session_id": "sess_abc123",
    "event": "user_request",
    "summary": "User asked me to optimize the database query for the user 
                search endpoint. I found that the query was missing an index 
                on the email column. After adding the index, query time 
                dropped from 2.3s to 45ms.",
    "entities": ["database", "user_search", "email_index"],
    "outcome": "success",
    "lessons": "Always check for missing indexes when queries are slow."
}
```

### 3.6 Semantic Memory

Semantic memory stores general facts and knowledge, abstracted from specific events. It represents the agent's accumulated understanding of the user, the project, and the domain.

```
semantic_memory = {
    "category": "user_preferences",
    "facts": [
        "User prefers Python over JavaScript for backend development",
        "User uses pytest for testing with the pytest-asyncio plugin",
        "User's code style follows Black formatting with 88 char line length",
        "User prefers explicit error handling over broad exception catches"
    ],
    "confidence": 0.95,
    "last_updated": "2026-03-28T14:30:00Z",
    "source_episodes": ["ep_001", "ep_005", "ep_012"]
}
```

### 3.7 Procedural Memory

Procedural memory stores knowledge about how to perform tasks—workflows, strategies, and heuristics that the agent has found effective. This is distinct from the agent's general capabilities (which come from the model's training) and represents task-specific knowledge accumulated through experience.

```
procedural_memory = {
    "task": "code_review",
    "procedure": [
        "1. Read the diff to understand the scope of changes",
        "2. Check for security issues (SQL injection, XSS, auth bypasses)",
        "3. Check for performance issues (N+1 queries, missing pagination)",
        "4. Check for style consistency with the existing codebase",
        "5. Verify test coverage for new functionality",
        "6. Note: User prefers inline comments over summary-only reviews"
    ],
    "effectiveness": 0.92,
    "last_used": "2026-03-27T10:00:00Z"
}
```

## 4. Two-Tier Memory Architecture

### 4.1 The Core Idea

The most practical memory architecture for production LLM agents is a two-tier system that separates a small, fast **hot tier** from a large, persistent **cold tier**. The hot tier is the agent's working context—information that is immediately available for the current task. The cold tier is an external store—a vector database, knowledge graph, or structured file system—that holds the agent's full accumulated memory. The agent's memory management system moves information between tiers based on relevance, recency, and the current task.

This mirrors how operating systems manage memory hierarchies (CPU cache → RAM → disk) and how human cognition works (working memory → long-term memory). The key insight is that an agent rarely needs all of its memories at once. Most memories are irrelevant to the current task. The two-tier system keeps the context window focused on what matters right now, while ensuring that any memory can be retrieved when needed.

### 4.2 Hot Tier: The Active Context

The hot tier consists of everything currently loaded into the model's context window for the active turn. This includes:

- **System prompt and instructions**: Fixed configuration that defines the agent's behaviour.
- **Active memories**: A curated subset of long-term memories that are relevant to the current task, retrieved from the cold tier at the start of the session or on demand.
- **Session state**: The current conversation history (possibly summarised), intermediate results, and the current plan or task list.
- **Scratchpad**: A mutable working area where the agent can store intermediate reasoning, notes, and temporary data that it needs for the current multi-step task but that should not be persisted long-term.

The hot tier has a strict budget, typically expressed in tokens. A common allocation for a 128K context model might be:

| Component | Budget |
|---|---|
| System prompt + tool definitions | 5,000–15,000 tokens |
| Active memories (retrieved from cold tier) | 2,000–8,000 tokens |
| Conversation history (recent or summarised) | 10,000–40,000 tokens |
| Tool results and retrieved documents | 20,000–50,000 tokens |
| Reserved for model generation | 10,000–30,000 tokens |

Managing this budget is the central challenge of agent memory. Every token spent on one component is unavailable for another. Effective memory management means loading the right information at the right time and evicting information that is no longer needed.

### 4.3 Cold Tier: The Persistent Store

The cold tier holds the agent's full memory—every fact learned, every experience recorded, every preference observed—in a durable external store. The cold tier is never loaded in its entirety into the context. Instead, the agent queries the cold tier and retrieves specific memories that are relevant to the current situation.

Cold tier implementations vary:

- **Vector database** (Pinecone, Weaviate, Qdrant, Chroma): Memories are stored as text chunks with embeddings. Retrieval is by semantic similarity to the current query. Good for "find memories related to X" queries.
- **Relational database** (PostgreSQL with pgvector): Combines structured queries (filter by user, timestamp, type) with vector similarity. Good for systems that need both structured filtering and semantic retrieval.
- **Knowledge graph** (Neo4j, or in-memory graph): Memories are stored as entities and relationships. Good for "what do I know about X's relationship to Y" queries.
- **File system** (Markdown files, JSON): Simple, human-readable, version-controllable. Claude Code's CLAUDE.md and memory directory are examples. Good for systems where memory should be inspectable and editable by humans.

The choice of cold tier depends on the application's requirements for scale, query complexity, and human inspectability. Many production systems combine multiple backends—a vector store for semantic retrieval alongside a relational database for structured metadata.

### 4.4 The Retrieval Bridge

The mechanism that moves information from the cold tier to the hot tier is the retrieval bridge. At its simplest, this is a vector similarity search run at the start of each turn. More sophisticated implementations use multiple retrieval strategies:

**Query-time retrieval.** Before each model call, the system formulates a retrieval query (often from the last user message or the current task description), searches the cold tier, and injects the top-k results into the context. This is the standard RAG pattern applied to agent memory.

**Proactive retrieval.** The system predicts what memories might be useful based on the conversation trajectory, not just the latest message. If the user mentions "the authentication bug," the system retrieves memories about authentication, the specific bug, and the fix that was applied, even if the user has not explicitly asked for this context.

**Model-directed retrieval.** The agent itself decides when to query memory. Using a tool like `search_memory(query)`, the model formulates its own retrieval queries during reasoning. This is the MemGPT approach—the model actively manages its own memory, pulling information from the cold tier when it determines that it needs more context.

**Continuous background retrieval.** A background process monitors the conversation and continuously updates the set of active memories in the hot tier, adding newly relevant memories and evicting those that are no longer pertinent. This keeps the hot tier fresh without requiring explicit retrieval calls.

### 4.5 Writeback: Hot Tier to Cold Tier

Memory is not only read from the cold tier; it is also written back. During a conversation, the agent learns new information—user preferences, project facts, task outcomes, corrections. This information starts in the hot tier (the conversation context) and must be persisted to the cold tier for use in future sessions.

Writeback can be:

- **Explicit**: The model calls a `save_memory(content, type)` tool to persist a specific piece of information. This gives the model control over what to remember but depends on the model's judgment about what is worth saving.
- **Automatic**: A background process extracts salient information from the conversation at session end (or periodically during the session) and writes it to the cold tier. This ensures nothing important is lost but may store irrelevant information.
- **Hybrid**: The model explicitly saves important memories, and an automatic process catches anything the model missed. This is the most robust approach but requires deduplication logic.

The quality of writeback directly determines the quality of future retrieval. Poorly written memories (too vague, too verbose, missing context, wrong facts) degrade future interactions.

## 5. Episodic Memory in Depth

### 5.1 What Episodic Memory Captures

Episodic memory records the agent's experiences as discrete events, each grounded in a specific time and context. Unlike semantic memory (which stores abstracted facts), episodic memory preserves the narrative structure of what happened: the sequence of events, the causal relationships, the context in which decisions were made, and the outcomes of actions taken.

A well-structured episodic memory entry contains:

- **Temporal grounding**: When the event occurred (absolute timestamp and relative position within the session).
- **Context**: What was the user's goal? What task was the agent working on? What was the state of the project?
- **Event description**: What happened—the user's request, the agent's actions, any tool calls made, any errors encountered.
- **Outcome**: Did the action succeed or fail? Was the user satisfied? What was the result?
- **Causal links**: What caused this event? What did it cause? ("The user reported a bug → I investigated → found missing null check → fixed it → tests pass.")
- **Lessons learned**: What general principle can be derived from this experience? ("Always check for null values when handling optional API response fields.")

```
{
    "id": "ep_2026_0328_001",
    "timestamp": "2026-03-28T14:30:00Z",
    "session_id": "sess_abc123",
    "user_goal": "Optimize slow API endpoint",
    "context": {
        "project": "user-service",
        "file": "src/api/search.py",
        "prior_discussion": "User mentioned search was timing out in production"
    },
    "events": [
        {"action": "read_file", "target": "src/api/search.py", "result": "Found N+1 query pattern"},
        {"action": "read_file", "target": "src/models/user.py", "result": "No index on email column"},
        {"action": "edit_file", "target": "src/api/search.py", "change": "Added select_related() to queryset"},
        {"action": "edit_file", "target": "migrations/0042_add_email_index.py", "change": "Added database index"},
        {"action": "run_tests", "result": "All 47 tests pass"},
        {"action": "benchmark", "result": "Query time: 2.3s → 45ms"}
    ],
    "outcome": "success",
    "user_feedback": "positive",
    "lessons": [
        "Check for N+1 queries when API endpoints are slow",
        "Missing database indexes are a common cause of slow queries in this project",
        "This project uses Django ORM; select_related/prefetch_related are the standard fixes"
    ],
    "entities": ["search_endpoint", "email_index", "N+1_query", "Django_ORM"],
    "related_episodes": ["ep_2026_0315_003"]
}
```

### 5.2 Event Boundary Detection

Not every model turn constitutes a distinct episode. A 50-turn conversation about debugging a single issue is one episode, not fifty. Effective episodic memory requires detecting event boundaries—the points where one coherent episode ends and another begins.

Boundary detection heuristics include:

- **Topic shift**: The user's request changes subject. ("OK, enough about the database. Let's work on the authentication flow.")
- **Task completion**: A task reaches a definitive endpoint (success or failure). ("Great, the tests pass. Let's move on.")
- **Time gap**: A significant pause in the conversation suggests a natural boundary.
- **Explicit markers**: The user explicitly starts a new task. ("New topic: I need to set up CI/CD.")

In practice, many systems use the LLM itself to detect boundaries. At session end (or periodically during long sessions), the model is asked to segment the conversation into distinct episodes and summarise each one. This produces cleaner episodic memories than storing raw conversation chunks.

### 5.3 Episodic Retrieval Strategies

Retrieving the right episodic memories is more nuanced than standard RAG. The retrieval query must consider:

- **Temporal relevance**: Recent episodes are generally more relevant than old ones, but not always. A bug fix from six months ago becomes highly relevant when the same bug recurs.
- **Situational similarity**: Episodes that occurred in similar contexts (same file, same type of task, same error pattern) are more useful than episodes that happen to share keywords.
- **Causal chains**: If the current situation was caused by a previous event, the causal predecessor should be retrieved. ("This error is caused by the migration we ran last week.")
- **Outcome filtering**: For learning purposes, both successes and failures are valuable. For providing context, successful resolutions of similar problems are most useful.

A composite retrieval score might weight these factors:

```
score = α × semantic_similarity(query, episode)
      + β × recency_decay(episode.timestamp)
      + γ × context_overlap(current_context, episode.context)
      + δ × outcome_relevance(episode.outcome, current_task_type)
```

### 5.4 Episodic Memory and Learning

The most valuable function of episodic memory is enabling the agent to learn from experience. When the agent encounters a situation similar to a previous one, it can retrieve the relevant episode and apply the lessons learned:

- **Avoiding repeated mistakes**: "The last time I tried to use `eval()` in this codebase, the user corrected me and said to use `ast.literal_eval()` instead."
- **Reusing successful strategies**: "When I optimised the search endpoint last month, adding an index solved the problem. Let me check if the same approach applies here."
- **Adapting to user preferences**: "The last three times I suggested adding comments, the user removed them. I should write self-documenting code instead of adding comments."

This experience-based learning is distinct from the model's training-time knowledge. It is specific to the user, the project, and the agent's own history of interactions.

## 6. Semantic Memory in Depth

### 6.1 From Episodes to Knowledge

While episodic memory stores what happened, semantic memory stores what the agent *knows*—abstracted facts, relationships, and principles that are not tied to any specific event. Semantic memory is the agent's accumulated knowledge base about the user, the project, the domain, and its own effective strategies.

The relationship between episodic and semantic memory is one of abstraction. Episodic memories are the raw material; semantic memories are the distilled knowledge. After multiple episodes where the user corrects the agent's use of semicolons in JavaScript, the episodic memories record each correction event, while the semantic memory records the abstracted fact: "This user's JavaScript style omits semicolons."

This abstraction process—sometimes called memory distillation or consolidation—can be:

- **Manual**: The agent explicitly extracts knowledge when instructed. ("Remember that I prefer tabs over spaces.")
- **Automatic**: A background process periodically reviews recent episodic memories and extracts recurring patterns, preferences, and facts.
- **Model-driven**: The agent recognises during a conversation that it has learned something worth persisting and stores it proactively.

### 6.2 Semantic Memory Schema Design

The structure of semantic memories significantly affects retrieval quality. A flat list of facts becomes unwieldy at scale. Effective schemas organise semantic knowledge into categories with metadata:

```
{
    "id": "sem_user_pref_012",
    "category": "user_preferences",
    "subcategory": "code_style",
    "fact": "User uses Black formatter with 88-character line length",
    "confidence": 0.98,
    "evidence_count": 7,
    "first_observed": "2026-02-15T10:00:00Z",
    "last_confirmed": "2026-03-28T14:30:00Z",
    "source_episodes": ["ep_001", "ep_005", "ep_012", "ep_019", "ep_023", "ep_031", "ep_042"],
    "supersedes": "sem_user_pref_008",
    "tags": ["python", "formatting", "black"]
}
```

Key schema design principles:

- **Categorise facts**: user preferences, project facts, domain knowledge, effective strategies. Categories enable filtered retrieval ("give me all project facts") rather than relying solely on semantic similarity.
- **Track confidence**: Facts observed many times across multiple sessions are high-confidence. A single mention is low-confidence. Retrieval can prioritise high-confidence facts.
- **Track provenance**: Link each fact back to the episodic memories that support it. This enables verification ("why do I believe this?") and allows the fact to be re-evaluated if source episodes are later found to be unreliable.
- **Track currency**: When was this fact last confirmed? Old facts may be outdated. The user may have switched from Django to FastAPI since the agent last checked.
- **Support supersession**: When a fact is updated, the old fact is not deleted but marked as superseded. This maintains audit trails and allows rollback if the update was incorrect.

### 6.3 Fact Lifecycle Management

Semantic facts are not static. They go through a lifecycle:

**Observation**: A new fact is observed in a conversation. "The user's project is called 'nexus' and uses PostgreSQL 16." This is stored as a low-confidence fact with a single source episode.

**Confirmation**: The same fact is observed again in a subsequent interaction. Confidence increases. The source episode list grows. After multiple confirmations, the fact is high-confidence.

**Update**: New information modifies the fact. "The user migrated from PostgreSQL 16 to 17." The old fact is superseded, and a new fact is created with the updated information. The update is itself an episodic event.

**Contradiction**: A new observation contradicts an existing fact. "The user said they use MySQL, but semantic memory says PostgreSQL." The system must resolve the contradiction—typically by trusting the more recent observation, but flagging the conflict for the agent's attention.

**Decay**: A fact that has not been confirmed or accessed for a long time decays in confidence. It is not deleted but is deprioritised in retrieval. If the user has not mentioned Django in six months and recent conversations reference FastAPI, the Django fact should decay.

**Retirement**: A fact is explicitly invalidated, either by the user ("Forget that I use Django, we switched") or by the consolidation process determining that the fact is no longer supported by evidence.

### 6.4 Semantic Memory Retrieval

Retrieving semantic facts differs from retrieving episodic memories. Episodic retrieval seeks narratives; semantic retrieval seeks facts. The optimal retrieval strategy depends on the query type:

- **Direct fact lookup**: "What language does the user's project use?" → search by category (project_facts) and keywords (language, programming).
- **Preference queries**: "How does the user like their code formatted?" → search by category (user_preferences) and subcategory (code_style).
- **Contextual enrichment**: The system automatically loads relevant semantic facts based on the current task context. If the agent is about to write Python code, load all semantic memories categorised under Python, code_style, and project conventions.
- **Contradiction checking**: Before the agent states a fact, it checks semantic memory to ensure consistency. If the agent is about to suggest using Django, it should first check whether the project still uses Django.

In practice, semantic retrieval often works best as a combination of categorical filtering (narrow by type and tags) followed by semantic similarity ranking within the filtered set. This avoids the noise of searching the entire memory store with a single embedding query.

### 6.5 The Episodic-Semantic Bridge

The most sophisticated memory systems maintain an explicit bridge between episodic and semantic memory. Every semantic fact links back to the episodes that support it. Every episode can be queried for the semantic knowledge it contributed. This bidirectional linkage enables:

- **Evidence-based reasoning**: The agent can explain why it believes a fact. "I know your project uses FastAPI because you mentioned it on March 15 when we set up the API routes (ep_023) and again on March 22 when we added middleware (ep_031)."
- **Confidence calibration**: A semantic fact supported by many independent episodes is more trustworthy than one supported by a single episode. The evidence count directly informs retrieval ranking.
- **Cascading updates**: When an episodic memory is corrected or deleted (perhaps the user says "actually, that session was about a different project"), the semantic facts derived from it can be re-evaluated and potentially revised.
- **Active learning**: When the agent notices that a semantic fact has low evidence count or has not been confirmed recently, it can proactively ask the user. "I have on record that your project uses PostgreSQL 16—is that still accurate?"

## 7. Conversation Buffer and Sliding Window

### 4.1 Full Conversation Buffer

The simplest memory implementation is a full conversation buffer: every message in the conversation is stored and included in the context for every model call. This works well for short conversations but fails for long ones because the accumulated history eventually exceeds the context window.

```
class ConversationBufferMemory:
    def __init__(self):
        self.messages = []
    
    def add_message(self, role, content):
        self.messages.append({"role": role, "content": content})
    
    def get_context(self):
        return self.messages
```

**When to use**: Short conversations (fewer than 20-30 turns), tasks where all previous context is likely relevant, applications where the context window is much larger than the expected conversation length.

### 4.2 Sliding Window

The sliding window approach keeps only the most recent N messages in context, discarding older messages. This ensures that the context never exceeds a fixed size, but it loses historical context.

```
class SlidingWindowMemory:
    def __init__(self, window_size=20):
        self.messages = []
        self.window_size = window_size
    
    def add_message(self, role, content):
        self.messages.append({"role": role, "content": content})
    
    def get_context(self):
        return self.messages[-self.window_size:]
```

**When to use**: Long conversations where recent context is most relevant, chat applications where older messages are unlikely to be referenced, applications with tight context window budgets.

**Limitation**: Important information from early in the conversation (the user's name, the task description, key decisions) is lost when it falls outside the window. This can cause the agent to contradict earlier statements or forget the purpose of the conversation.

### 4.3 Token-Based Window

A refinement of the sliding window approach counts tokens rather than messages. This provides more precise control over context usage.

```
class TokenWindowMemory:
    def __init__(self, max_tokens=50000):
        self.messages = []
        self.max_tokens = max_tokens
    
    def add_message(self, role, content):
        self.messages.append({"role": role, "content": content})
    
    def get_context(self):
        context = []
        total_tokens = 0
        
        # Work backwards from most recent
        for message in reversed(self.messages):
            message_tokens = count_tokens(message["content"])
            if total_tokens + message_tokens > self.max_tokens:
                break
            context.insert(0, message)
            total_tokens += message_tokens
        
        return context
```

## 8. Summary Memory

### 5.1 Concept

Summary memory addresses the information loss of sliding window approaches by maintaining a running summary of the conversation. As older messages fall outside the window, they are summarized and the summary is included in the context. This preserves the essential information from earlier in the conversation without consuming the full token cost of the original messages.

### 5.2 Implementation

```
class SummaryMemory:
    def __init__(self, model, summary_threshold=30):
        self.messages = []
        self.summary = ""
        self.model = model
        self.summary_threshold = summary_threshold
        self.summarized_count = 0
    
    def add_message(self, role, content):
        self.messages.append({"role": role, "content": content})
        
        # Trigger summarization when unsummarized messages exceed threshold
        unsummarized = len(self.messages) - self.summarized_count
        if unsummarized >= self.summary_threshold:
            self._update_summary()
    
    def _update_summary(self):
        # Get messages to summarize
        new_messages = self.messages[self.summarized_count:]
        
        prompt = f"""Update the following conversation summary with 
        the new messages. Preserve all important information including 
        decisions, facts, user preferences, and task context.
        
        Current summary: {self.summary}
        
        New messages: {format_messages(new_messages)}
        
        Updated summary:"""
        
        self.summary = self.model.generate(prompt)
        self.summarized_count = len(self.messages)
    
    def get_context(self):
        # Include summary + recent unsummarized messages
        context = []
        if self.summary:
            context.append({
                "role": "system", 
                "content": f"Conversation summary so far: {self.summary}"
            })
        context.extend(self.messages[self.summarized_count:])
        return context
```

### 5.3 Summary Quality Challenges

The quality of the summary determines the quality of the memory. Common problems include:

**Information loss**: The summary may omit details that later turn out to be important. A summary that drops the specific error code from a debugging session makes it impossible to reference that error code later.

**Drift**: Successive summarizations can cause the summary to drift from the original content. Each summarization is lossy, and the cumulative loss over many iterations can be significant.

**Bias**: The model may emphasize recent or salient information at the expense of important but mundane details. User preferences and project decisions may be dropped in favor of more dramatic events.

**Cost**: Each summarization requires a model call, adding latency and cost. For long conversations with frequent summarization, this overhead can be significant.

### 5.4 Hybrid: Summary + Window

The most common practical approach combines summary memory with a sliding window. Older messages are summarized, while recent messages are kept in full. This provides the best of both approaches: the summary preserves historical context while the window preserves recent detail.

```
class SummaryWindowMemory:
    def __init__(self, model, window_size=10, summary_threshold=20):
        self.messages = []
        self.summary = ""
        self.model = model
        self.window_size = window_size
        self.summary_threshold = summary_threshold
    
    def get_context(self):
        context = []
        
        # Add summary of older messages
        if self.summary:
            context.append({
                "role": "system",
                "content": f"Summary of earlier conversation: {self.summary}"
            })
        
        # Add recent messages in full
        context.extend(self.messages[-self.window_size:])
        
        return context
```

## 9. RAG-Backed Long-Term Memory

### 6.1 Concept

Retrieval-Augmented Generation (RAG) provides a natural architecture for long-term memory. Past interactions, documents, and facts are stored in a database (typically a vector store) and retrieved when relevant to the current interaction. This allows the agent to access a virtually unlimited memory store while only including the most relevant memories in the current context.

### 6.2 Architecture

```
class RAGMemory:
    def __init__(self, vector_store, embedding_model, top_k=5):
        self.vector_store = vector_store
        self.embedding_model = embedding_model
        self.top_k = top_k
    
    def store(self, text, metadata=None):
        embedding = self.embedding_model.embed(text)
        self.vector_store.insert(
            text=text,
            embedding=embedding,
            metadata=metadata or {}
        )
    
    def retrieve(self, query, filters=None):
        query_embedding = self.embedding_model.embed(query)
        results = self.vector_store.search(
            embedding=query_embedding,
            top_k=self.top_k,
            filters=filters
        )
        return results
    
    def get_relevant_memories(self, current_message, user_id=None):
        filters = {"user_id": user_id} if user_id else None
        memories = self.retrieve(current_message, filters)
        return format_memories(memories)
```

### 6.3 What to Store

The design of what to store as long-term memories is critical. Storing every message verbatim creates a large, noisy store that is expensive to search and likely to return irrelevant results. Common strategies include:

**Store summaries of conversations.** After each session, generate a summary and store it as a memory. This is efficient but may lose important details.

**Store key facts and decisions.** Extract specific facts, decisions, and preferences from the conversation and store them individually. This enables precise retrieval but requires a reliable extraction process.

**Store user messages only.** The user's messages contain the highest-density information about their needs, preferences, and context. Storing only user messages (not assistant responses) reduces noise.

**Store tagged memories.** The model explicitly identifies information worth remembering and stores it with relevant tags. For example, "Remember: the user's production database is PostgreSQL 15 on AWS RDS."

### 6.4 Retrieval Strategies

**Semantic similarity.** Retrieve memories whose embeddings are most similar to the current query. This is the default RAG approach and works well when the current conversation is semantically related to past memories.

**Recency-weighted retrieval.** Weight search results by recency, so more recent memories are preferred over older ones. This helps when the user's context changes over time (e.g., they switched projects).

**Importance-weighted retrieval.** Weight search results by importance, where importance is a score assigned when the memory is created (based on the significance of the information). This ensures that important but old memories are not overshadowed by recent but trivial ones.

**Hybrid retrieval.** Combine semantic similarity, recency, and importance into a single score:

```
score = (alpha * semantic_similarity + 
         beta * recency_score + 
         gamma * importance_score)
```

Where alpha, beta, and gamma are tunable weights.

**Entity-based retrieval.** When the current conversation mentions specific entities (people, projects, tools), retrieve memories tagged with those entities regardless of semantic similarity.

## 10. Vector Store Memory

### 7.1 Architecture

Vector store memory is the most common implementation of RAG-backed long-term memory. Memories are embedded into vector representations and stored in a vector database. Retrieval is performed through approximate nearest neighbor (ANN) search.

### 7.2 Embedding Choices

The choice of embedding model significantly affects retrieval quality. Key considerations include:

**Dimensionality.** Higher-dimensional embeddings capture more nuance but require more storage and slower search. Common dimensions range from 256 to 3072.

**Domain specificity.** General-purpose embeddings (like OpenAI's `text-embedding-3-large` or Cohere's `embed-v3`) work well for most applications. Domain-specific embeddings (trained on code, legal documents, medical text) may perform better for specialized agents.

**Chunk size compatibility.** The embedding model should be matched to the chunk size of the stored memories. Models trained on short text (sentences) perform poorly on long documents, and vice versa.

### 7.3 Chunking Strategies

How memories are chunked before embedding affects retrieval quality:

**Fixed-size chunks.** Split text into chunks of a fixed token count (e.g., 256 tokens). Simple but may split important information across chunks.

**Semantic chunks.** Split text at semantic boundaries (paragraph breaks, topic changes). More complex but preserves the coherence of each chunk.

**Conversation-turn chunks.** Store each conversation turn as a separate chunk. Natural for conversation memory but may miss multi-turn context.

**Summary chunks.** Generate summaries of conversation segments and store the summaries rather than the raw text. Efficient but lossy.

### 7.4 Vector Database Options

Popular vector databases for agent memory include:

- **Pinecone**: Managed service, good scaling, metadata filtering
- **Weaviate**: Open-source, supports hybrid search (vector + keyword)
- **Chroma**: Lightweight, easy to embed in applications
- **Qdrant**: Open-source, strong filtering capabilities
- **pgvector**: PostgreSQL extension, useful when the application already uses PostgreSQL
- **FAISS**: Facebook's library, very fast for in-memory search, no built-in persistence

## 11. Knowledge Graph Memory

### 8.1 Concept

Vector stores excel at fuzzy semantic retrieval but struggle with structured relationships. If the agent needs to remember that "Alice manages the authentication team, which is responsible for the login service, which uses OAuth 2.0," a vector store might retrieve relevant fragments but not the complete chain of relationships.

Knowledge graph memory stores information as entities and relationships in a graph structure. This enables precise, structured queries that follow relationship chains—capabilities that vector stores do not provide.

### 8.2 Architecture

```
class KnowledgeGraphMemory:
    def __init__(self, graph_db):
        self.graph = graph_db
    
    def add_entity(self, name, entity_type, properties=None):
        self.graph.create_node(
            name=name,
            type=entity_type,
            properties=properties or {}
        )
    
    def add_relationship(self, entity1, relationship, entity2, properties=None):
        self.graph.create_edge(
            from_node=entity1,
            relationship=relationship,
            to_node=entity2,
            properties=properties or {}
        )
    
    def query(self, question, model):
        # Convert natural language question to graph query
        graph_query = model.generate(
            f"Convert this question to a graph query: {question}\n"
            f"Available entity types: {self.graph.get_types()}\n"
            f"Available relationship types: {self.graph.get_relationships()}"
        )
        
        results = self.graph.execute(graph_query)
        return results
    
    def extract_and_store(self, text, model):
        # Extract entities and relationships from text
        extraction_prompt = f"""Extract entities and relationships from 
        this text. Format as:
        ENTITY: name | type | properties
        RELATIONSHIP: entity1 | relationship | entity2
        
        Text: {text}"""
        
        extractions = model.generate(extraction_prompt)
        
        for entity in parse_entities(extractions):
            self.add_entity(**entity)
        for relationship in parse_relationships(extractions):
            self.add_relationship(**relationship)
```

### 8.3 Hybrid: Vector Store + Knowledge Graph

The most effective memory systems combine vector stores (for fuzzy semantic retrieval) with knowledge graphs (for structured relationship queries). The agent retrieves relevant memories from the vector store and relevant relationships from the knowledge graph, combining both into a rich context.

```
class HybridMemory:
    def __init__(self, vector_store, knowledge_graph, model):
        self.vector_store = vector_store
        self.knowledge_graph = knowledge_graph
        self.model = model
    
    def retrieve(self, query, user_id=None):
        # Retrieve from vector store
        semantic_results = self.vector_store.search(
            query=query, 
            filters={"user_id": user_id}
        )
        
        # Extract entities from query
        entities = self.model.extract_entities(query)
        
        # Retrieve related entities and relationships from graph
        graph_results = []
        for entity in entities:
            related = self.knowledge_graph.get_neighborhood(
                entity, depth=2
            )
            graph_results.extend(related)
        
        # Combine and format
        return format_combined_results(semantic_results, graph_results)
```

## 12. Memory Consolidation

### 9.1 Concept

Human memory undergoes consolidation—a process where short-term memories are transformed into long-term memories, with irrelevant details discarded and important patterns strengthened. LLM agent memory systems benefit from a similar process.

Without consolidation, the memory store grows indefinitely, retrieval quality degrades as noise accumulates, and contradictory or outdated memories coexist with current information. Consolidation addresses these problems by periodically reviewing, summarizing, and restructuring the memory store.

### 9.2 Consolidation Processes

**Deduplication.** Identify and merge duplicate or near-duplicate memories. If the agent has stored "User prefers Python" three times across different sessions, consolidate into a single memory with higher confidence.

**Conflict resolution.** Identify contradictory memories and resolve them. If one memory says "User's project uses Django" and a later memory says "User migrated to FastAPI," the consolidation process should update the semantic memory to reflect the current state.

**Summarization.** Summarize groups of related episodic memories into higher-level summaries. Ten individual memories about debugging database issues can be summarized into a semantic memory about the project's database challenges and solutions.

**Pruning.** Remove memories that are no longer relevant—old project details, outdated preferences, resolved issues. Pruning keeps the memory store focused and improves retrieval quality.

### 9.3 Implementation

```
class MemoryConsolidator:
    def __init__(self, memory_store, model):
        self.memory_store = memory_store
        self.model = model
    
    def consolidate(self):
        # Step 1: Find similar memories
        clusters = self.memory_store.cluster_similar(threshold=0.85)
        
        for cluster in clusters:
            if len(cluster) > 1:
                # Step 2: Merge duplicates
                merged = self.model.generate(
                    f"Merge these related memories into a single, "
                    f"comprehensive memory:\n{format_memories(cluster)}"
                )
                self.memory_store.replace(cluster, merged)
        
        # Step 3: Resolve conflicts
        all_memories = self.memory_store.get_all()
        conflicts = self.model.generate(
            f"Identify any contradictory information in these memories:\n"
            f"{format_memories(all_memories)}"
        )
        
        if conflicts:
            resolved = self.model.generate(
                f"Resolve these contradictions, keeping the most recent "
                f"information:\n{conflicts}"
            )
            self.memory_store.update_resolved(resolved)
        
        # Step 4: Prune old, low-relevance memories
        old_memories = self.memory_store.get_older_than(days=90)
        for memory in old_memories:
            if memory.access_count < 2 and memory.importance < 0.3:
                self.memory_store.delete(memory)
```

### 9.4 When to Consolidate

Consolidation can be triggered by:

- **Time**: Run consolidation daily, weekly, or on a schedule
- **Volume**: Run consolidation when the memory store exceeds a size threshold
- **Session boundaries**: Run consolidation at the end of each agent session
- **Retrieval quality**: Run consolidation when retrieval results show declining relevance

## 13. MemGPT and Virtual Context Management

### 10.1 The MemGPT Concept

MemGPT (Packer et al., 2023) introduced a paradigm shift in how LLM memory is managed. Rather than relying on fixed rules (sliding windows, periodic summarization) to manage what is in the model's context, MemGPT gives the model itself the ability to manage its own memory through explicit memory operations.

The analogy is to operating system virtual memory. Just as an OS manages the movement of data between fast RAM and slow disk, MemGPT manages the movement of information between the model's limited context window (analogous to RAM) and an external memory store (analogous to disk). The model can explicitly load information into its context, write information to external memory, and search external memory for relevant information.

### 10.2 Architecture

MemGPT's architecture has three memory tiers:

**Main context (working memory).** The model's current context window. This contains the system prompt, recent conversation messages, and any information the model has explicitly loaded from external memory. This is the only memory the model can directly reason about.

**Recall storage (conversation history).** A searchable store of all past conversation messages. The model can search this store by keyword or time range and load specific messages into main context.

**Archival storage (long-term memory).** A vector store for long-term information. The model can write arbitrary text to archival storage and search it semantically.

### 10.3 Memory Operations

MemGPT gives the model access to explicit memory management functions:

```
# Core memory operations
core_memory_append(section, content)   # Add to persistent context
core_memory_replace(section, old, new) # Update persistent context

# Conversation search
conversation_search(query, page=0)     # Search past messages

# Archival memory operations
archival_memory_insert(content)        # Store in long-term memory
archival_memory_search(query, page=0)  # Retrieve from long-term memory
```

The model decides when to use these operations as part of its normal reasoning process. If it needs information from a previous conversation, it searches recall storage. If it wants to remember something important, it writes to archival storage. If its core context needs updating (e.g., the user's preferences have changed), it uses core memory operations.

### 10.4 Self-Editing Memory

A distinctive feature of MemGPT is that the model can edit its own persistent context. The "core memory" section of the system prompt contains information about the user and the agent's self-description. The model can update this section—for example, adding a new user preference or updating a project description. This edited core memory persists across conversation turns and is always included in the context.

```
# Example: Model updates its core memory
core_memory_append("user", "Prefers concise responses with code examples")
core_memory_replace("user", 
    "Working on: unknown", 
    "Working on: E-commerce platform migration from Django to FastAPI"
)
```

### 10.5 Impact and Adoption

MemGPT demonstrated that LLMs can effectively manage their own memory when given the right tools. The approach has been adopted and adapted by several systems. Letta (the company founded by the MemGPT creators) has built a production platform around the concept. The core insight—that memory management should be a model capability rather than a system heuristic—has influenced the design of memory systems across the industry.

## 14. Practical Implementations

### 11.1 LangChain Memory Types

LangChain, the most widely used LLM application framework, provides several built-in memory types:

**ConversationBufferMemory.** Stores all messages. Simplest possible memory.

**ConversationBufferWindowMemory.** Keeps the last K messages. Simple sliding window.

**ConversationSummaryMemory.** Maintains a running summary of the conversation.

**ConversationSummaryBufferMemory.** Hybrid of summary and buffer. Summarizes old messages, keeps recent messages in full.

**ConversationTokenBufferMemory.** Token-based sliding window.

**ConversationEntityMemory.** Tracks entities mentioned in the conversation and stores facts about each entity. When an entity is mentioned, its stored facts are retrieved.

**VectorStoreRetrieverMemory.** RAG-based memory using a vector store. Past messages are embedded and stored; relevant ones are retrieved based on the current input.

### 11.2 LangGraph Memory and Checkpointing

LangGraph, LangChain's framework for building stateful agents, provides a persistence layer that automatically saves and restores agent state. The state includes:

- The current node in the agent's execution graph
- All state variables (conversation history, intermediate results, accumulated knowledge)
- Any custom state defined by the developer

This persistence enables long-running agents that can be interrupted and resumed, as well as multi-session agents that maintain state across user sessions. The persistence layer supports multiple backends (SQLite, PostgreSQL, custom stores).

### 11.3 Anthropic's Approach

Anthropic's Claude models support memory through two mechanisms:

**Projects with project knowledge.** Claude's Projects feature allows users to upload documents that persist across conversations. These documents are included in the model's context for every conversation within the project. This provides a form of semantic memory—the model always has access to the project's documentation, code, and context.

**Extended context.** Claude's large context windows (up to 200K tokens) reduce the need for external memory by allowing more conversation history to be maintained in context. For many use cases, a sufficiently large context window eliminates the need for separate memory management.

**Memory feature.** Claude's memory feature, introduced in 2025, allows the model to remember facts and preferences across conversations. The model can explicitly save memories and retrieve them in future conversations. This provides long-term memory without requiring the developer to implement a custom memory system.

### 11.4 OpenAI's Approach

OpenAI has implemented memory for ChatGPT through a built-in memory feature that allows the model to store and retrieve facts across conversations. The user can view, edit, and delete stored memories. Developers using the API can implement similar functionality through the Assistants API's thread and file storage capabilities.

## 15. Multi-Session Memory

### 12.1 The Multi-Session Challenge

Multi-session memory is the most challenging memory problem for LLM agents. The agent must:

1. Determine what information from the current session is worth remembering
2. Store that information in a way that enables future retrieval
3. In future sessions, determine what past memories are relevant to the current context
4. Include relevant memories without overwhelming the context window
5. Handle the evolution of information over time (user preferences change, projects evolve)

### 12.2 Session Boundary Processing

At the end of each session, the memory system should process the conversation to extract valuable information:

```
class SessionMemoryProcessor:
    def __init__(self, model, memory_store):
        self.model = model
        self.memory_store = memory_store
    
    def process_session(self, conversation, user_id, session_id):
        # Extract key information
        extraction_prompt = f"""Review this conversation and extract:
        1. Key facts learned about the user
        2. Decisions made during the conversation
        3. Unresolved issues or pending tasks
        4. User preferences observed
        5. Technical details worth remembering
        
        Conversation: {format_conversation(conversation)}"""
        
        extractions = self.model.generate(extraction_prompt)
        
        # Generate session summary
        summary_prompt = f"""Write a concise summary of this conversation 
        that would help a future AI assistant understand what was discussed 
        and accomplished.
        
        Conversation: {format_conversation(conversation)}"""
        
        summary = self.model.generate(summary_prompt)
        
        # Store memories
        self.memory_store.store_episodic(
            summary=summary,
            user_id=user_id,
            session_id=session_id,
            timestamp=now()
        )
        
        for fact in parse_facts(extractions):
            self.memory_store.store_semantic(
                fact=fact,
                user_id=user_id,
                source_session=session_id
            )
```

### 12.3 Session Start Retrieval

At the beginning of a new session, the memory system retrieves relevant memories to provide context:

```
class SessionStartRetriever:
    def __init__(self, memory_store, model):
        self.memory_store = memory_store
        self.model = model
    
    def get_session_context(self, user_id, initial_message=None):
        memories = []
        
        # Always include: user profile
        user_profile = self.memory_store.get_user_profile(user_id)
        if user_profile:
            memories.append(f"User profile: {user_profile}")
        
        # Always include: most recent session summary
        recent = self.memory_store.get_recent_sessions(user_id, n=1)
        if recent:
            memories.append(f"Last session: {recent[0].summary}")
        
        # If there's an initial message, retrieve relevant memories
        if initial_message:
            relevant = self.memory_store.search(
                query=initial_message,
                user_id=user_id,
                top_k=5
            )
            for memory in relevant:
                memories.append(f"Relevant memory: {memory.text}")
        
        return format_session_context(memories)
```

## 16. Privacy Considerations

### 13.1 The Privacy Challenge

Memory systems store personal information—user preferences, conversation contents, project details, potentially sensitive data. This creates privacy obligations that do not exist for stateless LLM interactions.

### 13.2 Data Minimization

Store only the minimum information needed for the agent's function. Avoid storing raw conversation transcripts when summaries would suffice. Avoid storing personally identifiable information (PII) when anonymized or abstracted information would serve the same purpose.

### 13.3 User Control

Users should have visibility and control over what the agent remembers:

- **View memories**: Users should be able to see what the agent has stored about them.
- **Edit memories**: Users should be able to correct inaccurate memories.
- **Delete memories**: Users should be able to delete specific memories or all memories.
- **Opt out**: Users should be able to disable memory entirely.

### 13.4 Retention Policies

Implement automatic retention policies that delete memories after a defined period. Episodic memories from specific conversations might be retained for 90 days. Semantic memories (user preferences) might be retained indefinitely but reviewed periodically. The retention period should be communicated to users and comply with applicable regulations.

### 13.5 Access Control

In multi-user or multi-agent systems, ensure that one user's memories are not accessible to another user or another user's agent. Implement strict access controls on the memory store, with user-level isolation.

### 13.6 Regulatory Compliance

Memory systems that store personal data must comply with relevant regulations:

- **GDPR** (EU): Right to access, right to rectification, right to erasure, data minimization, purpose limitation.
- **CCPA** (California): Right to know, right to delete, right to opt out of sale.
- **Other regulations**: Various jurisdictions have additional requirements for personal data storage and processing.

## 17. Advanced Topics

### 14.1 Memory-Augmented Reasoning

Memory retrieval can be integrated directly into the model's reasoning process, rather than being a preprocessing step. The model reasons about a problem, determines that it needs information from memory, formulates a memory query, retrieves results, and continues reasoning. This is the MemGPT approach applied to reasoning—the model actively manages what information it needs.

### 14.2 Collaborative Memory

In multi-agent systems, agents may share a memory pool. One agent's memories can inform another agent's reasoning. This requires careful management to prevent memory pollution (one agent storing incorrect information that misleads another) and to ensure that memories are attributed to their source.

### 14.3 Memory Distillation

Over time, episodic memories can be "distilled" into semantic memories—general knowledge abstracted from specific experiences. After five conversations about database optimization, the episodic memories ("On March 15, we optimized the user query by adding an index") can be distilled into semantic knowledge ("The user's database frequently benefits from index optimization; always check for missing indexes when queries are slow").

```
class MemoryDistiller:
    def __init__(self, model, memory_store):
        self.model = model
        self.memory_store = memory_store
    
    def distill(self, user_id, topic):
        # Retrieve all episodic memories related to a topic
        episodes = self.memory_store.search_episodic(
            user_id=user_id,
            topic=topic,
            min_count=3  # Only distill when we have enough episodes
        )
        
        if len(episodes) < 3:
            return  # Not enough data to distill
        
        # Generate semantic knowledge from episodes
        prompt = f"""Based on these experiences, what general knowledge 
        or patterns can be extracted?
        
        Experiences:
        {format_episodes(episodes)}
        
        Extract general principles, patterns, and knowledge:"""
        
        knowledge = self.model.generate(prompt)
        
        # Store as semantic memory
        self.memory_store.store_semantic(
            knowledge=knowledge,
            user_id=user_id,
            source_episodes=[e.id for e in episodes],
            confidence=calculate_confidence(len(episodes))
        )
```

### 14.4 Forgetting

Not all information should be remembered forever. Intentional forgetting is important for:

- **Relevance**: Old, unused memories clutter the store and reduce retrieval quality.
- **Accuracy**: Information changes over time, and old memories may be wrong.
- **Privacy**: Users may want certain information forgotten.
- **Cost**: Larger memory stores are more expensive to maintain and search.

A forgetting mechanism should consider:
- How recently the memory was accessed (unused memories are candidates for forgetting)
- How recently the memory was created (older memories are more likely to be outdated)
- The importance of the memory (important memories should be retained longer)
- User-explicit requests to forget

## 18. Design Considerations for Practitioners

### 15.1 Choosing a Memory Architecture

The choice of memory architecture depends on the application:

| Application Type | Recommended Memory | Reason |
|---|---|---|
| Simple chatbot | Sliding window | Low complexity, short conversations |
| Customer service | Summary + window | Need session context, but sessions are short |
| Personal assistant | RAG + knowledge graph | Need long-term personalization |
| Coding agent | Session state + vector store | Need project context and past solutions |
| Research agent | Vector store + episodic | Need to build on previous research |
| Multi-session agent | MemGPT-style | Need active memory management |

### 15.2 Memory Budget

Decide how much of the context window to allocate to memory. A common allocation:

- System prompt + tool definitions: 20-30%
- Retrieved memories: 10-20%
- Conversation history: 30-40%
- Model reasoning and response: 20-30%

These percentages vary by application and context window size, but the key principle is to budget explicitly rather than allowing any single category to dominate.

### 15.3 Evaluation

Evaluate memory systems on:

- **Retrieval relevance**: Are the retrieved memories relevant to the current context?
- **Coverage**: Does the memory system retain important information?
- **Freshness**: Are retrieved memories up-to-date?
- **Coherence**: Do retrieved memories present a consistent picture?
- **Impact on task performance**: Does memory improve the agent's output quality compared to a memoryless baseline?

### 15.4 Start Simple

Begin with the simplest memory that could work for your application. A sliding window with session summaries covers many use cases. Add vector store retrieval only when the agent demonstrably needs information from past sessions. Add knowledge graphs only when structured relationships are critical. Add MemGPT-style active memory management only when the simpler approaches fail to provide adequate context.

## 19. Conclusion

Memory systems transform LLM agents from stateless responders into persistent entities that learn, adapt, and maintain continuity over time. The taxonomy of memory types—working, short-term, long-term, episodic, semantic, and procedural—provides a framework for designing systems that balance completeness with efficiency, detail with abstraction, and persistence with privacy.

The field has progressed rapidly from simple conversation buffers to sophisticated architectures that combine vector stores, knowledge graphs, active memory management, and consolidation processes. MemGPT demonstrated that models can effectively manage their own memory. Production implementations from Anthropic, OpenAI, and framework providers like LangChain have made memory accessible to practitioners.

For engineers building memory-augmented agents, the key lessons are pragmatic. Start with the simplest memory that serves your use case. Budget your context window explicitly. Store the minimum information needed. Give users control over their data. Evaluate memory quality as rigorously as you evaluate the model's output. And remember that the goal of memory is not to store everything but to provide the right context at the right time—the information the agent needs to be helpful in this specific moment.
