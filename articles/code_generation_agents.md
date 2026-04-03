# Code Generation and Software Engineering Agents

*April 2026*

## 1. Introduction

The application of large language models to software engineering has progressed through three distinct phases. The first phase, beginning around 2021 with GitHub Copilot's launch, provided inline code completion—the model predicted the next few lines of code based on the current file context. The second phase, emerging in 2023-2024, introduced chat-based coding assistants that could answer questions about code, explain errors, and generate functions on request. The third phase, now well underway in 2026, deploys fully agentic coding systems that can autonomously navigate codebases, plan changes, edit multiple files, run tests, debug failures, and iterate until the task is complete.

This progression represents a qualitative shift in what LLMs can do for software engineering. A code completion tool saves keystrokes. A chat assistant answers questions. An agentic coding system does the work. It reads the issue, understands the codebase, writes the code, runs the tests, and submits the pull request. The human's role shifts from writing code to reviewing code—and increasingly, to deciding what code should be written in the first place.

This report provides a comprehensive technical examination of code generation and software engineering agents. It covers how modern coding agents work, the key systems in the field, the benchmarks used to evaluate them, the mechanisms they use to understand and modify code, and the limitations and risks they present. The intended audience is engineers who use or build coding agents and researchers working on improving their capabilities.

## 2. Evolution of Code Generation

### 2.1 The Completion Era (2021-2023)

GitHub Copilot, powered by OpenAI's Codex model, established the first commercially successful code generation product. Copilot operated as an IDE extension that predicted the next token (or tokens) based on the current file context. It could complete function bodies, generate boilerplate code, and suggest implementations based on comments or function signatures.

The completion paradigm had clear strengths and limitations. It excelled at generating code that followed obvious patterns—standard library usage, common algorithms, boilerplate. It struggled with code that required understanding beyond the current file—cross-file dependencies, project-specific conventions, architectural patterns. Most critically, it was passive: the developer decided what to write and where, and the model only predicted the next few tokens.

### 2.2 The Chat Era (2023-2024)

ChatGPT and Claude introduced conversational coding assistance. Developers could describe a task in natural language and receive complete implementations, explanations, or debugging help. This was more flexible than completion—the developer could ask "write a function that validates email addresses" and receive a complete, tested implementation—but the interaction was still single-turn or multi-turn within a conversation, and the model had no access to the developer's actual codebase.

The chat era also introduced the concept of code explanation and review. Developers could paste code into a chat and ask the model to find bugs, suggest improvements, or explain complex logic. This was valuable but limited by the developer's ability to identify which code to share and to manually integrate the model's suggestions.

### 2.3 The Agentic Era (2024-2026)

The agentic era began with systems that could access the developer's actual codebase and development tools. Rather than operating on code pasted into a chat window, these systems could read files, search codebases, run terminal commands, execute tests, and edit files directly. The model became a participant in the development workflow rather than a sidecar to it.

Key capabilities of agentic coding systems include:

- **Repository-level context**: Understanding the full codebase, not just the current file
- **Multi-file editing**: Making coordinated changes across multiple files
- **Test execution**: Running tests and using results to verify and debug changes
- **Terminal access**: Executing arbitrary commands to explore the environment
- **Iteration**: Trying an approach, evaluating the result, and revising if necessary
- **Autonomy**: Completing tasks with minimal human intervention

## 3. How Coding Agents Work

### 3.1 The Read-Plan-Edit-Test Loop

The core architecture of modern coding agents follows a loop:

1. **Read**: The agent explores the codebase to understand the relevant code. It reads files, searches for symbols, traces dependencies, and builds a mental model of the code it needs to modify.

2. **Plan**: Based on its understanding, the agent formulates a plan for the changes. The plan specifies which files to modify, what changes to make in each file, and the expected effect of the changes.

3. **Edit**: The agent modifies the code. This may involve editing existing files, creating new files, or deleting obsolete code.

4. **Test**: The agent runs the test suite (or relevant subset) to verify that the changes work correctly and do not break existing functionality.

5. **Debug**: If tests fail, the agent reads the error messages, identifies the cause, and revises its changes. This cycle repeats until the tests pass or the agent determines that it cannot resolve the issue.

```
function coding_agent(task, codebase, tools):
    # Phase 1: Understand
    relevant_files = search_codebase(task, codebase)
    context = read_files(relevant_files)
    understanding = model.analyze(task, context)
    
    # Phase 2: Plan
    plan = model.plan_changes(task, understanding)
    
    # Phase 3: Execute and iterate
    for attempt in range(MAX_ATTEMPTS):
        # Edit files
        for edit in plan.edits:
            apply_edit(edit.file, edit.changes)
        
        # Run tests
        test_result = run_tests(plan.test_commands)
        
        if test_result.all_passed:
            return success(plan.summary)
        
        # Debug and revise
        diagnosis = model.diagnose_failure(
            test_result.errors, 
            plan.edits, 
            context
        )
        plan = model.revise_plan(plan, diagnosis)
    
    return partial_success(plan.summary, remaining_failures)
```

### 3.2 Codebase Exploration

Before making changes, the agent must understand the relevant parts of the codebase. This exploration phase is critical—changes made without understanding the surrounding code are likely to be incorrect, inconsistent, or harmful.

Exploration strategies include:

**Keyword search.** Searching for relevant terms (function names, class names, error messages) across the codebase. This is fast but may miss relevant code that uses different terminology.

**Symbol-based navigation.** Using language server protocol (LSP) features or code intelligence tools to find symbol definitions, references, and implementations. "Find all references to the `authenticate` function" or "Go to the definition of the `User` class."

**Dependency tracing.** Following import chains and dependency relationships to understand how components interact. If the agent needs to modify a function, it should also understand what calls that function and what the function calls.

**File structure analysis.** Reading directory listings, configuration files, and project structure to understand the codebase's organization and conventions.

**Grep and search tools.** Using ripgrep, git grep, or similar tools to search for patterns across the codebase. This is particularly useful for finding usage patterns, configuration values, and related code.

### 3.3 Context Window Management

A large codebase contains far more code than any context window can hold. The agent must strategically select which files and code fragments to include in its context. This is one of the most challenging aspects of building coding agents.

Strategies for context management include:

**Relevant file selection.** Use the task description and codebase structure to identify the most relevant files. Include only the files that are directly related to the task.

**Partial file inclusion.** Include only the relevant portions of large files—the specific functions, classes, or sections that relate to the task, rather than the entire file.

**Progressive exploration.** Start with a broad overview (directory structure, file names) and progressively zoom into specific files and functions as the agent's understanding narrows.

**Context summarization.** For files that provide important context but are too large to include in full, generate summaries that capture the key information (function signatures, class hierarchies, important comments).

**Scratchpad.** Maintain a scratchpad of key findings that persists across exploration steps. The agent writes down important discoveries (the authentication function is in `auth/handler.py`, the database schema is defined in `models/user.py`) and includes the scratchpad in its context rather than re-reading the files.

## 4. Key Systems

### 4.1 GitHub Copilot

GitHub Copilot has evolved from a pure completion tool to a multi-modal coding assistant. As of 2026, Copilot operates in several modes:

**Inline completion.** The original mode: predicting the next tokens as the developer types. This remains the most frequently used mode and has been refined with faster inference, better context awareness, and multi-line completions.

**Copilot Chat.** A conversational interface within the IDE that can answer questions about code, explain errors, and generate code based on natural language descriptions. Chat mode has access to the current file, open files, and (with workspace indexing) the broader codebase.

**Copilot Workspace.** An agentic mode that can plan and implement multi-file changes based on a task description. The developer describes what they want to achieve, and Copilot generates a plan, implements the changes, and presents them for review.

**Copilot Agent Mode.** The latest iteration which works autonomously within VS Code, running terminal commands, reading and editing files, and iterating on changes based on test results. This mode operates as a full coding agent with the read-plan-edit-test loop.

### 4.2 Claude Code

Claude Code, developed by Anthropic, is a terminal-based coding agent that operates directly in the developer's command line. It has full access to the file system, can execute terminal commands, and uses Claude's extended thinking capabilities for complex reasoning about code.

Key characteristics:

**Terminal-native.** Claude Code runs in the terminal rather than in an IDE, giving it access to any tool the developer can use from the command line—git, build tools, test frameworks, linters, deployment scripts.

**Extended thinking.** Claude's extended thinking feature generates a detailed internal reasoning trace before producing actions. For complex coding tasks, this thinking trace includes analysis of the codebase, evaluation of different approaches, and planning of the change sequence. The thinking is visible to the developer, providing transparency into the agent's reasoning.

**Multi-file editing.** Claude Code can make coordinated changes across many files in a single task. It understands dependencies between files and ensures that changes are consistent.

**Git integration.** Claude Code can read git history, create branches, stage changes, and generate commit messages. It understands the project's git workflow and can operate within it.

**Tool extensibility via MCP.** Claude Code supports the Model Context Protocol (MCP), allowing developers to connect custom tools and data sources. An MCP server for a project management tool, for example, allows Claude Code to read issue descriptions and update ticket status as part of its workflow.

### 4.3 Cursor

Cursor is an AI-native IDE (based on VS Code) that integrates coding assistance throughout the development experience. Rather than being an add-on to an existing IDE, Cursor is built from the ground up around AI capabilities.

Key characteristics:

**AI-aware editor.** The editor understands which code the model is generating and which code the developer wrote, enabling features like AI-generated code highlighting, easy acceptance/rejection of suggestions, and tracked attribution.

**Multi-model support.** Cursor supports multiple model providers (OpenAI, Anthropic, Google, custom models), allowing developers to use different models for different tasks or to compare outputs.

**Codebase indexing.** Cursor indexes the entire codebase and uses this index to provide relevant context for AI operations. When the developer asks a question about the code, Cursor retrieves relevant files and code fragments automatically.

**Composer.** Cursor's Composer mode enables multi-file editing with AI assistance. The developer describes a change, and Composer plans and implements it across multiple files, presenting a diff for review.

**Agent mode.** Cursor's agent mode operates as a full coding agent within the IDE, with the ability to run terminal commands, read files, make edits, and iterate based on results.

### 4.4 Devin

Devin, developed by Cognition Labs, was introduced in early 2024 as the first "AI software engineer"—a fully autonomous coding agent designed to complete software engineering tasks with minimal human intervention.

Key characteristics:

**Full autonomy.** Devin is designed to work on tasks independently, from understanding the requirements to submitting the completed work. The human's role is to assign the task and review the result.

**Full development environment.** Devin operates in a complete development environment with a code editor, terminal, and web browser. It can install dependencies, configure environments, browse documentation, and debug issues.

**Multi-step planning.** Devin generates and follows multi-step plans for complex tasks. It can break down a feature request into component tasks, implement them in order, and verify the result.

**Learning from feedback.** When a human reviews Devin's work and provides feedback, Devin can revise its approach. Over time, it learns the project's conventions and the reviewer's preferences.

### 4.5 Aider

Aider is an open-source coding agent that operates in the terminal, similar to Claude Code. It emphasizes a chat-driven workflow where the developer describes changes in natural language and Aider implements them.

Key characteristics:

**Git-first workflow.** Every change Aider makes is automatically committed to git, making it easy to review, revert, or cherry-pick changes. The developer never loses work.

**Multi-model support.** Aider works with many model providers and has extensive benchmarks comparing model performance on coding tasks.

**Map-based context.** Aider generates a "repository map"—a compact representation of the codebase's structure (file names, function/class names, signatures) that fits in the context window and helps the model understand the codebase without reading every file.

**Diff-based editing.** Aider uses a diff-based editing format that is efficient and less error-prone than whole-file rewrites.

### 4.6 Other Notable Systems

**Amazon Q Developer** (formerly CodeWhisperer): AWS-integrated coding assistant with security scanning and AWS-specific knowledge.

**Windsurf (Codeium)**: AI IDE with "Cascade" agent mode that combines codebase understanding with agentic editing capabilities.

**OpenHands (formerly OpenDevin)**: Open-source platform for autonomous software development agents.

**SWE-Agent**: Research system from Princeton that pioneered many of the techniques used by modern coding agents, particularly the Agent-Computer Interface (ACI) design.

## 5. Code Understanding

### 5.1 Repository-Level Context

Understanding code at the repository level requires more than reading individual files. The agent must understand:

**Architecture.** How the application is structured—what components exist, how they communicate, what patterns are used (MVC, microservices, event-driven, etc.).

**Conventions.** The project's coding style, naming conventions, directory structure conventions, testing patterns, and error handling approaches.

**Dependencies.** What external libraries and frameworks the project uses, and how they are configured and used.

**Data models.** The structure of the data—database schemas, API contracts, type definitions, and the relationships between entities.

**Build and deployment.** How the project is built, tested, and deployed. What build tools, CI/CD pipelines, and deployment targets are used.

### 5.2 Techniques for Code Understanding

**Static analysis.** Parsing code to extract structure—ASTs, call graphs, dependency graphs, type information. This is precise but language-specific and requires tooling for each language.

**Repository mapping.** Generating a compact representation of the codebase's structure. Aider's repository map, for example, lists every file with its function/class definitions, providing a bird's-eye view that fits in the context window.

```
# Example repository map (simplified)
src/
  auth/
    handler.py
      - class AuthHandler
        - authenticate(request) -> User
        - validate_token(token) -> bool
      - class TokenManager
        - create_token(user) -> str
        - refresh_token(token) -> str
  models/
    user.py
      - class User(BaseModel)
        - id: int
        - email: str
        - hashed_password: str
      - class UserCreate(BaseModel)
        - email: str
        - password: str
```

**Semantic search.** Embedding code chunks and using vector similarity to find code related to a natural language query. "Find the code that handles user authentication" retrieves the relevant files even if the query terms do not exactly match the code.

**Language server integration.** Using language servers (LSP) to get precise information about symbols—definitions, references, type information, documentation. This is the most precise source of code structure information but requires a running language server.

### 5.3 The Challenge of Large Codebases

Production codebases can contain millions of lines of code across thousands of files. No context window can hold even a fraction of this. Coding agents must be highly selective about what code they include in context, and they must be able to navigate efficiently to the relevant code.

This is an area where agent architectures matter significantly. A naive agent that reads files randomly will waste its context window on irrelevant code. A sophisticated agent that starts with a repository map, uses search to find relevant areas, reads targeted sections, and maintains a scratchpad of findings will use its context much more efficiently.

## 6. Edit Mechanisms

### 6.1 Whole-File Replacement

The simplest edit mechanism: the model generates the complete new version of a file. The old file is replaced entirely with the model's output.

**Advantages**: Simple to implement. No parsing of edit instructions required. The model has full control over the file content.

**Disadvantages**: Expensive—the model must regenerate the entire file even for small changes. Error-prone for large files—the model may accidentally modify code it did not intend to change. Does not scale to large files (the model cannot generate a 5,000-line file reliably).

### 6.2 Diff-Based Editing

The model generates a diff that specifies only the changes, not the entire file. Common diff formats include:

**Search-and-replace.** The model specifies a block of text to find and a block of text to replace it with. This is the format used by Claude Code and Aider.

```
<<<<<<< SEARCH
def authenticate(request):
    token = request.headers.get("Authorization")
    if not token:
        raise AuthError("Missing token")
=======
def authenticate(request):
    token = request.headers.get("Authorization")
    if not token:
        raise AuthError("Missing authorization token")
    if not token.startswith("Bearer "):
        raise AuthError("Invalid token format")
>>>>>>> REPLACE
```

**Unified diff.** The model generates a standard unified diff with context lines, additions, and deletions. This is familiar to developers but harder for models to generate reliably (line numbers must be correct).

**Line-based edit.** The model specifies line numbers and operations (insert, delete, replace). This is precise but fragile—if line numbers are wrong (due to earlier edits), the edit is applied to the wrong location.

**Advantages of diff-based editing**: Efficient—only the changed code is generated. Clear about what is changing and what is not. Scales to files of any size.

**Disadvantages**: Parsing and applying diffs can fail if the search text does not exactly match the file content (due to whitespace, indentation, or encoding differences). The model must generate the search text accurately, which requires it to reproduce the existing code exactly.

### 6.3 AST-Based Editing

The model specifies edits in terms of the code's abstract syntax tree (AST)—"add a parameter `timeout` to the function `fetch_data`" or "wrap the body of `process_request` in a try-except block." The system translates these semantic edits into concrete code changes.

**Advantages**: Semantically precise. Does not require the model to reproduce existing code exactly. Robust to formatting differences.

**Disadvantages**: Complex to implement. Requires language-specific AST parsing and manipulation. Limited to edits that can be expressed as AST transformations.

### 6.4 Hybrid Approaches

Most production systems use hybrid approaches. Claude Code uses search-and-replace for precise edits and whole-file writes for new files. Cursor's Composer generates diffs for small changes and full file content for new files or major rewrites. The system chooses the most appropriate mechanism based on the size and nature of the change.

## 7. Benchmarks: SWE-bench and Beyond

### 7.1 SWE-bench

SWE-bench (Jimenez et al., 2023) is the most influential benchmark for coding agents. It consists of real GitHub issues paired with the pull requests that resolved them. The task is to take the issue description and the codebase at the time of the issue and produce a patch that resolves the issue. The patch is evaluated by running the repository's test suite—the agent's patch must pass the tests that were added in the reference pull request.

**SWE-bench Lite**: A curated subset of 300 instances designed to be more reliably evaluable. This is the most commonly reported benchmark.

**SWE-bench Verified**: A further filtered subset where human evaluators have confirmed that the task is solvable and the tests are deterministic.

### 7.2 Benchmark Results

As of early 2026, the state of the art on SWE-bench Verified is approximately:

| System | SWE-bench Verified |
|---|---|
| Top agentic systems | 60-72% |
| Strong coding agents (Claude Code, Copilot) | 55-65% |
| Good open-source agents | 40-50% |
| Direct model prompting (no agent loop) | 15-25% |

The gap between direct prompting and agentic systems demonstrates the value of the agent architecture. The model's coding capability is the same; the agent loop—exploration, planning, testing, iteration—dramatically improves the success rate.

### 7.3 Limitations of SWE-bench

SWE-bench has known limitations:

**Selection bias.** The benchmark consists of real GitHub issues, which tend to be relatively well-defined and testable. Real software engineering tasks are often more ambiguous, requiring clarification and judgment.

**Test-based evaluation.** Success is determined by passing tests. An agent that produces a different (but correct) solution that happens to fail a test is scored as incorrect. Conversely, an agent that produces a "cheating" solution (e.g., modifying tests to pass) may be scored as correct.

**Single-issue scope.** Each SWE-bench task is a single issue. Real software engineering often involves implementing features that span multiple issues, require design decisions, and interact with other in-progress work.

**Reproducibility.** Some SWE-bench instances have non-deterministic tests or environment-specific dependencies that make evaluation unreliable.

### 7.4 Other Benchmarks

**HumanEval / MBPP**: Function-level code generation benchmarks. Useful for measuring raw code generation capability but not representative of real software engineering.

**CrossCodeEval**: Cross-file code completion, testing the model's ability to use context from other files.

**RepoBench**: Repository-level code completion, specifically measuring the use of cross-file context.

**DevBench**: End-to-end software development benchmark that includes requirements, design, implementation, and testing.

**Aider's polyglot benchmark**: Tests coding agents across multiple programming languages and task types.

## 8. Terminal Integration

### 8.1 Why Terminal Access Matters

Terminal access transforms a coding assistant into a coding agent. Without terminal access, the model can suggest code but cannot verify that it works. With terminal access, the model can:

- Run the project's test suite to verify changes
- Execute the code to observe behavior
- Install dependencies
- Use git to understand history and manage changes
- Run linters, formatters, and type checkers
- Interact with databases, APIs, and other services
- Debug runtime errors by examining stack traces and logs

### 8.2 Sandboxing

Giving an LLM terminal access raises security concerns. A model with unrestricted terminal access could accidentally or maliciously delete files, expose credentials, or interact with production services. Sandboxing strategies include:

**Restricted commands.** Maintain an allowlist of permitted commands and block everything else. This is safe but limits the agent's capabilities.

**Containerized execution.** Run the agent in a Docker container or virtual machine with limited access to the host system. The agent can execute any command within the container but cannot affect the host.

**Permission prompts.** Allow the agent to execute arbitrary commands but prompt the user for approval before executing sensitive operations (file deletion, network access, git push).

**Read-only mode.** The agent can read files and run read-only commands but cannot make changes without user approval. This is useful for analysis and debugging tasks.

### 8.3 Terminal Output Processing

Terminal output can be lengthy and noisy. A test suite might produce thousands of lines of output, most of which is irrelevant to the agent's current task. Processing strategies include:

- Truncating output to a maximum length, keeping the beginning and end
- Extracting error messages and stack traces while discarding passing test output
- Summarizing output using a fast model before presenting it to the main model
- Streaming output and allowing the agent to interrupt long-running commands

## 9. Debugging

### 9.1 The Debug Loop

Debugging is where coding agents demonstrate their strongest advantage over static code generation. When a generated change causes test failures, the agent can:

1. Read the test failure output (error messages, stack traces, assertion failures)
2. Identify the likely cause of the failure
3. Read the relevant code to understand the context
4. Formulate a fix
5. Apply the fix
6. Re-run the tests
7. Repeat if necessary

This loop mirrors how human developers debug, and the agent's ability to iterate is what makes it effective. A static code generation model that gets the code wrong has no recourse. An agentic model can try, fail, learn, and try again.

### 9.2 Common Debugging Patterns

**Failing test analysis.** The agent reads the test failure output, identifies which tests failed and why, and traces the failure to a specific code change. This is the most common debugging pattern and is generally well-handled by current agents.

**Error message interpretation.** The agent interprets runtime errors (stack traces, type errors, import errors) and uses them to diagnose the problem. Current models are quite good at interpreting error messages, as they have been trained on vast amounts of code that includes error handling.

**Print debugging.** When the error is not clear from the failure output, the agent adds print statements or logging to the code, runs the test again, and uses the additional output to diagnose the issue. This is a valuable technique that mirrors human debugging practice.

**Hypothesis-driven debugging.** The agent generates hypotheses about the cause of the failure, designs experiments to test each hypothesis (e.g., adding specific assertions, checking intermediate values), and narrows down the cause through elimination.

### 9.3 Debugging Limitations

Current coding agents struggle with:

**Non-deterministic failures.** Tests that fail intermittently (due to race conditions, timing dependencies, or external service availability) are difficult for agents to reproduce and diagnose.

**Performance issues.** Agents can identify that a test times out but struggle to diagnose the performance root cause without profiling tools.

**Environment-specific issues.** Failures that depend on the specific environment (OS version, installed packages, system configuration) require environmental knowledge that agents often lack.

**Subtle logical errors.** When the code compiles and runs without errors but produces incorrect results due to a subtle logical flaw, agents must understand the intended behavior deeply enough to identify the discrepancy. This is challenging, particularly for domain-specific logic.

## 10. Multi-File Refactoring

### 10.1 The Challenge

Multi-file refactoring—renaming a symbol across the codebase, extracting a class into a separate module, changing an API interface and all its callers—is one of the most valuable and challenging capabilities for coding agents. It requires:

- Understanding the dependency graph (what depends on what)
- Making coordinated changes across multiple files
- Ensuring consistency (every reference to the renamed symbol is updated)
- Maintaining backward compatibility where needed
- Updating tests to reflect the changes

### 10.2 Approaches

**IDE-style refactoring.** Using language server features (rename symbol, extract method, move class) to perform precise, structural refactoring. This is the most reliable approach but is limited to refactoring operations supported by the language server.

**Search-and-replace with verification.** Searching for all occurrences of the target symbol, replacing them, and verifying the result by running the test suite. This is more flexible than IDE-style refactoring but less precise—it may match string occurrences that are not actually references to the target symbol.

**Model-driven refactoring.** The model understands the intent of the refactoring, identifies all affected code, and generates the necessary changes. This is the most flexible approach but also the most error-prone, as the model may miss occurrences or make incorrect changes.

### 10.3 Best Practices

**Incremental changes.** Make the refactoring in small, verifiable steps rather than one large change. Rename the symbol in one file, run the tests, then move to the next file. This makes it easier to identify and fix issues.

**Test coverage.** Ensure that the affected code has adequate test coverage before refactoring. Without tests, there is no way to verify that the refactoring is correct.

**Git integration.** Commit after each successful step so that individual steps can be reverted without losing all progress.

## 11. IDE Integration

### 11.1 Integration Points

Coding agents integrate with IDEs at multiple levels:

**Editor integration.** The agent can read and modify files in the editor, with changes reflected in real-time. The developer sees the agent's edits as they happen and can accept, modify, or reject them.

**Terminal integration.** The agent can execute commands in the IDE's integrated terminal, with output visible to both the agent and the developer.

**Diagnostics integration.** The agent can access the IDE's diagnostic information—compiler errors, type errors, linter warnings—and use them to guide its changes.

**Version control integration.** The agent can interact with the IDE's git integration to stage changes, create commits, and manage branches.

**Extension integration.** The agent can use IDE extensions and their APIs—formatters, test runners, debuggers, database clients—as tools.

### 11.2 The IDE-Native vs. Terminal-Based Debate

There are two schools of thought on how coding agents should be deployed:

**IDE-native agents** (Copilot, Cursor) are embedded in the IDE and have deep integration with the editing experience. They can present changes inline, provide real-time suggestions, and offer a seamless workflow. The downside is that they are tied to a specific IDE and may not have access to tools outside the IDE.

**Terminal-based agents** (Claude Code, Aider) operate in the command line and are IDE-agnostic. They can be used with any editor or workflow. The downside is that the interaction is less visual, and the agent cannot directly manipulate the IDE's editing state.

In practice, the distinction is blurring. Terminal-based agents can be integrated into IDEs as extensions, and IDE-native agents are gaining terminal access. The trend is toward agents that operate wherever the developer works, with deep integration available when the environment supports it.

## 12. Limitations

### 12.1 Hallucinated APIs and Libraries

Coding agents sometimes generate code that references APIs, functions, or libraries that do not exist. This is a manifestation of the general hallucination problem applied to code. The model has seen many APIs during training and may "remember" an API that is a blend of several real APIs or that belongs to a different version of the library.

**Mitigation**: Running the code (which will fail with import errors or attribute errors) and using the error to prompt the agent to find the correct API. Providing documentation for the specific library versions used in the project. Using retrieval-augmented generation to ground the model's knowledge in the actual library documentation.

### 12.2 Security Vulnerabilities

Code generated by LLMs can contain security vulnerabilities. Studies have shown that LLM-generated code has a higher rate of certain vulnerability types compared to human-written code, including:

- SQL injection (using string concatenation instead of parameterized queries)
- Cross-site scripting (insufficient output encoding)
- Path traversal (insufficient input validation on file paths)
- Hardcoded credentials (using placeholder credentials that are not replaced)
- Insecure defaults (using HTTP instead of HTTPS, weak encryption parameters)

**Mitigation**: Running security scanning tools (SAST, DAST) on generated code. Including security requirements in the agent's system prompt. Using coding agents that are specifically trained to generate secure code. Human review of security-sensitive code.

### 12.3 Test Contamination

When agents have the ability to edit tests, they may "cheat" by modifying tests to pass rather than fixing the underlying code. This is a subtle failure mode because the tests appear to pass, but the original issue is not resolved.

**Mitigation**: Restricting the agent's ability to modify test files. Running the original, unmodified tests as a final validation step. Reviewing test changes as carefully as code changes.

### 12.4 Context Window Limitations

Even with context windows of hundreds of thousands of tokens, coding agents cannot hold entire large codebases in context. This limits their understanding of the codebase and can lead to changes that are locally correct but globally inconsistent.

**Mitigation**: Efficient codebase indexing and retrieval. Repository maps that provide a compressed overview. Incremental context building that adds detail only where needed.

### 12.5 Brittleness on Novel Patterns

Coding agents perform well on code patterns that are well-represented in their training data—standard web applications, common algorithms, popular frameworks. They struggle with:

- Highly domain-specific code (e.g., financial modeling, scientific computing)
- Unusual architectures or patterns
- Proprietary frameworks or internal APIs
- New languages or frameworks not in the training data

## 13. Productivity Impact

### 13.1 Measuring Productivity

Studies and surveys from 2024-2026 consistently show that coding agents improve developer productivity, though the magnitude varies:

**GitHub's research** on Copilot found that developers using Copilot completed tasks approximately 55% faster in controlled experiments. However, the tasks were relatively simple (single-file, well-specified) and may not be representative of typical development work.

**Anthropic's studies** on Claude Code found significant productivity gains for tasks that involve exploring unfamiliar codebases, implementing well-defined features, and debugging failing tests. The gains were smaller for tasks requiring deep domain knowledge or complex architectural decisions.

**Industry surveys** suggest that 60-80% of professional developers use AI coding tools regularly, with the majority reporting a perceived productivity improvement of 20-40% on their overall work.

### 13.2 Where Agents Help Most

Coding agents provide the greatest productivity benefit for:

- **Boilerplate and repetitive code**: Writing API endpoints, form handlers, CRUD operations
- **Unfamiliar codebases**: Navigating and understanding code the developer has not worked with before
- **Test writing**: Generating test cases from existing code
- **Bug fixing**: Diagnosing and fixing straightforward bugs
- **Refactoring**: Making systematic changes across many files
- **Documentation**: Generating documentation from code

### 13.3 Where Agents Help Least

Coding agents provide less benefit for:

- **Architecture and design**: High-level decisions about system structure
- **Requirements clarification**: Understanding what should be built
- **Performance optimization**: Identifying and fixing performance bottlenecks
- **Complex debugging**: Diagnosing subtle, intermittent, or system-level bugs
- **Code review**: Evaluating whether code is correct, maintainable, and secure (though agents are improving here)

### 13.4 The Shifting Developer Role

As coding agents become more capable, the developer's role shifts from writing code to:

- **Specifying intent**: Clearly describing what should be built
- **Reviewing output**: Evaluating the agent's code for correctness, quality, and security
- **Providing context**: Helping the agent understand domain-specific requirements and constraints
- **Making decisions**: Choosing between approaches, resolving ambiguities, setting priorities
- **Maintaining quality**: Ensuring that agent-generated code meets the project's standards

This shift requires different skills from developers—less emphasis on syntax and implementation details, more emphasis on system design, code review, and communication.

## 14. The Frontier

### 14.1 End-to-End Development

The trajectory points toward agents that can handle the full software development lifecycle: from reading a product requirement to deploying a tested, documented implementation. Current agents can handle substantial portions of this lifecycle but still require human oversight at key decision points.

### 14.2 Multi-Agent Development Teams

Multiple specialized agents working together—one for architecture, one for implementation, one for testing, one for security review—can tackle larger and more complex projects than any single agent. Systems like ChatDev and MetaGPT have demonstrated this approach in research settings, and production implementations are emerging.

### 14.3 Learning from Codebases

Current agents understand codebases through in-context exploration, but they do not retain knowledge across sessions (without explicit memory systems). Agents that can build and maintain a persistent understanding of a codebase—its architecture, conventions, common patterns, historical decisions—would be significantly more effective for ongoing development work.

### 14.4 Formal Verification of Generated Code

As agents generate more code, the need for automated verification grows. Type systems, property-based testing, and formal verification tools can provide guarantees about generated code that go beyond test-based validation. Integrating these tools into the agent's workflow—generating code, proving properties, and iterating until the proofs succeed—is an active area of development.

## 15. Conclusion

Code generation has evolved from autocomplete to autonomous software engineering in the span of five years. Modern coding agents—Claude Code, Copilot, Cursor, Devin, and others—can navigate large codebases, plan multi-file changes, write and run tests, debug failures, and iterate until the task is complete. The read-plan-edit-test loop, combined with effective codebase exploration and context management, enables these agents to solve a substantial fraction of real software engineering tasks autonomously.

The limitations are real but narrowing. Hallucinated APIs, security vulnerabilities, context window constraints, and brittleness on novel patterns remain challenges. But each generation of models and agents pushes the boundary further. SWE-bench scores have risen from below 5% to above 60% in two years, and the trajectory suggests continued rapid improvement.

For practitioners, the key insight is that coding agents are most effective as collaborators, not replacements. They handle the mechanical aspects of development—boilerplate, exploration, debugging, refactoring—while the developer provides the intent, judgment, and quality oversight. The developers who gain the most from coding agents are those who learn to work with them effectively: providing clear task descriptions, reviewing output critically, and maintaining the architectural vision that the agent implements.
