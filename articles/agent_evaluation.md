# Evaluating LLM Agents

*April 2026*

## 1. Introduction

Evaluating a language model is hard. Evaluating an LLM agent is harder by an order of magnitude. A language model takes text in and produces text out. You can compare the output to a reference, compute a metric, and move on. An agent takes a goal, executes a sequence of actions in an environment, and produces an outcome that may depend on dozens of intermediate decisions, tool calls, environment states, and stochastic branching points. The output is not a string to be compared but a trajectory to be judged.

This distinction matters because the industry is moving rapidly from model deployment to agent deployment. Coding agents that resolve GitHub issues, customer service agents that handle multi-step workflows, research agents that search and synthesize information, computer-use agents that navigate web interfaces—these systems are being deployed in production. The question of whether they work, how well they work, and when they fail is not academic. It determines whether an agent can be trusted with real tasks that affect real users.

This report provides a comprehensive examination of how LLM agents are evaluated. It covers why agent evaluation is fundamentally harder than model evaluation, the major benchmarks used to measure agent capabilities, the dimensions along which agents are assessed, the tension between automated and human evaluation, the reproducibility crisis in agent benchmarks, how to build custom evaluation suites, and how evaluation integrates into the development and deployment lifecycle. The intended audience is engineers building and deploying LLM agents and researchers working on agent capabilities.

## 2. Why Agent Evaluation Is Harder Than Model Evaluation

### 2.1 The Stochastic Action Problem

A language model's output is stochastic—the same prompt can produce different completions. But for most evaluation purposes, this variability is manageable. You can run multiple samples, compute majority-vote accuracy, or use temperature 0 for deterministic outputs. The output space is text, and text can be compared.

An agent's behavior is stochastic at every step of a multi-step trajectory. The first action may vary, which changes the environment state, which changes the available actions, which changes the second action, and so on. A coding agent working on a bug might start by reading one file or another. That choice cascades through the entire trajectory. Even with temperature 0, differences in API responses, environment state, or timing can produce radically different trajectories.

This means that a single evaluation run tells you very little. An agent might succeed on run 1 and fail on run 2 of the exact same task. Reliable evaluation requires multiple runs per task, which multiplies cost and time. The number of runs needed to estimate success probability with reasonable confidence intervals is often 5-10 per task, turning a 100-task benchmark into 500-1000 agent executions.

### 2.2 Multi-Step Dependencies

Model evaluation typically involves independent examples. Each question in a benchmark is scored independently—getting question 7 wrong does not affect question 8. Agent tasks are inherently sequential. A wrong action at step 3 may make success at step 15 impossible, or it may not—the agent might recover. This makes it difficult to attribute failures to specific decisions.

Consider a coding agent tasked with resolving a GitHub issue. The agent reads the issue, explores the codebase, identifies the relevant files, understands the bug, writes a fix, and runs tests. Failure at any point can cascade. But partial success is meaningful—an agent that correctly identifies the bug but writes a flawed fix is more capable than one that explores the wrong files entirely. Scoring must account for this nuance, which simple pass/fail metrics miss.

### 2.3 Environment Interaction

Models operate in a vacuum—they receive input and produce output. Agents operate in environments. A coding agent interacts with a filesystem, a shell, and a test suite. A browser agent interacts with a web page that has dynamic content, JavaScript, and state. A customer service agent interacts with APIs, databases, and possibly other humans.

These environments introduce variables that are outside the agent's control and outside the evaluator's control. A web page may change its layout between evaluation runs. An API may return different results. A database may have different state. These environmental variables inject noise into evaluation results that is not attributable to the agent's capabilities.

### 2.4 The Cost Problem

Evaluating a language model on a benchmark is cheap. You send prompts, collect responses, and score them. The cost is proportional to the number of tokens processed. Evaluating an agent is expensive. Each task may involve dozens of LLM calls, each with tool definitions that inflate the prompt. A single SWE-bench task might cost $1-5 in API calls. Running the full SWE-bench Verified benchmark (500 tasks) with 3 runs per task costs $1,500-7,500 per evaluation. Running it across multiple model versions, prompt variants, and architecture changes during development makes evaluation a significant budget item.

The cost problem creates a tension between evaluation thoroughness and development speed. Teams often compromise by evaluating on subsets, reducing runs per task, or using cheaper models as proxies—all of which reduce evaluation reliability.

## 3. Key Benchmarks

### 3.1 SWE-bench and SWE-bench Verified

**What it measures.** SWE-bench evaluates an agent's ability to resolve real GitHub issues from popular Python repositories. Each task provides an issue description and a codebase; the agent must produce a patch that resolves the issue. Success is determined by whether the patch passes the repository's test suite (including held-out tests that verify the fix).

**The original SWE-bench** (Jimenez et al., 2024) contains 2,294 tasks drawn from 12 popular Python repositories including Django, Flask, scikit-learn, matplotlib, and others. The tasks range from simple bug fixes to complex feature additions. This benchmark became the standard for evaluating coding agents, but it has known issues: some tasks have ambiguous issue descriptions, some test suites are flaky, and some tasks are effectively unsolvable without information not present in the issue.

**SWE-bench Verified** addresses these issues by having human developers review and filter the original tasks. The verified subset contains 500 tasks that are confirmed to be solvable given the issue description and codebase. This subset has become the primary benchmark for coding agent evaluation.

**Current state of the art.** As of early 2026, the best coding agents resolve approximately 60-72% of SWE-bench Verified tasks. This represents rapid progress from roughly 20% in early 2024. The leading systems combine frontier models (Claude, GPT-4o, Gemini) with sophisticated scaffolding that includes codebase exploration, test generation, iterative debugging, and multi-agent architectures.

**Scoring.** SWE-bench uses binary pass/fail scoring per task. A patch either passes all held-out tests or it does not. There is no partial credit. This is a deliberate choice—in software engineering, a patch that mostly fixes a bug but introduces a regression is not acceptable. However, it means that an agent that correctly diagnoses the problem but makes a minor implementation error scores the same as one that does not understand the problem at all.

**Limitations.** SWE-bench is Python-only, focuses on well-tested open-source repositories, and uses test suites as ground truth (some valid fixes may fail tests due to test limitations). It also does not measure code quality, readability, or efficiency—only correctness as defined by tests.

### 3.2 GAIA

**What it measures.** GAIA (General AI Assistants) evaluates an agent's ability to answer complex questions that require using multiple tools and reasoning across multiple steps. Unlike SWE-bench, which focuses on a single domain (coding), GAIA tests general-purpose agent capabilities.

**Task design.** GAIA tasks are questions that a human could answer with access to the internet and standard tools, but that require multiple steps. Examples include: "What was the population of the city where the author of [specific paper] was born, according to the most recent census?" Answering requires identifying the paper's author, finding where they were born, and looking up census data. Each task has a single, unambiguous correct answer.

**Levels.** GAIA defines three difficulty levels:
- Level 1: Tasks requiring 1-2 tool calls and straightforward reasoning.
- Level 2: Tasks requiring 3-5 tool calls with some reasoning complexity.
- Level 3: Tasks requiring many tool calls, complex reasoning, and sometimes creative problem-solving.

**Current performance.** Frontier agents score well on Level 1 (80-90%) but performance degrades significantly on Level 3 (30-50%). Human performance on GAIA is approximately 92% across all levels, demonstrating a persistent gap.

**Strengths.** GAIA tasks have ground-truth answers, making scoring unambiguous. The tasks test a broad range of capabilities (web search, calculation, document reading, multi-step reasoning) in a general-purpose setting.

**Limitations.** Some GAIA tasks depend on web content that may change, making results non-reproducible over time. The benchmark also does not measure the efficiency of the agent's approach—an agent that uses 20 tool calls to answer a question that requires 3 is scored the same as one that does it in 3 calls.

### 3.3 WebArena and VisualWebArena

**What they measure.** WebArena evaluates web agents—systems that navigate and interact with web applications to accomplish tasks. The agent must click links, fill forms, navigate between pages, and perform multi-step workflows in realistic web environments.

**Environment.** WebArena provides self-hosted web applications that mimic real-world sites: a Reddit-like forum, a shopping site, a content management system, a GitLab instance, and a map application. These are fully functional web applications deployed in Docker containers, providing a reproducible environment.

**Task examples.** "Find the cheapest one-way flight from New York to London on December 15th and book it." "Create a new repository on GitLab with a specific name and add a README file." "Post a reply to the most recent comment in the technology forum." Tasks require understanding the web application's interface, navigating to the right pages, and performing the correct sequence of actions.

**VisualWebArena** extends WebArena with tasks that require visual understanding—the agent must interpret images, charts, and visual layouts to complete tasks, not just text content.

**Scoring.** Tasks are scored by checking the environment state after the agent finishes. Did the flight get booked? Was the repository created? Was the reply posted? This is evaluated through programmatic checks against the web application's database or DOM state.

**Current performance.** Web agents score approximately 25-40% on WebArena, with significant variation by task type. Navigation-heavy tasks are easier than tasks requiring complex form filling or multi-page workflows. Human performance is approximately 78%, indicating substantial room for improvement.

### 3.4 tau-bench

**What it measures.** tau-bench (Tool Agent User benchmark) evaluates agents specifically on their ability to use tools correctly in realistic customer service scenarios. It focuses on the intersection of natural language understanding, policy adherence, and tool use.

**Design.** tau-bench presents agents with simulated customer interactions in two domains: retail (order management, returns, product inquiries) and airline (booking, cancellations, flight changes). Each scenario includes a customer profile, a conversation history, a set of available tools (e.g., lookup_order, process_return, search_flights), and a company policy that the agent must follow.

**What makes it distinctive.** tau-bench explicitly tests whether the agent follows policies while using tools. A customer might ask for a refund on a non-refundable order. The agent must use the lookup tool to check the order details, determine that the item is non-refundable per policy, and decline the request politely—not just execute whatever the customer asks for. This tests the agent's ability to balance helpfulness with constraint adherence.

**Scoring.** tau-bench scores on multiple dimensions: whether the correct tools were called with the correct arguments, whether the final outcome matches the expected outcome, and whether the agent's behavior complied with the stated policies.

### 3.5 OSWorld

**What it measures.** OSWorld evaluates agents that interact with full desktop operating systems—not just web browsers, but the entire computing environment including file managers, text editors, terminal applications, and system settings.

**Environment.** OSWorld provides virtual machines running Ubuntu or Windows with standard desktop applications. The agent observes the screen (via screenshots) and acts through mouse clicks, keyboard input, and other desktop interactions.

**Task examples.** "Download the PDF from this URL, extract all tables, and save them as a CSV file." "Open the system settings and change the display resolution to 1920x1080." "Create a presentation with three slides about machine learning." Tasks span the full range of desktop computing activities.

**Current performance.** Desktop agents score approximately 10-20% on OSWorld, making it one of the most challenging agent benchmarks. The combination of visual understanding, spatial reasoning, multi-application coordination, and long action sequences makes these tasks extremely difficult for current systems. Even the best models struggle with tasks that require more than 10-15 sequential actions.

### 3.6 AgentBench

**What it measures.** AgentBench is a comprehensive benchmark suite that evaluates agents across eight distinct environments: operating system (bash), database (SQL), knowledge graph, digital card game, lateral thinking puzzles, household tasks (ALFWorld), web shopping, and web browsing.

**Design philosophy.** AgentBench's breadth is deliberate. It tests whether agent capabilities generalize across very different environments and task types. An agent that excels at coding but fails at web navigation has narrow capabilities; AgentBench reveals these imbalances.

**Scoring.** Each environment has its own success criteria, and scores are aggregated across environments. This aggregation can mask domain-specific strengths and weaknesses, so per-environment scores are also reported.

**Current findings.** AgentBench results consistently show that proprietary frontier models (GPT-4, Claude, Gemini) significantly outperform open-weight models as agents, even when the gap is smaller on traditional language benchmarks. This suggests that agent capabilities are particularly sensitive to model quality, especially reasoning ability and instruction following.

### 3.7 Emerging Benchmarks

Several newer benchmarks address specific gaps:

**SWE-bench Multilingual** extends coding evaluation beyond Python to JavaScript, TypeScript, Java, Go, and other languages. This is important because many production agents operate in polyglot codebases.

**DevBench** evaluates agents on the full software development lifecycle, including project setup, dependency management, testing, and documentation, not just bug fixing.

**AssistantBench** focuses on evaluating AI assistants on real-world web tasks that require free-form text answers rather than binary success/failure, attempting to bridge the gap between benchmarks and actual user experiences.

**MiniWoB++** provides simplified web interaction tasks (clicking buttons, filling forms, navigating menus) that are less realistic than WebArena but enable faster, cheaper evaluation and more controlled experimentation.

## 4. Evaluation Dimensions

### 4.1 Task Success Rate

The most fundamental metric: did the agent accomplish the task? This is typically measured as a binary pass/fail per task, with the overall score being the percentage of tasks successfully completed. Pass@1 measures success on a single attempt. Pass@k measures whether any of k attempts succeeds, giving a more generous assessment of capability.

The distinction between pass@1 and pass@k matters. An agent with 40% pass@1 but 70% pass@3 is capable but inconsistent—it can solve the task but does not do so reliably. Whether this matters depends on the deployment context. For a human-in-the-loop system where the user can retry, pass@3 may be the relevant metric. For an autonomous system that runs without supervision, pass@1 is what matters.

**Partial credit.** Binary scoring loses information. Some benchmarks have adopted more granular scoring:
- **Milestone-based scoring**: Define intermediate milestones for each task and score what fraction the agent achieves. A coding agent that correctly identifies the bug and localizes it to the right file gets credit even if the final patch is wrong.
- **Weighted scoring**: Assign different weights to different tasks based on difficulty, so that solving hard tasks contributes more to the overall score.
- **Functional correctness with gradations**: For coding tasks, measure what fraction of tests pass rather than requiring all tests to pass.

### 4.2 Efficiency and Cost

Two agents might both solve a task, but one does it in 3 steps costing $0.05, while the other takes 30 steps costing $2.00. Efficiency metrics capture this difference:

**Step count.** The number of actions or tool calls the agent takes to complete the task. Fewer steps generally means a more efficient agent, though this can penalize agents that verify their work.

**Token usage.** The total number of input and output tokens consumed. This is directly proportional to cost for API-based models. Long-context agents that stuff entire codebases into context may succeed more often but at much higher token cost.

**Wall-clock time.** The total time from task start to completion. This matters for user-facing agents where latency affects experience. It includes model inference time, tool execution time, and any environment interaction time.

**Dollar cost per task.** The total API cost for completing a task. This is the metric that matters most for production deployment. An agent that costs $5 per task to resolve a GitHub issue is a very different proposition from one that costs $0.50 per task with the same success rate.

**Cost-adjusted success rate.** Success rate divided by average cost per task. This penalizes agents that achieve high success rates through brute-force approaches (e.g., generating dozens of candidate patches and testing each one). Some teams use a cost-weighted scoring where the score is success_rate / average_cost, making efficiency a first-class concern.

### 4.3 Safety and Guardrail Adherence

For deployed agents, safety is not optional. Evaluation must cover:

**Boundary adherence.** Does the agent stay within its authorized scope? A customer service agent should not modify database records it is not supposed to touch. A coding agent should not delete files unrelated to the task. A browser agent should not navigate to unauthorized sites.

**Policy compliance.** Does the agent follow specified policies? In tau-bench, this is explicitly tested. But every production agent has policies—refund limits, information disclosure rules, escalation procedures. Evaluation must verify that the agent follows these policies even when the user pushes back.

**Harmful action avoidance.** Does the agent refuse to take actions that would cause harm? This includes actions that are technically possible but undesirable: deleting production data, sending emails to wrong recipients, making financial transactions above authorized limits.

**Prompt injection resistance.** Can the agent be manipulated into taking unauthorized actions through adversarial inputs? A web agent navigating a page with hidden instructions, a coding agent processing a repository with malicious code comments, a customer service agent receiving crafted messages designed to bypass policies—all of these are realistic attack vectors.

**Information leakage.** Does the agent inadvertently expose information it should not? This includes system prompts, tool definitions, internal company data, or other users' information.

### 4.4 Tool Use Accuracy

For tool-using agents, the quality of tool use is a critical evaluation dimension:

**Tool selection accuracy.** Does the agent choose the right tool for the job? Given a set of available tools, does it select the most appropriate one for each step?

**Argument correctness.** Does the agent provide correct and complete arguments to the tools it calls? Missing arguments, wrong types, hallucinated values, and incorrectly formatted arguments all reduce tool use quality.

**Tool call necessity.** Does the agent make unnecessary tool calls? An agent that calls a search tool for information it already has in context is wasting resources. An agent that calls the same tool multiple times with the same arguments is being redundant.

**Result interpretation.** Does the agent correctly interpret and use tool results? A search tool might return irrelevant results—does the agent recognize this and try a different query, or does it blindly use the results?

### 4.5 Trajectory Quality

Beyond whether the agent succeeds, the quality of its trajectory matters:

**Reasoning quality.** Does the agent's chain of thought show sound reasoning? Does it correctly understand the task, formulate a plan, and adapt when things go wrong?

**Recovery from errors.** When the agent makes a mistake or a tool call fails, does it recover gracefully? Does it try alternative approaches, or does it get stuck in loops?

**Exploration efficiency.** Does the agent explore the environment effectively? In coding tasks, does it read relevant files or does it waste time reading unrelated ones? In web tasks, does it navigate efficiently or click randomly?

## 5. Human Evaluation vs. Automated Evaluation

### 5.1 The Case for Automated Evaluation

Most agent benchmarks use automated evaluation—programmatic checks that determine whether the task was completed successfully. This has clear advantages:

**Scalability.** Automated evaluation can score thousands of agent runs without human involvement, enabling large-scale benchmarking.

**Reproducibility.** The same automated checker produces the same score for the same agent output, eliminating inter-rater variability.

**Speed.** Results are available immediately after the agent finishes, enabling rapid iteration.

**Cost.** Once the evaluation harness is built, marginal evaluation cost is near zero (beyond the agent's API costs).

### 5.2 The Case for Human Evaluation

Automated evaluation misses important aspects of agent quality:

**Response quality.** An agent that completes a customer service task but is rude, confusing, or unhelpful scores the same as one that is clear and professional. Automated metrics do not capture communication quality.

**Approach quality.** Two agents might both resolve a coding task, but one writes clean, well-documented code while the other writes a hacky workaround. Test-based evaluation scores them equally.

**Edge case handling.** Automated checks test for expected outcomes but may miss unexpected behaviors—the agent accomplishing the task but also producing undesirable side effects.

**User preference.** Ultimately, what matters is whether users prefer the agent's behavior. Human evaluation through A/B testing, preference ranking, and satisfaction surveys captures this directly.

### 5.3 LLM-as-Judge

A middle ground between human and automated evaluation is using an LLM as a judge. A capable model (often a different model from the one being evaluated) reviews the agent's trajectory and scores it on multiple dimensions.

**How it works.** The judge model receives the task description, the agent's complete trajectory (all actions, tool calls, observations, and reasoning), and the final outcome. It is prompted to evaluate the trajectory along specific dimensions (correctness, efficiency, safety, communication quality) and provide scores with explanations.

**Advantages.** LLM-as-judge scales better than human evaluation while capturing aspects that automated metrics miss. It can evaluate communication quality, reasoning soundness, and approach appropriateness.

**Limitations.** LLM judges have systematic biases: they tend to prefer longer, more verbose responses; they may be overly generous or overly harsh depending on the prompt; they can be fooled by confident-sounding but incorrect reasoning; and they have limited ability to verify factual claims. Judge models also have their own errors—they may score a correct trajectory as incorrect or vice versa. Calibrating LLM judges against human judgments is essential.

**Best practices for LLM-as-judge.**
- Use a different (ideally more capable) model as the judge.
- Provide detailed rubrics with specific criteria and scoring guidelines.
- Include reference solutions when available so the judge can compare.
- Validate judge accuracy against human labels on a calibration set.
- Use multiple judge calls and aggregate scores to reduce variance.

### 5.4 Hybrid Approaches

The most robust evaluation systems combine automated, LLM-based, and human evaluation:

1. **Automated metrics** for task success, cost, latency, and tool use accuracy—these are objective and scalable.
2. **LLM-as-judge** for trajectory quality, reasoning, and communication—these are subjective but important.
3. **Human evaluation** for a sample of runs, validating both automated and LLM-based scores and catching issues that neither misses.

## 6. The Reproducibility Problem

### 6.1 Stochastic Environments

Agent evaluation environments are often stochastic in ways that are hard to control:

**API variability.** Web search tools return different results at different times. Database contents change. External services have variable latency and occasional failures. An agent that succeeds when a search returns relevant results in the first page might fail when the same search returns different results a week later.

**Environment state.** Even containerized environments have variability. Race conditions, timing-dependent behavior, and non-deterministic system operations can affect results.

**Model API changes.** API providers update their models without notice. A model version that was available yesterday may behave differently today due to a silent update. This is particularly problematic for long-running evaluation campaigns where different tasks are evaluated at different times.

### 6.2 API Changes and Model Versioning

A major reproducibility challenge is that model capabilities change over time. OpenAI, Anthropic, and Google all update their models—sometimes with version bumps (GPT-4-0125, GPT-4-0513), sometimes silently. An evaluation run in January may not be comparable to one in March, even if the benchmark and agent code are identical.

Best practices for managing this:

**Pin model versions.** Always record the exact model version used (not just "gpt-4" but the specific snapshot identifier). Use version-pinned API calls where available.

**Record all parameters.** Log temperature, max tokens, system prompt, tool definitions, and any other parameters that affect model behavior.

**Snapshot the environment.** Use Docker images or VM snapshots to capture the exact environment state. Tag evaluation runs with the environment version.

**Time-stamp evaluations.** Record when each evaluation run was performed, so that results can be contextualized against known model updates.

### 6.3 Cost Variance

Agent evaluation costs are highly variable. The same task might cost $0.50 on one run and $3.00 on another, depending on the agent's trajectory. This variance makes cost comparisons between agents unreliable unless many runs are aggregated.

Additionally, API pricing changes over time. A benchmark result reported as "$0.50 per task" becomes misleading when the underlying model's pricing changes. Reporting costs in tokens rather than dollars provides a more stable metric.

### 6.4 The Leaderboard Problem

Public leaderboards for agent benchmarks create reproducibility challenges of their own. Teams report their best results, which may come from cherry-picked runs, specific model versions, or custom evaluation setups. Without standardized evaluation protocols, leaderboard rankings are often not directly comparable.

SWE-bench has addressed this partially by providing a standardized evaluation harness and requiring submissions to use it. But even with a standard harness, differences in model versions, API parameters, and run counts make direct comparisons imprecise.

## 7. Building Custom Agent Evaluations

### 7.1 When You Need Custom Evaluations

Standard benchmarks are valuable for comparing agents against a common baseline, but they often do not match your specific use case. If you are building a customer service agent for an insurance company, SWE-bench tells you nothing about whether your agent can handle insurance claims correctly. Custom evaluations are necessary when:

- Your agent operates in a specific domain with domain-specific tools and policies.
- Your agent must satisfy specific quality requirements (latency, cost, accuracy thresholds).
- You need to test specific failure modes or edge cases relevant to your deployment.
- Standard benchmarks do not cover your agent's interaction modality (voice, multimodal, etc.).

### 7.2 Task Design

Designing evaluation tasks is the most important step. Good tasks have these properties:

**Representativeness.** Tasks should reflect the actual distribution of requests your agent will receive in production. If 60% of production requests are simple lookups and 40% are complex multi-step workflows, your evaluation should have a similar distribution.

**Clear success criteria.** Each task must have unambiguous success criteria. "Handle this customer inquiry" is not a testable criterion. "Determine that this order is eligible for a refund, process the refund, and send a confirmation" is testable.

**Difficulty stratification.** Include tasks at multiple difficulty levels. This helps you understand where your agent's capabilities break down and where to focus improvement efforts.

**Edge cases.** Include tasks that test boundary conditions: ambiguous requests, conflicting instructions, incomplete information, adversarial inputs, and error conditions. These edge cases are often where agents fail in production.

**Ground truth.** Each task should have a known correct answer or outcome. For some tasks, this is straightforward (the refund was processed). For others, it requires expert judgment to define what the correct outcome is.

### 7.3 Environment Setup

The evaluation environment must be realistic but controlled:

**Isolation.** Each evaluation run should start from a clean environment state. Agent actions from one run should not affect subsequent runs. This typically means using containers, VMs, or database snapshots that are reset between runs.

**Determinism where possible.** Mock external services that would otherwise be non-deterministic. If your agent calls a weather API, use a mock that returns consistent results. If it searches the web, use a cached set of search results. This improves reproducibility without sacrificing realism.

**Realistic tools.** The tools available in the evaluation environment should match what the agent has in production. If the production agent has access to a CRM, the evaluation environment should have a CRM (or a realistic mock of one) with representative data.

**Monitoring.** Instrument the environment to capture everything: all tool calls with arguments and results, all LLM API calls with prompts and completions, environment state changes, timing information, and error traces. This data is essential for debugging failures.

### 7.4 Success Criteria and Metrics

Define success criteria at multiple levels:

**Task-level criteria.** Binary pass/fail per task. For each task, specify exactly what constitutes success. This might be a combination of: the correct tool calls were made, the final state matches the expected state, no prohibited actions were taken, and the agent's response was appropriate.

**Aggregate metrics.** Overall success rate, average cost per task, average step count, p50/p90/p99 latency, and failure categorization (what types of tasks fail most often).

**Quality metrics.** For tasks where the agent produces text (responses, code, documents), quality metrics capture aspects beyond correctness: clarity, tone, completeness, adherence to style guides.

**Safety metrics.** The fraction of runs where the agent violated a policy, took a prohibited action, leaked information, or produced harmful content. Unlike success rate where higher is better, safety violation rate should be as close to zero as possible.

### 7.5 Evaluation Harness Architecture

A production-quality evaluation harness has these components:

```
Task Repository
    |
    v
Task Runner (orchestrates agent execution)
    |
    ├── Agent Under Test (the system being evaluated)
    ├── Environment (tools, databases, mocked services)
    ├── Monitor (captures all interactions)
    |
    v
Evaluator (scores each run)
    |
    ├── Automated Checks (state verification, tool call validation)
    ├── LLM Judge (trajectory quality, response quality)
    |
    v
Reporter (aggregates scores, generates dashboards)
```

The task runner manages the lifecycle: reset the environment, start the agent, feed it the task, let it execute, capture the trajectory, score the result, log everything. It should support parallel execution across multiple tasks to reduce wall-clock evaluation time.

### 7.6 Iterating on Evaluations

Evaluations evolve as your agent improves:

**Start simple.** Begin with 20-50 tasks that cover your most important use cases. This is enough to catch major regressions and guide initial development.

**Expand as you learn.** As you discover failure modes in production, add tasks that test those specific scenarios. Your evaluation suite should grow to reflect the full range of situations your agent encounters.

**Retire stale tasks.** Tasks that your agent solves 100% of the time across many runs are no longer informative. Keep them as regression tests but focus evaluation attention on tasks near the agent's capability boundary.

**Version your evaluations.** Track changes to tasks, success criteria, and scoring logic. When comparing evaluation results over time, ensure you are comparing against the same evaluation version.

## 8. Evaluation-Driven Development

### 8.1 The Development Loop

Evaluation should drive agent development, not just measure it. The development loop is:

1. **Run evaluation.** Execute the agent on the evaluation suite.
2. **Analyze failures.** Examine failed tasks in detail. Why did the agent fail? Was it a reasoning error, a tool use error, a policy violation, or an environment issue?
3. **Categorize failures.** Group failures by root cause. Common categories include: incorrect tool selection, wrong arguments, failure to recover from errors, context window overflow, instruction following failures, and hallucination.
4. **Prioritize fixes.** Address the failure categories that account for the most failures and that are most feasible to fix.
5. **Implement changes.** Modify the agent (prompt, tools, architecture, model) to address the prioritized issues.
6. **Re-evaluate.** Run the evaluation again to verify the fix and check for regressions.

### 8.2 Red-Teaming Agents

Red-teaming—deliberately trying to make the agent fail or behave unsafely—is essential for deployed agents. Red-teaming for agents goes beyond prompt injection:

**Adversarial task design.** Create tasks that are specifically designed to exploit known weaknesses: ambiguous instructions, conflicting constraints, tasks that tempt the agent to take shortcuts, requests that would violate policies if fulfilled.

**Environmental adversaries.** Modify the environment to create adversarial conditions: tools that return errors, misleading tool results, web pages with deceptive content, database entries with unexpected values.

**Multi-turn adversarial conversations.** In conversational agents, simulate users who gradually escalate requests, trying to get the agent to take unauthorized actions through a series of seemingly innocuous steps.

**Automated red-teaming.** Use a separate LLM to generate adversarial inputs. The red-team model is prompted to find inputs that cause the agent to fail, violate policies, or produce harmful outputs. This can discover failure modes that human red-teamers miss.

### 8.3 Regression Testing

Every agent change—prompt updates, tool modifications, model upgrades, architecture changes—can introduce regressions. A regression test suite is a set of tasks that the agent is expected to pass; any failure is flagged for investigation.

Regression tests should be drawn from:
- Tasks that the agent previously failed and was fixed to handle.
- Critical use cases that must always work.
- Safety-critical scenarios where failure has severe consequences.

### 8.4 CI/CD for Agents

Integrating agent evaluation into CI/CD pipelines enables automated quality gates:

**On every PR.** Run a small, fast evaluation suite (20-50 tasks) that covers the most critical use cases. This catches major regressions before they are merged.

**Nightly.** Run the full evaluation suite. This provides comprehensive coverage and detects subtle regressions that the fast suite misses.

**Before deployment.** Run the full evaluation suite plus safety-focused evaluations. The results must meet defined quality bars (e.g., success rate >= 85%, safety violation rate = 0%) before the agent is deployed.

**Post-deployment monitoring.** Continue evaluating the agent in production through A/B testing, sampling real interactions for evaluation, and monitoring automated metrics (task completion rate, user satisfaction, escalation rate).

**Cost budgets.** Set cost budgets for CI/CD evaluations. If evaluation costs are unbounded, teams will reduce evaluation frequency, leading to quality degradation. Allocate a specific budget for evaluation and design evaluation suites that fit within it.

## 9. Leaderboard Gaming and Benchmark Saturation

### 9.1 The Gaming Problem

Public benchmarks create incentives for gaming. Common tactics include:

**Overfitting to the benchmark.** Training on the exact tasks or closely related tasks. SWE-bench has addressed this by publishing after training data cutoffs, but variants and related tasks can still influence training data.

**Prompt optimization.** Extensive prompt engineering specifically for benchmark tasks. The resulting prompts may not generalize to other tasks, making the benchmark score misleading as a measure of general capability.

**Architecture tuning.** Building agent architectures that are optimized for the benchmark's specific characteristics. A coding agent might be optimized for Python-only single-file changes (which dominate SWE-bench) at the expense of multi-file or multi-language changes.

**Cherry-picking runs.** Reporting best-of-N results without disclosing N. An agent with 40% pass@1 has 78% pass@3 and 93% pass@8. Reporting the pass@8 result as if it were the standard metric inflates the apparent performance.

**Selective task filtering.** Evaluating only on a subset of tasks where the agent performs well, and characterizing the excluded tasks as "invalid" or "ambiguous."

### 9.2 Benchmark Saturation

A benchmark is saturated when the best agents approach human-level or near-perfect performance, making it no longer discriminative between systems. When a benchmark saturates:

- Marginal improvements become statistically insignificant given the benchmark's size and variance.
- The benchmark no longer reflects the hardest problems in the domain—it has been "solved" even if real-world performance on similar tasks is much lower.
- Research effort shifts to optimizing for the benchmark rather than improving general capability.

SWE-bench Verified is approaching saturation for the best systems, with scores above 70%. The benchmark's creators have responded by developing harder variants (SWE-bench Multimodal, SWE-bench Multilingual). The broader community response is to create new benchmarks that target unsolved capabilities.

### 9.3 Mitigation Strategies

**Private holdout sets.** Maintain a private evaluation set that is never published. Use the public benchmark for comparison with other systems and the private set for internal evaluation.

**Dynamic benchmarks.** Create benchmarks that are updated regularly with new tasks, so that overfitting to a specific set is not possible. GAIA follows this approach with a held-out test set that is evaluated through a submission system.

**Multi-benchmark evaluation.** Evaluate on multiple benchmarks rather than optimizing for a single one. An agent that scores well on SWE-bench, GAIA, and WebArena is more likely to be genuinely capable than one that excels only on SWE-bench.

**Process evaluation.** In addition to outcome-based evaluation, evaluate the agent's process. Did it take reasonable steps? Did it demonstrate understanding of the task? This is harder to game because it requires genuine capability.

## 10. Real-World vs. Benchmark Performance Gaps

### 10.1 Why Benchmarks Overestimate Real-World Performance

Agent benchmarks almost always overestimate how well the agent will perform in production. There are structural reasons for this:

**Clean problem specifications.** Benchmark tasks have clear, well-written descriptions. Real-world tasks are often ambiguous, incomplete, or contradictory. A customer might say "fix my order" without specifying what is wrong with it. A coding task might be described in a Slack thread with context scattered across multiple messages.

**Limited scope.** Benchmark tasks are scoped to be solvable. Real-world tasks sometimes are not solvable with the available tools, or require capabilities the agent does not have. A well-designed agent must recognize when a task is beyond its capabilities and escalate rather than flail.

**Controlled environments.** Benchmark environments are stable and well-maintained. Production environments have unexpected states, integration failures, race conditions, and edge cases that benchmarks cannot fully capture.

**No user interaction.** Most benchmarks provide the task upfront and evaluate the final result. In production, agents interact with users mid-task. Users change their minds, provide incorrect information, interrupt the agent, and have expectations about communication style and pace that benchmarks do not measure.

**Distribution shift.** The distribution of tasks in a benchmark may not match the distribution in production. Benchmarks often over-represent interesting or challenging tasks and under-represent the mundane, repetitive tasks that constitute the majority of production workload.

### 10.2 Why Benchmarks Underestimate Real-World Performance

In some cases, agents perform better in production than on benchmarks:

**User collaboration.** In production, users provide feedback, correct misunderstandings, and guide the agent. This collaborative loop helps the agent succeed on tasks it might fail on independently.

**Narrower scope.** A production agent typically handles a narrower range of tasks than a benchmark tests. An insurance claims agent only needs to handle insurance claims—it can be specialized in ways that a general-purpose benchmark does not reward.

**Domain-specific tools.** Production agents often have access to better, more specific tools than benchmark environments provide. A proprietary API that returns exactly the right data is more helpful than a generic web search.

### 10.3 Closing the Gap

**Production monitoring as evaluation.** Treat production as a continuous evaluation environment. Sample real interactions, score them (with automated metrics, LLM judges, and human review), and use the results to improve the agent.

**Feedback loops.** When users report agent failures, convert them into evaluation tasks. This continuously enriches your evaluation suite with realistic scenarios.

**Shadowing.** Before deploying a new agent version, run it in shadow mode—it processes real requests but its outputs are not delivered to users. Compare the shadow outputs to the current agent's outputs to identify regressions and improvements.

**Staged rollout.** Deploy new agent versions to a small fraction of traffic first, monitor metrics, and gradually increase if performance is acceptable. This limits the blast radius of regressions.

## 11. Advanced Evaluation Topics

### 11.1 Multi-Agent Evaluation

When a system consists of multiple agents working together (a planner agent, a coder agent, a reviewer agent), evaluation must cover both individual agent performance and system-level performance. An individual agent might perform well in isolation but poorly in the system if agents miscommunicate, duplicate work, or create deadlocks.

System-level evaluation tests the end-to-end outcome. Individual evaluation tests each agent's contribution. Attribution—determining which agent is responsible for a system-level failure—requires trace analysis that follows the flow of information and decisions through the multi-agent system.

### 11.2 Long-Horizon Evaluation

Some agent tasks take hours or days to complete—large refactoring projects, multi-stage research tasks, complex workflow automation. Evaluating these tasks is expensive and time-consuming, but they represent important real-world use cases.

Approaches include:
- **Milestone-based evaluation**: Break long tasks into intermediate milestones and evaluate progress at each milestone.
- **Surrogate tasks**: Create shorter tasks that test the same capabilities required for long tasks. If the agent can do the components, it is more likely to succeed at the full task.
- **Sampling**: Evaluate only a sample of long tasks, accepting higher variance in exchange for feasibility.

### 11.3 Cross-Model Evaluation

Agents are often built to be model-agnostic—the same scaffolding can run with different underlying models. Cross-model evaluation tests the same agent architecture across multiple models to determine which model produces the best agent performance. This is valuable because model rankings on traditional benchmarks (MMLU, HumanEval) do not always predict agent rankings. A model that scores higher on code generation benchmarks may score lower as a coding agent if it is worse at planning, error recovery, or tool use.

### 11.4 Ablation Studies

Understanding which components of an agent matter most requires ablation studies—systematically removing or modifying components and measuring the impact:

- Remove the planning step: How much does performance drop?
- Remove the error recovery mechanism: How many more tasks fail?
- Reduce the context window: At what point does performance degrade?
- Simplify the system prompt: Which instructions matter?
- Remove specific tools: Which tools are most valuable?

Ablation studies are expensive because each ablation requires a full evaluation run, but they are essential for understanding and improving agent architectures.

### 11.5 Evaluating Agent Safety at Scale

Safety evaluation requires a different methodology from capability evaluation. While capability evaluation asks "can the agent do this?", safety evaluation asks "does the agent ever do this?" Safety violations may be rare—an agent might behave correctly 99.9% of the time and fail dangerously 0.1% of the time. Detecting these rare failures requires:

- **Large evaluation sets** to get sufficient statistical power for detecting rare events.
- **Targeted adversarial evaluation** that specifically probes for known safety failure modes.
- **Stress testing** under unusual conditions (high load, unusual inputs, edge case environments).
- **Formal verification** for critical agent properties, where feasible—proving that certain actions are impossible given the agent's tool set and constraints.

## 12. The State of Evaluation Tooling

### 12.1 Open-Source Frameworks

Several open-source frameworks support agent evaluation:

**Inspect AI** (UK AI Safety Institute) provides a framework for building evaluation harnesses with support for multiple model providers, sandboxed execution environments, and built-in scoring metrics. It has become a standard tool for AI safety evaluations.

**METR** (Model Evaluation and Threat Research) develops evaluation tools for assessing autonomous capabilities and risks of AI systems, with a focus on long-horizon tasks and safety-critical evaluations.

**AgentEval frameworks** from various research groups provide benchmark-specific evaluation harnesses (SWE-bench's harness, WebArena's harness, etc.).

**Braintrust, Arize, Langsmith** and similar platforms provide commercial evaluation and observability tools that support agent evaluation through trace logging, metric computation, and comparison dashboards.

### 12.2 What Good Tooling Provides

A mature evaluation tool should provide:

- **Task management**: Define, version, and organize evaluation tasks.
- **Environment management**: Spin up, reset, and tear down evaluation environments.
- **Execution orchestration**: Run evaluation tasks in parallel across multiple agents and configurations.
- **Trace capture**: Record complete agent trajectories including all LLM calls, tool calls, and environment interactions.
- **Automated scoring**: Apply programmatic checks to score task outcomes.
- **LLM judging**: Integrate LLM-as-judge scoring for subjective quality dimensions.
- **Comparison**: Compare results across agent versions, model versions, and configurations.
- **Statistical analysis**: Compute confidence intervals, significance tests, and handle the high variance inherent in agent evaluation.
- **Cost tracking**: Monitor and report the cost of both the evaluation itself and the agent's execution.

## 13. Conclusion

Evaluating LLM agents is fundamentally harder than evaluating language models. The combination of stochastic multi-step trajectories, environment interaction, high cost, and the gap between benchmark and real-world performance creates challenges that the field is still learning to address.

The current state of agent evaluation has clear strengths: benchmarks like SWE-bench Verified, GAIA, and WebArena provide standardized measures of important capabilities; automated evaluation enables rapid iteration; and the community is increasingly adopting rigorous evaluation practices.

But significant challenges remain. Reproducibility is fragile—model updates, environment changes, and stochastic variation make it hard to compare results across time and teams. Benchmark gaming and saturation reduce the informativeness of public leaderboards. The gap between benchmark and real-world performance means that evaluation results must be interpreted cautiously.

For practitioners building production agents, the most important investments are: building a custom evaluation suite that reflects your specific use case, integrating evaluation into your development workflow (CI/CD for agents), monitoring production performance as a continuous evaluation signal, and maintaining a healthy skepticism about benchmark numbers—including your own. An agent that passes your evaluation is not guaranteed to work in production. An agent that fails your evaluation is guaranteed to fail in production. Evaluation is necessary but not sufficient. Build it, trust it, but verify it continuously.
