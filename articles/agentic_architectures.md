# Agentic Architectures for LLMs

*April 2026*

## 1. Introduction

The transition from language models as text generators to language models as autonomous agents represents one of the most significant shifts in applied AI. A standard LLM inference call is stateless: the model receives a prompt, produces a completion, and retains no memory or ability to act on the world. An agentic architecture wraps the model in a loop that allows it to observe its environment, reason about what to do next, take actions through tools, and incorporate the results of those actions into subsequent reasoning. The model becomes a component in a system that can pursue multi-step goals, recover from errors, and adapt its strategy based on feedback.

This shift is not merely an engineering convenience. It changes the fundamental capability profile of the system. A single-turn LLM can answer questions about code. An agentic coding system can read a codebase, identify a bug, write a fix, run the test suite, observe the failures, revise the fix, and submit a pull request. The same underlying model is responsible for the reasoning, but the architecture determines whether that reasoning can be applied iteratively toward a goal.

This report provides a comprehensive technical examination of the architectures used to build agentic LLM systems. It covers the major patterns—ReAct, plan-and-execute, reflection, inner monologue, hierarchical planning, and others—with pseudocode for each. It examines the design tradeoffs between patterns, the real systems that implement them, and the practical considerations that determine which architecture to use for a given task. The intended audience is engineers building agentic systems and researchers studying the design space.

## 2. Foundations: What Makes a System Agentic

### 2.1 Defining Agency in the LLM Context

An LLM-based system is agentic to the degree that it exhibits the following properties:

**Goal-directed behavior.** The system pursues an objective over multiple steps, rather than responding to a single prompt in isolation. The objective may be explicitly stated ("fix this bug") or inferred from context ("the user wants their email summarized and the key action items extracted").

**Environmental interaction.** The system can observe and modify its environment through tools—reading files, executing code, querying APIs, browsing the web, sending messages. The environment provides feedback that influences subsequent actions.

**Autonomous decision-making.** The system decides which actions to take and in what order, without requiring human input at every step. The degree of autonomy varies: some systems make all decisions independently, while others pause for human approval at critical junctures.

**Iterative refinement.** The system can evaluate the results of its actions and adjust its approach. If a tool call fails, the system can try a different approach. If the output does not meet the goal, the system can revise it.

**State management.** The system maintains state across steps—tracking what it has done, what it has learned, and what remains to be done. This state may be explicit (a task list) or implicit (the accumulated conversation history).

These properties exist on a spectrum. A chatbot with a calculator tool is minimally agentic. A system that autonomously navigates a codebase, plans a multi-file refactoring, executes it, runs tests, and iterates until the tests pass is highly agentic. The architecture determines where on this spectrum the system falls.

### 2.2 The Core Loop

Despite their diversity, virtually all agentic architectures share a common structure: a loop in which the model alternates between thinking and acting. The simplest formulation is:

```
while not done:
    observation = get_current_state()
    thought = model.reason(observation, history)
    action = model.select_action(thought)
    result = environment.execute(action)
    history.append(observation, thought, action, result)
    done = model.check_completion(result, goal)
```

The differences between architectures lie in how they structure the reasoning step, how they decompose goals into subgoals, how they select and sequence actions, and how they handle failure. The following sections examine each major pattern in detail.

### 2.3 The Importance of the System Prompt

In every agentic architecture, the system prompt plays a critical role in defining the agent's behavior. The system prompt specifies the agent's role, its available tools, its constraints, its output format, and its decision-making heuristics. A well-designed system prompt can compensate for architectural simplicity—a basic loop with a carefully crafted system prompt often outperforms a sophisticated architecture with a poorly designed prompt.

The system prompt is also where the architecture is communicated to the model. If the agent is expected to follow a ReAct pattern, the system prompt must explain the Thought/Action/Observation format. If the agent is expected to decompose tasks before executing them, the system prompt must instruct it to do so. The model has no inherent knowledge of the architecture it is embedded in; it must be told.

## 3. The ReAct Pattern

### 3.1 Origin and Concept

ReAct (Reasoning + Acting), introduced by Yao et al. in 2022, is the foundational agentic pattern. It interleaves reasoning traces with action execution in a single sequence. The model generates a thought that explains its reasoning, then generates an action to take, then observes the result, and repeats. The key insight is that reasoning and acting are synergistic: reasoning helps the model plan which action to take, and action results provide new information that informs further reasoning.

Before ReAct, there were two separate research threads: chain-of-thought prompting (reasoning without acting) and action generation (acting without explicit reasoning). ReAct unified these, demonstrating that the combination outperforms either approach alone on tasks requiring both knowledge retrieval and multi-step reasoning.

### 3.2 The Thought-Action-Observation Cycle

The ReAct pattern structures each step of the agent loop as a three-part cycle:

1. **Thought**: The model reasons about the current situation—what it knows, what it needs to find out, and what action would be most helpful next.
2. **Action**: The model selects and parameterizes a tool call based on its reasoning.
3. **Observation**: The tool is executed and the result is appended to the context.

This cycle repeats until the model determines that it has enough information to produce a final answer or that the task is complete.

### 3.3 Pseudocode

```
function react_agent(goal, tools, model):
    history = []
    system_prompt = format_react_prompt(tools)
    
    while True:
        prompt = system_prompt + format_history(history) + goal
        response = model.generate(prompt)
        
        thought = parse_thought(response)
        action = parse_action(response)
        
        if action.type == "finish":
            return action.result
        
        observation = execute_tool(action.tool, action.args)
        history.append({
            "thought": thought,
            "action": action,
            "observation": observation
        })
        
        if len(history) > MAX_STEPS:
            return "Maximum steps reached without completion"
```

### 3.4 Example Trace

Consider a ReAct agent tasked with answering "What is the population of the city where the Python programming language was created?"

```
Thought: I need to find where Python was created. Python was created by 
Guido van Rossum. I believe he created it in Amsterdam, Netherlands, but 
I should verify this.

Action: search("where was Python programming language created")

Observation: Python was conceived in the late 1980s by Guido van Rossum 
at Centrum Wiskunde & Informatica (CWI) in Amsterdam, Netherlands.

Thought: Python was created in Amsterdam. Now I need to find the current 
population of Amsterdam.

Action: search("population of Amsterdam 2026")

Observation: The population of Amsterdam is approximately 921,000 as of 
2025 estimates.

Thought: I now have the information needed to answer the question.

Action: finish("The population of Amsterdam, where Python was created, 
is approximately 921,000.")
```

### 3.5 Strengths and Limitations

**Strengths.** ReAct is simple to implement, easy to debug (the reasoning traces are visible), and effective for a wide range of tasks. The interleaved reasoning helps the model stay focused and reduces hallucination by grounding reasoning in observed facts. The pattern works with any model that can follow the prescribed format.

**Limitations.** ReAct is fundamentally sequential—each step depends on the previous step's observation. This makes it slow for tasks that could benefit from parallel execution. The model has no explicit planning phase; it decides what to do next based only on its current reasoning, which can lead to meandering or suboptimal action sequences. For complex tasks, the lack of upfront planning means the agent may pursue dead ends before discovering the correct approach.

### 3.6 Variants

**ReAct with self-consistency.** Multiple ReAct traces are generated independently, and the final answers are aggregated (e.g., by majority vote). This improves reliability at the cost of increased computation.

**ReAct with retrieval.** The observation step is augmented with retrieval from a knowledge base, providing the model with relevant context even when no explicit tool call is made.

**Structured ReAct.** Instead of free-form text for thoughts and actions, the model generates structured output (JSON) that is parsed programmatically. This improves reliability of action parsing at the cost of some reasoning flexibility.

## 4. Plan-and-Execute

### 4.1 Concept

The plan-and-execute pattern separates planning from execution into distinct phases. First, the model generates a complete plan—a sequence of steps needed to achieve the goal. Then, the system executes each step, potentially using the same or a different model. This separation addresses one of ReAct's key limitations: the lack of upfront planning.

The intuition is that humans typically plan before acting on complex tasks. A software engineer does not start typing code the moment they receive a bug report; they first understand the problem, identify the likely cause, and plan an approach. Plan-and-execute brings this same structure to agentic systems.

### 4.2 Architecture

The architecture consists of two components:

1. **Planner**: Receives the goal and generates a list of steps. The planner may have access to tool descriptions (to know what actions are possible) but does not execute tools itself.
2. **Executor**: Takes each step from the plan and executes it, which may involve one or more tool calls. The executor may use a ReAct-style loop for individual steps.

Optionally, a **replanner** component monitors execution progress and revises the plan if a step fails or produces unexpected results.

### 4.3 Pseudocode

```
function plan_and_execute_agent(goal, tools, model):
    # Phase 1: Planning
    plan_prompt = format_planning_prompt(goal, tools)
    plan = model.generate_plan(plan_prompt)
    steps = parse_steps(plan)
    
    results = []
    
    # Phase 2: Execution
    for step in steps:
        executor_prompt = format_executor_prompt(step, results, tools)
        step_result = execute_step(step, executor_prompt, model, tools)
        results.append(step_result)
        
        # Phase 3: Replanning (optional)
        if step_result.status == "failed":
            replan_prompt = format_replan_prompt(goal, steps, results)
            revised_plan = model.generate_plan(replan_prompt)
            steps = merge_remaining_steps(revised_plan, steps, results)
    
    # Phase 4: Synthesis
    synthesis_prompt = format_synthesis_prompt(goal, results)
    final_answer = model.generate(synthesis_prompt)
    return final_answer

function execute_step(step, prompt, model, tools):
    # Each step may itself use a ReAct-style loop
    return react_agent(step.description, tools, model)
```

### 4.4 Advantages Over Pure ReAct

**Global coherence.** By planning upfront, the agent avoids the myopic decision-making of step-by-step ReAct. The plan provides a roadmap that keeps the agent focused on the overall goal.

**Parallelism.** Independent steps in the plan can be executed concurrently. If the plan includes "fetch weather data" and "fetch calendar events," these can be executed in parallel rather than sequentially.

**Model specialization.** The planner and executor can use different models. A powerful model can generate the plan, while a faster, cheaper model executes individual steps. This optimizes cost and latency.

**Transparency.** The plan is an explicit artifact that can be reviewed, modified, or approved by a human before execution begins. This is valuable for high-stakes tasks where autonomous execution is risky.

### 4.5 Challenges

**Plan rigidity.** If the plan is generated without knowledge of what the environment will reveal, it may be based on incorrect assumptions. The replanning mechanism mitigates this but adds complexity and latency.

**Granularity.** Choosing the right level of detail for plan steps is difficult. Steps that are too coarse require complex execution logic. Steps that are too fine-grained produce plans that are brittle and may not generalize to unexpected situations.

**Context window pressure.** The plan, accumulated results, and current step context all consume tokens. For complex tasks with many steps, the context window can become a binding constraint.

## 5. Reflection and Self-Critique

### 5.1 Concept

Reflection architectures add a self-evaluation step to the agent loop. After producing an output or completing an action sequence, the model (or a separate evaluator model) critiques the result and identifies ways to improve it. The system then uses this critique to generate a revised output. This pattern is inspired by the human practice of reviewing one's own work and making revisions.

Reflection can be applied at multiple levels: within a single reasoning step (checking the logic of an argument), at the action level (evaluating whether a tool call returned useful information), or at the task level (assessing whether the overall output meets the goal).

### 5.2 The Reflexion Framework

Reflexion, introduced by Shinn et al. in 2023, is a prominent implementation of the reflection pattern. It maintains a memory of verbal feedback from previous attempts. When the agent fails a task, it generates a self-reflection that identifies what went wrong and how to improve. This reflection is stored and included in the context for subsequent attempts. The system learns from its mistakes within a single task, without any weight updates.

### 5.3 Pseudocode

```
function reflection_agent(goal, tools, model, max_attempts=3):
    reflections = []
    
    for attempt in range(max_attempts):
        # Generate or revise output
        prompt = format_prompt(goal, tools, reflections)
        output = agent_execute(prompt, model, tools)
        
        # Evaluate output
        evaluation = evaluate(output, goal)
        
        if evaluation.success:
            return output
        
        # Generate reflection
        reflection_prompt = format_reflection_prompt(
            goal, output, evaluation.feedback
        )
        reflection = model.generate(reflection_prompt)
        reflections.append(reflection)
    
    return best_output(attempts)

function agent_execute(prompt, model, tools):
    # Can use ReAct, plan-and-execute, or any inner pattern
    return react_agent(prompt, tools, model)
```

### 5.4 Self-Critique Patterns

Several specific self-critique patterns have emerged:

**Output validation.** After generating a response, the model is asked to evaluate whether the response is correct, complete, and consistent with the instructions. Common prompts include "Review your answer for errors" or "Does this response fully address the user's question?"

**Constitutional self-critique.** The model evaluates its output against a set of principles (a "constitution"). For each principle, it identifies whether the output violates the principle and generates a revision. This was introduced by Anthropic as part of Constitutional AI but applies broadly to agentic self-improvement.

**Adversarial self-critique.** The model is asked to find flaws in its own output by adopting an adversarial perspective: "What would a skeptical reviewer say about this analysis?" or "What errors or oversights does this code contain?" This perspective shift often surfaces issues that straightforward self-review misses.

**Test-driven self-critique.** For coding tasks, the model generates tests, runs them against its code, and uses the test results as feedback for revision. This is objective self-critique—the tests provide ground truth about whether the code is correct, independent of the model's subjective assessment.

### 5.5 Limitations of Self-Critique

Self-critique is only effective when the model can identify its own errors. Research has shown that models are often unable to detect their own mistakes, particularly factual errors and subtle logical flaws. A model that confidently generates incorrect information will often confidently confirm that the information is correct when asked to review it. This is sometimes called the "blind spot" problem: the same knowledge gaps that cause the error also prevent the model from detecting it.

Using a separate, potentially more capable model for critique mitigates this problem but increases cost. Using external validators (test suites, type checkers, factual databases) is more reliable but limits the scope of critique to domains where automated validation is possible.

## 6. Inner Monologue

### 6.1 Concept

Inner monologue is a pattern where the model generates extended reasoning that is not shown to the user but is used to guide its actions. Unlike chain-of-thought prompting, where the reasoning is typically short and integrated into the response, inner monologue produces a separate, often lengthy stream of reasoning that the system uses internally.

The pattern is motivated by the observation that complex tasks require more reasoning than users want to read. A user who asks an agent to debug a failing test does not want to see the agent's internal deliberation about which files to read, which hypotheses to consider, and which debugging strategy to pursue. They want the agent to do this reasoning and then present the result.

### 6.2 Architecture

The inner monologue architecture separates the model's output into two streams:

1. **Internal reasoning**: Extended thought process including hypothesis generation, strategy selection, self-questioning, and planning. This is appended to the model's context but not shown to the user.
2. **External output**: The user-facing response, which summarizes the results of the internal reasoning in a concise, useful form.

### 6.3 Pseudocode

```
function inner_monologue_agent(goal, tools, model):
    internal_history = []
    user_messages = []
    
    while True:
        # Generate internal reasoning (hidden from user)
        internal_prompt = format_internal_prompt(
            goal, internal_history, user_messages, tools
        )
        internal_thought = model.generate(
            internal_prompt, 
            stop_token="[ACTION]" or "[RESPOND]"
        )
        internal_history.append(internal_thought)
        
        if internal_thought.ends_with("[RESPOND]"):
            # Generate user-facing response
            response_prompt = format_response_prompt(
                goal, internal_history
            )
            response = model.generate(response_prompt)
            user_messages.append({"role": "assistant", "content": response})
            return response
        
        elif internal_thought.ends_with("[ACTION]"):
            action = parse_action(internal_thought)
            result = execute_tool(action.tool, action.args)
            internal_history.append({
                "action": action,
                "result": result
            })
```

### 6.4 Relationship to Extended Thinking

Claude's extended thinking feature is a production implementation of the inner monologue pattern at the model level. When extended thinking is enabled, the model generates a thinking trace before producing its response. The thinking trace is available to the developer via the API but is visually separated from the response in the user interface.

The key difference between prompt-level inner monologue (where the system prompt instructs the model to reason internally) and model-level extended thinking (where the model has been trained to produce a separate thinking stream) is reliability. Prompt-level inner monologue depends on the model following the prescribed format, which it may fail to do. Model-level extended thinking is built into the model's generation process and does not depend on prompt adherence.

### 6.5 Benefits

**Better reasoning.** Generating extended reasoning before acting improves the quality of both actions and responses. The model has more "space" to think through complex problems.

**Cleaner user experience.** Users see concise results without being overwhelmed by the agent's deliberation process. The reasoning is available for debugging but does not clutter the interaction.

**Reduced hallucination.** By reasoning through a problem before committing to a response, the model is more likely to identify and correct errors in its own thinking.

## 7. Task Decomposition

### 7.1 Concept

Task decomposition breaks a complex goal into smaller, manageable subtasks. This is related to plan-and-execute but focuses specifically on the decomposition process rather than the execution pattern. The key question is: how should a complex task be broken down, and at what granularity?

Task decomposition is critical because LLMs perform significantly better on focused, well-defined tasks than on broad, ambiguous ones. A model asked to "build a web application" will produce vague, incomplete output. A model asked to "write a Python function that validates email addresses using a regular expression" will produce specific, actionable output. Decomposition bridges this gap by converting broad goals into specific subtasks.

### 7.2 Decomposition Strategies

**Sequential decomposition.** The task is broken into a linear sequence of steps, where each step depends on the output of the previous step. This is the simplest form and maps directly to the plan-and-execute pattern.

```
function sequential_decompose(goal, model):
    prompt = f"""Break the following goal into a sequence of steps.
    Each step should be specific and actionable.
    
    Goal: {goal}
    
    Steps:"""
    response = model.generate(prompt)
    return parse_steps(response)
```

**Hierarchical decomposition.** The task is broken into subtasks, which are themselves broken into sub-subtasks, forming a tree. This is useful for complex tasks where the top-level steps are themselves too complex to execute directly.

```
function hierarchical_decompose(goal, model, depth=0, max_depth=3):
    steps = sequential_decompose(goal, model)
    
    if depth >= max_depth:
        return steps
    
    for step in steps:
        if is_complex(step, model):
            step.substeps = hierarchical_decompose(
                step.description, model, depth + 1, max_depth
            )
    
    return steps
```

**Dependency-aware decomposition.** The task is decomposed into a directed acyclic graph (DAG) of subtasks, where edges represent dependencies. Subtasks without dependencies on each other can be executed in parallel.

```
function dag_decompose(goal, model):
    prompt = f"""Break the following goal into subtasks.
    For each subtask, list which other subtasks it depends on.
    
    Goal: {goal}
    
    Format each subtask as:
    Task: [description]
    Depends on: [list of task IDs, or "none"]"""
    
    response = model.generate(prompt)
    tasks = parse_dag(response)
    return topological_sort(tasks)
```

### 7.3 Adaptive Decomposition

Static decomposition—generating the full task breakdown before execution begins—suffers from the same rigidity problem as static planning. Adaptive decomposition generates the next level of decomposition on demand, based on the results of previous subtasks.

```
function adaptive_decompose_and_execute(goal, tools, model):
    # Generate only the first level of decomposition
    steps = sequential_decompose(goal, model)
    results = []
    
    for i, step in enumerate(steps):
        # Execute the step
        result = execute_step(step, tools, model)
        results.append(result)
        
        # Reassess remaining steps based on result
        if result.requires_redecomposition:
            remaining_goal = synthesize_remaining(goal, results)
            steps[i+1:] = sequential_decompose(remaining_goal, model)
    
    return synthesize_results(results)
```

### 7.4 When Decomposition Helps vs. Hurts

Decomposition improves performance when the task is genuinely complex and the model cannot solve it in a single pass. But decomposition can also hurt performance when the task is simpler than the decomposition overhead, when the subtasks have strong interdependencies that are lost in decomposition, or when the model makes errors in the decomposition itself that propagate through execution.

A useful heuristic: if a capable model can solve the task with a single well-crafted prompt, decomposition is unnecessary overhead. If the task requires multiple tool interactions, involves multiple distinct subtasks, or exceeds what the model can hold in its context at once, decomposition is likely beneficial.

## 8. Hierarchical Planning

### 8.1 Concept

Hierarchical planning extends task decomposition with a management structure. Instead of a flat list of subtasks, the system creates a hierarchy of planners and executors. A high-level planner generates strategic goals, mid-level planners convert these into tactical plans, and low-level executors carry out specific actions. This mirrors organizational structures in human enterprises and military planning.

### 8.2 Architecture

```
function hierarchical_agent(goal, tools, model):
    # Level 1: Strategic planner
    strategy = strategic_plan(goal, model)
    
    results = []
    for strategic_goal in strategy.goals:
        # Level 2: Tactical planner
        tactical_plan = tactical_plan(strategic_goal, tools, model)
        
        tactical_results = []
        for task in tactical_plan.tasks:
            # Level 3: Executor
            result = react_agent(task, tools, model)
            tactical_results.append(result)
            
            # Tactical-level error handling
            if result.status == "failed":
                recovery = tactical_recover(
                    task, result, tactical_plan, model
                )
                tactical_plan.update(recovery)
        
        results.append(synthesize(tactical_results))
        
        # Strategic-level progress check
        if not strategy.on_track(results):
            strategy = strategic_replan(goal, results, model)
    
    return final_synthesis(results, model)
```

### 8.3 Model Assignment in Hierarchical Systems

A key design decision in hierarchical systems is which model to use at each level. Common approaches include:

**Capability-based assignment.** The most capable (and expensive) model handles strategic planning, where errors are most costly. Cheaper, faster models handle tactical execution, where individual errors can be caught and retried. For example, Claude Opus or GPT-4 for planning, Claude Haiku or GPT-4o-mini for execution.

**Specialization-based assignment.** Different models are chosen based on the domain of each subtask. A coding-specialized model handles code generation tasks, a reasoning-specialized model handles analytical tasks, and a general-purpose model handles coordination.

**Adaptive assignment.** The system starts with a cheaper model and escalates to a more capable model only when the cheaper model fails or the task is estimated to be sufficiently complex. This minimizes cost while maintaining quality for difficult subtasks.

### 8.4 Real-World Example: Devin-Style Coding Agents

Devin and similar autonomous coding agents use hierarchical planning extensively. At the strategic level, the system understands the overall software engineering task (implement a feature, fix a bug, refactor a module). At the tactical level, it breaks this into concrete steps: read relevant files, understand the codebase structure, write the implementation, write tests, run the tests, iterate. At the execution level, each step involves specific tool calls: file reads, code edits, terminal commands.

The hierarchical structure allows the agent to maintain strategic focus (achieving the overall goal) while handling tactical complexity (navigating a large codebase) and execution details (formatting code correctly, handling edge cases).

## 9. Observe-Think-Act Loops

### 9.1 Concept

The observe-think-act (OTA) loop is a generalization of the ReAct pattern that emphasizes the observation phase. Where ReAct focuses on interleaving reasoning with actions, OTA explicitly structures the system around a perceive-reason-act cycle inspired by cognitive science and robotics.

The distinction matters in systems where observation is non-trivial. In a web browsing agent, observation involves interpreting a screenshot or DOM structure—a complex perceptual task. In a coding agent, observation involves understanding the current state of the codebase, the test results, and the error messages. The OTA pattern gives observation first-class status rather than treating it as a simple tool output.

### 9.2 Pseudocode

```
function ota_agent(goal, environment, tools, model):
    state = initial_state()
    
    while not goal_achieved(state, goal):
        # Observe: Gather and interpret environmental state
        raw_observations = environment.observe()
        interpreted_state = model.interpret(
            raw_observations, 
            state.history,
            goal
        )
        state.update_observations(interpreted_state)
        
        # Think: Reason about what to do next
        reasoning = model.reason(
            state.observations,
            state.history,
            state.plan,
            goal
        )
        state.update_reasoning(reasoning)
        
        # Act: Execute the chosen action
        action = model.select_action(reasoning, tools)
        result = environment.execute(action)
        state.update_action(action, result)
        
        # Optional: Update plan based on new information
        if state.plan_needs_revision(result):
            state.plan = model.replan(state, goal)
    
    return state.final_output()
```

### 9.3 Application: Web Browsing Agents

Web browsing agents are the canonical application of the OTA pattern. The agent observes the current page (via screenshot analysis or DOM parsing), thinks about what action to take (click a button, fill a form, navigate to a URL), acts by executing the chosen browser action, and observes the result. The observation phase is critical because web pages are complex visual environments where the agent must identify relevant elements, understand page structure, and determine whether its previous action had the intended effect.

Systems like WebArena and VisualWebArena benchmark agents on this loop, and production agents from Anthropic (computer use), OpenAI (Operator), and others implement variations of the OTA pattern for web interaction.

## 10. Scratchpad Mechanisms

### 10.1 Concept

A scratchpad is a persistent working memory that the agent can write to and read from across multiple steps. Unlike the conversation history, which grows monotonically and cannot be edited, a scratchpad is a mutable workspace where the agent can store intermediate results, maintain running summaries, track task state, and organize information.

The scratchpad addresses a fundamental limitation of the conversation-as-memory paradigm: as the conversation grows, older information is pushed out of the context window or becomes difficult for the model to attend to effectively. A scratchpad allows the agent to maintain a compact, organized representation of its current state.

### 10.2 Implementation Patterns

**Key-value store.** The scratchpad is a dictionary of named values. The agent can set, get, update, and delete entries. This is simple and effective for tracking discrete pieces of information.

```
function scratchpad_agent(goal, tools, model):
    scratchpad = {}
    history = []
    
    while True:
        # Include scratchpad in context
        prompt = format_prompt(goal, scratchpad, history[-WINDOW:], tools)
        response = model.generate(prompt)
        
        actions = parse_actions(response)
        
        for action in actions:
            if action.type == "scratchpad_write":
                scratchpad[action.key] = action.value
            elif action.type == "scratchpad_read":
                # Value is already in context; this is a no-op
                pass
            elif action.type == "scratchpad_delete":
                del scratchpad[action.key]
            elif action.type == "tool_call":
                result = execute_tool(action.tool, action.args)
                history.append({"action": action, "result": result})
            elif action.type == "finish":
                return action.result
```

**Structured document.** The scratchpad is a structured document (e.g., markdown) that the agent can edit. This allows more complex organization—sections, lists, tables—and supports richer working memory.

**Append-only log with summarization.** The scratchpad is a log of observations and conclusions. Periodically, the agent summarizes the log into a compact form, discarding details that are no longer relevant. This balances the need for detailed working memory with the constraint of limited context.

### 10.3 Relationship to MemGPT

MemGPT (described in detail in the companion article on memory systems) is an influential system that formalizes the scratchpad concept. It provides the agent with explicit memory management operations—writing to a main context (working memory), archiving information to a longer-term store, and retrieving archived information when needed. The key innovation is treating context management as a task the model itself performs, rather than relying on fixed rules (like sliding windows) to manage what is in context.

## 11. Autonomous vs. Semi-Autonomous Agents

### 11.1 The Autonomy Spectrum

Agentic systems vary widely in how much autonomy they grant the model. The spectrum ranges from fully manual (human makes every decision, model only provides suggestions) to fully autonomous (model makes all decisions and takes all actions without human involvement).

In practice, most production systems are semi-autonomous: the model makes routine decisions independently but escalates to a human for high-stakes actions, ambiguous situations, or error recovery. The challenge is designing the escalation policy—deciding when the model should act independently and when it should ask for guidance.

### 11.2 Autonomy Patterns

**Human-in-the-loop.** Every action (or every action of a certain type) requires human approval before execution. The model proposes actions, and the human accepts, modifies, or rejects them. This maximizes safety but minimizes throughput and is only practical for low-frequency, high-stakes tasks.

```
function human_in_the_loop_agent(goal, tools, model):
    history = []
    
    while True:
        proposed_action = model.propose_action(goal, history, tools)
        
        # Present to human for approval
        human_decision = present_for_approval(proposed_action)
        
        if human_decision == "approve":
            result = execute_tool(proposed_action.tool, proposed_action.args)
        elif human_decision == "modify":
            modified_action = human_decision.modified_action
            result = execute_tool(modified_action.tool, modified_action.args)
        elif human_decision == "reject":
            result = {"status": "rejected", "feedback": human_decision.feedback}
        
        history.append({"action": proposed_action, "decision": human_decision, "result": result})
```

**Autonomous with guardrails.** The model acts independently but within defined boundaries. Actions that exceed the boundaries (e.g., deleting files, sending emails, spending money) trigger a confirmation request. Actions within the boundaries (e.g., reading files, running searches) are executed automatically.

```
function guardrailed_agent(goal, tools, model, policy):
    history = []
    
    while True:
        action = model.select_action(goal, history, tools)
        
        if action.type == "finish":
            return action.result
        
        if policy.requires_approval(action):
            approved = request_human_approval(action)
            if not approved:
                history.append({"action": action, "result": "blocked by policy"})
                continue
        
        result = execute_tool(action.tool, action.args)
        history.append({"action": action, "result": result})
```

**Fully autonomous with rollback.** The model acts independently on all decisions, but actions are executed in a sandboxed environment with the ability to roll back. After the task is complete, a human reviews the results and either accepts or reverts the changes. This is common in coding agents, where changes can be made on a branch and reviewed via pull request.

### 11.3 Choosing the Right Level of Autonomy

The appropriate level of autonomy depends on several factors:

- **Reversibility of actions.** Reversible actions (editing code on a branch) can be safely delegated. Irreversible actions (sending emails, deleting data) require more oversight.
- **Cost of errors.** If the worst-case failure is a wasted API call, full autonomy is reasonable. If the worst-case failure is a security breach or data loss, human oversight is essential.
- **Task complexity.** Simple, well-defined tasks can be safely automated. Complex, ambiguous tasks benefit from human guidance at key decision points.
- **Model reliability.** More capable models can be trusted with more autonomy. The autonomy level should be calibrated to the model's demonstrated reliability on similar tasks.
- **Frequency.** High-frequency tasks must be automated for practical reasons. Low-frequency tasks can afford human oversight.

## 12. Real Systems and Their Architectures

### 12.1 Claude Extended Thinking and Tool Use

Anthropic's Claude models implement agentic capabilities through two primary mechanisms: extended thinking and tool use.

**Extended thinking** implements the inner monologue pattern at the model level. When enabled, Claude generates a thinking trace before producing its response. The thinking trace includes reasoning about the task, evaluation of different approaches, identification of needed information, and planning of the response or tool use strategy. This thinking is separate from the response and is available to developers through the API.

**Tool use** implements a structured ReAct-like loop. Claude receives tool definitions (as JSON schemas), generates tool call requests when it needs to interact with external systems, receives tool results, and continues reasoning. The key architectural features are:

- Tool calls are structured (JSON) rather than free-form text, ensuring reliable parsing.
- Multiple tool calls can be generated in a single turn (parallel tool use).
- The model can chain tool calls across multiple turns, using results from earlier calls to inform later ones.
- The system prompt can constrain which tools are available and under what conditions.

Claude Code, Anthropic's coding agent, builds on these primitives with an architecture that combines extended thinking for complex reasoning with tool use for file operations, terminal commands, and code execution. The system uses a combination of plan-and-execute (for multi-step coding tasks) and reflection (running tests and iterating on failures).

### 12.2 OpenAI Function Calling and Assistants

OpenAI's agentic architecture is built around function calling and the Assistants API.

**Function calling** provides the primitive: the model receives function definitions, generates structured function call arguments, and the host application executes the calls and returns results. The function calling loop is the core of OpenAI's agentic pattern—the model generates a function call, the host executes it, the result is appended to the conversation, and the model continues.

**The Assistants API** adds persistence and orchestration. An Assistant is a configured agent with a system prompt, tools, and a persistent thread (conversation history). The Assistants API manages the agentic loop: when the model generates a tool call, the API can execute certain tools automatically (code interpreter, file search) or return the call to the host for execution (custom functions).

**Responses API** (introduced in 2025) represents OpenAI's latest iteration on agentic architecture. It provides built-in tools (web search, code execution, file search, computer use) and supports multi-step tool use with the model autonomously deciding when to invoke tools, processing the results, and generating additional tool calls as needed—all within a single API request.

**OpenAI's Agents SDK** provides an opinionated framework for building multi-step agentic workflows. It includes agent handoff mechanisms (one agent transferring control to another), guardrails (input/output validation), and tracing for observability.

### 12.3 AutoGPT and Autonomous Agent Planning

AutoGPT, launched in early 2023, was one of the first widely known autonomous agent systems. Its architecture is significant not because it was optimal but because it established patterns that influenced subsequent systems.

AutoGPT's core architecture is a loop with explicit planning and memory:

1. **Goal decomposition**: The user provides a high-level goal, and the system breaks it into subgoals.
2. **Action selection**: At each step, the model considers its goals, available commands (tools), and recent history to select an action.
3. **Execution**: The selected command is executed (web search, file operations, code execution, etc.).
4. **Memory storage**: Key findings and results are stored in a vector database for later retrieval.
5. **Progress evaluation**: The system evaluates whether the current subgoal has been achieved and whether to proceed, pivot, or escalate.

AutoGPT's primary contribution was demonstrating that an LLM could be embedded in an autonomous loop with meaningful tool access. Its primary limitation was reliability—the system frequently got stuck in loops, pursued irrelevant tangents, or failed to make progress. These failures highlighted the need for better planning, error recovery, and human oversight mechanisms.

### 12.4 Google's Agent Architectures

Google has contributed several research architectures and production systems:

**Gemini's tool use** follows a similar structured function calling pattern to Claude and GPT, with tool definitions, structured calls, and multi-turn chaining. Gemini's distinctive feature is deep integration with Google's tool ecosystem (Search, Maps, YouTube, etc.) and native multimodal capabilities that support visual observation in OTA loops.

**Vertex AI Agent Builder** provides a managed platform for building agents with structured planning, tool use, and conversation management. It supports both code-defined agents (using the Vertex AI SDK) and declaratively configured agents.

### 12.5 Open-Source Agent Systems

Several open-source systems have established important architectural patterns:

**BabyAGI** introduced the concept of an autonomous task management agent that maintains a task list, executes tasks, and generates new tasks based on results. Its architecture is essentially plan-and-execute with continuous replanning.

**CrewAI** implements multi-agent architectures with role-based agent assignment. Each agent has a defined role, goal, and backstory, and agents collaborate to complete tasks. The architecture supports both sequential (pipeline) and hierarchical (manager/worker) execution patterns.

**LangGraph** provides a framework for building agents as state machines. Each node in the graph performs a specific function (reasoning, tool execution, evaluation), and edges define the transitions between nodes. This approach makes the agent's control flow explicit and debuggable, in contrast to the implicit control flow of prompt-driven agents.

## 13. Choosing an Architecture

### 13.1 Decision Framework

Choosing the right agentic architecture requires evaluating the task along several dimensions:

**Task complexity.** Simple tasks (single tool call, straightforward reasoning) need no architecture beyond a basic tool use loop. Complex tasks (multi-step, requiring exploration) benefit from plan-and-execute or hierarchical planning.

**Error tolerance.** If errors are cheap (can be retried easily), simpler architectures are fine. If errors are costly (irreversible actions, user-facing outputs), add reflection and human-in-the-loop.

**Latency requirements.** ReAct and plan-and-execute are sequential and can be slow. If latency matters, consider parallel execution with DAG-based decomposition or simpler architectures with fewer steps.

**Predictability requirements.** If the agent's behavior must be predictable and auditable, explicit planning architectures are preferable to implicit ReAct reasoning. The plan artifact provides a clear record of intent.

**Context window constraints.** All agentic architectures consume context window capacity. Long-running agents with many steps will exhaust the context window. Scratchpad mechanisms, summarization, and hierarchical architectures help manage this constraint.

### 13.2 Architecture Selection Guide

| Task Type | Recommended Architecture | Reason |
|---|---|---|
| Simple Q&A with tool use | Basic tool loop | Minimal overhead, one or two tool calls |
| Research with multiple sources | ReAct | Need to iteratively search and synthesize |
| Multi-step coding task | Plan-and-execute + reflection | Need upfront planning and test-based validation |
| Complex project management | Hierarchical planning | Multiple levels of abstraction and coordination |
| Web navigation | OTA loop | Observation of visual state is primary challenge |
| Writing and revision | Reflection / self-critique | Iterative improvement is the core pattern |
| Long-running autonomous task | Scratchpad + adaptive planning | Must manage state across many steps |

### 13.3 Composing Architectures

In practice, production systems combine multiple patterns. A coding agent might use hierarchical planning at the top level, plan-and-execute for individual coding tasks, ReAct for tool interactions within each task, reflection for evaluating test results, and a scratchpad for maintaining state across the overall project. The architecture is not a single pattern but a composition of patterns at different levels.

The key design principle is to match the pattern to the level of abstraction. Strategic decisions benefit from planning. Tactical tool use benefits from ReAct. Quality assurance benefits from reflection. State management benefits from scratchpads. A well-designed agent composes these patterns appropriately.

## 14. Practical Considerations

### 14.1 Error Handling and Recovery

Errors are inevitable in agentic systems. Tools fail, APIs return unexpected results, the model makes incorrect decisions, and the environment changes between planning and execution. Robust error handling is not optional; it is a core architectural requirement.

Common error handling patterns include:

**Retry with backoff.** For transient errors (network failures, rate limits), simply retry the action after a delay. This is trivial to implement and handles a large fraction of errors in practice.

**Fallback tools.** If the primary tool fails, try an alternative. If a web search fails, try a different search engine. If a code execution environment fails, try a simpler approach.

**Error-informed replanning.** When an action fails, include the error information in the context and ask the model to select a different approach. The error message often contains useful information about what went wrong and how to fix it.

**Graceful degradation.** If the agent cannot complete the full task, it should complete as much as possible and clearly communicate what it accomplished and what it could not. A coding agent that fixes three of four failing tests has made meaningful progress even if it did not fully succeed.

### 14.2 Cost Management

Agentic architectures are inherently more expensive than single-turn inference. Each step of the loop requires a model call, and each model call processes the accumulated history plus the current step. The cost compounds: step N processes the context from all previous N-1 steps.

Cost management strategies include:

**Context pruning.** Periodically summarize the history and discard detailed step records. Keep only the information needed for the current phase of the task.

**Model tiering.** Use expensive models for planning and evaluation, cheap models for execution. Most of the value comes from good planning; execution is often straightforward.

**Early termination.** Set budgets (token count, step count, wall-clock time) and stop the agent when the budget is exhausted. Return whatever progress has been made.

**Caching.** Cache tool results to avoid redundant calls. If the agent reads the same file twice, the second read should be served from cache.

### 14.3 Observability and Debugging

Agentic systems are notoriously difficult to debug because the control flow is determined by the model at runtime. The same prompt can lead to different action sequences on different runs. Effective debugging requires comprehensive logging of every step: the prompt, the model's reasoning, the selected action, the tool result, and the state transitions.

Observability tools like LangSmith, Phoenix (Arize), and Braintrust provide visualization and analysis of agent traces. These tools display the full execution graph, highlight bottlenecks and failures, and enable comparison across runs. Investing in observability early in the development process pays dividends as the system grows in complexity.

### 14.4 Evaluation

Evaluating agentic systems is harder than evaluating single-turn model outputs. The final output depends on the entire trajectory of actions, and the same correct final output can be reached through efficient or wasteful paths. Evaluation must consider:

- **Task success rate.** Does the agent achieve the goal? This is the primary metric but is often binary and does not capture partial success.
- **Efficiency.** How many steps, tool calls, and tokens does the agent use? An agent that succeeds in 5 steps is better than one that succeeds in 50.
- **Robustness.** Does the agent succeed consistently, or does it fail unpredictably? Variance across runs is a critical metric.
- **Error recovery.** When the agent encounters errors, does it recover gracefully or get stuck?
- **Safety.** Does the agent stay within its boundaries? Does it respect human oversight requirements? Does it avoid dangerous actions?

Benchmarks like SWE-bench (for coding agents), WebArena (for web agents), and GAIA (for general agents) provide standardized evaluation protocols. However, benchmark performance does not always predict real-world performance, and task-specific evaluation is often necessary.

## 15. The Future of Agentic Architectures

### 15.1 Learned Architectures

Current agentic architectures are designed by humans—the loop structure, the planning strategy, the error handling policy are all hand-engineered. A promising research direction is learning the architecture itself. Given a distribution of tasks, can a meta-learning system discover the optimal loop structure, decomposition strategy, and error handling policy? Early work on learned tool use and learned planning suggests this is feasible, but the field is nascent.

### 15.2 Formal Verification

As agents take more consequential actions, the need for formal guarantees about their behavior increases. Can we prove that an agent will never take a harmful action? Can we verify that a plan is consistent with a set of constraints? Formal verification of agentic systems is extremely challenging because the model's decision-making is opaque, but progress in constrained decoding, output validation, and runtime monitoring provides building blocks for more rigorous safety guarantees.

### 15.3 Multi-Modal Agents

Current agentic architectures are predominantly text-based: the model reasons in text, generates text-based actions, and processes text-based observations. Multi-modal models that can process images, audio, and video open new possibilities for agentic architectures. An agent that can observe a screen (vision), listen to a conversation (audio), and interact with a physical environment (robotics) requires architectural patterns that handle multiple observation modalities and coordinate actions across modalities.

### 15.4 Persistent Agents

Current agents are typically ephemeral—they are created for a task, execute the task, and are discarded. Persistent agents that maintain long-term memory, learn from experience, and improve over time represent a significant architectural evolution. The challenges include managing unbounded state, preventing memory corruption, and ensuring that learned behaviors remain aligned with the user's goals.

## 16. Conclusion

Agentic architectures transform language models from passive text generators into active problem-solvers. The core patterns—ReAct, plan-and-execute, reflection, inner monologue, task decomposition, hierarchical planning, and observe-think-act—each address specific challenges in building autonomous systems. No single pattern is optimal for all tasks; effective agents compose multiple patterns, matching each pattern to the level of abstraction and the nature of the problem.

The field is maturing rapidly. Early autonomous agents like AutoGPT demonstrated the potential but also the fragility of LLM-based autonomy. Current production systems—Claude Code, OpenAI's Assistants, Google's agent platforms—have significantly improved reliability through better models, better architectures, and better engineering. But significant challenges remain: cost management, error handling, evaluation, safety, and the fundamental tension between autonomy and control.

For practitioners building agentic systems, the most important lesson is pragmatic: start with the simplest architecture that can accomplish the task, add complexity only when simpler approaches fail, invest heavily in observability and evaluation, and always include mechanisms for human oversight. The goal is not to build the most sophisticated agent but to build one that reliably accomplishes the task at hand.
