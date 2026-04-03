# Reasoning Models and Test-Time Compute

*April 2026*

## 1. Introduction

For most of the history of deep learning, improving model performance meant increasing training compute—training larger models on more data for more steps. The resulting models produced their output in a single forward pass, spending the same amount of computation on every problem regardless of difficulty. A simple factual question and a complex mathematical proof received the same computational budget at inference time.

Reasoning models break this paradigm. They spend variable amounts of computation at inference time, thinking longer about harder problems and producing rapid answers for easy ones. The mechanism is deceptively simple: the model generates a chain of reasoning tokens before producing its final answer, and the length of this chain scales with the difficulty of the problem. But the implications are profound. It means that model capability is no longer fixed at training time—it can be increased at inference time by allocating more computation. It means that a single model can operate at different capability levels depending on how much time and money the user is willing to spend. And it means that the scaling laws that govern AI progress now have a second axis: test-time compute.

This report provides a comprehensive technical examination of reasoning models and test-time compute. It covers the mechanics of chain-of-thought reasoning, the major reasoning models (OpenAI o1/o3, Claude extended thinking, DeepSeek-R1), the training techniques that produce reasoning capabilities, the scaling laws that govern test-time compute, and the practical implications for engineers deploying these models. The intended audience is engineers building applications with reasoning models and researchers studying the test-time compute paradigm.

## 2. The Shift to Variable Test-Time Compute

### 2.1 Fixed vs. Variable Compute

A standard transformer model performs a fixed number of operations per token: one forward pass through the model's layers. The total compute for generating a response is proportional to the number of output tokens times the model size. A 70B parameter model uses the same amount of compute per token whether it is completing a sentence or solving a differential equation.

This fixed-compute paradigm is fundamentally mismatched to the difficulty distribution of real-world tasks. Some tasks are trivial ("What is the capital of France?") and do not benefit from additional computation. Other tasks are extraordinarily difficult ("Prove that there are infinitely many twin primes") and would benefit from vastly more computation. A system that allocates the same compute to both is either wasting resources on easy tasks or under-investing in hard ones.

Reasoning models address this mismatch by generating a variable-length reasoning chain before producing the final answer. Easy problems receive short chains (or no chain at all). Hard problems receive long chains that may span thousands of tokens. The total compute scales with the difficulty of the problem, as assessed by the model during generation.

### 2.2 How This Differs from Chain-of-Thought Prompting

Chain-of-thought (CoT) prompting, introduced by Wei et al. in 2022, demonstrated that models perform better on reasoning tasks when prompted to "think step by step." This was the first evidence that generating intermediate reasoning tokens improves accuracy. However, CoT prompting has important limitations:

**Prompt-dependent.** The model only generates reasoning steps when prompted to do so. Without the right prompt, the model reverts to direct answer generation.

**Shallow training signal.** The model was not trained to reason step by step; it was trained to predict the next token. CoT prompting exploits emergent reasoning capability but does not optimize for it.

**No quality signal.** The model generates reasoning tokens, but there is no mechanism to evaluate whether the reasoning is correct. Bad reasoning is generated with the same confidence as good reasoning.

Reasoning models address all three limitations. They are trained to generate reasoning chains (not prompted to do so). The training process optimizes for the quality of the reasoning (not just the final answer). And the model learns to allocate more reasoning to harder problems (not the same amount to every problem).

### 2.3 The Two Scaling Axes

AI capability now scales along two axes:

**Training compute.** Larger models trained on more data for more steps. This axis determines the model's base capability—its knowledge, its ability to follow instructions, its grasp of language.

**Test-time compute.** More reasoning tokens generated at inference. This axis determines how effectively the model applies its base capability to specific problems—how deeply it can reason, how many approaches it can explore, how thoroughly it can verify its conclusions.

The interaction between these axes is multiplicative: a more capable base model benefits more from additional test-time compute, and more test-time compute benefits more from a stronger base model. This has important implications for the economics of AI deployment, as we discuss in Section 11.

## 3. Chain-of-Thought Mechanics

### 3.1 What Happens During Reasoning

When a reasoning model encounters a problem, it generates a sequence of reasoning tokens before producing the final answer. These tokens represent the model's "thought process"—a step-by-step progression toward the solution. The reasoning chain may include:

**Problem analysis.** The model restates the problem in its own words, identifies the key variables and constraints, and clarifies what is being asked.

**Strategy selection.** The model considers different approaches to the problem and selects the most promising one. It may explicitly compare approaches ("I could use dynamic programming or recursion; dynamic programming will be more efficient here...").

**Step-by-step execution.** The model works through the selected strategy step by step, showing intermediate calculations, logical deductions, or code execution traces.

**Self-verification.** The model checks its own work—re-doing calculations, verifying that conclusions follow from premises, testing edge cases.

**Error correction.** If the model detects an error in its reasoning, it backs up and tries a different approach. This backtracking is a key capability that distinguishes reasoning models from standard CoT prompting.

### 3.2 Example: Mathematical Reasoning

Consider a reasoning model solving: "If $f(x) = 2x^3 - 3x^2 + x - 5$, find $f'(2)$."

A standard model might directly output "13" or attempt a direct computation. A reasoning model generates something like:

```
I need to find f'(2), which means I need to:
1. First find the derivative f'(x)
2. Then evaluate it at x = 2

Step 1: Finding f'(x)
f(x) = 2x³ - 3x² + x - 5

Using the power rule:
- d/dx(2x³) = 6x²
- d/dx(-3x²) = -6x
- d/dx(x) = 1
- d/dx(-5) = 0

So f'(x) = 6x² - 6x + 1

Step 2: Evaluating f'(2)
f'(2) = 6(2)² - 6(2) + 1
     = 6(4) - 12 + 1
     = 24 - 12 + 1
     = 13

Let me verify: 6(4) = 24, 24 - 12 = 12, 12 + 1 = 13. ✓

The answer is 13.
```

The reasoning chain makes the model's logic explicit, enables verification, and reduces the probability of computational errors.

### 3.3 Reasoning Token Budget

Reasoning models allocate different numbers of tokens to different problems. This allocation is learned during training, not hard-coded. The model has learned correlations between problem features and the amount of reasoning required:

- Simple factual questions: 0-50 reasoning tokens
- Moderate reasoning tasks: 100-500 reasoning tokens
- Complex multi-step problems: 500-5,000 reasoning tokens
- Very difficult problems: 5,000-100,000+ reasoning tokens

Some APIs allow the user to set a maximum reasoning token budget, which controls the tradeoff between quality and latency/cost. A lower budget forces the model to reason more concisely (potentially sacrificing accuracy). A higher budget allows more thorough reasoning (at greater cost and latency).

## 4. OpenAI o1 and o3

### 4.1 o1: The First Reasoning Model

OpenAI's o1 (released September 2024) was the first commercially available reasoning model. It demonstrated that training a model specifically for chain-of-thought reasoning could dramatically improve performance on tasks requiring multi-step reasoning, particularly mathematics, coding, and science.

Key characteristics of o1:

**Hidden reasoning.** o1 generates a chain-of-thought reasoning trace that is not shown to the user. The user sees only the final answer (though a summary of the reasoning may be shown). This was a deliberate design choice—the reasoning trace may contain errors, dead ends, and backtracking that would be confusing if presented directly.

**Significant latency.** Because the model generates potentially thousands of reasoning tokens before the final answer, o1 responses take significantly longer than standard model responses. Simple tasks might take 5-10 seconds; complex tasks might take 30-60 seconds or more.

**Strong performance on reasoning benchmarks.** o1 showed dramatic improvements on math competitions (AIME, AMC), coding challenges, and science problems. On some benchmarks, it exceeded human expert performance.

**Weaker on non-reasoning tasks.** For tasks that do not benefit from extended reasoning—simple classification, straightforward text generation, factual recall—o1 performed similarly to or worse than GPT-4, while being slower and more expensive.

### 4.2 o3 and o4-mini

OpenAI's o3 (released early 2025) and o4-mini (released April 2025) represent the second generation of reasoning models:

**o3** improved on o1 with better reasoning quality, reduced latency, and the ability to use tools during reasoning. o3 can call functions, browse the web, and execute code as part of its reasoning process—not just before or after reasoning, but integrated into the reasoning chain itself. This tool-augmented reasoning significantly expands what the model can accomplish.

**o4-mini** provides reasoning capabilities in a smaller, faster, cheaper model. It makes reasoning accessible for applications that cannot afford the latency and cost of the full o3 model. The quality-cost tradeoff is explicit: o4-mini reasons less deeply but more affordably.

### 4.3 Configurable Reasoning Effort

OpenAI introduced configurable reasoning effort levels. Users can specify how much reasoning the model should perform:

- **Low effort**: Minimal reasoning, fast responses, suitable for simple tasks
- **Medium effort**: Moderate reasoning, balanced latency
- **High effort**: Thorough reasoning, higher latency, suitable for complex tasks

This gives developers explicit control over the quality-latency-cost tradeoff, allowing them to match the reasoning budget to the task difficulty.

## 5. Claude Extended Thinking

### 5.1 Architecture

Anthropic's approach to reasoning is implemented as "extended thinking" in Claude models. When extended thinking is enabled, the model generates a thinking trace before producing its response. Unlike OpenAI's approach, Claude's thinking traces are visible to the developer through the API (shown in thinking blocks), providing transparency into the model's reasoning.

### 5.2 Key Design Decisions

**Visible thinking.** Claude's thinking traces are available to the developer (and can be shown to the user). This transparency enables debugging, evaluation, and trust-building. Developers can inspect the reasoning to understand why the model reached a particular conclusion.

**Budget control.** Developers can set a maximum token budget for thinking. This controls the tradeoff between reasoning depth and latency/cost. A budget of 1,024 tokens allows brief reasoning; a budget of 128,000 tokens allows deep, extended reasoning.

**Streaming.** Extended thinking supports streaming—the thinking tokens are streamed as they are generated, so the developer can observe the reasoning in real-time. This is valuable for long-running reasoning tasks where the developer wants to monitor progress.

**Tool use integration.** Claude can use tools during extended thinking, integrating tool calls into the reasoning chain. The model can search for information, execute code, and read files as part of its reasoning process.

### 5.3 Thinking Trace Characteristics

Claude's thinking traces exhibit several characteristic behaviors:

**Problem decomposition.** The model breaks complex problems into sub-problems and addresses each one.

**Hypothesis generation and testing.** The model generates hypotheses about the answer and evaluates them against the evidence.

**Self-correction.** When the model detects an error in its reasoning, it explicitly notes the error and corrects course. "Wait, that's not right. I made an error in step 3. Let me redo this..."

**Exploration of alternatives.** For problems with multiple possible approaches, the model may explore several before committing to one.

**Verification.** After reaching a conclusion, the model often verifies it by checking the logic, testing edge cases, or approaching the problem from a different angle.

## 6. DeepSeek-R1

### 6.1 Significance

DeepSeek-R1 (January 2025) was significant for several reasons. It was the first open-weight reasoning model that matched or exceeded o1's performance on key benchmarks. Its training methodology was published in detail, providing the research community with insights into how reasoning capabilities can be trained. And it demonstrated that reasoning models could be built without proprietary techniques, accelerating the democratization of reasoning capabilities.

### 6.2 Training Methodology

DeepSeek-R1's training proceeded through several stages:

**Stage 1: Cold start.** A small number of high-quality chain-of-thought examples were used to fine-tune the base model (DeepSeek-V3), giving it the basic ability to generate reasoning chains.

**Stage 2: Reinforcement learning.** The model was trained using Group Relative Policy Optimization (GRPO) with rule-based rewards. The reward signal was based on the correctness of the final answer (for math problems, whether the answer matched the reference; for coding, whether the code passed test cases). Importantly, the reasoning chain itself was not directly supervised—the model was free to develop its own reasoning style as long as the final answers were correct.

**Stage 3: Rejection sampling and SFT.** The RL-trained model generated many reasoning traces, and the best ones (those leading to correct answers with clear, readable reasoning) were selected for supervised fine-tuning. This stage improved the readability and consistency of the reasoning chains.

**Stage 4: Second RL pass.** A final round of reinforcement learning refined the model's reasoning on a broader set of tasks.

### 6.3 Emergent Behaviors

During RL training, DeepSeek-R1 developed several interesting emergent behaviors:

**Self-verification.** The model spontaneously learned to verify its own answers, re-checking calculations and testing edge cases without being explicitly trained to do so.

**Backtracking.** The model learned to recognize when it was going down an unproductive path and backtrack to try a different approach. This behavior emerged from the RL reward signal—trajectories that reached correct answers after backtracking received positive reward.

**Extended reasoning.** The model learned to allocate more reasoning tokens to harder problems, spending thousands of tokens on complex mathematical proofs while answering simple questions concisely.

**Language mixing.** During early RL training, the model sometimes mixed languages or used shorthand notation in its reasoning. The subsequent SFT stage corrected this.

### 6.4 Distillation

DeepSeek published distilled versions of R1—smaller models (1.5B, 7B, 8B, 14B, 32B, 70B parameters) trained on R1's reasoning traces. These distilled models demonstrated remarkably strong reasoning for their size, suggesting that the reasoning capability can be transferred from large models to small ones through distillation. The 32B distilled model, in particular, achieved performance competitive with much larger models on reasoning benchmarks.

## 7. How Reasoning Training Works

### 7.1 The Training Pipeline

Training a reasoning model typically involves these stages:

1. **Pre-training**: Standard next-token prediction on a large corpus. This creates the base model with broad knowledge and language capability.

2. **Instruction fine-tuning**: Standard supervised fine-tuning on instruction-following data. This teaches the model to follow instructions and produce helpful responses.

3. **Reasoning data generation**: Creating chain-of-thought training data, either through human annotation, distillation from a stronger model, or self-play.

4. **Reasoning fine-tuning**: Supervised fine-tuning on reasoning data. This teaches the model the format and basic patterns of reasoning.

5. **Reinforcement learning**: Training with RL to improve reasoning quality. The model learns to generate better reasoning chains by receiving rewards for correct final answers.

6. **Optional distillation**: Using the reasoning model's traces to train smaller models.

### 7.2 Reinforcement Learning on Chain-of-Thought

The RL stage is where the model learns to reason effectively, rather than just imitating reasoning patterns. The key components are:

**Policy.** The language model itself, which generates reasoning tokens and final answers.

**Reward signal.** A signal that evaluates the quality of the model's output. This can be:

- **Outcome reward**: Based on whether the final answer is correct. Simple and reliable but provides sparse feedback (the model only knows if it was right or wrong, not where in the reasoning the error occurred).
- **Process reward**: Based on the correctness of individual reasoning steps. Richer feedback but requires evaluating each step, which is more complex and expensive.

**Optimization algorithm.** The algorithm used to update the model's weights based on the reward signal. Common choices include PPO (Proximal Policy Optimization), GRPO (Group Relative Policy Optimization), and variants.

### 7.3 Outcome Reward Models (ORMs)

An outcome reward model evaluates the final answer produced by the reasoning model. For mathematical problems, the ORM might simply check if the answer matches the reference. For open-ended tasks, the ORM is itself a trained model that evaluates answer quality.

```
function outcome_reward(question, reasoning_chain, final_answer, reference):
    if answer_matches(final_answer, reference):
        return 1.0  # Correct
    else:
        return 0.0  # Incorrect
```

**Advantages**: Simple, objective, easy to implement for domains with verifiable answers (math, coding).

**Disadvantages**: Sparse feedback—the model does not know where in its reasoning chain the error occurred. Does not reward good reasoning that leads to wrong answers due to a final calculation error. Does not penalize bad reasoning that happens to reach the correct answer.

### 7.4 Process Reward Models (PRMs)

A process reward model evaluates individual reasoning steps, providing step-by-step feedback on the quality of the reasoning chain.

```
function process_reward(question, reasoning_steps):
    rewards = []
    for step in reasoning_steps:
        step_quality = prm.evaluate(question, reasoning_steps[:step], step)
        rewards.append(step_quality)
    return rewards
```

**Advantages**: Dense feedback—the model knows which steps were good and which were wrong. Rewards good reasoning even if the final answer is incorrect. Penalizes lucky guesses.

**Disadvantages**: Training a PRM requires step-level annotations, which are expensive to produce. The PRM itself can make errors, propagating incorrect feedback. The definition of a "correct step" is subjective for many reasoning tasks.

### 7.5 The PRM vs. ORM Debate

Research has shown that PRMs generally outperform ORMs for training reasoning models, particularly on tasks where the reasoning chain is long and the final answer is a small fraction of the total output. However, the cost of training and deploying PRMs is significantly higher. In practice, many systems use a combination:

- ORM for the initial RL training (cheap, objective)
- PRM for refinement (expensive, but provides better signal)
- ORM for evaluation (since it directly measures task success)

### 7.6 Self-Play and Iterative Improvement

Some training approaches use self-play: the model generates reasoning traces, evaluates them (using an ORM or PRM), and uses the best traces as training data for the next iteration. This creates a virtuous cycle where the model's reasoning improves with each iteration, generating better training data that further improves reasoning.

```
function iterative_self_play(model, problems, iterations):
    for i in range(iterations):
        traces = []
        for problem in problems:
            # Generate multiple reasoning traces
            candidates = [model.reason(problem) for _ in range(N)]
            
            # Evaluate and select the best
            scores = [evaluate(candidate, problem) for candidate in candidates]
            best = candidates[argmax(scores)]
            traces.append((problem, best))
        
        # Fine-tune model on best traces
        model = fine_tune(model, traces)
    
    return model
```

## 8. Test-Time Compute Scaling Laws

### 8.1 The Scaling Relationship

Research from OpenAI, DeepSeek, and others has established that reasoning model performance scales predictably with test-time compute. Specifically:

**Log-linear scaling.** Performance (measured by accuracy on reasoning benchmarks) scales approximately log-linearly with the number of reasoning tokens. Doubling the reasoning budget improves accuracy by a roughly constant amount.

**Diminishing returns.** The marginal improvement from additional reasoning tokens decreases as the total budget increases. The first 100 tokens of reasoning provide a large improvement. The next 1,000 provide a smaller improvement per token. Beyond 10,000 tokens, the improvement per token is small for most problems.

**Problem-dependent ceiling.** Each problem has a difficulty ceiling—a level of reasoning beyond which additional tokens do not help. For easy problems, this ceiling is reached quickly. For very hard problems, it may not be reached within practical token budgets.

### 8.2 Compute-Optimal Allocation

Given a fixed inference budget, how should compute be allocated across problems? Research suggests:

**Allocate proportionally to difficulty.** Spend more reasoning on harder problems and less on easier ones. This outperforms uniform allocation because the marginal value of reasoning is higher for hard problems.

**Use difficulty estimation.** Before allocating the full reasoning budget, estimate the problem's difficulty (using a fast initial pass or a difficulty classifier). Allocate reasoning tokens accordingly.

**Adaptive allocation.** Allow the model to decide how much reasoning to do, rather than setting a fixed budget. The model naturally generates more tokens for harder problems (this is what reasoning models are trained to do). Setting a maximum budget provides a safety valve against runaway computation.

### 8.3 Test-Time Compute vs. Training Compute

A key finding is that test-time compute and training compute are partially substitutable. A smaller model with more reasoning can match the performance of a larger model with less reasoning on many tasks. This has important economic implications:

- For rare, high-value tasks (e.g., solving a novel mathematical problem), investing in test-time compute is efficient because the cost is incurred only when the problem appears.
- For frequent, routine tasks (e.g., classifying customer support tickets), investing in training compute (larger model or more training) is efficient because the per-task inference cost must be low.

The optimal balance depends on the task frequency, difficulty distribution, and cost constraints of the specific application.

## 9. Latency-Quality Tradeoff

### 9.1 The Fundamental Tension

Reasoning models trade latency for quality. More reasoning means better answers but longer wait times. This tradeoff is manageable for batch processing or asynchronous tasks but can be problematic for interactive applications where users expect responses in seconds.

### 9.2 Latency Characteristics

Typical latency ranges for reasoning models:

| Task Difficulty | Reasoning Tokens | Typical Latency |
|---|---|---|
| Trivial | 0-50 | 1-2 seconds |
| Simple | 50-200 | 2-5 seconds |
| Moderate | 200-1,000 | 5-15 seconds |
| Complex | 1,000-5,000 | 15-45 seconds |
| Very complex | 5,000-50,000 | 45-180 seconds |

These numbers vary with model size, hardware, and provider. But the general pattern is that reasoning models can take 10-100x longer than non-reasoning models for complex tasks.

### 9.3 Managing Latency

**Streaming.** Stream the thinking trace (where visible) and the final answer to provide a sense of progress. Users tolerate longer waits when they can see the model working.

**Budget caps.** Set maximum reasoning token budgets to bound latency. Accept potentially lower quality in exchange for predictable response times.

**Difficulty routing.** Route easy tasks to fast, non-reasoning models and hard tasks to reasoning models. A classifier determines task difficulty and selects the appropriate model.

**Speculative reasoning.** Begin generating the reasoning trace while the user is still typing, using the partial input to predict the full question. When the user submits, the model may already have completed part of its reasoning.

**Caching reasoning patterns.** For tasks with common patterns (e.g., many users ask similar math questions), cache reasoning traces and adapt them to new instances rather than generating from scratch.

## 10. Faithfulness of Reasoning Chains

### 10.1 The Faithfulness Problem

A critical question about reasoning models is whether the visible reasoning chain actually reflects the model's decision-making process. If the model reaches its answer through one process and then generates a reasoning chain that rationalizes the answer through a different process, the chain is unfaithful—it is a post-hoc explanation rather than a genuine trace of the model's reasoning.

### 10.2 Evidence for Unfaithfulness

Several lines of evidence suggest that reasoning chains can be unfaithful:

**Biased reasoning.** When the model has a strong prior about the answer (due to training data), it may generate reasoning that supports this prior regardless of the actual evidence. The reasoning chain becomes a rationalization rather than a derivation.

**Correct conclusions from wrong reasoning.** Models sometimes generate reasoning chains that contain errors but still reach the correct answer. This suggests that the answer was determined by a process other than the visible reasoning.

**Sycophantic reasoning.** When a user suggests an answer before asking the model to reason, the model tends to generate reasoning that supports the user's suggestion, even when it is wrong. The reasoning is being shaped by the desired conclusion rather than the other way around.

**Steganographic concerns.** Theoretical work has raised the possibility that models could encode information in the reasoning chain that is not visible to human readers but influences the model's subsequent generation. This would make the visible reasoning chain incomplete or misleading.

### 10.3 Implications

If reasoning chains are not faithful, their value for transparency and debugging is limited. A developer who inspects a wrong reasoning chain and tries to understand the error may be misled if the chain does not reflect the actual cause of the error.

Research on improving faithfulness is active. Approaches include:

- Training models specifically for faithfulness (rewarding reasoning chains that causally influence the answer)
- Probing the model's internal representations to verify that they are consistent with the reasoning chain
- Testing faithfulness by modifying the reasoning chain and checking whether the answer changes accordingly
- Evaluating faithfulness using perturbation studies—changing input details and verifying that the reasoning chain changes appropriately

## 11. Cost Implications

### 11.1 The Cost Structure

Reasoning models have a distinctive cost structure:

**Input tokens.** The same as any model—the cost of processing the input prompt.

**Reasoning tokens.** The cost of generating the reasoning chain. These tokens are generated by the model and consume compute, but many providers charge for them at a different rate than output tokens (some charge the same rate, others offer reduced pricing for reasoning tokens, and some include them in the output token count).

**Output tokens.** The cost of generating the final answer.

For complex tasks, reasoning tokens can dominate the total cost. A task that requires 10,000 reasoning tokens and 500 output tokens has 20x more reasoning tokens than output tokens. If reasoning tokens are priced equally to output tokens, the cost is approximately 20x what a non-reasoning model would charge.

### 11.2 Cost Comparison

Approximate cost per problem for a moderately complex reasoning task (early 2026 pricing):

| Model | Reasoning Tokens | Output Tokens | Approx. Cost |
|---|---|---|---|
| GPT-4o (no reasoning) | 0 | 500 | $0.005 |
| o3 (medium effort) | 5,000 | 500 | $0.06-0.10 |
| o3 (high effort) | 20,000 | 500 | $0.25-0.40 |
| Claude 3.5 Sonnet (no thinking) | 0 | 500 | $0.008 |
| Claude 3.7 Sonnet (extended thinking) | 10,000 | 500 | $0.05-0.15 |
| DeepSeek-R1 (self-hosted) | 10,000 | 500 | Variable (compute cost) |

The cost differential is significant—10-50x for reasoning models compared to non-reasoning models. This makes reasoning models impractical for high-volume, low-value tasks but highly cost-effective for low-volume, high-value tasks where accuracy matters.

### 11.3 Cost Optimization

**Task routing.** Use a cheap model to classify task difficulty and route only hard tasks to reasoning models. This reduces the average cost per task while maintaining quality for hard tasks.

**Budget tuning.** Experiment with different reasoning budgets to find the minimum budget that achieves acceptable quality for each task type.

**Batch processing.** For non-interactive tasks, batch problems and process them during off-peak hours when inference capacity is cheaper.

**Model distillation.** Distill reasoning traces from large models into smaller models that can reason at lower cost. DeepSeek's success with distilled R1 models demonstrates the viability of this approach.

## 12. Limitations

### 12.1 Overthinking

Reasoning models sometimes overthink simple problems, generating lengthy reasoning chains for tasks that should be answered directly. A model might spend 1,000 tokens reasoning about "What is 2 + 2?" when "4" is the immediate correct answer. This wastes tokens and latency without improving quality.

Overthinking is partly a training artifact—the model is rewarded for reasoning, so it reasons even when unnecessary—and partly a difficulty estimation failure—the model misjudges the problem's difficulty.

### 12.2 Hallucinated Reasoning

Just as models can hallucinate facts, they can hallucinate reasoning—generating plausible-looking but incorrect logical steps. A model might apply a mathematical technique incorrectly, use a theorem that does not apply, or draw a conclusion that does not follow from the premises. The reasoning looks convincing, making the error harder to detect than a direct incorrect answer.

### 12.3 Reasoning Loops

Reasoning models can get stuck in loops, repeatedly reconsidering the same approach without making progress. The model generates reasoning, reaches an inconclusive state, backtracks, tries the same approach again, reaches the same inconclusive state, and so on. This consumes tokens without converging on an answer.

### 12.4 Sensitivity to Problem Framing

Reasoning model performance can be sensitive to how the problem is framed. Rephrasing the same problem in a different way can lead to different (and differently correct) reasoning chains. This sensitivity makes reasoning models less predictable than non-reasoning models, complicating application design and testing.

### 12.5 Limited Self-Knowledge

Reasoning models do not have reliable knowledge of their own capabilities. A model may confidently generate a long reasoning chain for a problem that is fundamentally beyond its capability, consuming resources without producing a correct answer. Better calibration—the model's ability to assess whether it can solve a problem—would improve the efficiency of reasoning allocation.

## 13. Distilling Reasoning

### 13.1 The Distillation Approach

Reasoning distillation transfers the reasoning capability of a large teacher model to a smaller student model. The teacher generates reasoning traces for a set of problems, and the student is trained (via supervised fine-tuning) to reproduce those traces. The student learns the teacher's reasoning patterns without going through the expensive RL training process.

### 13.2 Effectiveness

DeepSeek's distillation experiments showed striking results:

- A 7B parameter model distilled from R1 outperformed larger non-reasoning models on reasoning benchmarks
- A 32B distilled model approached R1's full performance on many tasks
- Even a 1.5B distilled model showed meaningful reasoning capability

These results suggest that reasoning capability is more about the training signal (learning to generate step-by-step reasoning) than about model scale (having more parameters). A small model that has learned to reason can outperform a large model that has not.

### 13.3 Limitations of Distillation

**Capability ceiling.** The distilled model cannot exceed the teacher's capability. If the teacher makes errors on certain types of problems, the student inherits those errors.

**Format without substance.** The student may learn the format of reasoning (generating step-by-step text) without fully learning the substance (actually performing valid reasoning). This produces reasoning chains that look correct but contain logical errors.

**Domain transfer.** Reasoning distilled on mathematical problems may not transfer well to coding or scientific reasoning. Domain-specific distillation data is often needed.

## 14. Practical Considerations for Deployment

### 14.1 When to Use Reasoning Models

Reasoning models are appropriate when:

- The task requires multi-step reasoning (math, coding, logic)
- Accuracy matters more than latency
- The cost per task is justified by the value of the result
- The task is too complex for a standard model to solve reliably

Reasoning models are not appropriate when:

- The task is simple and does not benefit from reasoning
- Low latency is critical
- High volume makes reasoning costs prohibitive
- The task requires knowledge the model does not have (reasoning does not create knowledge)

### 14.2 Configuring Reasoning

Key configuration parameters:

**Thinking budget.** Set appropriately for the task difficulty. Start with a moderate budget and adjust based on accuracy-latency tradeoffs.

**Temperature.** Reasoning models often perform best with low temperature (0.0-0.3) for the reasoning chain, as creative variation in reasoning is usually harmful. Some implementations use different temperatures for reasoning and final output.

**Streaming.** Enable streaming for interactive applications to provide feedback during the reasoning phase.

**Tool use during reasoning.** If available, enable tool use during reasoning so the model can ground its reasoning in real data (calculator, code execution, web search).

### 14.3 Evaluating Reasoning Quality

Beyond task accuracy, evaluate:

- **Reasoning correctness.** Are the individual reasoning steps valid?
- **Efficiency.** Does the model reach the answer with a reasonable number of reasoning tokens?
- **Consistency.** Does the model produce consistent reasoning across similar problems?
- **Calibration.** Does the model express appropriate confidence in its reasoning?

### 14.4 Combining Reasoning with Agent Architectures

Reasoning models integrate naturally with agentic architectures. The reasoning trace can include tool calls, observations, and planning—not just abstract logical reasoning. A reasoning model embedded in an agent loop can:

1. Reason about the task (what needs to be done)
2. Call tools to gather information (during reasoning)
3. Reason about the information (what does it mean)
4. Plan the next action (what to do next)
5. Execute the action and observe the result
6. Continue reasoning based on the new information

This integration of reasoning and acting is a key frontier for agentic systems.

## 15. The Future of Reasoning Models

### 15.1 Deeper Reasoning

Current reasoning models can solve problems that require tens of reasoning steps. Future models may be capable of hundreds or thousands of coherent reasoning steps, enabling solutions to problems that currently exceed model capability—complex proofs, long-horizon planning, scientific discovery.

### 15.2 Multi-Modal Reasoning

Reasoning that integrates text, images, code, and mathematical notation in a unified reasoning chain. A model that can look at a diagram, reason about its structure, write code to test a hypothesis, and produce a mathematical proof—all in one reasoning chain.

### 15.3 Collaborative Reasoning

Multiple models reasoning together, with each model contributing specialized expertise. A mathematics model, a coding model, and a scientific knowledge model collaborating on a complex research problem, each reasoning in its domain and sharing conclusions.

### 15.4 Verifiable Reasoning

Reasoning chains that are machine-verifiable—each step can be formally checked for logical validity. This would provide guarantees that the reasoning is correct, not just plausible. Integration with formal verification systems (proof assistants like Lean, Isabelle) is one path toward this goal.

## 16. Conclusion

Reasoning models represent a paradigm shift in how language models approach difficult problems. By spending variable computation at inference time, these models can adapt their reasoning effort to the difficulty of the problem, achieving dramatically better performance on tasks requiring multi-step reasoning. The training techniques—reinforcement learning on chain-of-thought traces, process reward models, self-play—have proven effective across multiple research groups and model architectures.

The practical implications are significant. Engineers now have a second lever for improving AI performance: in addition to choosing a more capable model (training compute), they can allocate more reasoning at inference time (test-time compute). The tradeoff between quality, latency, and cost is explicit and configurable. For high-value tasks where accuracy matters, reasoning models offer a substantial capability improvement at a manageable cost. For high-volume, low-latency tasks, standard models remain the better choice.

The field is evolving rapidly. The gap between reasoning and non-reasoning models is widening on complex tasks. Distillation is making reasoning accessible at smaller scales. And the integration of reasoning with tool use and agent architectures is creating systems that can think, act, and learn in increasingly sophisticated ways. For practitioners, the key is to understand where reasoning adds value, to configure it appropriately for each use case, and to evaluate reasoning quality rigorously—not just whether the answer is correct, but whether the reasoning that produced it is faithful, efficient, and trustworthy.
