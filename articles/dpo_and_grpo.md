# Direct Preference Optimization and GRPO: The Post-PPO Era of LLM Alignment

*April 2026 • Technical Report*

## 1. Introduction

For the first three years after RLHF became the dominant approach to LLM alignment, "RLHF" was effectively synonymous with PPO — Proximal Policy Optimization, the on-policy reinforcement learning algorithm popularized by OpenAI in 2017 and adapted to language models in InstructGPT (2022). The pipeline was: collect preference data, train a reward model, then run PPO with the reward model as the optimization signal. The approach worked, and produced ChatGPT, GPT-4, and Claude — but it was complicated, expensive, and finicky. Training a reward model was hard. Tuning PPO hyperparameters was harder. Reward hacking was a constant risk. The whole pipeline required infrastructure that few organizations outside the big labs could afford.

In May 2023, a paper from Stanford titled "Direct Preference Optimization: Your Language Model is Secretly a Reward Model" argued that you could skip most of this. By rearranging the math of RLHF, the authors showed that the entire two-stage pipeline (reward model training + PPO) could be collapsed into a single supervised loss applied directly to preference pairs. No reward model. No reinforcement learning. No PPO hyperparameters to tune. The result was DPO, which over the following two years displaced PPO as the default approach for fine-tuning open-source models on preference data.

A year and a half later, a different shift occurred. DeepSeek's R1 paper (2024) demonstrated that for reasoning-heavy tasks where the reward signal is verifiable (math problems with known answers, code with passing tests), a simpler reinforcement learning approach called GRPO (Group Relative Policy Optimization) outperformed both DPO and PPO. GRPO has since become the standard recipe for training reasoning models, used by DeepSeek, Qwen, and several other frontier teams.

This report examines DPO, GRPO, and the broader landscape of preference and reward optimization methods that have largely replaced classical RLHF.

## 2. Why PPO Was Hard

To understand why DPO and GRPO took over so quickly, it helps to understand what was painful about PPO. PPO is a general-purpose reinforcement learning algorithm, and applying it to language models requires several layers of adaptation that introduce their own problems.

### 2.1 The Reward Model Step

PPO needs a scalar reward for each output, and human preferences come as pairwise comparisons. Bridging this requires training a reward model — a neural network that takes a prompt and a response and outputs a scalar score, learned from preference pairs by minimizing a Bradley-Terry loss. The reward model is itself a sizable LLM (often the same architecture as the policy being trained), and it needs to be reasonably well-calibrated for PPO to work.

The reward model is the leakiest part of the pipeline. It is trained on a finite preference dataset and overfits to spurious patterns ("prefer longer responses," "prefer responses that hedge"). It generalizes poorly out of distribution. It can be exploited by the policy through reward hacking — finding outputs that score well according to the reward model but are unhelpful or wrong by human standards. Mitigating these issues requires careful dataset curation, regularization, and ensembling, none of which are trivial.

### 2.2 The PPO Loop

Once you have a reward model, you run PPO. The policy generates responses to prompts, the reward model scores them, and the policy is updated to increase the probability of high-reward responses while staying close to a reference policy (to avoid drifting too far from the original model). The "staying close" is enforced through a KL penalty term, which has its own hyperparameter that must be tuned.

PPO requires several copies of the model in memory simultaneously: the policy being trained, a reference policy (frozen copy of the original), the reward model, and (in some implementations) a value function for advantage estimation. For a 70B-parameter model, this can mean 4× the memory of normal supervised training, plus the overhead of generating samples on-the-fly. The compute cost of PPO training is typically 10-30× higher than the equivalent supervised fine-tuning step.

### 2.3 Hyperparameter Sensitivity

PPO has roughly a dozen hyperparameters that interact in non-obvious ways: learning rate, KL coefficient, clipping threshold, advantage normalization, generation temperature, batch size, number of PPO epochs per batch, value function coefficient, and so on. Getting these right requires extensive sweeps and experience that took years for the community to accumulate. A misconfigured PPO run could collapse the model entirely (the policy diverges to nonsense) or fail to make any progress (the policy never moves from the reference).

The result was that PPO worked, but only for teams who had invested heavily in the infrastructure. The barrier to entry was high enough that most academic groups and smaller companies stuck with supervised fine-tuning and never fine-tuned on preference data at all.

## 3. Direct Preference Optimization

### 3.1 The Key Insight

The DPO paper begins with a remarkable observation. The optimal policy under RLHF — the one that maximizes the reward model's reward subject to the KL constraint — has a closed-form expression in terms of the reference policy and the reward function. Specifically:

```
π*(y|x) ∝ π_ref(y|x) · exp(r(x,y) / β)
```

This says: the optimal policy assigns probabilities proportional to the reference policy's probabilities, multiplied by an exponential of the reward (scaled by the KL coefficient β). This is a standard result from constrained optimization, but the DPO authors' key move was to invert it: given a target policy and a reference policy, you can solve for the implicit reward function:

```
r(x,y) = β · log(π(y|x) / π_ref(y|x)) + constant
```

This means that any policy implies a reward function. If you have preference data (chosen and rejected responses for each prompt), you can directly optimize the policy to assign higher implicit reward to chosen responses than to rejected ones — without ever explicitly training a reward model.

### 3.2 The DPO Loss

The DPO loss takes preference pairs (x, y_chosen, y_rejected) and minimizes:

```
L_DPO = -log(σ(β · [log(π(y_c|x)/π_ref(y_c|x)) - log(π(y_r|x)/π_ref(y_r|x))]))
```

In words: increase the log-ratio of the policy probability to the reference policy probability for the chosen response, and decrease it for the rejected response. The loss is fully differentiable and can be optimized by ordinary supervised training tooling — no rollouts, no value function, no separate reward model.

### 3.3 Why DPO Works

DPO's mathematical equivalence to RLHF is approximate (the original paper claimed exact equivalence under certain assumptions, which were later refined). In practice, DPO produces policies that are similar in quality to PPO-trained policies on the same preference data, with substantially lower compute cost and complexity. The training is stable, the hyperparameters (mainly the β coefficient) are easy to tune, and the implementation fits in a few hundred lines of code.

The catch is that DPO is sensitive to the quality of the preference data. Because there is no reward model to abstract over noise in individual preferences, every preference pair directly affects the loss. Datasets with ambiguous preferences, mislabeled examples, or distributional skew can produce DPO models with subtle quality regressions. Reward-model-based RLHF averaged over many preferences before producing a reward signal, providing some robustness; DPO has less of this averaging.

### 3.4 Variants and Extensions

DPO spawned a family of variants:

- **IPO (Identity Preference Optimization)**: Replaces the log-sigmoid in DPO with an identity-based loss to reduce overfitting to noisy preferences.
- **KTO (Kahneman-Tversky Optimization)**: Optimizes against unpaired preference signals (just "good" or "bad" labels rather than pairs), making it useful for datasets where pairwise comparisons are not available.
- **SimPO**: Removes the reference policy from the loss, simplifying training and improving performance in some benchmarks.
- **ORPO**: Combines preference optimization with a supervised fine-tuning objective in a single loss.
- **DPO with SFT regularization**: Adds a supervised fine-tuning loss on the chosen responses to prevent the model from learning to suppress chosen responses (a known DPO failure mode).

The proliferation of variants reflects ongoing research into the right loss for preference data. As of 2025, the dominant production choice for preference fine-tuning of open-source models is DPO with mild SFT regularization, though SimPO and ORPO are gaining traction.

## 4. The Limits of DPO

DPO is excellent for the preference-data setting it was designed for, but it has limitations.

### 4.1 No Online Exploration

DPO is offline: it learns from a fixed preference dataset and never generates new samples during training. PPO, by contrast, is on-policy: it generates new responses, scores them, and updates the policy on those. This matters when the preference data does not cover the response distribution well — DPO can only optimize within the support of its training data, while PPO can explore new regions of response space.

For tasks where the model's behavior is mostly within the training distribution (instruction following, general chat), this distinction does not matter much. For tasks requiring genuine exploration of new strategies (reasoning, agentic behavior), the on-policy nature of PPO becomes more valuable.

### 4.2 No Verifiable Rewards

DPO uses preference data, which is inherently subjective and noisy. For tasks with verifiable rewards — math problems with known answers, code that either passes tests or doesn't, theorem-proving that either succeeds or fails — preferences are an inefficient signal. The reward is already known with certainty; turning it into a preference pair (sample two responses, mark the correct one as preferred) discards information.

This is the regime where DPO's advantages over reward-model-based RL evaporate, because the "reward model" is just an objective scorer that you don't need to train. And it is the regime where DeepSeek's GRPO took over.

## 5. GRPO: Group Relative Policy Optimization

### 5.1 The Setting

GRPO, introduced in the DeepSeekMath paper (2024) and subsequently refined in the DeepSeek-R1 paper (2025), targets tasks where you can verify whether a response is correct. The setting is typically:

- A math problem with a known final answer
- A coding task with unit tests
- A logical reasoning task with a verifiable answer
- A multi-step task where success is checkable

For each prompt, you sample multiple responses from the current policy (typically 8-64 samples per prompt). You score each response using the verifier — usually a simple function returning 1 if correct, 0 if incorrect, with optional partial credit. You then compute a per-response advantage relative to the group's mean reward, and update the policy to increase the probability of above-average responses and decrease the probability of below-average ones.

### 5.2 The Loss

GRPO's loss is essentially a simplified PPO loss with a group-relative advantage estimate:

```
L_GRPO = -E[ ratio · A - β · KL ]
```

where:
- `ratio` is the probability ratio of the new policy to the old policy for each token
- `A` is the advantage, computed as (reward - mean_reward_in_group) / std_reward_in_group
- `β · KL` is a KL penalty against the reference policy

The key innovations relative to PPO are:

1. **No value function**: PPO uses a learned value function to estimate the baseline for advantage computation. GRPO uses the empirical mean of the group as the baseline, eliminating the need for a value network entirely. This saves memory and avoids the difficulty of training a stable value function.
2. **Group-based normalization**: Advantages are normalized within each group (each set of samples for the same prompt), making the loss scale-invariant and more stable across diverse prompts.
3. **Verifiable reward**: The reward comes from a deterministic verifier, not a learned reward model, eliminating reward hacking and reward model maintenance.

### 5.3 Why GRPO Works for Reasoning

The DeepSeek-R1 results demonstrated that GRPO could elicit complex reasoning behaviors — chain-of-thought, self-correction, exploration of multiple solution paths — by simply rewarding correct final answers, with no explicit supervision of the reasoning process. The model learned to think through problems step-by-step because step-by-step thinking was rewarded by leading to correct answers more often.

This was a striking result. It suggested that the reasoning capabilities visible in models like o1 and R1 are not the product of carefully labeled chain-of-thought data, but the emergent product of a simple reinforcement learning objective applied to verifiable rewards. The recipe was elegant: collect or generate a large dataset of math and coding problems, run GRPO with binary reward, and let the model figure out the reasoning strategy on its own.

### 5.4 Computational Profile

GRPO is more expensive than DPO (because it requires generating samples on-policy) but cheaper than PPO (because it skips the value function and reward model). A typical GRPO training run uses 8-32 samples per prompt and ~10× the compute of an equivalent supervised fine-tuning run. The memory footprint is roughly 2× that of supervised training (policy + reference), much less than the 4-5× of PPO.

The on-policy sampling is the dominant cost. For each training step, the model generates 8-32 completions for each prompt in the batch, which takes substantially longer than a forward pass. Frameworks like TRL, OpenRLHF, and verl have optimized this with batch-aware generation, KV cache reuse, and integration with high-throughput inference engines like vLLM during sampling.

## 6. Hybrid and Alternative Approaches

### 6.1 RLHF Variants Combining DPO and PPO

Some teams use DPO for the initial alignment phase and PPO (or GRPO) for a subsequent refinement phase. The DPO step gets the model to a reasonable starting point cheaply, and the on-policy step polishes the behavior in regions where DPO's offline distribution is insufficient.

### 6.2 Expert Iteration

Expert iteration is an older technique that has seen renewed interest as a complement to GRPO. It generates samples, filters out the correct ones, and uses them as supervised training data for the next iteration. There is no policy gradient, no reward model, no advantage estimation — just supervised training on filtered samples. This is essentially a degenerate form of reinforcement learning that works surprisingly well for some reasoning tasks.

### 6.3 RLAIF and Self-Improvement Loops

RL from AI Feedback (RLAIF) replaces human preference labelers with another LLM, allowing preference data to be generated cheaply. Combined with DPO or GRPO, this enables self-improvement loops where the model generates and evaluates its own training data. The risk is reward hacking against the AI labeler, which has been observed in practice.

### 6.4 Process Reward Models

For multi-step reasoning, some work has explored process reward models that score each step of a reasoning chain rather than just the final answer. These provide denser learning signals and can speed up training, at the cost of needing labeled process data. The DeepSeek-R1 paper notably reported that process reward models did not outperform GRPO with outcome-only rewards in their setting.

## 7. Practical Recipes

### 7.1 Open Source Toolchains

The TRL library from Hugging Face has emerged as the dominant open-source framework for preference optimization. It supports DPO, IPO, KTO, ORPO, PPO, and GRPO with consistent APIs and efficient implementations. OpenRLHF and verl provide alternative implementations focused on scalability for larger models. NVIDIA's NeMo Aligner and DeepSpeed-Chat offer enterprise-grade RLHF pipelines.

### 7.2 Recipe by Use Case

A practical guide for which method to use:

- **Instruction following, chat quality**: DPO (or SimPO/ORPO) on preference data. This is the standard recipe for fine-tuning open-source models on preference datasets like UltraFeedback or HelpSteer.
- **Math, coding, reasoning**: GRPO on verifiable rewards. This is the recipe used by DeepSeek-R1, Qwen-Coder, and most reasoning-specialized models.
- **Safety alignment**: DPO with carefully curated preference data from safety experts. Some teams use RLHF (PPO with reward model) for the highest-stakes safety training, on the grounds that the additional control is worth the complexity.
- **Tool use, agent training**: GRPO with task success as the reward, plus auxiliary supervised losses for tool call format compliance.

### 7.3 Common Failure Modes

DPO failure modes include reduction of probability mass for chosen responses (the model learns to suppress all responses, just suppressing rejected ones harder), distributional drift away from useful behaviors not covered in preference data, and overfitting to noise in the preference dataset. Mitigations include SFT regularization, careful preference data curation, and conservative β values.

GRPO failure modes include reward hacking (the model finds shortcuts to verifiable rewards that don't generalize), instability when the verifier is imperfect, and degeneration of off-task capabilities. Mitigations include strict KL regularization, verifier robustness checks, and mixed training with general instruction-following data.

## 8. The Bigger Picture

The shift from PPO to DPO to GRPO illustrates a broader pattern in deep learning: the methods that win in practice are usually the simplest ones that achieve the desired result, not the most sophisticated. PPO was sophisticated and worked, but DPO worked equally well with much less machinery. GRPO worked for reasoning tasks where DPO did not, with minimal additional machinery beyond DPO.

Each transition reduced the amount of moving parts: PPO needed a reward model, value function, and complex hyperparameter tuning. DPO eliminated the reward model and value function. GRPO kept on-policy sampling but eliminated the reward model (when verifiable rewards exist) and the value function (using group-mean baselines). The remaining ingredients are: a policy, a reference policy, a way to score responses, and a way to compare scores.

The next iteration is likely to push further. There is active research on offline GRPO variants (eliminating on-policy sampling), pure value-free methods, and methods that share more compute with the underlying inference stack. The trajectory suggests that LLM alignment will continue to become simpler, cheaper, and more accessible — which is good news for everyone outside the well-funded labs that pioneered the original RLHF approaches.

## 9. Conclusion

DPO and GRPO represent the post-PPO era of LLM alignment. They are simpler, cheaper, and (in their respective domains) at least as effective as the methods they replaced. The pace of innovation in this space has been remarkably fast: from "PPO is the only way" in 2022 to "DPO replaces PPO for most use cases" in 2023 to "GRPO is the right tool for reasoning" in 2024. Each transition was driven by a small algorithmic insight that unlocked dramatic practical improvements.

For practitioners, the practical guidance is clear. If you have preference data and want a cheaply-tuned model with better outputs, use DPO. If you have verifiable tasks and want to elicit reasoning, use GRPO. Reach for full PPO with an explicit reward model only if you have specific reasons that DPO and GRPO fall short — and be prepared for the engineering overhead that PPO entails.

The deeper lesson is that the optimization problem of LLM alignment is, at its core, simpler than the early RLHF infrastructure suggested. Once the right loss function is found, much of the complexity falls away. There is reason to believe that further simplifications are still ahead.
