# Scaling Laws for Large Language Models

*April 2026*

## 1. Introduction

Scaling laws are empirical relationships that describe how the performance of a language model changes as a function of its size, training data, and compute budget. These relationships, first rigorously characterized by Kaplan et al. at OpenAI in 2020 and subsequently refined by Hoffmann et al. at DeepMind in 2022, have become the most important tool in the LLM field for making resource allocation decisions. They answer the most consequential question in large-scale machine learning: given a fixed budget of compute, how should that budget be divided between model parameters and training data to achieve the best possible performance?

The practical impact of scaling laws cannot be overstated. They are the reason Chinchilla exists. They are the reason Meta trained Llama models the way it did. They are the reason that industry investment in AI infrastructure has exceeded hundreds of billions of dollars. Scaling laws provide the theoretical justification—or at least the empirical justification—for the belief that performance will continue to improve predictably as resources increase. They are also at the center of the most important open debate in AI: whether this improvement will continue indefinitely, slow down, or hit fundamental limits.

This report provides a comprehensive technical examination of scaling laws. It covers the original formulations, the Chinchilla revision, practical applications and deviations, emergent abilities, test-time compute scaling, and the physical and economic limits that constrain the scaling paradigm. The intended audience is engineers, researchers, and decision-makers who need to understand scaling laws in order to make training decisions, evaluate model releases, or assess the trajectory of AI capabilities.

## 2. The Kaplan Scaling Laws (2020)

### 2.1 The Original Formulation

In January 2020, Jared Kaplan and colleagues at OpenAI published "Scaling Laws for Neural Language Models," a paper that systematically measured the relationship between model performance (measured as cross-entropy loss on a held-out test set) and three key variables: the number of model parameters (N), the size of the training dataset (D, measured in tokens), and the amount of compute used for training (C, measured in FLOPs).

The central finding was that loss follows a power-law relationship with each variable when the others are not bottlenecks:

```
L(N) = (N_c / N)^α_N       where α_N ≈ 0.076
L(D) = (D_c / D)^α_D       where α_D ≈ 0.095
L(C) = (C_c / C)^α_C       where α_C ≈ 0.050
```

Here, N_c, D_c, and C_c are constants that set the scale, and the exponents α determine how quickly loss improves as each variable increases. The key observation is that these are power laws, not exponential relationships. Performance improves smoothly and predictably as a straight line on a log-log plot, with no sudden transitions or plateaus over the range of scales tested.

### 2.2 Key Findings

Several findings from the Kaplan paper shaped the field's approach to scaling:

**Smooth power-law behavior.** The loss curves were remarkably clean. Across multiple orders of magnitude, the relationship between log(loss) and log(scale) was approximately linear for all three variables. This meant that performance was highly predictable—given the scaling exponents, one could estimate the loss of a larger model without training it, simply by extrapolating the power law.

**Model size matters more than data size.** The Kaplan exponents suggested that increasing model parameters was more efficient than increasing data for reducing loss. Specifically, if you had 10x more compute, the optimal allocation was to increase the model size by roughly 5x and the data by only 2x. This led to a widely adopted rule of thumb: for compute-optimal training, the number of tokens should be roughly equal to the number of parameters (D ≈ N).

**Architecture details matter less than scale.** The paper found that the scaling exponents were relatively consistent across different model architectures (varying depth, width, attention heads) as long as the total parameter count was held constant. This suggested that scale was the primary driver of performance, with architecture contributing at most second-order effects.

**Training efficiency.** The paper argued that large models are more sample-efficient than small models—they extract more information per training token. This meant that for a fixed compute budget, it was better to train a very large model for fewer steps than a small model for more steps.

### 2.3 The Compute-Optimal Frontier

Kaplan et al. defined the compute-optimal frontier as the set of (N, D) pairs that minimize loss for a given compute budget C. Since C ≈ 6ND for a transformer (where the factor of 6 accounts for the forward and backward pass), any compute budget can be allocated along the N-D trade-off curve. The Kaplan scaling laws implied a specific allocation rule:

```
N_opt ∝ C^0.73
D_opt ∝ C^0.27
```

This says that as compute increases, the model should grow much faster than the dataset. A 10x increase in compute should correspond to roughly 5.4x more parameters but only 1.9x more data. This recommendation heavily influenced the training of GPT-3 (175B parameters, 300B tokens)—a model that was, by the later Chinchilla analysis, significantly undertrained relative to its size.

### 2.4 Limitations of the Original Analysis

The Kaplan analysis had several limitations that would be identified by subsequent work:

**Learning rate schedule dependence.** The experiments used a fixed learning rate schedule that may not have been optimal for all model sizes. If smaller models were trained with suboptimal learning rates, this would bias the scaling exponents to overstate the advantage of larger models.

**Limited scale range for data.** The experiments varied model size over a much wider range than data size, potentially leading to less accurate estimates of the data scaling exponent.

**Cosine schedule not tuned per run.** Later work showed that properly tuning the learning rate schedule for each model size could significantly change the optimal allocation between parameters and data.

## 3. The Chinchilla Revision (2022)

### 3.1 Hoffmann et al.'s Findings

In March 2022, Jordan Hoffmann and colleagues at DeepMind published "Training Compute-Optimal Large Language Models," universally known as the "Chinchilla paper." This paper fundamentally revised the Kaplan scaling laws by showing that the optimal allocation of compute between model parameters and training data was much more balanced than Kaplan had suggested.

The Chinchilla paper used three complementary approaches to estimate the optimal N-D allocation:

1. **Approach 1 (fixed model size):** For several fixed model sizes, vary the number of training tokens and fit the loss curve for each.
2. **Approach 2 (fixed FLOP budget):** For several fixed compute budgets, vary the model size and training tokens simultaneously, finding the loss-minimizing allocation for each budget.
3. **Approach 3 (parametric loss function):** Fit a parametric model L(N, D) = E + A/N^α + B/D^β to all training runs simultaneously, then derive the optimal allocation analytically.

All three approaches converged on the same conclusion: the optimal number of training tokens scales approximately linearly with model parameters. The headline result was:

```
D_opt ≈ 20 × N
```

A compute-optimal model should be trained on roughly 20 tokens per parameter. This was a radical departure from the Kaplan recommendation, which implied much less data relative to model size. It meant that GPT-3, with 175B parameters trained on 300B tokens (1.7 tokens per parameter), was undertrained by a factor of roughly 10x.

### 3.2 The Chinchilla Model

To demonstrate their findings, the DeepMind team trained Chinchilla: a 70B parameter model trained on 1.4 trillion tokens (20 tokens per parameter). Despite having only 25% the parameters of the then-state-of-the-art Gopher model (280B parameters), Chinchilla outperformed Gopher on the majority of benchmarks. This was a striking demonstration: a smaller, cheaper-to-serve model could beat a larger one simply by training it on more data.

The implications were immediate and profound:

**Inference cost reduction.** A model with 4x fewer parameters is roughly 4x cheaper to serve (in terms of memory, compute, and latency). If scaling data rather than parameters could achieve the same or better quality, inference costs would drop dramatically.

**Data becomes the bottleneck.** If compute-optimal training requires 20 tokens per parameter, then a 1T parameter model would need 20T tokens of training data. High-quality text data of this volume does not obviously exist, making data curation and generation a critical concern.

**Re-evaluation of existing models.** Many existing large models—including GPT-3, Gopher, PaLM, and others—were revealed to be significantly undertrained by the Chinchilla criterion. This suggested that simply training existing architectures on more data could yield large improvements.

### 3.3 The Parametric Loss Model

Approach 3 of the Chinchilla paper fit the following parametric model to the empirical data:

```
L(N, D) = E + A/N^α + B/D^β
```

Where:
- E ≈ 1.69 is the irreducible loss (the entropy of natural language itself, or at least the best the model family can achieve)
- A ≈ 406.4, α ≈ 0.34
- B ≈ 410.7, β ≈ 0.28

The first term (A/N^α) captures the "approximation error"—the gap between the model and the best possible model of unlimited size, due to the finite number of parameters. The second term (B/D^β) captures the "estimation error"—the gap due to seeing a finite number of training examples. The irreducible error E represents the entropy of the data distribution itself, which no model can reduce regardless of scale.

Given a compute budget C = 6ND, minimizing L with respect to N and D (subject to the compute constraint) yields the optimal allocation. The roughly equal exponents (α ≈ 0.34, β ≈ 0.28) are what produce the approximately linear D ∝ N relationship, in contrast to Kaplan's asymmetric exponents.

### 3.4 Subsequent Confirmations and Refinements

The Chinchilla scaling laws have been broadly confirmed by subsequent work, though with refinements:

**Llama team analysis.** Meta's Llama technical reports have noted that the 20:1 token-to-parameter ratio is a good starting point but that the optimal ratio depends on the quality of training data and the specifics of the training recipe.

**Yi, Qwen, and other Chinese lab findings.** Multiple Chinese AI labs training large models have reported scaling behavior broadly consistent with Chinchilla, though some have argued for slightly different exponents depending on the data distribution and tokenizer.

**Epoch AI analysis.** The research organization Epoch has conducted meta-analyses of published training runs and found that the D ≈ 20N rule approximately holds for most models trained after the Chinchilla paper, with some labs deliberately deviating from compute-optimality for inference efficiency reasons (discussed in Section 4).

## 4. Beyond Compute-Optimality: Training for Inference

### 4.1 The Inference-Aware Perspective

The Chinchilla scaling laws define "compute-optimal" with respect to training cost: given a fixed training budget, what model achieves the lowest loss? But training cost is a one-time expense, while inference cost is ongoing and often dominates the total cost of ownership. This creates a different optimization problem: given a total budget that includes both training and anticipated inference, what model minimizes total cost for a target performance level?

The answer, in practice, is to train a smaller model on more data than the Chinchilla-optimal amount. This increases training cost (because you train longer) but decreases inference cost (because the model is smaller and cheaper to serve). If the model will handle billions of inference requests over its lifetime, the inference savings far outweigh the additional training cost.

### 4.2 Llama as a Case Study

Meta's Llama models are the most prominent example of this inference-aware scaling philosophy. Llama 1 (February 2023) trained a 7B model on 1 trillion tokens—roughly 143 tokens per parameter, far exceeding the Chinchilla-optimal 20:1 ratio. The 13B model was trained on the same 1T tokens (77 tokens per parameter), and even the 65B model saw 1.4T tokens (22 tokens per parameter, close to Chinchilla-optimal).

The explicit justification was that the Llama team was not optimizing for the cheapest training run to achieve a given loss. They were optimizing for the best possible model at a given inference cost. A 7B model, no matter how much it is trained, will always be cheaper to serve than a 13B model. If training the 7B model on 5x more data than Chinchilla suggests gets it close to the performance of a Chinchilla-optimal 13B model, the total cost (amortized training plus inference) is lower.

Llama 2 (July 2023) continued this trend, training on 2 trillion tokens. Llama 3 (April 2024) pushed even further, training the 8B model on 15 trillion tokens—roughly 1,875 tokens per parameter, nearly 100x the Chinchilla-optimal ratio. The 70B model trained on the same 15T tokens (214 tokens per parameter). Llama 3.1 405B trained on 15.6T tokens (39 tokens per parameter).

### 4.3 Diminishing Returns from Overtraining

Training beyond the Chinchilla-optimal point follows a specific diminishing returns curve. The Chinchilla loss model predicts that once the data term (B/D^β) is much smaller than the model term (A/N^α), additional data provides diminishing improvements. The model's representational capacity becomes the binding constraint, and no amount of additional data can compensate for having too few parameters.

Empirically, this manifests as a "data scaling wall" for a given model size. The 7B Llama models appear to gain meaningful improvements up to roughly 10-15T tokens but show diminishing returns beyond that. The exact point at which overtraining becomes wasteful depends on the data quality, the training recipe, and the evaluation criteria, but the general pattern is clear: there is a ceiling for each model size that data alone cannot break through.

### 4.4 Formalization: Inference-Adjusted Scaling Laws

Several groups have attempted to formalize inference-adjusted scaling. The general framework is:

```
Total_cost = C_train + n_queries × C_inference_per_query
```

Where C_train = 6ND and C_inference_per_query depends on N (roughly proportional to N for autoregressive generation). Minimizing total cost for a target loss L requires balancing training and inference costs. The result is that the optimal model is smaller and more heavily trained than the Chinchilla-optimal model, with the degree of overtraining depending on the expected number of inference queries.

For models expected to serve billions of queries (like GPT-4, Claude, or Gemini), the inference-optimal model can be dramatically smaller and more heavily trained than the compute-optimal one. For a model used for a niche research task with few inference queries, Chinchilla-optimal training is appropriate.

## 5. Data-Constrained Scaling

### 5.1 The Data Wall

The Chinchilla scaling laws assume an unlimited supply of high-quality training data. In practice, this assumption is increasingly strained. By 2024, frontier models were training on datasets of 10-15 trillion tokens, consuming a substantial fraction of the publicly available text on the internet. Epoch AI estimated in late 2024 that the stock of high-quality text data (books, articles, curated web text) totals roughly 10-20 trillion tokens, and that lower-quality web scrape data totals perhaps 50-100 trillion tokens (with significant noise).

This creates a data wall: the point at which further scaling of training data is limited by the available supply rather than by compute. If a 10T parameter model requires 200T tokens of compute-optimal training data, and only 50-100T tokens of usable text exist, the model is necessarily trained below compute-optimality.

### 5.2 Repeated Epochs

One response to data scarcity is to train on the same data for multiple epochs. Traditional deep learning in vision and other domains routinely uses hundreds of epochs, but the LLM scaling literature initially assumed single-epoch training. The question is whether repeated epochs provide diminishing returns that are qualitatively different from the diminishing returns of additional unique data.

Muennighoff et al. (2023, "Scaling Data-Constrained Language Models") studied this systematically and found:

**Multi-epoch training works, with diminishing returns.** Training for up to 4 epochs on the same data provides approximately the benefit of training on 60% as much unique data. Beyond 4 epochs, returns diminish more sharply, and the model begins to show signs of memorization and overfitting.

**The value of data repetition depends on data quality.** High-quality data can be repeated more times with less degradation than low-quality data, likely because high-quality data has more extractable structure per token.

**A modified scaling law for repeated data.** The authors proposed extending the Chinchilla parametric model to account for repeated data:

```
L(N, D, R) = E + A/N^α + B/D_eff(D, R)^β
```

Where D_eff is an "effective data" function that accounts for the diminishing value of repeated tokens. This provides a framework for deciding when to invest in acquiring new data versus training longer on existing data.

### 5.3 Synthetic Data Augmentation

The most actively pursued response to the data wall is synthetic data generation: using existing language models to generate additional training data. This takes several forms:

**Rephrasing and augmentation.** Existing text is paraphrased, reformatted, or extended by a language model, creating new surface forms that express the same underlying content. This increases the effective diversity of the dataset without requiring new source material.

**Instruction generation.** Models generate question-answer pairs, instructions, and other structured data from unstructured text. This is the basis of the "self-instruct" paradigm and its successors (Alpaca, Vicuna, etc.), which generate instruction-tuning data from a strong model and use it to train a weaker model.

**Reasoning traces.** Models generate step-by-step reasoning for mathematical and logical problems, creating training data that teaches chain-of-thought reasoning. This is particularly valuable because human-written reasoning traces are relatively scarce.

**Code generation.** Models generate code from specifications, documentation, or natural language descriptions, augmenting the supply of code training data.

**Domain-specific generation.** For specialized domains (medical, legal, scientific), models generate domain-specific text that augments the limited supply of real domain data.

The critical concern with synthetic data is "model collapse"—the risk that training on model-generated data introduces biases, reduces diversity, and degrades the distribution of the training set. Shumailov et al. (2023) showed that iterative training on model-generated data can lead to progressive loss of information in the tails of the distribution, eventually causing the model to converge on a narrow, low-diversity distribution. This imposes a limit on the fraction of training data that can be synthetic, though the exact limit depends on the quality of the generator and the curation applied to the synthetic data.

### 5.4 Data Quality as a Scaling Axis

An increasingly important insight is that data quality functions as an independent scaling axis. Training on curated, high-quality data produces better models per token than training on uncurated web scrape. This is reflected in the scaling laws through the constants A and B—better data effectively shifts the entire loss curve downward, equivalent to a multiplicative increase in effective data or effective parameters.

The Phi series of models from Microsoft Research is the most prominent demonstration of this principle. Phi-1 (1.3B parameters), Phi-2 (2.7B), and Phi-3 (3.8B) achieved performance competitive with much larger models by training on carefully curated "textbook-quality" data. The key insight was that filtering for high-quality data could substitute for raw scale, at least up to a point.

This has led to a modified perspective on scaling laws: the relevant variable is not just the number of tokens D but the "effective data" D_eff, which depends on both the volume and the quality of the training data. High-quality data has a higher effective token count per raw token, shifting the scaling curve.

## 6. Emergent Abilities and Phase Transitions

### 6.1 The Original Claims

In 2022, Wei et al. published "Emergent Abilities of Large Language Models," arguing that certain capabilities—including chain-of-thought reasoning, multi-step arithmetic, and word unscrambling—appear suddenly and unpredictably as model scale increases. Below a critical size threshold, models show near-zero accuracy on these tasks; above the threshold, accuracy jumps to significantly above chance. The paper argued that these "emergent abilities" could not be predicted by extrapolating from smaller models and represented a qualitatively different phenomenon from the smooth scaling observed for loss.

This claim was enormously influential. It suggested that scaling could produce genuinely novel capabilities, not just quantitative improvements—that there were "phase transitions" in model ability that made the trajectory of AI capabilities fundamentally unpredictable. This narrative shaped public discourse, policy debates, and investment decisions.

### 6.2 The Critique: Measurement Artifacts

In 2023, Schaeffer et al. published "Are Emergent Abilities of Large Language Models a Mirage?", arguing that the appearance of emergent abilities is largely an artifact of the evaluation metrics used. The key argument is:

**Nonlinear metrics create the appearance of sudden transitions.** Many of the tasks showing "emergent" abilities were evaluated using discontinuous metrics like exact-match accuracy or multiple-choice accuracy. These metrics have a threshold effect: if a model assigns 45% probability to the correct answer, it scores 0% on exact-match; if it assigns 55%, it scores 100%. The model's underlying capability (its log-probability on the correct answer) may be improving smoothly with scale, but the metric only reflects this improvement once the probability crosses the threshold.

**Continuous metrics show smooth improvement.** When the same tasks are evaluated using continuous metrics like Brier score, log-probability of the correct answer, or token-level perplexity, the smooth improvement with scale becomes visible. There is no sudden transition in the model's underlying capability—only a transition in the metric's ability to detect it.

**Statistical resolution.** Small models evaluated on difficult tasks produce near-zero accuracy with high variance. With enough evaluation examples, a slight improvement at intermediate scales might be detectable, but typical benchmark sizes are too small to resolve the small improvements that precede the apparent "emergence."

### 6.3 The Current Consensus

The current consensus, as of early 2026, is nuanced:

**Loss scales smoothly.** The aggregate cross-entropy loss on held-out data scales smoothly and predictably with model/data/compute, as described by the Kaplan and Chinchilla scaling laws. This has been confirmed across many orders of magnitude and is not seriously disputed.

**Downstream task accuracy may show threshold effects.** Even if the underlying capabilities improve smoothly, the practical utility of a model on specific tasks can show threshold behavior. A model that writes code with 40% correctness is essentially useless for production code generation; a model with 70% correctness can be productive with human review. This is not "emergence" in the model's capabilities but a threshold in practical utility.

**True qualitative emergence remains debated.** Whether there are capabilities that are genuinely absent below a certain scale and genuinely present above it—as opposed to gradually improving but crossing a usability threshold—remains an open empirical question. Claims of true emergence are difficult to verify because they require ruling out the metric and statistical artifacts described by Schaeffer et al.

**Predictability is the practical question.** For practitioners and decision-makers, the relevant question is not whether "emergence" is real in some philosophical sense but whether the performance of larger models on specific tasks can be predicted from smaller models. The evidence suggests that aggregate loss is predictable, individual task accuracy is somewhat predictable (especially with continuous metrics), and the appearance of entirely new capabilities is the least predictable.

## 7. Scaling Laws for Fine-Tuning

### 7.1 Distinct Scaling Regime

Pre-training scaling laws describe the relationship between scale and loss on a held-out test set drawn from the pre-training data distribution. Fine-tuning scaling laws describe a different relationship: how does performance on a specific downstream task change as a function of the pre-trained model's size, the fine-tuning dataset size, and the fine-tuning compute?

The key insight is that fine-tuning operates in a fundamentally different regime from pre-training. Pre-training learns general representations from a broad data distribution. Fine-tuning adapts those representations to a specific task, typically with much less data and much less compute. The scaling dynamics are different because the "base" from which improvement occurs is not random initialization but a pre-trained model that already has substantial knowledge and capability.

### 7.2 Empirical Findings

Several studies have characterized fine-tuning scaling:

**Performance scales with pre-trained model size.** Larger pre-trained models generally fine-tune to better performance on downstream tasks, given the same fine-tuning data. This is unsurprising—the larger model has more general knowledge to adapt—but the scaling exponents differ from pre-training scaling.

**Fine-tuning data requirements scale sub-linearly with model size.** Larger models require relatively less fine-tuning data to achieve a given level of task performance, because they have more relevant pre-trained knowledge. A 70B model may need only 2-3x the fine-tuning data of a 7B model to achieve proportionally better results, rather than the 10x that the size ratio might suggest.

**LoRA and parameter-efficient methods follow their own scaling laws.** Low-rank adaptation (LoRA) and similar methods add a relatively small number of parameters during fine-tuning. The relationship between the rank of the adaptation (the number of added parameters) and the resulting performance follows a power law with different exponents than full fine-tuning. Higher ranks improve performance up to a point, after which they provide diminishing returns and eventually risk overfitting.

**Instruction tuning scales differently from task-specific fine-tuning.** Instruction tuning—training on diverse (instruction, response) pairs to improve general instruction following—shows a scaling relationship more similar to pre-training than to task-specific fine-tuning. Performance improves with both the diversity and volume of instruction data, with diversity generally being more important than volume.

### 7.3 Implications for Practitioners

The fine-tuning scaling laws have several practical implications:

**Start with the largest affordable pre-trained model.** Because larger models fine-tune better with less data, the most cost-effective approach is usually to fine-tune the largest pre-trained model that fits within the inference budget, rather than fine-tuning a smaller model on more data.

**Diminishing returns from fine-tuning data are steep.** Fine-tuning data is typically much more expensive to acquire than pre-training data (because it requires task-specific annotation). The steep diminishing returns mean that small, high-quality fine-tuning datasets are often sufficient, and investing in data quality is more important than investing in data quantity.

**Fine-tuning compute is negligible relative to pre-training.** Even for the largest models, fine-tuning typically requires 0.01-0.1% of the pre-training compute. This means that fine-tuning scaling laws are primarily about data, not compute.

## 8. Test-Time Compute Scaling

### 8.1 The Shift to Inference-Time Scaling

The traditional scaling paradigm focuses exclusively on training: spend more compute on training to get a better model, which then runs with fixed compute at inference time. Beginning in 2024, a complementary paradigm emerged: scaling the compute spent at inference time to improve performance on individual queries. This approach, exemplified by OpenAI's o1 and o3 models and subsequently by DeepSeek-R1, Claude's extended thinking, and Gemini's thinking mode, represents a fundamental shift in how compute is allocated for performance.

The core idea is that some problems benefit from "thinking longer." Just as a human expert might solve a straightforward question in seconds but spend hours on a difficult proof, a language model can allocate variable compute to different queries. Simple queries are answered with a single forward pass (the standard regime); difficult queries trigger extended chain-of-thought reasoning, search over multiple solution paths, self-verification, and iterative refinement.

### 8.2 Mechanisms for Test-Time Compute

Several mechanisms enable test-time compute scaling:

**Chain-of-thought reasoning.** The model generates intermediate reasoning steps before producing a final answer. Longer chains consume more inference compute (more tokens generated, each requiring a forward pass) but can improve accuracy on reasoning-intensive tasks. The scaling law here is roughly logarithmic: doubling the length of the reasoning chain provides diminishing but positive returns.

**Best-of-N sampling.** The model generates N independent responses to the same query, and a selection mechanism (a verifier, majority vote, or the model itself) chooses the best one. Performance improves as a power law with N, but the compute cost also scales linearly with N. This is particularly effective for tasks where correctness can be verified (e.g., math problems, code generation).

**Tree search.** Rather than generating independent completions, the model explores a tree of partial solutions, backtracking when a branch appears unpromising. This is more compute-efficient than best-of-N for structured problems but requires a mechanism for evaluating partial solutions.

**Self-verification and refinement.** The model generates an initial response, critiques it, and produces an improved version. This can be repeated for multiple rounds, with each round consuming additional compute.

### 8.3 The Test-Time Compute Scaling Law

Snell et al. (2024, "Scaling LLM Test-Time Compute Optimally Can be More Effective than Scaling Model Parameters") provided the first systematic analysis of test-time compute scaling. Their key findings:

**Test-time compute follows a power law.** Performance on reasoning tasks (measured by accuracy on math and logic benchmarks) improves as a power law with the amount of inference compute, analogous to the power-law improvement with training compute. The exponent is different and task-dependent, but the functional form is similar.

**Test-time scaling can substitute for model scaling.** A smaller model with sufficient test-time compute can match or exceed a larger model running with standard inference. This creates a trade-off: spend the compute budget on a larger model with cheap inference, or a smaller model with expensive inference? The answer depends on the distribution of query difficulty.

**Optimal compute allocation is problem-dependent.** Easy problems benefit little from additional test-time compute; hard problems benefit enormously. The optimal strategy allocates compute adaptively, spending little on easy queries and much more on hard ones. This is the basis for "reasoning" model APIs that charge based on the number of tokens generated (including thinking tokens), not just the output tokens.

### 8.4 Implications

Test-time compute scaling fundamentally changes the economics of LLM deployment. It introduces a new trade-off axis: instead of choosing between a large model and a small model at the architecture level, operators can choose a smaller model and allocate variable compute at inference time based on query difficulty. This is closely related to model routing (discussed in the companion article on model routing and cascading) and has significant implications for capacity planning (discussed in the companion article on capacity planning).

The most profound implication is that "the model" is no longer a fixed entity with static capabilities. A model with test-time compute scaling is better described as a compute-performance curve: for any given amount of inference compute, it achieves a specific level of performance, and more compute always buys more performance (up to a limit). This changes how models are evaluated, deployed, and priced.

## 9. Mixture-of-Experts Scaling

### 9.1 MoE Architecture and Scaling Properties

Mixture-of-experts (MoE) models decouple the total number of parameters from the compute required per forward pass. In a standard dense transformer, every parameter participates in every forward pass. In an MoE transformer, each token is routed to a subset of "expert" modules (typically 1-2 out of 8-64 experts per layer), so the active compute per token is much less than the total parameter count would suggest.

This creates a different scaling dynamic. An MoE model with 1.8T total parameters but 111B active parameters per token (as in Mixtral 8x22B) performs more like a dense model somewhere between 111B and 1.8T parameters, depending on the task. The total parameters provide a larger "memory" for storing knowledge, while the active parameters determine the compute cost per token.

### 9.2 MoE Scaling Laws

The scaling laws for MoE models are less well-characterized than for dense models, but several findings have emerged:

**Loss scales with active parameters and total parameters.** Clark et al. (2022) and subsequent work have shown that MoE loss can be modeled as a function of both active parameters (N_active) and total parameters (N_total):

```
L(N_active, N_total, D) ≈ E + A/N_active^α_1 × f(N_total/N_active) + B/D^β
```

Where f is a function that captures the benefit of having more total parameters relative to active parameters. The benefit is real but sublinear—doubling the number of experts (and total parameters) while keeping active parameters constant provides a meaningful but diminishing improvement.

**MoE models are more compute-efficient for training.** Because an MoE model with a given active parameter count has more total parameters than a dense model with the same active count, it can achieve lower loss for the same training compute. This makes MoE attractive when training compute is the binding constraint.

**MoE models have complex inference characteristics.** The memory required to serve an MoE model is proportional to total parameters, not active parameters. This means a 1.8T parameter MoE model requires the same memory as a 1.8T dense model, even though it uses much less compute per token. For inference-bound deployments, the memory overhead of MoE can negate its compute advantages.

### 9.3 Expert Utilization and Routing

The effectiveness of MoE scaling depends critically on expert utilization—how evenly tokens are distributed across experts. If some experts are rarely used, their parameters are wasted. Routing mechanisms (typically a learned linear layer that computes expert assignment probabilities) must be designed to balance load across experts while still allowing specialization.

Auxiliary load-balancing losses encourage even utilization but can reduce specialization. Expert choice routing (where experts select tokens rather than tokens selecting experts) provides better load balance. DeepSeek's MoE models have introduced shared experts that process all tokens alongside routed experts, providing a baseline of dense computation augmented by sparse expert computation.

## 10. Prediction Accuracy and Extrapolation

### 10.1 How Well Do Scaling Laws Predict?

The practical value of scaling laws depends on their ability to predict the performance of models that have not yet been trained. Several groups have evaluated this:

**Loss prediction is accurate.** For models within the same architecture family, training recipe, and data distribution, scaling laws predict held-out loss with remarkable accuracy—often within 1-2% over 1-2 orders of magnitude of extrapolation. This means that a lab can train several small models, fit a scaling law, and predict the loss of a model 10-100x larger with reasonable confidence.

**Benchmark accuracy prediction is less reliable.** Predicting specific benchmark scores (e.g., MMLU accuracy, HumanEval pass rate) from scaling laws is less reliable because the relationship between loss and benchmark accuracy is nonlinear and benchmark-specific. Some groups have developed "downstream scaling laws" that predict benchmark scores as a function of model scale, but these require benchmark-specific calibration and are less transferable.

**Extrapolation beyond the fitted range is risky.** Scaling laws are empirical fits to observed data. Extrapolating far beyond the observed range is unreliable, as the true functional form may deviate from the assumed power law at extreme scales. The history of the field includes cases where models at unprecedented scales performed differently than scaling laws predicted—sometimes better (suggesting the scaling law was conservative), sometimes worse (suggesting the scaling law missed a constraint).

### 10.2 What Scaling Laws Do Not Predict

Several important properties of language models are not well-predicted by scaling laws:

**Specific capability thresholds.** When a model will become "good enough" at a specific task (e.g., reliable multi-step reasoning, accurate code generation for a specific language) cannot be predicted from loss scaling alone. The mapping from loss to specific capabilities is complex and task-dependent.

**Safety properties.** A model's tendency to produce harmful, biased, or incorrect outputs is not a simple function of scale. Larger models can be both more capable and more problematic—they are better at following instructions, including harmful instructions, and better at generating plausible-sounding misinformation.

**Post-training improvements.** Scaling laws characterize the base model after pre-training. The impact of RLHF, DPO, instruction tuning, and other post-training techniques is not captured by pre-training scaling laws and can significantly change the model's effective capability on practical tasks.

**Qualitative capability shifts.** Whether a much larger model will exhibit qualitatively new capabilities (e.g., the ability to reliably plan multi-step tasks, or to perform novel scientific reasoning) cannot be predicted from current scaling laws. This is the practical implication of the emergent abilities debate: even if loss scales smoothly, the mapping from loss to practically useful capabilities may be discontinuous.

## 11. Limits of Scaling

### 11.1 The Data Wall

As discussed in Section 5, the finite supply of high-quality text data imposes a practical limit on scaling. Current frontier models are already training on a substantial fraction of available high-quality text. Strategies to extend this limit—multi-epoch training, synthetic data, multimodal data—provide additional headroom but do not eliminate the constraint. Some researchers believe that synthetic data generation can provide an essentially unlimited supply of useful training data, while others argue that synthetic data will eventually suffer from distributional collapse and that the fundamental limit is the stock of genuinely novel information produced by humans.

### 11.2 The Energy Wall

Training and serving large language models requires enormous amounts of electricity. The energy required for training scales linearly with compute, and the scaling laws demand exponentially growing compute for linear improvements in loss. This creates an energy scaling challenge:

A 10x improvement in loss requires roughly 10^(1/α_C) ≈ 10^20 increase in compute (using the Kaplan exponent α_C ≈ 0.050). Even with hardware efficiency improvements, the energy requirements for continued scaling are staggering. Training runs for frontier models in 2025-2026 are estimated to consume tens to hundreds of gigawatt-hours. At current scaling rates, within a few generations, training runs would require power plant-scale energy production dedicated to a single model.

Industry responses include building dedicated power plants for AI data centers, securing long-term power purchase agreements, and investing in nuclear and renewable energy sources. Microsoft's deal with Constellation Energy for Three Mile Island nuclear power and Amazon's nuclear power investments are concrete manifestations of the energy wall's impact on industry strategy.

### 11.3 The Hardware Wall

Moore's Law has slowed, and the rate of improvement in transistor density and chip performance is no longer sufficient to maintain historical scaling rates without increases in total hardware. The semiconductor industry faces physical limits on transistor miniaturization, and the cost of new fabrication nodes is increasing. This means that scaling compute requires scaling the number of chips, not just the efficiency of each chip.

The practical consequence is that training the next generation of frontier models requires larger and more expensive GPU clusters, with the capital expenditure measured in billions of dollars. NVIDIA's dominant position in the GPU market, combined with supply constraints on advanced packaging (CoWoS) and HBM memory, creates additional bottlenecks.

### 11.4 The Algorithmic Wall

Scaling laws assume a fixed model architecture and training recipe. If a fundamentally better architecture or training method were discovered, it would shift the scaling curve—equivalent to increasing the effective compute per FLOP. Some researchers argue that the current transformer architecture is far from optimal and that architectural innovations could provide order-of-magnitude improvements. Others argue that the transformer is already well-optimized for autoregressive language modeling and that remaining gains from architecture search are modest.

Historical evidence is mixed. Attention was a major architectural innovation that shifted the scaling curve. Flash attention and related algorithmic improvements have provided significant constant-factor improvements. But no post-attention architectural change has shifted the scaling exponents themselves—the power-law relationship between loss and compute remains intact, with the same approximate slopes, across architectures.

### 11.5 The Irreducible Loss

The Chinchilla parametric model includes an irreducible loss term E, representing the entropy of the data distribution—the best loss achievable by any model on the given data. As models approach this limit, further scaling provides diminishing returns that are even steeper than the power law suggests. The estimated irreducible loss for internet text is around 1.5-1.7 nats per token, and current frontier models are approaching losses of 2.0-2.2 nats, suggesting that the remaining gap to the irreducible limit is relatively small in absolute terms (though still represents significant practical capability differences).

## 12. Industry Investment and Strategic Implications

### 12.1 The Scaling Hypothesis as Investment Thesis

The scaling laws underpin the largest capital allocation decisions in the history of the technology industry. The "scaling hypothesis"—the belief that continued scaling will continue to produce economically valuable capability improvements—is the primary justification for the tens of billions of dollars being invested in AI infrastructure by Microsoft, Google, Amazon, Meta, and others.

The investment logic is straightforward: if scaling laws hold, then the company that trains the largest model on the most data with the most compute will have the most capable model, which will capture the most economic value. This creates a race dynamic where companies invest preemptively, building infrastructure for the next generation of models before the current generation has been fully deployed.

### 12.2 The Risk of Scaling Plateaus

The investment thesis is vulnerable to several risks:

**Scaling exponents could worsen.** The power-law exponents observed at current scales might not hold at much larger scales. If the exponents decrease (loss improves more slowly with scale), the return on additional investment diminishes, potentially making further scaling uneconomical.

**Practical utility may not track loss.** Even if loss continues to improve with scale, the practical utility of those improvements may diminish. The difference between a loss of 2.0 and 1.9 may represent enormous capability improvements, or it may represent only marginal gains on tasks that matter economically.

**Competition could erode moats.** If scaling laws are universal, then any organization with sufficient resources can achieve the same performance. This means that scale alone does not provide a durable competitive advantage—it only provides a temporary lead that competitors can close with sufficient investment.

**Smaller, specialized models may win in practice.** For many economic applications, a smaller, fine-tuned model that costs 10x less to serve may be preferable to a frontier model that is marginally better but dramatically more expensive. If the market for AI services is more sensitive to cost than to the last increment of quality, the value of frontier scaling may be limited.

### 12.3 The Role of Scaling Laws in Labs' Strategy

Frontier AI labs use scaling laws operationally in several ways:

**Predicting next-generation performance.** Before committing to a multi-hundred-million-dollar training run, labs train small-scale versions and use scaling laws to predict the expected performance of the full-scale model. This reduces the risk of an expensive training run that fails to meet expectations.

**Optimal resource allocation.** Scaling laws inform decisions about model size, data requirements, cluster sizing, and training duration. The Chinchilla result directly changed how labs allocate compute between model parameters and training data.

**Competitive intelligence.** Published scaling laws and model details allow labs to estimate competitors' capabilities. If a competitor announces a training run with a given compute budget, scaling laws provide an estimate of the resulting model's quality.

**Communicating with investors.** Scaling laws provide a quantitative framework for explaining to investors why additional investment in compute will yield additional capability, making AI scaling legible in terms familiar to technology investors.

## 13. Scaling for Specific Modalities and Tasks

### 13.1 Vision-Language Scaling

Scaling laws for vision-language models (VLMs) are less well-characterized than for text-only models, but initial findings suggest similar power-law relationships. The key additional variable is the balance between vision and language capacity in the model, and the ratio of image/video data to text data in the training set.

Models like GPT-4V, Gemini, and Claude's multimodal capabilities suggest that vision-language scaling benefits from both modalities—visual data provides information that is complementary to text, rather than redundant, leading to super-additive scaling in some regimes.

### 13.2 Code Generation Scaling

Code generation has attracted particular scaling attention because code quality is relatively easy to evaluate (via execution against test cases), enabling precise measurement of scaling curves. Findings from Codex, StarCoder, and Code Llama indicate that code generation performance scales smoothly with model size and training data, with code-specific data being much more valuable per token than general text data for code tasks.

### 13.3 Mathematical Reasoning Scaling

Mathematical reasoning appears to scale more steeply with model size than general language tasks, with larger models showing disproportionate improvements on multi-step reasoning tasks. This is one area where claims of "emergent" abilities have been most persistent—small models appear to lack the capacity for multi-step mathematical reasoning regardless of training data, while models above a certain size acquire it seemingly quickly. Whether this reflects true emergence or threshold effects in evaluation metrics remains debated.

## 14. Scaling Laws in Historical Context

### 14.1 Scaling Laws in Other Fields

Power-law scaling relationships are not unique to language models. Physics has a long tradition of scaling laws—allometric scaling in biology (metabolic rate scales as body mass to the 3/4 power), Kleiber's law, Zipf's law in linguistics, and scaling laws in turbulence and critical phenomena. The observation that loss scales as a power law with model size has prompted speculation about whether there is a deeper theoretical explanation, analogous to the renormalization group arguments that explain scaling in physics.

### 14.2 Theoretical Explanations

Several theoretical frameworks have been proposed to explain scaling laws:

**Statistical mechanics analogies.** Bahri et al. (2024) and others have drawn analogies between neural network scaling and statistical mechanics, arguing that the power-law scaling of loss arises from the same mathematical structures that produce power laws in physical systems near phase transitions.

**Random feature models.** Simplified models of neural networks (kernel methods, random feature models) exhibit power-law scaling with well-understood exponents. Whether these simplified models capture the essential features of transformer scaling is debated.

**Data manifold theory.** The argument that scaling laws reflect the intrinsic dimensionality and structure of the data manifold—that the power-law exponents depend on the geometric properties of the data distribution, not on the specific model architecture. This would explain why different architectures give similar exponents.

No theoretical framework has yet provided a fully satisfactory explanation of the observed scaling laws, and the empirical exponents remain essentially unexplained from first principles.

## 15. Practical Recommendations

### 15.1 For Training

**Use scaling laws to plan.** Before committing to a large training run, train small-scale models (0.1-1% of the target size) and fit a scaling law. This provides a reliable estimate of the expected loss and helps identify problems (data quality issues, training instabilities) early.

**Deviate from Chinchilla for inference-heavy workloads.** If the model will serve many inference requests, train a smaller model on more data than the Chinchilla-optimal amount. The inference savings will more than compensate for the additional training cost.

**Invest in data quality.** Improving data quality shifts the entire scaling curve. A 2x improvement in effective data quality can be equivalent to a 4-8x increase in data volume, which is often much cheaper to achieve.

**Plan for multi-epoch training.** If high-quality data is limited, plan for up to 4 epochs on the best data, supplemented by lower-quality data for additional volume. Monitor for memorization and overfitting.

### 15.2 For Deployment

**Consider test-time compute.** For reasoning-intensive applications, deploying a smaller model with test-time compute scaling (thinking tokens, best-of-N sampling) may be more cost-effective than deploying a larger model with standard inference.

**Match model size to task difficulty.** Scaling laws apply on average, but specific tasks may be "easy" or "hard" relative to the model's capabilities. Use model routing to direct easy queries to small models and hard queries to large models, optimizing the cost-quality trade-off.

**Monitor for scaling ceiling effects.** As models approach the irreducible loss or the limits of their architecture's representational capacity, further scaling provides diminishing returns. Track the model's performance on your specific tasks to identify when you are in the diminishing-returns regime.

## 16. Conclusion

Scaling laws are the most powerful empirical tool available for understanding and predicting the behavior of large language models. The core finding—that loss improves as a smooth power law with model size, data, and compute—has been confirmed across hundreds of experiments, multiple organizations, and several orders of magnitude. The Chinchilla revision showed that the optimal allocation between model parameters and training data is approximately balanced (D ≈ 20N), overturning earlier beliefs about the primacy of model size. The inference-aware perspective introduced by Llama and others showed that practical deployments should deviate from compute-optimality in favor of smaller, more heavily trained models.

But scaling laws are not a guarantee of continued progress. They are empirical observations, not physical laws. They describe what has happened, not what must happen. The data wall, energy wall, hardware wall, and algorithmic wall all represent potential limits to the scaling paradigm. The irreducible loss sets a fundamental floor. And the relationship between loss and practically useful capabilities—the question that actually matters for economic and social impact—is not well-characterized by scaling laws at all.

The most important lesson from the scaling laws literature is epistemic humility about what scaling can and cannot tell us. Scaling laws are excellent at predicting aggregate loss on held-out data within the range of observed scales. They are unreliable at predicting specific capabilities, safety properties, or the qualitative character of much larger models. The field's future trajectory—whether scaling continues to produce useful capability improvements, or whether it encounters diminishing returns that redirect research toward other approaches—is a question that scaling laws themselves cannot answer. It will be answered by the models we build, the data we train them on, and the problems we ask them to solve.
