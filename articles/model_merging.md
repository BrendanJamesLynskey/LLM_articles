# Model Merging: TIES, DARE, and the Art of Combining Trained LLMs

*April 2026 • Technical Report*

## 1. Introduction

In 2023, a surprising observation began circulating in the open-source ML community: if you took two fine-tuned versions of the same base model and averaged their weights together, the resulting model was often as good as either parent, sometimes better. The technique required no retraining, no GPU compute beyond the cost of loading the weights, and no labeled data. The combined model could inherit capabilities from both parents — a coder fine-tune averaged with a math fine-tune produced a model that was reasonable at both.

This was the seed of model merging, a family of techniques that combine multiple trained models into a single model through arithmetic operations on their weights. The early experiments used simple averaging; subsequent work developed more sophisticated methods (Task Arithmetic, TIES, DARE, Model Soups, SLERP) that handle the issues that arise when naively averaging weights causes destructive interference.

By 2024, model merging had become a staple of the open-source LLM ecosystem. The Open LLM Leaderboard was dominated for months by merged models — combinations of Mistral, Llama, Yi, and Qwen fine-tunes that no single team had trained — and the practice had spawned its own tooling (mergekit), conventions, and culture. Model merging is now a standard step in the production pipeline for many open-source model releases, alongside fine-tuning, DPO, and quantization.

This report examines the techniques, theory, and practice of model merging, with particular attention to the methods that have proven most effective.

## 2. Why Merging Works At All

The first question to address is why averaging the weights of two neural networks should produce anything coherent. Naive intuition suggests that neural network weights are a chaotic, high-dimensional landscape where averaging two arbitrary points should land you somewhere meaningless. For randomly initialized networks, this intuition is correct — averaging two networks trained independently from scratch produces noise.

The key insight is that fine-tuned models share a common starting point. When you fine-tune the same base model on two different tasks, both fine-tunes start from the same initial weights and move only modestly during training. The resulting models are close to each other in weight space, in the sense that the "fine-tuning trajectory" for each is a small displacement from the shared base. Averaging two nearby points in weight space typically lands somewhere also nearby, and within the basin of attraction that contains the base model and its useful descendants.

This phenomenon — that fine-tuned models from a common ancestor live in connected, low-loss regions of weight space — is called linear mode connectivity, and it has been studied extensively in neural network optimization. For randomly initialized networks, mode connectivity does not hold. For fine-tuned-from-base networks, it does, and it is what makes model merging possible.

### 2.1 Task Vectors

A useful conceptual framework is the task vector. Define a task vector as the difference between a fine-tuned model and its base:

```
τ_task = θ_finetuned - θ_base
```

The task vector is the displacement that fine-tuning introduced. It captures everything the model learned during fine-tuning. Adding a task vector to the base model recovers the fine-tuned model. The space of task vectors has interesting properties:

- **Negation**: Subtracting a task vector from the base model can "unlearn" the task, sometimes producing a model that is worse at that task than the base.
- **Addition**: Adding multiple task vectors combines their capabilities, with interference depending on how the tasks overlap.
- **Scaling**: Multiplying a task vector by a coefficient less than 1 dampens the fine-tuning effect; greater than 1 amplifies it.

Most modern merging methods can be understood as operations on task vectors, with different strategies for handling the interference that occurs when multiple task vectors are combined.

## 3. Naive Averaging and Model Soups

The simplest merging method is uniform weight averaging:

```
θ_merged = (θ_1 + θ_2 + ... + θ_n) / n
```

For models that share a base, this often works surprisingly well. The "Model Soups" paper (2022) demonstrated that averaging multiple fine-tunes of the same base model on related tasks consistently produced models with better in-distribution accuracy and dramatically better out-of-distribution generalization than any single fine-tune. The technique is sometimes called "uniform soup" (averaging all the candidates) or "greedy soup" (incrementally adding fine-tunes whose inclusion improves a held-out metric).

Uniform averaging has limits, though. When the input models have learned conflicting things — one model wants weight w to be 0.5, another wants it to be -0.3 — averaging produces 0.1, which captures neither learned behavior. For closely related fine-tunes (different random seeds, different orderings of the same data), this conflict is rare. For divergent fine-tunes (math task vs. coding task), it is pervasive.

This is the central problem that more sophisticated merging methods address: how to handle weight conflicts intelligently rather than averaging through them.

## 4. Task Arithmetic

Task Arithmetic (introduced in "Editing Models with Task Arithmetic," 2022) formalizes the task-vector framework. To combine multiple fine-tuned models, compute their task vectors, sum them with optional coefficients, and add the result to the base:

```
θ_merged = θ_base + α_1 · τ_1 + α_2 · τ_2 + ... + α_n · τ_n
```

The coefficients α control the contribution of each task. Setting all α to 1/n recovers uniform averaging. Setting some α to negative values produces "task negation" — actively removing a learned behavior from the merged model.

Task Arithmetic works moderately well for related tasks but suffers from the same conflict problem as naive averaging: when two task vectors point in opposing directions on a given parameter, summing them cancels out the learned behavior. The next generation of methods focused on resolving these conflicts.

## 5. TIES: Trim, Elect Sign, Disjoint Merge

TIES-Merging (2023) was the first widely-adopted method that explicitly handles weight conflicts. It operates in three steps applied to each task vector:

### 5.1 Trim

Most parameters in a fine-tuned model do not change much from their base values. The fine-tuning concentrates on a relatively small fraction of the weights. TIES trims each task vector by keeping only the top-k% of parameters (by magnitude) and zeroing out the rest. This isolates the parameters that the fine-tuning actually relied on, discarding the noisy small-magnitude changes.

The trim percentage is typically 20-40% — keeping the top 20-40% of parameters by magnitude and zeroing the rest. This may sound aggressive, but the discarded parameters carry very little task-specific information.

### 5.2 Elect Sign

For each parameter position, multiple task vectors may want to push the weight in different directions. TIES resolves these conflicts by electing a single sign per parameter — typically the sign of the largest-magnitude vote across all task vectors. Parameters where the elected sign differs from a particular task vector's sign are zeroed out for that task vector.

This is the key conflict-resolution step. By choosing one sign per parameter and discarding the dissenting opinions, TIES avoids the cancellation that plagues naive averaging.

### 5.3 Disjoint Merge

Finally, TIES averages the remaining (non-zero, sign-aligned) values for each parameter across all task vectors. The resulting merged task vector is added to the base model.

In practice, TIES significantly outperforms naive averaging and Task Arithmetic when merging fine-tunes for unrelated tasks. It became the de facto baseline for model merging in the open-source community by mid-2024.

## 6. DARE: Drop And REscale

DARE ("Language Models are Super Mario," 2023) takes a different approach to the same problem. It drops a large fraction of each task vector's parameters at random, then rescales the surviving parameters to compensate.

### 6.1 Random Dropping

DARE selects a fraction p of each task vector's parameters (typically p = 0.7 to 0.99) and zeros them out. This leaves only a small subset of the task vector active. Surprisingly, dropping up to 99% of fine-tuning changes often has minimal impact on the model's performance — the bulk of fine-tuning information is concentrated in a small subset of weights.

### 6.2 Rescaling

After dropping, DARE rescales the surviving weights by 1/(1-p) to maintain the expected magnitude of the task vector. This is analogous to how dropout in training rescales activations. The rescaled task vector has the same expected effect as the original but is much sparser.

### 6.3 DARE + TIES

DARE is often combined with TIES — apply DARE first to sparsify the task vectors, then apply TIES to resolve sign conflicts and average. The resulting method (DARE-TIES) is currently the most popular merging recipe in the open-source community, used by hundreds of merged models on Hugging Face.

The intuition for why DARE works is that fine-tuning produces highly redundant updates: many of the parameter changes are correlated and can be removed without losing the underlying learned behavior. DARE exploits this redundancy by keeping a sparse skeleton of the task vector that suffices for the task.

## 7. SLERP: Spherical Linear Interpolation

SLERP merges two models by interpolating along the surface of a hypersphere rather than along a straight line in weight space. The motivation is that the angle between two weight vectors is a more meaningful distance than the Euclidean distance, especially for normalized weight matrices.

For two models θ_1 and θ_2, SLERP computes:

```
θ_merged = sin((1-t)·Ω)/sin(Ω) · θ_1 + sin(t·Ω)/sin(Ω) · θ_2
```

where Ω is the angle between θ_1 and θ_2, and t ∈ [0, 1] controls the interpolation point.

SLERP works well for merging exactly two models (especially when both are fine-tunes of the same base on related tasks) and is limited to two-model merges in its standard form. It is the default method for two-model merges in mergekit.

## 8. mergekit: The Tooling Ecosystem

mergekit is the dominant open-source tool for model merging. Created by Charles Goddard in 2023, it implements all the methods described above (linear, task arithmetic, TIES, DARE, SLERP, plus several others) and provides a YAML-based configuration interface for specifying merge recipes. A typical mergekit config looks like:

```yaml
models:
  - model: NousResearch/Nous-Hermes-2-Mistral-7B
    parameters:
      density: 0.5
      weight: 0.5
  - model: HuggingFaceH4/zephyr-7b-beta
    parameters:
      density: 0.5
      weight: 0.5
merge_method: dare_ties
base_model: mistralai/Mistral-7B-v0.1
parameters:
  int8_mask: true
dtype: bfloat16
```

This recipe merges two Mistral-7B fine-tunes using DARE-TIES with equal weights and 50% density. The output is a single merged model, written to disk in the standard Hugging Face format and immediately usable by transformers, vLLM, llama.cpp, etc.

mergekit also supports several "exotic" merging operations: layer-wise interpolation (different merge weights at different depths of the network), Frankenmerging (interleaving layers from different models to produce a deeper network with more parameters), and module replacement (swapping specific submodules between models). These techniques produce models that mergekit's own documentation describes as "experimental" — they sometimes work spectacularly well, sometimes catastrophically fail, with no good way to predict in advance.

## 9. The Open LLM Leaderboard Era

For most of 2024, the Open LLM Leaderboard at Hugging Face was dominated by merged models. The pattern was striking: a team would release a fine-tune that scored well, another team would release a different fine-tune that also scored well, and within days a third team would publish a merge of the two that scored higher than either parent. The leaderboard became an evolutionary system, with models combining and recombining at a rapid pace.

The phenomenon was both genuine and partly artificial. Genuine, in that merged models really did combine capabilities from their parents and often improved on benchmark performance. Artificial, in that some of the improvement was due to overfitting to the specific benchmarks used in the leaderboard — a merged model could be tuned (by trying many merge configurations and picking the best) to score well on MMLU and HellaSwag without genuinely improving general capability.

The eventual response from the Open LLM Leaderboard maintainers was to add more diverse benchmarks (the v2 leaderboard with IFEval, BBH, MATH, GPQA, MuSR, MMLU-Pro) and to deprioritize models with no clear training methodology. Merging is still common but no longer dominates the way it did in early 2024.

## 10. When Merging Helps and When It Doesn't

### 10.1 When It Helps

Model merging tends to help when:

- **The constituent models share a base.** Merging two Llama-3.1-8B fine-tunes is reasonable; merging Llama-3.1-8B with Qwen2.5-7B (different architectures, different bases) is not.
- **The tasks are complementary, not conflicting.** A coding fine-tune and a math fine-tune are complementary (the skills don't interfere). A safety fine-tune and a helpfulness fine-tune may conflict (they pull the model in opposing directions).
- **The fine-tunes are mild rather than extreme.** Models that have been heavily retrained on narrow data may have moved too far in weight space for merging to recover useful behavior.
- **You have a way to evaluate the merged model.** Merging without an evaluation loop is gambling.

### 10.2 When It Doesn't

Merging tends to fail or underperform when:

- **The models are from different bases.** Weight-space averaging doesn't work across architectural lineages.
- **The tasks are in genuine conflict.** No amount of TIES sign election can reconcile fundamentally incompatible learned behaviors.
- **The benchmarks don't reflect the use case.** A model that improves on MMLU through merging may not improve on real applications.
- **You need long-term stability.** Merged models can have subtle bugs (degraded behavior on rare inputs, slightly worse calibration) that are hard to detect in standard evaluations but matter in production.

## 11. The Theory Lag

Model merging is a textbook example of practice running ahead of theory. The methods work, the empirical evidence is strong, but the theoretical understanding of why and when merging produces good models remains incomplete. There is active research on:

- Why linear mode connectivity holds for fine-tuned models but not for randomly initialized ones
- How to predict in advance whether a particular merge will succeed
- Whether merging is fundamentally limited or whether better methods can extract more from the input models
- The relationship between merging and other techniques like ensembling, distillation, and continual learning

For now, model merging is a craft as much as a science. Practitioners know that DARE-TIES with density 0.5 and equal weights tends to work well for two-model merges of related fine-tunes, but the rationale is empirical rather than first-principles.

## 12. Conclusion

Model merging is one of the most cost-effective tools in the open-source LLM toolkit. For the price of loading some weights and running a few minutes of CPU computation, you can produce a model that combines capabilities from multiple parents, often outperforming any single parent on the metrics that matter. The technique requires no retraining, no labeled data, and no GPU compute beyond what's needed to evaluate the result.

The methods have matured from simple averaging through Task Arithmetic, TIES, DARE, and SLERP, with mergekit emerging as the standard tool. Best practices have stabilized: use DARE-TIES for multi-model merges, use SLERP for two-model merges of closely related fine-tunes, evaluate aggressively, and don't expect merging to bridge architectural lineages.

The deeper significance is that model merging demonstrates how much "free" capability is sitting in the open-source ecosystem, waiting to be combined. Every fine-tune released to Hugging Face is a potential ingredient in someone else's merge. The compositional nature of the technique creates network effects: as more high-quality fine-tunes are released, the value of merging grows because there are more useful combinations to try. This is one of the genuinely unique advantages of the open-source LLM ecosystem over closed alternatives, and it will continue to drive innovation in the post-foundation-model era of AI development.
