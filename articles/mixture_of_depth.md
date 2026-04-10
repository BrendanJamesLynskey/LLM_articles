# Mixture of Depth and Adaptive Compute for Transformers

*April 2026 • Technical Report*

## 1. Introduction

A standard transformer applies the same amount of compute to every token. Whether the token is `the`, `however`, or `transmogrification`, the model runs it through every layer of the network with the same number of operations. This uniformity is mathematically convenient — the entire forward pass can be batched and parallelized — but it is computationally wasteful. Some tokens are much harder to predict than others, and the easy tokens consume the same compute as the hard ones.

The intuition that different tokens deserve different compute budgets is not new. Adaptive computation time was proposed by Alex Graves in 2016 for RNNs, allowing the model to take more recurrent steps on hard inputs. PonderNet (DeepMind, 2021) extended the idea to general neural networks. Universal Transformers (Google, 2018) used iterative refinement with a halting mechanism. None of these became the dominant paradigm — the engineering complexity of dynamic compute outweighed the efficiency gains for most applications.

In 2024, a new variation called Mixture of Depth (MoD), proposed in a Google DeepMind paper, revived this line of thinking specifically for transformers. The key insight was that you could let each layer choose which tokens to process, rather than forcing every token through every layer. The result was a transformer that processed the same content with significantly less compute, with quality competitive with the standard architecture. MoD has since been combined with Mixture of Experts (MoE-MoD models), and the broader idea of adaptive compute has reentered the architecture conversation.

This report examines what Mixture of Depth is, how it differs from related approaches, and what the architectural shift toward adaptive compute might mean for the next generation of LLMs.

## 2. The Wasteful Uniformity of Standard Transformers

To appreciate why MoD matters, consider the work that a standard transformer does for a typical sentence. For the sentence "The cat sat on the mat," each token is processed by every layer of the network, even though the easy tokens (`the`, `cat`, `sat`, `on`, `the`, `mat`) require very little contextual reasoning. A model trained on enough data already "knows" each of these tokens almost from context-free statistics. The deep contextual processing that the transformer's many layers provide is largely wasted on them.

For the sentence "The transmogrification of the simulacrum yielded a pleasing aesthetic resolution," the situation is different. The unusual words and complex syntactic structure benefit from the deep contextual processing. Every layer adds value.

A standard transformer treats both sentences identically. Every token gets every layer. The compute spent on the easy tokens of the first sentence could, in principle, be spent on the harder tokens of the second sentence, producing better results without increasing the total compute budget. But the standard architecture does not provide a mechanism for this redistribution.

## 3. The Mixture of Depth Approach

### 3.1 The Core Idea

MoD modifies the transformer so that each layer can choose to skip some tokens. The mechanism is simple:

1. At each transformer layer, a small router network looks at each token and produces a routing score
2. Based on the scores, the top-k tokens (where k is some fraction of the total tokens, often 50%) are selected to pass through the layer's full computation (attention + FFN)
3. The remaining tokens skip the layer entirely, passing through unchanged via a residual connection

The router is trained jointly with the rest of the model, learning to identify which tokens benefit most from each layer. Different layers may select different tokens, so a token that skips one layer may be selected for the next.

The compute savings come from the fact that only a fraction of tokens go through the full computation at each layer. With k=50%, each layer does roughly half the work it would in a standard transformer, with corresponding savings in FLOPs and inference latency.

### 3.2 The Top-k Choice

The choice of how many tokens to process at each layer is a critical hyperparameter. The MoD paper found that k=12.5% (processing only one in eight tokens at each layer) produced reasonable quality with substantial compute savings. Larger values of k provide more compute and better quality; smaller values save more compute at the cost of quality.

Crucially, the top-k selection is differentiable — the router scores are real-valued, the "selection" is implemented via a learned weighting that is dominated by the highest-scoring tokens. This allows the router to be trained with standard backpropagation.

In practice, the implementation uses a hard top-k at inference time (only the top-k tokens are actually processed) with a soft approximation during training that allows gradients to flow through. This is the same trick used in MoE training, and it works similarly well.

### 3.3 What MoD Saves

For a model with depth L, the standard transformer does O(L × N) work per sequence (where N is the sequence length and the constant factor depends on the model size). MoD with top-k fraction p reduces this to O(L × N × p) work per sequence — a savings of (1-p)×.

For p=0.5, the savings is 50%. For p=0.125, the savings is 87.5%. The actual wall-clock improvement is somewhat less due to the routing overhead and the fact that the skip operations still take some time, but the FLOPs reduction translates relatively directly to throughput improvement.

### 3.4 Quality Tradeoffs

The MoD paper reports that MoD models match the performance of standard transformers at the same compute budget. This is the key claim: by spending compute more efficiently (on the tokens that need it), MoD can achieve the same quality with less total work.

Subsequent work has refined this finding. MoD models tend to slightly underperform standard transformers when both are trained for the same number of tokens with the same effective compute. They overperform when the compute budget is tight. The crossover point depends on the specific task and data distribution, but for many practical settings, MoD provides meaningful efficiency gains without significant quality loss.

## 4. MoD vs. Other Adaptive Compute Approaches

### 4.1 vs. Universal Transformers

Universal Transformers (UT) use a single transformer block applied iteratively, with a halting mechanism that decides when to stop processing each position. UT can apply different amounts of compute to different positions, similar in spirit to MoD, but the mechanism is different — UT iterates within a layer rather than skipping layers.

UT has the disadvantage that the iterative computation is harder to parallelize than the standard transformer's stacked-layer architecture. MoD preserves the stacked-layer structure, which makes it more compatible with existing GPU hardware and inference frameworks.

### 4.2 vs. PonderNet

PonderNet generalizes the halting idea to arbitrary neural network architectures. Each step of the network's computation can choose to halt and produce an output, with the probability of halting learned during training. PonderNet has not seen widespread adoption for transformers, partly because the integration with attention is awkward.

### 4.3 vs. Early Exit Networks

Early exit networks attach intermediate output heads to a deep network, allowing predictions to be made at multiple depths. Easy inputs use only the shallow heads; hard inputs continue through the entire network. This provides per-input adaptive compute but at the cost of training complexity (each exit head needs its own loss term).

MoD is more fine-grained: it allows per-token rather than per-input adaptive compute. A single sequence can have some tokens processed by all layers and other tokens processed by only a few.

### 4.4 vs. Mixture of Experts

MoE and MoD are complementary. MoE applies different expert subnetworks to different tokens at each layer; MoD applies the same layer to different subsets of tokens. They can be combined, and recent work has explored MoE-MoD models that get both kinds of adaptive compute.

The combination is intuitively appealing. MoE provides specialization (different experts for different content); MoD provides efficiency (skipping layers for easy tokens). Together they should provide both better quality and better compute efficiency than either alone.

## 5. Implementation Challenges

### 5.1 Routing Overhead

The router that decides which tokens to process is itself a small neural network. Its compute is added to the per-layer cost. For most models, the router is small enough to be negligible (a tiny MLP per layer), but it does eat into the compute savings.

The router also adds memory bandwidth — its weights must be loaded from HBM, and its activations must be computed. For memory-bound inference, the router's bandwidth cost can be a meaningful fraction of the per-layer bandwidth.

### 5.2 Load Imbalance

The set of selected tokens at each layer is dynamic, which creates load balancing challenges similar to MoE's. Different layers may select different fractions of tokens, and the variance can cause some GPUs in a parallel deployment to have more or less work than others.

The standard MoE techniques apply: capacity-aware routing, auxiliary load balancing losses, and expert-parallel deployment patterns. The implementation is messier than MoE because the imbalance occurs within a single layer rather than between layers, but the basic approaches transfer.

### 5.3 KV Cache and Causal Masking

The KV cache for attention assumes that every token contributes a key and value at every layer. With MoD, some tokens skip some layers, which means the KV cache at those layers does not contain those tokens. This complicates the attention computation because the cache is no longer dense.

The standard fix is to keep the KV cache "as if" every token participates, with the skipped tokens contributing their previous-layer values. This loses some of the compute savings but avoids breaking causal masking and the standard attention infrastructure. More aggressive variants compute attention only over the participating tokens, requiring custom attention kernels.

### 5.4 Training Stability

Training MoD models is harder than training standard transformers. The router can collapse to a degenerate solution (always selecting the same tokens, always skipping the same tokens), or it can oscillate between different routing strategies during training. Several stabilization techniques are needed: warm-up periods where the router is gradually introduced, auxiliary losses that encourage diverse routing, and careful learning rate schedules.

These training considerations are similar to those for MoE, and the same body of empirical knowledge applies. Training MoD models is more art than science, and the published recipes are still evolving.

## 6. Production Deployments

As of 2026, MoD is still a research-stage technique. Several papers have explored variations and applications, but no production frontier model has publicly disclosed using MoD as its primary architecture. The technique is mentioned in the context of exploration and ablations rather than as a deployed feature.

Several reasons explain the slow adoption. The compute savings, while meaningful, are not as dramatic as some other innovations (FlashAttention, GQA, MoE) that became standard much faster. The implementation complexity is significant. The training stability issues are real. And the standard transformer is good enough for most use cases that the marginal improvement of MoD does not justify the engineering investment.

The most likely path forward is adoption in compute-constrained settings: on-device inference, edge deployment, or hyperscale serving where every percent of efficiency matters. As inference costs grow, the calculus may shift.

## 7. Related Ideas: Conditional Computation in General

MoD is part of a broader trend toward conditional computation in neural networks. The principle is that not every input deserves the same processing, and architectures that can adapt to input difficulty should be more efficient than uniform ones. Several related ideas have emerged:

**Layer-wise Token Pruning**: Drop tokens (rather than just skip them) at deeper layers, on the theory that deep features are more compressed and don't need every token. Used in some vision transformer optimizations.

**Adaptive Token Merging**: Merge similar tokens to reduce the effective sequence length at deeper layers. Particularly effective for vision and video models where adjacent patches are highly correlated.

**Contextual Pruning**: Drop attention heads or FFN dimensions based on the input. Different architectures emerge for different inputs.

**Mixture of Tokens**: Generalize MoE so that each "expert" sees a different subset of tokens, not just a different parameter set.

None of these have become dominant, but they share the underlying intuition that uniform compute is wasteful and that learning where to spend compute is a powerful optimization axis.

## 8. The Hardware Mismatch

A persistent challenge for adaptive compute architectures is that GPU hardware is optimized for uniform compute. Tensor cores expect rectangular matrix multiplications. Memory access patterns are most efficient when they are predictable. Branch prediction and instruction caching work best when control flow is consistent across threads. Adaptive compute, with its variable per-token work, fights all of these design assumptions.

The result is that the theoretical FLOPs savings of adaptive compute do not always translate into proportional wall-clock improvements. A model that does 50% fewer FLOPs may run only 30-40% faster in practice, because the irregular compute pattern reduces hardware efficiency.

This is one of the main reasons MoD has not seen explosive adoption. The benefit-to-complexity ratio is lower than alternatives that work with the hardware's grain rather than against it.

The longer-term hope is that hardware will evolve to better support adaptive compute. Some researchers have proposed accelerators with native support for sparse and dynamic computation, which would close the gap between the theoretical and practical efficiency of MoD-style architectures. None are commercial yet.

## 9. The Bigger Architectural Question

MoD raises a deeper question about transformer architecture: is uniform compute the right default? The standard transformer's success suggests that uniform compute, while wasteful, has been good enough. The success of MoE — which is also a form of conditional compute — suggests that there are gains to be had from non-uniformity. The mixed reception of MoD suggests that the right kind of non-uniformity matters more than the existence of non-uniformity.

The next several years will likely see more experimentation in this space. Some of the experiments will fail. Some will succeed. The successful ones will probably look quite different from MoD — perhaps combining adaptive compute with specialized hardware support, or combining it with other architectural innovations in ways that compound rather than fight each other.

For now, MoD is best understood as an early data point rather than a conclusive answer. It demonstrates that adaptive compute can work for transformers, that the engineering challenges are tractable, and that meaningful efficiency gains are achievable. Whether it becomes the dominant paradigm or remains a footnote depends on what comes next.

## 10. Conclusion

Mixture of Depth is an attractive idea: spend compute on the tokens that need it, save compute on the tokens that don't. The mechanism is simple, the implementation is tractable, and the published results demonstrate meaningful efficiency gains. The barriers to adoption are not technical — they are pragmatic. Standard transformers are good enough for most use cases, the engineering complexity of MoD is non-trivial, and the hardware does not give MoD as much benefit as the FLOPs counts would suggest.

For practitioners, MoD is worth knowing about but not yet worth deploying outside of research settings. The efficient inference of standard transformers — through the techniques discussed in many other articles in this collection — provides better cost reduction at lower complexity for most workloads. MoD becomes interesting when those techniques are exhausted and the only remaining axes of improvement are architectural.

The deeper significance is that MoD is part of a larger questioning of the uniform-compute assumption that has been the bedrock of deep learning since AlexNet. As models grow and the cost of uniform compute becomes increasingly visible, the pressure to find non-uniform alternatives will grow. MoD will not be the final answer, but it is an important early step in a direction that the field is collectively heading.
