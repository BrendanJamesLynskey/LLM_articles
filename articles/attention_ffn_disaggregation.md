# Attention–FFN Disaggregation: Why the Future of LLM Inference Splits the Transformer in Two

## Introduction

In March 2026, NVIDIA announced Groq 3 LPX — a rack-scale inference accelerator designed not to replace GPUs, but to work alongside them. The core idea is striking: split the transformer's decode loop across two fundamentally different kinds of hardware. GPUs handle attention. A new class of SRAM-first processors handle the feed-forward network (FFN) and Mixture-of-Experts (MoE) layers. Intermediate activations shuttle between the two on every single token.

This design, called Attention–FFN Disaggregation (AFD), is not a quirk of NVIDIA's product strategy. It reflects a deep architectural insight about the transformer itself — that attention and feed-forward computation have such different hardware requirements during token generation that running them on the same chip forces a compromise neither side wins.

To understand why AFD works, and why it has emerged now rather than three years ago, you need to understand three things in sequence: what the transformer actually computes at each layer, what makes those computations hard for hardware, and how Mixture-of-Experts models amplify the divergence between attention and FFN to the point where disaggregation becomes not just viable but necessary.

This article builds that understanding from first principles.

---

## Part 1: Inside a Transformer Layer

A decoder-only transformer — the architecture behind GPT, LLaMA, DeepSeek, and essentially every modern large language model — is a stack of identical layers, each applied in sequence to produce one output token at a time. A model like DeepSeek-V3 has 61 such layers. LLaMA 3 405B has 126.

Each layer has two main stages: an attention mechanism followed by a feed-forward network. Between and around these stages sit residual connections and normalisation operations, but the computational weight lies overwhelmingly in attention and FFN.

### 1.1 The Attention Mechanism

Self-attention is the operation that allows each token to incorporate information from every other token in the context. At its core, it works as follows.

The input to each layer is a matrix of hidden states — one vector per token, each of dimensionality *d* (for example, d = 7168 in DeepSeek-V3). Three linear projections transform each token's hidden state into a query vector (Q), a key vector (K), and a value vector (V). These projections are learned weight matrices.

The query of the current token is compared against the keys of all previous tokens using a scaled dot product. This produces a set of attention scores — essentially, a measure of how relevant each past token is to the current one. These scores are normalised via softmax to form a probability distribution, which is then used to compute a weighted sum of the value vectors. The result is a context-aware representation of the current token that has selectively incorporated information from the entire preceding sequence.

In practice, this is done across multiple "heads" in parallel — multi-head attention (MHA) — so the model can attend to different kinds of relationships simultaneously. Variants like Grouped-Query Attention (GQA) and Multi-Head Latent Attention (MLA) reduce the memory cost of storing keys and values, but the fundamental operation remains the same.

The critical detail for our purposes is what happens during the *decode* phase — when the model is generating output tokens one at a time. For each new token, the Q/K/V projections are computed for that single token, but the attention score computation must reference the key and value vectors of *every* previous token. These cached keys and values — the KV cache — grow with every generated token and with the length of the original prompt.

### 1.2 The Feed-Forward Network

After attention, each token's representation passes through a feed-forward network. In the original transformer, this is two linear transformations with a nonlinear activation between them:

    FFN(x) = W₂ · activation(W₁ · x)

In modern models, the activation is typically SwiGLU, which introduces a third weight matrix and a gating mechanism, but the essential character is the same: the FFN is a position-wise operation. Each token passes through the same weights independently. There is no interaction between tokens at this stage — the FFN processes each token's vector in isolation.

The FFN is where the majority of a transformer's parameters live. In a standard dense transformer, the FFN's hidden dimension is typically 4× the model dimension (or ~8/3× with SwiGLU), meaning the FFN weight matrices contain roughly two-thirds of the total parameters in each layer. The remaining third sits in the attention projections.

Crucially, the FFN is *stateless*. It has no memory of what it computed for previous tokens. It applies the same fixed function to whatever vector it receives. This stands in sharp contrast to attention, which is inherently stateful — the KV cache makes attention's computation dependent on the entire history of the sequence.

### 1.3 Putting It Together

A single forward pass through one transformer layer during decode looks like this:

1. **Attention**: Project the new token into Q, K, V. Store the new K and V in the cache. Compute attention scores against all cached K vectors. Weight the cached V vectors accordingly. Project the result back to the model dimension.
2. **FFN**: Pass the attention output through the feed-forward network — two or three large matrix multiplications with a nonlinear activation.
3. **Residual + Norm**: Add the result back to the input and normalise.

This entire sequence repeats for every layer. Then the final output is projected into vocabulary logits, a token is sampled, and the whole process begins again for the next token.

---

## Part 2: Why Attention and FFN Have Different Hardware Needs

The two stages of a transformer layer are both "matrix multiplications," but they stress hardware in fundamentally different ways. To understand why, we need the concept of *arithmetic intensity*.

### 2.1 Arithmetic Intensity and the Roofline Model

Arithmetic intensity is the ratio of floating-point operations (FLOPs) to bytes of data moved from memory, measured in operations per byte (op/B). It captures how much useful work a computation does for each byte of data it has to fetch.

Every processor has two ceilings: a peak compute rate (measured in FLOPS) and a peak memory bandwidth (measured in bytes per second). If a computation has low arithmetic intensity, the processor runs out of data to work on before it runs out of compute capacity — it is *memory-bandwidth bound*. If a computation has high arithmetic intensity, the processor's compute units are fully utilised and memory bandwidth has headroom — it is *compute bound*.

The crossover point is called the *ridge point*. For an NVIDIA H100, with ~2000 TFLOPS of FP8 compute and ~3.35 TB/s of HBM bandwidth, the ridge point is roughly 600 op/B. Any operation below this intensity is memory bound on that hardware.

### 2.2 Attention During Decode Is Memory Bound

During decode — generating tokens one at a time — attention has a severe arithmetic intensity problem.

For each new token, the attention mechanism must load the entire KV cache from memory. If the context length is *S* tokens, the cache contains 2 × S × d bytes per head per layer (one set of keys, one set of values), typically in FP16 or FP8. Against this massive data movement, the actual computation is modest: a single query vector is dot-producted against S key vectors, and S value vectors are combined. The FLOPs scale as O(S × d) per head, but the bytes loaded also scale as O(S × d).

This gives attention an arithmetic intensity of approximately 1–2 op/B during single-token decode. For comparison, the ridge point of a modern GPU is in the hundreds. Attention is deep in the memory-bound regime — the GPU's tensor cores sit largely idle, waiting for the memory system to deliver KV cache data.

Batching multiple requests together helps, because different requests can share the weight loads (for the Q/K/V projections), but the KV caches are per-request and cannot be shared. This means that as context lengths grow, even batching cannot rescue attention from memory-boundedness. At 128K context on a model like DeepSeek-V3, the KV cache dominates the memory traffic.

This is why attention benefits enormously from large, high-bandwidth memory systems — exactly what GPUs with their massive HBM pools provide.

### 2.3 FFN During Decode Is (Potentially) Compute Bound

The FFN tells a different story. During decode, each token passes through the same weight matrices. The weights are large (two-thirds of the per-layer parameters), but they are the same for every token in the batch. This means the weights can be loaded once from memory and reused across every token in the batch.

If the batch size is B tokens, the arithmetic intensity of an FFN layer is approximately 2B op/B (in FP16) — it scales linearly with batch size. At batch size 1, the FFN is memory bound (intensity ~2, well below the ridge point). But at batch sizes in the hundreds, the FFN becomes thoroughly compute bound.

For a dense model, this means that with sufficient batching, FFN layers can fully utilise a GPU's tensor cores. The problem is that achieving those batch sizes during decode is not always possible — latency constraints, memory limits for KV caches, and the sequential nature of generation all conspire to keep per-user batch sizes small.

Still, the key structural insight holds: *FFN arithmetic intensity is tuneable via batching, while attention arithmetic intensity is not* (because the KV cache is per-request). This asymmetry is the seed from which AFD grows.

### 2.4 Summary of the Asymmetry

| Property | Attention (Decode) | FFN (Decode) |
|---|---|---|
| **Stateful?** | Yes — depends on KV cache | No — stateless, same weights for all tokens |
| **Memory footprint** | Grows with context length | Fixed (model weights only) |
| **Arithmetic intensity** | ~1–2 op/B (memory bound) | ~2B op/B; scales with batch |
| **What it needs** | Large memory, high bandwidth | High compute throughput, or high bandwidth if batching is low |

---

## Part 3: Mixture of Experts — Amplifying the Split

Mixture of Experts is the architectural trend that turns the attention/FFN asymmetry from a theoretical curiosity into an urgent systems problem.

### 3.1 What MoE Replaces

MoE applies exclusively to the FFN portion of the transformer. The attention mechanism is left completely untouched. In a standard transformer layer, every token passes through a single, shared FFN. In an MoE layer, that single FFN is replaced by multiple independent FFNs — the "experts" — plus a small router (or gating) network that decides which experts each token should use.

This is worth emphasising because it directly answers the question of whether MoE's efficiency gains extend beyond the FFN: **they do not**. MoE is a technique for making the FFN sparse. It does not affect the attention computation, the KV cache, the Q/K/V projections, or any other part of the transformer. The attention half of every layer remains fully dense, fully activated, and carries the same computational and memory costs as in a non-MoE model.

### 3.2 How MoE Works in Detail

Consider DeepSeek-V3 as a concrete example. It has 671 billion total parameters, but only 37 billion are activated for any given token. The vast majority of those total parameters — and the vast majority of the sparsity savings — live in the MoE layers.

In each of DeepSeek-V3's 61 layers (excepting the first 3, which use dense FFNs), the FFN is replaced by:

- **256 routed experts**, each a small FFN with a hidden dimension of 2048
- **1 shared expert** that is always activated
- **A router** that assigns each token to its top 8 experts (out of 256)

So for each token, 8 routed experts plus 1 shared expert are activated — 9 small FFNs in total. The remaining 247 routed experts do nothing for that token. Their weights are not loaded, their computations are not performed.

Each expert is itself a standard FFN: two or three linear projections with a nonlinear activation. The router is a small linear layer that takes the token's hidden state as input and produces a score for each expert. The top-k scores determine which experts fire.

### 3.3 Why MoE Exists: Decoupling Parameters from Compute

The fundamental motivation for MoE is that model quality scales with total parameter count, but inference cost scales with *activated* parameter count. A 671B-parameter MoE model stores far more "knowledge" in its weights than a 37B dense model, but generating each token costs roughly the same as running a 37B dense model — because only 37B parameters are active for any given token.

This is sometimes called the "decoupling" of capacity from compute. The model has the knowledge capacity of a very large model and the per-token cost of a much smaller one.

But this decoupling has a catch: all those inactive expert weights still exist in memory. They must be stored somewhere, even if they are not used on any particular token. And the *which-experts-to-activate* decision changes from token to token, so the system cannot simply discard the unused experts.

### 3.4 The Memory Problem MoE Creates

A dense 37B model has 37 billion parameters — perhaps 37 GB in FP8, or 74 GB in FP16. These weights must be loaded from memory for each token during decode, but the total memory footprint is manageable.

DeepSeek-V3's 671B parameters require roughly 671 GB in FP8. The active parameters per token (37B) represent about 5.5% of the total. But the full 671 GB must be resident in memory (or at least quickly accessible), because different tokens route to different experts, and the router's decisions are not known in advance.

This creates a situation where:

- The **attention** portion of each layer has a modest, fixed parameter count, but its KV cache grows with context length and is per-request.
- The **MoE/FFN** portion has an enormous total parameter count distributed across experts, but only a small fraction is activated per token, and it has no per-request state.

The two halves of the transformer have almost opposite memory profiles: attention is state-heavy and weight-light; MoE is weight-heavy and state-free.

### 3.5 MoE and Arithmetic Intensity

MoE further complicates the arithmetic intensity picture. In a dense FFN, all tokens in a batch share the same weights, giving excellent data reuse and high arithmetic intensity at moderate batch sizes. In an MoE FFN, tokens are scattered across different experts. If batch size is B and there are E experts with top-k routing, each expert sees on average only B × k / E tokens.

For DeepSeek-V3 with 256 experts and top-8 routing, each expert sees B × 8/256 = B/32 tokens on average. To reach the same arithmetic intensity as a dense model at batch size B, you would need a batch 32× larger. Concretely, for an H100 at FP8, reaching the compute-bound regime with a dense FFN might require a batch of ~120 tokens. For DeepSeek-V3's MoE, the equivalent requirement is ~3,840 tokens — a batch size that is difficult to achieve during interactive decode while meeting latency targets.

This means MoE experts during decode are often **memory-bandwidth bound**, even more so than a dense FFN would be. The sparsity that makes MoE efficient in total FLOPs also fragments the workload in a way that reduces per-expert utilisation.

### 3.6 Expert Parallelism and the All-to-All Communication Pattern

To serve a large MoE model, the experts are typically distributed across many accelerators using *expert parallelism* (EP). Each GPU hosts a subset of experts. When a batch of tokens arrives at an MoE layer, each token must be routed to its assigned experts — which may reside on different GPUs. This requires an *all-to-all* communication step: every GPU sends tokens to every other GPU (dispatch), the experts process their assigned tokens, and the results are sent back (combine).

This all-to-all pattern is structurally quite different from the communication patterns in attention (which is dominated by KV cache access and typically uses tensor parallelism within a node). The MoE communication pattern is inherently cross-node and involves variable, data-dependent routing.

This creates a natural *seam* in the computation — a point where the data flow changes character dramatically. Attention wants local, bandwidth-rich access to a large per-request state. MoE wants to scatter and gather tokens across a distributed pool of experts. These two patterns pull in different directions, and trying to serve both on the same tightly coupled cluster of GPUs creates inefficiency on both sides.

---

## Part 4: The Three Levels of Disaggregation

The LLM inference community has progressively discovered that the transformer's internal structure maps onto a hierarchy of disaggregation opportunities.

### 4.1 Level 1: Prefill–Decode Disaggregation

The first level of disaggregation, now standard practice, separates the *prefill* phase (processing the input prompt) from the *decode* phase (generating output tokens one at a time).

During prefill, the model processes all input tokens in parallel. This involves large matrix-matrix multiplications and is heavily compute bound. During decode, the model generates one token at a time, mostly performing matrix-vector operations against cached state. Decode is memory-bandwidth bound.

Running both on the same hardware forces a compromise: either the hardware is sized for prefill's compute demands (wasting capacity during decode) or for decode's bandwidth demands (bottlenecking prefill). Prefill-decode disaggregation, pioneered by systems like DistServe, separates these phases onto different pools of hardware that can be independently scaled.

### 4.2 Level 2: Attention–FFN Disaggregation (AFD)

The second level recognises that even within the decode phase, the two halves of each transformer layer have different bottlenecks.

Attention during decode is stateful and memory-bandwidth bound. It needs large memory capacity (for the KV cache), high memory bandwidth (to read that cache quickly), and relatively modest compute. It benefits from hardware with large HBM pools and high bandwidth-to-compute ratios.

FFN/MoE during decode is stateless and, depending on batching, either memory-bandwidth bound (at low batch) or compute bound (at high batch). It needs access to a large weight store but has no per-request state. For MoE specifically, it needs high-bandwidth interconnects for expert dispatch/combine, and benefits from being able to aggregate tokens across many concurrent requests to increase per-expert batch sizes.

AFD splits these two computations onto different hardware, exchanging the intermediate activation tensor (the hidden state of each token) between them on every token. The attention nodes compute attention and send the result to FFN nodes; the FFN nodes compute the FFN/MoE and send the result back for the next layer's attention. This ping-pong repeats for every layer of the model, for every token generated.

### 4.3 Level 3: Experts-as-a-Service

A more aggressive variant, sometimes called EaaS, fully decouples the MoE expert pool from both attention and the model's layer structure. Experts are deployed as independent, stateless microservices that can be scaled, load-balanced, and routed to independently. This enables even finer-grained resource allocation but introduces significant orchestration complexity and latency sensitivity.

---

## Part 5: NVIDIA's Vera Rubin + LPX Architecture

NVIDIA's Groq 3 LPX system is a concrete instantiation of AFD, designed to operate alongside the Vera Rubin NVL72 GPU cluster.

### 5.1 The GPU Side: Vera Rubin NVL72

The Rubin GPU is NVIDIA's next-generation general-purpose GPU, succeeding Blackwell. Deployed in the NVL72 configuration (72 GPUs per rack with high-bandwidth NVLink interconnect), it handles:

- **Prefill**: Processing input prompts. This is a large, compute-heavy operation that benefits from the GPU's massive FLOPS and large HBM capacity.
- **Decode attention**: Computing the attention mechanism during token generation. This requires access to the KV cache, which resides in GPU HBM. The GPU's large memory pool and high HBM bandwidth make it well suited to this memory-bound operation.

The GPU does *not* run the FFN/MoE computation during decode in the AFD configuration. Instead, after computing attention, it sends the intermediate activation tensor to the LPX system.

### 5.2 The LPU Side: Groq 3 LPX

The LPX rack contains 256 Groq 3 LPU accelerators — a fundamentally different kind of processor. Where GPUs are built around massive HBM pools and dynamic hardware scheduling, the LPU is built around:

- **On-chip SRAM as primary working memory**: 500 MB per chip, 128 GB at rack scale. SRAM is far faster than HBM — the rack delivers 40 PB/s of on-chip bandwidth, roughly 10× what a comparable GPU cluster achieves from HBM.
- **Deterministic, compiler-orchestrated execution**: Instead of dynamic hardware schedulers, the LPU's compiler explicitly schedules every computation, memory access, and data movement. Execution timing is predictable down to the cycle.
- **High-bandwidth chip-to-chip interconnect**: 2.5 TB/s bidirectional per chip, 640 TB/s at rack scale, enabling fast activation exchange and expert dispatch.

The LPU handles the FFN/MoE computation during decode. Expert weights are distributed across the 256 chips' SRAM, and tokens are routed to the appropriate experts via the chip-to-chip fabric.

### 5.3 Why This Split Makes Sense

The FFN/MoE computation during decode has two key properties that the LPU architecture exploits:

**First, it is stateless.** The FFN has no KV cache — it needs only the current token's activation and the expert weights. This means the LPU does not need large HBM pools for per-request state. The expert weights can be pre-loaded into SRAM and reused across all requests.

**Second, its performance is dominated by weight-loading bandwidth at low batch sizes.** The SRAM-first architecture of the LPU delivers vastly more bandwidth per FLOP than a GPU's HBM can. For memory-bound FFN/MoE operations, this translates directly into lower latency: the weights are already in fast on-chip memory, and the deterministic execution model eliminates the jitter and scheduling overhead that GPUs incur.

Meanwhile, attention stays on the GPU because it *is* stateful — the KV cache is large, per-request, and grows with context. The GPU's HBM is the right home for this data, and the GPU's flexible execution model handles the variable-length, per-request nature of attention well.

### 5.4 The AFD Decode Loop in Practice

For each generated token, the decode loop works as follows:

1. **GPU**: Compute attention for the new token — load Q/K/V projections, update KV cache, compute attention scores, produce attention output.
2. **GPU → LPX**: Send the intermediate activation tensor (the attention output) to the LPX system over the network.
3. **LPX**: Route the activation to the appropriate MoE experts, compute the FFN, combine expert outputs.
4. **LPX → GPU**: Send the FFN output back to the GPU.
5. **GPU**: Apply residual connection and normalisation, then proceed to the next layer.

This ping-pong repeats for every layer. The activation tensor being exchanged is small — just the hidden state for each token in the batch (e.g., 7168 × batch_size in FP8), far smaller than the KV cache or the expert weights. This makes the communication cost per step manageable.

### 5.5 The Role of NVIDIA Dynamo

Making this heterogeneous loop operational in production requires sophisticated orchestration. NVIDIA Dynamo handles:

- **Request routing**: Classifying incoming requests by latency sensitivity and directing them to the appropriate serving path (GPU-only for throughput, GPU+LPX for low latency).
- **Activation transfer**: Moving intermediate tensors between GPU and LPX with minimal overhead and predictable timing.
- **KV-aware scheduling**: Ensuring that the GPU side can efficiently manage KV caches across many concurrent requests while the LPX side aggregates FFN work from multiple requests to improve expert utilisation.
- **Tail latency control**: Preventing stragglers and jitter from degrading the user-visible token generation rate.

---

## Part 6: Speculative Decoding and AFD

AFD also enables a natural hardware split for speculative decoding. In speculative decoding, a small "draft" model generates candidate tokens quickly, and a larger "target" model verifies them in parallel. When candidates are accepted, multiple tokens are committed at once, increasing effective throughput.

LPX is well suited to run the draft model. The draft model is small enough to fit in the LPX's SRAM, and the LPU's deterministic, low-latency execution model makes it very fast at generating candidate tokens — exactly the property you want from a draft engine. The GPU, with its larger memory and higher compute throughput, handles the verification step efficiently.

This creates a clean division: LPX drafts, GPU verifies. Both processors operate in their respective sweet spots.

---

## Part 7: The Bigger Picture — Why Now?

### 7.1 MoE Has Gone Mainstream

Two years ago, MoE was an exotic architecture used by a handful of models (Mixtral, early Switch Transformers). Today, the most capable open-weight models — DeepSeek-V3/R1, Qwen3, LLaMA 4 Maverick — all use MoE. This means that the attention/FFN asymmetry is no longer an edge case; it is the default computational profile of frontier models.

### 7.2 Context Lengths Are Growing

As context windows extend to 128K tokens and beyond, the KV cache grows proportionally. Attention becomes more memory-bound, the KV cache consumes more HBM, and the gap between attention's needs and FFN's needs widens.

### 7.3 Interactive Latency Demands Are Rising

Agentic workloads — where AI systems chain multiple inference calls together with tool use, retrieval, and reasoning — make per-token latency compound. A coding assistant that calls a model 50 times in a reasoning chain is 50× more sensitive to individual token latency than a single-turn chatbot. This pushes systems toward latency-optimised decode, which is exactly where AFD provides its advantage.

### 7.4 Prefix Caching Shifts the Balance

As prefix caching and prompt reuse become standard, the prefill phase gets cheaper — shared prompts are computed once and cached. This means a larger fraction of the total compute per request is spent in decode, further increasing the payoff from optimising decode specifically.

### 7.5 The Economics

NVIDIA claims the Vera Rubin NVL72 + LPX combination delivers up to 35× higher tokens-per-second per megawatt at 400 tokens/second per user, compared to the previous-generation GB200 NVL72. The economic implication is significant: by specialising hardware for the two halves of the decode loop, data centres can serve high-value, latency-sensitive workloads (coding assistants, agentic workflows, voice interfaces) far more efficiently than with a homogeneous GPU fleet.

---

## Part 8: Challenges and Open Questions

AFD is not a free lunch. Several challenges remain.

**Communication overhead.** Every layer of every token requires a round trip between GPU and LPU. The activation tensor is small, but the latency of each transfer adds up across 60+ layers. High-bandwidth, low-latency interconnects are essential, and any network jitter directly impacts per-token latency.

**Dense models are harder.** Current AFD implementations work best with MoE models, where the all-to-all expert dispatch already introduces a communication pattern that AFD can piggyback on. For dense models, the FFN has no natural scatter/gather step, and the communication overhead of AFD is proportionally larger relative to the compute savings. Research from the Hao AI Lab at UCSD notes that AFD for dense models remains an open challenge.

**Load balancing.** The attention side's per-step latency varies with context length (longer contexts mean more KV cache to read). The FFN side's latency is more stable. If the two are not carefully balanced, one side idles while the other finishes, creating pipeline bubbles. The optimal ratio of attention-to-FFN hardware depends on model architecture, context length distribution, and batch size — and may change dynamically as workload mix shifts.

**Scaling.** AFD introduces a bipartite communication topology (M attention nodes talking to N FFN nodes) that is more complex than the symmetric all-to-all of traditional expert parallelism. Scheduling, routing, and fault tolerance all become harder.

**Maturity.** As of early 2026, production-quality AFD systems are emerging (StepFun's Step-3, ByteDance's MegaScale-Infer, and now NVIDIA's LPX), but the technique is still young. Frameworks like vLLM are actively implementing AFD support, and the optimal design patterns are still being discovered.

---

## Conclusion

The transformer is not a monolithic block of computation. It is a composite of two fundamentally different operations — attention and feed-forward — that happen to be stacked inside the same layer. For years, inference systems treated the transformer as a single unit and optimised it on uniform hardware. This worked well enough when models were small, contexts were short, and dense architectures dominated.

The rise of MoE models, long contexts, and latency-sensitive agentic workloads has exposed the seam between attention and FFN as the central tension in inference system design. Attention is stateful, memory-bandwidth bound, and tied to per-request KV caches. The FFN — especially in MoE form — is stateless, weight-heavy, and benefits from a fundamentally different memory hierarchy.

Attention–FFN Disaggregation is the recognition that this tension is not a problem to be papered over with better batching or bigger GPUs. It is a structural property of the transformer itself, and the right response is to match each half of the computation to the hardware that serves it best. NVIDIA's Vera Rubin + LPX architecture is the first major commercial system built around this principle, but the underlying idea — that the decode loop is a relay race, not a solo sprint — will shape inference infrastructure for years to come.

---

## References and Further Reading

- NVIDIA, "Inside NVIDIA Groq 3 LPX: The Low-Latency Inference Accelerator for the NVIDIA Vera Rubin Platform," NVIDIA Technical Blog, March 2026.
- Zhong et al., "DistServe: Disaggregating Prefill and Decoding for Goodput-optimized Large Language Model Serving," OSDI 2024.
- Zhang et al., "Janus: Disaggregating Attention and Experts for Scalable MoE Inference," arXiv:2512.13525, December 2025.
- Wang et al., "Step-3 is Large yet Affordable: Model-system Co-design for Cost-effective Decoding," arXiv:2507.19427, July 2025.
- Song et al., "Theoretically Optimal Attention/FFN Ratios in Disaggregated LLM Serving," arXiv:2601.21351, January 2026.
- Xiao et al., "Revealing the Challenges of Attention-FFN Disaggregation for Modern MoE Models and Hardware Systems," arXiv:2602.09721, February 2026.
- Kwon et al., "AiDE: Attention-FFN Disaggregated Execution for Cost-Effective LLM Decoding on CXL-PNM," IEEE CAL, 2025.
- DeepSeek-AI, "DeepSeek-V3 Technical Report," arXiv:2412.19437, December 2024.
- Fedus et al., "Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity," JMLR, 2022.
- Shazeer et al., "Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer," ICLR 2017.
- Williams et al., "Roofline: An Insightful Visual Performance Model for Multicore Architectures," SC 2009.
- Hao AI Lab, "Disaggregated Inference: 18 Months Later," Blog, November 2025.
