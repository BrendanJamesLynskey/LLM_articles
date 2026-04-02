**Post-Transformer Language Models**

Architectures Beyond Standard Attention

March 2026 • Technical Report

Table of Contents

1\. Introduction

The transformer architecture, introduced in 2017, has dominated language modeling for nearly a decade. Its core mechanism — dense self-attention over the full sequence — provides remarkable expressiveness and is straightforward to parallelize during training. However, the quadratic scaling of attention with sequence length, the linear growth of the key-value cache during autoregressive inference, and the fixed computation applied to every token regardless of difficulty have driven a sustained effort to develop alternative architectures that preserve the transformer's quality while addressing its efficiency limitations.

This report examines the major architectural families that aim to move beyond the standard transformer: state space models, linear attention variants, modern recurrent networks, retention mechanisms, and hybrid architectures that combine elements from multiple paradigms. These approaches share a common goal — achieving sub-quadratic or linear complexity in sequence length — but differ substantially in their mathematical foundations, training characteristics, and practical trade-offs.

2\. Transformer Limitations and the Case for Alternatives

2.1 Quadratic Attention Complexity

Standard self-attention computes a score between every pair of tokens, yielding O(n²) time and memory complexity for a sequence of length n. While FlashAttention and related techniques reduce the memory overhead, the fundamental compute cost remains quadratic. For a 128K-token context, the attention computation is 16 times more expensive than for a 32K context. This scaling makes very long sequences prohibitively costly for both training and inference, even on modern hardware.

2.2 KV Cache Growth

During autoregressive generation, transformers must store the key and value vectors for every previously generated token across every layer and attention head. For a model like Llama 3 70B with 80 layers and 8 KV heads using grouped-query attention, the KV cache grows by roughly 0.33 MB per token in FP16. At 128K tokens, this amounts to approximately 40 GB of KV cache alone — consuming half the memory of a single H100 GPU. This growth is the dominant memory bottleneck for long-context generation and limits the batch sizes achievable during serving.

2.3 Fixed Computation Per Token

Transformers apply identical computation to every token regardless of its complexity or importance. The word "the" receives the same number of FLOPS as a token that requires complex reasoning. This uniform allocation is wasteful for easy tokens and potentially insufficient for hard ones. While mixture-of-experts provides some conditional computation at the FFN level, the attention computation remains uniformly applied.

2.4 Inference Efficiency

For autoregressive decoding, the transformer must attend to all previous tokens for each new token generated. This means generation latency grows linearly with sequence length even within a single decode step. For applications requiring real-time generation over long contexts — streaming agents, long-form dialogue, continuous document processing — this scaling presents a practical barrier that sub-quadratic architectures aim to eliminate.

3\. State Space Models

3.1 Foundations: S4 and HiPPO

State space models (SSMs) reformulate sequence modeling through the lens of continuous-time dynamical systems. The foundational work, S4 (Structured State Spaces for Sequence Modeling), introduced in late 2021 by Gu et al., models a sequence as a discretized linear time-invariant system:

h'(t) = A h(t) + B x(t), y(t) = C h(t) + D x(t)

where A is the state matrix, B the input matrix, C the output matrix, and h(t) is a hidden state. After discretization with step size Δ, this becomes a linear recurrence that can be computed either recurrently (one step at a time) or convolutionally (over the full sequence in parallel).

The critical insight of S4 was the HiPPO (High-order Polynomial Projection Operators) initialization for the A matrix, which encodes a mathematical framework for optimally compressing continuous signals into a finite-dimensional state. HiPPO initialization enables the state to maintain a compressed history of the entire input sequence, with more recent inputs represented at higher fidelity — a property that is particularly well-suited to language and time-series modeling.

S4 achieved state-of-the-art results on the Long Range Arena benchmark, which tests sequence models on tasks requiring reasoning over sequences of 1,024 to 16,384 tokens. Its performance on the Path-X task (classifying 16K-length sequences) was a breakthrough that dense attention models could not match.

3.2 How SSMs Work: Dual View

The elegance of SSMs lies in their dual computational modes:

**Recurrent mode.** At inference time, the model maintains a fixed-size hidden state h and updates it one token at a time: h_t = Ā h_{t-1} + B̄ x_t, where Ā and B̄ are the discretized state and input matrices. This gives O(1) computation per step and O(1) memory regardless of sequence length — no KV cache is needed. The state vector (typically 16 to 256 dimensions per channel) serves as a compressed representation of the entire history.

**Convolutional mode.** During training, the same linear recurrence can be unrolled into a global convolution over the sequence, enabling efficient parallel computation on GPUs. The convolution kernel is derived from the SSM parameters (A, B, C, Δ) and can be computed using FFT in O(n log n) time. This parallel training mode is critical because recurrent computation, while memory-efficient, is inherently sequential and underutilizes modern GPU architectures.

3.3 From S4 to Language Modeling: S5, H3, and Hyena

The path from S4 to practical language models involved several intermediate architectures. **S5** (Simplified State Space Layers) replaced S4's complex diagonal-plus-low-rank parameterization with a simpler diagonal structure and parallel scan computation, improving both speed and stability. **H3** (Hungry Hungry Hippos) combined two SSM layers with multiplicative gating to approximate attention-like recall capabilities, demonstrating for the first time that SSMs could approach transformer quality on language modeling tasks at the 125M to 1.3B parameter scale.

**Hyena** generalized the approach by replacing attention with long convolutions parameterized by implicit neural networks, achieving sub-quadratic complexity while matching transformer perplexity at moderate scales. StripedHyena extended this with alternating Hyena and attention layers to better handle tasks requiring precise token recall.

3.4 Mamba: Selective State Spaces

Mamba, introduced by Gu and Dao in December 2023, represents the most significant advance in SSMs for language modeling. Its key innovation is **selectivity** — making the SSM parameters (B, C, and Δ) functions of the input, rather than fixed learned parameters. This input-dependent parameterization allows the model to selectively propagate or forget information based on content, analogous to the gating mechanisms in LSTMs but within the SSM framework.

In a standard (non-selective) SSM, the same dynamics are applied to every input regardless of content. This makes the model a linear time-invariant system, which is mathematically convenient but limited in expressiveness — it cannot, for example, perform content-based copying or selective recall. By making B, C, and Δ input-dependent, Mamba breaks this linearity and gains the ability to perform input-dependent filtering.

However, selectivity introduces a challenge: because the parameters now vary with input, the convolutional mode used for efficient parallel training no longer applies directly. Mamba addresses this with a **hardware-aware parallel scan algorithm** that exploits the GPU memory hierarchy, keeping the scan computation in fast SRAM rather than materializing large intermediate states in HBM. This implementation, combined with kernel fusion, enables Mamba to train 3-5x faster than equivalent transformers of the same FLOP count.

At the 1.4B parameter scale, Mamba matched or exceeded the quality of transformer models (Pythia, GPT-Neo) with the same training compute on standard language modeling benchmarks. It demonstrated linear scaling in sequence length for both training and inference, and achieved 5x higher throughput than a transformer at 1 million token sequences.

3.5 Mamba-2: Structured State Space Duality

Mamba-2, published in mid-2024, revealed a deep theoretical connection between selective SSMs and attention. The **Structured State Space Duality (SSD)** framework shows that selective SSMs with a specific structure (diagonal state matrices with scalar-times-identity structure) are mathematically equivalent to a form of structured masked attention where the mask is determined by the cumulative product of a scalar gate. This duality means SSMs and attention are not fundamentally different paradigms but rather different computational strategies for related operations.

Practically, Mamba-2 leverages this duality to implement the SSM computation as a series of matrix multiplications that map efficiently onto tensor cores, achieving 2-8x speedups over Mamba-1. The SSD layer uses a larger state dimension (typically 64-256 versus 16 in Mamba-1), improving model quality, while the tensor-core-friendly computation keeps wall-clock time competitive. Mamba-2 matches transformer quality more consistently across tasks at the 2.7B parameter scale and simplifies the implementation compared to the custom CUDA kernels required by Mamba-1.

3.6 Why SSMs Matter for Inference

The inference advantages of SSMs are substantial:

  ----------------------------- ---------------------------------- ----------------------------------
  **Property**                  **Transformer**                    **SSM (Mamba)**

  Per-step computation          O(n) — attend to all past tokens   O(1) — fixed state update

  Memory for past context       O(n) — KV cache                   O(1) — fixed state vector

  Generation at 1K tokens       Baseline                           ~5x faster

  Generation at 64K tokens      Very slow (large KV cache)         Same speed as 1K tokens

  Prefill computation           O(n²) attention                    O(n) scan/convolution

  Memory scaling                Linear in context length           Constant
  ----------------------------- ---------------------------------- ----------------------------------

For Mamba-1.4B, the hidden state is approximately 1.5 MB regardless of sequence length, compared to a KV cache that would be approximately 200 MB at 8K tokens for an equivalently sized transformer. This constant memory footprint makes SSMs particularly attractive for deployment on memory-constrained devices and for applications requiring very long contexts.

4\. Linear Attention and Variants

4.1 The Linear Attention Framework

Standard attention computes: Attention(Q, K, V) = softmax(QK^T / √d) V. The softmax creates a non-linear coupling that prevents factoring the computation and forces the O(n²) materialization of the attention matrix. Linear attention removes the softmax and uses kernel functions φ to approximate the attention pattern: LinearAttn(Q, K, V) = φ(Q)(φ(K)^T V). By applying the kernel to K and V first (an n×d times d×n multiplication that produces a d×d matrix, where d is the head dimension), the computation becomes O(n d²) — linear in sequence length.

The original linear attention formulation by Katharopoulos et al. (2020) used φ(x) = elu(x) + 1 as the kernel function. While theoretically appealing, early linear attention models suffered significant quality degradation compared to softmax attention, particularly on tasks requiring precise token retrieval.

4.2 Performer and Random Feature Approximations

The **Performer** (Choromanski et al., 2021) introduced FAVOR+ (Fast Attention Via positive Orthogonal Random features), which uses random feature maps to approximate the softmax kernel. By projecting queries and keys into a random feature space, Performer achieves unbiased estimation of softmax attention with linear complexity. However, the approximation error proved problematic in practice — Performer consistently underperformed standard transformers on language modeling tasks, with perplexity gaps of 1-3 points at the 125M scale that widened at larger scales. The variance of the random feature estimator also introduced training instability.

4.3 cosFormer

**cosFormer** (Qin et al., 2022) replaced the softmax with a cosine-based reweighting scheme that decomposes into a linear attention form while incorporating a position-dependent decay that mimics the local bias of softmax attention. cosFormer achieved closer perplexity to standard transformers than Performer, narrowing the gap to approximately 0.5-1.0 perplexity points on WikiText-103, but still fell short on tasks requiring sharp attention patterns.

4.4 TransNormer and Its Variants

**TransNormer** (Qin et al., 2022) introduced the use of normalization in linear attention, applying a DiagAttention structure with RMSNorm to stabilize the attention computation. **TransNormerLLM** (2023) scaled this approach to large language models, incorporating Lightning Attention for efficient training, positional encoding through exponential decay, and gating mechanisms. TransNormerLLM achieved perplexity competitive with transformers at the 1B to 7B scale on several benchmarks, representing a significant improvement over earlier linear attention approaches.

4.5 Gated Linear Attention

**Gated Linear Attention (GLA)**, introduced by Yang et al. in 2024, adds data-dependent gating to the linear attention framework. Each head maintains a running key-value state S that is updated with a gating matrix G: S_t = G_t ⊙ S_{t-1} + k_t v_t^T. The gate G allows the model to selectively forget information, addressing the unbounded state growth problem of vanilla linear attention. GLA achieves quality competitive with Mamba and approaches transformer quality at the 1.3B to 2.7B scale, while maintaining the recurrent inference properties of linear attention. GLA benefits from a chunk-wise parallel training algorithm that achieves 80-95% of FlashAttention-2's training throughput.

4.6 Lightning Attention

**Lightning Attention** is a family of IO-aware algorithms for training linear attention models efficiently on GPUs, analogous to what FlashAttention does for softmax attention. Lightning Attention-1 used tiling and accumulation to avoid materializing large intermediate states. Lightning Attention-2 (Lin et al., 2024) separated the computation into intra-block (causal, requiring triangular masking) and inter-block (fully computed) components, enabling different optimization strategies for each. This achieves near-hardware-peak throughput for linear attention training, making linear attention models practical to train at scale.

4.7 Limitations of Linear Attention

Despite substantial progress, linear attention models still exhibit measurable quality gaps on certain tasks. Associative recall (retrieving a specific value associated with a key seen earlier in the context) remains challenging because the fixed-size state compresses all past key-value associations. Tasks requiring precise attention to a small number of specific tokens — such as copying, exact retrieval, or multi-hop reasoning — tend to favor softmax attention, which can place arbitrarily sharp probability mass on individual tokens. These limitations have motivated hybrid architectures that combine linear attention or SSMs with selective use of full attention.

5\. Modern RNN Approaches

5.1 RWKV: Reinventing RNNs for the Transformer Era

**RWKV** (Receptance Weighted Key Value), developed by Bo Peng and collaborators, is an architecture that combines the training parallelism of transformers with the constant-memory inference of RNNs. RWKV has evolved through multiple versions, each refining the core mechanism:

**RWKV-4 (2023)** introduced the core WKV (Weighted Key Value) mechanism, which computes a weighted sum of values where weights decay exponentially with distance, modulated by learned per-channel decay rates. The model uses a **token shift** operation — linearly interpolating between the current and previous token representations — to introduce local context before computing the attention-like operation. Crucially, the WKV computation can be expressed both as a parallel scan (for training) and as a simple recurrence (for inference), giving O(n) training complexity and O(1) per-step inference.

**RWKV-5 (Eagle, 2024)** introduced multi-headed matrix-valued states, replacing the scalar decay with learned matrix-valued receptance and decay parameters per head. This increased the model's capacity to maintain and manipulate state, narrowing the quality gap with transformers.

**RWKV-6 (Finch, 2024)** added data-dependent decay rates — the forget gate becomes a function of the current input, similar to Mamba's selectivity. This allows the model to dynamically decide how much past context to retain based on what it is currently processing, significantly improving performance on tasks requiring selective memory.

**RWKV-7 (Goose, 2025)** further refined the architecture with improved state management, in-context learning mechanisms, and an expanded state size. At the 7B parameter scale, RWKV-7 achieves quality competitive with similarly-sized transformers on a range of standard benchmarks including MMLU, HellaSwag, and ARC, while maintaining constant-memory inference.

RWKV models have been trained at scales up to 14B parameters, with the RWKV-6 14B model achieving benchmark performance roughly comparable to Llama 2 13B. The RWKV project is notable for being fully open-source with an active community, and models are available under Apache 2.0 licensing.

5.2 xLSTM: Extending the LSTM

**xLSTM** (Extended Long Short-Term Memory), introduced by Sepp Hochreiter's group in 2024, revisits the LSTM architecture with modern scaling techniques. xLSTM introduces two key variants:

**sLSTM (scalar LSTM)** enhances the traditional LSTM with exponential gating — replacing the sigmoid gates with exponential functions followed by normalization. This provides a much larger dynamic range for the gates, allowing the model to more decisively open or close information flow. sLSTM retains the scalar memory cell of the original LSTM but with improved gradient flow and training stability.

**mLSTM (matrix LSTM)** replaces the scalar memory cell with a matrix-valued memory, storing key-value associations in a d×d matrix that is updated via an outer-product rule. This is closely related to linear attention and provides a higher-capacity memory that can store and retrieve associative information more effectively. The mLSTM update rule is: C_t = f_t C_{t-1} + i_t (v_t k_t^T), where C is the memory matrix, f_t is the forget gate, and i_t is the input gate, both using exponential gating.

xLSTM blocks are arranged in a residual architecture similar to transformers, with alternating sLSTM and mLSTM layers. At the 1.3B parameter scale, xLSTM achieves perplexity competitive with Mamba and approaches transformer quality, with mLSTM blocks contributing most of the quality advantage. However, xLSTM has seen less adoption than Mamba or RWKV in practice, partly due to the complexity of the sLSTM blocks (which require sequential computation for backpropagation, limiting training throughput) and the relative immaturity of the training infrastructure.

5.3 Griffin and Hawk

**Griffin** and **Hawk**, introduced by Google DeepMind in early 2024, are RNN-based architectures designed for efficient language modeling. **Hawk** uses a simple single-head gated linear recurrence: h_t = a_t ⊙ h_{t-1} + (1 - a_t) ⊙ x_t, where a_t is a learned gating vector. This is essentially a gated exponential moving average with input-dependent gates.

**Griffin** extends Hawk by adding a multi-head recurrent block with a gated linear recurrence alongside local (sliding window) attention. The recurrence provides long-range context propagation while the local attention handles precise short-range interactions. At the 7B parameter scale trained on 300B tokens, Griffin matched the quality of a comparably sized transformer on most benchmarks while requiring significantly less memory during inference. On long-context evaluations, Griffin outperformed transformers with the same training context length, suggesting that the recurrent state generalizes to longer sequences more naturally than positional encodings.

5.4 HGRN and HGRN2

**HGRN** (Hierarchical Gated Recurrent Network) applies gated linear recurrence at multiple timescales within each layer, using different forget-gate biases for different portions of the hidden state to create a hierarchy from fast-changing to slowly-changing dimensions. **HGRN2** (Qin et al., 2024) extends this with outer-product-based state expansion, similar to mLSTM, achieving quality competitive with Mamba at the 1.3B to 3B scale while maintaining efficient recurrent inference. HGRN2 is notable for achieving strong results with a relatively simple architecture and clean implementation.

6\. Retention and Hybrid Mechanisms

6.1 RetNet: Retentive Network

**RetNet** (Sun et al., 2023, from Microsoft Research) introduces a **retention** mechanism designed to unify training parallelism, efficient inference, and good quality. The retention mechanism uses an exponentially decaying attention pattern: Retention(Q, K, V)_n = Σ_{m≤n} γ^{n-m} (Q_n K_m^T) V_m, where γ is a learned decay rate per head.

RetNet supports three computation modes. **Parallel mode** computes retention as a matrix operation for training, similar to standard attention but with the decay mask applied element-wise. **Recurrent mode** computes retention as a linear RNN for O(1) per-step inference. **Chunk-wise mode** divides the sequence into chunks, using parallel computation within chunks and recurrent propagation between chunks, providing a middle ground for throughput-oriented inference.

Multi-scale retention uses different decay rates γ across heads, with some heads having rapid decay (focusing on local context) and others having slow decay (maintaining long-range information). At the 6.7B parameter scale, RetNet was reported to achieve quality comparable to transformers with 8.4x faster decoding and 3.4x lower memory usage during inference.

6.2 Hybrid Architectures

The most practically successful post-transformer models have been hybrids that combine recurrent or SSM layers with selective use of attention. These architectures are motivated by the observation that attention excels at precise retrieval and associative tasks while recurrent/SSM layers excel at long-range propagation and efficient inference.

**Jamba (AI21 Labs, March 2024)** was the first production-scale hybrid SSM-attention model. Jamba interleaves Mamba layers with transformer attention layers at a ratio of approximately 7:1 (7 Mamba layers per 1 attention layer), and incorporates mixture-of-experts on some layers. The resulting 52B parameter model (12B active) supports a 256K token context window while fitting on a single 80GB GPU for inference — a significant practical advantage. On benchmarks, Jamba matched Mixtral 8x7B and Llama 2 70B quality with substantially lower inference cost. Jamba 1.5, released later in 2024, scaled to larger sizes and refined the layer ratio, confirming the viability of the hybrid approach.

**Zamba (Zyphra, 2024)** takes a different hybrid approach: it uses shared attention layers interleaved among Mamba blocks, where the attention layers share weights across their occurrences. This weight sharing reduces the total parameter count while maintaining the benefits of occasional full attention. Zamba demonstrated that even a small number of shared attention layers (1-2 sets of shared weights used at multiple positions) significantly improves recall and in-context learning compared to pure Mamba models.

**Samba (Microsoft, 2024)** combines Mamba layers with sliding window attention (rather than full attention), using the SSM for long-range context and the local attention for precise short-range interactions. This design avoids the quadratic cost of full attention entirely while still benefiting from the exact token-level matching that attention provides within a local window.

**RecurrentGemma (Google DeepMind, 2024)** builds on the Griffin architecture, combining gated linear recurrences with local sliding window attention. The 2B and 9B parameter models were trained on substantially more data than comparable models and achieved quality competitive with Gemma 2B and other small transformers. RecurrentGemma demonstrated that with sufficient training data, hybrid recurrent architectures could match transformer quality at small to medium scales.

**StripedHyena** uses alternating Hyena (long convolution) and attention layers, with the Hyena layers handling long-range dependencies and attention layers providing precise short-range interactions. StripedHyena-7B was one of the first open models to demonstrate competitive quality with a sub-quadratic architecture at the 7B scale.

7\. Test-Time Computation Models

7.1 Adaptive Compute Per Token

A separate but related line of research addresses the fixed-computation limitation of transformers by allowing the model to allocate different amounts of computation to different tokens.

**Mixture of Depths (Raposo et al., Google DeepMind, 2024)** allows tokens to skip certain transformer layers via a learned routing mechanism. Each layer has a capacity budget (e.g., 50% of tokens), and a lightweight router decides which tokens are processed and which bypass the layer via a residual connection. This reduces the average FLOPS per token while maintaining quality on standard benchmarks, as many tokens (articles, common words, punctuation) require less processing than tokens at decision points or in complex reasoning chains. Training uses a top-k routing strategy similar to MoE but applied to depth rather than width.

7.2 Early Exit Strategies

Early exit architectures add prediction heads at intermediate layers, allowing confident predictions to exit the network before reaching the final layer. For language modeling, this means easy-to-predict tokens (which are the majority during generation) can be produced with a fraction of the full model's computation. CALM (Confident Adaptive Language Modeling) demonstrated 2-3x speedups on generation tasks with minimal quality loss, using a learned confidence threshold to decide when to exit. The challenge is calibrating the exit criterion — exiting too aggressively harms quality on difficult tokens, while conservative exit provides minimal speedup.

7.3 Think Tokens and Pause Tokens

**Think tokens** (or **pause tokens**) insert special tokens into the sequence that allow the model to perform additional computation steps without producing output. By generating a series of pause tokens before the actual answer, the model effectively gets more "thinking time" — more forward passes to refine its internal representations. Goyal et al. (2024) showed that training transformers with pause tokens improved performance on reasoning and question-answering tasks, with the benefit scaling with the number of pause tokens. This approach is orthogonal to architectural changes and can be applied to any autoregressive model, though it increases generation latency proportionally to the number of inserted tokens.

8\. Comparison and Benchmarks

8.1 Quality Comparison

  --------------------------------- -------------------- ---------------------- --------------------- ------------------- ---------------------
  **Architecture**                  **Scale Tested**     **MMLU (approx)**      **HellaSwag (approx)**  **Training Data**   **Quality vs Transformer**

  Mamba                             1.4B--2.8B           ~26 (1.4B)             ~60 (1.4B)            300B tokens         Comparable at ≤3B

  Mamba-2                           2.7B                 ~28                    ~64                   300B tokens         ≈ Transformer

  RWKV-6 (Finch)                    1.6B--14B            ~25 (1.6B)             ~59 (1.6B)            1T+ tokens          ~95% of Transformer

  RWKV-7 (Goose)                    Up to 7B             ~32 (7B)               ~72 (7B)              1T+ tokens          ≈ Transformer

  Griffin                           7B                   ~31                    ~71                   300B tokens         ≈ Transformer

  xLSTM                             1.3B                 ~26                    ~59                   300B tokens         Comparable at ≤1.3B

  RetNet                            6.7B                 ~29                    ~68                   Undisclosed         Slightly below

  GLA                               1.3B--2.7B           ~26                    ~60                   100B tokens         ≈ Mamba

  Jamba                             52B (12B active)     ~67                    ~87                   Undisclosed         ≈ Mixtral 8x7B

  Jamba 1.5                         94B (12B active)     ~70                    ~88                   Undisclosed         Above Mixtral 8x7B

  RecurrentGemma                    2B--9B               ~30 (2B)               ~71 (2B)              2T tokens           ≈ Gemma 2B
  --------------------------------- -------------------- ---------------------- --------------------- ------------------- ---------------------

Note: Benchmark numbers are approximate and sourced from respective papers and community evaluations. Direct comparisons are complicated by differences in training data, tokenization, and evaluation methodology.

8.2 Inference Speed and Memory

  --------------------------------- ---------------------- ---------------------- ---------------------- ----------------------
  **Architecture**                  **Prefill Speed**      **Decode Speed**       **Memory at 8K ctx**   **Memory at 128K ctx**

  Transformer (FlashAttn-2)         Baseline               Baseline               Baseline               ~16x baseline

  Mamba                             ~1.5x faster           ~5x faster             ~0.3x                  ~0.3x

  RWKV-6                            ~1.3x faster           ~4x faster             ~0.3x                  ~0.3x

  Griffin                            ~1.2x faster           ~4x faster             ~0.5x (has local attn) ~0.5x

  RetNet                            ~1.5x faster           ~8x faster             ~0.3x                  ~0.3x

  Jamba (hybrid)                    ~1.2x faster           ~3x faster             ~0.5x                  ~5x (attention layers)

  GLA                               ~1.2x faster           ~5x faster             ~0.3x                  ~0.3x
  --------------------------------- ---------------------- ---------------------- ---------------------- ----------------------

The decode speed advantages grow with sequence length. At very long contexts (100K+ tokens), pure SSM/RNN models can be 10-20x faster than transformers because their per-step cost is independent of context length, while the transformer's per-step cost scales linearly.

8.3 Scaling Behavior

A critical question is whether sub-quadratic architectures follow the same scaling laws as transformers. Evidence to date suggests that:

SSMs and linear attention models benefit from increased scale, but the scaling exponent may differ from transformers. Mamba-1 showed consistent improvement with scale from 130M to 2.8B parameters, with a scaling curve roughly parallel to but slightly below the transformer curve. Mamba-2 and RWKV-7 have narrowed this gap, with some evaluations showing equivalent scaling at the 1-7B range.

The largest pure SSM/RNN models trained to date remain relatively small compared to frontier transformers. No pure SSM model has been publicly trained at the 70B+ scale with comparable data, making it difficult to confirm whether the quality parity observed at smaller scales persists. The hybrid models (Jamba at 52B active parameters including attention layers) provide the strongest evidence that SSM-based architectures can scale to production quality.

8.4 Long-Context Performance

On tasks requiring retrieval from long contexts (needle-in-a-haystack, multi-document QA, long-range dependency resolution), the picture is nuanced:

Pure SSMs struggle with precise retrieval from very long contexts. The fixed-size state must compress the entire history, and information can be lost — particularly when the model must retrieve a specific detail stored many thousands of tokens earlier without being "primed" by related context. Mamba models show degraded needle-in-a-haystack performance beyond approximately 16K tokens compared to transformers with full attention.

Hybrid models largely solve this problem. By including periodic attention layers, Jamba and similar hybrids achieve near-perfect retrieval performance while maintaining most of the efficiency benefits of SSMs for the majority of layers.

RWKV-6 and RWKV-7 demonstrate improved recall compared to Mamba, attributed to the data-dependent decay mechanism, but still fall short of full attention on the most demanding retrieval tasks.

8.5 The Hybrid Tax

Hybrid models aim to combine the quality of attention with the efficiency of recurrence, but this comes at a cost. The attention layers, even if constituting only 10-15% of total layers, dominate the KV cache memory usage and create a bottleneck during long-context inference. Jamba's KV cache, while much smaller than a pure transformer's, still grows linearly with sequence length due to its attention layers. The "hybrid tax" also manifests in engineering complexity — these models require two different computational paradigms, two sets of optimized kernels, and careful balancing of layer ratios during architecture design.

Despite this tax, hybrid architectures currently offer the best practical trade-off for production deployment, combining near-transformer quality with significantly reduced inference cost.

9\. Practical Considerations

9.1 Training Infrastructure

Transformers benefit from a decade of optimized training infrastructure. Frameworks like Megatron-LM, DeepSpeed, and FSDP are heavily optimized for attention computation. SSM and RNN architectures require custom kernels for efficient parallel training (scan operations, specialized convolutions), and these are less mature. Mamba's custom CUDA kernels and the Triton-based implementations for GLA and RWKV have reached reasonable maturity, but the ecosystem is smaller and less tested at frontier scale.

Training SSM models at the largest scales (tens of billions of parameters across thousands of GPUs) has not been publicly demonstrated to the same extent as transformers. The parallel scan operation requires careful implementation of sequence parallelism, and the interaction with tensor parallelism and pipeline parallelism is less well understood.

9.2 Hardware Utilization

Modern GPUs are optimized for large matrix multiplications, which is the dominant operation in transformer attention and FFN layers. The recurrent operations in SSMs and RNNs, while theoretically more efficient, can underutilize GPU resources if not carefully implemented. Mamba-2's SSD framework addresses this by reformulating the SSM computation as structured matrix multiplications that map onto tensor cores, achieving high hardware utilization. Nevertheless, transformers generally achieve higher FLOP utilization during training due to the maturity of optimized attention kernels.

During inference, the situation reverses: the small matrix operations in transformer autoregressive decoding (batch_size × d matrix-vector products for attention) severely underutilize GPU compute, while SSM state updates, though also small, are fixed-cost and avoid the memory-bandwidth-bound KV cache lookups.

9.3 Ecosystem and Tooling

The transformer ecosystem includes mature tooling for quantization (GPTQ, AWQ, GGUF formats), serving (vLLM, TGI, TensorRT-LLM), fine-tuning (LoRA, QLoRA), and evaluation. SSM and RNN models have partial support in these tools — Mamba is supported in several serving frameworks, and RWKV has its own inference stack — but the coverage is less comprehensive. Jamba and other hybrid models require special handling in serving systems that must manage both attention KV caches and SSM states.

Quantization of SSM models presents both opportunities and challenges. The recurrent state is sensitive to quantization error accumulation over long sequences, requiring careful handling of state precision even when weights are quantized. RWKV models have been successfully quantized to 4-bit and lower with GGUF-compatible formats, running efficiently on consumer hardware through the rwkv.cpp ecosystem.

9.4 Task-Architecture Fit

Different architectures show strengths on different task types:

  --------------------------------- --------------------------------- --------------------------------- ---------------------------------
  **Task Type**                     **Best Architecture**             **Why**                           **Alternative**

  Long-form generation              SSM/RNN                           Constant memory, fast decode      Hybrid

  Retrieval-heavy tasks             Transformer or Hybrid             Precise attention needed          RWKV-7 (improving)

  Real-time streaming               SSM/RNN                           O(1) per step                     Local attention

  In-context learning               Transformer                      Sharp attention patterns           Hybrid (closing gap)

  Code generation                   Transformer or Hybrid             Structure-sensitive tasks          Mamba-2 (improving)

  Long-document summarization       SSM or Hybrid                     Long-range context propagation    Extended-context Transformer

  Constrained deployment            SSM/RNN                           Small memory footprint            Quantized Transformer
  --------------------------------- --------------------------------- --------------------------------- ---------------------------------

10\. Future Outlook

10.1 Replacement or Augmentation?

The evidence to date suggests that transformers are more likely to be augmented than fully replaced. Hybrid architectures that use attention selectively — for the tasks and layers where it provides the most benefit — appear to be the dominant trajectory. The question is shifting from "SSMs vs. transformers" to "what is the optimal ratio and placement of attention layers within a predominantly recurrent architecture."

10.2 The Convergence Hypothesis

A striking observation is that many of these architectures are converging mathematically. Mamba-2's SSD framework shows that selective SSMs are equivalent to structured attention. GLA, RetNet, and RWKV can all be viewed as different parameterizations of gated linear recurrences with matrix-valued states. xLSTM's mLSTM is closely related to linear attention. This convergence suggests that the field may be discovering a common underlying computational primitive — a gated linear recurrence with input-dependent parameters — that can be implemented in multiple equivalent ways. The differences between architectures may ultimately reduce to implementation details and hardware-specific optimizations rather than fundamental algorithmic distinctions.

10.3 Scaling to Frontier Size

The most important open question is whether sub-quadratic architectures (pure or hybrid) can match transformer quality at the frontier scale of 100B+ active parameters trained on 10T+ tokens. There are theoretical reasons for both optimism (the architectural principles are sound and quality parity at smaller scales is established) and caution (subtle quality gaps may widen with scale, and the training infrastructure has not been validated at this scale). AI21's Jamba and Zyphra's Zamba represent the largest public efforts, but neither approaches the scale of GPT-4 or Claude. A definitive 100B+ SSM or hybrid model with frontier-quality evaluations would significantly accelerate adoption.

10.4 Open Research Questions

Several fundamental questions remain unresolved:

**State capacity limits.** What is the minimum state size needed to match transformer quality at a given model scale? The relationship between state dimension, model dimension, and effective context capacity is not well characterized theoretically.

**Optimal hybrid ratios.** What fraction of layers should be attention vs. recurrence? Does this ratio change with model scale, and should it vary by position within the model (e.g., more attention in later layers)?

**Training stability.** Recurrent architectures with multiplicative state updates can exhibit gradient instability at scale. Exponential gating (as in xLSTM) and careful normalization help, but robust training recipes for 100B+ scale remain to be established.

**Hardware co-design.** Current GPUs are optimized for the dense matrix multiplications of attention. Custom hardware designed for scan operations and recurrent computation (such as Groq's LPU or specialized FPGA implementations) could dramatically shift the efficiency calculus in favor of recurrent architectures.

**Distillation and conversion.** Can a pretrained transformer be distilled into an SSM or hybrid architecture without full retraining? Early results from approaches like Mamba distillation and the Linearizing Transformers framework suggest this is partially possible but with quality loss. Efficient architecture conversion could allow leveraging existing transformer training investments.

11\. Conclusion

The landscape of post-transformer language models has evolved rapidly from theoretical curiosities to production-viable architectures. State space models, modern RNNs, linear attention variants, and hybrid architectures each address different aspects of the transformer's limitations — quadratic attention cost, KV cache growth, and fixed per-token computation. Mamba and its successors have demonstrated that linear-time sequence models can approach transformer quality at moderate scales, while hybrid architectures like Jamba have shown that selective use of attention within a predominantly recurrent architecture can deliver the best practical trade-off between quality and efficiency.

The convergence of these approaches — revealed through frameworks like Structured State Space Duality — suggests that the field is moving toward a unified understanding of sequence modeling that transcends the attention-vs-recurrence dichotomy. The coming years will determine whether this understanding can be scaled to frontier model sizes and whether the efficiency advantages of sub-quadratic architectures will reshape the economics of large-scale language model deployment. For now, the transformer remains the default for the largest and most capable models, but its position is no longer uncontested.
