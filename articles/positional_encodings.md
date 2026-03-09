**Positional Encodings for Large Language Models**

From Sinusoidal Embeddings to RoPE, ALiBi, and YaRN

March 2026 • Technical Report

Table of Contents

1\. Introduction

Transformers are fundamentally permutation-invariant: without positional information, a transformer treats the sentence "the cat sat on the mat" identically to "mat the on sat cat the." Every modern LLM must inject positional information so that the model can distinguish token order, learn local syntactic patterns, and attend to the right parts of long contexts. The design of positional encodings has evolved from an architectural afterthought into one of the most consequential decisions in transformer design, directly determining the model's maximum context length, its ability to generalise to sequences longer than those seen during training, and the efficiency of KV cache reuse during inference.

This report traces the evolution of positional encoding from the original sinusoidal scheme through the two dominant modern approaches, Rotary Position Embeddings (RoPE) and Attention with Linear Biases (ALiBi), and covers the family of context-extension methods (Position Interpolation, NTK-aware scaling, and YaRN) that have enabled trained models to be extended to context lengths far beyond their original training horizon.

2\. Absolute Positional Encodings

2.1 Sinusoidal Encodings

The original Transformer (Vaswani et al., 2017) added fixed sinusoidal signals to the token embeddings at the input layer. Each position receives a unique vector defined by sine and cosine functions at geometrically spaced frequencies. Pairs of dimensions oscillate at different rates, providing a unique fingerprint for each position. The lowest-frequency dimensions have wavelengths that span thousands of positions, while the highest-frequency dimensions cycle every few tokens.

2.2 Learned Absolute Encodings

BERT and early GPT models replaced the fixed sinusoidal functions with learned position embeddings: a lookup table mapping each position index to a trainable vector. This is simple and effective but introduces a hard maximum context length (the number of rows in the embedding table) and provides no inductive bias for generalisation beyond that length.

2.3 Limitations

Both sinusoidal and learned absolute encodings share a fundamental problem: they encode absolute position rather than relative distance. The model must learn from data that "position 5" and "position 105" encode the same relative relationship (100 tokens apart) as "position 200" and "position 300." More critically, absolute encodings offer no principled mechanism for handling sequences longer than those seen during training. Empirical work has shown that transformers with absolute encodings exhibit catastrophic performance degradation when evaluated on sequences even slightly longer than the training length, because the model encounters position indices for which it has no learned representation or for which the learned features are not calibrated.

3\. Rotary Position Embeddings (RoPE)

3.1 Core Mechanism

Rotary Position Embeddings, proposed by Su et al. in 2021, encode positional information by rotating the query and key vectors before computing attention scores. The key idea is that the dot product between a rotated query at position m and a rotated key at position n depends only on the relative distance m minus n, not on the absolute positions. This provides relative positional encoding without additional parameters and without modifying the attention score computation beyond the rotation.

Concretely, RoPE divides the query and key vectors into pairs of dimensions and applies a 2D rotation to each pair. The rotation angle for the i-th pair at position m is m times theta_i, where theta_i equals base raised to the power of negative 2i over d (with d being the head dimension and base typically 10,000). The lower-indexed pairs rotate quickly (high frequency, short wavelength) and encode fine-grained local position, while higher-indexed pairs rotate slowly (low frequency, long wavelength) and encode broad positional context.

3.2 Properties

RoPE has several attractive properties that have made it the default positional encoding for most modern LLMs including Llama 2/3, Mistral, Gemma, and Qwen. It is parameter-free: no additional weights are needed beyond the base frequency. It is relative by construction: the attention score between two tokens depends only on their distance, not their absolute positions. It is streaming-friendly: the rotation is applied once when writing to the KV cache and does not need to be recomputed during decoding. And it composes naturally with the existing attention mechanism, adding no overhead beyond two rotation operations per head per token.

3.3 Limitations

Despite its elegance, RoPE has practical limitations. The rotation angles are defined for all positions on the real number line, but the model is only trained on positions up to its maximum training context length. Positions beyond this length are out-of-distribution: the rotation angles visit regions of the sine-cosine space that the model has never seen, leading to performance degradation. Additionally, in mixed-precision inference (FP16 or BF16), very large position indices produce rotation angles where the sine and cosine values differ by less than the precision of the floating-point format, effectively collapsing distinct positions to the same encoding. These limitations motivate the context-extension methods discussed in Section 5.

4\. Attention with Linear Biases (ALiBi)

4.1 Mechanism

ALiBi, proposed by Press, Smith, and Lewis in 2022, takes a fundamentally different approach. Rather than modifying token embeddings or applying rotations, ALiBi adds a static, position-dependent bias directly to the attention scores. The bias for a query at position i and a key at position j is negative m times the absolute value of i minus j, where m is a head-specific slope that is fixed (not learned) and varies geometrically across heads. This means that distant tokens receive increasingly negative attention biases, creating an inductive recency bias: the model naturally attends more strongly to nearby tokens.

4.2 Properties and Trade-offs

ALiBi's simplicity is its primary advantage. The bias is added after computing the raw attention scores and before softmax, requiring no modification to the query or key projections and no additional parameters. The linear penalty is trivial to compute and adds negligible overhead. ALiBi was designed specifically for length extrapolation and demonstrated better zero-shot performance on sequences longer than the training length compared to sinusoidal encodings and learned absolute encodings.

However, ALiBi has not achieved the same widespread adoption as RoPE. The linear penalty imposes a strong recency bias that may be too aggressive for tasks requiring attention to distant tokens. Empirical comparisons have found that ALiBi can struggle with long-range retrieval tasks where the model must attend precisely to information in the distant past. Models trained with ALiBi are less common in the current generation of frontier LLMs, with RoPE being the dominant choice.

5\. Context Window Extension

5.1 The Extension Problem

A model trained with RoPE on sequences of length L encounters out-of-distribution (OOD) rotation angles when processing sequences longer than L. Naively extrapolating beyond L leads to rapid quality degradation. The challenge is to extend the effective context window to a target length L-prime (where L-prime is much larger than L) without full retraining, ideally with minimal or no fine-tuning.

5.2 Position Interpolation

The simplest approach, proposed by Chen et al. in 2023, is Position Interpolation (PI). Instead of extrapolating to positions beyond L, PI compresses all positions into the range zero to L by dividing each position index by the scaling factor s equals L-prime over L. This ensures that the rotation angles remain within the range seen during training. The trade-off is that the effective resolution between adjacent positions is reduced: the model must distinguish positions that are now s times closer together in the embedding space. PI requires a modest amount of fine-tuning (typically a few hundred steps) to recover quality and has been shown to work reliably up to scaling factors of about 8 before degradation becomes noticeable.

5.3 NTK-Aware Scaling

Position Interpolation scales all frequency components uniformly, which over-compresses the high-frequency dimensions that encode fine-grained local position. NTK-aware scaling, first proposed as a community contribution on Reddit, addresses this by increasing the base frequency instead of scaling position indices. Specifically, it replaces the base with a scaled base equal to the original base times s raised to the power d over (d minus 2). This spreads the interpolation pressure across dimensions: low-frequency dimensions (which encode global position and are more OOD) receive more interpolation, while high-frequency dimensions (which encode local position and are less OOD) receive less. NTK-aware scaling requires no fine-tuning and provides reasonable quality for moderate extension factors.

5.4 Dynamic NTK Scaling

Dynamic NTK scaling applies NTK-aware scaling only when the input sequence exceeds the training length, and adjusts the scaling factor dynamically based on the current sequence length. This means that the model behaves identically to the original for sequences within the training length and gracefully extends as sequences grow longer. It is particularly useful for inference-time extension where the serving system does not know in advance how long each request will be.

5.5 YaRN

YaRN (Yet Another RoPE Extension), published at ICLR 2024 by Peng et al., combines NTK-aware scaling with two additional refinements. First, it uses an NTK-by-parts strategy that classifies each dimension into three categories based on the ratio of its wavelength to the training length: dimensions with wavelengths shorter than the training length are left unscaled (they are already well-trained), dimensions with wavelengths much longer than the training length are fully interpolated, and intermediate dimensions receive partial interpolation. Second, YaRN introduces an attention temperature factor that adjusts the softmax temperature to preserve the sharpness of attention distributions after scaling.

YaRN achieved state-of-the-art context extension performance, extending Llama 2 models to 128K tokens after fine-tuning on less than 0.1 percent of the original pre-training data. Combined with Dynamic Scaling, it enables inference-time extension to more than twice the fine-tuned context length.

5.6 Llama 3 and Modern Practice

Llama 3 uses RoPE with a specific frequency-aware scaling scheme that is conceptually similar to YaRN's NTK-by-parts approach. The scaling factors are determined per-dimension based on whether each dimension's wavelength falls within, near, or beyond the original training length. This approach, combined with continued pre-training on long-context data, enabled Llama 3 to support 128K-token contexts. The general trend in the field has been to combine careful positional encoding scaling with targeted long-context fine-tuning, rather than relying on any single extension technique in isolation.

6\. Comparison of Approaches

  ---------------------- ----------------------- ------------------------ ----------------------- -------------------------
  **Method**             **Type**                **Learned Parameters**   **Extrapolation**       **Adopted By**

  Sinusoidal             Absolute                None                     Poor                    Original Transformer

  Learned Absolute       Absolute                O(L x d)                 None (hard cutoff)      BERT, GPT-2

  RoPE                   Relative (rotation)     None                     Moderate (with ext.)    Llama, Mistral, Gemma

  ALiBi                  Relative (bias)         None                     Good (zero-shot)        MPT, BLOOM

  RoPE + PI              Relative (rotation)     None                     Good (with fine-tune)   CodeLlama

  RoPE + YaRN            Relative (rotation)     None                     Excellent               Llama 3 (similar)
  ---------------------- ----------------------- ------------------------ ----------------------- -------------------------

7\. Practical Considerations

For practitioners, the choice of positional encoding is usually made at model design time and has long-lasting consequences. RoPE is the dominant default for new models in 2025 and 2026, owing to its simplicity, parameter-free nature, and well-understood extension methods. ALiBi remains a viable alternative for workloads that prioritise zero-shot length generalisation and where the recency bias is acceptable.

When extending an existing RoPE-based model to a longer context, YaRN-style scaling with a small amount of long-context fine-tuning is the current best practice. Dynamic NTK scaling is a reasonable zero-fine-tuning fallback for moderate extension factors (2 to 4 times). For very large extensions (8 times or more), continued pre-training on long-context data is generally necessary.

The choice of RoPE base frequency affects the trade-off between local resolution and long-range discrimination. Higher base values spread the frequency spectrum and improve long-range differentiation but reduce local resolution. Lower base values provide better local discrimination but may struggle with very long contexts.

8\. Future Directions

Several research directions continue to push positional encoding forward. LongRoPE2 (2025) uses evolutionary search to find optimal per-dimension rescaling factors guided by needle-in-a-haystack retrieval tasks, achieving near-lossless extension to 128K tokens. Resonance RoPE addresses the feature gap between trained and extended positions by identifying critical dimensions whose wavelengths align with the training length. Axial and 2D RoPE variants extend the rotary framework to vision transformers and multi-modal models where the input has spatial rather than purely sequential structure. The broader trend is toward increasingly sophisticated, dimension-aware scaling strategies that treat each frequency component of RoPE independently rather than applying uniform transformations.

9\. Conclusion

Positional encoding has evolved from a simple add-on (sinusoidal or learned vectors appended to embeddings) into a carefully engineered component that directly determines a model's context capabilities. RoPE's insight that relative position can be encoded through rotation, combined with the family of frequency-aware extension methods, has enabled the dramatic expansion of context windows from a few thousand tokens to hundreds of thousands and beyond. Understanding the frequency structure of RoPE, the out-of-distribution behaviour at the boundary of the training length, and the mechanisms by which extension methods correct this behaviour is essential for anyone working on long-context LLM deployment.
