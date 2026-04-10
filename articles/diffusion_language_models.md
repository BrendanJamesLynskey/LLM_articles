# Diffusion Language Models: An Alternative to Autoregressive Generation

*April 2026 • Technical Report*

## 1. Introduction

The dominant paradigm for language modeling is autoregressive: a model predicts the next token given all previous tokens, then samples from that distribution, then repeats. Every modern LLM that has shaped the field — GPT, Claude, Llama, Gemini, Mistral — works this way. The architecture is the transformer decoder, the training objective is next-token prediction, and the inference loop generates one token at a time. This has become so ubiquitous that "language model" and "autoregressive transformer" are often used interchangeably.

But autoregression is not the only way to build a generative language model. In the image generation world, diffusion models displaced GANs as the dominant approach around 2021, and they now power Stable Diffusion, DALL-E, Midjourney, Imagen, and Flux. The success of diffusion in images raised an obvious question: could the same approach work for text? A series of papers from 2022 to 2025 explored this idea, producing models like Diffusion-LM, SEDD, Plaid, AR-Diffusion, and most recently Inception Labs' Mercury — a 7B-parameter diffusion language model that achieves competitive quality with autoregressive models of similar size while generating text 5-10× faster.

This report examines what diffusion language models are, how they differ from autoregressive models, and why they may matter for the future of language model inference even if they never displace the dominant paradigm.

## 2. Why Diffusion for Text?

### 2.1 The Sequential Bottleneck of Autoregression

Autoregressive generation has one fundamental constraint: it is sequential. To generate token N, you must first generate tokens 1 through N-1. This is true at training time (where it can be parallelized via teacher forcing) and at inference time (where it cannot). The decode phase of LLM inference is bottlenecked by the latency of producing one token at a time, not by GPU compute. A modern LLM serving 1,000 tokens of output takes 1,000 forward passes through the model, each of which produces one token.

For users, this manifests as latency. The model takes a long time to produce long responses, and there is no easy way to make it faster without sacrificing quality. Speculative decoding and Medusa-style approaches accelerate this loop by guessing multiple tokens at once and verifying them, but they offer at most 2-3× speedup and add complexity.

A model that could generate text in parallel — producing many tokens simultaneously rather than one at a time — would break this bottleneck. Diffusion models, which generate images by parallel denoising of all pixels at once, are the obvious candidate.

### 2.2 The Promise of Parallelism

A diffusion model generates output by starting from random noise and iteratively refining it toward a clean sample, with each refinement step processing the entire output in parallel. For a 1,024-pixel image, all 1,024 pixels are updated at once in each step, and a typical diffusion process uses 20-100 steps. The total work is fixed (steps × parameters), but the wall-clock latency depends on parallelism, not on output length.

Applying this to text would mean generating an N-token output in K diffusion steps rather than N autoregressive steps, where K is much smaller than N. For N = 1,000 and K = 30, this would be a 33× reduction in the number of sequential operations needed. Even after accounting for the higher per-step cost (each step must process the full output), the wall-clock improvement could be substantial.

The catch is that text is harder to diffuse than images. Pixels are continuous values; tokens are discrete. The mathematical machinery of diffusion (Gaussian noise, score matching, smooth probability distributions) does not directly apply to discrete data. Adapting diffusion to text required several years of research to overcome these issues.

## 3. Continuous and Discrete Diffusion

### 3.1 Continuous Diffusion: Diffusion-LM and Plaid

The first generation of diffusion language models worked by embedding tokens into a continuous space, applying continuous Gaussian diffusion in the embedding space, and then mapping back to tokens at the end. This is the approach used by Diffusion-LM (2022), Plaid (2023), and several other early models.

The training procedure looks like:

1. Start with a target text and embed each token into a vector
2. Add Gaussian noise to the embeddings, with noise scale determined by a timestep t
3. Train a neural network to predict the original embeddings from the noisy embeddings, conditioned on t
4. At inference, start with random noise and iteratively denoise it using the trained network
5. After the final denoising step, snap the resulting embeddings to the nearest token in the vocabulary

This worked but had quality problems. The discrete-to-continuous embedding step was lossy, and the continuous denoising process did not always produce embeddings close to any valid token. The "snap to nearest" step often produced incoherent text. Plaid and subsequent works improved on this with better embedding spaces and self-conditioning, but the quality gap with autoregressive models remained large.

### 3.2 Discrete Diffusion: D3PM, SEDD

A different line of work tackled diffusion directly in the discrete token space. Instead of adding Gaussian noise to embeddings, these models add discrete noise — either by replacing tokens with a special [MASK] token or by transitioning to other tokens according to a stochastic matrix. Discrete Denoising Diffusion Probabilistic Models (D3PM, 2021) introduced this framework, and SEDD (Score Entropy Discrete Diffusion, 2024) refined it with a new training objective that produced significantly better results.

The forward (corruption) process looks like:

1. Start with a clean sequence of tokens
2. At each timestep, replace some tokens with [MASK] (or transition them to other tokens) with increasing probability
3. After T steps, the sequence is fully masked (or random)

The reverse (denoising) process learns to predict the original tokens from a partially-masked sequence. Training this is closely related to BERT-style masked language modeling, but with a richer set of masking ratios and a different loss function.

SEDD demonstrated that discrete diffusion could approach autoregressive quality on standard benchmarks like LAMBADA and Wikitext, while supporting parallel generation. It was the first diffusion language model whose quality was competitive enough to take seriously.

### 3.3 Mercury and the Inception Labs Approach

Mercury, released by Inception Labs in early 2025, is the first commercially-significant diffusion language model. It is a 7B-parameter model trained on roughly the same data as Llama 7B, using a discrete diffusion objective with several innovations over SEDD. The published benchmarks show Mercury matching or exceeding Llama 2 7B on coding and reasoning tasks while generating output 5-10× faster on equivalent hardware.

Inception Labs' value proposition is exactly the parallelism argument: by generating multiple tokens per forward pass, Mercury achieves higher throughput per GPU. The company offers Mercury through an API at competitive prices and emphasizes use cases (real-time agents, low-latency code completion) where speed matters more than the absolute quality ceiling. As of mid-2025, Mercury is the only diffusion language model in commercial production at significant scale, and its existence has revived interest in diffusion approaches across the research community.

## 4. The Diffusion Language Model Forward Pass

### 4.1 Training

Training a discrete diffusion language model looks like training a BERT-style masked language model with one important twist: the masking ratio varies across training examples. Instead of always masking 15% of tokens, the model sees examples with masking ratios from 0% (no masking) to 100% (everything masked), and it must learn to predict the original tokens at every masking level.

The loss is typically the cross-entropy of the model's predictions over the masked tokens, weighted by a function of the masking ratio. Heavily masked examples teach the model to generate from minimal context; lightly masked examples teach it to fill in details. The final model is a single network that can do both.

### 4.2 Sampling

To generate text with a diffusion language model, you start with a fully-masked sequence of length N and run the model for K denoising steps. At each step:

1. The model takes the current (partially masked) sequence as input
2. It predicts a probability distribution over the vocabulary for each masked position
3. Some subset of the masked positions are "unmasked" by sampling from the predicted distributions
4. The sequence becomes less masked, and the next step starts

The number of positions unmasked per step is a hyperparameter. Aggressive unmasking (many positions per step) gives faster but lower-quality output. Conservative unmasking (few positions per step) gives slower but higher-quality output. Mercury reportedly uses around 30 steps for a 1,000-token output, unmasking roughly 33 tokens per step.

The order in which positions are unmasked also matters. Some implementations unmask in left-to-right order (mimicking autoregression but with multi-token chunks). Others use a confidence-based ordering, unmasking the positions with the most confident predictions first. The latter approach allows the model to commit to easy parts of the output first and use those committed tokens as context for harder positions.

## 5. Quality vs. Speed Tradeoffs

### 5.1 The Quality Gap

Diffusion language models are still slightly behind autoregressive models of equivalent parameter count on most quality benchmarks. SEDD, Plaid, and Mercury all report 1-5% gaps on standard benchmarks like MMLU, HellaSwag, and ARC compared to their autoregressive counterparts. The gap is small enough that for many applications it does not matter, but it is consistent.

Several factors contribute to this gap. Autoregressive models have a strict probabilistic interpretation (the joint probability factorizes cleanly over tokens), while diffusion models have a more approximate one. Autoregressive models can use causal masking and the resulting computational savings to train more efficiently per parameter. The autoregressive token-by-token decoding gives the model the chance to condition each token on all previous tokens in their finalized form, while diffusion models commit to multiple tokens simultaneously and may make mistakes that cannot be revised.

The community is actively working to close this gap. Improvements in training objectives, sampling schedules, and hybrid approaches (combining autoregressive and diffusion components) are all areas of active research.

### 5.2 The Speed Advantage

The flip side is that diffusion language models are dramatically faster for long outputs on parallel hardware. A 1,000-token output with 30 diffusion steps requires 30 forward passes, compared to 1,000 forward passes for an autoregressive model. Even if each diffusion forward pass is more expensive (typically 1.5-2× the cost of an autoregressive forward pass due to the lack of KV caching benefits), the total wall-clock time is much lower.

For batch inference scenarios, the comparison becomes more nuanced. Autoregressive models with continuous batching achieve very high GPU utilization by processing many requests in parallel — at any given time, the model is generating one token for hundreds of requests simultaneously. Diffusion models can also be batched, but the per-step parallelism is already high, so the batch-level parallelism gains are smaller.

The clearest speed advantage of diffusion models is for low-batch, latency-sensitive scenarios. A single-user code completion service, an interactive agent, or a real-time voice assistant all benefit from the reduced sequential bottleneck. For high-throughput batch serving, autoregressive models with continuous batching remain competitive.

## 6. Hybrid Approaches

Several recent papers have explored hybrid models that combine autoregressive and diffusion components.

**AR-Diffusion** uses an autoregressive backbone with diffusion-based refinement of the output. The model generates a draft autoregressively, then refines the draft with a diffusion process that can edit any token. This combines the quality of autoregression with the parallel refinement capability of diffusion.

**Block-diffusion models** generate text in blocks of multiple tokens, with diffusion operating on each block in parallel and autoregression connecting the blocks. A block size of 8-16 tokens balances the parallelism of diffusion with the sequential coherence of autoregression.

**Speculative diffusion** uses a small diffusion model to draft multiple candidate continuations in parallel, then verifies them with a larger autoregressive model. This is structurally similar to speculative decoding but with diffusion as the draft model.

These hybrid approaches are early-stage research, but they suggest that the autoregressive vs. diffusion question may not have a binary answer. The future may involve models that combine both paradigms in ways that neither pure approach can match.

## 7. Comparison with Image Generation Lessons

The image generation field went through a similar paradigm shift around 2020, when diffusion models displaced GANs as the dominant approach. The lessons from that transition are partially applicable to text:

**Quality matters more than speed.** Diffusion image models are slower than GANs but produce dramatically better images. Users prefer the quality. For text, the quality gap is smaller and the speed gap favors diffusion, so the calculus may go in the opposite direction.

**Different modalities have different inductive biases.** Image diffusion benefits from the local correlation structure of pixels (nearby pixels are usually similar). Text does not have this structure — adjacent tokens have varying degrees of correlation depending on their roles. This makes text diffusion harder.

**Tooling matters.** The maturity of the diffusion image ecosystem (Stable Diffusion, ComfyUI, ControlNet, LoRAs) accelerated adoption. The diffusion text ecosystem is in its infancy, with no equivalent of Stable Diffusion's open weights and tooling. This is a barrier to widespread experimentation.

## 8. Use Cases Where Diffusion May Win

Even if diffusion language models don't become the dominant paradigm, they may carve out important niches:

**Long-form generation with hard latency targets**. Generating a 10,000-token document with autoregression takes ~10 seconds per response on a fast GPU. With diffusion, the same length can be produced in 1-2 seconds. For users who want long-form output now, this matters.

**Code completion and editing**. Code is heavily structured, with long-range dependencies that diffusion handles naturally. Diffusion models can edit existing code by treating the existing tokens as a partially-denoised sequence and refining them, which is awkward for autoregressive models.

**Constrained generation**. Diffusion models can incorporate hard constraints (specific tokens at specific positions, regex patterns, structural templates) more naturally than autoregressive models. The denoising process can be conditioned on the constraints throughout, ensuring the output respects them.

**Real-time interactive applications**. Voice assistants, live transcription, and interactive agents benefit from the reduced sequential bottleneck. A voice agent that generates a 100-token response in 200 ms (vs. 2 seconds for autoregression) feels qualitatively more responsive.

## 9. Limitations and Open Questions

Diffusion language models are not a panacea, and several technical challenges remain unresolved:

**KV cache equivalent**. Autoregressive models benefit enormously from KV caching, which lets them reuse computation across tokens. Diffusion models do not have an obvious equivalent — each denoising step processes the full sequence, with little reuse between steps. This makes diffusion inference relatively more expensive per token of useful output, even with the parallelism benefit.

**Variable-length outputs**. Autoregressive models naturally handle variable-length outputs by stopping when they generate an end-of-sequence token. Diffusion models must commit to an output length up front, which is awkward when the right length is not known. Several workarounds exist (generating to a maximum length and truncating, or using a variable-length variant of diffusion) but none are as clean as the autoregressive default.

**Streaming**. Autoregressive generation streams naturally — each token can be displayed as it is produced. Diffusion generation produces all tokens at once, after the full denoising process. Streaming-friendly diffusion variants exist but sacrifice the parallelism that makes diffusion attractive in the first place.

**Quality at scale**. Most current diffusion language models are at the 1B-10B parameter scale. How they will behave at the 70B or 400B scale is unknown. Whether the quality gap with autoregressive models grows, shrinks, or stays constant at larger scales is an open empirical question.

## 10. Conclusion

Diffusion language models are an alternative paradigm to autoregressive transformers, with different tradeoffs and a different set of strengths. They are not yet as good as autoregressive models on quality benchmarks, but they are dramatically faster for long outputs in low-batch scenarios. They are unlikely to displace autoregressive models as the dominant paradigm, but they may carve out important niches in latency-sensitive applications where the quality gap is acceptable.

The bigger lesson is that the autoregressive transformer is not the only way to build a useful language model. After several years where every new release was "another transformer trained on more data," it is interesting to see genuine architectural diversification reappearing. State space models (Mamba), diffusion language models, and other alternatives are pushing back against the architectural monoculture, and the competition is producing both better autoregressive models and better alternatives. Whether or not diffusion ever wins outright, the existence of credible alternatives makes the field healthier and more open to future innovations.

For practitioners, the practical takeaway is to keep an eye on the diffusion approach for applications where latency matters most. Mercury is available today, and it is good enough to use in production for many tasks. The performance characteristics — much faster but slightly lower quality — are unusual enough that they deserve a place in the toolbox alongside conventional autoregressive options.
