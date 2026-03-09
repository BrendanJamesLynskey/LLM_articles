**LLM Training Stages**

Pre-Training, Mid-Training, and Post-Training in Large Language Model Development

March 2026 • Technical Report

Table of Contents

1\. Introduction

Building a large language model is not a single monolithic training run. Modern LLM development follows a multi-stage pipeline, each stage with distinct objectives, data requirements, and computational profiles. Pre-training establishes the model's broad knowledge and linguistic competence by learning from trillions of tokens of text. Mid-training, an increasingly recognized intermediate phase, adapts and extends the pre-trained model through continued training with modified data mixtures, longer context windows, or domain-specific corpora. Post-training transforms the capable but unrefined base model into a useful, safe, and steerable assistant through supervised fine-tuning, preference optimization, and safety training.

This report examines each of these three stages in detail, covering the data pipelines, optimization strategies, infrastructure requirements, and design decisions that define modern LLM training. It also addresses the cross-cutting themes of compute allocation, the interactions between stages, and the evolving balance of effort across the pipeline.

2\. Pre-Training

2.1 Objective and Formulation

Pre-training is the foundational stage of LLM development, consuming the vast majority of total compute. The dominant objective for modern LLMs is causal language modeling (CLM), also known as next-token prediction: given a sequence of tokens, the model learns to predict the probability distribution over the next token. The training loss is the cross-entropy between the model's predicted distribution and the actual next token, averaged over all positions in the training data. This deceptively simple objective, applied at sufficient scale, produces models with broad linguistic knowledge, reasoning ability, and world knowledge.

Encoder-decoder models like T5 and UL2 use a masked span prediction objective, where contiguous spans of the input are replaced with sentinel tokens and the model must reconstruct the missing spans. While encoder-decoder architectures remain relevant for specific tasks, the decoder-only causal language model has become the dominant paradigm for general-purpose LLMs, used by GPT-4, Claude, Llama, Gemini, and most other frontier models. The simplicity of the causal LM objective, its natural fit for generation tasks, and its favorable scaling properties have driven this convergence.

2.2 Data Collection and Curation

Pre-training data quality is one of the most critical determinants of model quality. Modern LLMs are trained on datasets ranging from 1 trillion to over 15 trillion tokens, drawn from diverse sources. The primary sources include web crawls (primarily Common Crawl, which provides petabytes of raw web text), digitized books, academic papers, code repositories (GitHub, GitLab), conversational data (forums, Q&A sites), and curated reference material (Wikipedia, encyclopedias).

Raw web data is extremely noisy, containing spam, boilerplate, duplicate content, low-quality machine-generated text, and harmful material. The data curation pipeline typically involves several stages:

**Language identification and filtering.** Documents are classified by language, and non-target languages are removed or separated. Fasttext-based classifiers are commonly used for this purpose.

**Quality filtering.** Multiple approaches are used to separate high-quality from low-quality text. Perplexity filtering scores documents using a language model trained on a known high-quality corpus (such as Wikipedia) and removes documents with excessively high perplexity, indicating noisy or incoherent text. Classifier-based filtering trains binary classifiers to distinguish high-quality documents (using positive examples from curated sources) from low-quality documents. Heuristic filters remove documents based on rules such as excessive repetition, insufficient word count, high symbol-to-word ratios, or abnormal character distributions. Llama 3's data pipeline used both a fasttext classifier and a RoBERTa-based quality classifier to filter web data, keeping only roughly 15% of the raw Common Crawl content.

**Deduplication.** Training data contains enormous amounts of duplication, which wastes compute, biases the model toward memorizing duplicated content, and increases the risk of training data extraction. Exact deduplication removes byte-identical documents using hash comparisons. Near-duplicate detection, typically using MinHash with Locality-Sensitive Hashing (LSH), identifies and removes documents that are highly similar but not identical. Some pipelines also perform line-level or paragraph-level deduplication to remove boilerplate text (navigation menus, cookie notices, legal disclaimers) that appears across many documents. The Llama 3 pipeline used both URL-level, document-level, and line-level deduplication.

**Content filtering.** Toxic, harmful, or personally identifiable content is identified and removed using classifiers, keyword lists, and regular expressions. This filtering must balance thoroughness against overly aggressive removal that could bias the dataset.

The relative mixture of data sources is itself a critical design decision. Empirical research has consistently shown that upweighting high-quality sources (books, Wikipedia, academic papers, curated code) relative to their natural proportion in web crawls improves model quality. The Llama 3 pre-training mixture allocated roughly 50% to web data, 25% to code, 15% to books and academic papers, and 10% to other curated sources. These ratios are typically determined through ablation studies on smaller models.

2.3 Tokenizer Training and Vocabulary Design

Before text can be fed to a model, it must be converted into a sequence of integer tokens. Modern LLMs overwhelmingly use subword tokenization algorithms, primarily Byte Pair Encoding (BPE) and its variants. The tokenizer is trained on a representative sample of the training data by iteratively merging the most frequent adjacent byte or character pairs into new tokens, building up a vocabulary of common subwords.

Vocabulary size is a consequential design choice. Larger vocabularies (50,000 to 200,000 tokens) improve compression of common text, reducing sequence length and training cost per document, but increase the size of the embedding and output projection layers. GPT-4 uses a vocabulary of roughly 100,000 tokens. Llama 3 expanded its vocabulary to 128,000 tokens, up from Llama 2's 32,000, significantly improving tokenization efficiency for code and multilingual text. The tokenizer also determines how well the model handles different languages, scripts, and domains: a tokenizer trained primarily on English text will fragment non-Latin scripts into many tokens, degrading performance and increasing cost for those languages.

2.4 Architecture Decisions

The transformer architecture has been nearly universal for large language models since GPT-2, but significant design choices remain within the transformer framework.

**Decoder-only vs. encoder-decoder.** Decoder-only architectures (GPT, Llama, Mistral, Claude) use causal attention masks and generate tokens autoregressively. Encoder-decoder architectures (T5, Flan-UL2) use bidirectional attention in the encoder and causal attention in the decoder. Decoder-only models have dominated at scale due to simpler training, more straightforward scaling, and natural suitability for generation. Virtually all frontier models since 2023 are decoder-only.

**Model size and shape.** The number of parameters is determined by the hidden dimension, number of layers, number of attention heads, and the intermediate dimension of the feed-forward layers. Models range from 1 billion parameters (small, useful for research and edge deployment) to over 1 trillion parameters (frontier models like GPT-4 and Gemini Ultra, which use mixture-of-experts architectures). The ratio of depth (number of layers) to width (hidden dimension) affects training dynamics and the types of computation the model can perform. Deeper models tend to be better at sequential reasoning, while wider models have more capacity per layer.

**Context length.** The maximum context length at pre-training time determines the baseline sequence length the model can handle. Early GPT models used 1,024 or 2,048 tokens. Modern models pre-train with 4,096 to 8,192 tokens and extend to longer contexts during mid-training. Longer pre-training context windows increase memory and compute costs quadratically with standard attention (or linearly with efficient attention variants), so there is a strong incentive to pre-train with moderate context lengths and extend later.

**Positional encoding.** Rotary Position Embeddings (RoPE) have become the standard positional encoding for modern LLMs, replacing learned absolute position embeddings. RoPE encodes relative positions through rotation matrices applied to query and key vectors, and crucially enables extrapolation to longer contexts than those seen during training, which is essential for mid-training context extension.

**Normalization and activation functions.** RMSNorm has largely replaced LayerNorm due to its computational efficiency and comparable performance. Pre-normalization (applying normalization before the attention and feed-forward sublayers rather than after) has become standard as it improves training stability. SwiGLU activations have replaced the standard ReLU or GELU activations in the feed-forward layers of most modern architectures, following findings that gated activations improve quality at equivalent compute.

2.5 Scaling Laws

Scaling laws describe the predictable relationship between model performance (measured by cross-entropy loss) and the key variables of model size (number of parameters N), dataset size (number of tokens D), and compute budget (total FLOPs C). The foundational Kaplan et al. (2020) scaling laws from OpenAI established that loss decreases as a power law in each of these variables, and that larger models are more sample-efficient, suggesting that compute budgets should be allocated primarily to increasing model size rather than dataset size.

The Chinchilla scaling laws (Hoffmann et al., 2022) from DeepMind revised this conclusion significantly. Chinchilla's analysis showed that the Kaplan laws had overestimated the optimal model size for a given compute budget. The compute-optimal relationship is approximately D = 20N, meaning a compute-optimal model should be trained on roughly 20 tokens per parameter. Under these revised laws, a 70-billion-parameter model should be trained on approximately 1.4 trillion tokens, and many existing models were significantly undertrained (GPT-3, for example, trained its 175B parameters on only 300B tokens).

In practice, the Chinchilla-optimal point optimizes for training compute, not inference compute. Since inference costs dominate for widely deployed models, many recent models are deliberately "over-trained" beyond the Chinchilla-optimal point by training smaller models on far more data than the 20:1 ratio suggests. Llama 3 8B was trained on 15 trillion tokens, nearly 250 tokens per parameter, roughly 12x the Chinchilla-optimal data-to-parameter ratio. This produces a smaller model that is cheaper to serve while achieving quality closer to what a larger Chinchilla-optimal model would achieve.

2.6 Training Infrastructure

Pre-training a frontier LLM requires thousands of accelerators running for weeks to months. Llama 3 405B was trained on 16,384 NVIDIA H100 GPUs. GPT-4 reportedly used a similar or larger cluster. These clusters are organized into nodes (typically 8 GPUs per node connected by NVLink) interconnected by high-bandwidth networks (InfiniBand or custom fabrics like Google's TPU interconnect).

The training system combines multiple parallelism strategies. Tensor parallelism splits individual layers across GPUs within a node. Pipeline parallelism assigns different groups of layers to different nodes. Data parallelism replicates the full pipeline across groups of nodes, with each replica processing different batches. For mixture-of-experts models, expert parallelism adds a fourth dimension. This 3D or 4D parallelism is necessary because no single strategy can efficiently scale to the cluster sizes required for frontier model training.

Mixed precision training is universal. Modern training uses BF16 (bfloat16) for forward and backward pass computation, with FP32 master copies of weights maintained by the optimizer. BF16 preserves the dynamic range of FP32 (same number of exponent bits) while halving memory and doubling throughput on modern GPUs. Some operations, particularly the loss computation and gradient accumulation, are kept in FP32 for numerical stability.

Checkpointing saves the full training state (model weights, optimizer states, data loader position, learning rate schedule state) periodically, typically every few hundred to few thousand steps. Given the aggregate memory footprint of a frontier training run (tens to hundreds of terabytes of optimizer state), checkpointing requires high-bandwidth storage systems. Asynchronous checkpointing overlaps the checkpoint write with ongoing training to minimize overhead.

2.7 Training Dynamics

The learning rate schedule is one of the most important hyperparameters for stable pre-training. The standard approach is a warmup phase (linearly increasing the learning rate from near-zero over the first few thousand steps) followed by a cosine decay schedule that gradually reduces the learning rate to a small fraction (typically 10%) of the peak value over the course of training. The warmup phase is critical for training stability, as large learning rates applied to randomly initialized weights can cause immediate divergence.

The global batch size is typically ramped up during training, starting smaller (for better gradient signal per token at the start of training when the loss landscape is most volatile) and increasing to the final batch size over the first 5-10% of training. Llama 3 ramped the batch size from 4 million tokens to 16 million tokens. Frontier training runs use batch sizes of 4 to 60 million tokens per step.

Loss curves during pre-training show a characteristic pattern: rapid initial decrease as the model learns basic linguistic patterns, followed by a long, slow, approximately power-law decrease as the model acquires increasingly subtle knowledge. The training loss for a well-configured run is remarkably smooth and predictable, which enables extrapolation of final loss from early training dynamics, a critical capability for making go/no-go decisions early in expensive training runs.

2.8 Compute Costs and Duration

The compute cost of pre-training is enormous and has been growing exponentially. The training compute for Llama 3 405B is estimated at approximately 3.8 x 10^25 FLOPs, consuming roughly 30.8 million GPU-hours on H100s. At typical cloud rental rates of $2-3 per H100-hour, this represents $60-90 million in compute cost alone, not including electricity, engineering time, failed runs, or infrastructure. GPT-4's training cost has been widely estimated at over $100 million. The wall-clock time for these runs is typically 2 to 4 months, constrained by cluster size and the desire to complete training within a reasonable timeframe.

These costs impose a strong practical constraint: most pre-training decisions must be validated through smaller-scale experiments (typically at 1-10 billion parameter scale) before committing to the full run. The predictability of scaling laws is what makes this approach viable: if a configuration works well at small scale with the expected scaling relationship, it can be trusted to work at full scale.

2.9 Common Failure Modes

**Loss spikes.** Sudden, sharp increases in training loss, sometimes by an order of magnitude or more, occur during pre-training and can be caused by data quality issues (a batch of corrupted or highly unusual data), learning rate problems, or numerical instability. Small loss spikes are common and the model typically recovers within a few hundred steps. Large, persistent loss spikes may require rolling back to a previous checkpoint and skipping the problematic data. Llama 3 reported 466 job interruptions during its 54-day training run, including hardware failures, loss spikes, and other issues.

**Training divergence.** If the learning rate is too high, the model's weights can grow without bound, causing NaN loss values and irrecoverable training collapse. This is most common early in training before the warmup phase stabilizes the dynamics, or when training is resumed from a checkpoint with mismatched hyperparameters.

**Data contamination.** If benchmark evaluation data is present in the pre-training corpus, the model may memorize evaluation examples, inflating performance metrics and giving a misleading picture of actual capability. Careful decontamination involves identifying and removing evaluation set content from the training data, though near-duplicate variants can be difficult to catch.

**Slow degradation.** Subtle data quality issues or hyperparameter misconfigurations may not cause obvious instability but can result in a model that trains stably yet converges to a higher loss than expected. These issues are detectable only through careful comparison against scaling law predictions and ablation experiments.

3\. Mid-Training

3.1 What Mid-Training Is

Mid-training refers to a phase of continued pre-training that occurs after the initial pre-training run but before the post-training alignment phase. It has emerged as a distinct stage because certain capabilities and adaptations are most effectively introduced at a scale and in a manner that is neither pure pre-training nor post-training. Mid-training typically involves tens of billions to low trillions of additional tokens, uses pre-training-style objectives (next-token prediction), but modifies the data mixture, context length, learning rate schedule, or model architecture relative to the original pre-training configuration.

The term "mid-training" was not widely used before 2024, but the practice dates back earlier. CodeLlama (2023) is a canonical example: Meta took the Llama 2 base model and continued pre-training on 500 billion additional tokens of predominantly code data, producing a code-specialized model that retained its general capabilities. What has changed is the recognition that this phase deserves deliberate planning from the start, with the pre-training configuration designed to support mid-training modifications.

3.2 Long-Context Extension

One of the most important applications of mid-training is extending the model's effective context length far beyond what was used in pre-training. Pre-training with very long contexts is expensive (attention cost scales quadratically with sequence length for standard attention, and even with FlashAttention the memory and compute cost scales linearly with sequence length), so it is more efficient to pre-train with moderate context lengths (e.g., 8,192 tokens) and extend during mid-training.

**RoPE frequency scaling.** The primary technique for context extension modifies the base frequency of the RoPE positional encoding. By increasing the base frequency (e.g., from 10,000 to 500,000 or 1,000,000), the rotational frequencies are compressed, allowing the model to represent positions beyond its original training range without catastrophic loss of positional information. This approach, used by Llama 3, Code Llama, and many other models, is often called RoPE scaling or NTK-aware interpolation. YaRN (Yet another RoPE extensioN) provides a more refined version that scales different frequency components differently, preserving short-range position resolution while extending long-range capacity.

**Progressive context extension.** Rather than jumping directly from 8K to 128K tokens, mid-training typically extends context length progressively, training for a period at 16K, then 32K, then 65K, then 128K. Each stage allows the model to adapt its attention patterns to the new length before extending further. Llama 3 used a six-stage progressive extension from 8,192 to 131,072 tokens over the course of roughly 800 billion mid-training tokens.

**Data requirements.** Long-context mid-training requires data with genuine long-range dependencies, not just short documents padded or concatenated to fill the context window. Book-length texts, long code files, multi-document collections, and synthetic long-context tasks (such as needle-in-a-haystack retrieval with varying document lengths) are used to train the model to actually attend to and utilize information across the full extended context.

3.3 Domain Adaptation

Mid-training can specialize a general-purpose model for a specific domain by continuing pre-training on domain-specific data. This has been demonstrated effectively for code (Code Llama, DeepSeek-Coder), mathematics (Llemma, DeepSeek-Math), medicine (Med-PaLM, PMC-LLaMA), legal text, and scientific literature. The key challenge is catastrophic forgetting: if the domain-specific data completely replaces the general pre-training mixture, the model loses its general capabilities. Effective domain adaptation mixes domain-specific data with a portion of general data (typically 20-50%) to maintain broad competence while shifting the distribution toward the target domain.

The learning rate for domain adaptation is typically set lower than the peak pre-training learning rate, often at or near the final pre-training learning rate, to avoid disrupting the learned representations. Some approaches use a brief warmup period before continued training.

3.4 Data Mixture Adjustments and Annealing

Even without changing the context length or targeting a specific domain, mid-training can improve model quality by shifting the data mixture toward higher-quality sources. The rationale is that during pre-training, the model needs exposure to vast, diverse data to learn broad patterns, but during the later stages of training, higher-quality data provides more useful gradient signal. Llama 3 used an annealing phase in which the proportion of high-quality data (curated code, math, reasoning examples, and carefully filtered text) was increased while the learning rate was decayed to zero. This annealing phase over the final 40 million tokens of pre-training functionally served as a mid-training step.

Learning rate annealing during mid-training typically follows a cosine decay from the starting learning rate to near zero. The combination of high-quality data and decreasing learning rate allows the model to settle into a region of the loss landscape that reflects the distribution of the high-quality data without the instability that would come from training on this more concentrated mixture at higher learning rates.

3.5 Depth Upscaling and Model Surgery

A more experimental approach to mid-training involves modifying the model architecture itself. Depth upscaling initializes a deeper model by duplicating layers from the pre-trained model and continuing training. For example, a 32-layer model can be expanded to 48 layers by duplicating the middle 16 layers, allowing increased capacity without training from scratch. This technique requires careful handling of the learning rate (lower for the copied layers, which already carry learned representations) and is sensitive to which layers are duplicated.

Width expansion, dense-to-MoE conversion (splitting dense feed-forward layers into multiple experts), and other forms of model surgery have been explored as ways to increase model capacity during mid-training without paying the full cost of pre-training a larger model from scratch. These techniques are less mature than context extension and domain adaptation but represent an active research frontier.

3.6 Notable Examples

**Llama 3.** Meta's Llama 3 training pipeline explicitly included a mid-training phase after the primary 15-trillion-token pre-training run. This phase extended the context length from 8,192 to 131,072 tokens over approximately 800 billion tokens of continued training, adjusted the data mixture to emphasize long-context data, and included an annealing phase on high-quality data.

**Code Llama.** Starting from the Llama 2 70B base model, Meta continued pre-training on 500 billion tokens of predominantly code data, followed by a long-context fine-tuning stage that extended the context from 4,096 to 100,000 tokens. This two-step mid-training process (domain adaptation followed by context extension) produced a code-specialized model that significantly outperformed the base model on coding benchmarks.

**DeepSeek-V2 and V3.** DeepSeek's models used multi-stage training with explicit mid-training phases for context extension and data mixture refinement, demonstrating the value of treating mid-training as a planned part of the pipeline from the beginning.

4\. Post-Training

4.1 Overview and Purpose

Post-training encompasses all the techniques applied after pre-training (and mid-training) to transform a base language model into a useful and safe AI assistant. The base model, despite its broad knowledge, is not inherently helpful: it will generate completions that reflect the statistical distribution of its training data, which includes harmful content, unhelpful formats, and unreliable information alongside high-quality text. Post-training teaches the model to follow instructions, produce helpful and coherent responses, refuse harmful requests, use tools, and engage in multi-turn conversation.

Post-training typically uses orders of magnitude less compute than pre-training. Where pre-training may consume 10^25 FLOPs, post-training might use 10^21 to 10^23 FLOPs. However, this ratio has been shifting: recent frontier models invest significantly more compute in post-training than earlier models did, reflecting the growing recognition that post-training quality is a primary differentiator between models with similar pre-training quality.

4.2 Supervised Fine-Tuning (SFT)

Supervised fine-tuning is typically the first post-training step. The model is trained on a curated dataset of instruction-response pairs, where human annotators (or, increasingly, stronger AI models) have written high-quality responses to diverse prompts. The training objective is the same as pre-training (next-token prediction), but applied only to the response portion of each example, with the instruction portion provided as context.

SFT datasets range from tens of thousands to millions of examples and cover a wide distribution of tasks: question answering, summarization, creative writing, code generation, mathematical reasoning, multi-turn conversation, and more. The quality of SFT data is far more important than the quantity; a small dataset of expertly written responses typically outperforms a large dataset of average-quality responses. The LIMA paper demonstrated that as few as 1,000 carefully curated examples could produce remarkably capable instruction-following behavior, although frontier models use substantially more data for broader coverage.

SFT teaches the model the format and behavioral patterns expected in assistant interactions, but it is limited by the quality ceiling of the demonstrations. If the demonstration data contains errors, inconsistencies, or suboptimal responses, SFT will teach the model to replicate those flaws. This limitation motivates the preference optimization stage that follows.

4.3 RLHF: Reward Modeling and PPO

Reinforcement Learning from Human Feedback addresses SFT's limitations by optimizing the model against a learned measure of response quality rather than individual demonstrations. The RLHF pipeline involves three steps: training a reward model on human preference data, then using reinforcement learning to optimize the language model (the policy) to maximize the reward.

**Reward model training.** Human annotators compare pairs of model responses to the same prompt and indicate which response is better. These pairwise comparisons are used to train a reward model (typically a transformer initialized from the SFT model with a scalar output head) that assigns quality scores to prompt-response pairs. The reward model is trained using the Bradley-Terry model of pairwise preferences. Effective reward models require diverse, high-quality comparison data with good annotator calibration and agreement.

**PPO optimization.** Proximal Policy Optimization, adapted from the reinforcement learning literature, optimizes the language model to generate responses that receive high reward scores. The training loop generates responses from the policy, scores them with the reward model, computes advantage estimates, and updates the policy to increase the probability of high-reward responses. A KL-divergence penalty between the policy and the SFT model prevents the policy from deviating too far from the starting point, mitigating reward hacking, where the model exploits spurious patterns in the reward model.

PPO-based RLHF is operationally complex, requiring four models to be active simultaneously (the policy, the reference SFT model, the reward model, and a value model for advantage estimation), and is sensitive to hyperparameters including the KL penalty coefficient, the clipping ratio, and the learning rate. Despite this complexity, PPO-based RLHF remains the approach used by several frontier labs for their most capable models, as it provides the strongest optimization signal when the reward model is sufficiently accurate.

4.4 DPO and Preference Optimization Variants

Direct Preference Optimization (DPO) simplifies the RLHF pipeline by eliminating the separate reward model and RL optimization step. DPO derives a closed-form mapping between reward functions and optimal policies, enabling direct optimization on preference data using a modified supervised learning objective. The DPO loss increases the log probability of preferred responses and decreases the log probability of dispreferred responses, with an implicit KL constraint controlled by a temperature parameter.

DPO's simplicity, requiring only the SFT model and a preference dataset, has driven widespread adoption. Several variants address perceived limitations: **IPO** (Identity Preference Optimization) regularizes more robustly against overfitting to preference data. **KTO** (Kahneman-Tversky Optimization) works with unpaired examples (individual ratings rather than pairwise comparisons), reducing data collection requirements. **ORPO** (Odds Ratio Preference Optimization) combines SFT and preference optimization into a single stage. **SimPO** (Simple Preference Optimization) uses the average log probability of responses as an implicit reward, eliminating the need for a reference model.

The relative performance of DPO versus PPO-based RLHF at the frontier remains actively debated. Some evidence suggests that PPO provides stronger optimization for models with very high-quality reward models, while DPO is more stable and accessible for smaller-scale training.

4.5 Constitutional AI and RLAIF

Constitutional AI (CAI), developed by Anthropic, introduces a principles-based approach to alignment. Rather than relying solely on human preferences, CAI defines a set of principles (a constitution) that the model uses to self-critique and revise its own outputs. The model generates an initial response, critiques it against the principles, generates an improved response, and the resulting preference pair (original vs. revised) is used for preference optimization. This self-generated preference data supplements or replaces human preference annotations.

Reinforcement Learning from AI Feedback (RLAIF) extends this concept by using a capable AI model to generate preference judgments at scale. RLAIF has been shown to produce alignment quality comparable to human-annotated RLHF for many tasks, while dramatically reducing the cost and turnaround time for generating preference data. This enables more rapid iteration on alignment, with new preference data generated on demand for emerging failure modes.

4.6 Safety Training and Red-Teaming

Safety training is a specialized component of post-training focused on preventing the model from generating harmful, dangerous, or inappropriate content. It encompasses several activities:

**Red-teaming.** Teams of human testers (and increasingly automated systems) attempt to elicit harmful behavior from the model through adversarial prompting, including jailbreak attempts, social engineering, requests for dangerous information, and exploitation of edge cases. The discovered vulnerabilities are used to generate training data for safety fine-tuning.

**Safety-specific SFT.** The model is trained on examples of correctly refusing harmful requests while remaining helpful for benign requests. The challenge is teaching the model to distinguish genuinely harmful requests from superficially similar but benign ones (the over-refusal problem).

**Safety RLHF.** Separate reward models or preference data focused specifically on safety dimensions ensure that the model's safety behavior is optimized alongside helpfulness. Multi-objective optimization balances helpfulness against harmlessness, as excessive safety training can degrade model usefulness.

4.7 Tool Use and Function Calling

Modern LLMs are increasingly trained to use external tools: web search, code execution, calculators, APIs, and arbitrary functions. Tool-use training occurs during post-training and involves SFT on examples of correct tool invocation (generating properly formatted function calls given a set of available tools and their schemas) and tool result interpretation (incorporating tool outputs into coherent responses). More advanced tool-use training teaches the model to plan multi-step tool-use sequences, handle tool errors gracefully, and decide when tool use is necessary versus when the model should respond directly.

4.8 Chat Formatting and System Prompts

Post-training also establishes the model's conversation format, including special tokens that delineate system instructions, user messages, and assistant responses. The model learns to respect system prompt instructions that modify its behavior (persona, constraints, output format), maintain context across multi-turn conversations, and handle conversation-level features like context windows and conversation resets. The specific chat template (e.g., ChatML, Llama's template, or custom formats) is baked into the model during SFT and becomes part of its expected interface.

4.9 Evaluation During Post-Training

Post-training involves continuous evaluation against multiple benchmarks and test suites. Standard evaluations include instruction-following benchmarks (IFEval, MT-Bench, AlpacaEval), knowledge and reasoning benchmarks (MMLU, GPQA, ARC), coding benchmarks (HumanEval, MBPP, SWE-bench), math benchmarks (GSM8K, MATH), and safety evaluations. Automated LLM-as-judge evaluations, where a strong model rates the quality of responses, have become essential for rapid iteration, though human evaluation remains the gold standard for subtle quality differences.

Evaluation results across these benchmarks guide decisions about data mixture adjustments, training duration, and when to advance to the next stage or declare training complete. Regressions on any dimension may trigger additional targeted training.

4.10 Iteration Cycles and the Data Flywheel

Post-training is not a single pass but an iterative process. Each round of evaluation reveals weaknesses, which are addressed through targeted data collection, additional SFT examples, or modified preference optimization. Deployed models generate conversation logs that, with appropriate consent and privacy protections, provide signal about real-world failure modes and user needs. This creates a data flywheel: deployment generates data, data improves post-training, improved post-training produces better models, and better models generate more useful deployment data.

The iterative nature of post-training means that the total post-training compute may be distributed across many training runs with different data compositions and objectives, rather than a single continuous training run. This iterative approach makes post-training more flexible but also more labor-intensive than pre-training, requiring significant human judgment about what to train on and when.

5\. Cross-Cutting Topics

5.1 How the Three Stages Interact

The three stages are not independent; decisions in each stage constrain and enable the others. Pre-training data quality sets a ceiling on the knowledge and capabilities available to mid-training and post-training. The choice of tokenizer in pre-training determines how efficiently the model handles code, math, and multilingual text throughout all stages. Mid-training context extension enables post-training tasks that require long-context capabilities (such as document analysis or repository-level code understanding). Post-training can compensate for some pre-training deficiencies (e.g., improving factual accuracy through targeted SFT), but it cannot instill fundamental capabilities that the pre-trained model lacks entirely.

Feedback also flows backward: insights from post-training inform the next generation's pre-training. If post-training repeatedly struggles to teach a capability, this signals that the pre-training data mixture or training duration should be adjusted. If safety training is unable to suppress certain harmful behaviors, the pre-training data may need more aggressive content filtering.

5.2 Relative Compute Allocation

Pre-training dominates the total compute budget, typically consuming 95-99% of total FLOPs for the overall pipeline. Mid-training, when used, accounts for 1-5% of the pre-training compute (e.g., 800 billion tokens of mid-training versus 15 trillion tokens of pre-training for Llama 3). Post-training historically used a small fraction of pre-training compute, but this fraction has been growing. Reports suggest that frontier models from 2025-2026 allocate significantly more compute to post-training than earlier models, with some estimates placing post-training compute at 5-10% of pre-training compute or higher.

  ------------------- --------------------------------- ----------------------------------------
  **Stage**           **Typical Token Count**           **Approximate % of Total Compute**

  Pre-Training        1T -- 15T+ tokens                 90 -- 98%

  Mid-Training        100B -- 1T tokens                 1 -- 5%

  Post-Training       1B -- 100B+ tokens (multi-round)  1 -- 10% (trending upward)
  ------------------- --------------------------------- ----------------------------------------

5.3 The Trend Toward Longer Post-Training

The increasing emphasis on post-training reflects several factors. As pre-training approaches the limits of available data (high-quality internet text is finite, and synthetic data for pre-training is still maturing), the marginal return on additional pre-training compute decreases. Meanwhile, post-training techniques like RLHF and iterative refinement continue to yield significant quality improvements. The development of techniques for scaling post-training compute effectively, including reinforcement learning on verifiable tasks (math, code) where correctness can be checked automatically, has opened new avenues for post-training investment.

There is also a growing recognition that model differentiation increasingly happens in post-training. When multiple organizations train models of similar size on similar data, the quality of post-training, specifically the data curation, alignment methodology, and safety training, determines which model is more useful, safer, and more pleasant to interact with. This competitive dynamic drives investment in post-training research and engineering.

5.4 Open vs. Closed Approaches

The three training stages vary significantly in transparency across the industry. Pre-training methodologies are relatively well-documented, with detailed technical reports from Meta (Llama), Mistral, and others describing data pipelines, architectures, and training configurations. Mid-training techniques are moderately documented, with papers on context extension and domain adaptation providing reproducible methods. Post-training is the least transparent stage: frontier labs rarely publish detailed accounts of their alignment data, RLHF configurations, safety training procedures, or evaluation criteria. This asymmetry is partly driven by competitive concerns (post-training is a key differentiator) and partly by safety considerations (detailed descriptions of safety training could enable adversaries to circumvent it).

Open-weight models like Llama, Mistral, and Qwen typically release the final post-trained model but not the intermediate checkpoints, training data, or post-training recipes. Fully open efforts like OLMo and DBRX have released more of the pipeline, but even these projects omit some post-training details. This creates a significant reproducibility gap in the field, where the most consequential training decisions are the least visible.

6\. Conclusion

The three-stage training pipeline of pre-training, mid-training, and post-training represents the current state of the art in LLM development. Pre-training remains the dominant consumer of compute, establishing the model's foundational knowledge through next-token prediction on trillions of tokens. Mid-training has emerged as a distinct and increasingly important phase, enabling context extension, domain adaptation, and data quality refinement without the cost of full pre-training. Post-training, encompassing SFT, preference optimization, safety training, and tool-use training, transforms the base model into a useful assistant and is receiving growing investment as the field recognizes its outsized impact on user-facing model quality.

The boundaries between these stages are not rigid, and the field continues to evolve in how it allocates effort across them. The trend toward longer post-training, more deliberate mid-training, and more efficient pre-training suggests that future model development will look less like a single massive training run with minor adjustments, and more like a carefully orchestrated multi-stage pipeline where each phase is optimized for its specific contribution to the final model.
