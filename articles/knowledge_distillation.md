# Knowledge Distillation for Large Language Models

*April 2026 · Technical Report*

## 1. Introduction

The relentless scaling of large language models has produced systems of remarkable capability — but also of remarkable cost. A 400-billion-parameter model that achieves state-of-the-art performance on reasoning benchmarks may require multiple high-end GPUs for inference, consume hundreds of watts of power, and cost dollars per million tokens to serve. For many deployment scenarios — edge devices, cost-sensitive applications, latency-critical services, and privacy-sensitive contexts requiring local inference — these frontier models are impractical.

Knowledge distillation addresses this tension by transferring the knowledge and capabilities of a large, expensive "teacher" model into a smaller, more efficient "student" model. The student, once trained, can replicate a significant fraction of the teacher's capabilities at a fraction of the computational cost. The technique, originally proposed by Hinton, Vinyals, and Dean in 2015, has become one of the most important and widely used methods in the LLM ecosystem, underpinning models like Gemma, Phi, Orca, and many open-source fine-tuned models that achieve surprising capability relative to their size.

This report provides a comprehensive technical examination of knowledge distillation for LLMs: the classical formulation and its extensions, the distinction between white-box and black-box distillation, intermediate layer matching, prominent distilled models, progressive and self-distillation techniques, the relationship between distillation and synthetic data, and the practical limitations and legal considerations that shape real-world application.

## 2. Classical Knowledge Distillation

### 2.1 The Hinton Framework

Geoffrey Hinton, Orion Vinyals, and Jeff Dean's 2015 paper "Distilling the Knowledge in a Neural Network" established the foundational framework for knowledge distillation. The core insight is that the full probability distribution over classes produced by a trained model (the "soft targets") contains more information than the hard class labels alone. When a model classifies an image of a dog, the probabilities it assigns to other categories — higher probability for "wolf" than for "airplane" — encode the model's learned understanding of inter-class relationships.

The classical distillation loss combines two terms:

**Hard label loss**: The standard cross-entropy loss between the student's predictions and the ground-truth labels. This ensures the student learns the correct task.

**Soft label loss (distillation loss)**: The KL divergence between the teacher's softened probability distribution and the student's softened probability distribution. Both distributions are "softened" using a temperature parameter T that smooths the probability distribution, making the relative probabilities of incorrect classes more visible.

The combined loss is:

```
L = α · L_hard(y, p_student) + (1 - α) · T² · KL(p_teacher^(T) || p_student^(T))
```

where `p^(T)` denotes the probability distribution computed with temperature T (dividing logits by T before the softmax), and α controls the weight between the hard and soft losses. The T² factor compensates for the magnitude reduction caused by the temperature scaling.

### 2.2 The Role of Temperature

The temperature parameter T is critical to effective distillation. At T=1 (standard softmax), the teacher's probability distribution is typically very peaked — the correct class has probability near 1.0, and all other classes have probabilities near 0. The soft targets provide little information beyond the hard label.

At higher temperatures (T=2 to T=20), the distribution is smoothed. The relative probabilities of non-target classes become more visible, revealing the teacher's learned similarity structure. A temperature of T=4 might reveal that the teacher considers "wolf" 100× more likely than "airplane" for a dog image — information invisible at T=1 but highly informative for the student.

Too high a temperature flattens the distribution excessively, approaching a uniform distribution that provides no useful signal. The optimal temperature depends on the task, model capacity, and data, but values in the range of 2–10 are typical for classification tasks.

### 2.3 From Classification to Language Modeling

Adapting classical distillation to language models requires extending the framework from classification (a single probability distribution over classes) to sequence generation (a probability distribution over the vocabulary at each position in the sequence). The distillation loss at each token position measures the divergence between the teacher's and student's distributions over the vocabulary:

```
L_distill = Σ_t KL(p_teacher(·|x_1,...,x_{t-1}) || p_student(·|x_1,...,x_{t-1}))
```

where the sum is over all token positions t in the sequence, and the KL divergence is computed over the full vocabulary at each position.

This per-token distillation loss is computationally expensive because it requires computing the teacher's full vocabulary distribution at every position — which means running the teacher model on every training example. For large teacher models (hundreds of billions of parameters), this cost can be prohibitive, motivating the distinction between white-box and black-box distillation.

## 3. White-Box vs. Black-Box Distillation

### 3.1 White-Box Distillation

In white-box distillation, the student has access to the teacher's internal representations: logits (pre-softmax scores), intermediate layer activations, attention patterns, and hidden states. This rich access enables the student to learn not just the teacher's output behavior but its internal reasoning process.

**Advantages**:
- The full logit distribution provides maximal information about the teacher's knowledge.
- Intermediate layer matching (see Section 5) can guide the student's internal representations.
- Attention transfer can teach the student where to attend.
- The temperature parameter can be optimized precisely.

**Requirements**:
- Full access to the teacher model's weights and architecture.
- The ability to run forward passes through the teacher (on training infrastructure that can accommodate the teacher's size).
- Knowledge of the teacher's architecture for designing layer-matching schemes.

White-box distillation is feasible when both teacher and student are developed within the same organization (e.g., distilling a proprietary frontier model into a smaller deployment model) or when both models are open-weight. It is the most effective form of distillation but also the most resource-intensive.

### 3.2 Black-Box Distillation

In black-box distillation, the student has access only to the teacher's outputs — generated text sequences — without access to logits, hidden states, or model internals. This is the dominant form of distillation in the open-source LLM ecosystem, where the teacher is typically a proprietary API model (GPT-4, Claude) that exposes only generated text.

Black-box distillation is effectively supervised fine-tuning on teacher-generated data:

1. Curate a set of prompts/instructions.
2. Generate responses from the teacher model via its API.
3. Fine-tune the student model on (prompt, teacher response) pairs using standard next-token prediction loss.

**Advantages**:
- No access to teacher internals required — only API access.
- Works with any teacher, including proprietary models.
- Simple to implement (standard fine-tuning pipeline).

**Limitations**:
- The student only learns from the teacher's generated text, not from the full distribution over possible responses. This loses the "dark knowledge" encoded in the probabilities of non-generated tokens.
- The student cannot learn from the teacher's internal representations.
- Quality is bounded by the generated text — errors, hallucinations, and biases in the teacher's outputs become training signal for the student.
- The lack of distributional information means the student may learn the surface form of the teacher's responses without learning the underlying reasoning.

### 3.3 Practical Comparison

In practice, the gap between white-box and black-box distillation depends on the amount of generated data and the capability gap between teacher and student. With enough high-quality generated data and moderate capability gaps, black-box distillation can be surprisingly effective — the Alpaca and Vicuna models demonstrated strong instruction-following despite being distilled purely from generated text.

However, for tasks requiring precise calibration, nuanced reasoning, or knowledge of the teacher's uncertainty, white-box distillation's access to the full probability distribution provides advantages that black-box distillation cannot match.

## 4. Task-Specific vs. General Distillation

### 4.1 Task-Specific Distillation

Task-specific distillation trains the student to replicate the teacher's behavior on a specific task or narrow set of tasks. The training data consists of task-specific examples (e.g., sentiment analysis examples for a sentiment classifier, or code generation examples for a coding model).

The advantage is efficiency: a small student model can achieve near-teacher performance on a specific task with relatively little training data. The DistilBERT model (Sanh et al., 2019) demonstrated this effectively — a 6-layer transformer distilled from 12-layer BERT retained 97% of BERT's performance on GLUE benchmarks while being 60% smaller and 60% faster.

The disadvantage is narrow applicability. A task-specifically distilled model performs well on the target task but may lose the teacher's general capabilities. For LLM applications where users interact with the model across diverse tasks, task-specific distillation is insufficient.

### 4.2 General Distillation

General distillation aims to transfer the teacher's broad capabilities across all tasks, producing a smaller model that serves as a general-purpose replacement for the teacher. This is much more challenging, as the student must learn the teacher's knowledge and reasoning abilities across the full range of possible inputs.

General distillation for LLMs typically involves:

1. **Pre-training distillation**: Training the student from scratch (or continuing pre-training) with a distillation objective that matches the teacher's logit distribution on a large, diverse corpus.
2. **Instruction-tuning distillation**: Fine-tuning the student on a diverse set of instruction-following examples generated by the teacher, covering many task types and domains.
3. **Preference distillation**: Training the student to match the teacher's preferences (via DPO or RLHF) on a diverse set of response comparisons.

Each stage transfers a different aspect of the teacher's capabilities: pre-training distillation transfers world knowledge and language modeling ability, instruction-tuning distillation transfers task-following and formatting abilities, and preference distillation transfers alignment and response quality.

### 4.3 Curriculum-Based General Distillation

Effective general distillation often follows a curriculum — presenting training examples in an order that facilitates learning:

1. **Start with simple tasks**: Basic language modeling, simple Q&A, factual recall.
2. **Progress to intermediate tasks**: Multi-step reasoning, summarization, analysis.
3. **End with complex tasks**: Mathematical proofs, extended code generation, nuanced argumentation.

This curriculum approach, used in Orca and Orca 2, produces student models that learn foundational capabilities before attempting to match the teacher's performance on challenging tasks.

## 5. Intermediate Layer Matching

### 5.1 Beyond Output Matching

White-box distillation can go beyond matching the teacher's output distribution to matching intermediate representations. The intuition is that the teacher's internal representations encode useful computational patterns (how to represent syntactic structure, how to route information between attention heads, how to encode factual knowledge) that the student can benefit from imitating.

### 5.2 Hidden State Matching

The simplest form of intermediate layer matching minimizes the distance between the teacher's and student's hidden states at corresponding layers:

```
L_hidden = Σ_l ||f(h_student^l) - h_teacher^{g(l)}||²
```

where `h_student^l` is the hidden state of the student's layer l, `h_teacher^{g(l)}` is the hidden state of a corresponding teacher layer (determined by a mapping function g), and `f` is a learned projection that maps the student's hidden dimension to the teacher's.

Since the teacher and student typically have different numbers of layers and different hidden dimensions, the mapping function g and the projection f must be carefully designed. Common strategies include:

- **Uniform mapping**: Map every k-th teacher layer to successive student layers (if the teacher has 32 layers and the student has 8, map teacher layers 4, 8, 12, 16, 20, 24, 28, 32 to student layers 1–8).
- **Last-layer mapping**: Only match the final layers of each.
- **Learned mapping**: Train a small network to determine the optimal layer correspondence.

### 5.3 Attention Transfer

Attention transfer matches the attention patterns (attention weight matrices) between teacher and student. The attention weights encode which tokens attend to which other tokens — a form of structural knowledge about language that can be directly transferred.

The attention transfer loss is typically:

```
L_attn = Σ_l Σ_h ||A_student^{l,h} - A_teacher^{g(l),h'}||²
```

where A denotes attention matrices and the summation is over layers l and attention heads h (with appropriate mapping between teacher and student heads).

Attention transfer is most useful when the teacher's attention patterns capture meaningful linguistic structure that the student benefits from learning. However, there is evidence that matching attention patterns too rigidly can hurt student performance — the student may benefit from developing its own attention patterns that are better suited to its smaller architecture.

### 5.4 Embedding Matching

Matching the teacher's and student's token embeddings (input representations) is another form of intermediate matching. This is particularly useful when the teacher and student use the same tokenizer, as it directly aligns the representation spaces at the input level.

### 5.5 MiniLLM and Advanced Objectives

MiniLLM (Gu et al., 2024) introduced a reverse KL divergence objective for LLM distillation, arguing that the standard forward KL divergence (used in classical distillation) encourages the student to spread probability mass across all tokens the teacher considers plausible, which can lead to incoherent generation. The reverse KL divergence instead encourages the student to focus on the modes of the teacher's distribution, producing more focused and coherent outputs at the cost of reduced diversity.

MiniLLM combines the reverse KL objective with reinforcement learning techniques (policy gradient methods) to optimize the student's generation quality, achieving better results than standard distillation objectives for open-ended text generation tasks.

## 6. Notable Distilled Models

### 6.1 Gemma

Google's Gemma models (2B and 7B parameters) were trained with knowledge from the much larger Gemini family. While Google has not published full details of the distillation process, the Gemma models demonstrate capabilities that significantly exceed what would be expected from models of their size trained independently, suggesting substantial knowledge transfer from the larger Gemini models.

Gemma 2 (2024) more explicitly incorporated distillation. The Gemma 2 2B and 9B models were distilled from the Gemma 2 27B model, using a combination of logit distillation (matching the teacher's token probability distributions) and on-policy data generation (training on data generated by the student, scored by the teacher). The distillation was integrated into the training process rather than applied as a separate fine-tuning step.

### 6.2 Phi Models

Microsoft's Phi models (discussed in detail in the companion report on Synthetic Data Generation) use distillation through synthetic data. GPT-3.5 and GPT-4 serve as teachers, generating "textbook quality" training data that is then used to train small student models (1.3B to 14B parameters). This is black-box distillation at scale — the teacher's knowledge is transferred entirely through generated text, without access to logits or internal representations.

The Phi models demonstrate that black-box distillation can be extraordinarily effective when the generated data is high quality and carefully curated. Phi-3 (3.8B) and Phi-4 (14B) achieve performance comparable to models many times their size on reasoning and knowledge benchmarks.

### 6.3 Orca and Orca 2

Microsoft's Orca (June 2023) and Orca 2 (November 2023) represent a systematic approach to distillation that emphasizes learning the reasoning process:

**Orca 1**: A 13B model distilled from GPT-4 using 5 million instruction-following examples. The key innovation was the system prompt design — GPT-4 was prompted to "think step by step" and "explain your reasoning," producing training examples that taught the student not just what to answer but how to reason about the answer.

**Orca 2**: Extended the approach with "Cautious Reasoning" — teaching the student to select different reasoning strategies for different types of problems (step-by-step for math, extractive for factual questions, direct for simple tasks). Orca 2 used a curriculum that progressively increased task difficulty, and it was trained at both 7B and 13B scales.

Orca 2 demonstrated that a 13B model, through carefully designed distillation, could match or exceed GPT-4's performance on specific reasoning benchmarks — a striking result that highlighted the power of reasoning-focused distillation.

### 6.4 Alpaca, Vicuna, and the Instruction-Tuning Wave

The wave of instruction-tuned models in early 2023 — Alpaca, Vicuna, Koala, Dolly, and many others — all used some form of black-box distillation from GPT-3.5 or GPT-4. These models demonstrated that even simple distillation (generating instruction-following examples and fine-tuning) could produce usable instruction-following capabilities in small open-weight models.

Vicuna (March 2023) achieved particular prominence by fine-tuning LLaMA-13B on approximately 70K conversation examples shared by users on ShareGPT — conversations with ChatGPT that were publicly posted. GPT-4 evaluation (a then-novel evaluation method) assessed Vicuna's responses as achieving 90% of ChatGPT's quality, establishing a benchmark for distillation effectiveness.

### 6.5 DistilBERT and TinyBERT

Before the LLM era, DistilBERT and TinyBERT demonstrated distillation for encoder models:

**DistilBERT** (Sanh et al., 2019): A 6-layer, 66M-parameter model distilled from BERT-base (12 layers, 110M parameters). Used a combination of output distillation, intermediate hidden state matching, and a cosine embedding loss. Retained 97% of BERT's performance on GLUE while being 60% smaller and 60% faster.

**TinyBERT** (Jiao et al., 2020): Extended DistilBERT's approach with more comprehensive layer matching and a two-stage distillation process (general pre-training distillation followed by task-specific distillation). Achieved strong results with even smaller student models.

### 6.6 Distilled Reasoning Models

A recent trend is distilling the chain-of-thought reasoning capabilities of large models. DeepSeek-R1-Distill models (2025) distilled DeepSeek-R1's extended thinking capabilities into smaller models (1.5B to 70B parameters), producing students that could perform multi-step reasoning with explicit thinking traces. These models demonstrated that the reasoning patterns of very large models — not just their factual knowledge — could be transferred through distillation.

## 7. Progressive Distillation

### 7.1 The Progressive Approach

Progressive distillation trains a sequence of increasingly smaller student models, each distilled from the previous one. Rather than directly distilling a 400B teacher into a 7B student (a large capability gap), the pipeline might proceed:

400B → 70B → 13B → 7B

Each step bridges a smaller capability gap, potentially preserving more knowledge than a single large step.

### 7.2 Theoretical Motivation

The motivation for progressive distillation comes from the observation that very large capability gaps between teacher and student can lead to poor distillation outcomes. When the teacher's behavior is too complex for the student to approximate, the student may learn shallow heuristics rather than the teacher's actual reasoning process. By reducing the gap at each step, progressive distillation gives each student a more achievable learning target.

### 7.3 Practical Considerations

Progressive distillation has practical tradeoffs:

**Advantages**:
- Each step bridges a manageable capability gap.
- Intermediate models are useful products themselves.
- Error accumulation can be managed by re-including the original teacher's signal at each step.

**Disadvantages**:
- Multiple training runs increase total compute cost.
- Errors compound — each step may lose information, and these losses accumulate.
- The intermediate model sizes must be chosen carefully.

In practice, progressive distillation is less commonly used for LLMs than direct distillation, partly because the compute cost of multiple training runs is significant and partly because direct distillation from very capable teachers has proven surprisingly effective even with large capability gaps.

## 8. On-Policy vs. Off-Policy Distillation

### 8.1 Off-Policy Distillation

In off-policy distillation, the training data is generated by the teacher model (or comes from a fixed dataset). The student learns from the teacher's behavior on examples that the teacher generated or that were drawn from a fixed distribution. This is the standard approach: generate data from the teacher, train the student on that data.

The problem with off-policy distillation is distribution mismatch. During inference, the student generates text autoregressively, conditioning each token on its own previous outputs. But during training, it conditioned on the teacher's outputs. If the student makes an error early in a sequence, it may encounter states that never appeared in the training data (because the teacher would not have made that error), and its behavior in these states is unpredictable.

### 8.2 On-Policy Distillation

On-policy distillation addresses distribution mismatch by generating training data from the student model itself, then using the teacher to evaluate or correct the student's outputs. Several approaches fall under this umbrella:

**ImitKD (Imitation-based Knowledge Distillation)**: The student generates sequences, and the teacher provides token-level probability distributions for the student's generated sequences. The distillation loss matches these distributions. Because the training data comes from the student's own distribution, the distribution mismatch is eliminated.

**GKD (Generalized Knowledge Distillation)**: A framework that interpolates between off-policy and on-policy distillation by generating training data from a mixture of the student and teacher distributions. The mixing ratio controls the tradeoff between training efficiency (off-policy is more efficient when the student is weak) and distribution matching (on-policy is better when the student is closer to the teacher).

**Rejection sampling with teacher scoring**: The student generates multiple responses, the teacher scores them (either through logit comparison or through a reward model trained to represent the teacher's preferences), and the student is trained on its own high-scoring responses. This is on-policy (data comes from the student) with teacher guidance (the teacher selects which student outputs to train on).

### 8.3 Gemma 2's Approach

Google's Gemma 2 distillation used an on-policy approach. The student model generated text, and the teacher (Gemma 2 27B) provided logits for the student's generated sequences. The distillation objective matched the teacher's logits on the student's distribution, combining the benefits of on-policy data (distribution matching) with white-box distillation (access to teacher logits). Google reported that this on-policy approach was critical to the quality of the distilled models.

## 9. Self-Distillation

### 9.1 The Concept

Self-distillation is the counterintuitive technique of distilling a model from itself. The model serves as both teacher and student, typically with some form of asymmetry:

- **Temporal asymmetry**: A later checkpoint distills into an earlier checkpoint (or the model is trained to match its own earlier predictions).
- **Architectural asymmetry**: The full model distills into a subset of itself (e.g., using only a subset of layers or heads).
- **Ensemble asymmetry**: An ensemble of the same model (with different random seeds or dropout masks) serves as the teacher for a single instance.
- **Augmentation asymmetry**: The model's predictions on augmented inputs (back-translation, paraphrasing) serve as soft targets for the original inputs.

### 9.2 Self-Distillation in Practice

Self-distillation has been shown to improve model performance without changing model size — a result that seems paradoxical (how can a model learn from itself?). The explanation is that the soft targets from self-distillation provide a richer training signal than hard labels. The model's own predictions encode learned relationships between classes/tokens that the hard labels do not capture, and training on these soft targets amounts to a form of label smoothing that improves generalization.

Born-Again Networks (Furlanello et al., 2018) demonstrated that training a student of identical architecture to the teacher, using the teacher's soft targets, consistently produced a student that outperformed the teacher. This counterintuitive result suggests that the distillation process itself — not just the teacher-student capacity gap — provides a beneficial training signal.

### 9.3 Self-Distillation for LLMs

In the LLM context, self-distillation takes several forms:

**Self-rewarding models**: The model generates responses and evaluates them, using its own evaluations as training signal for improvement. Meta's Self-Rewarding Language Models (2024) demonstrated this approach, achieving iterative improvement through self-evaluation.

**Self-play**: The model generates both sides of a debate or conversation, and the resulting data is used for further training. This can improve reasoning and argumentation capabilities.

**Iterative refinement**: The model generates a response, critiques it, revises it, and is trained on the revised version. This Constitutional AI-style approach is a form of self-distillation where the teacher is the model's own critique-and-revision capability.

## 10. Distilling Chain-of-Thought Reasoning

### 10.1 The Challenge

Chain-of-thought (CoT) reasoning — generating explicit intermediate reasoning steps before producing a final answer — is a capability that emerges primarily in very large models. Directly prompting a small model to "think step by step" often produces incoherent or incorrect reasoning chains, even when the model has the factual knowledge to answer the question directly. Distilling CoT reasoning from large to small models is therefore a particularly valuable and challenging form of distillation.

### 10.2 Approaches to CoT Distillation

**Direct CoT distillation**: Generate CoT reasoning traces from the teacher and train the student on (question, CoT reasoning, answer) triples. The student learns to produce reasoning chains that mimic the teacher's style. This works for straightforward reasoning tasks but may fail for complex reasoning where the student lacks the capacity to faithfully reproduce the teacher's reasoning.

**Step-by-step distillation**: Hsieh et al. (2023) proposed extracting rationales (reasoning explanations) from the teacher and training the student on these rationales alongside the task labels. The student learns from both the reasoning process and the final answer, with a multi-task loss that weights both objectives.

**Reasoning chain distillation with verification**: Generate reasoning chains from the teacher, verify each chain's correctness (using automated verification for math/code, or a judge model for open-ended reasoning), and train the student only on verified correct chains. This ensures the student learns from accurate reasoning rather than from the teacher's errors.

**Process reward models**: Train a process reward model (PRM) that evaluates the correctness of individual reasoning steps, then use the PRM to filter and score teacher-generated reasoning chains. The student is trained on high-scoring chains, learning step-by-step reasoning that is more likely to be correct.

### 10.3 DeepSeek-R1 Distillation

DeepSeek-R1 (January 2025) demonstrated large-scale distillation of extended reasoning capabilities. The DeepSeek-R1 model (a 671B parameter mixture-of-experts model) was trained to produce lengthy reasoning traces before answering, similar to OpenAI's o1 model. The reasoning capabilities were then distilled into smaller models:

- DeepSeek-R1-Distill-Qwen-1.5B
- DeepSeek-R1-Distill-Qwen-7B
- DeepSeek-R1-Distill-Qwen-14B
- DeepSeek-R1-Distill-Qwen-32B
- DeepSeek-R1-Distill-Llama-8B
- DeepSeek-R1-Distill-Llama-70B

The distillation used approximately 800K carefully curated reasoning examples generated by the full R1 model. The distilled models demonstrated strong reasoning capabilities, with the 14B and 32B variants achieving performance competitive with or exceeding much larger non-reasoning models on math and coding benchmarks. This demonstrated that the extended thinking paradigm could be effectively distilled, though with some capability loss — the distilled models' reasoning was less reliable on the most complex problems compared to the full model.

## 11. Quantization-Aware Distillation

### 11.1 The Intersection of Distillation and Quantization

Quantization (reducing the numerical precision of model weights from FP32/FP16 to INT8/INT4) is a complementary model compression technique that can be combined with distillation. Quantization-aware distillation trains the student with quantized weights from the start, ensuring that the student's representations are optimized for the target precision rather than being adapted post-hoc.

### 11.2 Approaches

**QAT with distillation**: Quantization-aware training (QAT) simulates quantization effects during training by rounding weights to the target precision in the forward pass while maintaining full-precision gradients in the backward pass. Combining QAT with distillation means the student learns to produce good outputs despite operating at reduced precision, with the teacher's soft targets guiding the learning.

**Distillation-guided quantization**: Use the teacher's representations to guide the choice of quantization parameters (scale factors, zero points) for the student. This can produce better quantized models than standard PTQ (post-training quantization) because the quantization parameters are optimized to minimize the distillation loss rather than a generic reconstruction loss.

**QuIP# and AQLM**: Advanced quantization methods that use codebook-based quantization combined with distillation objectives to achieve very low bitrates (2–3 bits per weight) with minimal quality loss. The distillation objective ensures that the quantized model's outputs remain close to the full-precision teacher's outputs despite extreme compression.

### 11.3 Practical Benefits

Combining distillation and quantization produces models that are compressed on two axes simultaneously: fewer parameters (from the smaller student architecture) and lower precision per parameter (from quantization). A 7B student model quantized to 4-bit can be served with approximately 3.5 GB of memory — small enough to run on a smartphone or a consumer laptop without a dedicated GPU.

## 12. Relationship to Synthetic Data

### 12.1 Distillation as Synthetic Data Generation

Black-box distillation and synthetic data generation are fundamentally the same process viewed from different perspectives. When a teacher model generates instruction-following examples that are used to train a student model, this is simultaneously:

- **Distillation**: Transferring the teacher's knowledge to the student.
- **Synthetic data generation**: Creating training data using an AI model.

The distinction is largely one of framing. The "distillation" framing emphasizes the teacher-student relationship and knowledge transfer. The "synthetic data" framing emphasizes the data pipeline and quality engineering. In practice, the same techniques (quality filtering, diversity optimization, curriculum design) apply regardless of framing.

### 12.2 When Distillation and Synthetic Data Diverge

The two framings diverge in important ways:

- **Distillation focuses on the teacher-student relationship**: The quality of distillation depends on the capability gap between teacher and student, the fidelity of knowledge transfer, and the student's capacity to absorb the teacher's knowledge.
- **Synthetic data focuses on the data distribution**: The quality of synthetic data depends on its diversity, accuracy, and coverage of the target capability space, regardless of any specific teacher-student pairing.

In practice, the best approaches combine both perspectives: generating diverse, high-quality synthetic data (the synthetic data perspective) while ensuring it effectively transfers capabilities from a strong teacher to the target student (the distillation perspective).

## 13. Legal Concerns

### 13.1 Terms of Service

Most commercial API providers' terms of service contain provisions relevant to distillation. OpenAI's usage policies, for example, prohibit using API outputs to "develop models that compete with OpenAI." Similar provisions exist in other providers' terms. These restrictions are specifically targeted at the distillation use case — preventing competitors from using a frontier model's capabilities as free training signal for competing products.

The enforceability and scope of these restrictions are subjects of ongoing legal debate. Key questions include:

- Does fine-tuning an open-weight model on API-generated data constitute "developing a model that competes"?
- Does using outputs for research purposes fall under a different legal category than commercial use?
- Are terms-of-service restrictions on model outputs enforceable, given that the outputs are generated in response to the user's specific prompts?

### 13.2 Intellectual Property

The intellectual property status of distilled knowledge is legally unsettled. If a student model learns factual knowledge, reasoning patterns, and writing styles from a teacher's outputs, does this constitute copying the teacher? The analogy to students learning from textbooks suggests not — but the scale and directness of the transfer may lead to different legal conclusions.

Several lawsuits have touched on this question, though no definitive precedent has been established as of early 2026. The outcome may depend on the specifics: distilling a model that closely mimics a specific teacher's distinctive style or reproduces its exact phrasings may face stronger legal challenges than distilling general capabilities.

### 13.3 Licensing of Open-Weight Models

Many open-weight models come with licenses that restrict or permit distillation. The Llama 3 license permits using model outputs for most purposes, including training other models. Gemma's license similarly permits distillation. Some models (particularly those with non-commercial licenses) explicitly restrict the use of model outputs for training competing models.

Understanding these licensing terms is important for organizations building distillation pipelines. Using a model's outputs in violation of its license creates legal risk even if the technical capability to do so exists.

## 14. When Distillation Fails

### 14.1 Capability Cliffs

Distillation does not smoothly transfer all capabilities. Some capabilities exhibit "cliffs" — they work well above a certain model size but fail completely below it. Common examples include:

- **Complex multi-step reasoning**: A 70B teacher may solve 5-step reasoning problems reliably, but a 7B student may solve 1–2 step problems while completely failing at 5-step problems. The student doesn't learn a slightly degraded version of 5-step reasoning — it simply cannot do it.
- **Long-context coherence**: The ability to maintain coherence over long contexts may not transfer to smaller models that lack the capacity to manage complex state.
- **Rare knowledge**: Factual knowledge about less common topics may not transfer, as the student's smaller parametric memory cannot store all the teacher's knowledge.

### 14.2 Imitation vs. Understanding

A persistent concern with distillation is that the student may learn to imitate the teacher's surface-level behavior without learning the underlying reasoning. Gudibande et al. (2023) argued that distillation from proprietary models primarily transfers the teacher's "style" (formatting, verbosity, confidence) rather than its capabilities (factual knowledge, reasoning ability). They found that distilled models performed well on human preference evaluations (which rewarded confident, well-formatted responses) but poorly on factual knowledge benchmarks (which required actual knowledge).

This "imitation gap" is most pronounced when:
- The capability gap between teacher and student is large.
- The distillation data lacks diversity in reasoning strategies.
- Evaluation metrics reward surface-level similarity to the teacher's style.

### 14.3 Hallucination Transfer

Teachers hallucinate, and their hallucinations become "ground truth" in the student's training data. Unlike factual errors in human-written text (which are randomly distributed), teacher hallucinations are systematically correlated with the teacher's weaknesses. If the teacher consistently hallucinates about a particular topic, the student will learn those same hallucinations with high confidence.

Verification and filtering of teacher outputs partially mitigate this problem but cannot eliminate it entirely, particularly for domains where automated verification is difficult.

### 14.4 Distributional Narrowing

Distilled models often produce outputs that are less diverse than the teacher's. The distillation process tends to capture the modes of the teacher's distribution while losing the tails. This manifests as:

- More formulaic responses (converging to the teacher's most common output patterns).
- Reduced creativity and variety in open-ended generation.
- Less calibrated uncertainty (the distilled model may be overconfident because it learned from the teacher's most confident outputs).

## 15. Best Practices

### 15.1 Teacher Selection

Choose a teacher that is:
- Significantly more capable than the student on the target tasks.
- Well-calibrated (its confidence correlates with its accuracy).
- Strong on the specific capabilities you want to transfer.
- Available with sufficient access (logits for white-box, API for black-box).

### 15.2 Data Quality

Invest heavily in data quality:
- Use multiple quality filters (reward models, judge models, automated verification).
- Ensure diversity (topic, format, difficulty, reasoning strategy).
- Include negative examples and edge cases, not just the teacher's best outputs.
- Verify factual claims when possible.

### 15.3 Training Strategy

- Use a curriculum (easy → hard).
- Start with a pre-trained student (not random initialization).
- Combine distillation loss with standard next-token prediction loss.
- For white-box distillation, combine output distillation with intermediate layer matching.
- Monitor for distribution mismatch (on-policy data generation helps).

### 15.4 Evaluation

Evaluate distillation effectiveness on:
- Task-specific benchmarks (does the student match the teacher on target tasks?).
- General benchmarks (has the student maintained broad capabilities?).
- Factual accuracy (has the student learned the teacher's knowledge or just its style?).
- Output diversity (is the student producing varied, creative responses?).
- Calibration (does the student's confidence correlate with accuracy?).

## 16. Conclusion

Knowledge distillation has become an essential technique in the LLM ecosystem, enabling the capabilities of frontier models to be deployed at scale through smaller, more efficient student models. The technique spans a spectrum from classical white-box distillation (matching teacher logits and internal representations) to the ubiquitous black-box distillation (training on teacher-generated text) that underlies much of the open-source model ecosystem.

The most successful distilled models — Gemma, Phi, Orca, and the DeepSeek-R1 distillation series — demonstrate that careful distillation can produce small models with remarkable capabilities. But distillation has real limitations: capability cliffs, imitation without understanding, hallucination transfer, and distributional narrowing all constrain what distillation can achieve. The student is not a miniature version of the teacher — it is a different model with different strengths and weaknesses, shaped by both the teacher's knowledge and the student's architectural constraints.

The field is evolving toward more sophisticated distillation approaches: on-policy methods that address distribution mismatch, process-level supervision that transfers reasoning strategies rather than just final answers, and multi-teacher approaches that combine the strengths of diverse teachers. As these methods mature, the practical capabilities available at small model scales will continue to increase, broadening access to capable AI systems.

## References

1. Hinton, G., Vinyals, O., and Dean, J. "Distilling the Knowledge in a Neural Network." NIPS Workshop 2015.
2. Sanh, V., et al. "DistilBERT, a Distilled Version of BERT: Smaller, Faster, Cheaper and Lighter." 2019.
3. Jiao, X., et al. "TinyBERT: Distilling BERT for Natural Language Understanding." EMNLP 2020.
4. Hsieh, C.-Y., et al. "Distilling Step-by-Step! Outperforming Larger Language Models with Less Training Data and Smaller Model Sizes." ACL 2023.
5. Mukherjee, S., et al. "Orca: Progressive Learning from Complex Explanation Traces of GPT-4." 2023.
6. Mitra, A., et al. "Orca 2: Teaching Small Language Models How to Reason." 2023.
7. Gu, Y., et al. "MiniLLM: Knowledge Distillation of Large Language Models." ICLR 2024.
8. Agarwal, R., et al. "GKD: Generalized Knowledge Distillation for Auto-Regressive Sequence Models." 2024.
9. Gemma Team, Google. "Gemma 2: Improving Open Language Models at a Practical Size." 2024.
10. DeepSeek-AI. "DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning." 2025.
11. Gudibande, A., et al. "The False Promise of Imitating Proprietary LLMs." 2023.
12. Furlanello, T., et al. "Born Again Neural Networks." ICML 2018.
13. Yuan, W., et al. "Self-Rewarding Language Models." 2024.
14. Gunasekar, S., et al. "Textbooks Are All You Need." 2023.
15. Taori, R., et al. "Stanford Alpaca: An Instruction-following LLaMA Model." 2023.
