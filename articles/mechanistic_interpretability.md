# Mechanistic Interpretability: Reverse-Engineering Neural Networks

*April 2026 • Technical Report*

## 1. Introduction

A trained large language model is a black box. The weights are a collection of billions of numbers learned through gradient descent on enormous datasets. The model's behavior — its ability to generate coherent text, answer questions, write code, reason about novel problems — emerges from the interaction of these weights with input prompts. But the relationship between specific weights and specific behaviors is opaque. You cannot point to a row of a matrix and say "this is the part of the model that knows about French syntax." The knowledge is distributed across the network in ways that resist straightforward inspection.

Mechanistic interpretability is the research program of changing this. The goal is to reverse-engineer trained neural networks into human-understandable algorithms — to look at the actual weights and activations and explain, in mechanical terms, what computation the network is performing and why. The field has produced striking results in the past several years, identifying specific circuits inside language models that implement specific computations, and developing tools (sparse autoencoders being the most prominent) that decompose model representations into interpretable features.

If mechanistic interpretability succeeds at scale, it would transform AI safety. We could understand what models actually know, detect deceptive or misaligned behavior, and verify that models are doing what we want them to do for the reasons we want them to do it. If it fails — if neural networks remain too complex for human understanding even with the best tools — we will be deploying increasingly capable systems whose internals we cannot inspect.

This report examines the techniques, the major findings, the open questions, and the practical implications of mechanistic interpretability.

## 2. The Goal: Understanding from First Principles

Mechanistic interpretability is distinct from other forms of model interpretability in its ambition. **Behavioral interpretability** (asking the model questions and analyzing its answers) tells you what the model can do but not why. **Attribution methods** (gradient-based saliency, attention visualization) tell you which inputs influenced an output but not the algorithmic structure of the influence. **Probing classifiers** (training simple models on hidden representations to predict properties of interest) tell you what information is present in a representation but not how the model uses it.

Mechanistic interpretability aims for something more rigorous. The goal is to identify the actual computation: which neurons fire under which conditions, how they connect to other neurons, what role each plays in the larger algorithm. The product is not a heatmap or a probability score — it is a circuit diagram that explains the model's behavior in a specific situation.

The closest analogy is to neuroscience or to reverse-engineering an unknown computer program. You know the inputs and outputs and can observe internal activity, but the original "source code" is unavailable. The job is to construct a hypothesis about the algorithm from observation.

## 3. Early Findings: Induction Heads

The first major result of modern mechanistic interpretability was the discovery of induction heads in transformer language models. Induction heads, identified in a 2022 Anthropic paper, are pairs of attention heads that work together to implement a specific algorithm: copying patterns from earlier in the input.

The mechanism is elegant. The first attention head ("previous token head") looks at each position and shifts its attention to the previous token. The second attention head ("induction head") then looks for places earlier in the input where the previous token matches the current token, and copies the token that came after that match. The combined effect is that if the input contains the pattern "[A][B] ... [A]", the induction head will predict [B] as the next token.

This circuit explains a wide range of model behaviors: in-context learning of patterns, memorization of repeated phrases, completion of structured templates. It is not the only mechanism the model uses, but it is a clear, identifiable component that can be located in specific attention heads at specific layers and verified by ablation experiments (removing those heads breaks the behavior).

The induction head finding was important not just for what it revealed but for what it demonstrated: that mechanistic interpretability could produce real, verifiable explanations of model behavior. The skeptics' position — that neural networks are fundamentally too complex to understand — was partially refuted. At least some computations could be reverse-engineered, even in models with hundreds of millions of parameters.

## 4. Sparse Autoencoders

The technique that has done the most to advance mechanistic interpretability is the sparse autoencoder (SAE). Introduced in earlier neural network research and adapted to language models in 2023-2024, SAEs are auxiliary neural networks trained to decompose model activations into interpretable features.

The setup: pick a layer of a language model. Take the activation vector at that layer for many input examples (millions or billions). Train a sparse autoencoder — a small network with an over-complete hidden layer and a sparsity penalty — to encode the activations into a sparse vector and decode them back. The hidden layer of the SAE has many more dimensions than the original activation, but the sparsity penalty ensures that only a few of those dimensions are active at a time.

The hope (largely confirmed empirically) is that the SAE's sparse hidden dimensions correspond to interpretable features. A specific dimension might activate when the model is processing tokens about France, or when it is about to predict a number, or when it is reading code. By inspecting which inputs activate each SAE feature, researchers can label the features and understand what concepts the model represents at that layer.

Anthropic's "Towards Monosemanticity" paper (October 2023) demonstrated this approach on a small language model and identified thousands of interpretable features. The follow-up "Scaling Monosemanticity" paper (May 2024) applied the technique to Claude 3 Sonnet and identified millions of features, including some with surprising properties: a feature for the Golden Gate Bridge that activated specifically on text mentioning the bridge, a feature for code vulnerabilities, features for emotional tone, features for specific historical figures.

Crucially, these features could be manipulated. By artificially activating or suppressing specific features, researchers could change the model's behavior in predictable ways. The "Golden Gate Claude" demo, which clamped the Golden Gate Bridge feature to a high value and let users chat with the resulting model, became a viral demonstration: the model became obsessed with the bridge, weaving it into every response regardless of topic. This was a direct, mechanistic intervention on a specific concept inside the model.

### 4.1 Why SAEs Work

The intuition behind SAEs is the superposition hypothesis: language models represent more concepts than they have neurons, by overlapping representations in a way that is recoverable through the right decoder. The SAE is essentially a learned decoder that disentangles the superposed features.

The theoretical basis is the compressed sensing literature, which shows that sparse signals can be recovered from low-dimensional measurements when the underlying basis is appropriately structured. The SAE learns this basis from data, finding the directions in activation space that correspond to discrete, interpretable concepts.

The downside is that SAEs are not the only valid decomposition. Different SAEs trained on the same activations may produce different feature sets, and there is no canonical answer to "what features does this model use?" The features are useful for analysis and intervention, but they are tools for understanding rather than ground truth about the model's internal representation.

## 5. Circuits and Algorithms

Sparse autoencoders identify features. The next step is identifying circuits — groups of features and weights that work together to perform specific computations. This is harder, but several papers have made progress.

The most rigorous work has been on small models (single-layer attention heads, two-layer MLPs) where circuits can be exhaustively characterized. For these toy models, researchers have identified algorithms for tasks like modular arithmetic, sorting, and small-scale pattern matching. The algorithms are surprisingly elegant — the model often discovers efficient solutions that resemble what a human would design.

For larger models, full circuit identification is computationally infeasible. The number of possible feature interactions grows combinatorially with the model size. The compromise is to identify circuits for specific narrow tasks: how does the model predict that the next token is an opening parenthesis? How does it complete arithmetic problems? How does it identify when a question is being asked? These narrow circuit analyses produce real findings but cover only a tiny fraction of the model's overall computation.

The hope is that circuits compose: if you can understand how each individual circuit works, you can understand how they interact to produce more complex behavior. Whether this is actually true at scale is one of the open questions of mechanistic interpretability.

## 6. The Anthropic Investment

Mechanistic interpretability has been most aggressively pursued by Anthropic, where it is treated as a core safety research area. Anthropic has published multiple major papers on the topic, including the induction head paper, the monosemanticity papers, and several follow-ups on circuit identification and feature manipulation. The company has built specific tools (Garcon, Neuronpedia) for SAE-based interpretation and has invested in the engineering required to apply these tools to frontier-scale models.

The motivation is explicit: Anthropic believes that understanding what models are doing internally is essential for ensuring they are safe to deploy. If you cannot inspect a model's reasoning, you cannot detect deception or misalignment. If you can inspect it, you might be able to verify that the model is behaving for the right reasons.

Other organizations have also invested in interpretability research. OpenAI has published work on neuron-level interpretation. DeepMind has explored similar techniques. The academic community has produced significant contributions, particularly through groups at MIT, Harvard, and Berkeley. But Anthropic is the most visible commercial sponsor and has pushed the techniques the furthest at frontier scale.

## 7. What Has Been Learned

Several findings have emerged from mechanistic interpretability work that have changed how researchers think about language models:

**Models represent abstract concepts, not just surface features.** The features identified in SAEs include high-level abstractions (concepts like "deception," "uncertainty," "moral reasoning") rather than just shallow patterns (specific words, syntax structures). This suggests that models develop genuine semantic representations during training.

**Computation is highly distributed.** Even simple-looking model behaviors involve many features and many layers working together. There is rarely a single "neuron" responsible for a specific behavior; the responsibility is spread across the network.

**Models contain features for things they shouldn't.** Some features identified in frontier models correspond to deceptive behaviors, manipulation tactics, or unsafe content. The features are present even when the behaviors are not actively expressed, suggesting that the model has learned representations of these concepts even if it has been trained not to use them.

**Some computations are surprisingly efficient.** The algorithms identified in toy models often use clever tricks that the researchers did not expect. The model sometimes discovers solutions that are more efficient than what human programmers would write.

**Other computations are surprisingly inefficient.** The same models also waste large amounts of computation on tasks that could be done more simply. There is no consistent story about whether neural networks are "smart" or "dumb" at the algorithmic level — they are both, depending on the task.

## 8. The Limits and Skepticism

Despite the progress, mechanistic interpretability has serious limits. The most important is scale. Identifying circuits in a 100-million-parameter model is hard but tractable. Identifying them in a 100-billion-parameter model is much harder. The number of potential features grows with model size, and the techniques that work at small scale do not always extend cleanly.

A related issue is completeness. Identifying that a circuit exists for one behavior does not mean you have understood the model. The model has many behaviors, and characterizing all of them at the circuit level is enormously expensive. In practice, mechanistic interpretability provides spotlights on specific aspects of model behavior, not a complete map.

Some skeptics go further. They argue that the features identified by SAEs are artifacts of the analysis method rather than real internal structure of the model. The argument is that the SAE imposes a sparse structure on the activations that may not actually exist in the underlying computation. The resulting features are "interpretable" in the sense that they can be labeled, but they may not correspond to anything the model is "really" doing.

This is a live debate. The empirical evidence (intervention experiments, prediction of behavior from features, robustness across different SAE training runs) supports the view that features are more than artifacts. But the theoretical basis for why SAE features are meaningful is not airtight.

## 9. Practical Applications

Beyond the basic research, mechanistic interpretability has begun to inform practical applications:

**Model debugging.** When a model produces unexpected behavior, identifying the features and circuits responsible can help diagnose the issue. This is particularly useful for fine-tuning regressions, where a model that previously worked well starts behaving badly after fine-tuning.

**Safety verification.** Identifying features for deceptive or unsafe behaviors and monitoring whether they activate during inference provides a form of runtime safety check. This is being explored as a complement to (not replacement for) other safety techniques.

**Targeted interventions.** Suppressing specific features can change model behavior in controlled ways. This has been demonstrated for removing certain biases, eliminating certain kinds of errors, and modifying tone.

**Knowledge editing.** Identifying the circuits that store specific facts allows for targeted knowledge updates without retraining. If you can find the circuit that knows "Paris is the capital of France," you can modify it to know "Paris is the capital of Germany" — useful for fictional applications and (more practically) for correcting outdated facts.

These applications are early and limited. Mechanistic interpretability is not yet a mature deployment technology. But the techniques are improving, and the practical benefits are growing alongside the basic research findings.

## 10. The Big Picture

Mechanistic interpretability is one of the most ambitious research programs in modern AI. The goal — to understand what neural networks are actually doing, mechanistically — would have seemed naive a few years ago and is now treated as a serious research direction by the leading AI labs. The progress has been faster than the skeptics predicted, with real findings and real applications emerging from sustained effort.

The progress has also been slower than the optimists predicted. Frontier-scale models remain mostly opaque. The features identified by SAEs cover a small fraction of the model's representational capacity. The circuits that have been characterized cover narrow slices of the model's behavior. The dream of a complete, mechanistic understanding of a frontier model remains far away.

The next several years will determine whether mechanistic interpretability can scale to the models that matter most. If it can, the safety and reliability of AI systems will improve substantially. If it cannot, we will be deploying increasingly powerful systems whose internals remain mysterious — a situation that is, at minimum, uncomfortable.

For practitioners, the relevance depends on the use case. If you are building systems that need verifiable behavior or that handle high-stakes decisions, the interpretability literature is worth following. The techniques are not yet plug-and-play, but the trajectory is toward usable tools, and the early adopters will have advantages. For most application developers, mechanistic interpretability remains a research topic with limited immediate practical relevance — but its outcomes will shape the broader trustworthiness of the AI systems everyone is building on top of.

## 11. Conclusion

Mechanistic interpretability is the most promising approach to understanding what neural networks actually do, and its progress has been faster than many expected. Sparse autoencoders, induction heads, and circuit analysis have all produced real findings that change how researchers think about language models. The techniques are not yet adequate for frontier-scale model understanding, but they are improving, and the practical applications are beginning to emerge.

The deeper significance is the implicit bet on scientific understanding as a basis for AI safety. Other safety approaches — RLHF, constitutional AI, content filtering — work without requiring understanding of the model's internals. Mechanistic interpretability is different: it bets that understanding is achievable, that it will be useful, and that it will provide guarantees that black-box approaches cannot. This is a high-risk bet, but the potential payoff is correspondingly large. If we can actually understand what models are doing, we can build safer, more reliable, and more trustworthy AI systems. The pursuit is worth taking seriously, and the early results suggest that the bet is at least not obviously wrong.
