# Structured Output and Constrained Decoding

*March 2026*

## 1. The Problem: Free-Form Text in a Structured World

Large language models are, at their core, next-token predictors over natural language. They excel at producing fluent, contextually appropriate prose, but the systems that consume their output rarely want prose. API backends expect JSON. Code generators need syntactically valid programs. Tool-using agents require function calls with precisely typed arguments. The gap between what an LLM naturally produces and what a downstream system can parse has become one of the most practically important problems in applied ML engineering.

Prompt engineering alone offers no guarantees. You can instruct a model to "respond only in valid JSON," and it will comply most of the time, but "most of the time" is insufficient when a malformed response means a 500 error in production. Even highly capable models occasionally emit trailing commas, unquoted keys, or markdown fences wrapping their JSON. The need for deterministic structural correctness, not probabilistic compliance, is what motivates the family of techniques collectively known as constrained decoding.

## 2. JSON Mode and Schema-Constrained Generation

The simplest form of structured output is JSON mode, now offered by most major API providers. When enabled, the model's generation is post-processed or lightly constrained to ensure the output parses as valid JSON. This is useful but limited: it guarantees syntactic validity without enforcing any particular schema. A response might be valid JSON yet contain the wrong field names, incorrect types, or missing required properties.

Schema-constrained generation goes further. Given a JSON Schema, the decoding process ensures that every generated token is consistent with the schema at every step. This means the model cannot produce an integer where a string is expected, cannot omit required fields, and cannot introduce properties not permitted by the schema. The result is output that is both syntactically valid and semantically conformant, by construction rather than by hope.

## 3. Grammar-Based Decoding: CFGs and PDA-Guided Token Masking

The most general approach to constrained decoding uses formal grammars. A context-free grammar (CFG) can describe the syntax of JSON, SQL, Python, or any other structured language. At each decoding step, the system maintains a pushdown automaton (PDA) that tracks the current parse state. Before the model's softmax distribution is sampled, tokens that would lead to an invalid parse state are masked out, their logits set to negative infinity. The model can only choose among tokens that keep the output within the grammar.

This approach is powerful but comes with subtleties. The token vocabulary of a language model does not align neatly with the terminal symbols of a grammar. A single token might span multiple grammar productions, or a grammar terminal might require multiple tokens to express. Efficient implementations precompute token masks for each grammar state, amortizing the cost of intersection between the token vocabulary and the set of valid continuations. Libraries such as llama.cpp's GBNF grammar support and the grammar engine in vLLM implement this strategy.

## 4. Finite-State Machine Approaches

When the target language is regular rather than context-free, finite-state machines (FSMs) offer a simpler and faster alternative to full PDA-based decoding. Regex-constrained generation is the canonical example: given a regular expression describing the desired output format, the system compiles it into a deterministic finite automaton (DFA), then masks tokens at each step according to the DFA's current state and transition function.

The Outlines library pioneered this approach, demonstrating that regex-constrained generation could be implemented efficiently by precomputing, for each DFA state, the set of vocabulary tokens whose string representation would drive the automaton into a valid successor state. This precomputation produces an index that makes per-token masking nearly free at inference time. The technique works well for formats like dates, phone numbers, enumerations, and simple structured fields where a regular expression suffices.

## 5. Libraries and Tooling: Outlines, Guidance, and lm-format-enforcer

Several open-source libraries have emerged to make constrained decoding accessible. Outlines provides regex and CFG-based constrained generation with deep integration into the Hugging Face ecosystem. Guidance from Microsoft offers a template-based approach where users interleave literal text with generation directives, enabling fine-grained control over which parts of the output are fixed and which are model-generated. The lm-format-enforcer library takes a JSON Schema as input and builds the appropriate token masks automatically, targeting ease of use for the common case of structured JSON output.

These libraries differ in their abstraction level and performance characteristics, but they share a common principle: intervene at the logit level during decoding, not after the fact. This is fundamentally more reliable than post-hoc parsing and retry loops, because it makes invalid output impossible rather than merely unlikely.

## 6. Token Healing and Logit Processors

Token healing addresses a subtle problem that arises when constrained decoding forces a particular prefix. Because BPE tokenization is context-dependent, inserting a forced string at a generation boundary can produce a token sequence that the model would never have generated naturally, degrading output quality. Token healing backs up by one token and re-decodes, allowing the model to "heal" the boundary between the forced prefix and the generated continuation into a natural token sequence.

More broadly, logit processors provide the general mechanism for constrained decoding. A logit processor is a function that modifies the raw logit vector before sampling. Logit bias, where specific token logits are increased or decreased by fixed amounts, is the crudest form. Grammar-based masks, regex DFA masks, and schema-derived masks are all implemented as logit processors. This clean abstraction has enabled frameworks to compose multiple constraints and integrate them into existing inference pipelines with minimal architectural changes.

## 7. Function Calling as Structured Output

Function calling, as implemented by OpenAI, Anthropic, and others, can be understood as a specialized form of structured output. The model is presented with function signatures, including parameter names, types, and descriptions, and is trained or prompted to emit structured invocations. Under the hood, many providers apply constrained decoding to ensure the emitted arguments conform to the function's schema.

This framing unifies what might otherwise seem like separate features. Whether you need a JSON object matching a schema, a SQL query conforming to a grammar, or a function call with typed arguments, the underlying mechanism is the same: guide the model's token selection so that the output is guaranteed to inhabit a particular formal language.

## 8. Performance Implications

Constrained decoding is not free. At each generation step, the system must compute which tokens are valid given the current grammar or automaton state, then apply the mask before sampling. For complex grammars, this per-step overhead can be significant. Precomputation helps enormously, turning what would be a per-token grammar intersection into a table lookup, but the precomputation itself can be expensive for large vocabularies and complex grammars, and the resulting index consumes memory.

The interaction with speculative decoding deserves particular attention. Speculative decoding accelerates inference by drafting multiple tokens with a small model, then verifying them in a single forward pass of the large model. When constrained decoding is active, the draft model must also respect the constraints, and rejected speculative tokens that would have been valid unconstrained might now be rejected for grammar violations, potentially reducing the acceptance rate and negating some of the speedup. Implementations in vLLM and SGLang have developed strategies to mitigate this, including constraint-aware draft selection.

## 9. Reliability vs. Prompt-Only Approaches

The case for constrained decoding rests on a simple empirical observation: prompt-only approaches to structured output are unreliable at scale. Even with few-shot examples, system prompts, and carefully tuned instructions, models produce malformed output at a non-trivial rate. In high-throughput production systems processing millions of requests, even a 0.1% failure rate translates into thousands of errors per day. Retry loops add latency and cost. Post-hoc repair heuristics are fragile and format-specific.

Constrained decoding eliminates this entire failure mode. The output is structurally valid by construction. This shifts the reliability concern from "does the output parse?" to "does the output contain the right content?", a question that constrained decoding cannot answer but that is strictly easier to address when structural correctness is guaranteed.

## 10. Framework Support

The major open-source inference frameworks have all adopted constrained decoding as a core feature. vLLM supports JSON Schema-based constrained generation through its integration with Outlines and xgrammar, exposing it via the OpenAI-compatible API's `response_format` parameter. SGLang provides grammar-based decoding with its own optimized FSM engine, achieving particularly strong performance through aggressive precomputation and caching. TensorRT-LLM offers logit processor hooks that enable constrained decoding with custom grammars, though with somewhat more manual integration work.

On the proprietary side, OpenAI's structured outputs feature guarantees schema conformance for JSON generation, and Anthropic has introduced similar capabilities for tool use. The convergence across both open and closed ecosystems signals that constrained decoding has moved from research novelty to production necessity.

## 11. Practical Patterns

Several patterns have emerged as best practices. First, prefer schema-constrained generation over prompt-only instructions whenever the output format is known at request time. Second, use the narrowest constraint that captures your requirements: a regex for simple formats, a JSON Schema for structured data, a full CFG only when the target language demands it. Third, keep schemas simple. Deeply nested optional fields and complex union types increase the constraint-checking overhead and can confuse the model's content generation even when the structure is guaranteed. Fourth, test constrained outputs for semantic correctness, not just structural validity. A perfectly formatted JSON object with hallucinated field values is still wrong.

## 12. Conclusion

Structured output and constrained decoding represent a maturing of LLM deployment practices. The early era of hoping the model would follow formatting instructions is giving way to a more rigorous approach grounded in formal language theory. By intervening at the logit level with grammars, automata, and schema-derived masks, we can guarantee structural properties of model output without sacrificing the flexibility and fluency that make language models useful in the first place. As inference frameworks continue to optimize these techniques and reduce their overhead, constrained decoding is becoming not an optional enhancement but a default mode of operation for any production system that consumes LLM output programmatically.
