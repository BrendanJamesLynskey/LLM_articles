# LLM Observability: Tracing, Evaluation, and Production Monitoring

*April 2026 • Technical Report*

## 1. Introduction

When a traditional web service misbehaves, operators have a well-developed toolkit. Logs describe what happened, metrics describe how often, traces show how requests flowed across services, and dashboards aggregate the whole picture. Decades of investment in observability tooling — Prometheus, Grafana, Jaeger, Datadog, the ELK stack — make it possible to answer questions like "why did the checkout flow get slower this afternoon?" with reasonable speed. The methods are mature, the tooling is mature, and the discipline of observability engineering has well-known best practices.

LLM applications break this comfortable pattern. The "service" is a model with non-deterministic outputs, opaque internal reasoning, and a quality dimension that cannot be reduced to status codes and latencies. A response that returns HTTP 200 in 800 milliseconds may still be wrong, biased, hallucinated, or off-topic. A small change in the system prompt can silently degrade quality across an entire class of inputs. A new model version released by the upstream provider can break carefully-tuned applications without warning. The conventional observability stack tells you nothing about these problems.

LLM observability is the discipline of building tooling to handle this gap. It encompasses tracing (recording every prompt, response, and tool call in detail), evaluation (scoring outputs against quality criteria), drift detection (noticing when behavior changes over time), and the dashboards and alerts that put it all together. This report examines the techniques, tools, and operational patterns that have emerged for monitoring LLM applications in production.

## 2. The Problem Space

### 2.1 Why Conventional Observability Falls Short

Standard application monitoring is built around a few simple signals: request rate, error rate, duration, and resource utilization (the "four golden signals"). These are sufficient for most services because the definition of "working correctly" is binary or near-binary — the response either succeeded or failed.

LLM applications have a much more complex notion of working correctly. A response can:
- Succeed technically but be factually wrong
- Be factually correct but unhelpful
- Be helpful for the right reasons but use the wrong tone
- Be helpful for the wrong reasons (lucky guess based on training data)
- Be a hallucination that the user has no easy way to detect
- Leak sensitive information from the prompt
- Be jailbroken by adversarial input
- Refuse a legitimate request because of overly-cautious safety training
- Fail to call a required tool, or call the wrong tool

None of these failure modes show up in standard metrics. A 200 OK response containing a confidently-stated falsehood is, from an HTTP perspective, exactly the same as a correct response. Operators need entirely new signals to detect these problems.

### 2.2 The Cost Dimension

A second pressure on LLM observability is cost. LLM API calls are expensive — often the largest line item in a modern AI application's infrastructure budget. Tracking which features, users, and prompts consume the most tokens is essential for cost management and capacity planning. Conventional service metrics don't capture this either; they tell you that requests are being made, not how much they cost.

A typical LLM observability dashboard tracks token consumption at the level of individual users, individual features, individual prompts, and individual tools, with breakdowns by model. This level of detail is not standard in conventional service monitoring but is essential for LLM applications.

## 3. Tracing LLM Applications

### 3.1 The Trace as the Fundamental Unit

The fundamental unit of LLM observability is the trace: a complete record of an LLM interaction including the input prompt, the system prompt, the model's response, any tool calls and their results, the model identifier and parameters, the latency, and the token counts. A trace is to LLM applications what a request log is to web applications, but with much richer content.

A typical trace for a single user interaction might contain:

- The original user message
- The system prompt and any context injected from RAG
- The conversation history up to this point
- The model's internal reasoning (if visible)
- Each tool call the model made, with arguments and results
- The final response shown to the user
- Latency breakdowns for prefill, decode, and tool execution
- Token counts (input, output, cached, total)
- Cost in dollars
- Any user feedback (thumbs up/down, follow-up message)

For complex agent workflows, a single user interaction may produce hundreds of LLM calls and tool calls, all of which need to be captured and connected into a unified trace. The trace format must support this hierarchical structure.

### 3.2 OpenTelemetry and the Semantic Conventions

OpenTelemetry has emerged as the standard for instrumenting LLM applications, just as it has for general distributed services. The OTel community has defined semantic conventions for GenAI workloads that specify standard attribute names for prompts, responses, model identifiers, token counts, and other LLM-specific data. Most major LLM observability platforms accept OTel-format traces, which means that an application instrumented once can export to multiple observability backends.

The standard attributes include:
- `gen_ai.system`: The provider (e.g., "openai", "anthropic")
- `gen_ai.request.model`: The model identifier
- `gen_ai.usage.input_tokens` / `gen_ai.usage.output_tokens`: Token counts
- `gen_ai.response.finish_reason`: Why generation stopped
- `gen_ai.prompt` / `gen_ai.completion`: The actual content (typically gated by privacy settings)

By using these standard names, applications can switch between observability vendors without re-instrumentation, and observability platforms can present a consistent view across applications written by different teams.

### 3.3 Sampling and Storage

Storing every trace in full fidelity is expensive — a typical LLM application might produce gigabytes per day of trace data, dominated by the verbose prompt and response content. Sampling strategies are essential for keeping storage costs manageable.

Common sampling strategies include:
- **Head-based sampling**: Decide at the start of each trace whether to keep it, based on a fixed percentage (e.g., 10% of all traces) or on attributes like the user identity (e.g., always sample admin users)
- **Tail-based sampling**: Decide at the end of each trace based on the outcome — always keep traces with errors, slow latencies, or low quality scores; sample successful traces at lower rates
- **Adaptive sampling**: Adjust the sampling rate based on traffic levels to maintain a constant trace volume

Most production LLM applications use a combination: 100% sampling of error and slow traces, 10% sampling of normal traces, and full sampling for any trace marked as "interesting" (flagged by user feedback, quality scores, or specific user segments).

## 4. LLM Evaluation

### 4.1 Evaluation as Continuous Monitoring

Traditional software has tests that run in CI and validate that the code is correct before deployment. LLM applications have a similar concept — evaluation suites — but the role is different. Because the model is non-deterministic and the input distribution evolves over time, evaluation cannot be a one-time check. It must run continuously in production, sampling real traffic and scoring it against quality criteria.

The continuous evaluation model looks like:

1. Application traffic generates traces in real time
2. A sampling component selects traces for evaluation (e.g., 5% of all traces)
3. Each selected trace is scored along several dimensions (correctness, helpfulness, safety, format compliance)
4. The scores are stored alongside the traces for analysis
5. Aggregate quality metrics are computed and displayed on dashboards
6. Alerts fire when quality metrics drop below thresholds

This makes evaluation a live signal rather than a pre-deployment check, which is necessary because LLM behavior can shift in production for reasons that have nothing to do with code changes — input distribution drift, upstream model updates, prompt template changes, or accumulated context bloat.

### 4.2 LLM-as-Judge

The dominant approach to scoring LLM outputs is to use another LLM as the judge. The judge LLM receives the original input, the model's response, and a rubric, and returns a score along with an explanation. Judge models are typically GPT-4 class or better, on the assumption that a stronger model can reliably evaluate a weaker one (and that even imperfect judgments at scale are useful).

LLM-as-judge has well-known issues. Judges can be fooled by stylistic features (verbose responses score higher than concise ones), can be biased toward responses that resemble their own training distribution, and can be inconsistent across evaluations of the same input. Best practices include:
- Asking the judge to provide reasoning before the score, which improves consistency
- Using multiple judges and aggregating their scores
- Calibrating judge scores against human ratings on a sample of traces
- Designing rubrics that focus on specific, measurable criteria rather than vague quality

Despite these issues, LLM-as-judge is the most scalable evaluation approach available, and it works well enough in practice for most production use cases.

### 4.3 Heuristic and Programmatic Evaluation

For specific quality criteria, heuristic checks are cheaper and more reliable than LLM judges. Examples include:
- **Format compliance**: Does the output parse as valid JSON when it's supposed to be JSON?
- **Required fields**: Does the response include all the fields the user asked for?
- **Forbidden content**: Does the response avoid mentioning competitors, internal codenames, or other forbidden terms?
- **Length**: Is the response within an acceptable length range?
- **Citations**: Does the response include source citations when retrieving from a knowledge base?
- **Hallucination indicators**: Does the response include claims that don't appear in the retrieved documents?

These checks are deterministic, fast, and free. They should always be the first line of evaluation, with LLM judges reserved for the harder questions that heuristics can't answer.

### 4.4 Human Feedback and Annotation

For the highest-stakes evaluation, human review remains the gold standard. Production LLM applications typically include:
- **Implicit feedback**: User behavior signals like "did the user click again," "did they re-prompt," "did they accept the suggestion"
- **Explicit feedback**: Thumbs up/down, star ratings, or detailed review prompts after specific interactions
- **Annotation queues**: A subset of traces routed to human annotators who score them against rubrics, providing ground-truth labels for the LLM judges to be calibrated against

The volume of human annotation is necessarily limited by cost, so it must be focused on the most informative traces — typically the ones where automated evaluation flags potential problems, where users left negative feedback, or where the trace falls into a specific high-priority category (legal advice, medical information, financial transactions).

## 5. Drift Detection

### 5.1 Why LLMs Drift

A traditional service that has been running unchanged for months should produce identical outputs for identical inputs (modulo time-varying things like database state). LLM services can drift even when the application code is unchanged:

- **Provider model updates**: OpenAI, Anthropic, and Google periodically update their hosted models without changing the API model name. A "gpt-4o" call in January may behave differently than the same call in June.
- **Prompt template drift**: As applications add features, prompts grow and change. Small tweaks to the system prompt can have outsized effects on behavior across the entire input distribution.
- **Input distribution drift**: User behavior changes over time as the user base grows and evolves. A response that was perfect for early users may be inappropriate for the current user base.
- **Context bloat**: Conversational applications accumulate more context per turn over time, which can degrade quality in subtle ways.
- **Tool ecosystem changes**: Adding or removing tools changes the model's behavior, sometimes in unexpected ways.

Detecting these drifts requires monitoring quality metrics over time and alerting on regressions.

### 5.2 Statistical Drift Detection

The simplest drift detection is statistical: compute aggregate quality metrics over a rolling window and alert when the metric falls outside a confidence interval based on historical data. For example, if the LLM judge gives an average score of 4.2/5 over the past month, and the score for the past day drops to 3.8/5, that is a drift signal worth investigating.

More sophisticated approaches use change-point detection algorithms (CUSUM, Bayesian online change-point detection) to identify the exact moment when behavior shifted. These algorithms can detect even small drifts that would not trigger threshold-based alerts but represent genuine regressions.

### 5.3 Distributional Comparison

A complementary approach is to compare the distribution of model outputs over time. If the model used to produce responses with median length 200 tokens, and now produces responses with median length 350 tokens, something has changed even if quality scores are stable. The same applies to topic distributions, response formats, and tool call patterns.

Some observability tools track these distributional metrics automatically, computing histograms of various output features and flagging significant changes from historical baselines.

## 6. Prompt and Version Management

### 6.1 Prompts as First-Class Artifacts

In a mature LLM application, prompts are first-class artifacts subject to versioning, review, and rollback. They are too important to live as string literals scattered across the codebase. Mature observability platforms include prompt registries that:

- Store every version of every prompt with timestamps and diff history
- Associate each LLM call in production with the specific prompt version it used
- Allow rollback to a previous prompt version with a single click
- Enable A/B testing of prompt variants in production

This is critical because prompt changes are the most common cause of LLM application regressions, and being able to attribute a quality drop to a specific prompt change is essential for diagnosis.

### 6.2 Model Version Pinning

The corresponding question for models is whether to pin to a specific version (e.g., "gpt-4-0613" rather than "gpt-4") or accept the provider's automatic updates. Pinning provides stability at the cost of missing improvements. Auto-updating provides improvements at the cost of unexpected regressions.

Most production applications pin to specific versions and explicitly upgrade when they have time to test the new version. The observability stack helps with this by making it easy to compare quality metrics across model versions during the upgrade evaluation.

## 7. The Tool Landscape

A vibrant ecosystem of LLM observability tools emerged in 2023-2025. The major players include:

### 7.1 Specialized LLM Observability Platforms

- **LangSmith**: The observability platform from the LangChain team, with deep integration into LangChain and LangGraph applications. Provides tracing, evaluation, prompt management, and dataset curation.
- **Langfuse**: An open-source alternative to LangSmith with similar features. Self-hostable, which is appealing for organizations with data residency requirements.
- **Arize Phoenix**: Open-source LLM observability with strong evaluation tooling. Integrates with Arize's broader ML observability platform.
- **Helicone**: Proxy-based observability that requires no code changes — applications point at the Helicone proxy instead of directly at the LLM API, and Helicone records all the traffic.
- **Weights & Biases Weave**: Built on top of W&B's experiment tracking, with strong support for evaluations and training-time observability that extends into production.

### 7.2 General APM Platforms with LLM Support

- **Datadog**: Added GenAI monitoring features in 2024, including prompt/response tracking and token cost metrics. Useful for organizations already on Datadog.
- **New Relic, Honeycomb, Lightstep**: Similar story — adding LLM-specific support to their existing APM platforms.

### 7.3 OpenTelemetry-Native Approaches

For organizations preferring open standards, the OpenTelemetry ecosystem provides:
- Auto-instrumentation libraries for major LLM SDKs
- Collectors that route OTel data to any compatible backend
- Open-source frontends like SigNoz and Jaeger that can visualize LLM traces

The choice between specialized LLM platforms and general APM tools depends on the organization's existing tooling, the depth of LLM-specific features needed, and the integration with the rest of the infrastructure stack.

## 8. Cost Tracking and Optimization

LLM costs deserve their own observability category because they can dwarf other infrastructure costs. A single GPT-4 call can cost more than a thousand database queries. Effective cost observability requires:

- **Per-feature attribution**: Which features in the application consume which fraction of the total token budget?
- **Per-user attribution**: Which users (or which user tiers) account for the most cost?
- **Trend analysis**: Is cost growing linearly with usage or super-linearly (suggesting context bloat or runaway agent loops)?
- **Outlier detection**: Are there individual requests that consume 100× more tokens than normal?

The observability platforms listed above all provide some form of cost tracking, with varying levels of granularity. For organizations spending significant amounts on LLM APIs, the cost dashboard is often the most-watched panel in the observability stack.

## 9. Privacy and Compliance Considerations

LLM observability creates a new privacy concern: prompts and responses often contain sensitive user data. A trace that captures every input/output pair is essentially a copy of every conversation users have had with the application. This data must be handled with the same care as any other PII.

Best practices include:
- **PII redaction**: Automatically detect and mask PII in traces (names, emails, phone numbers, credit card numbers) before storing them
- **Tokenization**: For applications with regulated data, tokenize the prompts before sending them to observability platforms, then de-tokenize only when authorized
- **Access control**: Restrict access to trace data on a need-to-know basis, with audit logging of who viewed which traces
- **Retention policies**: Automatically delete trace data after a defined retention period (often 30-90 days)
- **Geographic restrictions**: Ensure trace data is stored in the same region as the user data, in compliance with GDPR, CCPA, and other regulations

Several observability platforms now offer "PII-aware" modes that automatically apply redaction and access control.

## 10. Conclusion

LLM observability is a young discipline that has evolved rapidly from "nice to have" to "essential for production." The combination of non-deterministic outputs, complex quality dimensions, drifting behavior, and high cost makes LLM applications fundamentally harder to operate than conventional services, and conventional observability tools are inadequate to the task.

The good news is that the tooling ecosystem has matured quickly. OpenTelemetry's GenAI semantic conventions provide a standard data model, multiple commercial and open-source platforms offer comprehensive observability features, and best practices for evaluation, drift detection, and cost tracking are stabilizing across the industry. A team starting an LLM application today has access to dramatically better tooling than was available even two years ago.

The remaining challenges are organizational and methodological rather than technical. Teams must invest in evaluation rubrics, build the human annotation pipelines that calibrate automated judges, develop on-call runbooks for LLM-specific incidents, and integrate LLM quality metrics into their broader service health discussions. The organizations that do this well will operate LLM applications with the same rigor they bring to traditional services. The ones that don't will be perpetually surprised by silent quality regressions, runaway costs, and angry users.

For practitioners, the practical advice is to start early. Instrument your LLM application from day one with at least basic tracing and cost tracking. Add evaluation as soon as you have a coherent definition of quality. Iterate from there. The cost of retrofitting observability onto an LLM application that has been running blind for months is much higher than the cost of building it in from the start.
