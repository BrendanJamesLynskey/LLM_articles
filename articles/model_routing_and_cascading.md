# Model Routing and Cascading

*April 2026*

## 1. Introduction

Model routing is the practice of directing different queries to different language models based on the characteristics of each query. The core insight is that not all queries require the same level of model capability. A simple factual question, a greeting, or a straightforward summarization task can be handled by a small, cheap model just as well as by a large, expensive one. A complex reasoning problem, a nuanced analysis, or a multi-step coding task requires the capabilities of a larger model. By routing each query to the most cost-effective model that can handle it at the required quality level, an organization can reduce inference costs by 30-70% while maintaining—or even improving—overall quality.

Cascading is a related but distinct pattern: a query is first processed by a cheap model, and if the result is deemed insufficient (by a confidence check, a quality classifier, or another heuristic), the query is escalated to a more capable (and expensive) model. Routing decides where a query goes before any model processes it. Cascading processes the query with a cheap model first and escalates only when needed. Both patterns exploit the same fundamental observation—that query difficulty varies widely—but they differ in latency characteristics, cost profiles, and implementation complexity.

This report provides a comprehensive technical examination of model routing and cascading. It covers the architectural patterns, routing algorithms, production systems, and practical considerations for implementing these strategies. The intended audience is engineers and architects designing LLM-powered systems that must balance cost, quality, and latency at scale.

## 2. Why Routing Matters

### 2.1 The Query Difficulty Distribution

Empirical analysis of production LLM workloads consistently shows that query difficulty follows a long-tailed distribution. In a typical customer support application, 60-80% of queries are routine (greetings, simple lookups, FAQ answers) and can be handled by a small model. 15-30% are moderate (multi-turn conversations, nuanced policy questions) and benefit from a mid-tier model. 5-10% are complex (escalations, edge cases, complaints requiring careful reasoning) and genuinely require a large model's capabilities.

Without routing, all queries go to the same model. If that model is the large, expensive one, the organization overpays for the 60-80% of routine queries. If it is the small, cheap one, the 5-10% of complex queries receive inadequate responses. Routing allows the organization to use the right model for each query, optimizing the cost-quality trade-off across the entire distribution.

### 2.2 The Cost Multiplier

The cost difference between model tiers is substantial. As of early 2026, representative per-million-token output costs are:

- Small models (Haiku-class, GPT-4o-mini): $0.10-$1.25
- Mid-tier models (Sonnet-class, GPT-4o): $1.50-$15
- Large models (Opus-class, GPT-4.5): $8-$75
- Reasoning models (o3, extended thinking): $40-$100+

The ratio between the cheapest and most expensive tier is 50-1000x. Even a modest routing strategy that diverts 50% of queries to a small model saves 25-50% of total inference cost, assuming the routed queries maintain quality.

### 2.3 Quality Equivalence on Easy Queries

The economic case for routing rests on the observation that small models achieve equivalent quality to large models on easy queries. This is well-supported empirically. On simple factual questions, basic summarization, format conversion, and straightforward instruction following, the quality gap between a 7B model and a 400B model is often negligible—both produce correct, well-formatted responses.

The gap widens on tasks requiring complex reasoning, nuanced judgment, extensive world knowledge, or handling of edge cases. Routing exploits this variable gap: route to the small model when the gap is negligible, and to the large model when the gap is significant.

## 3. Routing Architectures

### 3.1 Classifier-Based Routing

The simplest routing architecture uses a trained classifier to predict the difficulty of each query and route accordingly. The classifier can be:

**A traditional ML model.** A logistic regression, gradient-boosted tree, or small neural network trained on features extracted from the query (length, vocabulary complexity, topic, presence of specific keywords). These classifiers are extremely fast (microsecond-level inference) and cheap to run, adding negligible overhead to the routing decision.

**A small language model.** An embedding model (like a sentence transformer) that maps the query to a vector, followed by a classification head that predicts difficulty. This captures more semantic information than simple features but is more expensive to run (~1-10ms).

**A fine-tuned classification model.** A small LM (1-3B parameters) fine-tuned specifically to predict query difficulty. This provides the best routing accuracy but adds meaningful latency and cost to each query.

The training data for the classifier typically comes from running the same queries through both the small and large models, comparing quality scores, and labeling queries where the small model matches the large model as "easy" and queries where it does not as "hard." This requires a quality evaluation mechanism (human annotation, automated metrics, or LLM-as-judge).

### 3.2 Confidence-Based Routing

Rather than predicting difficulty before any model runs, confidence-based routing sends the query to the small model first and uses the model's own confidence to decide whether to escalate. If the small model is confident in its answer, the response is returned. If the model is uncertain, the query is escalated to the larger model.

Confidence can be estimated from:

**Token-level log-probabilities.** The mean or minimum log-probability of the generated tokens. Low probabilities indicate that the model is uncertain about its generation. Most inference APIs can return log-probabilities, making this approach easy to implement.

**Entropy of the output distribution.** High entropy at any generation step indicates uncertainty. This requires access to the full output distribution, which is available in self-hosted deployments but not always through APIs.

**Consistency checking.** Generate multiple responses (e.g., 3-5) at temperature > 0 and check consistency. If all responses agree, the model is confident. If they disagree, escalate. This is more expensive (3-5x the cost of a single generation) but provides a reliable confidence signal.

**Self-reported confidence.** Ask the model to rate its confidence in its answer. This is unreliable for calibration (models are often overconfident) but can be useful as a relative signal, especially when combined with other measures.

### 3.3 LLM-as-Judge Routing

A more sophisticated approach uses a language model to evaluate whether the small model's response is adequate. The flow is:

1. Send query to the small model.
2. Send the query and the small model's response to a judge model.
3. If the judge approves, return the small model's response.
4. If the judge rejects, send the query to the large model.

The judge model can be the large model itself (in which case you pay for a judgment call instead of a full response, which is cheaper if the judgment prompt is shorter than the full response), a specialized evaluation model, or even a fine-tuned small model trained specifically for quality judgment.

The trade-off is that the judge adds latency and cost. If the judge is expensive and rejects frequently, the total cost can exceed simply running the large model on all queries. The approach is most effective when the judge is cheap and accurate, and when the small model handles most queries correctly.

### 3.4 Semantic Routing

Semantic routing uses the semantic content of the query—its topic, intent, complexity, and domain—to select the model, rather than a generic difficulty score. This is particularly useful when different models have different strengths:

**Topic-based routing.** A coding question might be routed to a code-specialized model; a creative writing request to a model optimized for creative tasks; a factual question to a model with strong knowledge retrieval.

**Intent-based routing.** A classification model identifies the user's intent (information retrieval, task execution, conversation, creative generation, analysis) and routes to the model best suited for that intent type.

**Complexity-based routing with domain awareness.** Simple medical questions go to a small general model; complex medical questions go to a medical-domain-specialized model or a large general model.

Semantic routing can be implemented using embedding-based similarity (comparing the query embedding to centroids representing each model's area of strength) or using explicit intent classification.

## 4. Cascading Patterns

### 4.1 Simple Cascading

The simplest cascading pattern is:

1. Send query to Model A (cheap).
2. If Model A's response passes a quality check, return it.
3. Otherwise, send query to Model B (expensive), and return its response.

The quality check can be any of the mechanisms described in Section 3.2 (confidence, consistency, judge). The key parameter is the escalation rate—the fraction of queries that are forwarded to Model B. If the escalation rate is too high, cascading provides little cost savings. If it is too low, quality suffers because some queries that should have been escalated are not.

### 4.2 Multi-Stage Cascading

Extending to more than two stages:

1. Model A (cheapest): handles routine queries.
2. Model B (mid-tier): handles queries that A cannot.
3. Model C (most expensive): handles queries that B cannot.

Each stage has an escalation criterion and a quality check. Multi-stage cascading can provide better cost-quality trade-offs than two-stage cascading, especially when the query difficulty distribution has multiple modes. However, the added complexity (more models to manage, more escalation thresholds to tune) means that the incremental benefit of the third stage is typically smaller than the benefit of the first stage.

### 4.3 Cascading with Parallel Verification

Instead of sequential escalation, some systems run multiple models in parallel and use a verification step to select the best response:

1. Send query to Model A and Model B simultaneously.
2. A verifier (rule-based, classifier, or LLM) selects the better response.
3. Return the selected response.

This eliminates the latency penalty of cascading (no sequential waiting) but increases cost (both models run on every query). The approach is cost-effective only when the combined cost of running both models plus the verifier is less than always running the expensive model, which is typically the case when Model A is very cheap and the verifier is efficient.

### 4.4 Speculative Routing

Speculative routing is inspired by speculative decoding. The small model generates a complete response, and the large model verifies (or partially regenerates) the response. If the small model's response is correct, the verification is cheaper than generating from scratch because the large model only needs to confirm rather than create.

In practice, this is implemented by:

1. Small model generates a response.
2. Large model is given the query and the small model's response, and asked to either approve it or generate a replacement.
3. If approved, the large model's cost is just the evaluation (cheaper than full generation). If replaced, the cost is the full generation plus the evaluation overhead.

This pattern is similar to cascading with LLM-as-judge but explicitly frames the large model's task as verification rather than independent generation, which can reduce cost when the small model is usually correct.

## 5. Learned Routers

### 5.1 RouteLLM

RouteLLM (2024) is a framework for training routing models that learn to predict which model should handle a given query. The key innovation is training the router on preference data—human or automated judgments about which model produced a better response for each query—rather than on abstract difficulty labels.

The RouteLLM approach trains several types of routing models:

**Matrix factorization router.** Represents queries and models as vectors in a shared embedding space, and predicts quality based on the inner product. This is fast and effective for routing among a small number of models.

**BERT-based classifier.** A fine-tuned BERT model that takes the query as input and outputs a routing decision. This captures more semantic information than the matrix factorization approach but is more expensive to run.

**Causal LM router.** Uses a small causal language model to predict the routing decision, leveraging the same architecture as the models being routed to.

RouteLLM demonstrated that trained routers can achieve significant cost savings (50-70%) at minimal quality loss (1-3% on benchmarks) compared to always using the large model. The key finding is that router accuracy does not need to be very high to be useful—even a router that is correct 70% of the time provides substantial savings, because the cost of a routing error (sending a hard query to the small model and getting a suboptimal response) is often small relative to the savings on correctly routed easy queries.

### 5.2 Router Training Data

The quality of a learned router depends critically on the training data. The standard pipeline is:

1. **Collect a diverse query set.** Representative of the production distribution.
2. **Run all queries through all candidate models.** Collect responses from each model.
3. **Evaluate response quality.** Using human annotation, automated metrics (BLEU, code execution, factual accuracy), or LLM-as-judge.
4. **Label queries with routing decisions.** For each query, identify the cheapest model that produces an acceptable response.
5. **Train the router.** Using the labeled data, train a classifier that maps queries to model selections.

The evaluation step (step 3) is the most expensive and most important. The quality of the routing labels depends entirely on the quality of the evaluation, and evaluation errors propagate directly into routing errors.

### 5.3 Online Learning and Adaptation

Production routers can benefit from online learning—updating the routing model based on feedback from deployed queries. This is particularly valuable because:

**Query distributions shift over time.** New types of queries emerge, existing patterns change, and the difficulty distribution evolves. A static router trained on historical data will gradually become less effective.

**Model capabilities change.** As models are updated or replaced, the difficulty boundaries shift. A query that required the large model before a model update might be handleable by the small model after the update.

**Feedback signals are available.** User satisfaction ratings, explicit corrections, task completion metrics, and other feedback can serve as labels for updating the router without explicit annotation.

Online router adaptation can be implemented through periodic retraining on recent data, bandit algorithms that explore model assignments, or continual learning methods that update the router incrementally.

## 6. The Cost-Quality Pareto Frontier

### 6.1 Defining the Frontier

For a given set of models, the cost-quality Pareto frontier is the set of routing strategies that achieve the best possible quality for each cost level. No strategy below the frontier can achieve the same quality at lower cost, and no strategy above the frontier can achieve lower cost at the same quality.

The endpoints of the frontier are:
- **Minimum cost, minimum quality:** Route all queries to the cheapest model.
- **Maximum quality, maximum cost:** Route all queries to the most expensive model.

Between these endpoints, routing strategies trace a curve that represents the achievable trade-offs. A good router approaches the Pareto frontier, while a poor router (or no routing at all) falls below it.

### 6.2 Measuring the Frontier

To measure the Pareto frontier empirically:

1. Collect a benchmark set of N queries with ground-truth quality labels for each model.
2. For each possible routing threshold (from 0% escalation to 100% escalation), compute the average quality and average cost.
3. Plot quality vs. cost to obtain the frontier.

The shape of the frontier depends on the query distribution and the capability gap between models. If most queries are easy (and the small model handles them well), the frontier has a long flat region where cost can be reduced substantially with minimal quality loss. If many queries are hard, the frontier is steeper, and cost reduction comes at a greater quality cost.

### 6.3 Operating Point Selection

Choosing where to operate on the Pareto frontier is a business decision. The relevant factors are:

**Quality requirements.** Applications with strict quality requirements (medical, legal, financial) operate near the high-quality end of the frontier. Applications with more tolerance for occasional lower-quality responses (casual chat, brainstorming) can operate further along the cost-reduction axis.

**Budget constraints.** A fixed monthly budget constrains the average cost per query, determining the maximum operating point on the cost axis.

**Latency requirements.** Cascading adds latency when queries are escalated. If latency is critical, the escalation rate must be kept low or parallel strategies must be used.

**User segment differentiation.** Different users may warrant different operating points—premium users get all-large-model routing, free-tier users get aggressive cost optimization.

## 7. Mixture-of-Agents

### 7.1 The MoA Architecture

Mixture-of-agents (MoA) is a related but distinct pattern where multiple models process the same query and their outputs are combined (rather than selecting one). The combination can be:

**Aggregation.** Multiple models generate responses, and a synthesis model combines them into a single response that is better than any individual response. Wang et al. (2024) demonstrated that this approach can exceed the quality of any individual model in the mixture.

**Voting.** Multiple models generate responses, and the most common response is selected. This is particularly effective for tasks with verifiable correct answers (math, factual questions, classification).

**Ranking.** Multiple models generate responses, and a ranker selects the best one. This is similar to best-of-N sampling but across different models rather than samples from the same model.

### 7.2 Cost Considerations

MoA is more expensive than routing because it runs multiple models on every query. The cost is the sum of all model costs plus the aggregation/selection cost. This makes MoA suitable only when the quality improvement justifies the cost—typically for high-value queries where the cost of errors is high.

In practice, MoA is most commonly used for:
- **Evaluation and benchmarking.** Running multiple models and comparing outputs to identify quality issues.
- **High-stakes responses.** Generating multiple candidate responses for important queries and having a human or automated judge select the best one.
- **Self-improvement.** Using the combined output of multiple models as training data for a single model (distillation from the mixture).

## 8. Production Routing Systems

### 8.1 OpenRouter

OpenRouter is an API aggregation platform that provides access to hundreds of models from multiple providers through a unified API. Its routing capabilities include:

**Automatic model selection.** Users can specify a desired capability level or cost budget, and OpenRouter selects the model that best matches the requirements.

**Fallback chains.** If the primary model is unavailable or rate-limited, requests are automatically routed to a fallback model.

**Cost optimization.** OpenRouter tracks pricing across providers and can route to the cheapest provider for a given model.

OpenRouter's value proposition is simplifying multi-model deployment by abstracting away the differences between provider APIs and pricing models. However, it does not implement query-level routing based on difficulty—it routes based on user preferences and provider availability rather than query characteristics.

### 8.2 Martian

Martian is an AI infrastructure company focused specifically on model routing. Its router analyzes each query and selects the model that provides the best cost-quality trade-off for that specific query. The routing decision is based on:

- Query complexity analysis
- Historical performance data for each model on similar queries
- Real-time model availability and latency
- User-specified cost and quality constraints

Martian's approach is closer to the learned router paradigm (Section 5), using historical performance data to train routing models that predict per-query quality for each available model.

### 8.3 Custom Routing in Production

Many organizations implement custom routing rather than using a third-party service. Common patterns:

**Rule-based routing.** The simplest approach: route based on explicit rules (e.g., "route all queries with fewer than 20 tokens to the small model," "route all code generation queries to the code-specialized model"). Easy to implement and debug, but brittle and difficult to optimize.

**Embedding-based routing.** Compute the query embedding, compare it to cluster centroids representing different difficulty levels or topic categories, and route based on the nearest centroid. This provides semantic awareness without the cost of a full classification model.

**A/B testing-informed routing.** Run A/B tests comparing models on different query types, measure quality and cost, and use the results to configure routing rules. This is the most empirical approach and provides the highest confidence in routing decisions, but requires significant traffic and evaluation infrastructure.

## 9. Latency Considerations

### 9.1 Routing Latency

The routing decision itself adds latency to the request. The overhead depends on the routing mechanism:

- **Rule-based routing:** Negligible (<1ms)
- **Feature-based classification:** ~1-5ms
- **Embedding-based routing:** ~5-20ms (depends on embedding model size)
- **LLM-based classification:** ~50-500ms (depends on classifier model size)

For applications where time-to-first-token is critical, the routing overhead must be carefully managed. A rule-based or feature-based router adds negligible latency, while an LLM-based router can add hundreds of milliseconds—potentially longer than the time-to-first-token from the small model itself.

### 9.2 Cascading Latency

Cascading inherently adds latency for escalated queries. In the worst case, a query is processed by the small model, evaluated, and then reprocessed by the large model—consuming time equal to the sum of both models' response times plus the evaluation time. For the queries that are escalated, the latency can be 2-3x higher than routing directly to the large model.

Strategies to mitigate cascading latency:

**Early termination.** If the small model shows signs of uncertainty early in generation (low log-probabilities on the first few tokens), abort and escalate before generating the full response. This reduces wasted compute on the small model.

**Parallel execution with cancellation.** Start both models simultaneously, but cancel the large model's request if the small model's confidence is high. This eliminates the latency penalty for escalated queries but increases cost (the large model starts processing before it is known to be needed) and wastes resources when the small model is sufficient.

**Speculative execution.** Send the query to the small model and speculatively to the large model simultaneously. If the small model's response is adequate, use it and discard the large model's response. If not, use the large model's response, which is already in progress. This minimizes latency at the cost of running both models on every query—economical only when the small model handles most queries and the large model's cost for discarded responses is acceptable.

### 9.3 Latency-Quality-Cost Three-Way Trade-off

Routing introduces a three-way trade-off between latency, quality, and cost. Without routing, there is a two-way trade-off: choose a model that balances quality and cost, and accept its latency. With routing, the routing overhead adds latency, and cascading adds latency for escalated queries, but the cost savings can be substantial.

The optimal strategy depends on the relative importance of the three factors. For real-time applications (conversational chatbots, code completion), latency is critical, and the routing strategy must minimize overhead—prefer pre-routing (classify before generating) with fast classifiers, and avoid cascading if possible. For batch applications (document processing, data analysis), latency is unimportant, and cascading is attractive because it minimizes cost without latency constraints.

## 10. Implementing Routing: Practical Patterns

### 10.1 The Router as Middleware

The most common implementation pattern is to insert the router as middleware between the application and the model APIs:

```
Application → Router Middleware → Model A API
                                → Model B API
                                → Model C API
```

The router middleware:
1. Receives the query from the application.
2. Extracts routing features (query text, metadata, user tier, conversation history).
3. Applies the routing logic (rules, classifier, embedding similarity).
4. Forwards the query to the selected model.
5. Receives the response and returns it to the application.
6. Logs the routing decision and model response for monitoring and training.

This pattern has the advantage of being transparent to the application—the application sends queries to the router as if it were a single model API, and the router handles model selection, failover, and load balancing.

### 10.2 Implementing Cascading

Cascading requires additional logic beyond simple routing:

```python
def cascade(query, models, quality_check):
    for model in models:  # ordered from cheapest to most expensive
        response = model.generate(query)
        if quality_check(query, response):
            return response, model.name
    # If no model passes, return the most expensive model's response
    return response, models[-1].name
```

The quality_check function is the critical component. Common implementations:

**Log-probability threshold:**
```python
def quality_check(query, response):
    avg_logprob = mean(response.token_logprobs)
    return avg_logprob > THRESHOLD
```

**Consistency check:**
```python
def quality_check(query, response):
    responses = [model.generate(query) for _ in range(3)]
    return all_consistent(responses)
```

**LLM judge:**
```python
def quality_check(query, response):
    judgment = judge_model.evaluate(query, response)
    return judgment.score >= THRESHOLD
```

### 10.3 Configuration and Tuning

The routing configuration includes:

**Model registry.** The set of available models, their capabilities, costs, and latency characteristics.

**Routing thresholds.** The difficulty thresholds that determine which model handles each query. These should be tuned on a representative evaluation set to optimize the cost-quality trade-off.

**Fallback rules.** What happens when a model is unavailable, rate-limited, or returns an error. Typically, the query is escalated to the next model in the cascade.

**Monitoring metrics.** The metrics used to evaluate routing quality: average quality score, cost per query, routing distribution (fraction of queries going to each model), escalation rate, and latency.

### 10.4 Evaluation Methodology

Evaluating a routing system requires measuring quality and cost simultaneously:

1. **Collect a labeled evaluation set.** A diverse set of queries with quality ratings for each model's response.
2. **Simulate routing.** Apply the routing strategy to the evaluation set and compute the resulting quality and cost.
3. **Compare to baselines.** Compare against always using the small model, always using the large model, and random routing.
4. **Measure the Pareto gap.** How close is the routing strategy to the theoretical Pareto frontier?

The evaluation should be repeated periodically, as model capabilities, query distributions, and costs all change over time.

## 11. When Routing Helps vs. Hurts

### 11.1 Routing Helps When

**The cost gap between models is large.** If the small model costs 10-50x less than the large model, even modest routing accuracy provides significant savings.

**The query distribution is bimodal or long-tailed.** If most queries are easy (handled well by the small model) and a few are hard (requiring the large model), routing captures most of the savings with minimal quality impact.

**Quality requirements are task-specific.** If the application has well-defined quality criteria (correct code, accurate facts, appropriate tone), it is possible to build reliable quality checks that enable effective cascading.

**The application has sufficient volume.** The engineering investment in building and maintaining a routing system is justified only when the inference cost savings are significant in absolute terms. For low-volume applications, the engineering cost outweighs the inference savings.

### 11.2 Routing Hurts When

**The cost gap is small.** If models are similarly priced, routing adds complexity without significant savings.

**All queries are similarly difficult.** If the query distribution is uniform in difficulty—most queries are moderately hard—there is no "easy" fraction to divert to the small model, and routing provides little benefit.

**Quality requirements are holistic and subjective.** If quality is hard to evaluate automatically (e.g., creative writing, empathetic conversation), building reliable quality checks for routing is difficult, and routing errors are costly.

**Routing errors are expensive.** If sending a hard query to the small model has severe consequences (a wrong medical answer, an incorrect legal opinion, a failed code deployment), the risk of routing errors may outweigh the cost savings.

**Latency is critical and cascading is the only option.** If the application cannot tolerate the additional latency of cascading, and pre-routing is inaccurate, the routing system may degrade the user experience.

### 11.3 The Minimum Viable Router

For organizations evaluating routing, the minimum viable implementation is:

1. **Two models:** One cheap, one expensive.
2. **Rule-based routing:** Simple rules based on query length, detected intent (classification, translation, chat, reasoning), or user tier.
3. **A/B testing:** Measure quality on both models for each query type to validate routing decisions.
4. **Manual threshold tuning:** Adjust routing rules based on A/B test results.

This can be implemented in a day, provides 20-40% cost savings in many applications, and serves as a foundation for more sophisticated routing strategies.

## 12. Advanced Topics

### 12.1 Multi-Objective Routing

In practice, routing must optimize multiple objectives simultaneously:

- **Quality:** Maximize the average quality of responses.
- **Cost:** Minimize the average cost per query.
- **Latency:** Minimize the response time for each query.
- **Fairness:** Ensure consistent quality across user segments, query types, and topics.
- **Privacy:** Route sensitive queries to models or providers that meet privacy requirements.

Multi-objective routing can be formulated as a constrained optimization problem:

```
Minimize: expected cost per query
Subject to: expected quality ≥ quality threshold
            P(latency > latency_limit) ≤ latency_violation_rate
            quality variance across segments ≤ fairness_threshold
```

Solving this requires a model of each objective as a function of the routing strategy, which can be learned from data or estimated from model characteristics.

### 12.2 Context-Aware Routing

Routing decisions can incorporate context beyond the current query:

**Conversation history.** In multi-turn conversations, the difficulty of the current query depends on the preceding context. A follow-up question to a complex analysis may require the large model even if the question itself seems simple.

**User profile.** Repeat users develop patterns—some consistently ask easy questions, others consistently ask hard ones. User-level routing models can learn these patterns and improve routing accuracy.

**Time and load context.** During peak load, aggressive routing to the small model can maintain responsiveness. During off-peak periods, more queries can be routed to the large model to improve quality.

### 12.3 Routing for Tool Use and Agents

Agentic applications—where the LLM uses tools, plans multi-step actions, and iterates on results—present unique routing challenges:

**Planning vs. execution.** The planning step (deciding which tools to use and in what order) may require a large model, while the execution step (calling tools and processing results) may be handled by a small model.

**Error cost.** In agentic workflows, an error in one step can propagate through subsequent steps, amplifying the cost of routing errors. This argues for more conservative routing (favoring the large model) in agentic contexts.

**Variable step difficulty.** Different steps in an agentic workflow may have different difficulty levels. Step-level routing (routing each tool call or sub-task to the appropriate model) can provide better cost optimization than query-level routing.

### 12.4 Federated Routing

In organizations with multiple LLM deployments (different teams, different regions, different use cases), federated routing provides a unified routing layer:

- A central router knows the capabilities, costs, and availability of all deployed models across the organization.
- Queries from any team are routed to the globally optimal model, considering cross-team resource sharing and load balancing.
- This prevents the common pattern where each team independently deploys (and pays for) its own large model, even when a shared smaller model would suffice for most queries.

## 13. Measuring Routing Effectiveness

### 13.1 Key Metrics

**Cost reduction.** The percentage reduction in inference cost compared to always using the most expensive model, at a given quality level.

**Quality preservation.** The percentage of queries where the routed response matches or exceeds the quality of the expensive model's response.

**Routing accuracy.** The percentage of queries correctly routed (easy queries to the small model, hard queries to the large model).

**Escalation rate.** The percentage of queries that are escalated from the small model to the large model in a cascading setup. An optimal escalation rate balances cost and quality.

**Latency impact.** The additional latency introduced by routing or cascading, measured as the difference in P50 and P99 latency compared to direct model access.

### 13.2 Pitfalls in Evaluation

**Benchmark bias.** Routing evaluated on benchmarks may not reflect production performance, because benchmark queries are often harder and more uniform than production queries.

**Quality metric sensitivity.** The choice of quality metric affects the measured routing effectiveness. Exact-match metrics may show routing as more beneficial (because small models often produce approximately correct answers that fail exact match), while human evaluation may show less benefit (because approximately correct answers are often acceptable).

**Distribution shift.** Routing tuned on one query distribution may perform poorly when the distribution changes. Production evaluation with live traffic is essential for validating routing effectiveness.

## 14. The Future of Routing

### 14.1 Model-Native Routing

Future LLM architectures may incorporate routing natively—internally deciding how much computation to allocate to each query. Mixture-of-experts models already do a form of internal routing (selecting which experts process each token). Extending this to variable-depth computation (early exit when a query is easy) or variable-width computation (using fewer experts for easy queries) would make external routing less necessary.

### 14.2 Continuous Model Spectrums

Rather than choosing among discrete model sizes, future serving infrastructure may offer a continuous spectrum of model capabilities. A request could specify a target quality level or cost budget, and the infrastructure would dynamically allocate the appropriate amount of computation—more for harder queries, less for easier ones—without the application needing to manage routing explicitly.

### 14.3 Routing as a Core Infrastructure Primitive

As the LLM ecosystem matures, routing is likely to become a standard infrastructure primitive rather than a custom-built system. Just as load balancers, CDNs, and database query routers are standard components of web infrastructure, model routers will become standard components of AI infrastructure, handling model selection, fallback, cost optimization, and quality monitoring as a service.

## 15. Conclusion

Model routing and cascading are among the most impactful practical optimizations available for LLM-powered applications. By directing each query to the most cost-effective model that can handle it, organizations can reduce inference costs by 30-70% while maintaining quality on the queries that matter most. The fundamental insight—that query difficulty varies widely and cheap models handle easy queries as well as expensive ones—is well-supported by empirical evidence and has been validated in production systems.

The implementation spectrum ranges from simple rule-based routing (implementable in a day, providing 20-40% savings) to sophisticated learned routers with online adaptation (requiring significant engineering investment, providing 50-70% savings). The right level of sophistication depends on the query volume, cost structure, and quality requirements of each application.

The key practical advice is to start simple. Implement two-tier routing with basic rules, measure the cost savings and quality impact, and iterate toward more sophisticated approaches as justified by the data. Routing is an optimization that compounds—small improvements in routing accuracy translate to meaningful cost savings at scale—and the monitoring infrastructure built for simple routing provides the data needed to train more sophisticated routers later.

As LLM applications scale and the model ecosystem continues to diversify, routing will transition from a cost optimization technique to a core architectural pattern. Understanding the trade-offs—cost vs. quality vs. latency, pre-routing vs. cascading, static vs. learned, rule-based vs. semantic—is essential for building LLM-powered systems that are both capable and economical.
