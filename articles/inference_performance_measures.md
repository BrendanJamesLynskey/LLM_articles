# Inference Performance Measures

*April 2026*

## 1. Introduction

"Is the model fast enough?" is the wrong question to ask of a modern LLM
serving system. Speed is not a scalar. A stack that produces 200 tokens
per second on a single idle request may deliver 20 tokens per second to
each of 10 concurrent users, or 1 second to first token on a cold cache
and 80 milliseconds on a warm one. Two systems with identical "tokens per
second" headlines can feel radically different to the same user on the
same workload — one conversational, the other visibly stuttering.

Measuring inference performance properly means separating the quantities
the user actually perceives (how long until I see something, how fast
does it stream) from the quantities the operator cares about (how many
requests can one GPU handle before an SLA is breached). These two views
are connected, but not in a way that can be collapsed into a single
number. This report enumerates the measures that matter, what each one
actually captures, and how they interact under load. It is a reference
for the vocabulary used in later articles on agent performance, capacity
planning, and SLA design.

## 2. The Two Phases and Why They Matter

LLM inference splits into two computationally distinct phases. Every
performance metric is tied to one phase, the other, or the seam between
them.

**Prefill.** All input tokens are processed in a single parallel forward
pass. This is compute-bound and scales roughly with prompt length. A
4K-token prompt on an H100 might take 50–200 ms depending on model size
and tensor parallelism.

**Decode.** Output tokens are generated one at a time autoregressively.
Each step reads the full model weights to emit a single token per
in-flight sequence. This phase is memory-bandwidth-bound. Per-request
decode speed is flat in the batch size until the arithmetic intensity
crosses over to compute-bound, at which point adding more sequences
starts to slow each one down.

The consequence for measurement: prefill latency grows with input
length; per-token decode latency grows with batch size. Any single
"latency" number that fails to separate these will mislead.

## 3. User-Facing Latency Measures

### 3.1 Time to First Token (TTFT)

TTFT is the wall-clock time from request submission to the first
response token emerging on the wire. It includes:

- Network ingress and load-balancer hop
- Admission-control queuing at the inference engine
- Tokenization of the input
- Prefill compute on the GPU
- First decode step
- Streaming framing and egress back to the client

TTFT is the latency the user feels before anything happens. For chat
and streaming assistants it is the single most important UX metric — a
2-second TTFT feels sluggish even if subsequent tokens stream at 100
tok/s. For autocomplete-style uses (IDE completions) TTFT budgets are
tighter still, typically under 200 ms.

### 3.2 Time Per Output Token (TPOT) and Inter-Token Latency (ITL)

TPOT is the average time between successive output tokens once
streaming begins. ITL is the same quantity reported per-step rather
than averaged. Both measure the decode phase directly.

TPOT is what governs perceived streaming speed. A TPOT of 25 ms
corresponds to 40 tok/s, which reads faster than most humans. A TPOT
of 200 ms (5 tok/s) feels painful. TPOT is also where batching
interference appears: a request sharing a GPU with 15 others will have
a higher TPOT than the same request alone, even though throughput per
GPU goes up.

### 3.3 End-to-End Latency

End-to-end latency is TTFT plus the sum of all inter-token intervals up
to the stop token. For a 500-token response it is approximately:

```
E2E ≈ TTFT + (500 - 1) × TPOT
```

End-to-end is the right metric for non-streaming APIs (JSON responses,
structured-output calls that are consumed whole) and for agentic tool
calls where nothing downstream can start until the full response is in
hand. For streaming chat the user never waits for E2E; for an agent
waiting to parse a tool-call argument, E2E is the only thing that
matters.

### 3.4 Interactivity

"Interactivity" has become a named metric, most prominently in NVIDIA's
inference serving discussions and MLPerf's derived reporting. The term
is used in two overlapping senses:

1. **Per-user output speed** measured as tokens per second as
   experienced by one client. Effectively `1 / TPOT`, reported in
   user-facing units.
2. **Sustained per-user speed under concurrency.** The same quantity
   measured while the serving system is also handling N other
   simultaneous users. An engine that delivers 100 tok/s at batch = 1
   but 12 tok/s at batch = 32 has high peak speed but poor
   interactivity under load.

The distinction matters because optimizers trade these off. Bigger
batches raise aggregate throughput but lower per-user interactivity.
Speculative decoding raises interactivity (each accepted draft token
shortens TPOT) but costs extra compute and only helps when the draft
model's guesses are accepted frequently. Disaggregated prefill/decode
raises interactivity for decode at the cost of a second hardware pool.

A useful framing: **interactivity is the slope of the streaming
response, TTFT is its intercept.** A responsive system has both a low
intercept and a steep slope, and holds that steepness as concurrency
grows.

## 4. Operator-Facing Throughput Measures

### 4.1 Requests per Second (RPS) and Tokens per Second

RPS is the most common capacity figure but it is badly defined on its
own, because a request can be 50 tokens in / 50 tokens out or 100K
tokens in / 2K tokens out. Token-oriented measures are less ambiguous:

- **Input tokens/sec.** Prefill throughput — governs cost of long
  prompts, RAG augmentation, and context-heavy workloads.
- **Output tokens/sec.** Decode throughput — governs cost of long
  generations, reasoning models, and agent loops that produce many
  tokens.

These should be reported separately. A system that is prefill-limited
behaves very differently from one that is decode-limited, and the
optimizations are different (chunked prefill, prefill/decode
disaggregation, KV-cache reuse on the prefill side; batching,
quantization, speculative decoding on the decode side).

### 4.2 Goodput

Raw throughput counts every token the server produces, including those
that arrived too slowly to be useful. **Goodput** counts only tokens
delivered while a latency SLA was being met. Formally:

```
goodput = Σ output_tokens_in_requests_meeting_SLA / elapsed_time
```

A system can double its throughput by packing larger batches, yet halve
its goodput because TTFT or TPOT on those larger batches breaches the
SLA. Goodput is the metric that aligns operator-facing and user-facing
performance — it is the throughput that the user actually wanted.

### 4.3 Utilization

GPU utilization as reported by `nvidia-smi` is a near-useless measure
for LLM inference. It reports whether any kernel is resident, not
whether the arithmetic units are doing useful work. Two better signals:

- **Model FLOPs Utilization (MFU).** Achieved FLOPs divided by
  theoretical peak FLOPs. Prefill workloads can hit 40–60% MFU on
  modern GPUs; decode rarely exceeds 5–15% because it is
  bandwidth-bound.
- **Memory-bandwidth utilization.** Achieved DRAM bandwidth divided by
  peak HBM bandwidth. This is the right efficiency metric for decode
  and usually sits in the 60–85% range for a well-optimized engine.

### 4.4 Concurrency and In-Flight Requests

The number of sequences sharing a GPU at any instant determines the
batch size the decode loop sees. For continuous-batching engines this
is not a fixed configuration parameter — it is a state that evolves at
each iteration. Concurrency distributions should be reported alongside
latency percentiles, because a p99 latency measured at an average
concurrency of 4 tells you nothing about what the same system will do
at an average concurrency of 40.

## 5. Percentile Statistics and Their Pitfalls

Means lie. A system that averages 500 ms TTFT might have a p99 of 6
seconds — that 1% of requests is where the complaints come from.
Production inference SLAs are written in percentiles, not averages.

| Percentile | What it captures | Typical use |
|---|---|---|
| p50 (median) | Typical user experience | Capacity sizing on the happy path |
| p95 | Bad-day experience | Most customer SLAs |
| p99 | Near-worst-case | Tight interactive SLAs, voice agents |
| p99.9 | Tail events | Multi-step agents where every step hits p99.9 eventually |

The p99.9 row matters for agents specifically. A 10-step agent whose
per-step p99 is 2 seconds will have an end-to-end p99 far worse than 20
seconds, because each step independently rolls the tail dice. Tail
amplification is discussed further in the companion article on agentic
performance.

Percentiles must be reported with the measurement window and the load
level. "p99 = 800 ms" is meaningless without "over a 5-minute window at
30 RPS with average input length 1200 tokens."

## 6. Streaming-Specific Measures

Streaming APIs introduce measures that don't exist in request-response
systems.

**Time to first chunk.** Some frameworks emit an empty or
server-sent-events handshake chunk before the first token. TTFT should
be measured from request submission to the first chunk containing an
actual content token, not the first framing byte.

**Stream smoothness / jitter.** The variance of inter-token intervals.
Two systems with identical TPOT can feel very different: one delivers
tokens at a steady 40 ms cadence, the other delivers bursts of 8 tokens
then pauses for 320 ms. Humans read the steady stream as faster.
Measure it as the standard deviation of ITL, or as the probability
that any single inter-token gap exceeds some threshold (e.g., P(ITL >
200 ms) < 1%).

**Stall rate.** The fraction of requests that experience an ITL gap
longer than the SLA mid-stream — typically caused by preemption in
priority-aware schedulers or by batch re-composition events.

## 7. Benchmark Regimes: MLPerf Inference

MLPerf Inference standardizes four scenarios, each corresponding to a
different way performance can be reported:

| Scenario | What is measured | Real-world analog |
|---|---|---|
| Single-stream | Latency of one request at a time | Minimum possible TTFT/TPOT, on-device use |
| Multi-stream | Worst-case latency across several fixed streams | Voice assistants, low-concurrency edge |
| Server | p99 latency at the maximum sustainable Poisson arrival rate | Chat APIs, web-scale serving |
| Offline | Throughput with no latency target | Batch processing, embedding generation, RLHF rollouts |

A vendor reporting 10,000 tok/s "MLPerf throughput" without specifying
the scenario is either talking about offline or is quoting a number
that does not generalize. The server scenario is the one that binds an
SLA to capacity and is the number to look at when sizing an
interactive deployment.

MLPerf Inference has also added derived measures such as
**tokens-per-second-per-user at 99%-percentile-latency bounded** — an
explicit interactivity-under-load metric. Expect more of these
multi-axis measures as the benchmark suite evolves.

## 8. Summary of Measures

| Measure | Phase | Who cares | Typical target |
|---|---|---|---|
| TTFT | Prefill | User (perceived latency) | < 500 ms chat, < 200 ms autocomplete |
| TPOT / ITL | Decode | User (stream speed) | < 50 ms (≥ 20 tok/s) |
| Interactivity | Decode, under concurrency | User and operator | ≥ 20 tok/s/user at target load |
| End-to-end | Both | User (non-streaming), agents | Application-dependent |
| RPS | System | Operator | Maximum at goodput SLA |
| Tokens/sec (in/out) | Each | Operator | Hardware-dependent |
| Goodput | Both | Everyone | ≥ 80% of raw throughput |
| MFU / BW utilization | Each | Operator | 40%+ prefill, 60%+ decode BW |
| p50/p95/p99 latency | Both | SLA owners | Defined by SLA |
| Stream jitter | Decode | User | Std(ITL) < TPOT/2 |

## 9. Common Mis-Measurements

**Measuring on an idle machine.** The numbers you get benchmarking one
request at a time on a fresh GPU bear no resemblance to what you will
see at batch 16 under load. Report all metrics at the concurrency you
expect in production.

**Using average latency.** Averages hide tails. Report p50, p95, p99 at
minimum, and p99.9 if you run agents.

**Mixing input and output token counts.** Input tokens consumed during
prefill are not interchangeable with output tokens generated during
decode. Report them separately, otherwise a prompt-caching optimization
that eliminates 90% of prefill work will look like a 90% throughput
win when it is actually a TTFT win.

**Ignoring TTFT's long tail.** TTFT p99 is dominated by queueing, not
by compute. A system that looks great on median TTFT may have a
pathological p99 because its scheduler admits long prefills greedily.

**Reporting interactivity without concurrency.** Single-user tok/s is
the easy number. Sustained tok/s per user under the concurrency the
system is designed for is the honest one.

## 10. How These Measures Feed Agent Performance

Every measure in this article is a per-call property of the underlying
model-serving layer. An agent is a composition of many such calls —
typically a loop that issues an inference call, parses its output,
invokes a tool, and feeds the result back into another inference call.
The agent's end-user-visible latency is a sum of:

- Per-call TTFT and E2E
- Tool invocation round-trip time (network, storage, external API)
- Orchestration overhead on the CPU (parsing, prompt assembly,
  tokenization, memory retrieval)

The tail percentiles of the per-call distribution matter far more for
agents than for single-call products, because each additional step
rolls the tail dice again. The companion article on agentic
performance analyses this composition and the coupling to CPU,
network, and storage that determines how a good model-serving layer
does or does not translate into a good agent experience.

## 11. Abbreviations

| Acronym | Expansion |
|---|---|
| API | Application Programming Interface |
| BW | Bandwidth |
| CPU | Central Processing Unit |
| DRAM | Dynamic Random-Access Memory |
| E2E | End-to-End (latency) |
| FLOPs | Floating-Point Operations (per second) |
| FP16 / BF16 / FP8 / INT4 | 16-bit / bfloat16 / 8-bit / 4-bit numeric formats for weights and activations |
| GPU | Graphics Processing Unit |
| HBM | High-Bandwidth Memory |
| IDE | Integrated Development Environment |
| ITL | Inter-Token Latency |
| JSON | JavaScript Object Notation |
| KV cache | Key / Value cache (attention intermediate state) |
| LLM | Large Language Model |
| MFU | Model FLOPs Utilization |
| MLPerf | Machine Learning Performance (MLCommons benchmark suite) |
| p50 / p95 / p99 / p99.9 | 50th / 95th / 99th / 99.9th percentile |
| RAG | Retrieval-Augmented Generation |
| RLHF | Reinforcement Learning from Human Feedback |
| RPS | Requests Per Second |
| SLA | Service-Level Agreement |
| SSE | Server-Sent Events (streaming over HTTP) |
| TP | Tensor Parallelism |
| TPOT | Time Per Output Token |
| TTFT | Time To First Token |
| UX | User Experience |

## 12. Conclusion

A single "tokens per second" number is almost never the right answer
to the question of how fast an inference system is. The useful answers
separate prefill from decode, user-facing latency from operator-facing
throughput, average behaviour from the tail, and idle benchmarks from
loaded ones. The specific metrics — TTFT, TPOT, interactivity,
goodput, and the p50/p95/p99 latency distribution — each capture a
different aspect of performance that matters to a different stakeholder.
Getting the vocabulary right is a prerequisite for designing SLAs that
describe real user experience, for capacity planning that holds up
under real traffic, and for agent architectures that extend
single-call performance into multi-step workflows without amplifying
the tail.
