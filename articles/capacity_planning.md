# Capacity Planning for LLM Infrastructure

*April 2026*

## 1. Introduction

Capacity planning for LLM infrastructure is the process of determining how much hardware to provision, how to configure it, and how to manage it over time to meet performance, cost, and reliability requirements. It is fundamentally more complex than traditional web service capacity planning because LLMs have unique resource profiles: they consume enormous amounts of GPU memory (making model placement a bin-packing problem), they are bottlenecked by memory bandwidth rather than compute for autoregressive generation (making throughput non-intuitive), and they exhibit complex interactions between batch size, sequence length, and latency (making performance modeling difficult).

A capacity planning error in the wrong direction has immediate and visible consequences. Under-provisioning leads to high latency, request timeouts, and degraded user experience. Over-provisioning wastes expensive GPU resources—at $2-4 per GPU-hour, idle capacity costs add up rapidly. The stakes are high: a fleet of 1,000 H100 GPUs costs $2-4 million per month in cloud rental or represents $25-35 million in capital expenditure. Getting the capacity plan right is one of the most consequential technical decisions in LLM deployment.

This report provides a comprehensive technical examination of capacity planning for LLM inference and training infrastructure. It covers resource estimation, performance modeling, autoscaling, multi-model serving, cost optimization, and monitoring. The intended audience is infrastructure engineers, SREs, and platform architects responsible for operating LLM-serving systems at scale.

## 2. Sizing GPU Clusters for Inference

### 2.1 VRAM Requirements Per Model

The first constraint in capacity planning is fitting the model in GPU memory. The VRAM required depends on the model size and the numerical precision:

**Model weights:**

| Model Size | FP16/BF16 | FP8 | INT4 |
|-----------|-----------|-----|------|
| 7-8B | 14-16 GB | 7-8 GB | 3.5-4 GB |
| 13B | 26 GB | 13 GB | 6.5 GB |
| 34B | 68 GB | 34 GB | 17 GB |
| 70B | 140 GB | 70 GB | 35 GB |
| 405B | 810 GB | 405 GB | ~200 GB |

**KV cache (per request):**

For a standard 70B model with GQA (8 KV heads, 128 head dim, 80 layers), the KV cache per request at FP16 is approximately:

```
2 (K+V) × 80 (layers) × 8 (KV heads) × 128 (head dim) × seq_len × 2 (bytes/FP16)
= 327,680 × seq_len bytes
≈ 0.31 MB per token of context
```

For a 4096-token context: ~1.28 GB per request.
For a 32,768-token context: ~10.2 GB per request.

**Activations and overhead.** The inference engine's runtime, CUDA context, and intermediate activations consume an additional 1-4 GB per GPU.

### 2.2 Tensor Parallelism Requirements

When a model does not fit on a single GPU, it must be split across multiple GPUs using tensor parallelism (TP). The key constraints:

**Memory-driven TP.** If the model weights plus KV cache plus overhead exceed a single GPU's VRAM, TP is required. The number of GPUs must be at least enough to hold the model.

**Efficiency-driven TP.** Higher TP degrees provide more aggregate memory bandwidth, increasing decode throughput. However, TP communication overhead (all-reduce operations after each attention and FFN layer) reduces efficiency. The optimal TP degree balances bandwidth gain against communication overhead.

Typical TP configurations:

| Model Size | FP16 | FP8 | INT4 |
|-----------|------|-----|------|
| 7-8B | 1 GPU | 1 GPU | 1 GPU |
| 13B | 1 GPU (80GB) | 1 GPU | 1 GPU |
| 34B | 1-2 GPUs | 1 GPU | 1 GPU |
| 70B | 2-4 GPUs | 1-2 GPUs | 1 GPU (80GB) |
| 405B | 8-16 GPUs | 4-8 GPUs | 2-4 GPUs |

**NVLink and NVSwitch.** TP within a single node (up to 8 GPUs connected by NVLink/NVSwitch at 900 GB/s bidirectional on H100) is much faster than TP across nodes (connected by InfiniBand at 400-800 Gb/s). Cross-node TP should be avoided when possible due to the 5-10x lower bandwidth.

### 2.3 Pipeline Parallelism for Very Large Models

For models too large for a single node (requiring >8 GPUs at the chosen precision), pipeline parallelism (PP) splits the model's layers across nodes. This avoids the high bandwidth requirements of cross-node TP but introduces pipeline bubble overhead and complicates request scheduling.

PP is typically used in combination with TP: TP within each node (exploiting NVLink) and PP across nodes (using InfiniBand). For example, a 405B model at FP8 might use TP=8 within 2 nodes (one node for the first half of layers, one for the second half) for a total of 16 GPUs.

## 3. Throughput Estimation

### 3.1 Tokens Per Second Per GPU

The throughput of LLM inference depends on whether the workload is prefill-dominated (compute-bound) or decode-dominated (memory-bandwidth-bound):

**Prefill throughput (compute-bound):**

```
Tokens/s = GPU_TFLOPS × MFU / (2 × N_params)
```

For an H100 (1,979 TFLOP/s BF16) with 50% MFU serving a 70B model:

```
Tokens/s = 1,979 × 10¹² × 0.5 / (2 × 70 × 10⁹) ≈ 7,068 tokens/s
```

This means the GPU can process approximately 7,000 input tokens per second during prefill. For a 2-GPU setup (TP=2), throughput per GPU is similar (the work is split, but so is the compute), with some overhead for communication.

**Decode throughput (memory-bandwidth-bound, single request):**

```
Tokens/s = Memory_bandwidth / (Bytes_per_param × N_params)
```

For an H100 (3.35 TB/s) serving a 70B model at FP16 (2 bytes/param):

```
Tokens/s = 3.35 × 10¹² / (2 × 70 × 10⁹) ≈ 24 tokens/s
```

This is the throughput for a single request—each token requires reading the full model from memory. With TP=2 (each GPU reads half the model), per-request throughput approximately doubles to ~48 tokens/s.

**Decode throughput with batching:**

At batch size B, the throughput scales approximately as B × single-request throughput until the workload transitions from memory-bandwidth-bound to compute-bound:

```
B_transition ≈ GPU_TFLOPS / (2 × Memory_bandwidth_TFLOPS)
```

For H100: B_transition ≈ 1,979 / (2 × 3.35 × 10¹² / 2) ≈ 590 for FP16.

In practice, KV cache memory limits the batch size well before this theoretical transition point for most model sizes and sequence lengths.

### 3.2 Request-Level Throughput

The throughput in requests per second depends on the mix of input and output tokens:

```
Requests/s ≈ min(Prefill_throughput / avg_input_tokens, Decode_throughput / avg_output_tokens)
```

For a 70B model on 2× H100 GPUs with 1000 input tokens, 500 output tokens, and batch size 16:

- Prefill capacity: ~14,000 tokens/s → 14 concurrent prefills at 1000 tokens each → each prefill takes ~71ms
- Decode capacity: ~48 tokens/s per request × 16 batch = ~768 output tokens/s → each request takes ~651ms for 500 tokens
- Total time per request: ~722ms
- Approximate throughput: ~22 requests/s

This is a rough estimate; actual throughput depends on the serving framework's scheduling, continuous batching implementation, and system overhead.

### 3.3 Throughput vs. Latency Trade-off

Higher batch sizes increase throughput (requests per second) but also increase latency (time per request):

**Time to first token (TTFT).** At low load, TTFT is determined by prefill time. At high load, TTFT increases because requests wait in queue while previous prefills complete. TTFT is the metric most sensitive to load.

**Inter-token latency (ITL).** The time between consecutive output tokens. At batch size B, each decode step takes longer (proportional to B for compute, constant for memory bandwidth up to the transition point). ITL increases linearly with batch size in the compute-bound regime.

**Time per output token (TPOT).** Average time per output token, including queuing and scheduling overhead. This determines the perceived "typing speed" of the model.

The capacity planner must choose an operating point that balances throughput (to handle the request rate) and latency (to meet SLA requirements). Running at high utilization maximizes throughput but degrades latency; running at low utilization provides good latency but wastes resources.

## 4. Request Patterns and Load Modeling

### 4.1 Bursty vs. Steady Traffic

LLM inference traffic is rarely uniform. Common patterns include:

**Diurnal patterns.** For consumer-facing applications (chatbots, search), traffic follows a daily cycle with peaks during working hours and troughs at night. Peak-to-trough ratios of 3-5x are typical.

**Weekly patterns.** Business applications show lower traffic on weekends. Consumer applications may show higher weekend traffic.

**Event-driven bursts.** Product launches, news events, and marketing campaigns can cause sudden spikes of 10-100x above normal traffic.

**API usage patterns.** Enterprise API consumers may send large batches (e.g., processing a document corpus) followed by periods of no usage. The resulting traffic is highly irregular.

### 4.2 Characterizing the Request Distribution

Capacity planning requires characterizing not just the average request rate but the distribution of request characteristics:

**Input length distribution.** The distribution of input token counts. Long-tailed distributions (most requests are short, a few are very long) are common. Long requests consume more prefill compute and more KV cache memory.

**Output length distribution.** The distribution of output token counts. Also typically long-tailed. Long outputs dominate decode time and KV cache memory.

**Request priority.** Some requests may have strict latency SLAs (interactive chat), while others are latency-tolerant (batch processing, background tasks).

**Model distribution.** If serving multiple models, the fraction of requests going to each model. This determines the allocation of GPUs across models.

### 4.3 Load Testing

Capacity planning should be validated through load testing before production deployment:

**Synthetic load generation.** Generate requests with realistic input/output length distributions at target request rates. Tools like llmperf, vllm-benchmark, and custom load generators can simulate production traffic.

**Key measurements:**
- Throughput (requests/s) at each load level
- TTFT distribution (P50, P95, P99) at each load level
- TPOT distribution at each load level
- GPU utilization, memory utilization, and KV cache utilization at each load level
- Error rate (timeouts, OOM errors) at each load level

**Identify the saturation point.** The request rate at which latency (P99 TTFT or P99 TPOT) exceeds the SLA defines the maximum capacity of the system. The capacity plan should provision for peak traffic with headroom below this saturation point.

## 5. Queuing Theory for Inference

### 5.1 The M/G/c Model

LLM inference can be modeled as a queuing system: requests arrive (with some arrival process), wait in a queue (if all servers are busy), and are served (with some service time distribution).

The most applicable classical model is M/G/c: Poisson arrivals (M), general service time distribution (G), and c servers (GPUs or GPU groups). The service time is not exponential—it depends on input/output length, which varies across requests—making the general (G) distribution more appropriate than an exponential (M) assumption.

### 5.2 Little's Law

Little's Law relates three fundamental queuing metrics:

```
L = λ × W
```

Where:
- L = average number of requests in the system (in queue + in service)
- λ = average arrival rate (requests/second)
- W = average time in the system (queuing time + service time)

For capacity planning, this provides:

```
W = L / λ
```

If the system has capacity for L concurrent requests and the arrival rate is λ, the average wait time is L/λ. For example, if the system can process 50 concurrent requests and the arrival rate is 100 requests/second, the average time in the system is 0.5 seconds.

### 5.3 Utilization and Latency

The most important queuing result for capacity planning is the relationship between utilization and latency. For a single server with Poisson arrivals:

```
Average_wait = Service_time × ρ / (1 - ρ)
```

Where ρ = utilization (fraction of time the server is busy), defined as λ × Service_time / c.

This relationship has a critical property: as utilization approaches 1.0, the average wait time goes to infinity. In practice, this means:

- At 50% utilization: wait time ≈ 1× service time
- At 70% utilization: wait time ≈ 2.3× service time
- At 80% utilization: wait time ≈ 4× service time
- At 90% utilization: wait time ≈ 9× service time
- At 95% utilization: wait time ≈ 19× service time

The practical implication is that running GPUs at >80% utilization leads to rapidly increasing latency. Capacity plans that assume 90%+ utilization will experience latency spikes during any traffic variability. A target utilization of 60-75% provides a good balance between cost efficiency and latency stability.

### 5.4 Handling Variability

The queuing effects are worse when request characteristics are highly variable. A mix of short requests (100 output tokens) and long requests (4000 output tokens) creates more queuing delay than a uniform distribution of medium requests (1000 output tokens), even at the same average.

Continuous batching in modern serving frameworks (vLLM, TensorRT-LLM, SGLang) partially addresses this by processing requests at different stages simultaneously—new requests can start prefilling while existing requests are still decoding. This reduces the queuing delay compared to static batching (where all requests in a batch must complete before new requests can start) but does not eliminate it.

Priority scheduling—serving latency-sensitive requests before latency-tolerant ones—further improves latency for high-priority requests at the expense of increasing latency for low-priority requests. This is particularly useful when the workload includes both interactive (chat) and batch (document processing) traffic.

## 6. Autoscaling Strategies

### 6.1 Why Autoscaling Is Harder for LLMs

Autoscaling LLM inference is more challenging than scaling traditional web services for several reasons:

**Slow startup.** Loading a model onto a new GPU takes 30-120 seconds (depending on model size and storage speed). This means new capacity cannot be brought online instantly in response to traffic spikes.

**Large granularity.** Each model replica requires a fixed number of GPUs (1, 2, 4, or 8, depending on the model and TP configuration). Scaling by one replica adds 1-8 GPUs worth of capacity—a much larger increment than adding one web server.

**State management.** Active requests have KV cache state that cannot be migrated between replicas (without complex live migration protocols). This means that scaling down (removing a replica) requires draining active requests, which takes time.

**Cost of idle capacity.** GPU-hours are expensive. Maintaining idle capacity for potential traffic spikes is costly. The cost of being wrong (idle GPUs running at $3/hour each) creates pressure to scale down aggressively, which conflicts with the slow startup time.

### 6.2 Reactive Autoscaling

Reactive autoscaling adjusts capacity based on observed metrics:

**Request rate-based.** Scale up when the request rate exceeds a threshold per replica; scale down when it drops below another threshold. Simple but reactive—by the time the metric triggers scaling, latency may already be degraded.

**Queue depth-based.** Scale up when the request queue depth exceeds a threshold; scale down when it drops. More responsive than rate-based scaling because queue depth is a leading indicator of latency degradation.

**Latency-based.** Scale up when P95 or P99 TTFT exceeds the SLA; scale down when it drops below a lower threshold. Directly targets the metric that matters (latency) but is reactive to degradation rather than predictive.

**KV cache utilization-based.** Scale up when KV cache memory utilization exceeds a threshold (indicating that the system is memory-constrained and cannot accept more concurrent requests). This is specific to LLM inference and captures a constraint that generic metrics miss.

### 6.3 Predictive Autoscaling

Predictive autoscaling uses forecasting to provision capacity before demand arrives:

**Time-of-day forecasting.** Based on historical diurnal patterns, pre-provision capacity for the expected peak and deprovision for the expected trough. This handles the predictable component of traffic variability.

**Trend-based forecasting.** Track the growth rate of traffic over days/weeks and increase baseline capacity accordingly.

**Event-based pre-provisioning.** For known events (product launches, marketing campaigns), manually or automatically increase capacity in advance.

### 6.4 Warm Pools

A warm pool is a set of pre-provisioned GPUs with the model already loaded, ready to serve requests immediately. The warm pool absorbs traffic spikes without the delay of model loading. The cost is maintaining idle GPUs (which is expensive but provides instant availability).

Warm pool sizing is a classic inventory problem: too few warm GPUs, and spikes cause latency degradation; too many, and the idle GPUs waste money. The optimal warm pool size depends on the traffic variability, the model loading time, and the cost of latency degradation relative to the cost of idle GPUs.

### 6.5 Serverless Inference

Serverless inference platforms (AWS SageMaker Serverless, Google Cloud Vertex AI, various startups) abstract away capacity planning entirely: the user sends requests, and the platform handles provisioning, scaling, and model loading. The user pays per request rather than per GPU-hour.

The trade-off is that serverless platforms typically have higher per-request costs (the platform adds margin and must manage utilization across its fleet) and less predictable latency (cold starts when the model must be loaded onto a new GPU). Serverless is attractive for low-volume, variable-traffic workloads where the alternative is maintaining dedicated GPUs at low utilization.

## 7. Multi-Model Serving

### 7.1 The Bin-Packing Problem

Most organizations serve multiple models—different model sizes, different fine-tuned variants, different model versions. Each model must be placed on GPUs, and the placement must respect memory constraints (each GPU's VRAM is limited) while maximizing utilization.

This is a variant of the bin-packing problem: GPUs are "bins" with limited capacity (VRAM), and models are "items" with known sizes (weight memory + KV cache memory). The goal is to minimize the number of GPUs used (cost) while meeting throughput and latency requirements for each model.

### 7.2 Multi-Model Strategies

**Dedicated GPUs per model.** Each model gets its own set of GPUs. Simple but potentially wasteful if some models have low utilization. Best for high-traffic models where isolation prevents noisy-neighbor effects.

**Model multiplexing.** Multiple small models share the same GPU, with each model using a portion of VRAM. For example, two 3B INT4 models (~2 GB each) can coexist on a single 80 GB GPU with ample room for KV cache. This maximizes utilization for small models.

**Model swapping.** Multiple models share a GPU, but only one is loaded at a time. When a request arrives for a different model, the current model is unloaded and the requested model is loaded. This is appropriate only for very low-traffic models where the loading time (30-120 seconds) is acceptable.

**LoRA serving.** For fine-tuned variants that differ from the base model only in LoRA adapter weights, the base model is loaded once and different LoRA adapters are swapped on a per-request basis. The adapter weights are small (10-100 MB), so swapping is fast (<100ms). This enables serving dozens of fine-tuned variants from a single set of GPUs. Frameworks like vLLM and LoRAX support this natively.

### 7.3 Traffic-Aware Placement

The placement of models on GPUs should consider traffic patterns:

**Co-locate complementary traffic patterns.** If Model A has peak traffic during business hours and Model B peaks during evenings, co-locating them on shared GPUs allows each to use the other's capacity during off-peak times.

**Isolate latency-sensitive models.** Models serving interactive users should not share GPUs with batch processing models, because the batch traffic can cause latency spikes for the interactive model.

**Reserve capacity for burst handling.** Maintain some unallocated GPU capacity that can be assigned to any model during traffic spikes.

## 8. Capacity for Training

### 8.1 Training Cluster Sizing

Training cluster sizing is driven by the target training time and the model/data scale:

**Total FLOPs.** The training compute is C = 6 × N × D. For Llama 3 405B on 15T tokens: C ≈ 3.6 × 10²⁵ FLOPs.

**Time constraint.** If the target training time is T seconds:

```
GPUs_needed = C / (T × GPU_TFLOPS × MFU)
```

For C = 3.6 × 10²⁵, T = 54 days = 4.67 × 10⁶ seconds, H100 at 1,979 TFLOP/s, MFU = 0.40:

```
GPUs = 3.6 × 10²⁵ / (4.67 × 10⁶ × 1.979 × 10¹⁵ × 0.40) ≈ 9,738 GPUs
```

Meta used 16,384 H100s for Llama 3, providing additional headroom for fault tolerance and overhead.

### 8.2 Checkpointing and Fault Tolerance

Large training clusters experience frequent hardware failures. At the scale of 16,000 GPUs, individual GPU failures occur multiple times per day, node failures occur daily, and network partitions occur weekly.

**Checkpointing.** The training state (model weights, optimizer state, data loader position) is saved to persistent storage at regular intervals—typically every 30-60 minutes. If a failure occurs, training restarts from the last checkpoint, losing at most one checkpoint interval of work.

**Checkpoint storage.** The checkpoint size for a 405B model with optimizer state is approximately:

```
Weights (FP32): 405B × 4 bytes = 1.6 TB
Optimizer state (Adam momentum + variance): 405B × 8 bytes = 3.2 TB
Total: ~5 TB per checkpoint
```

Writing 5 TB to storage every 30 minutes requires sustained write throughput of ~2.8 GB/s, which requires fast distributed storage (parallel file systems like GPFS or Lustre, or cloud object storage with high throughput).

**Redundancy strategies.** To minimize the impact of failures:

- Elastic training: the training job continues with fewer GPUs when a failure occurs, automatically incorporating replacement GPUs when they become available.
- In-memory checkpoint replication: checkpoints are stored in memory on other nodes in addition to persistent storage, enabling faster recovery.
- Preemptive migration: if monitoring detects degraded hardware (e.g., ECC errors approaching threshold), the affected GPU's work is migrated before failure occurs.

### 8.3 Cluster Utilization

The utilization of a training cluster is the fraction of time the GPUs are actively training (as opposed to idle due to failures, checkpointing, evaluation, or scheduling gaps):

```
Effective utilization = MFU × uptime_fraction × scheduling_efficiency
```

Where:
- MFU (Model FLOPs Utilization): 30-55%, representing the fraction of theoretical peak compute actually used during training.
- Uptime fraction: 90-98%, representing the fraction of time the cluster is available (not experiencing failures or maintenance).
- Scheduling efficiency: 95-99%, representing the fraction of available time that is used for training (vs. checkpointing, evaluation, data loading).

Total effective utilization is typically 25-50% of theoretical peak. A cluster of 16,000 H100s at 40% effective utilization provides the equivalent compute of 6,400 H100s at theoretical peak.

### 8.4 Mixed Training and Inference

Some organizations use the same GPU cluster for both training and inference, dynamically allocating capacity based on need:

- During active training runs: all GPUs are dedicated to training.
- Between training runs: GPUs are used for inference, evaluation, or experimentation.
- During off-peak inference hours: surplus inference GPUs can be used for training experiments or hyperparameter search.

This requires a scheduling system that can dynamically partition the cluster and manage the transitions (loading/unloading models, migrating work). Kubernetes-based orchestration with GPU scheduling plugins (NVIDIA GPU Operator, RunAI, Volcano) is the most common approach.

## 9. Monitoring and Alerting

### 9.1 Key Metrics

Effective capacity management requires monitoring the following metrics:

**Latency metrics:**
- Time to first token (TTFT): P50, P95, P99
- Time per output token (TPOT): P50, P95, P99
- End-to-end request latency: P50, P95, P99

**Throughput metrics:**
- Requests per second (total and per model)
- Tokens per second (input and output, per model)
- Successful request rate (excluding errors and timeouts)

**Utilization metrics:**
- GPU compute utilization (SM utilization)
- GPU memory utilization (VRAM used / total)
- KV cache utilization (KV cache memory used / available)
- Batch size distribution (average and max concurrent requests)

**Queue metrics:**
- Queue depth (number of requests waiting to be processed)
- Queue wait time (time requests spend in queue before processing begins)

**Error metrics:**
- Timeout rate (requests exceeding the latency SLA)
- OOM rate (requests failing due to out-of-memory errors, typically from KV cache exhaustion)
- Hardware error rate (GPU errors, network errors)

### 9.2 Alerting Thresholds

Typical alerting thresholds:

| Metric | Warning | Critical |
|--------|---------|----------|
| P99 TTFT | > 2× SLA | > 3× SLA |
| P99 TPOT | > 1.5× SLA | > 2× SLA |
| Queue depth | > 50th percentile of historic max | > 90th percentile |
| KV cache utilization | > 80% | > 90% |
| GPU memory utilization | > 85% | > 95% |
| Error rate | > 1% | > 5% |

### 9.3 Dashboards

A comprehensive LLM serving dashboard should include:

**Real-time panels:** Request rate, TTFT/TPOT distributions, queue depth, GPU utilization, KV cache utilization, error rate.

**Trend panels:** Hourly/daily request rate trends, latency trends, utilization trends. These reveal patterns (diurnal cycles, growth trends) that inform capacity decisions.

**Capacity planning panels:** Current capacity vs. projected demand, headroom analysis (how much additional traffic can be absorbed before SLA degradation), cost per request trend.

## 10. Cost Optimization Techniques

### 10.1 Right-Sizing

Ensure each model is served on the minimum hardware that meets SLA requirements:

- Use quantization (FP8 or INT4) to reduce the number of GPUs required per model.
- Use GQA/MQA models to reduce KV cache memory, enabling larger batch sizes on fewer GPUs.
- Evaluate whether a smaller model (which requires fewer GPUs) can meet quality requirements through fine-tuning or better prompting.

### 10.2 Spot/Preemptible Instances

Cloud providers offer spot instances (AWS), preemptible instances (GCP), or spot VMs (Azure) at 50-80% discounts. These instances can be reclaimed by the provider with short notice (30 seconds to 2 minutes).

Spot instances are suitable for:
- **Batch inference.** Latency-tolerant workloads where requests can be retried on new instances if an instance is reclaimed.
- **Training.** With checkpointing, training can resume on new instances after a spot reclamation, losing at most one checkpoint interval.
- **Overflow capacity.** Use spot instances for traffic spikes, falling back to on-demand instances if spot capacity is unavailable.

Spot instances are not suitable for latency-sensitive interactive inference, where a 30-second interruption is unacceptable.

### 10.3 Reserved vs. On-Demand

For predictable, steady-state workloads, reserved instances (1-3 year commitments) provide 30-60% savings over on-demand pricing. The capacity plan should:

1. Estimate the minimum sustained traffic (the "base load") that will persist for the commitment period.
2. Reserve capacity for the base load.
3. Use on-demand (or spot) instances for traffic above the base load.

This hybrid approach minimizes cost while maintaining flexibility.

### 10.4 Request-Level Optimization

Reducing the compute required per request directly reduces capacity requirements:

**Prompt caching.** For applications with shared system prompts or common query prefixes, caching the KV states from the shared prefix avoids redundant prefill computation. This can reduce prefill compute by 50-90% for many applications.

**Early stopping.** Stop generation when the model produces a stop token or when the output is sufficient for the application's needs, rather than always generating to max_tokens.

**Output length control.** Set appropriate max_tokens limits. Generating 1000 tokens when 200 are sufficient wastes 80% of the decode compute.

**Model routing.** Direct easy queries to smaller models, reducing the average compute per request (detailed in the companion article on model routing).

## 11. Reserved Capacity and SLAs

### 11.1 Defining Inference SLAs

Inference SLAs should specify:

**Latency targets:**
- TTFT: P99 < X milliseconds (e.g., 2000ms for chat, 200ms for autocomplete)
- TPOT: P99 < Y milliseconds (e.g., 100ms for streaming chat)
- End-to-end: P99 < Z seconds

**Throughput targets:**
- Minimum sustained requests per second at the specified latency targets.

**Availability:**
- Uptime percentage (e.g., 99.9% = 8.76 hours downtime per year).
- Maximum continuous outage duration.

**Error rate:**
- Maximum percentage of requests that result in errors (timeout, OOM, internal error).

### 11.2 Capacity Headroom

To meet SLAs under traffic variability, capacity must include headroom above the expected peak:

**Headroom for traffic spikes.** Provision 20-50% above the expected peak to absorb unexpected spikes without SLA degradation.

**Headroom for hardware failure.** If N GPUs are required for peak traffic, provision N/(1-f) GPUs, where f is the expected failure rate. For a 2% GPU failure rate, this means provisioning 2% more GPUs.

**Headroom for scaling lag.** If autoscaling takes 2 minutes to add new capacity, the existing capacity must absorb the traffic increase during those 2 minutes.

The total headroom is the product of these factors. For an expected peak of 100 GPUs:

```
100 / (1 - 0.02) × 1.3 (30% spike headroom) × 1.1 (scaling lag) ≈ 146 GPUs
```

This means provisioning roughly 1.5× the minimum required capacity—a significant cost that must be justified by the SLA requirements.

## 12. Real-World Deployment Patterns

### 12.1 Single-Model, High-Volume

The simplest deployment: a single model serving high-volume traffic. Example: a chatbot powered by a 70B model.

- **Configuration:** Multiple replicas of the model, each on 2-4 H100s with TP, behind a load balancer.
- **Autoscaling:** Based on request rate and TTFT latency.
- **Cost optimization:** Quantize to FP8 to halve the GPU count per replica. Use prompt caching for the system prompt.

### 12.2 Multi-Model, Multi-Tier

A more complex deployment serving multiple models at different quality/cost tiers with model routing. Example: an API platform offering small, medium, and large models.

- **Configuration:** Separate GPU pools for each model tier, with independent autoscaling.
- **Routing:** A lightweight router directs requests to the appropriate tier based on user subscription level, query complexity, or explicit model selection.
- **Cost optimization:** LoRA serving for fine-tuned variants within each tier. Aggressive scaling of the small-model pool (which handles the majority of requests). Reserved instances for base load, on-demand for overflow.

### 12.3 Training + Inference Mixed Cluster

A unified cluster that handles both training and inference. Example: an AI startup with a 512-GPU cluster.

- **Configuration:** Kubernetes-based orchestration with GPU partitioning. Training jobs are scheduled as long-running pods; inference deployments are scheduled as auto-scaled replica sets.
- **Priority:** Training can be preempted during inference demand spikes (if the training job supports checkpointing and resumption).
- **Cost optimization:** Maximize cluster utilization by filling idle GPU time with training experiments, hyperparameter search, or evaluation jobs.

### 12.4 Edge + Cloud Hybrid

On-device models handle most queries, with cloud models handling complex queries. Example: a mobile assistant.

- **Configuration:** On-device model (3B INT4) runs on the phone. Cloud model (70B FP8) on a GPU cluster.
- **Capacity planning:** Cloud capacity sized for the escalation rate (10-20% of total queries), not the total query rate. On-device capacity is "free" (the user's hardware).
- **Cost optimization:** Aggressive routing to minimize cloud escalation. Cache cloud responses for common escalation queries.

## 13. Capacity Planning Process

### 13.1 The Planning Cycle

Capacity planning is an ongoing process, not a one-time decision:

1. **Forecast demand.** Estimate request rate, input/output lengths, and model mix for the planning horizon (typically 3-12 months).
2. **Model performance.** Run load tests to determine the throughput and latency of each model configuration at various utilization levels.
3. **Calculate capacity.** Determine the number of GPUs required to meet demand at target utilization (60-75%).
4. **Add headroom.** Add capacity for spikes, failures, and scaling lag.
5. **Optimize cost.** Choose the mix of reserved, on-demand, and spot instances that minimizes cost for the calculated capacity.
6. **Provision and deploy.** Procure hardware, set up infrastructure, deploy models.
7. **Monitor and adjust.** Continuously monitor actual demand vs. forecast, and adjust capacity as needed.
8. **Repeat.** Re-plan on a quarterly or monthly basis as demand evolves.

### 13.2 Forecasting Demand

Demand forecasting for LLM inference is difficult because the market is evolving rapidly:

- New features and integrations drive step-function increases in traffic.
- Pricing changes (both internal and competitive) affect demand.
- Model improvements (faster, cheaper models) can shift traffic between tiers.
- External events (product launches, seasonal patterns) create temporary surges.

A practical approach combines:
- **Bottom-up estimation.** For each product feature that uses the model, estimate the number of users, queries per user, and tokens per query.
- **Top-down trending.** Extrapolate historical traffic growth rates.
- **Scenario planning.** Define optimistic, pessimistic, and base-case scenarios and plan capacity for the base case with the ability to scale to the optimistic case.

## 14. Conclusion

Capacity planning for LLM infrastructure requires a different mental model than traditional web service planning. The key differences are: GPU memory is a hard constraint that determines the minimum hardware per model, memory bandwidth (not compute) is the bottleneck for autoregressive generation, KV cache is a hidden memory consumer that limits batch size and therefore throughput, and the cost of hardware is orders of magnitude higher than traditional compute, making both over-provisioning and under-provisioning expensive mistakes.

The most important capacity planning decisions are:

1. **Quantization level.** The choice between FP16, FP8, and INT4 determines the GPU count per model by 2-4x, making it the highest-leverage hardware decision.
2. **Target utilization.** Running at 60-75% utilization provides a good balance between cost and latency. Higher utilization saves money but risks SLA violations during traffic spikes.
3. **Autoscaling strategy.** Combining predictive scaling (for diurnal patterns) with reactive scaling (for unexpected spikes) and warm pools (for instant capacity) provides robust handling of traffic variability.
4. **Multi-model optimization.** LoRA serving, model multiplexing, and traffic-aware placement maximize the utilization of the GPU fleet across multiple models.

The field is evolving rapidly. New hardware generations (NVIDIA B200, Google TPU v6, AMD MI400) change the performance and cost parameters. New serving frameworks (vLLM, SGLang, TensorRT-LLM) improve throughput and reduce latency. New techniques (speculative decoding, disaggregated serving, prefix caching) change the performance model. Capacity plans should be revisited quarterly and load-tested with each major system change to ensure that the planned capacity continues to meet requirements as the technology evolves.
