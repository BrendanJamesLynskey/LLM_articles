**Hardware Implications of Hosting Multiple Open-Source Models for Multi-Agent Systems**

Architecture, Memory Planning, and Practical Configurations

March 2026

Technical Report

Table of Contents

1\. Introduction: The Multi-Agent Multi-Model Problem

Multi-agent systems have moved beyond research curiosity into practical deployment. Frameworks such as LangGraph, CrewAI, and AutoGen allow developers to compose pipelines where multiple LLM-backed agents collaborate on complex tasks, each agent assuming a specialised role within the pipeline. A supervisor agent might orchestrate the workflow using a large reasoning model, tool-calling agents handle structured function invocation with smaller instruction-tuned models, a code-generation agent runs a code-specialised model, and a retrieval-augmented generation (RAG) subsystem relies on an embedding model to index and query a vector store.

In practice, this means a single multi-agent application may depend on three to eight distinct models loaded simultaneously. A representative configuration might include Llama 3.1 70B for high-level reasoning, Qwen2.5-Coder 32B for code generation, Mistral 7B for lightweight tool-calling, and nomic-embed-text for embeddings. This is a fundamentally different hardware planning problem from the single-model serving scenario that dominates most inference optimization literature.

When serving a single model, capacity planning reduces to a relatively clean question: how many GPUs are needed to hold the model weights plus KV cache for the desired concurrency? Multi-model serving introduces combinatorial complexity. Each model carries its own memory footprint, each maintains separate KV caches for concurrent requests, and the system must decide which models stay resident in VRAM and which are swapped on demand. Interference effects between models sharing the same GPU, fragmentation from repeated loading and unloading, and the heterogeneous latency profiles of differently-sized models all compound the challenge.

This report examines the hardware implications of this multi-model multi-agent paradigm in detail, covering memory budgeting, GPU topology, model scheduling, quantisation strategy, network and storage considerations, and practical hardware configurations at several price points.

2\. Memory Budget Analysis

2.1 The Components of GPU Memory Consumption

For any single model loaded on a GPU, the total VRAM consumption consists of four components:

**Total VRAM = Model Weights + KV Cache + Activation Memory + Runtime Overhead**

Model weights are the static cost: the parameters of the network stored at whatever precision the quantisation format dictates. KV cache is the dynamic cost that grows with the number of concurrent sequences and their context lengths. Activation memory holds intermediate tensors during the forward pass and is typically modest during inference (unlike training, where activations are retained for the backward pass). Runtime overhead includes CUDA context, memory allocator fragmentation, framework buffers, and other fixed costs that typically consume 500MB to 1.5GB per model process.

The weight memory for a given model is straightforward to compute:

**Weight Memory (bytes) = Number of Parameters x Bytes per Parameter**

For a 70-billion-parameter model at 4-bit quantisation (approximately 0.5 bytes per parameter with overhead from scales and group metadata in formats like Q4_K_M), the weight memory is roughly 40GB. At FP16 (2 bytes per parameter), the same model requires 140GB.

KV cache memory per sequence per layer is:

**KV Cache per Layer = 2 x Hidden Dimension x Sequence Length x Bytes per Element**

The factor of 2 accounts for both key and value tensors. For a model with 80 layers, 8192 hidden dimension, and FP16 KV cache, a single sequence at 4096 tokens consumes approximately 2 x 8192 x 4096 x 2 x 80 = 10.7GB. With quantised KV caches (Q8 or Q4), this can be reduced by 2-4x.

2.2 Concrete Multi-Model Memory Budget

Consider a representative multi-agent deployment using quantised open-source models:

| Model | Role | Quant Format | Weight Memory | KV Cache (1 seq, 4K ctx) |
|-------|------|-------------|---------------|-------------------------|
| Llama 3.1 70B | Supervisor / Reasoning | Q4_K_M (GGUF) | ~40 GB | ~2.5 GB |
| Qwen2.5-Coder 32B | Code Generation | Q5_K_M (GGUF) | ~23 GB | ~1.2 GB |
| Mistral 7B v0.3 | Tool Calling | Q8_0 (GGUF) | ~7.7 GB | ~0.4 GB |
| nomic-embed-text v1.5 | RAG Embeddings | FP16 | ~0.5 GB | N/A (no autoregressive decoding) |

The weight memory alone totals approximately 71.2GB. Adding KV cache for even a single concurrent sequence per model brings the total to roughly 75.3GB. With runtime overhead of approximately 1GB per model process, the practical requirement approaches 79GB before accounting for any concurrency beyond a single request per model.

This immediately rules out any single GPU with less than 80GB of VRAM if all models must be loaded concurrently. An NVIDIA A100 80GB or H100 80GB is at the absolute edge, with almost no headroom for concurrent requests or longer contexts. An H200 with 141GB of HBM3e provides meaningful breathing room.

2.3 KV Cache Pressure Under Concurrency

Multi-agent systems often process multiple pipeline instances concurrently. If three users are running the agent pipeline simultaneously, each model must maintain KV caches for three active sequences. The KV cache for the 70B model jumps from 2.5GB (one sequence) to 7.5GB (three sequences) at 4K context. With longer contexts — 16K or 32K tokens, which are common for reasoning tasks with extensive prompts — the KV cache can easily exceed the weight memory for the model itself.

This multiplicative relationship between concurrency, context length, and per-model KV cache is the central memory planning challenge for multi-model deployments. A system that fits comfortably with one concurrent pipeline may fail catastrophically with three.

2.4 Memory Fragmentation

Repeatedly loading and unloading models creates memory fragmentation on the GPU. Even if the total free VRAM is sufficient for a new model, the free space may be scattered across non-contiguous regions, preventing allocation of the large contiguous blocks that model weights require. CUDA's memory allocator (and higher-level allocators in frameworks like PyTorch) mitigate this to some degree, but fragmentation remains a practical concern in long-running multi-model serving processes. Periodic defragmentation — effectively unloading all models and reloading them — may be necessary in production systems that experience frequent model swaps.

3\. GPU Topology and Multi-GPU Strategies

3.1 Single Large GPU

The simplest topology is a single GPU with enough VRAM to hold all models. As of early 2026, the relevant options are:

- **NVIDIA H100 80GB (HBM3):** Fits the example four-model stack with modest concurrency. The 3.35 TB/s memory bandwidth provides strong decode throughput for the large model. Cost: approximately $25,000-$30,000 per GPU.
- **NVIDIA H200 141GB (HBM3e):** Comfortably fits the full stack with room for concurrent KV caches and even additional models. The 4.8 TB/s bandwidth is the highest available. Cost: approximately $30,000-$40,000.
- **NVIDIA A100 80GB (HBM2e):** Fits the stack at the edge. The 2.0 TB/s bandwidth is adequate but noticeably slower than H-series for decode-heavy workloads. Available used at $8,000-$12,000, making it an attractive option.

A single-GPU deployment eliminates inter-GPU communication overhead. All models share the same memory space (though via separate allocations), and the inference server can manage VRAM as a unified pool. The limitation is that only one model's forward pass executes at a time on the GPU (unless using MPS or MIG partitioning), so agents cannot truly run in parallel — they are multiplexed on the GPU's compute resources.

3.2 Multi-GPU Model-per-GPU Assignment

A more scalable approach assigns different models to different GPUs. This allows true parallel execution: the supervisor's 70B model can be generating tokens on GPU 0 while the code agent's 32B model runs on GPU 1. Communication between agents happens through the host CPU and system memory, not through GPU interconnects, since agents exchange text (tokens), not tensors.

A practical configuration might be:

- **GPU 0 (48GB+):** Llama 3.1 70B Q4_K_M — dedicated GPU for the largest model
- **GPU 1 (24GB):** Qwen2.5-Coder 32B Q5_K_M — dedicated GPU for code generation
- **GPU 2 (24GB):** Mistral 7B Q8_0 + nomic-embed-text — shared GPU for small models

This configuration requires three GPUs but provides the highest throughput because there is no contention between models for compute resources. The inter-agent communication bandwidth requirement is minimal — agent messages are typically a few kilobytes of text — so even PCIe-connected GPUs work well. NVLink's high bandwidth (900 GB/s on NVLink 4.0) is only relevant if a single model is split across multiple GPUs via tensor parallelism, not for inter-agent communication.

3.3 Tensor Parallelism for the Large Model

When no single GPU has enough VRAM for the largest model, tensor parallelism splits the model across multiple GPUs. For Llama 3.1 70B at Q4_K_M (~40GB weights), two 24GB GPUs can host it with tensor parallelism, though the KV cache must also be split. This works but introduces latency overhead: every transformer layer requires an all-reduce operation across the GPUs, and the latency of this communication directly impacts per-token generation time.

With NVLink, the all-reduce overhead per layer is on the order of 10-20 microseconds, adding roughly 1-2 milliseconds to total per-token latency across 80 layers. With PCIe 4.0 (64 GB/s bidirectional), this overhead increases by 5-10x, potentially adding 5-15 milliseconds per token. For a multi-agent system where the 70B model might generate 500 tokens per supervisor call, this translates to 2.5-7.5 seconds of additional latency from interconnect overhead alone. NVLink is strongly preferred for tensor-parallel configurations.

3.4 Consumer GPU Considerations

For developers and small teams, consumer GPUs offer substantially better price-to-VRAM ratios:

- **2x RTX 4090 (2x 24GB = 48GB total):** At approximately $2,000 each, this provides 48GB of aggregate VRAM for $4,000. The GPUs are PCIe-connected (no NVLink on consumer cards since RTX 3090), so tensor parallelism across them incurs significant overhead. The better strategy is model-per-GPU assignment: the 32B code model on one GPU, and the 7B + embedding models on the other, with the 70B model using CPU offloading or being replaced with a smaller reasoning model.
- **1x RTX 4090 (24GB):** Can run 2-3 small models (7B-14B) concurrently, or one 32B quantised model. The 70B model does not fit at any practical quantisation level. Viable for development with model swapping.
- **1x RTX A6000 (48GB):** Professional card with ECC memory and 48GB VRAM. Can fit the 70B Q4_K_M model with limited KV cache headroom, or the 32B + 7B + embedding models comfortably. Available used for $3,000-$4,000.

3.5 The Apple Silicon Option

Apple's M-series chips with unified memory architecture offer an unconventional but increasingly practical alternative. The M4 Ultra, with up to 192GB of unified memory accessible to both CPU and GPU cores, can load the entire four-model stack with ample room for KV caches. The M4 Max with 128GB is also sufficient for many configurations.

The tradeoff is throughput. Apple Silicon's memory bandwidth — 819 GB/s for M4 Ultra — is roughly one-quarter of an H100's 3.35 TB/s. Since autoregressive decode is memory-bandwidth bound, token generation is proportionally slower. An M4 Ultra generating tokens from a 70B Q4_K_M model achieves approximately 10-15 tokens per second, compared to 40-60 tok/s on an H100. For development, prototyping, and low-concurrency applications, this is often acceptable, and the ability to load all models simultaneously without swapping simplifies the development workflow considerably.

The Mac Studio with M4 Ultra (192GB) costs approximately $7,000-$8,000, placing it between consumer GPU builds and professional datacenter hardware. Its silent operation, low power draw (under 200W for the full system), and macOS development environment make it attractive for individual developers and small research teams.

4\. Model Loading, Swapping, and Scheduling

4.1 Hot Models vs Cold Models

Not all models in a multi-agent pipeline are used with equal frequency. The supervisor model may be called at the start and end of every pipeline execution, making it the hottest model. The code agent might only be invoked for 30% of requests. An embedding model might run once at the beginning for retrieval and then sit idle. A rational VRAM management strategy keeps hot models permanently loaded and swaps cold models on demand.

The key metric is the **swap latency** — the time to load a model from storage into VRAM:

| Model Size (Quantised) | NVMe SSD (~7 GB/s) | SATA SSD (~550 MB/s) | Network Storage (~1 GB/s) |
|------------------------|--------------------|-----------------------|---------------------------|
| 7B Q8_0 (~7.7 GB) | ~1.1 s | ~14 s | ~7.7 s |
| 32B Q5_K_M (~23 GB) | ~3.3 s | ~42 s | ~23 s |
| 70B Q4_K_M (~40 GB) | ~5.7 s | ~73 s | ~40 s |

These are sequential read times; actual loading includes parsing, memory allocation, and GPU transfer overhead that typically adds 20-50% to the raw I/O time. NVMe storage is effectively mandatory for tolerable swap latencies with larger models. Loading the 70B model from SATA SSD takes over a minute, which is unacceptable for interactive multi-agent pipelines.

4.2 Ollama's Model Caching Behaviour

Ollama, the most popular tool for local model serving, implements a model caching strategy controlled by the `OLLAMA_KEEP_ALIVE` parameter (default: 5 minutes). After a model is used, it remains loaded in VRAM for the keep-alive duration. If another request arrives for the same model within that window, it is served immediately. If the keep-alive expires, the model is unloaded to free VRAM for other models.

For multi-agent workloads, the default 5-minute keep-alive is often too short. A pipeline that uses the supervisor model, then calls the code agent, then returns to the supervisor 6 minutes later will find the supervisor unloaded — incurring a multi-second reload penalty. Setting `OLLAMA_KEEP_ALIVE=-1` (never unload) keeps all used models resident but risks VRAM exhaustion. The optimal strategy depends on the pipeline's temporal access pattern and available VRAM.

Ollama's concurrency model also presents limitations. By default, Ollama processes requests sequentially per model. The `OLLAMA_NUM_PARALLEL` setting allows concurrent requests to the same model, and `OLLAMA_MAX_LOADED_MODELS` controls how many models can be resident simultaneously. Tuning these parameters is essential for multi-agent performance:

- `OLLAMA_MAX_LOADED_MODELS=4` allows all four models to stay loaded
- `OLLAMA_NUM_PARALLEL=2` on the supervisor model allows two pipeline instances to share it
- `OLLAMA_KEEP_ALIVE=-1` prevents unloading between pipeline stages

4.3 vLLM Multi-Model Serving

vLLM provides more sophisticated multi-model serving through its PagedAttention memory manager. Multiple models can be served from a single vLLM instance (or multiple instances sharing GPUs via CUDA MPS), with each model receiving a dynamically-sized allocation from the VRAM pool. PagedAttention's non-contiguous KV cache allocation is particularly valuable in multi-model scenarios because it eliminates the fragmentation problems that plague naive memory management when multiple models' KV caches grow and shrink independently.

vLLM's continuous batching also allows each model to process multiple concurrent requests efficiently, batching across pipeline instances. If three multi-agent pipelines are all waiting on the supervisor model, vLLM batches those three requests and processes them together, amortizing the memory bandwidth cost of reading the 70B model's weights across three sequences.

4.4 The "All Models Loaded" vs "Swap on Demand" Tradeoff

The decision between keeping all models loaded and swapping on demand is governed by a simple inequality:

**If Total Weight Memory of All Models < Available VRAM - Required KV Cache Headroom, load all models.**

When this inequality holds, there is no reason to swap. Every model is available with zero additional latency, and the system's response time is determined purely by inference speed. When it does not hold, the system designer must choose which models to keep hot and accept swap latency for the rest.

A hybrid approach maintains a priority queue of models ranked by access frequency and latency sensitivity. The supervisor model, being both frequently accessed and latency-critical (its output determines the next pipeline step), is always kept loaded. Embedding models, being small, are always loaded. The code model and tool-calling model are swapped based on recent usage patterns.

Preemption strategies handle VRAM exhaustion when a new model must be loaded but no free space exists. The least-recently-used model is evicted, its KV caches are either discarded (if no active requests) or spilled to CPU memory (if active requests exist). vLLM implements KV cache swapping to CPU memory, allowing preempted sequences to resume without regeneration when the model is reloaded.

5\. Inference Throughput and Latency in Multi-Agent Pipelines

5.1 Sequential Agent Chains

The simplest multi-agent pattern is a sequential chain: the supervisor calls agent A, which calls agent B, which returns results up the chain. The total end-to-end latency is the sum of per-agent latencies:

**T_total = T_supervisor_plan + T_agent_A + T_agent_B + T_supervisor_synthesise + T_overhead**

Where T_overhead includes agent framework processing, prompt construction, and inter-agent communication. For typical multi-agent pipelines:

- T_supervisor_plan (70B model, ~200 output tokens): 5-15 seconds depending on hardware
- T_agent_A (7B model, ~300 output tokens): 3-8 seconds
- T_agent_B (32B model, ~500 output tokens of code): 10-25 seconds
- T_supervisor_synthesise (70B model, ~150 output tokens): 4-12 seconds
- T_overhead: 0.5-2 seconds

Total: 22.5-62 seconds for a single pipeline execution. The large model calls dominate even though they produce fewer tokens, because per-token latency scales roughly linearly with model size in the memory-bandwidth-bound decode regime.

5.2 Parallel Agent Fan-Out

More sophisticated pipelines fan out multiple agent calls in parallel. The supervisor dispatches the code agent, tool-calling agent, and RAG retrieval simultaneously, then aggregates results. Here, the total latency for the parallel segment is determined by the slowest agent:

**T_parallel = max(T_code_agent, T_tool_agent, T_retrieval)**

This is faster than sequential execution but requires all invoked models to be loaded simultaneously (no swapping during the parallel phase) and requires sufficient GPU compute to serve multiple models concurrently. On a single GPU, the parallel calls are actually serialised at the compute level — the GPU multiplexes between them — so the benefit comes only from overlapping prefill with decode across models, and the speedup is modest. True parallelism requires multi-GPU configurations where each model has dedicated compute.

5.3 The Large Model Bottleneck

In almost every multi-agent pipeline, the largest model is the latency bottleneck. The 70B supervisor model takes 3-5x longer per token than the 7B tool-calling model, and it is typically called at the beginning and end of every pipeline execution. Two strategies mitigate this:

**Reduce calls to the large model.** Restructure the pipeline so the supervisor emits a detailed plan in a single call, rather than iteratively calling subordinate agents with supervision between each step. This reduces the number of 70B model invocations from N+1 (for N agents) to 2 (plan + synthesise).

**Reduce per-call latency.** Use speculative decoding with a smaller draft model (e.g., Llama 3.1 8B drafting for Llama 3.1 70B) to accelerate the large model's token generation. With high acceptance rates (70-90% for same-family draft models), this can improve the 70B model's effective throughput by 2-3x.

5.4 Batching Across Pipeline Instances

When multiple pipeline instances run concurrently — serving multiple users or processing a batch of tasks — the inference server can batch requests to the same model across pipelines. If five pipelines all need the supervisor model, those five requests are batched and processed together, increasing GPU compute utilization significantly compared to processing them sequentially.

This batching benefit is one of the strongest arguments for using a proper inference server (vLLM, TGI) rather than a simple model loading framework (Ollama in its default configuration). The throughput improvement from batching 8 concurrent requests to a 70B model can be 4-6x compared to sequential processing, because the memory bandwidth cost of reading model weights is amortised across all sequences in the batch.

5.5 Request Routing and Priority

Not all agent calls are equally latency-sensitive. The supervisor's planning step is on the critical path — nothing else can proceed until it completes. A tool-calling agent that is one of four parallel agents is less critical; if it finishes slightly later, the impact on total pipeline latency may be zero (if another parallel agent is slower).

Production multi-agent systems should implement priority-aware request routing. Supervisor model requests receive high priority and are scheduled immediately, potentially preempting lower-priority requests. Parallel agent requests receive normal priority. Background embedding or indexing requests receive low priority and yield VRAM and compute to interactive workloads.

6\. Quantisation Strategy for Multi-Model Deployments

6.1 Differentiated Quantisation by Role

A key insight for multi-model deployments is that not all models need the same quantisation quality. The reasoning model, whose output quality directly determines the pipeline's overall quality, benefits from higher-fidelity quantisation. Tool-calling models, which produce structured outputs (function names, JSON arguments), are more tolerant of aggressive quantisation because their output space is constrained.

A practical strategy:

| Model Role | Recommended Quant | Rationale |
|-----------|-------------------|-----------|
| Supervisor / Reasoning (70B) | Q5_K_M or Q6_K | Quality-sensitive; open-ended generation where subtle errors compound |
| Code Generation (32B) | Q5_K_M | Code correctness matters; moderate quantisation preserves syntax accuracy |
| Tool Calling (7B) | Q4_K_M or Q4_0 | Structured output; constrained vocabulary; aggressive quant tolerable |
| Embeddings | FP16 or Q8 | Small model; quantisation savings minimal; embedding quality matters for retrieval |

6.2 Memory Savings from Quantisation

The following table shows weight memory for each model at various quantisation levels:

| Model | FP16 | Q8_0 | Q6_K | Q5_K_M | Q4_K_M | Q4_0 |
|-------|------|------|------|--------|--------|------|
| Llama 3.1 70B | 140 GB | 74 GB | 57 GB | 49 GB | 40 GB | 37 GB |
| Qwen2.5-Coder 32B | 64 GB | 34 GB | 26 GB | 23 GB | 19 GB | 17 GB |
| Mistral 7B v0.3 | 14.5 GB | 7.7 GB | 5.9 GB | 5.1 GB | 4.1 GB | 3.8 GB |
| nomic-embed-text | 0.5 GB | 0.27 GB | — | — | — | — |
| **Total** | **219 GB** | **116 GB** | **89 GB** | **77 GB** | **63 GB** | **58 GB** |

The difference between FP16 and Q4_K_M is a 3.5x reduction, bringing the four-model stack from requiring multiple high-end GPUs to fitting on a single 80GB GPU. This illustrates why quantisation is not optional for multi-model deployments — it is a prerequisite.

6.3 GGUF Format Advantages

The GGUF format (used by llama.cpp and Ollama) is particularly well-suited for heterogeneous multi-model serving. GGUF files are self-contained, including model architecture metadata, tokenizer data, and quantised weights in a single file. This simplifies model management when running 4-8 different models — each is a single file that can be loaded independently without framework-specific conversion or configuration.

GGUF also supports flexible GPU/CPU offloading at layer granularity. The largest model can offload some layers to CPU memory while keeping the most performance-critical layers on the GPU. This allows a 70B Q4_K_M model to run on a 24GB GPU by offloading roughly 50% of layers to system RAM, at the cost of reduced throughput (tokens per second drops roughly in proportion to the fraction of layers offloaded, limited by CPU memory bandwidth of typically 50-100 GB/s versus GPU HBM bandwidth of 1-5 TB/s).

7\. Network and Storage I/O

7.1 Model Weight Storage

The total storage requirement for the four-model stack at the quantisation levels in Section 6 is approximately 63-77GB. This is modest by modern storage standards, but the I/O performance of the storage medium directly impacts model swap latency.

- **NVMe SSD (PCIe 4.0 x4):** Sequential read speeds of 5-7 GB/s. Loading the 70B Q4_K_M model takes 6-8 seconds. This is the minimum acceptable tier for multi-model serving with model swapping.
- **NVMe SSD (PCIe 5.0 x4):** Sequential reads of 10-14 GB/s. Halves swap latency compared to PCIe 4.0. Increasingly available and recommended for new deployments.
- **SATA SSD:** 500-550 MB/s sequential read. Model swaps take 30-80 seconds for large models. Not viable for interactive multi-agent workloads.
- **Network Storage (NFS, S3):** Highly variable. Dedicated 10GbE connections provide ~1 GB/s; 25GbE or 100GbE InfiniBand can approach local NVMe speeds. Relevant for shared storage in multi-node deployments.

7.2 Multi-Machine Distributed Multi-Agent

For large-scale deployments, models can be distributed across multiple machines. The supervisor's 70B model runs on a dedicated GPU server; small models run on a separate, less expensive machine. Inter-agent communication crosses the network, adding latency.

The bandwidth requirement for inter-agent messages is minimal. Agent messages are text — typically 1-50 KB per message. Even over a 1 Gbps network, a 50 KB message transfers in 0.4 milliseconds. The concern is not bandwidth but round-trip latency. Each inter-agent call adds one network round trip (typically 0.1-1 ms on a local network, 1-50 ms across datacenters). For a pipeline with 5-10 inter-agent calls, this adds 0.5-500 ms of network overhead — negligible compared to inference latency on a local network, but potentially significant across wide-area networks.

The transport protocol matters for reliability and overhead. gRPC with protocol buffers is the standard choice for inter-agent communication, offering bidirectional streaming (useful for token-by-token output), built-in load balancing, and efficient serialisation. HTTP/REST with JSON is simpler but adds serialisation overhead and does not natively support streaming. For co-located deployments (same machine or same rack), Unix domain sockets or shared memory eliminate network overhead entirely.

7.3 Co-located vs Distributed Tradeoffs

Co-locating all models on a single machine simplifies operations dramatically: no network configuration, no distributed state management, no partial failure modes where one machine is down while others are up. The cost is that the machine must have enough VRAM, CPU memory, and power capacity for all models.

Distribution across machines enables scaling beyond single-machine limits and provides isolation — a crash in the code agent's serving process does not affect the supervisor model. It also enables heterogeneous hardware: the 70B model runs on an H100, while small models run on cheaper RTX 4090 machines. The operational cost is managing a distributed system with its attendant complexity in deployment, monitoring, and failure handling.

For most teams below enterprise scale, co-location on a single well-provisioned machine is the pragmatic choice. The operational simplicity outweighs the flexibility of distribution until the workload exceeds single-machine capacity.

8\. Practical Hardware Configurations

The following tiers provide concrete hardware recommendations for different scales of multi-agent deployment.

8.1 Tier 1 — Hobbyist / Solo Developer

**Hardware:** Single NVIDIA RTX 4090 (24GB VRAM), 64GB system RAM, 2TB NVMe SSD

**Budget:** ~$3,000-$4,000 (full system)

**Capability:** Run 2-3 small models (7B-14B class) concurrently. The 70B model does not fit; use a 7B-14B reasoning model instead, or offload layers to CPU at reduced throughput. Swap models on demand with 1-3 second latency via NVMe. Single concurrent pipeline instance.

**Representative model stack:**
- Qwen2.5 14B Q5_K_M (~10 GB) — reasoning
- Mistral 7B Q8_0 (~7.7 GB) — tool calling
- nomic-embed-text FP16 (~0.5 GB) — embeddings
- Total: ~18.2 GB, leaving ~5 GB for KV caches

**Limitation:** No 70B-class reasoning capability. Pipeline throughput limited by single-GPU serialisation.

8.2 Tier 2 — Small Team / Startup

**Hardware:** 2x NVIDIA RTX 4090 (48GB total) or 1x RTX A6000 (48GB), 128GB system RAM, 4TB NVMe

**Budget:** ~$5,000-$8,000 (full system)

**Capability:** Run 4-5 models concurrently across two GPUs. The 70B model can fit with partial CPU offloading (~20 layers on GPU, ~60 on CPU), providing ~3-5 tok/s. Or, use a 32B reasoning model on one GPU and smaller models on the other. Two to three concurrent pipeline instances with careful memory management.

**Representative model stack (2x RTX 4090):**
- GPU 0: Qwen2.5 32B Q4_K_M (~19 GB) — reasoning + code — leaves 5 GB for KV cache
- GPU 1: Mistral 7B Q8_0 (~7.7 GB) + DeepSeek-R1-Distill-Llama 8B Q6_K (~6.6 GB) + nomic-embed-text (~0.5 GB) — tool calling, secondary reasoning, embeddings — total ~14.8 GB, leaves 9 GB for KV cache

**Limitation:** 70B models impractical at interactive speeds. PCIe-only interconnect prevents efficient tensor parallelism across GPUs.

8.3 Tier 3 — Production / Research Lab

**Hardware:** 1x NVIDIA A100 80GB or H100 80GB, 256GB system RAM, 8TB NVMe RAID

**Budget:** ~$15,000-$35,000 (GPU alone); ~$25,000-$50,000 (full server)

**Capability:** Load the full four-model stack (70B + 32B + 7B + embedding) concurrently. Moderate concurrency (3-5 pipeline instances) with the 70B model as the bottleneck. High single-stream throughput: 30-60 tok/s for the 70B model on H100.

**Representative model stack (H100 80GB):**
- Llama 3.1 70B Q4_K_M (~40 GB)
- Qwen2.5-Coder 32B Q4_K_M (~19 GB)
- Mistral 7B Q4_K_M (~4.1 GB)
- nomic-embed-text (~0.5 GB)
- Total weights: ~63.6 GB, leaving ~16 GB for KV caches and overhead

**Limitation:** Single GPU limits concurrency; the 70B model monopolises compute during its forward pass, serialising all other model requests. Scaling beyond 5 concurrent pipelines requires additional GPUs or model sharding.

8.4 Tier 4 — Enterprise

**Hardware:** Multi-node server with 4-8x H100 80GB GPUs per node, NVLink interconnect, 1TB+ system RAM, shared NVMe storage

**Budget:** ~$150,000-$400,000 per node

**Capability:** Dedicated GPU(s) per model class. The 70B model uses 2 GPUs with tensor parallelism for maximum throughput; the 32B model gets a dedicated GPU; small models share a GPU. Tens to hundreds of concurrent pipeline instances. Full production SLAs with redundancy and autoscaling.

**Representative allocation (8x H100 80GB):**
- GPUs 0-1: Llama 3.1 70B FP16 (140 GB) with tensor parallelism — maximum quality and throughput
- GPU 2: Qwen2.5-Coder 32B Q5_K_M (23 GB) — dedicated code generation
- GPU 3: Mistral 7B Q8_0 (7.7 GB) + additional tool models — shared small-model GPU
- GPUs 4-7: Reserved for scaling, redundancy, or additional model variants

8.5 Mac Studio Alternative

**Hardware:** Mac Studio with M4 Ultra, 192GB unified memory

**Budget:** ~$7,000-$8,000

**Capability:** Load all models concurrently in unified memory. The 819 GB/s memory bandwidth yields approximately 10-15 tok/s for the 70B model — 3-4x slower than an H100 but with no swapping delays and silent operation. Suitable for development, prototyping, and low-concurrency production use.

**Advantages:** Silent operation, low power (~150W under load), macOS development environment, no driver issues, all models always loaded.

**Limitations:** Slow inference for large models. Cannot scale horizontally (no multi-machine GPU clustering). Limited to the GGUF/llama.cpp/MLX ecosystem. No CUDA, so vLLM and TensorRT-LLM are unavailable.

8.6 Cost Comparison Summary

| Configuration | VRAM | Approx. Cost | 70B tok/s | Max Concurrent Models | Max Concurrent Pipelines |
|--------------|------|-------------|-----------|----------------------|-------------------------|
| 1x RTX 4090 | 24 GB | $3,500 | N/A (too large) | 2-3 small | 1 |
| 2x RTX 4090 | 48 GB | $6,000 | ~3-5 (CPU offload) | 4-5 | 2-3 |
| 1x A100 80GB | 80 GB | $25,000 | ~25-35 | 4-6 | 3-5 |
| 1x H100 80GB | 80 GB | $35,000 | ~40-60 | 4-6 | 5-8 |
| 1x H200 141GB | 141 GB | $40,000 | ~45-65 | 6-8 | 8-15 |
| Mac Studio M4 Ultra | 192 GB (unified) | $8,000 | ~10-15 | 6-8 | 1-2 |

9\. Software Frameworks and Orchestration

9.1 Ollama

Ollama provides the simplest path to multi-model serving. Models are pulled from the Ollama library (or imported from GGUF files) and served via a local HTTP API. Ollama manages model loading, VRAM allocation, and caching automatically. For multi-agent systems, each agent is configured with its model name, and Ollama handles loading and unloading.

Ollama's limitations for production multi-agent use include limited batching (no continuous batching as of early 2026), basic scheduling (FIFO queue per model), and modest concurrency control. It is well-suited for development and small-scale deployment but becomes a bottleneck at higher concurrency levels.

9.2 vLLM

vLLM is the most capable open-source inference engine for multi-model serving. Its PagedAttention memory management, continuous batching, and efficient CUDA kernels provide near-optimal throughput for each model. Multiple vLLM instances (one per model) can run on the same machine, each allocated to specific GPUs via CUDA_VISIBLE_DEVICES. Alternatively, vLLM's multi-model serving (via its OpenAI-compatible API server) can serve multiple models from a single process.

For multi-agent systems, vLLM's OpenAI-compatible API means each agent simply points to a different model name on the same endpoint. LangGraph, CrewAI, and AutoGen all support OpenAI-compatible APIs natively, making integration straightforward.

9.3 Text Generation Inference (TGI) with Router

Hugging Face's TGI can be deployed with a router that distributes requests across multiple model endpoints. Each model runs in its own TGI instance, and the router forwards requests based on the model name in the request. This provides isolation between models (a crash in one instance does not affect others) and allows independent scaling (run more replicas of the bottleneck model).

9.4 Ray Serve

Ray Serve provides a general-purpose model serving framework with autoscaling, multi-model deployment, and integration with the Ray distributed computing ecosystem. Each model is deployed as a Ray Serve deployment with configurable resource requirements (e.g., `num_gpus=1`). Ray's scheduler places deployments on available GPUs and can scale replicas up and down based on request queue depth.

For multi-agent systems, Ray Serve's composition API allows chaining model calls within the serving layer, reducing the overhead of external HTTP calls between agents. This is particularly valuable when the agent framework (LangGraph, etc.) is also running within the Ray ecosystem.

9.5 LiteLLM

LiteLLM provides a unified API that routes requests to different model backends — local (Ollama, vLLM), cloud (OpenAI, Anthropic), or hybrid. For multi-agent systems, LiteLLM enables a gradual migration path: start with cloud APIs for the expensive 70B model while running smaller models locally, then bring the 70B model on-premises when hardware is available. Each agent is configured with a model name, and LiteLLM transparently routes to the appropriate backend.

9.6 Kubernetes and GPU Scheduling

For production deployments, Kubernetes with the NVIDIA GPU Operator provides GPU-aware scheduling. Each model serving pod requests specific GPU resources, and the Kubernetes scheduler places pods on nodes with available GPUs. NVIDIA's Multi-Instance GPU (MIG) feature on A100 and H100 can partition a single GPU into isolated instances, allowing multiple small models to share a GPU without interference.

A typical Kubernetes deployment for a multi-agent system includes:
- A deployment for the large reasoning model (requesting 1 full GPU or 2 GPUs for tensor parallelism)
- A deployment for each smaller model (requesting 1 GPU or a MIG partition)
- A deployment for the agent orchestration framework (CPU only)
- Horizontal Pod Autoscalers that scale model replicas based on request queue depth

10\. Power, Thermal, and Operational Considerations

10.1 Power Draw

Multi-GPU systems consume substantial power under sustained inference load:

| Configuration | GPU TDP | System Total (est.) | Annual Electricity (24/7, $0.12/kWh) |
|--------------|---------|--------------------|-----------------------------------------|
| 1x RTX 4090 | 450W | ~650W | ~$685 |
| 2x RTX 4090 | 900W | ~1,150W | ~$1,210 |
| 1x A100 80GB | 300W | ~600W | ~$630 |
| 1x H100 80GB | 700W | ~1,000W | ~$1,050 |
| 4x H100 80GB | 2,800W | ~3,500W | ~$3,680 |
| Mac Studio M4 Ultra | N/A | ~150-200W | ~$175 |

For 24/7 multi-agent serving, electricity costs are a non-trivial operational expense. The Mac Studio's 150W power draw, roughly one-fifth of a single-H100 system, partially offsets its slower inference speed in cost-per-query terms for low-concurrency workloads.

10.2 Thermal Management

Consumer GPUs (RTX 4090) are designed for burst workloads in gaming, not sustained inference. Under continuous inference load, GPU temperatures can reach 80-85C, triggering thermal throttling that reduces clock speeds and throughput. Adequate case airflow, aftermarket cooling, or open-air server chassis are necessary for sustained operation.

Datacenter GPUs (A100, H100) are designed for sustained operation and include more robust cooling solutions, but they still require proper airflow. A single H100 SXM5 module dissipates 700W of heat, requiring server-grade cooling infrastructure.

Multi-GPU configurations multiply thermal challenges. Two RTX 4090 cards in a standard ATX case create a thermal bottleneck: the upper card's intake draws hot exhaust from the lower card. Spacer slots between GPUs, adequate case fans, and possibly water cooling are necessary for sustained multi-GPU inference.

10.3 Cloud vs On-Premises Cost Crossover

For teams deciding between cloud GPU instances and on-premises hardware, the cost crossover depends on utilization:

An H100 instance on major cloud providers costs approximately $2.50-$4.00 per GPU-hour. At 24/7 utilization, that is $21,900-$35,040 per year for a single GPU. An on-premises H100 costs approximately $35,000 upfront plus $1,050/year in electricity, with amortized cost roughly $12,000/year over a 3-year lifespan (including electricity and maintenance).

The crossover is straightforward: if average GPU utilization exceeds approximately 40-50%, on-premises is cheaper over a multi-year horizon. For a multi-agent system in active use by a development team or serving production traffic, utilization typically exceeds this threshold. However, cloud instances offer flexibility (scale up for demos, scale down at night), no upfront capital, and no hardware maintenance — factors that may dominate the pure cost calculation for smaller teams.

11\. Recommendations

Based on the analysis in this report, the following recommendations apply to teams deploying multi-agent systems with multiple open-source models:

**Size VRAM for worst-case concurrent model loading.** Identify the maximum number of models that must be loaded simultaneously (typically during parallel agent fan-out stages) and ensure sufficient VRAM exists for all their weights plus KV caches at the expected concurrency level. Undersizing VRAM leads to model swapping during latency-critical pipeline stages.

**Dedicate GPU resources to the largest model.** The 70B-class reasoning model is both the largest consumer of VRAM and the latency bottleneck. Give it dedicated GPU(s) and allocate remaining GPU(s) for smaller models. Do not time-share the large model's GPU with other models unless VRAM is sufficient for both and compute contention is acceptable.

**Apply differentiated quantisation.** Use Q5_K_M or Q6_K for the reasoning model where output quality is paramount. Use Q4_K_M or even Q4_0 for tool-calling agents that produce structured output. Use FP16 or Q8 for embedding models. This strategy minimizes total VRAM consumption while preserving quality where it matters most.

**Use NVMe storage for model weights.** Model swap latency is directly proportional to storage read speed. NVMe SSDs (PCIe 4.0 or 5.0) provide 5-14 GB/s sequential reads, keeping swap times under 10 seconds even for 70B models. SATA SSDs are unacceptable for interactive workloads.

**Monitor VRAM fragmentation.** In long-running serving processes with frequent model swaps, VRAM fragmentation can silently reduce effective capacity. Monitor actual vs theoretical VRAM availability and implement periodic defragmentation (full model reload) if fragmentation exceeds 10-15%.

**Set appropriate keep-alive durations.** In Ollama or similar caching model servers, set keep-alive durations based on your pipeline's access patterns. Models called in every pipeline execution should have infinite keep-alive. Models called sporadically can use shorter durations. Measure the actual interval between calls to each model under production load and set keep-alive accordingly.

**Consider model consolidation.** Before deploying 6 different models, ask whether a single capable 32B model (like Qwen2.5 32B) can serve multiple roles — reasoning, code generation, and tool calling — with different system prompts. Fewer models mean simpler operations, less VRAM consumption, and better batching efficiency. The quality trade-off from using one model for multiple roles is often smaller than expected, especially with well-crafted prompts.

**Profile your actual pipeline.** The theoretical analysis in this report provides starting points, but every multi-agent pipeline has a unique access pattern. Instrument your pipeline to measure per-agent latency, model utilization, VRAM consumption over time, and inter-agent wait times. The bottleneck is usually the large model — but profile to confirm, because the fix differs depending on whether the bottleneck is compute (need faster GPU), memory (need more VRAM), or scheduling (need better batching).

**Start with co-located single-machine deployment.** Unless your scale demands it, run all models on one machine. The operational simplicity of a single machine — one set of drivers, one monitoring agent, no network partitions between agents — is worth more than the theoretical scalability of a distributed deployment. Scale to multiple machines when the single machine's GPU capacity or throughput is demonstrably insufficient.

**Plan for growth.** Multi-agent systems tend to accumulate models over time as new agent roles are added. The code agent gets a specialized coding model; the data agent gets a SQL-specialized model; the security agent gets a model fine-tuned on vulnerability patterns. Budget VRAM for 50-100% more models than your initial design requires.