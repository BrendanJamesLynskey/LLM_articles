# Groq's Architecture for LLM Inference

*March 2026*

## 1. Introduction

The dominant paradigm for serving large language models in production is GPU-based inference, typically on NVIDIA A100 or H100 accelerators running frameworks like vLLM or TensorRT-LLM. GPUs are general-purpose parallel processors originally designed for graphics, later adapted for training and inference of neural networks. They are extraordinarily capable but carry architectural baggage—deep memory hierarchies, dynamic scheduling, cache-based memory systems—that introduces unpredictability and leaves performance on the table for inference workloads.

Groq represents the most radical departure from this paradigm. Founded in 2016 by Jonathan Ross and a team of engineers who previously designed Google's Tensor Processing Unit (TPU), Groq built a ground-up inference accelerator called the Language Processing Unit (LPU), based on an underlying chip architecture known as the Tensor Streaming Processor (TSP). The design philosophy is simple but extreme: eliminate every source of non-determinism from the hardware, replace off-chip memory with massive on-chip SRAM, and give the compiler total control over scheduling. The result is a system that achieves token generation speeds an order of magnitude faster than GPU-based alternatives for single-stream inference, at the cost of flexibility and per-chip model capacity.

This report examines the TSP architecture, how it achieves its speed advantage, how it scales across chips, and where its fundamental trade-offs lie relative to GPU-based inference and other specialized accelerators.

## 2. Why GPUs Are Suboptimal for LLM Inference

### 2.1 The Memory Bandwidth Bottleneck

LLM inference—particularly the autoregressive decode phase, which generates one token at a time—is dominated by memory bandwidth, not compute. Each generated token requires reading the full model weights from memory, performing a relatively small amount of arithmetic (a few matrix-vector multiplies), and writing back the result. The arithmetic intensity (FLOPs per byte of memory accessed) is extremely low, typically well below 1 for single-request decode on large models.

An NVIDIA H100 SXM provides approximately 3.35 TB/s of HBM3 bandwidth and 989 TFLOPS of FP16 compute. For a 70B parameter model in FP16 (140 GB of weights), generating a single token requires reading 140 GB of weights, which takes roughly 42 ms at peak bandwidth—setting an upper bound of about 24 tokens/second for a single stream regardless of how much compute is available. In practice, actual throughput is lower due to memory access inefficiencies, KV-cache reads, and other overhead.

The GPU's enormous compute capacity (hundreds of TFLOPS) sits largely idle during decode. The chip is bottlenecked waiting for data to arrive from HBM.

### 2.2 Sources of Unpredictability

GPUs rely on hardware-managed caches, dynamic warp scheduling, and speculative execution to handle a wide variety of workloads. These mechanisms are essential for general-purpose flexibility but introduce latency variance:

- **Cache misses** cause unpredictable stalls when data is not in L1/L2 cache and must be fetched from HBM.
- **Dynamic scheduling** means the order and timing of warp execution is determined at runtime by hardware schedulers, making worst-case latency difficult to bound.
- **Memory controller contention** arises when many streaming multiprocessors compete for HBM bandwidth simultaneously.
- **PCIe and NVLink traffic** for multi-GPU configurations adds further variance depending on network congestion.

For training, where throughput over large batches matters more than per-request latency, this unpredictability is acceptable. For real-time inference—voice assistants, interactive coding tools, low-latency API endpoints—it is a meaningful limitation.

### 2.3 The Case for Purpose-Built Hardware

The mismatch between GPU architecture and inference workload characteristics creates an opening for specialized hardware. An ideal LLM inference chip would maximize memory bandwidth per FLOP, eliminate latency variance, and be optimized for the specific computation patterns of transformer decode (matrix-vector products, attention, elementwise operations). This is precisely what Groq set out to build.

## 3. The Tensor Streaming Processor (TSP)

### 3.1 Overview

The TSP is a spatial, dataflow processor designed around the principle of deterministic execution. Rather than fetching instructions and data dynamically like a CPU or GPU, the TSP executes a statically scheduled program where every operation is assigned to a specific functional unit at a specific clock cycle at compile time. The hardware is an assembly line: data flows through a pipeline of functional units in a predetermined order, with the compiler having planned exactly where every byte of data will be at every cycle.

The first-generation TSP is a 25 mm x 29 mm die manufactured on GlobalFoundries' 14nm process, operating at a nominal clock frequency of 900 MHz. It achieves a computational density of more than 1 TeraOp/s per square millimeter.

### 3.2 Functional Units

The TSP organizes its compute resources into vertical "slices," each containing a stack of specialized functional units connected by data transport lanes (the "conveyor belts"). The primary functional unit types are:

| Functional Unit | Abbreviation | Role |
|---|---|---|
| Matrix Multiply Engine | MXM | Dense matrix operations; performs 320-byte x 320-byte dot products |
| Vector Execution Module | VXM | Elementwise vector arithmetic (activations, normalization, etc.) |
| Shift/Rotate Module | SXM | Data permutation and rearrangement operations |
| Memory Module | MEM | SRAM read/write; each slice holds ~2.5 MB of banked SRAM |
| Instruction Control Unit | ICU | Instruction fetch and dispatch (organized horizontally across slices) |

A single TSP contains 88 MEM slices, yielding approximately 220 MB of on-chip SRAM (often cited as ~230 MB including additional buffers). The MXM units support INT8 and FP16 operands with 32-bit accumulation and are fully pipelined: a 320-byte x 320-byte dot product loads in 20 cycles and executes in 20 cycles. The MEM slices are heavily banked to provide massive memory-level parallelism—up to 176-way concurrent access—feeding 20 TB/s of operand bandwidth to the compute units.

### 3.3 SRAM-Only Memory Design

This is the single most distinctive feature of the TSP architecture. The chip has no HBM, no DRAM, no off-chip memory on the compute path, and no caches. All model weights, activations, and intermediate values reside in on-chip SRAM during execution.

The implications are profound:

- **Bandwidth:** On-chip SRAM delivers approximately 80 TB/s of aggregate memory bandwidth—roughly 10x what HBM3 provides on an H100. This directly translates to faster weight reads during decode.
- **Latency:** SRAM access latency is sub-nanosecond and deterministic. HBM access latency is on the order of hundreds of nanoseconds and variable.
- **No cache hierarchy:** Because there are no caches, there are no cache misses, no cache coherence protocols, no eviction policies, and no associated unpredictability. Every memory access completes in a known, fixed number of cycles.
- **No memory controller contention:** The banked SRAM design eliminates the contention that occurs when many compute units compete for shared memory controllers.

The trade-off is capacity. At ~230 MB per chip, the SRAM is roughly 350x smaller than the 80 GB of HBM on an H100. This means even a 7B parameter model in FP16 (14 GB) requires approximately 60 TSP chips, and a 70B model requires hundreds of chips working in concert.

### 3.4 Deterministic Execution Model

On a GPU, the hardware decides at runtime which warps to schedule, when to issue memory requests, and how to handle contention. On the TSP, all of these decisions are made at compile time. The compiler produces a static schedule that maps every operation to a specific functional unit at a specific clock cycle. The hardware then executes this schedule without deviation.

This eliminates:

- **Branch prediction** — there are no branches in the execution path; the control flow is fully unrolled at compile time.
- **Dynamic scheduling** — the hardware has no scheduler; it simply executes the pre-planned instruction stream.
- **Speculative execution** — nothing is speculated; every operation is known to be needed.
- **Runtime arbitration** — there is no contention for shared resources because the compiler has guaranteed conflict-free access at every cycle.

The result is that every operation takes a known, fixed number of cycles. The compiler can compute exact latencies for any input, and the runtime behavior matches the compile-time prediction precisely. This determinism is not just a performance optimization—it is a fundamental architectural property that enables the multi-chip scaling described in Section 5.

### 3.5 Software-Scheduled vs. Hardware-Scheduled

The TSP's approach can be understood as transferring complexity from hardware to software. A GPU invests billions of transistors in scheduling logic, cache controllers, branch predictors, and memory management units. The TSP eliminates all of these, using the freed silicon area for more SRAM and compute units. The complexity budget is instead spent in the compiler, which must solve a sophisticated scheduling problem (mapping millions of operations to specific hardware resources at specific times) before execution begins.

This is analogous to the historical distinction between CISC and RISC architectures, or between out-of-order and VLIW processors. In each case, moving intelligence from hardware to software can yield efficiency gains when the workload is predictable—which LLM inference very much is, since the model graph is fixed and known ahead of time.

## 4. How the LPU Achieves Speed

### 4.1 The Bandwidth Advantage

The fundamental speed advantage is straightforward: the LPU can read model weights faster than a GPU because SRAM bandwidth vastly exceeds HBM bandwidth.

| Metric | NVIDIA H100 SXM | Groq TSP (per chip) | Groq (per-chip advantage) |
|---|---|---|---|
| Memory type | HBM3 | On-chip SRAM | — |
| Memory capacity | 80 GB | ~230 MB | 0.003x |
| Memory bandwidth | 3.35 TB/s | ~80 TB/s | ~24x |
| Memory latency | ~100s of ns | Sub-ns | ~100x+ lower |

Because decode-phase performance is memory-bandwidth-bound, the ~24x bandwidth advantage per chip translates directly into faster token generation, provided the model weights are distributed across enough chips to fit in SRAM.

### 4.2 Deterministic Latency

On a GPU, token generation latency varies from token to token due to cache behavior, scheduling decisions, and memory contention. On the LPU, every token takes the same amount of time because the execution schedule is identical for every forward pass (the only variable is the attention computation over the growing KV-cache, which is also deterministically scheduled). This predictability is valuable for real-time applications where tail latency matters—the p99 latency on a Groq system is essentially the same as the median.

### 4.3 No Batching Required

GPUs achieve good throughput by batching many requests together, amortizing the cost of weight reads across multiple input sequences. This is necessary because reading weights from HBM is slow, and computing a matrix multiply on a single vector wastes most of the GPU's arithmetic throughput. Larger batches increase throughput but also increase per-request latency.

The LPU's SRAM bandwidth is high enough that batching is not required to achieve high throughput on a single request. Weights are adjacent to compute units with 80 TB/s of bandwidth, so even a single-request forward pass keeps the compute units well-fed. This means Groq can deliver high tokens-per-second at batch size 1, whereas a GPU at batch size 1 is severely underutilized.

### 4.4 Published Benchmark Performance

Groq has demonstrated the following performance numbers on its GroqCloud API:

| Model | Groq Output Speed | Typical GPU Cloud Speed | Speedup |
|---|---|---|---|
| Llama 2 70B | ~300 tokens/s | ~30 tokens/s (H100) | ~10x |
| Llama 3 70B | ~284 tokens/s | ~25–35 tokens/s | ~8–11x |
| Mixtral 8x7B | ~430 tokens/s | ~40–60 tokens/s | ~7–10x |
| Llama 3 70B (with speculative decoding) | ~1,665 tokens/s | — | — |

These numbers represent single-stream output token throughput. The speedup is most dramatic at low batch sizes and for latency-sensitive, single-user interactions. At very high batch sizes where GPUs achieve peak throughput, the gap narrows significantly—Groq has reported being approximately 2.5x faster than a V100 at large batch sizes in image inference benchmarks, compared to 17.6x at batch size 1.

### 4.5 Latency Characteristics

Groq systems exhibit extremely low time-to-first-token (TTFT) and inter-token latency. Because there is no dynamic scheduling overhead, no cache warmup, and no contention-induced variance, the prefill phase completes quickly and decode tokens arrive at a consistent, rapid cadence. For interactive applications—chatbots, coding assistants, voice interfaces—this creates a qualitatively different user experience compared to GPU-served models, where the first few hundred milliseconds are often spent waiting for prefill and the first tokens to arrive.

## 5. Multi-Chip Scaling

### 5.1 The Scaling Challenge

With only ~230 MB of SRAM per chip, even modest models require many chips. A 7B parameter model in FP16 needs roughly 60 chips; a 70B model needs over 300. The multi-chip system must maintain the deterministic execution guarantees of a single chip while coordinating hundreds of processors—a fundamentally difficult problem.

### 5.2 Plesiosynchronous Interconnect

Groq developed a custom chip-to-chip communication protocol described as "plesiosynchronous" (nearly synchronous). Each chip has its own crystal oscillator, and the protocol compensates for the natural clock drift between chips through periodic software-initiated synchronization. This allows hundreds of LPUs to operate as if they were a single logical processor, with the compiler able to predict exactly when data will arrive at each chip.

This is in stark contrast to GPU clusters, where inter-chip communication (even over NVLink) is asynchronous and introduces variable latency. Groq's approach extends determinism from the single-chip level to the entire system.

### 5.3 Tensor Parallelism Across TSPs

Groq distributes models using tensor parallelism: each transformer layer's weight matrices are partitioned across multiple chips, with each chip holding a slice of the parameters in its local SRAM. A forward pass involves each chip computing its portion of the matrix multiply and exchanging partial results with its neighbors. Because the communication timing is deterministic, the compiler schedules these exchanges as part of the static instruction stream, interleaving communication with computation to hide transfer latency.

The system uses forward error correction (FEC) rather than retransmission for error handling on the inter-chip links. Retransmission would introduce non-deterministic delays, violating the architecture's core guarantee. FEC corrects errors in-place at each hop, maintaining the deterministic timing of every packet.

### 5.4 Physical Configurations

Groq packages its hardware in several form factors:

| Configuration | Description |
|---|---|
| **GroqCard** | Single TSP chip in a PCIe Gen 4 x16 form factor; 375W max / 240W average power |
| **GroqNode** | 8 interconnected GroqCards in a single server chassis; ~4 kW max power |
| **GroqRack** | Multiple GroqNodes in a rack; typically 9 GroqNodes (72 chips) per rack |

For serving Mixtral 8x7B, Groq reportedly deployed 8 racks of 9 servers each (576 chips total). For Llama 3 70B at FP8 precision (~70 GB of parameters), approximately 300+ LPU chips are required, each holding a ~230 MB slice of the model. A next-generation GroqNode design is expected to increase the chip count per node by mounting LPU chips directly on motherboards rather than using PCIe cards.

### 5.5 Scaling Limits

The TSP network architecture is designed to support systems of over 10,000 chips with minimal latency degradation. However, the practical scaling challenge is economic: serving a single 70B model requires an entire rack (or more) of specialized hardware, whereas a pair of H100 GPUs can hold the same model. The per-model hardware footprint is Groq's most significant scaling constraint.

## 6. Software Stack

### 6.1 The Groq Compiler

The compiler is arguably the most critical component of the Groq system—it was developed before the chip itself. The compiler takes a model graph (from PyTorch, TensorFlow, or ONNX) and produces a complete static schedule: every operation is mapped to a specific functional unit on a specific chip at a specific clock cycle. This involves solving a complex resource allocation and scheduling problem, sometimes described as a "Tetris problem" of fitting operations into the chip's spatiotemporal execution grid.

Key compiler responsibilities include:

- **Graph optimization:** Operator fusion, constant folding, dead code elimination, and other standard compiler optimizations adapted for the TSP's dataflow execution model.
- **Memory allocation:** Assigning every weight tensor, activation tensor, and intermediate buffer to a specific SRAM bank on a specific chip.
- **Instruction scheduling:** Determining the exact cycle at which each operation executes, ensuring no resource conflicts and that data dependencies are satisfied.
- **Multi-chip partitioning:** Deciding how to split the model across chips, where to place tensor parallel boundaries, and when to schedule inter-chip communication.
- **Quantization:** Supporting post-training quantization (typically to INT8 or FP8) to reduce model size and the number of chips required.

Compilation can be time-consuming—potentially hours for large models—because the scheduler must explore a vast search space. This makes Groq poorly suited for research workflows where models change frequently, but well-suited for production deployments where a model is compiled once and served for weeks or months.

### 6.2 GroqFlow

GroqFlow is an open-source tool that automates the process of taking a model from a standard framework and compiling it for GroqChip processors. It accepts PyTorch or ONNX models and handles conversion, optimization, quantization, and compilation in a streamlined pipeline. Under the hood, GroqFlow invokes the Groq Developer Tools package (groq-devtools) for compilation and the Groq Runtime package (groq-runtime) for execution on hardware.

### 6.3 GroqCloud API

For users who do not operate their own Groq hardware, GroqCloud provides an OpenAI-compatible REST API for inference. As of early 2026, supported models include Llama 3 (8B and 70B), Mixtral 8x7B, Gemma, and several other open-weight models. The API has attracted over 1.9 million developers, with enterprise deployments at organizations including Dropbox, Volkswagen, and Riot Games. Meta partnered with Groq in 2025 for an official Llama API offering.

### 6.4 Limitations of the Software Ecosystem

The Groq software stack is narrower than CUDA's ecosystem. There is no equivalent of the thousands of CUDA libraries, community tools, and framework integrations that NVIDIA has accumulated over two decades. Custom model architectures require going through the Groq compiler, which may not support all operators or graph patterns. The ahead-of-time compilation model means that dynamic control flow (variable-length loops, data-dependent branching) must be handled at the graph level rather than within the compiled program.

## 7. Trade-Offs and Limitations

### 7.1 Inference Only

The TSP is designed exclusively for inference. The deterministic, statically scheduled execution model does not support the stochastic gradient updates, dynamic loss scaling, and backward-pass computation required for training. Organizations using Groq must train models elsewhere (on GPUs or TPUs) and then compile and deploy them on LPU hardware.

### 7.2 SRAM Capacity Constrains Per-Chip Model Size

At ~230 MB per chip, the SRAM capacity is the binding constraint on system size. Every increase in model size requires a proportional increase in chip count, with no option to use cheaper, denser off-chip memory as a fallback. This creates a steep cost curve for large models.

| Model | Approximate Chip Count (FP8) | Approximate Chip Count (FP16) |
|---|---|---|
| 7B | ~30 | ~60 |
| 70B | ~300 | ~600+ |
| 405B | ~1,700+ | ~3,500+ |

By contrast, a single H100 (80 GB HBM) can hold a 70B model in FP8 entirely on-chip, or a pair of H100s can hold it in FP16.

### 7.3 Cost

The GroqCard has been estimated at approximately $20,000 per chip. Serving a 70B model at ~300 chips represents roughly $6 million in chip costs alone, excluding servers, networking, power, and cooling. An equivalent GPU deployment (2–8 H100s at $25,000–$40,000 each) costs $50,000–$320,000. Groq's cost advantage must come from higher throughput per dollar over time (lower cost per token) rather than lower upfront capital expenditure.

### 7.4 Batch Size Economics

Groq excels at low-batch, latency-sensitive workloads. GPUs excel at high-batch, throughput-optimized workloads. At large batch sizes, GPU compute utilization increases substantially (the arithmetic intensity rises as the same weights serve many sequences), and the GPU's massive FLOP capacity becomes the dominant advantage rather than memory bandwidth. For workloads like offline document processing, batch embedding generation, or high-concurrency API serving where latency is secondary to throughput and cost, GPUs often deliver better total cost of ownership.

### 7.5 Flexibility

The static compilation model means that changing model architecture requires recompilation. Dynamic features like variable-length generation, speculative decoding with variable acceptance rates, or adaptive computation (early exit) must be handled through pre-planned execution paths. The GPU's dynamic scheduling makes it inherently more flexible for workloads that cannot be fully characterized at compile time.

### 7.6 Power Efficiency

A GroqCard consumes up to 375W, comparable to a high-end GPU. However, because many more chips are needed per model, the total system power for serving a single model is substantially higher. A 300-chip Groq deployment for a 70B model could consume upwards of 72 kW (300 x 240W average), versus approximately 1.4 kW for two H100s serving the same model. The performance-per-watt equation favors Groq only if the throughput gain exceeds the power cost increase, which depends on the workload and pricing model.

## 8. Competitive Landscape

### 8.1 Other Inference Accelerators

Groq is one of several companies building specialized hardware for AI inference, each with a distinct architectural philosophy:

| Company | Architecture | Key Differentiator |
|---|---|---|
| **Cerebras** | Wafer-Scale Engine (WSE-3); single 46,225 mm² die | Massive on-chip SRAM (40 GB); eliminates multi-chip partitioning for many models |
| **SambaNova** | Reconfigurable Dataflow Unit (RDU) with three-tier memory (SRAM + HBM + DRAM) | Flexibility via reconfigurability; supports both training and inference |
| **Tenstorrent** | RISC-V based AI accelerators with mesh architecture | Open-source ISA; modular chiplet design; targets cost-efficient inference |
| **Google TPU** | Systolic array architecture with HBM | Tight integration with TensorFlow/JAX; massive scale in Google Cloud |

Cerebras has emerged as a particularly strong competitor in inference, with independent benchmarks from Artificial Analysis showing the WSE-3 achieving over 2,500 tokens/s on Llama 3.3 70B—roughly 6x Groq's speed on the same model. Cerebras's advantage comes from its 40 GB of on-chip SRAM (versus Groq's 230 MB per chip), which allows it to hold much larger models on a single wafer without the multi-chip communication overhead.

### 8.2 GPU Software Optimization

NVIDIA and the open-source community have made substantial progress in closing the inference performance gap through software:

- **TensorRT-LLM** provides fused kernels, FP8/INT4 quantization, in-flight batching, and paged KV-cache management, significantly improving H100 inference throughput.
- **vLLM** with PagedAttention and continuous batching has become the de facto open-source serving framework, achieving 2–4x throughput improvements over naive GPU inference.
- **Speculative decoding** allows GPUs to generate multiple tokens per forward pass, reducing the number of memory-bandwidth-bound decode steps.
- **FP8 inference** on H100 Tensor Cores effectively doubles throughput compared to FP16 with minimal accuracy loss.

These optimizations have substantially narrowed the gap between GPU-based inference and purpose-built accelerators, particularly for throughput-oriented workloads with large batch sizes.

### 8.3 Where Groq Excels

Groq's architecture is best suited for workloads with the following characteristics:

- **Latency-critical**: Real-time voice, interactive chat, coding assistance where time-to-first-token and inter-token latency directly impact user experience.
- **Low-batch or single-stream**: Applications serving individual users rather than batching hundreds of requests together.
- **Deterministic requirements**: Use cases where bounded, predictable latency is more important than average-case throughput (e.g., robotics, financial services, real-time decision-making).
- **Models that fit within the chip budget**: Small to medium models (7B–70B) where the chip count and cost remain manageable.

## 9. Benchmarks and Real-World Performance

### 9.1 GroqCloud API Performance

GroqCloud has demonstrated consistently fast inference in production. Independent benchmarks place Groq's Llama 3 70B output speed at approximately 280–300 tokens/second, compared to 25–35 tokens/second from GPU-based cloud providers (using single-stream measurements). Time-to-first-token is typically under 200 ms, and inter-token latency is consistently in the low single-digit millisecond range.

### 9.2 Comparison with GPU Cloud Providers

| Provider | Model | Output Tokens/s | TTFT | Notes |
|---|---|---|---|---|
| **Groq** (GroqCloud) | Llama 3 70B | ~284 | <200 ms | Single-stream, FP8 |
| **NVIDIA** (H100, TensorRT-LLM) | Llama 3 70B | ~30–35 | ~300–500 ms | Single-stream, FP8, 2 GPUs |
| **Cerebras** (Inference Cloud) | Llama 3.3 70B | ~2,500+ | N/A | Wafer-scale engine |
| **Together AI** (GPU-based) | Llama 3 70B | ~50–80 | ~200–400 ms | Optimized GPU serving |

These numbers shift significantly at higher batch sizes. GPU-based providers can increase aggregate throughput substantially by batching dozens or hundreds of requests, while Groq's per-chip throughput is largely independent of batch size (since it does not need batching to saturate bandwidth).

### 9.3 Cost-Performance Analysis

The cost-performance calculus depends on the metric:

- **Cost per token (at low batch):** Groq is competitive or superior, because its chips generate tokens much faster and the cost per token amortizes the higher hardware expense over more tokens per second.
- **Cost per token (at high batch):** GPUs are typically superior, because batching allows GPUs to amortize weight reads across many sequences, achieving high throughput per dollar.
- **Cost per unit of latency:** Groq wins by a wide margin, as no GPU configuration can match its single-stream speed.

Groq's GroqCloud API pricing has been competitive with GPU-based providers, suggesting that the company absorbs the higher hardware costs and competes on the value of speed rather than on raw cost-per-token at scale.

## 10. Future Directions

### 10.1 Next-Generation TSP

Groq's second-generation LPU (LPU v2) is being manufactured on Samsung's 4nm process node, a significant shrink from the first-generation's 14nm GlobalFoundries process. The move to 4nm is expected to deliver a 15–20x improvement in power efficiency and a substantial increase in on-chip SRAM capacity, which would directly reduce the number of chips required per model. If SRAM capacity scales to 1 GB+ per chip, a 70B model in FP8 would require roughly 70 chips instead of 300+, dramatically improving the cost equation.

### 10.2 The NVIDIA Acquisition

In December 2025, NVIDIA announced an agreement to acquire Groq's technology assets for approximately $20 billion—a record acquisition for NVIDIA. Groq founder Jonathan Ross and president Sunny Madra were reported to be joining NVIDIA as part of the deal. Groq has characterized this as a non-exclusive licensing arrangement. The acquisition signals NVIDIA's recognition that deterministic, SRAM-based architectures represent a meaningful complement to GPU-based inference, particularly for latency-sensitive workloads. It remains to be seen whether NVIDIA will integrate TSP concepts into future GPU designs, offer Groq hardware as a separate product line, or incorporate the technology into its inference-as-a-service offerings.

### 10.3 Scaling to Larger Models

As models continue to grow (Llama 3 405B, GPT-4-class models at rumored 1T+ parameters), the SRAM capacity constraint becomes more acute. Groq's path to serving these models depends on a combination of next-generation chips with more SRAM, aggressive quantization (FP4, sub-4-bit), and potentially hybrid architectures that selectively use off-chip memory for less latency-sensitive data (e.g., KV-cache for long contexts) while keeping weights in SRAM.

### 10.4 The Role of Specialized Hardware

The broader question raised by Groq's architecture is whether the LLM inference market will converge on GPUs (with ever-better software optimization) or fragment into specialized hardware for different workload profiles. The historical precedent from networking (ASICs vs. general-purpose processors), video encoding (hardware encoders vs. CPU-based), and cryptocurrency mining (ASICs vs. GPUs) suggests that sufficiently large, stable workloads eventually justify purpose-built hardware. LLM inference may be reaching that threshold, with Groq, Cerebras, and others demonstrating that meaningful performance gains are available to those willing to abandon GPU generality.

The most likely near-term outcome is coexistence: GPUs for training and flexible, high-batch inference; specialized accelerators like Groq for latency-critical, real-time serving; and wafer-scale processors like Cerebras for workloads that benefit from extreme on-chip memory capacity. The NVIDIA-Groq deal may accelerate the integration of these approaches into unified platforms.

## 11. Conclusion

Groq's LPU architecture represents a fundamentally different approach to LLM inference: trade memory capacity for memory bandwidth, trade hardware flexibility for deterministic execution, and trade runtime scheduling for compile-time planning. The result is a system that generates tokens at speeds no GPU can match for single-stream inference, with rock-solid latency guarantees and no dependence on batching for utilization.

The architecture's limitations are equally clear. The small per-chip SRAM capacity demands large chip counts for any meaningful model, driving up costs and power consumption. The static compilation model sacrifices flexibility. The inference-only design means the chips cannot participate in the training workflow. And the narrow software ecosystem places a higher burden on the compiler to support new model architectures.

Whether these trade-offs are worthwhile depends entirely on the application. For a real-time voice assistant that needs consistent sub-10ms inter-token latency, Groq is unmatched. For a batch processing pipeline that needs to classify millions of documents at minimum cost, a GPU cluster with vLLM will likely win. The LPU's contribution to the field is not in replacing GPUs but in demonstrating that the design space for inference hardware is far larger than the GPU-centric ecosystem had assumed—and that the memory bandwidth wall, not the compute wall, is the barrier worth breaking through.
