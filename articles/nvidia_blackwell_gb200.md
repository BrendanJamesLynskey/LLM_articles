# NVIDIA Blackwell GB200 NVL72: A Rack-Scale Computer for Trillion-Parameter Inference

*April 2026 • Technical Report*

## 1. Introduction

When NVIDIA unveiled the Blackwell architecture at GTC 2024, the headline was not the chip — it was the rack. The GB200 NVL72 is a single liquid-cooled cabinet that fuses 36 Grace CPUs and 72 Blackwell GPUs into what NVIDIA marketed as "one giant GPU": 13.5 TB of HBM3e visible to a single NVLink domain, 130 TB/s of all-to-all bandwidth, and roughly 1.4 exaFLOPS of FP4 inference throughput. The product designation reflects the count: 72 Blackwell dies in a single NVLink-connected coherent system, packaged as a 120 kW rack with direct-to-chip liquid cooling. By the time the first units shipped to hyperscalers in late 2024, the GB200 NVL72 had become the reference platform for trillion-parameter LLM inference and training.

This report examines the Blackwell GPU itself, the GB200 superchip that pairs it with Grace, the NVL72 rack architecture, and the implications of moving from node-scale to rack-scale computers for LLM workloads.

## 2. The Blackwell GPU

### 2.1 Dual-Die Architecture

Blackwell is the first NVIDIA datacenter GPU to use a multi-die package. Two reticle-limit dies, each fabricated on TSMC's custom 4NP process, are connected by a 10 TB/s die-to-die interconnect that NVIDIA calls NV-HBI (NVIDIA High Bandwidth Interface). The two dies present themselves to software as a single GPU — a programmer writes CUDA kernels against one logical device, and the driver handles partitioning work across the dies. This is a significant engineering achievement: previous multi-die GPUs (notably AMD's MI300) exposed cross-die latency to programmers; Blackwell hides it behind a coherent memory model and a transparent execution fabric.

The total transistor count across both dies is 208 billion — more than double the 80 billion of the Hopper H100. The package integrates eight HBM3e stacks for 192 GB of memory, providing 8 TB/s of aggregate bandwidth. The combined silicon delivers roughly 20 PFLOPS of FP8 compute and, with the new FP4 tensor core support, approximately 40 PFLOPS at half that precision.

### 2.2 Second-Generation Transformer Engine and FP4

The Transformer Engine, introduced in Hopper, dynamically scales numerical precision per layer based on activation distributions. Blackwell extends this with native support for FP4 — a 4-bit floating-point format with one sign bit, two exponent bits, and one mantissa bit. FP4 doubles arithmetic throughput compared to FP8 at the cost of significant precision loss, which the Transformer Engine compensates for with finer-grained scaling, micro-block quantization, and selective layer promotion to higher precision.

For inference workloads on quantized models, the throughput improvement is substantial. A 405-billion-parameter Llama 3.1 model running in FP4 on a single Blackwell GPU achieves roughly the same per-token latency as the same model running in FP8 on an H100 — but with half the memory footprint and half the energy per token. The catch is that FP4 is essentially useless for training and marginally useful for high-quality inference without careful calibration. Most production deployments use FP4 for the FFN weights (where activation outliers are less severe) and retain FP8 or BF16 for attention.

### 2.3 Decompression Engine and Sparsity

Blackwell adds a hardware decompression engine that accelerates the loading of compressed weights from HBM. For models stored in formats like GPTQ, AWQ, or NVIDIA's own NVFP4, the decompression happens on the fly as weights stream into the tensor cores, avoiding a separate dequantization pass. The same engine accelerates 2:4 structured sparsity, doubling effective tensor core throughput when 50% of weights are pruned in the required pattern.

## 3. The GB200 Superchip

### 3.1 Grace + Two Blackwells

The GB200 module pairs one Grace CPU (a 72-core ARM Neoverse V2 design with 480 GB of LPDDR5X) with two Blackwell GPUs over a 900 GB/s NVLink-C2C interconnect. The CPU and GPUs share a unified memory address space, allowing kernels to dereference CPU memory directly without explicit DMA, and allowing CPU code to access GPU memory through the same coherent fabric.

The coherent memory model has two practical consequences for LLM workloads. First, model weights too large for HBM can be staged in Grace's LPDDR5X and accessed by GPU kernels without explicit prefetching — a 1.5 trillion-parameter model can fit in the combined 480 GB CPU + 384 GB GPU memory of a single GB200 module. Second, KV cache spillover from GPU to CPU memory becomes nearly transparent: when a long-context inference request exceeds available HBM, the runtime can transparently offload the oldest cache pages to LPDDR5X, paying only the bandwidth cost on subsequent attention operations.

### 3.2 Power and Cooling

The GB200 module dissipates approximately 2,700 watts at peak load — roughly 1,200 W per Blackwell GPU and 300 W for the Grace CPU. This is well beyond the practical limits of forced-air cooling, which is why NVIDIA designed the module from the outset for direct-to-chip liquid cooling. Cold plates sit directly on the GPU and CPU dies, with coolant entering at approximately 25°C and exiting at 35°C. The module's power density makes traditional rack air cooling infeasible: a fully populated NVL72 cabinet rejects 120 kW of heat, more than ten times the density of a typical air-cooled server rack.

## 4. The NVL72 Rack Architecture

### 4.1 Topology

The NVL72 rack contains 18 compute trays, each holding two GB200 superchips, for a total of 36 Grace CPUs and 72 Blackwell GPUs. The trays are interconnected by nine NVLink switch trays, each containing two NVSwitch chips. Every Blackwell GPU has 18 fifth-generation NVLink ports, each providing 100 GB/s of bidirectional bandwidth, for a total of 1.8 TB/s per GPU. The NVSwitch fabric provides full all-to-all connectivity across all 72 GPUs at this rate, yielding 130 TB/s of aggregate bisection bandwidth across the rack.

The significance of this is hard to overstate. In an H100-based DGX system, a single NVLink domain spans 8 GPUs. Crossing the boundary to a 9th GPU requires going over InfiniBand at roughly 50 GB/s per GPU — a 18× drop in bandwidth and an order-of-magnitude increase in latency. The NVL72 raises the NVLink domain from 8 to 72 GPUs, eliminating this cliff entirely for models that fit within the rack. A 1.8 trillion-parameter model can be tensor-parallelized across 72 GPUs at full NVLink speed, with no InfiniBand involvement until cross-rack scaling is needed.

### 4.2 The "One Giant GPU" Programming Model

NVIDIA's marketing claim that the NVL72 presents itself as "one giant GPU" is partially accurate. From the perspective of CUDA tensor parallelism libraries — Megatron-LM, vLLM, TensorRT-LLM — the 72 GPUs appear as a single tensor-parallel group with uniform bandwidth between any pair. This dramatically simplifies the partitioning problem: rather than carefully placing layers to minimize inter-node communication, a programmer can use uniform tensor parallelism degree 72 and let the NVLink fabric handle the rest.

The model is not entirely uniform — there are still subtle latency differences depending on which switch hop a transfer takes — but for the synchronous collective operations that dominate transformer execution (all-reduce, all-gather, reduce-scatter), the latency variation is small enough to be ignored. NCCL on the NVL72 achieves all-reduce bandwidth efficiencies above 90% across all 72 GPUs, compared to 60-70% on typical 8-GPU H100 nodes scaled out over InfiniBand.

### 4.3 Memory Aggregation

With 72 Blackwell GPUs at 192 GB each, the rack provides 13.5 TB of HBM3e memory in a single NVLink domain. Adding the 36 Grace CPUs at 480 GB each yields another 17 TB of LPDDR5X, for a total of 30 TB of coherent memory. This is enough to hold the full weights and KV cache of any LLM in production today, including unannounced research models in the 5-10 trillion parameter range.

For inference, the practical implication is that an entire model fits in fast memory with significant headroom for KV cache. A 1.8T parameter model in FP8 occupies 1.8 TB of weight storage, leaving over 11 TB of HBM for KV cache and activations — enough to support thousands of concurrent inference sessions or a handful of sessions with multi-million-token contexts.

## 5. Inference at Rack Scale

### 5.1 Throughput vs. Latency Tradeoffs

The NVL72 enables a fundamentally different inference deployment topology. Where an H100 cluster forces operators to choose between latency-optimal small tensor parallelism (TP=8) and throughput-optimal large tensor parallelism with InfiniBand penalties, the NVL72 allows simultaneously high TP and low latency. Tensor parallelism degree 72 with 100 GB/s links between every GPU pair achieves all-reduce latencies under 50 microseconds — fast enough that the per-token latency of large models drops to levels previously achievable only on much smaller models.

NVIDIA's published benchmarks claim 30× inference throughput improvement over an H100 cluster of equivalent GPU count for trillion-parameter models. The 30× factor is composed of approximately 2.5× from FP4 vs. FP8, 2× from improved tensor core throughput, and 6× from eliminating the InfiniBand bottleneck for tensor parallelism. Real-world deployments typically see 8-15× improvements depending on model architecture and workload.

### 5.2 Disaggregated Prefill and Decode

The NVL72's massive NVLink domain makes disaggregated inference (separating prefill and decode onto different hardware pools) significantly more attractive. Prefill is compute-bound and benefits from large parallelism; decode is memory-bandwidth-bound and benefits from moderate parallelism with high HBM-per-token efficiency. On the NVL72, both pools can share the same NVLink domain, transferring KV cache between prefill and decode workers at NVLink speeds rather than InfiniBand speeds.

A typical disaggregated configuration might dedicate 24 GPUs to prefill (running TP=8 with 3 replicas) and 48 GPUs to decode (running TP=8 with 6 replicas), with KV cache transfer between them happening over NVLink. The transfer cost — moving roughly 100 KB of cache per token per layer — is negligible at NVLink speeds, where it would consume meaningful time on InfiniBand.

## 6. Training at Rack Scale

For training workloads, the NVL72 changes the calculus of 3D parallelism. The conventional approach uses tensor parallelism within a node (8 GPUs over NVLink), pipeline parallelism across nodes (over InfiniBand), and data parallelism across the cluster. The NVL72 collapses tensor and pipeline parallelism into a single NVLink domain, allowing TP × PP up to 72 within one rack at full bandwidth.

The result is that pipeline bubble overhead — the idle time GPUs spend waiting for activations to flow through the pipeline — drops dramatically. Where an H100 cluster might run pipeline parallelism degree 8 with bubble overhead of 10-15%, an NVL72 can run pipeline parallelism degree 1 (or 2 for memory reasons) and tensor parallelism degree 36 or 72, eliminating pipeline bubbles entirely. For a 1 trillion parameter model, this translates to roughly 25% better hardware utilization end-to-end.

## 7. Networking Beyond the Rack

A single NVL72 is rarely deployed alone. Production training clusters chain dozens or hundreds of NVL72 racks together over InfiniBand or NVIDIA's Spectrum-X Ethernet fabric. Each rack has a top-of-rack switch with 800 GB/s of uplink bandwidth, supporting non-blocking communication with up to seven other racks at full speed.

The hierarchy is now: ultra-fast NVLink within a rack (130 TB/s aggregate), fast InfiniBand between racks (a few TB/s aggregate per rack), and slower Ethernet between datacenters. Training a foundation model on a 16-rack NVL72 cluster requires careful placement: tensor parallelism stays within one rack, pipeline parallelism spans 2-4 racks, and data parallelism spans the remaining dimension. The NVIDIA Megatron-Core library has been updated to be NVL72-aware, automatically selecting parallelism degrees based on the topology.

## 8. Power, Cooling, and Datacenter Implications

A single NVL72 rack draws 120 kW under full load — more than 30 typical air-cooled server racks. The cooling infrastructure required is substantial: closed-loop liquid cooling with rear-door heat exchangers or in-rack coolant distribution units, supply temperatures around 25-30°C, and return temperatures up to 45°C. Most existing datacenters cannot accommodate NVL72 racks without significant retrofitting, which is why hyperscalers building Blackwell clusters are constructing purpose-built facilities — Microsoft's Wisconsin datacenter, Meta's Louisiana facility, and xAI's Memphis Colossus expansion are all designed around 100+ kW liquid-cooled racks.

The power-per-rack figure also affects how clusters scale. A 1,000-GPU H100 cluster fits in roughly 40 air-cooled racks drawing 30 kW each (1.2 MW total). A 1,000-GPU Blackwell NVL72 cluster fits in 14 racks drawing 120 kW each (1.7 MW total). The NVL72 cluster delivers roughly 5× the FP8 throughput at 1.4× the power — a substantial improvement in performance-per-watt despite the much higher absolute draw.

## 9. Software Stack and Adoption

CUDA 12.5 and later support Blackwell natively, with no source-level changes required for most workloads. The CUDA driver handles the dual-die abstraction transparently. cuDNN, cuBLAS, and CUTLASS have been extended with FP4 kernels and Blackwell-specific tensor core paths. NCCL has been updated to take advantage of the NVL72 topology, automatically detecting the rack-scale NVLink domain and configuring collectives accordingly.

For inference frameworks, vLLM, TensorRT-LLM, and SGLang all gained Blackwell support during 2024 and 2025. TensorRT-LLM in particular is the reference framework for FP4 inference on Blackwell, providing the calibration tooling needed to convert FP16 or FP8 checkpoints to FP4 with minimal accuracy loss. PyTorch added Blackwell support in version 2.5, including a `torch.compile` backend that targets the new tensor core paths.

## 10. Outlook

The Blackwell GB200 NVL72 represents a discrete jump in the granularity of AI computing — from server to rack as the basic unit of deployment. The next generation, Rubin (announced for 2026 production), continues this trajectory: NVIDIA has previewed a rack-scale system with 144 GPUs, 1.6 TB HBM4 per GPU, and a new NVLink-6 fabric providing 200 GB/s per port. The trend lines suggest that by the end of the decade, "one giant GPU" will mean an entire datacenter aisle, not a single rack.

For LLM operators, the implication is that infrastructure choices increasingly resemble mainframe procurement: long lead times, custom datacenter facilities, multi-year capacity commitments, and a small handful of vendors capable of delivering at scale. The era of building inference clusters from off-the-shelf servers is ending; the era of buying pre-integrated, liquid-cooled, rack-scale AI computers has begun.
