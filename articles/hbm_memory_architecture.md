# High Bandwidth Memory: Architecture, Generations, and the Hidden Bottleneck of LLM Inference

*April 2026 • Technical Report*

## 1. Introduction

Almost every conversation about LLM hardware focuses on compute. Tensor cores, FLOPs, sparsity acceleration, FP4 throughput, the relentless march of arithmetic capability — these are the headline numbers in every product launch. But for inference, the actual bottleneck is rarely compute. It is memory: the bandwidth at which the GPU can read model weights and KV cache from DRAM into the registers where the math actually happens. A modern LLM inference workload spends most of its time waiting for data, not computing on data, and the rate at which data can flow from off-chip storage into the compute units is the dominant performance constraint.

High Bandwidth Memory (HBM) is the technology that makes modern AI accelerators viable. By stacking DRAM dies vertically and connecting them to the host die through a silicon interposer with thousands of parallel wires, HBM achieves bandwidths an order of magnitude higher than the GDDR memory used in consumer GPUs and roughly 30× higher than DDR memory used in CPUs. Without HBM, even the fastest tensor cores would sit idle most of the time. With it, AI accelerators can keep their compute units fed at the rates the workloads demand.

This report examines what HBM is, how it differs from other memory technologies, the generations from HBM1 through HBM4 and beyond, and why memory bandwidth remains the critical constraint on LLM inference performance.

## 2. The Memory Wall

The fundamental problem HBM solves is the gap between compute and memory bandwidth. Moore's Law has historically improved compute by roughly 2× every 18-24 months, while DRAM bandwidth has improved by perhaps 1.3× over the same period. Over decades, this divergence creates an enormous gap: a modern GPU can perform tens of thousands of arithmetic operations in the time it takes to read a single value from off-chip memory. The "memory wall" — coined by Wm. Wulf and Sally McKee in 1995 — describes the situation where compute units are starved for data because memory cannot keep up.

For traditional graphics workloads, the memory wall is partially hidden by texture caching, latency tolerance through massive parallelism, and the locality of pixel-shading operations. For LLM inference, none of these escape hatches work. The arithmetic intensity (operations per byte of memory traffic) of decode is fundamentally low — a single multiply-accumulate per weight loaded — and there is no spatial locality to exploit beyond what GQA and KV caching already provide. Decode is memory-bandwidth-bound by physics, not by software inefficiency.

For inference of a 70B-parameter model in FP16, generating one token requires reading 140 GB of weights from memory. At 3.35 TB/s of HBM3 bandwidth (an H100), this takes about 42 milliseconds — and this is the absolute floor on per-token decode latency, regardless of how fast the tensor cores are. The math is exact: more bandwidth = faster decode. More compute, all else equal, does nothing.

## 3. From DDR to GDDR to HBM

To understand HBM, it helps to compare it to the memory technologies it replaced.

### 3.1 DDR

DDR (Double Data Rate) is the standard memory used in CPUs. It is optimized for capacity, latency, and cost. DDR5, the current generation, provides per-channel bandwidth of about 50 GB/s. A typical server CPU has 8-12 DDR channels, yielding aggregate bandwidth of 300-600 GB/s. DDR uses individually-packaged DIMM modules connected to the CPU through long PCB traces, which limits its data rate but allows easy capacity expansion (you can plug in more DIMMs).

### 3.2 GDDR

GDDR (Graphics DDR) is the memory used in consumer GPUs and gaming hardware. It trades latency and capacity for bandwidth. GDDR6 and GDDR6X provide per-chip bandwidth of 16-24 GB/s, and a typical consumer GPU uses 8-12 GDDR chips for aggregate bandwidth of 500-1000 GB/s. The chips are soldered close to the GPU on the same PCB. GDDR is a good fit for graphics workloads, which have high spatial locality and benefit from raw throughput, but it cannot scale to the bandwidth that AI workloads need.

### 3.3 HBM

HBM (High Bandwidth Memory) is a fundamentally different design. Instead of placing memory chips on the same PCB as the GPU, HBM places them on the same silicon package, connected through a silicon interposer. The interposer routes thousands of parallel wires between the GPU die and the HBM stacks at very short distances, allowing much higher signaling density without the physical limits of PCB traces. Each HBM stack contains 4-12 DRAM dies stacked vertically, connected to each other and to the base die through Through-Silicon Vias (TSVs).

The result is enormous bandwidth per stack. HBM3 provides about 800 GB/s per stack, and a GPU package typically contains 5-8 stacks. An H100 with 5 HBM3 stacks reaches 3.35 TB/s. A B200 with 8 HBM3e stacks reaches 8 TB/s. These bandwidths are unattainable with off-package memory, no matter how fast the underlying DRAM cells.

## 4. HBM Generations

Each HBM generation improves on bandwidth, capacity, and energy efficiency:

**HBM1** (2013-2015) was the first commercial generation. It provided 128 GB/s per stack and was used in AMD's Fiji GPUs and Nvidia's P100. Capacity was limited to 4 GB per stack, and the technology was expensive and yield-challenged.

**HBM2** (2016-2019) doubled bandwidth to 256 GB/s per stack and increased capacity to 8 GB. It was used in NVIDIA V100, AMD MI50/MI60, and several AI accelerators. The improved manufacturing maturity made it dramatically more practical at scale.

**HBM2e** (2019-2021) was an enhanced HBM2 with 460 GB/s per stack and 16 GB capacity. It powered the A100, the first GPU to truly bring HBM-class bandwidth to AI workloads at scale.

**HBM3** (2022-2024) increased bandwidth to 800 GB/s per stack and capacity to 24 GB. It was the memory in the H100 (5 stacks, 80 GB), and its higher bandwidth enabled the bigger models and longer contexts that became standard during the generative AI boom.

**HBM3e** (2024-2025) further increased bandwidth to 1.2 TB/s per stack and capacity to 36 GB. It powered the H200 and is used in the Blackwell B200 (8 stacks, 192 GB total). HBM3e is the current dominant generation in production AI accelerators.

**HBM4** (2025-2026) is sampling now. It targets 1.6-2.0 TB/s per stack and 48 GB capacity, with first deployments expected in next-generation AI chips (Rubin from NVIDIA, MI400 from AMD). The HBM4 spec also introduces a wider 2048-bit interface, double the 1024-bit interface of HBM3.

**HBM4e** is on the JEDEC roadmap for 2027 with another bandwidth increase, and HBM5 has been discussed for 2028-2030.

The pattern is consistent: each generation roughly doubles bandwidth and adds capacity, but the gains are increasingly hard-won. Power consumption rises with each generation, manufacturing yield is increasingly challenging, and the cost per GB has fallen more slowly than capacity has grown.

## 5. The Manufacturing Challenge

HBM is among the most difficult products in the semiconductor industry to manufacture. A single HBM3e stack contains 12 DRAM dies, a base die, and tens of thousands of TSVs that must be perfectly aligned. The dies are bonded with copper-to-copper hybrid bonding at micron precision. Yield issues at any layer affect the entire stack, and yield issues at the package level can render an otherwise-good GPU unusable.

The supply chain is dominated by three vendors: SK hynix (the dominant supplier and the first to ship HBM3 and HBM3e), Samsung (which ships HBM but has had qualification challenges), and Micron (the most recent entrant, shipping HBM3e to NVIDIA in 2024-2025). NVIDIA's relationship with SK hynix has been strategic for years, and the constraint on Blackwell production volumes through 2024-2025 was largely HBM availability rather than GPU silicon.

The cost is significant. HBM3e is estimated to cost $15-25 per GB to manufacture, compared to roughly $3-5 per GB for GDDR6 and $1-2 per GB for DDR5. A B200 with 192 GB of HBM3e contains $3,000-5,000 of HBM alone — a meaningful fraction of the total chip cost.

## 6. Why Bandwidth Matters Most

For LLM inference, the dominance of memory bandwidth over compute is easy to demonstrate. Consider the per-token decode time for a model with N parameters in fp16:

```
decode_time ≈ (2N bytes) / (HBM bandwidth)
```

For N = 70B and HBM bandwidth = 8 TB/s (B200), decode_time ≈ 17.5 ms per token. This is the floor — it cannot go faster regardless of compute capability. For the same model on an H100 (3.35 TB/s), the floor is about 42 ms.

The B200's compute is roughly 5× the H100's, but because both are bandwidth-bound for decode of large models, the actual speedup is closer to 2.4× — which matches the bandwidth ratio (8/3.35 ≈ 2.4). The compute headroom on the B200 is wasted on decode workloads; it can only be exploited by prefill and by smaller models that fit comfortably in the bandwidth budget.

This is why every major optimization in LLM inference targets memory: KV cache quantization (smaller cache → less bandwidth), GQA and MQA (smaller cache), prefix caching (avoid re-reading), continuous batching (amortize bandwidth across many requests), tensor parallelism (split bandwidth demand across GPUs), and disaggregated serving (separate the bandwidth-bound and compute-bound phases). All of these are techniques to escape the bandwidth wall, not the compute wall.

## 7. Capacity vs. Bandwidth

HBM provides both capacity and bandwidth, but at any point in time these two improve at different rates. HBM3e at 192 GB total per GPU is generous in capacity terms — enough to hold a 70B model with substantial KV cache headroom. But for trillion-parameter models, even 192 GB is insufficient, and the model must be distributed across multiple GPUs through tensor parallelism. This is one of the main reasons rack-scale architectures like the GB200 NVL72 matter: by aggregating HBM across 72 GPUs, you get 13.5 TB of fast memory in a single coherent address space.

The HBM4 generation will increase per-stack capacity to 48 GB, enabling future GPUs with 384-512 GB of HBM. This is enough to fit very large models on a single GPU, but the bandwidth still scales with stack count, so multi-GPU configurations remain necessary for high-throughput serving.

## 8. Power and Thermal Implications

HBM is power-hungry. An HBM3e stack draws roughly 30-50 watts at full bandwidth utilization, and a GPU with 8 stacks burns 250-400 watts in HBM alone. This is comparable to the power draw of the GPU's compute units. Cooling HBM stacks is a significant engineering challenge because the stacks sit on the same package as the hot GPU die and inherit much of its thermal envelope.

For the Blackwell B200, the power and thermal characteristics of HBM3e are one of the reasons the part requires liquid cooling. Air cooling cannot reliably remove the combined ~1200 W of GPU + HBM heat from a single package.

## 9. Compute-in-Memory and PIM

A long-running research direction is to move computation closer to (or into) the memory itself, eliminating the need to ship data to a separate compute die. Processing-in-Memory (PIM) and Compute Express Link (CXL) memory pooling are two flavors of this idea.

Samsung has shipped HBM-PIM products with simple compute units integrated into the HBM base die. These can perform reduction-style operations directly on the data as it streams out of the DRAM, reducing the bandwidth demand on the GPU. The performance gains for transformer workloads are real but modest (10-20% on memory-bound layers), and adoption has been limited.

SK hynix's AiM (Accelerator-in-Memory) is a similar product, with GDDR6-AiM chips that integrate multiply-accumulate units. These have been used in research demonstrations of near-memory transformer inference, but no commercial product has emerged.

The longer-term direction is unclear. PIM is conceptually appealing — moving compute to data is more efficient than moving data to compute — but it requires new programming models, new compilers, and new system designs. The maturity gap between standard HBM and PIM-augmented HBM is large, and NVIDIA's overwhelming software ecosystem creates strong inertia against alternative approaches.

## 10. The Future

The trajectory of HBM is clear: more bandwidth, more capacity, more power, and more cost per generation. HBM4 will arrive in 2026 with 1.6-2.0 TB/s per stack. HBM4e will follow in 2027. The fundamental physics of stacked DRAM and silicon interposers are well-understood, and barring a major manufacturing breakthrough, the rate of improvement is bounded by what these technologies can deliver.

The likely escape from the HBM cost trap is not better DRAM but better memory architectures around the DRAM. Hybrid HBM + LPDDR designs (like NVIDIA's Grace+Hopper, where Grace's LPDDR provides additional capacity at lower bandwidth) are one path. CXL memory pooling, where a separate memory tier sits between HBM and persistent storage, is another. Co-packaged optics may allow memory to live further from the compute die without losing too much bandwidth, easing the integration constraints that currently force HBM onto the GPU package.

For the next several years, however, HBM is the only game in town for AI accelerators. Every major chip vendor uses it. Every major model serving framework optimizes for it. And every benchmark that purports to measure LLM inference performance is, in the final analysis, a benchmark of HBM bandwidth. The hidden bottleneck of LLM inference is also the most expensive component in the chip — and the constraint that shapes nearly every architectural decision in modern AI hardware.

## 11. Conclusion

HBM is the unsung enabler of modern AI. While compute capability gets the headlines, memory bandwidth determines what is actually achievable in production. The progression from HBM1 to HBM4e represents one of the steepest hardware improvement curves in the industry, and it has been essential to every generation of large language model deployment.

For practitioners, the practical lesson is to treat memory bandwidth as the primary capacity-planning constraint for LLM inference. Compute is rarely the bottleneck for decode; bandwidth is. When choosing accelerators, the relevant number is HBM bandwidth, not peak FLOPs. When designing inference architectures, the relevant question is how to minimize bytes moved from HBM, not how to maximize operations performed. This framing leads to dramatically different decisions than the compute-first mental model that most engineers bring to GPU programming, and it produces systems that scale far better in production.
