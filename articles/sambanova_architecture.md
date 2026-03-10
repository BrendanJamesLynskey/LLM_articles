# SambaNova's SN40L Architecture: Dataflow Processing for LLM Inference

*March 2026*

## 1. Company Overview

SambaNova Systems, founded in 2017 in Palo Alto by a team of Stanford researchers including Kunle Olukotun, Christopher Re, and Rodrigo Liang, set out to rethink hardware for AI workloads from first principles. Rather than adapting general-purpose GPUs or designing fixed-function accelerators, SambaNova pursued a fundamentally different compute paradigm: reconfigurable dataflow processing. The company has raised over $1.1 billion in venture funding and has shipped hardware to national laboratories, enterprise customers, and government agencies, positioning itself as a credible alternative to NVIDIA-dominated infrastructure for large-scale AI inference and training.

## 2. Reconfigurable Dataflow Architecture

### 2.1 The RDA Concept

At the core of SambaNova's approach is the Reconfigurable Dataflow Architecture (RDA), which inverts the relationship between compute and data that defines conventional processor design. In a GPU, a grid of streaming multiprocessors fetches data from a memory hierarchy, executes instructions, and writes results back. The processor is the fixed entity; data moves to it. In SambaNova's dataflow model, data moves through a spatial arrangement of compute units, with each unit performing a transformation as data streams past. The computation graph of a neural network is mapped directly onto the hardware fabric, so that activations flow from one operation to the next without round-tripping through a global memory subsystem. This eliminates much of the memory bandwidth bottleneck that constrains GPU-based inference, particularly for autoregressive LLM decoding where each token generation step involves reading the entire KV cache.

### 2.2 Contrast with GPU and LPU Approaches

This philosophy differs markedly from both NVIDIA's GPU approach and Groq's Language Processing Unit (LPU). NVIDIA GPUs rely on massive parallelism across thousands of CUDA cores backed by high-bandwidth HBM, but still suffer from the von Neumann bottleneck of shuttling data between compute and memory. Groq's LPU takes a deterministic, SRAM-only approach that eliminates HBM entirely, achieving extraordinarily low latency but at the cost of limited model capacity per chip. SambaNova's RDA occupies a middle ground: it uses a reconfigurable spatial compute fabric combined with a three-tier memory hierarchy that balances capacity, bandwidth, and latency.

## 3. The SN40L Chip

### 3.1 Reconfigurable Dataflow Units

The SN40L is SambaNova's second-generation processor, built around what the company calls Reconfigurable Dataflow Units (RDUs). Each RDU contains a large array of pattern compute units (PCUs) and pattern memory units (PMUs) connected by a reconfigurable on-chip network. The PCUs handle arithmetic operations including FP16, BF16, FP8, and INT8 formats, while the PMUs provide distributed on-chip SRAM that acts as software-managed scratchpad memory. Unlike a GPU's fixed pipeline stages, the RDU's interconnect topology and the function of each compute unit can be reconfigured at runtime to match the specific dataflow pattern required by a given layer or model architecture.

### 3.2 Three-Tier Memory Hierarchy

The SN40L implements a distinctive three-tier memory system. The first tier is on-chip SRAM distributed across the PMUs, offering the highest bandwidth and lowest latency for frequently accessed data such as model weights in active layers and attention KV caches. The second tier is on-package HBM (High Bandwidth Memory), providing several tens of gigabytes of capacity for storing larger model partitions. The third tier is off-chip DDR memory, which extends total capacity into the hundreds of gigabytes per chip, enabling very large models to reside on fewer chips than would be required with HBM-only designs. The SambaFlow compiler orchestrates data placement and movement across these tiers, prefetching weights and caching activations to keep the dataflow pipeline fed without stalls.

## 4. Software-Defined Reconfiguration and SambaFlow

### 4.1 The SambaFlow Compiler Stack

Hardware reconfigurability is only as useful as the software that drives it. SambaFlow is SambaNova's compiler and runtime stack that takes neural network graphs defined in PyTorch or other frameworks and maps them onto dataflow patterns for the RDU. The compiler performs graph-level optimizations including operator fusion, tiling, and data layout transformations, then generates a spatial configuration that assigns operations to specific PCUs and routes data through the on-chip network. This process is conceptually similar to place-and-route in FPGA toolchains, but operates at a higher level of abstraction and targets the coarser-grained RDU fabric rather than individual logic elements.

### 4.2 Software-Defined Hardware Reconfiguration

A key advantage of the SambaFlow approach is that the hardware can be reconfigured between inference requests or even between layers of the same model. This enables what SambaNova calls software-defined hardware reconfiguration: the physical compute substrate adapts its topology and data routing to the specific demands of each operation. For transformer models, this means the attention layers can use a different spatial mapping than the feed-forward layers, optimizing utilization across both compute-bound and memory-bound phases of the forward pass.

## 5. Composition of Experts

One of SambaNova's most distinctive software innovations is Composition of Experts (CoE), a framework for running multiple specialized models on shared hardware with near-zero switching overhead. Rather than deploying a single monolithic model, CoE allows organizations to maintain a collection of fine-tuned expert models, each optimized for a specific domain or task, and route queries to the appropriate expert at inference time. Because the RDU's reconfigurable fabric can swap model configurations rapidly and the three-tier memory hierarchy can hold multiple model weight sets simultaneously, the switching latency between experts is on the order of milliseconds rather than the seconds required to reload weights on a GPU. This makes CoE practical for production deployments where diverse workloads must be served from a shared infrastructure pool.

## 6. Performance Characteristics

### 6.1 Throughput Benchmarks

SambaNova has published benchmark results demonstrating competitive token generation rates on popular open-weight models. On Llama 3.1 405B, the SN40L-based SambaNova Cloud service has reported output throughput exceeding 100 tokens per second per request, with time-to-first-token latencies well under one second. For DeepSeek-R1 671B, a significantly larger mixture-of-experts model, SambaNova demonstrated inference at over 50 tokens per second, leveraging the RDU's ability to efficiently activate only the relevant expert subnetworks while keeping inactive experts resident in the lower tiers of the memory hierarchy.

### 6.2 SambaNova Cloud

The SambaNova Cloud service, launched as a public API, provides access to these performance characteristics without requiring customers to deploy on-premises hardware. The service has gained attention for offering fast inference on very large models at competitive pricing, positioning SambaNova alongside Groq, Together AI, and Fireworks as an alternative to hyperscaler inference endpoints. The cloud offering also showcases the CoE capability, allowing users to route between multiple specialized models through a unified API.

## 7. Multi-Chip Scaling and DataScale Systems

### 7.1 Rack-Scale Architecture

The SN40L is designed for multi-chip scaling through SambaNova's DataScale systems. A single DataScale rack houses eight RDU nodes interconnected by a high-bandwidth chip-to-chip fabric that extends the dataflow paradigm across node boundaries. The SambaFlow compiler handles multi-chip partitioning automatically, splitting model graphs across RDUs using a combination of tensor, pipeline, and expert parallelism strategies. This rack-scale approach allows SambaNova to serve models with over a trillion parameters while maintaining the latency advantages of the dataflow execution model.

### 7.2 Energy Efficiency

SambaNova claims significant energy efficiency advantages over GPU-based systems, citing the elimination of redundant data movement as the primary factor. Because data flows through compute rather than being fetched from and written back to a shared memory hierarchy, the energy cost per operation is substantially lower. The company has published figures suggesting two to three times better performance per watt compared to equivalent NVIDIA A100 or H100 configurations for large model inference workloads, though independent verification of these claims at scale remains limited.

## 8. Limitations and Trade-Offs

Despite its architectural advantages, the SN40L platform faces meaningful challenges. The ecosystem surrounding SambaNova hardware is far less mature than NVIDIA's CUDA ecosystem, which benefits from decades of tooling, community support, and library development. Model support breadth is narrower: while popular architectures like Llama, Mistral, and DeepSeek are well-optimized, more exotic or rapidly evolving architectures may require additional compiler work before they can run efficiently on the RDU. The SambaFlow compilation process itself can be time-consuming for new model architectures, as the spatial mapping problem is fundamentally more complex than generating GPU kernel code. Additionally, the relatively small installed base means fewer practitioners have hands-on experience with the platform, creating a talent and knowledge gap that slows adoption. The three-tier memory hierarchy, while advantageous for capacity, introduces complexity in performance tuning that the compiler does not always resolve optimally without manual intervention.

## 9. Conclusion

SambaNova's SN40L architecture represents one of the most ambitious departures from conventional AI accelerator design in the current landscape. By committing fully to the dataflow paradigm and combining it with a reconfigurable compute fabric and a deep memory hierarchy, SambaNova has carved out a genuine performance niche, particularly for large-model inference workloads where memory bandwidth is the dominant bottleneck. The Composition of Experts framework adds a compelling software dimension that leverages the hardware's unique strengths. Whether the RDA approach can scale its ecosystem and tooling to match the reach of GPU-based platforms will determine whether SambaNova remains a specialized alternative or grows into a mainstream infrastructure choice for the next generation of AI deployment.
