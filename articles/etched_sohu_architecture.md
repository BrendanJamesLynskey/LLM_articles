# Etched Sohu: A Transformer-Specific ASIC

*March 2026*

## 1. Introduction to Etched

Etched is a semiconductor startup founded in 2022 with a premise that is either visionary or reckless, depending on one's confidence in the longevity of the transformer architecture. Backed by Y Combinator and having raised over $120 million in venture funding, Etched set out to build an application-specific integrated circuit (ASIC) that does exactly one thing: run transformer inference as fast as physically possible. The company's chip, called Sohu, achieves this by burning the core operations of the transformer architecture directly into silicon, eliminating the overhead of general-purpose compute hardware entirely. If transformers remain the dominant paradigm in AI for the next decade, Etched will have built one of the most efficient inference chips in history. If a fundamentally different architecture displaces transformers, the chip becomes an expensive paperweight.

This report examines the Sohu chip's design philosophy, its claimed performance characteristics, the architectural trade-offs it makes relative to GPUs and other accelerators, and the existential risk embedded in its single-architecture bet.

## 2. The Radical Premise: Architecture Burned into Silicon

### 2.1 Specialization as Strategy

The semiconductor industry spans a well-understood spectrum of specialization. At one end sit CPUs and GPUs: general-purpose processors that can execute arbitrary programs but carry enormous overhead in instruction decode, branch prediction, cache hierarchies, and scheduling logic. In the middle sit FPGAs, which offer reconfigurable hardware that can be tailored to specific workloads but sacrifice clock speed and density for flexibility. At the far end sit ASICs, where the computation is etched permanently into the chip's logic gates, yielding maximum efficiency for a single fixed workload at the cost of zero adaptability.

Etched positions Sohu at the extreme ASIC end of this spectrum, but with an unusually narrow target. Traditional AI ASICs like Google's TPU or Intel's Habana Gaudi hardcode general matrix multiply and convolution operations that are shared across many neural network architectures. Sohu goes further: it hardcodes the specific operations that define a transformer, including softmax, layer normalization, multi-head attention patterns, and matrix multiplications tuned to the specific dimension ratios common in transformer models. The chip does not merely accelerate transformers efficiently; it is physically incapable of running anything else.

### 2.2 What Gets Hardcoded

The transformer forward pass consists of a relatively small set of operations executed repeatedly across layers: query-key-value projections (dense matrix multiplies), scaled dot-product attention (batched matrix multiply followed by softmax and weighted sum), feed-forward network layers (two dense matrix multiplies with an activation function), layer normalization, and residual additions. On a GPU, each of these operations is dispatched as a separate kernel or fused kernel, decoded through the instruction pipeline, and executed on general-purpose CUDA cores that could equally well run a physics simulation or a video game. Sohu replaces this entire software stack with dedicated hardware blocks for each transformer operation. The softmax computation has its own fixed-function unit. Layer normalization has its own circuit. The attention mechanism's query-key dot product, scaling, masking, and value aggregation are implemented as a hardwired pipeline rather than a sequence of programmable instructions.

## 3. Eliminating the GPU's Generality Overhead

### 3.1 The Cost of Flexibility

A modern GPU like the NVIDIA H100 dedicates a substantial fraction of its die area and power budget to infrastructure that supports general-purpose computation. The instruction fetch and decode pipeline, warp schedulers, register files sized for arbitrary programs, L1 and L2 cache hierarchies, memory management units, and the PCIe/NVLink interface logic all consume silicon that contributes nothing to the core arithmetic of transformer inference. NVIDIA's own estimates suggest that less than half of an H100's transistor budget is dedicated to the Tensor Cores that actually perform matrix multiplication; the rest supports the programmable, flexible execution model that makes GPUs useful for everything from ray tracing to molecular dynamics.

### 3.2 Sohu's Approach

Sohu reclaims this overhead by eliminating everything that is not required for transformer inference. There are no general-purpose CUDA cores, no programmable shader pipelines, no instruction decoders for non-transformer operations, and no hardware branch predictors. The chip does not need a complex cache hierarchy because its dataflow is entirely predetermined by the transformer architecture. The silicon area that would have been spent on general-purpose infrastructure is instead allocated to more compute units, larger on-chip SRAM buffers, and wider data paths between functional blocks. Etched claims this approach yields roughly an order of magnitude more useful compute per watt and per square millimeter of silicon compared to a GPU running the same transformer workload.

## 4. Sohu Chip Design

### 4.1 Process Node and Physical Design

Sohu is fabricated on TSMC's advanced process technology, leveraging the same foundry that produces chips for NVIDIA, Apple, and AMD. The choice of TSMC provides access to leading-edge transistor density and power efficiency, which is critical for an ASIC that aims to maximize compute density. The specific process node positions Sohu competitively against the H100 (TSMC 4N) in terms of transistor density and operating voltage.

### 4.2 On-Chip SRAM vs. HBM

Like Groq's TSP, Sohu makes aggressive use of on-chip SRAM to provide high-bandwidth, low-latency access to model weights and activations. On-chip SRAM delivers dramatically higher bandwidth and lower latency than HBM, which is critical for the memory-bandwidth-bound decode phase of transformer inference. However, SRAM is far less dense than DRAM, creating the same fundamental capacity trade-off that every SRAM-heavy design faces: extraordinary bandwidth per byte at the cost of far fewer total bytes per chip. Etched addresses this by designing Sohu for multi-chip server configurations where model weights are distributed across chips, with each chip holding a partition of the parameters in its local SRAM.

### 4.3 Eight-Chip Server Configuration

Etched's reference deployment is an eight-chip server, where eight Sohu ASICs are interconnected with high-bandwidth, low-latency links within a single server chassis. This configuration is designed to hold large models like Llama 70B entirely across the collective on-chip SRAM of the eight chips, with each chip responsible for a tensor-parallel slice of the model. The eight-chip design balances the per-chip SRAM limitation against practical server form factors and interconnect complexity, providing enough aggregate memory capacity for production-scale models while keeping inter-chip communication latency low enough to maintain throughput.

## 5. Claimed Performance

### 5.1 Throughput Numbers

Etched has published performance claims that, if validated, would represent a step change in inference economics. The company claims that a single eight-chip Sohu server can generate over 500,000 tokens per second on Meta's Llama 70B model, representing more than a 10x throughput improvement over an equivalent NVIDIA H100-based server. These figures refer to aggregate server throughput across many concurrent requests rather than single-stream latency, but they imply a dramatic reduction in cost per token for high-volume inference workloads.

### 5.2 Context and Caveats

These performance claims should be interpreted with appropriate caution. Etched is a pre-revenue startup, and the benchmarks have not been independently verified by third parties at the time of writing. The comparison baseline matters significantly: a 10x advantage over a naively configured H100 is very different from a 10x advantage over an H100 running TensorRT-LLM with FP8 quantization, continuous batching, and speculative decoding. The specific batch sizes, sequence lengths, quantization levels, and measurement methodology all affect the comparison. Early-stage hardware companies have a strong incentive to present best-case numbers, and the AI accelerator space has a history of claims that do not survive contact with production workloads.

## 6. The Bet on Transformer Longevity

### 6.1 Why Etched Believes Transformers Will Dominate

Etched's entire business case rests on the assumption that the transformer architecture, introduced in 2017, will remain the dominant paradigm for large-scale AI for years to come. The company points to several factors in support of this thesis. First, the transformer has proven remarkably robust across modalities: it dominates not only natural language processing but also computer vision, speech, protein folding, and increasingly robotics and planning. Second, the scaling laws that govern transformer performance show no signs of saturating at current parameter counts and dataset sizes. Third, the enormous ecosystem of tools, frameworks, training recipes, and deployment infrastructure built around transformers creates deep switching costs that would slow any transition to an alternative architecture.

### 6.2 The Existential Risk

The counterargument is that the history of deep learning is a history of architectural disruption. Recurrent neural networks dominated sequence modeling until transformers displaced them almost overnight. Several alternative architectures have emerged that claim to match or exceed transformer performance on specific tasks while offering better computational scaling with sequence length. State-space models such as Mamba and S4, recurrence-based architectures like RWKV, and various hybrid approaches are under active research. If any of these alternatives achieves a decisive advantage on key benchmarks and gains ecosystem traction, Sohu's hardcoded transformer logic becomes not merely suboptimal but entirely useless. The chip cannot be reprogrammed, reconfigured, or adapted. It is a physical embodiment of a bet on one specific computational graph, and if that graph falls out of favor, the hardware has zero residual value.

## 7. Comparison with Other Specialized Chips

### 7.1 Sohu vs. Groq LPU

Groq's Tensor Streaming Processor shares Sohu's emphasis on SRAM bandwidth and deterministic execution but takes a fundamentally different approach to specialization. The TSP is a programmable dataflow processor with a compiler that can map arbitrary computation graphs onto its hardware; it is not limited to transformers. Groq's advantage is flexibility at the cost of some efficiency, while Sohu's advantage is maximum efficiency at the cost of all flexibility. Groq can run convolutional networks, mixture-of-experts models with non-standard routing, or future architectures that its compiler can schedule. Sohu cannot.

### 7.2 Sohu vs. Cerebras WSE

Cerebras takes yet another approach, building a wafer-scale engine with 40 GB or more of on-chip SRAM and 900,000 cores on a single massive die. Like Sohu, Cerebras benefits from enormous on-chip memory bandwidth, but the WSE is a general-purpose AI accelerator that supports both training and inference across diverse architectures. Cerebras sacrifices some per-operation efficiency relative to a purpose-built ASIC in exchange for architectural flexibility and the ability to serve as a complete AI compute platform rather than a single-architecture inference engine.

### 7.3 Sohu vs. GPUs

The comparison with GPUs is the most straightforward. GPUs offer maximum flexibility, a mature software ecosystem, support for training and inference, and compatibility with every neural network architecture ever devised. In exchange, they carry the overhead described in Section 3. For transformer inference specifically, Sohu aims to deliver dramatically better performance per watt and per dollar, but it cannot do anything else. Organizations adopting Sohu must maintain separate GPU infrastructure for training, non-transformer inference, and any experimental workloads.

## 8. Software Stack and Model Support

Etched provides a software stack that accepts models in standard formats and maps them onto Sohu's fixed-function hardware. Because the chip's dataflow is determined by the transformer architecture rather than by a general-purpose instruction set, the "compilation" process is more constrained than on a GPU or even a programmable ASIC. The software must verify that the input model conforms to the transformer patterns that Sohu supports, partition the model across the eight chips in a server, and configure the chip's data routing for the specific model dimensions. Etched has announced support for popular open-weight models including the Llama family, and the software stack is designed to handle standard transformer variants with different hidden dimensions, head counts, and layer depths. Custom architectures that deviate from the standard transformer template, such as models with novel attention patterns or non-standard normalization schemes, may require hardware-level validation to confirm compatibility.

## 9. Implications for Inference Economics

If Sohu delivers on its performance claims, the implications for inference economics are significant. A 10x improvement in throughput per server translates directly to a proportional reduction in the number of servers required to serve a given request volume, which in turn reduces capital expenditure, power consumption, cooling costs, and datacenter floor space. For hyperscale inference providers processing billions of tokens per day, even a 2-3x verified improvement in cost per token would justify the risk of architectural lock-in. The economic calculation becomes a bet: the cost savings from running transformers on purpose-built hardware for the next three to five years versus the risk of stranded assets if the industry shifts to a post-transformer architecture within that window.

## 10. Early Benchmarks and Industry Skepticism

The AI hardware community has responded to Etched's claims with a mixture of enthusiasm and skepticism. The enthusiasm stems from the genuine performance limitations of GPU-based inference and the appeal of a clean-sheet ASIC design unconstrained by backward compatibility. The skepticism stems from the difficulty of validating pre-production benchmarks, the history of AI hardware startups that failed to deliver on initial claims, and the fundamental risk of a single-architecture bet in a field that evolves rapidly. Several industry analysts have noted that Etched's claimed performance numbers would require not only superior silicon but also highly optimized data movement and memory management across the eight-chip server, which is a non-trivial systems engineering challenge. Independent benchmarks on production hardware will be essential to establishing Sohu's credibility.

## 11. Conclusion

Etched and the Sohu chip represent the logical extreme of the specialization thesis in AI hardware: if you know exactly what computation you need to perform, you can build hardware that performs it with unmatched efficiency by eliminating every transistor that does not contribute to that specific task. The transformer architecture, with its fixed and well-characterized computational graph, is an ideal target for this approach. The potential reward is enormous, with order-of-magnitude improvements in inference throughput and cost efficiency that could reshape the economics of AI deployment. The potential risk is equally stark, since a single architectural shift in the broader AI research community could render the entire hardware investment worthless. For ML engineers evaluating inference infrastructure, Sohu is worth watching closely as production hardware becomes available, but it is also a cautionary example of how deeply hardware bets are coupled to architectural assumptions that the research community has not yet settled.
