# Intel Gaudi Architecture: Habana Labs' AI Accelerator

*March 2026*

## 1. Habana Labs and Intel's AI Accelerator Strategy

Habana Labs was founded in 2016 in Israel by David Dahan and a team of chip architects with deep experience in networking and signal processing ASICs. The company's founding thesis was that AI accelerators needed to treat compute and networking as co-equal design priorities, rather than bolting network interfaces onto the side of a compute chip as an afterthought. This philosophy led to the Gaudi line of training and inference accelerators, which integrate high-speed Ethernet networking directly onto the processor die alongside dedicated matrix math engines. Intel acquired Habana Labs in December 2019 for approximately $2 billion, making it a central pillar of Intel's strategy to compete with NVIDIA in the data center AI market. The acquisition brought not only silicon but also the SynapseAI software stack, which Intel has continued to develop as an alternative to CUDA for PyTorch-based workloads.

## 2. Design Philosophy: Separation of Compute and Networking

### 2.1 The Network-on-Chip Approach

The defining architectural decision in every generation of Gaudi is the integration of RDMA-capable Ethernet network interfaces directly on the processor die. Where NVIDIA relies on proprietary NVLink for intra-node GPU communication and InfiniBand (via Mellanox/ConnectX adapters) for inter-node traffic, Gaudi embeds standard Ethernet RDMA NICs into the accelerator itself. This eliminates the need for external network interface cards, top-of-rack switches for scale-out within a pod, and the proprietary protocols that lock customers into a single vendor's networking ecosystem. The Gaudi approach means that the same Ethernet fabric handles both intra-node and inter-node communication, simplifying cluster design and reducing the bill of materials for large-scale deployments.

### 2.2 RDMA-Based Scale-Out Without External Switches

Because each Gaudi processor includes multiple 100 GbE or 200 GbE RDMA ports, a cluster of Gaudi accelerators can be wired in a direct-connect topology without requiring expensive external switches for the accelerator-to-accelerator communication plane. In an eight-card server, each Gaudi chip connects directly to every other chip via its integrated ports, forming a fully connected mesh. For multi-node scale-out, the remaining ports connect to standard Ethernet switches, but the total switch port count and cost are substantially lower than an equivalent InfiniBand deployment. This is a meaningful economic advantage for organizations building training or inference clusters at the 100- to 1,000-accelerator scale.

## 3. Gaudi 1

The first-generation Gaudi processor, launched in 2019, was fabricated on TSMC's 7nm process and featured eight Tensor Processor Cores (TPCs) alongside a pair of matrix math engines (MMEs) optimized for GEMM operations in BF16 and FP32. The chip included 32 GB of HBM2 memory providing approximately 1 TB/s of bandwidth, and ten 100 GbE RDMA NICs integrated on-die. Gaudi 1 was positioned as a training accelerator offering competitive price-performance against the NVIDIA A100, particularly for NLP workloads. Habana published MLPerf Training benchmarks showing Gaudi 1 achieving comparable time-to-train on BERT and ResNet-50 at a lower system cost. While Gaudi 1 did not achieve wide commercial adoption, it established the architectural template that would carry forward into subsequent generations and validated the integrated-networking approach in production deployments at several cloud providers.

## 4. Gaudi 2 Architecture

### 4.1 Compute and Memory

Gaudi 2, released in 2022 and fabricated on TSMC's 7nm process, represented a substantial generational leap. The chip doubled the matrix math engines to two, each capable of executing BF16 and FP32 GEMM operations, and added support for FP8 (both E4M3 and E5M2 formats) to enable mixed-precision training and inference with reduced memory footprint. The 24 TPCs were significantly enhanced with wider VLIW execution pipelines and expanded local SRAM. Memory was upgraded to 96 GB of HBM2e providing 2.45 TB/s of bandwidth, a substantial improvement over Gaudi 1's 32 GB at 1 TB/s. This capacity allowed Gaudi 2 to hold large models entirely in on-chip HBM, including 70B parameter models in FP8 precision on a single card.

### 4.2 Integrated Networking

Gaudi 2 expanded the integrated networking to 24 ports of 100 GbE RDMA, yielding an aggregate bisection bandwidth of 2.4 Tb/s per chip. These 24 ports are split between intra-node connectivity (typically 21 ports forming a full mesh among eight cards in a server) and inter-node scale-out (3 ports connecting to the external fabric). The integration of all networking on-die means that a Gaudi 2 server requires no NVLink bridges, no NVSwitch chips, and no dedicated network adapters for the accelerator interconnect, reducing system complexity and cost relative to an equivalent NVIDIA H100 HGX configuration.

## 5. Gaudi 3

### 5.1 Architecture Overview

Gaudi 3, announced in 2024 and fabricated on TSMC's 5nm process, is the most capable processor in the Habana lineage. The chip features 64 Tensor Processor Cores, a dramatic increase from Gaudi 2's 24, alongside enhanced matrix math engines with native support for FP8 delivering a claimed 4x improvement in FP8 throughput over the previous generation. Memory capacity rises to 128 GB of HBM2e with bandwidth exceeding 3.7 TB/s, placing it in the same memory class as the NVIDIA H100 (80 GB HBM3, 3.35 TB/s) while actually exceeding it in raw capacity. The additional HBM is particularly valuable for serving large language models, where KV cache memory consumption grows linearly with sequence length and batch size.

### 5.2 Networking and Scale-Out

Gaudi 3 upgrades the integrated networking to eight ports of 200 GbE RDMA (1.6 Tb/s aggregate per chip), reflecting the industry's transition from 100 GbE to 200 GbE fabric speeds. While the port count is lower than Gaudi 2's 24 ports, the per-port bandwidth doubles, maintaining high bisection bandwidth while simplifying cabling. The 200 GbE ports support RoCE v2 (RDMA over Converged Ethernet) for low-latency collective operations, enabling efficient all-reduce and all-gather patterns across multi-node clusters without proprietary interconnects.

## 6. Tensor Processor Core Architecture and VLIW ISA

### 6.1 TPC Design

The Tensor Processor Core is Habana's programmable compute unit, distinct from the fixed-function matrix math engines. Each TPC implements a Very Long Instruction Word (VLIW) instruction set architecture capable of issuing multiple independent operations per cycle across its functional units: a vector processing unit, a scalar processing unit, a load/store unit, and a special function unit. The VLIW approach allows the compiler to exploit instruction-level parallelism explicitly, packing independent operations into a single wide instruction word rather than relying on hardware to detect parallelism at runtime. Each TPC has its own local SRAM scratchpad, enabling data reuse patterns that reduce pressure on the HBM bandwidth.

### 6.2 Role in the Compute Pipeline

The TPCs handle operations that do not map well onto the systolic array structure of the MMEs: activation functions, layer normalization, softmax, rotary positional embeddings, and custom fused kernels. Because the TPCs are fully programmable via their VLIW ISA, developers can write custom TPC kernels in C-like syntax using the Habana TPC programming interface, giving Gaudi a level of programmability that fixed-function accelerators lack. This programmability is critical for supporting the rapidly evolving landscape of transformer variants, where new activation functions, attention mechanisms, and normalization schemes emerge regularly.

## 7. Software Stack

### 7.1 SynapseAI and the Habana Bridge for PyTorch

The SynapseAI software stack is Gaudi's equivalent of NVIDIA's CUDA ecosystem. At its core is the graph compiler, which takes a PyTorch model, captures the computation graph via the Habana PyTorch bridge (using the torch.hpu device), and lowers it through a series of optimization passes to native Gaudi instructions. The bridge provides a drop-in device abstraction: existing PyTorch code that runs on CUDA can typically be ported by replacing device references from cuda to hpu with minimal code changes. SynapseAI handles operator fusion, memory planning, TPC kernel selection, and scheduling of operations across the MMEs and TPCs.

### 7.2 DeepSpeed and Distributed Training Integration

Gaudi's software stack includes first-class integration with Microsoft's DeepSpeed library, supporting ZeRO stages 1 through 3, pipeline parallelism, and tensor parallelism for distributed training. The Habana DeepSpeed fork is maintained in close collaboration with Microsoft and supports the full range of DeepSpeed features including mixed-precision training, gradient accumulation, and activation checkpointing. For large-scale training runs, the integrated RDMA networking combined with DeepSpeed's communication-efficient algorithms allows Gaudi clusters to scale efficiently across hundreds of accelerators.

### 7.3 vLLM Support and Inference Serving

For inference workloads, Habana has contributed a Gaudi backend to the vLLM serving framework, enabling PagedAttention, continuous batching, and tensor-parallel inference on Gaudi hardware. This is a strategically important integration because vLLM has become the de facto standard for LLM serving, and supporting it means that organizations can deploy the same serving infrastructure on Gaudi that they use on NVIDIA GPUs. The Gaudi vLLM backend supports FP8 inference, chunked prefill, and multi-card tensor parallelism, with performance optimizations specific to the Gaudi memory hierarchy and networking topology.

## 8. Inference Performance and Cost-Performance

### 8.1 Llama Benchmarks

Intel has published benchmark results showing Gaudi 2 and Gaudi 3 achieving competitive inference throughput on Llama 2 and Llama 3 models. On Llama 2 70B, a single Gaudi 2 server (eight cards) delivers throughput comparable to an eight-GPU H100 HGX system while consuming less power and costing significantly less. Gaudi 3 extends this advantage with its FP8 performance improvements, showing particularly strong results on Llama 3 70B and Llama 3 8B inference. Independent benchmarks have generally confirmed that Gaudi offers competitive tokens-per-second performance at a lower price point, with the strongest results at moderate batch sizes typical of production inference serving.

### 8.2 Intel Tiber AI Cloud

Intel operates the Tiber AI Cloud (formerly the Intel Developer Cloud), which provides on-demand access to Gaudi 2 and Gaudi 3 instances. The cloud offering serves dual purposes: it provides a low-friction evaluation path for organizations considering Gaudi adoption, and it generates benchmark data in a controlled environment that Intel uses for competitive comparisons. Tiber AI Cloud pricing has been set aggressively below comparable NVIDIA GPU cloud instances, reflecting Intel's strategy of competing on cost-performance rather than raw performance leadership.

## 9. Competitive Comparison

### 9.1 NVIDIA H100 and H200

The NVIDIA H100 remains the dominant accelerator in the data center, offering 80 GB of HBM3 at 3.35 TB/s, 989 TFLOPS of FP16 compute, and a mature CUDA software ecosystem with thousands of optimized libraries. The H200, with 141 GB of HBM3e at 4.8 TB/s, extends this lead in memory capacity and bandwidth. Gaudi 3's 128 GB HBM2e at 3.7 TB/s places it between the H100 and H200 in memory bandwidth, and ahead of the H100 in capacity. However, NVIDIA's advantage lies less in raw hardware specifications and more in the depth of its software ecosystem, the breadth of model support, and the network effects of a developer community that has spent a decade building on CUDA.

### 9.2 AMD MI300X

AMD's Instinct MI300X offers 192 GB of HBM3 at 5.3 TB/s, the largest memory capacity among current accelerators. Like Gaudi, AMD competes primarily on cost-performance and memory capacity rather than ecosystem breadth. The MI300X uses standard Infinity Fabric for inter-chip communication and relies on ROCm as its software stack, which faces similar ecosystem maturity challenges to SynapseAI. Both AMD and Intel are pursuing a strategy of leveraging open-source frameworks (PyTorch, vLLM, DeepSpeed) as the compatibility layer that reduces the switching cost from CUDA.

## 10. Multi-Node Scaling

Gaudi's integrated Ethernet RDMA networking provides a distinctive advantage for multi-node scaling. In a typical Gaudi 3 cluster, the eight accelerators within a server are connected via their integrated 200 GbE ports in a mesh topology, while additional ports connect to spine switches for inter-node communication. Because the RDMA capability is built into the accelerator itself, collective operations (all-reduce, all-gather, reduce-scatter) execute with minimal software overhead and without the latency penalty of traversing a PCIe bus to reach an external network adapter. For training workloads that spend a significant fraction of time in gradient synchronization, this architectural advantage translates into higher scaling efficiency at 64-, 128-, and 256-node configurations compared to systems that rely on external network adapters.

## 11. Limitations

### 11.1 Software Ecosystem Lag

The most significant barrier to Gaudi adoption is the software ecosystem gap relative to CUDA. While SynapseAI supports the core PyTorch workflow and major frameworks like DeepSpeed and vLLM, the long tail of CUDA libraries, custom kernels, research code, and tooling does not exist for Gaudi. Organizations with complex inference pipelines that rely on TensorRT-LLM, Triton kernels, or custom CUDA extensions face a porting burden that may outweigh the cost savings. Intel has invested heavily in closing this gap, but the ecosystem deficit remains the primary reason that technically competitive Gaudi hardware has not achieved proportional market share.

### 11.2 Smaller Community and Fewer Supported Models

The Gaudi user community is orders of magnitude smaller than the CUDA community, which means fewer tutorials, fewer Stack Overflow answers, fewer pre-optimized model implementations, and slower identification and resolution of bugs. The Hugging Face Optimum Habana library provides optimized implementations of popular models, but the catalog of validated, optimized models is a fraction of what is available for NVIDIA GPUs. For organizations deploying well-established models like Llama or Mixtral, this is manageable. For those working with novel architectures or custom model variants, the lack of community support can be a significant friction point.

### 11.3 Market Perception and Strategic Uncertainty

Intel's broader AI strategy has undergone multiple pivots, from the Nervana acquisition to the Habana acquisition to the integration of both under the Intel Datacenter and AI Group. This organizational turbulence has created uncertainty about Intel's long-term commitment to the Gaudi line, particularly as Intel simultaneously develops its own GPU architecture (Ponte Vecchio, now Falcon Shores). Customers evaluating multi-year infrastructure investments need confidence that the hardware platform will receive continued software investment, driver updates, and next-generation silicon, and Intel's track record of strategic shifts gives some pause.

## 12. Conclusion

The Intel Gaudi architecture represents a thoughtfully differentiated approach to AI acceleration. By integrating RDMA-capable Ethernet networking directly on the processor die, Habana Labs addressed one of the most expensive and vendor-locked components of AI infrastructure. The combination of programmable Tensor Processor Cores with their VLIW ISA, dedicated matrix math engines, and generous HBM capacity creates a processor that is genuinely competitive with NVIDIA's offerings on performance benchmarks while offering meaningful cost advantages. Gaudi 3's 64 TPCs, 128 GB HBM2e, and 200 GbE integrated networking make it a credible choice for both training and inference at scale. The challenge, as with every NVIDIA alternative, is the software ecosystem: the depth of CUDA's libraries, the breadth of its community, and the inertia of existing deployments create a moat that raw hardware performance alone cannot cross. Gaudi's path to broader adoption depends on continued investment in SynapseAI, expansion of the validated model catalog, and the strategic patience to build a developer community that can sustain itself. For cost-conscious organizations willing to invest in the porting effort, Gaudi offers a compelling alternative that avoids proprietary networking lock-in and delivers competitive performance at a lower total cost of ownership.
