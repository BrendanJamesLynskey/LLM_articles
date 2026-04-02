# Tenstorrent Architecture: RISC-V AI Accelerators and the Open Hardware Bet

*March 2026*

## 1. Introduction

Tenstorrent occupies a unique position in the AI accelerator landscape. Founded in 2016 by Ljubisa Bajic in Toronto, the company spent its early years developing a novel AI processor architecture built around RISC-V cores and a packet-based dataflow execution model. The company's profile rose dramatically when Jim Keller, the legendary processor architect behind AMD's Zen, Apple's A-series, and contributions to x86-64 and ARM, joined as CTO in 2020 and subsequently became CEO. Keller brought not only technical credibility but a strategic vision: Tenstorrent would be the open-source alternative to NVIDIA's proprietary stack, betting that the AI hardware market would eventually reward openness over lock-in.

Unlike Groq's SRAM-only deterministic approach or Cerebras's wafer-scale integration, Tenstorrent's architecture is built on composability. The fundamental compute unit is the Tensix core, a heterogeneous processor tile containing five small RISC-V cores alongside dedicated matrix and vector engines. These Tensix cores are tiled into a 2D mesh connected by an Ethernet-based network-on-chip, and the same Ethernet protocol extends seamlessly from on-chip communication to chip-to-chip and rack-to-rack interconnect. The result is an architecture that scales from a single chip to multi-chip Galaxy systems without requiring proprietary interconnect technologies.

## 2. The RISC-V Bet

### 2.1 Open ISA vs. Proprietary Cores

Most AI accelerators use either proprietary control processors or license ARM cores for their management planes. Tenstorrent chose RISC-V, the open-source instruction set architecture originally developed at UC Berkeley. This decision has several strategic implications. First, RISC-V carries no per-chip licensing fees, which matters enormously at scale when shipping millions of units. Second, the open ISA allows Tenstorrent to customize the core microarchitecture without negotiating with a licensor. Third, it positions the company within a growing ecosystem of RISC-V tooling, compilers, and developer familiarity.

### 2.2 Customized RISC-V Implementations

Tenstorrent does not use off-the-shelf RISC-V cores. The company designs its own microarchitectures optimized for different roles within the Tensix tile. The compute-oriented baby cores are stripped down to the essentials needed for orchestrating matrix and vector operations, while the control core handles synchronization and scheduling. This heterogeneous approach allows each core to be area- and power-optimized for its specific function, rather than paying the silicon cost of a general-purpose core in every slot.

## 3. Tensix Core Architecture

### 3.1 The Five Baby Cores

Each Tensix core is a self-contained processing element built around five RISC-V baby cores, each dedicated to a specific role. Two cores handle compute orchestration, issuing instructions to the matrix and vector engines and managing data dependencies between operations. One core is responsible for data movement, coordinating transfers between local SRAM, the network-on-chip, and neighboring Tensix cores. A fourth core handles control flow, managing synchronization primitives and conditional execution. The fifth core manages Ethernet communication, handling packet formation and routing for both on-chip and off-chip data transfers.

### 3.2 Matrix and Vector Engines

Alongside the five RISC-V cores, each Tensix tile contains a dedicated matrix engine (for dense matrix multiplications in INT8, FP16, BF16, and FP32) and a vector engine (for elementwise operations, activations, normalization, and reduction). The matrix engine operates on tiles of data staged in the Tensix core's local SRAM, performing multiply-accumulate operations at high throughput. The vector engine handles the non-matmul operations that dominate the compute graph between matrix multiplications. This separation of concerns means the RISC-V cores are not performing arithmetic themselves but rather orchestrating when and how data flows through the fixed-function engines.

### 3.3 Local SRAM and Data Staging

Each Tensix core has a local SRAM buffer (typically around 1 MB) that serves as a scratchpad for operands, intermediate results, and output tiles. Data is staged into this SRAM by the data movement core before the compute cores begin issuing instructions to the matrix and vector engines. This explicit data staging model eliminates the need for hardware-managed caches and the associated unpredictability of cache misses. The programmer or compiler is responsible for orchestrating data movement, giving full control over memory access patterns.

## 4. Chip Generations

### 4.1 Grayskull (First Generation)

Grayskull was Tenstorrent's first-generation processor, featuring 120 Tensix cores arranged in a 2D mesh and operating at approximately 1 GHz. It was manufactured on a 12nm process and served primarily as a proof-of-concept for the Tensix architecture and the packet-based dataflow execution model. Grayskull demonstrated that the approach was viable for running transformer inference workloads and provided the foundation for the software stack. However, its interconnect bandwidth and memory subsystem were relatively modest, limiting performance on larger models.

### 4.2 Wormhole (Second Generation)

Wormhole represented a significant architectural refinement. It reduced the Tensix core count to 80 but substantially improved the per-core performance, interconnect bandwidth, and the Ethernet-based mesh topology. The key innovation in Wormhole was the introduction of native Ethernet links on every edge of the chip die, allowing direct chip-to-chip communication without bridges or switches. Each Wormhole chip exposes multiple 100 Gbps Ethernet ports that connect directly to neighboring chips, enabling seamless multi-chip scaling. Wormhole also introduced support for DRAM (DDR or GDDR) attached to the chip, providing capacity for model weights that exceed the aggregate on-chip SRAM.

### 4.3 Blackhole (Third Generation)

Blackhole is Tenstorrent's third-generation chip, bringing PCIe Gen5 host connectivity, support for both GDDR6 and HBM memory configurations, and an improved Tensix core with higher clock speeds and wider matrix engines. The Blackhole design targets both inference and training workloads, with improved floating-point throughput in BF16 and FP32 for gradient computation. The Ethernet mesh has been upgraded to higher per-link bandwidth, and the chip supports larger mesh topologies for multi-chip deployments. Blackhole represents the point at which Tenstorrent's architecture becomes competitive for production inference workloads on models in the 7B to 70B parameter range.

## 5. Conditional Execution and Packet-Based Dataflow

### 5.1 Data-Dependent Compute

One of the most distinctive features of the Tenstorrent architecture is its conditional execution model. Unlike GPUs, which execute the same instruction across all threads in a warp regardless of data values, Tensix cores can make data-dependent decisions about whether to execute operations. If a particular tile of data contains all zeros (common after ReLU activations or with sparse attention masks), the Tensix core can skip the associated matrix multiplication entirely. This conditional execution is managed by the control RISC-V core, which inspects metadata attached to data packets and decides whether to dispatch work to the matrix engine.

### 5.2 Packet-Based Dataflow

Data moves through the Tenstorrent architecture as packets rather than as bulk memory transfers. Each packet carries both data and metadata describing its destination, dimensions, and processing requirements. This packet-based approach enables the network-on-chip to route data dynamically based on runtime conditions, supporting workload patterns that vary across different inputs. The dataflow model means that computation is triggered by data arrival rather than by a centralized scheduler, which naturally pipelines operations across the mesh and reduces the need for global synchronization barriers.

## 6. Network-on-Chip and Ethernet Everywhere

### 6.1 Mesh Topology

The Tensix cores within a chip are connected by a 2D mesh network-on-chip (NoC) that uses Ethernet as its transport protocol. This is unusual: most AI accelerators use custom on-chip interconnects optimized for latency and bandwidth, reserving Ethernet for off-chip communication. Tenstorrent's decision to use Ethernet at every level of the hierarchy simplifies the architecture by making on-chip, chip-to-chip, and rack-to-rack communication use the same protocol and the same routing logic. A packet destined for a Tensix core on a neighboring chip traverses the same type of link as a packet moving between cores on the same die.

### 6.2 From Chip to Rack: Seamless Scaling

Because every link in the system speaks Ethernet, scaling from a single chip to a multi-chip system requires no proprietary interconnect hardware. Wormhole and Blackhole chips expose Ethernet ports on their die edges that connect directly to neighboring chips via standard PCB traces or short cables. This eliminates the need for NVLink-style proprietary bridges, InfiniBand switches, or custom ASICs for inter-chip communication. A rack of Tenstorrent chips forms a larger mesh that the software stack treats as a single logical fabric, with routing handled by the same on-chip Ethernet controllers.

### 6.3 Galaxy Multi-Chip Systems

Galaxy is Tenstorrent's reference multi-chip system architecture. A Galaxy configuration tiles multiple Wormhole or Blackhole chips into a 2D mesh, with each chip connected to its neighbors via Ethernet links. The system scales to hundreds of chips while maintaining a flat, uniform network topology. Because the Ethernet mesh is homogeneous, there is no distinction between "near" and "far" chips from a communication protocol perspective, though physical distance still affects latency. Galaxy systems are designed for both large-model inference (distributing weights across chips) and training (distributing data and gradients).

## 7. Software Stack

### 7.1 TT-Metalium

TT-Metalium is Tenstorrent's low-level programming SDK, providing direct access to the Tensix cores, the NoC, and the memory subsystem. Metalium exposes explicit control over data movement, kernel dispatch, and inter-core communication, analogous to writing CUDA kernels but targeting the Tensix architecture. Developers use Metalium to write custom kernels that orchestrate the five RISC-V baby cores, stage data into local SRAM, and issue operations to the matrix and vector engines. Metalium is open-source, published on GitHub under a permissive license.

### 7.2 TT-NN and Higher-Level Frameworks

TT-NN is a higher-level library built on top of Metalium that provides optimized implementations of common neural network operations: matrix multiplications, convolutions, attention mechanisms, layer normalization, and activation functions. TT-NN handles the complexity of tiling operations across Tensix cores, managing data layout, and exploiting the mesh topology for parallelism. Above TT-NN, Tenstorrent provides integration with PyTorch 2.0 through a custom backend, allowing users to run standard PyTorch models on Tenstorrent hardware with minimal code changes. The earlier TT-BUDA framework served a similar purpose but has been largely superseded by the TT-NN and PyTorch 2.0 integration path.

### 7.3 Open-Source Strategy

Tenstorrent has open-sourced not only its software stack but also significant portions of its hardware specifications. The Metalium SDK, TT-NN library, and associated tooling are available on GitHub. The company has also published architectural documentation for its chips, including details of the Tensix core, the NoC topology, and the Ethernet interconnect. This stands in sharp contrast to NVIDIA's closed CUDA ecosystem and even to other accelerator vendors who typically keep their hardware interfaces proprietary. The open-source strategy is a deliberate bet that community adoption and ecosystem growth will outweigh the competitive risk of disclosure.

## 8. LLM Inference Performance and Comparison

### 8.1 Inference on Tenstorrent Hardware

Tenstorrent has demonstrated inference performance on popular open-weight LLMs including Llama 2, Llama 3, Falcon, and Mistral models. On Wormhole-based n300 cards (two chips per card), the system achieves competitive token generation rates for models in the 7B to 13B parameter range. Blackhole-based systems extend this to 70B-class models distributed across Galaxy mesh configurations. Performance is most competitive on inference workloads that benefit from the conditional execution model, particularly those with sparse activations or structured sparsity in attention patterns.

### 8.2 Comparison with GPUs

Against NVIDIA GPUs, Tenstorrent's current-generation hardware does not match the raw throughput of an H100 on dense, fully optimized workloads. The H100 benefits from mature software (TensorRT-LLM, vLLM), HBM3 bandwidth, and two decades of CUDA ecosystem optimization. However, Tenstorrent competes on cost-per-token for specific model sizes and workload profiles, and its Ethernet-based scaling avoids the premium pricing of NVLink and InfiniBand infrastructure. The architecture is also better positioned for workloads with exploitable sparsity, where conditional execution can skip unnecessary computation that a GPU would perform regardless.

## 9. Business Model and Limitations

### 9.1 IP Licensing and Chips

Tenstorrent operates a dual business model. The company sells its own chips and systems (Wormhole and Blackhole cards, Galaxy systems), and it also licenses its RISC-V CPU core designs and Tensix IP to third parties for integration into their own SoCs. The IP licensing business provides a revenue stream that is less capital-intensive than chip manufacturing and positions Tenstorrent as a platform company rather than purely a chip vendor. Jim Keller has been explicit that this dual model is inspired by ARM's success in mobile and is intended to create a broad ecosystem around Tenstorrent's architecture.

### 9.2 Current Limitations

The Tenstorrent ecosystem remains early-stage in several important respects. The model zoo is limited compared to what runs optimally on NVIDIA hardware, and many popular architectures require manual optimization through Metalium to achieve peak performance rather than running efficiently out of the box through the PyTorch frontend. The software stack, while open-source, is less mature than CUDA and lacks the depth of profiling tools, debugging utilities, and community-contributed libraries that GPU developers expect. Performance gaps exist on workloads that are purely dense and regular, where GPUs excel due to their massive memory bandwidth and highly optimized kernels. Finally, as a smaller company competing against NVIDIA's entrenched ecosystem, Tenstorrent faces the classic chicken-and-egg problem: developers wait for hardware adoption before investing in software optimization, and customers wait for software maturity before committing to hardware purchases.

## 10. Conclusion

Tenstorrent's architecture represents a fundamentally different philosophy in AI accelerator design. Rather than building the fastest possible monolithic chip or the most deterministic execution engine, Tenstorrent bets on composability, openness, and programmability. The Tensix core's five RISC-V baby cores provide fine-grained control over compute, data movement, and communication. The Ethernet-everywhere interconnect eliminates proprietary networking dependencies. The conditional execution model exploits sparsity that GPU architectures leave on the table. And the open-source strategy invites the kind of ecosystem participation that has historically driven platform adoption in computing.

The limitations are real: the software stack needs maturation, the performance on dense workloads trails NVIDIA, and the model zoo requires expansion. But Tenstorrent's architecture is designed for a future in which AI hardware is not a monoculture. If the industry moves toward heterogeneous, composable AI infrastructure built on open standards, Tenstorrent is positioned to be the RISC-V of AI accelerators. Whether that future arrives, and how quickly, will determine whether the open hardware bet pays off.
