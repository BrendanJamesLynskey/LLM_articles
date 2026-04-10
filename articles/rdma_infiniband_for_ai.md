# RDMA and InfiniBand for AI Training Clusters

*April 2026 • Technical Report*

## 1. Introduction

The largest LLM training jobs in production today involve tens of thousands of GPUs working on a single model for weeks or months at a time. The compute is impressive, but the engineering achievement that makes it possible is not the GPUs themselves — it is the network that ties them together. A single gradient synchronization on a 16,000-GPU cluster moves roughly 20 GB of data through the fabric every few hundred milliseconds, with strict ordering and reliability requirements. The cluster's scaling efficiency depends almost entirely on whether this synchronization completes faster than the next forward-backward pass.

InfiniBand and Remote Direct Memory Access (RDMA) are the technologies that make this scale-out networking feasible. Unlike traditional TCP/IP networking, where every packet traverses the kernel and incurs latency for protocol processing, RDMA allows one machine's network adapter to write directly into another machine's memory with microsecond-scale latency and no CPU involvement. InfiniBand is the most mature physical and protocol stack supporting RDMA at scale, and it underpins essentially every large GPU training cluster in the world.

This report examines how RDMA and InfiniBand work, why they dominate AI cluster networking, and how the convergence with Ethernet via RoCE is reshaping the landscape.

## 2. RDMA Fundamentals

### 2.1 Bypassing the Kernel

Traditional networking on Linux involves a long chain of operations for each packet: an application calls `send()`, the kernel copies data from user space to a kernel buffer, the TCP stack processes the packet, the IP layer adds headers, the network driver pushes the packet to the NIC, and an interrupt handler processes the completion. This sequence takes microseconds per packet even when the underlying network is much faster, and it scales poorly because every packet costs CPU cycles.

RDMA bypasses essentially all of this. An application running on a machine with an RDMA-capable NIC (called a Host Channel Adapter or HCA in InfiniBand terminology) registers a memory region with the NIC, then issues work requests directly to a hardware queue. The NIC reads from the registered memory, formats InfiniBand packets, transmits them, and completes the operation without any kernel involvement. On the receiving side, the destination NIC writes the incoming data directly into the application's memory and signals completion via a hardware queue. The CPU is involved only at the queue submission and completion stages — the actual data movement is entirely DMA.

The latency improvement is dramatic. A round-trip RDMA write between two machines on the same InfiniBand fabric takes 1-2 microseconds, compared to 10-50 microseconds for TCP/IP over Ethernet. The CPU utilization is also dramatically lower: a 200 Gbps RDMA transfer consumes essentially no CPU time, compared to several cores' worth of CPU for the equivalent TCP/IP throughput.

### 2.2 RDMA Verbs and Operations

The RDMA programming model is based on "verbs" — operations that an application can request the NIC to perform. The main verbs are:

- **SEND/RECV**: A two-sided operation where the sender pushes data and the receiver consumes it from a pre-posted receive buffer. Similar in spirit to a TCP message but without kernel involvement.
- **WRITE**: A one-sided operation where the sender writes directly into a specific memory address on the remote machine. The remote CPU is not notified by default.
- **READ**: A one-sided operation where the sender reads directly from a specific memory address on the remote machine.
- **ATOMIC**: Hardware-level atomic operations (compare-and-swap, fetch-and-add) on remote memory.

For most AI workloads, the dominant pattern is SEND/RECV (used by NCCL for collective operations) and WRITE (used by some optimized communication libraries for one-sided patterns). The one-sided operations are unique to RDMA and have no equivalent in TCP/IP — they enable communication patterns where the receiver does not even know data is arriving until it inspects its memory.

### 2.3 Queue Pairs and Completion Queues

The RDMA programming abstraction is built around Queue Pairs (QPs). A QP is a pair of work queues — one send queue and one receive queue — associated with a specific connection to a remote endpoint. Applications post work requests (WRs) to the send queue, and the NIC processes them in order, generating completion events on a Completion Queue (CQ) when each WR finishes.

This architecture allows for very high concurrency: a single application can have hundreds of QPs open simultaneously, with the NIC processing work from all of them in parallel. NCCL uses this aggressively, opening dedicated QPs between every pair of GPUs in a multi-node collective and pipelining work across all of them. On a 1,024-GPU cluster, a single all-reduce operation may involve thousands of concurrent QP transactions.

## 3. InfiniBand Architecture

### 3.1 Lossless Fabric

InfiniBand is fundamentally different from Ethernet in that it is a lossless fabric: under normal operation, packets are never dropped due to congestion. This is achieved through a credit-based flow control system, where every packet sent consumes a credit at the receiver, and credits are returned only when the receiver has buffer space available. If credits run out, the sender simply stops transmitting until more become available.

The lossless property is essential for RDMA because the protocols above InfiniBand assume reliable delivery and have no retransmission logic. If packets were dropped, the higher-level software would have to handle retries, defeating much of the latency benefit of RDMA. Lossless delivery also avoids the long-tail latency spikes that plague TCP under congestion, making InfiniBand much more predictable for tightly-coupled workloads.

### 3.2 Subnet Manager and Routing

InfiniBand is a managed fabric. A central Subnet Manager (SM) discovers all switches and HCAs, computes routing tables, and pushes them to every switch. The SM uses a deterministic algorithm (typically minimum-hop or fat-tree-aware) to choose a single path between every pair of endpoints, ensuring in-order delivery at the cost of giving up multipath routing for individual flows.

This contrasts with Ethernet, which uses learning bridges and ECMP for routing decisions made at each hop. InfiniBand's centralized routing is simpler and more deterministic but requires the SM to be aware of the full topology and to recompute routes when failures occur. Production InfiniBand fabrics typically run a primary and standby SM, with failover taking sub-second time.

### 3.3 Topology: Fat-Tree and Dragonfly

The dominant topology for AI training clusters is the fat-tree (also called Clos network). A fat-tree connects compute nodes to leaf switches, leaf switches to spine switches, and spine switches potentially to a core layer. The bandwidth between layers grows by a factor matching the over-subscription ratio — a 1:1 fat-tree has equal bandwidth at every layer, providing full bisection bandwidth. AI clusters typically use 1:1 or 2:1 oversubscription for the leaf-to-spine layer.

A 1024-GPU cluster might use a two-tier fat-tree: 32 leaf switches each with 32 ports (16 down to GPUs, 16 up to spines) and 16 spine switches each with 32 ports. This provides 16,384 Gbps of bisection bandwidth — roughly 16 Gbps per GPU, which is sufficient for typical training workloads but tight for MoE models with heavy all-to-all traffic.

Larger clusters (10,000+ GPUs) use three-tier fat-trees or move to dragonfly topologies, which use a hierarchical structure of tightly-connected groups linked by long-distance optical fibers. Dragonfly minimizes the number of long fiber runs needed for very large clusters, at the cost of more complex routing and adaptive routing requirements.

## 4. RoCE: RDMA over Ethernet

InfiniBand's lossless fabric and proprietary protocol come with downsides: it requires dedicated InfiniBand switches (which only NVIDIA, since the Mellanox acquisition, manufactures at scale), it does not interoperate with Ethernet, and it creates vendor lock-in. RDMA over Converged Ethernet (RoCE) was developed to provide RDMA semantics over standard Ethernet hardware, eliminating these constraints.

RoCEv2 is the modern variant, which encapsulates RDMA packets inside UDP/IP and runs over standard Ethernet switches. The challenge is that Ethernet is not natively lossless — it can drop packets under congestion — so RoCE requires Priority Flow Control (PFC) and Explicit Congestion Notification (ECN) at the switch level to provide approximately lossless behavior. PFC pauses traffic for specific traffic classes when buffers fill, while ECN signals congestion by marking packets that the receiving NIC then uses to throttle senders.

In practice, RoCEv2 with well-tuned PFC and ECN achieves latency and bandwidth comparable to native InfiniBand for AI workloads. The configuration is more delicate — PFC misconfiguration can cause head-of-line blocking and even fabric-wide deadlocks — but the major hyperscalers have largely converged on RoCE for new deployments because it allows them to use standard Ethernet switches from multiple vendors and integrate AI clusters with their broader datacenter networks.

NVIDIA's Spectrum-X Ethernet platform is purpose-built for RoCE-based AI clusters. Spectrum-X switches have AI-specific features like adaptive routing, telemetry-driven congestion control, and packet spraying to maximize bisection bandwidth utilization. The Spectrum-X SuperNIC complements the switches with NIC-side congestion management. Together, they aim to match InfiniBand performance with the operational and procurement advantages of Ethernet.

## 5. NCCL on InfiniBand and RoCE

NCCL is the bridge between collective communication APIs (used by PyTorch, JAX, Megatron, etc.) and the underlying RDMA fabric. When NCCL initializes, it discovers the topology of the cluster, identifies which NICs are available on each node, builds a communication graph, and selects appropriate algorithms for each collective.

For inter-node communication, NCCL uses RDMA verbs directly, opening QPs between all pairs of GPUs that need to communicate and pipelining transfers across multiple QPs to saturate the available bandwidth. On a typical training run, NCCL maintains hundreds or thousands of open QPs per node and processes work across them concurrently.

NCCL's performance on RDMA fabrics is sensitive to several configuration parameters that production teams routinely tune. The number of channels controls how many parallel rings or trees NCCL uses for collectives — more channels allows better link utilization but increases per-collective overhead. The protocol selection (Simple, LL, LL128) trades off latency and bandwidth for different message sizes. The buffer size for collective operations affects the trade-off between memory consumption and pipelining efficiency.

For very large clusters, NCCL also supports hierarchical algorithms that perform intra-node collectives over NVLink first and then inter-node collectives over InfiniBand. This can substantially reduce the volume of data crossing the slower RDMA fabric for tensor-parallel and data-parallel patterns, at the cost of more complex algorithm scheduling.

## 6. GPUDirect RDMA

GPUDirect RDMA is the feature that allows the NIC to read from and write to GPU memory directly, bypassing the CPU and system memory entirely. Without GPUDirect, every GPU-to-GPU transfer over RDMA would require staging through CPU memory: GPU → CPU memory → NIC → network → NIC → CPU memory → GPU. Each of these copies adds latency and consumes PCIe bandwidth. With GPUDirect, the NIC and GPU communicate directly over the PCIe fabric, reducing the path to GPU → NIC → network → NIC → GPU.

The performance improvement is substantial. On an H100-based cluster with NDR InfiniBand (400 Gbps per port), GPUDirect RDMA reduces small-message latency from approximately 5 microseconds to 1.5 microseconds, and increases bandwidth efficiency from approximately 60% to 90% of the theoretical NIC peak.

GPUDirect RDMA depends on PCIe topology — the GPU and NIC must be on the same PCIe root complex (or at least the same NUMA node) for the direct DMA path to work efficiently. Server vendors design AI nodes with this in mind, placing each GPU and its associated NIC on a dedicated PCIe switch to ensure short, low-latency DMA paths.

## 7. Bandwidth Generations

InfiniBand and RoCE have evolved through several speed generations:

- **HDR (200 Gbps)**: Used in early H100 deployments and remains common in 2025 production clusters
- **NDR (400 Gbps)**: Standard for new H100 and Blackwell deployments since 2023
- **XDR (800 Gbps)**: Shipping in 2025 with ConnectX-8 and matching switches; standard for Blackwell-era clusters
- **GDR (1.6 Tbps)**: Announced for 2027

For comparison, NVLink 5 within a Blackwell rack provides 1.8 TB/s per GPU — roughly 18× the per-GPU bandwidth of XDR InfiniBand. This gap is what motivates the rack-scale architecture: all communication within the rack stays on the much faster NVLink fabric, and only cross-rack traffic uses the comparatively slower RDMA network.

## 8. Operational Realities

### 8.1 Failure Modes

InfiniBand and RoCE failures are notoriously difficult to diagnose. A degraded link may continue passing traffic but with elevated bit error rates, causing periodic retransmits and unpredictable latency spikes. A misconfigured PFC threshold may cause head-of-line blocking under load, manifesting as mysterious slowdowns under specific traffic patterns. A subnet manager failover may cause a brief routing inconsistency that interrupts in-flight collectives.

Production teams running large clusters invest heavily in fabric monitoring. NVIDIA's UFM (Unified Fabric Manager) is the standard tool for InfiniBand, providing topology visualization, real-time performance counters, and automated congestion analysis. For RoCE, vendors like Cisco and Arista provide similar capabilities, often integrated with their general-purpose network operations platforms.

### 8.2 Topology-Aware Scheduling

Cluster schedulers must be topology-aware to deliver good performance for AI workloads. Placing the GPUs of a single training job on nodes that are physically close in the network topology (same leaf switch, or at most one spine hop apart) dramatically reduces communication latency and avoids contention with other jobs. SLURM, Kubernetes (with the Volcano or Kueue schedulers), and proprietary platforms like Microsoft's Singularity all provide topology-aware placement features.

For the largest jobs that span an entire cluster, topology awareness is less about minimizing distance and more about avoiding pathological patterns like routing all all-reduce traffic through a single oversubscribed link. NCCL's adaptive routing helps with this on supported fabrics, but software-level scheduling decisions (which ranks are placed on which physical nodes) still matter.

### 8.3 Cost

InfiniBand fabrics are expensive. The HCAs, switches, and cabling for a 1,024-GPU cluster can run $5-10 million — a meaningful fraction of the GPU cost itself. RoCE deployments using standard Ethernet switches are typically 30-50% cheaper for equivalent bandwidth, though the savings are partly eaten by the need for higher-grade switches with deep buffers and AI-specific features.

The cost differential is one reason hyperscalers have largely moved to RoCE for new deployments, while neoclouds (CoreWeave, Lambda, Crusoe) and academic centers more often stay with InfiniBand for its operational simplicity and superior small-cluster performance.

## 9. The Future: Optical Interconnects

Copper cables for 800 Gbps RDMA are at the physical limits of what can be reliably transmitted over a few meters. The next bandwidth generations will require optical transceivers and fiber, which have historically been expensive and power-hungry. Co-packaged optics (CPO), where the optical engines sit directly on the switch ASIC package, are widely viewed as the path forward. Broadcom, NVIDIA, and Marvell are all developing CPO products targeted at 1.6 Tbps and beyond, with first deployments expected in 2026-2027.

The other emerging trend is silicon photonics for chip-to-chip optical communication, with companies like Ayar Labs and Lightmatter aiming to replace electrical NVLink-style interconnects with optical equivalents that can span tens of meters at lower power than copper. If successful, this would eliminate the rack as the natural boundary of the scale-up domain and allow datacenter-scale coherent compute fabrics.

## 10. Conclusion

RDMA and InfiniBand are the unglamorous but critical infrastructure that enables modern LLM training. Without them, the multi-thousand-GPU training jobs that produce frontier models would be impractical: gradient synchronization latencies would balloon, GPU utilization would collapse, and the scaling laws that motivate huge clusters would not be reachable in practice.

The technology stack is mature but evolving rapidly. The shift from InfiniBand to RoCE-on-Ethernet is transforming the operational landscape and breaking NVIDIA's effective monopoly on AI cluster networking. Bandwidth continues to grow at a faster pace than per-GPU compute, narrowing the gap between scale-up (NVLink) and scale-out (RDMA) interconnects. And optical technologies promise to extend the reach of high-bandwidth coherent fabrics beyond the rack, blurring the boundaries that have shaped cluster architecture for the past decade.

For practitioners, the practical lesson is that the network is rarely the bottleneck when it works well, but when it doesn't, no amount of GPU horsepower can compensate. Investing in fabric monitoring, topology-aware scheduling, and careful configuration of NCCL and the underlying RDMA stack pays dividends that often dwarf the gains from algorithmic optimizations at the model level.
