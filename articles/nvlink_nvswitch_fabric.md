# NVLink and NVSwitch: Building the Scale-Up Fabric for AI

*April 2026 • Technical Report*

## 1. Introduction

In any sufficiently large LLM deployment, the question of how GPUs talk to each other becomes as important as the question of how fast each individual GPU computes. A 70-billion-parameter model in FP16 occupies 140 GB of memory — more than fits on any single GPU. Training or serving it requires the model to be split across multiple GPUs, with frequent synchronization between them. The performance of the entire system is bounded by the bandwidth and latency of the GPU-to-GPU interconnect.

NVIDIA's NVLink, introduced with the Pascal P100 in 2016, was the first GPU interconnect designed specifically to address this problem. Where PCIe Gen 3 offered roughly 16 GB/s between a CPU and a GPU, NVLink offered 80 GB/s between two GPUs directly. NVSwitch, introduced two years later, generalized this point-to-point link into an all-to-all crossbar, allowing every GPU in a system to communicate with every other GPU at full NVLink speed simultaneously. Together, NVLink and NVSwitch define the "scale-up" interconnect domain that distinguishes GPU servers from generic compute clusters.

This report examines the architecture, evolution, and operational implications of NVLink and NVSwitch from their origins through the fifth-generation fabric used in the Blackwell GB200 NVL72.

## 2. NVLink Generations

### 2.1 NVLink 1 and 2

The original NVLink (2016) used a 20 GB/s per direction signaling rate, with each P100 GPU exposing four NVLink ports. A typical 8-GPU DGX-1 connected GPUs in a hybrid cube-mesh topology, providing direct point-to-point links between most pairs and requiring two-hop forwarding for the rest. The aggregate bandwidth was 80 GB/s per GPU, but the topology was non-uniform — some GPU pairs had higher bandwidth than others.

NVLink 2 (2018, V100) doubled the per-port bandwidth to 25 GB/s and increased the port count to six per GPU, raising aggregate bandwidth to 150 GB/s. More importantly, NVLink 2 introduced cache coherence with IBM POWER9 CPUs, demonstrating that NVLink could carry coherent memory traffic and not just bulk DMA. This foreshadowed the unified memory architectures of Grace Hopper and the GB200.

### 2.2 NVLink 3 and 4

NVLink 3 (2020, A100) again doubled to 50 GB/s per port, with twelve ports per GPU for 600 GB/s aggregate. NVLink 4 (2022, H100) raised the per-port speed to 100 GB/s and the port count to eighteen, yielding 900 GB/s aggregate. This was the configuration used by the DGX H100, where 8 GPUs were connected through four NVSwitch chips into a fully non-blocking all-to-all fabric.

### 2.3 NVLink 5

NVLink 5 (2024, Blackwell) maintained 100 GB/s per port but expanded the addressable domain dramatically. Each Blackwell GPU has eighteen NVLink 5 ports, providing 1.8 TB/s of aggregate bandwidth per GPU. More significantly, the NVLink 5 protocol supports addressing of up to 576 GPUs in a single coherent domain, compared to a maximum of 8 in NVLink 4 systems. The NVL72 takes advantage of this with 72 GPUs in one domain, and the upcoming NVL576 designs will reach the full protocol limit.

## 3. NVSwitch Architecture

### 3.1 Crossbar Topology

NVSwitch is the GPU-to-GPU equivalent of an Ethernet switch: a non-blocking crossbar that routes NVLink packets between any pair of connected GPUs. The first-generation NVSwitch (2018, used in DGX-2) had eighteen NVLink ports, allowing it to interconnect six GPUs with three ports each. A DGX-2 used twelve NVSwitch chips to provide all-to-all connectivity between sixteen GPUs at 300 GB/s per GPU.

NVSwitch 4 (used in DGX H100 and HGX H100 systems) has 64 NVLink 4 ports and provides 6.4 TB/s of aggregate switching bandwidth. Four NVSwitch chips in a DGX H100 deliver full bisection bandwidth between all 8 GPUs at 900 GB/s per GPU.

NVSwitch 5 (used in NVL72) is a major architectural step. It has 72 NVLink 5 ports, supports the much larger 576-GPU address space, and introduces SHARP (Scalable Hierarchical Aggregation and Reduction Protocol) acceleration directly on the switch. SHARP allows certain collective operations — particularly all-reduce — to be partially executed on the switch silicon, reducing the data movement and synchronization required from the GPUs.

### 3.2 In-Network Computing with SHARP

SHARP, originally developed for Mellanox InfiniBand switches, performs reductions in the network rather than at the endpoints. For an all-reduce across 72 GPUs, the traditional approach uses ring or tree algorithms that pass data between GPUs in stages, with each stage adding its local contribution. SHARP-enabled switches receive the partial sums, perform the addition in the switch ASIC, and forward only the final result to all GPUs. This reduces the number of data hops by roughly half and eliminates the need for the GPU to participate in intermediate aggregation steps.

For LLM training, where all-reduce dominates the inter-GPU communication time, SHARP can improve effective collective bandwidth by 1.5-2× on bandwidth-bound operations. The benefit is particularly pronounced for large messages (gradient buffers in tensor-parallel training) and large GPU counts. NCCL takes advantage of SHARP automatically when running on SHARP-enabled NVSwitch fabrics.

## 4. Topology and Bandwidth Domains

### 4.1 The "Scale-Up" Domain

NVLink and NVSwitch together define what is conventionally called the GPU "scale-up" domain — the set of GPUs that can communicate at NVLink speeds with full memory coherence. Within this domain, tensor parallelism and other tightly-coupled parallelism strategies are efficient because the bandwidth is high and the latency is low (single-digit microseconds). Outside this domain, GPUs communicate over PCIe and then over a network fabric (InfiniBand or Ethernet), with bandwidth one to two orders of magnitude lower and latency at least an order of magnitude higher.

The scale-up domain has grown over time:
- DGX-1 (P100): 8 GPUs in a non-uniform NVLink mesh
- DGX-2 (V100): 16 GPUs in a uniform NVSwitch fabric
- DGX A100: 8 GPUs in NVSwitch (DGX-2 was discontinued; A100 returned to 8-GPU systems)
- DGX H100: 8 GPUs in NVSwitch with 4× the per-GPU bandwidth
- NVL72 (Blackwell): 72 GPUs in NVSwitch
- NVL576 (planned Rubin): 576 GPUs in NVSwitch

The expansion from 8 to 72 GPUs is the most significant single jump, and it fundamentally changes which models fit within a single tensor-parallel group.

### 4.2 Implications for Tensor Parallelism

Tensor parallelism partitions individual layers (linear projections, attention heads, MoE experts) across multiple GPUs, requiring an all-reduce after each parallelized operation. The communication volume scales with the activation size and the parallelism degree. For tensor parallelism to be efficient, the all-reduce time must be small relative to the compute time of the partitioned layer.

On 8-GPU NVLink domains, this constraint limited tensor parallelism degree to 8 for most models. Beyond 8, the communication crossed into InfiniBand territory and the ratio of communication to compute degraded by an order of magnitude. Models too large for 8-GPU TP had to use pipeline parallelism, which introduces its own inefficiencies (pipeline bubbles) but has much lower communication requirements.

The NVL72 raises this ceiling to 72 GPUs in the scale-up domain. A 1.8-trillion-parameter model can be tensor-parallelized across all 72 GPUs at full NVLink speed, eliminating the need for pipeline parallelism entirely (or using it only at coarse granularity for memory management). The throughput improvement for very large models is substantial because pipeline bubbles disappear and tensor parallelism degree can match the GPU count.

## 5. Collective Communication Patterns

### 5.1 All-Reduce

All-reduce is the dominant collective in distributed deep learning. Every gradient update in data-parallel training, every layer-wise activation sync in tensor-parallel training, and every cross-rank reduction in parameter servers requires it. NVLink is optimized for the bandwidth-bound regime of all-reduce (large message sizes), achieving 80-95% of theoretical peak bandwidth on the NVL72 fabric.

The two main algorithms are ring all-reduce and tree all-reduce. Ring all-reduce has optimal bandwidth efficiency (uses every link equally) but linear latency in the number of participants. Tree all-reduce has logarithmic latency but uses links unevenly. NCCL automatically selects between them based on message size and topology — ring for large messages, tree for small ones, hybrid algorithms for the in-between regime. On the NVL72, NCCL also takes advantage of NVSwitch SHARP to perform in-network reduction, further improving bandwidth efficiency.

### 5.2 All-Gather and Reduce-Scatter

All-gather and reduce-scatter are the two halves of an all-reduce: all-gather distributes locally-held data to all participants, reduce-scatter aggregates contributions from all participants and distributes the result. They are used heavily in ZeRO-style sharded training (where parameters are split across GPUs and gathered before each layer's forward pass) and in tensor-parallel attention (where the partitioned outputs of attention heads must be gathered before the FFN).

NVLink's high bandwidth makes both operations cheap relative to the compute they enable, which is why ZeRO Stage 3 and FSDP — which would be impractical on PCIe-connected GPUs — work well on NVLink-connected GPUs.

### 5.3 All-to-All

All-to-all is the dominant collective for Mixture-of-Experts (MoE) models, where each token must be routed to its assigned expert (which may be on a different GPU) and the result returned. The communication pattern is N-to-N rather than N-to-1, and it stresses the bisection bandwidth of the fabric more than other collectives. NVSwitch's all-to-all topology is well-suited to this pattern, and NVLink-connected MoE models routinely achieve all-to-all efficiencies above 80%.

InfiniBand-connected MoE models, by contrast, often see all-to-all efficiencies of 30-50% because the cluster topology has bandwidth bottlenecks that all-to-all traffic exposes. This is one of the strongest arguments for the NVL72 in MoE training: the all-to-all bandwidth is roughly 10× higher per GPU than on InfiniBand-connected H100 clusters.

## 6. NCCL: The Communication Library

NCCL (NVIDIA Collective Communications Library) is the software layer that abstracts NVLink, NVSwitch, PCIe, and InfiniBand into a uniform collective API. NCCL detects the underlying topology at startup and selects optimal algorithms and message routing for each collective. PyTorch, TensorFlow, JAX, Megatron, DeepSpeed, and essentially every modern ML framework use NCCL for multi-GPU communication.

NCCL's value is in its topology awareness. On a DGX H100, it knows that GPUs 0-3 share an NVSwitch group with one set of links and GPUs 4-7 share another, and it routes traffic accordingly. On the NVL72, it knows the full 72-GPU NVLink topology and selects ring orderings that minimize switch hops. When a job spans multiple racks, NCCL automatically uses NVLink for intra-rack communication and InfiniBand for inter-rack communication, with appropriate algorithm selection at each level.

The library also supports collective operation overlap with computation, which is essential for efficient training. By dispatching collectives asynchronously and tracking their completion through CUDA streams, NCCL allows compute and communication to overlap, hiding much of the communication time behind useful work.

## 7. NVLink Beyond the Rack: NVLink Network and IB Convergence

The NVL576 design previewed for the Rubin generation hints at NVIDIA's longer-term direction: extending the NVLink address space across multiple racks. This requires NVLink-over-fiber (NVLink-NW) running over physical InfiniBand or custom optical links, with sufficient latency hiding to maintain coherent operation. The first version of NVLink-NW connects two NVL72 racks at NVLink speeds, with cross-rack latency of approximately 1 microsecond — a few times higher than intra-rack but still vastly better than InfiniBand.

The eventual convergence of NVLink and NVIDIA's Spectrum-X Ethernet products — both engineered to handle the same RDMA-style traffic patterns at increasing scales — is the most plausible long-term roadmap. The boundaries between scale-up (NVLink) and scale-out (InfiniBand/Ethernet) are blurring as NVIDIA pushes for ever-larger coherent compute domains.

## 8. Competing Approaches

NVLink is the most mature GPU interconnect, but it is not the only one. AMD's Infinity Fabric (used in MI300X) provides 896 GB/s of GPU-to-GPU bandwidth in an 8-GPU node, comparable to H100 NVLink. AMD's roadmap includes Infinity Fabric XGMI scaling to larger domains, though without an equivalent of NVSwitch's SHARP-style in-network compute. Intel's Xe Link in Gaudi 3 takes a different approach, integrating RDMA Ethernet directly into the accelerator, blurring the scale-up/scale-out distinction at the cost of slightly lower per-link bandwidth.

The Ultra Accelerator Link (UALink) consortium — formed by AMD, Intel, Google, AWS, Meta, Microsoft, HPE, Cisco, and Broadcom in 2024 — aims to create an open scale-up interconnect standard. The 1.0 spec targets bandwidth comparable to NVLink 5 and supports up to 1,024 accelerators in a single domain. Whether UALink achieves NVLink's mature ecosystem of software and tooling remains to be seen, but the consortium's existence reflects the strategic importance of scale-up interconnects for AI workloads.

## 9. Operational Considerations

Operating NVLink-connected systems imposes practical constraints often underappreciated by software teams. Failures of individual NVLinks degrade the entire fabric: NCCL can detect a link failure and route around it, but the surviving paths carry more traffic and become bottlenecks. Hyperscalers running Blackwell clusters report that NVLink errors are now a leading cause of training job interruptions, often surfacing as mysterious slowdowns rather than hard failures.

Monitoring NVLink health requires NVIDIA's DCGM (Datacenter GPU Manager) toolkit, which exposes per-link error counters, bandwidth utilization, and topology health. Production deployments typically scrape these metrics into Prometheus and alert on rising error rates before they become job-impacting.

Rebooting individual NVLink switches on the NVL72 requires draining the affected GPUs first, which means evacuating any running jobs to other parts of the cluster. The rack-scale nature of the NVL72 makes this more invasive than rebooting a single 8-GPU node — a single switch failure can take 8-16 GPUs offline simultaneously. Cluster schedulers like SLURM and Kubernetes need NVLink-topology awareness to handle these scenarios gracefully, and the tooling for this is still maturing in 2026.

## 10. Conclusion

NVLink and NVSwitch are the unsung heroes of modern LLM training and inference. The compute capability of individual GPUs has grown roughly 10× since the V100 era, but the interconnect bandwidth has grown roughly 30×, and the size of the coherent NVLink domain has grown roughly 9×. These together have made model parallelism strategies that were previously impractical — tensor parallelism degree 72, all-to-all MoE routing across an entire rack, ZeRO-3 sharding at scale — into the standard tools of large-model training.

The next decade will likely see the NVLink domain continue to grow, the boundary with scale-out networking continue to blur, and competing standards (UALink, InfinityFabric XGMI) attempt to break NVIDIA's interconnect lock-in. For now, NVLink+NVSwitch remains the gold standard for scale-up GPU communication, and the architectural choices it embodies — non-blocking crossbar topology, in-network compute, coherent memory addressing across many GPUs — define what "one giant GPU" actually means in practice.
