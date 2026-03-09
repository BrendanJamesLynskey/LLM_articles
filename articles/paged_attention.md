**PagedAttention**

Virtual Memory for the KV Cache: Eliminating Fragmentation in LLM Serving

March 2026 • Technical Report

Table of Contents

1\. Introduction

Autoregressive large language model inference is defined by a central tension: the key-value (KV) cache that makes token generation efficient is also the primary bottleneck limiting how many requests a serving system can handle concurrently. Each active sequence requires its own KV cache, and the memory consumed by these caches typically dwarfs the memory occupied by the model weights themselves once batch sizes and context lengths grow beyond modest values. Before 2023, the dominant approach to managing this memory was simple contiguous pre-allocation — reserve a slab of GPU memory for each sequence large enough to hold the maximum possible context length, and hope that most of it gets used. In practice, most of it did not. The resulting memory waste was enormous, and it placed a hard ceiling on serving throughput.

PagedAttention, introduced by Kwon et al. in the vLLM system (2023), broke through this ceiling by applying a concept borrowed directly from operating system design: virtual memory with paging. Instead of allocating one large contiguous buffer per sequence, PagedAttention divides the KV cache into small fixed-size blocks and maps each sequence's logical token positions to physical blocks through a block table, exactly analogous to how an OS maps virtual addresses to physical page frames. This single architectural insight — that KV cache memory need not be contiguous — eliminated the vast majority of memory waste in LLM serving and enabled a cascade of further optimizations that have reshaped the inference landscape.

This report provides a comprehensive examination of PagedAttention: the problem it solves, how it works, the system design that surrounds it, and the broad impact it has had on LLM inference infrastructure.

2\. The KV Cache Memory Problem

2.1 How the KV Cache Works

In a transformer-based language model, each layer contains a self-attention mechanism that computes attention over all previous tokens. During autoregressive decoding, the model generates one token at a time. Without caching, each new token would require recomputing the key and value projections for every previous token at every layer — an operation whose cost grows quadratically with sequence length. The KV cache eliminates this redundancy by storing the key and value tensors for all previously generated tokens so that each decoding step only needs to compute the key and value for the single new token and attend over the cached history.

For a model with L layers, H key-value heads, and head dimension d, storing a single token in the KV cache requires 2 × L × H × d elements (one key vector and one value vector per layer per head). In FP16, each element is 2 bytes. For a Llama 2 70B-class model (80 layers, 8 KV heads with GQA, head dimension 128), a single token requires 2 × 80 × 8 × 128 × 2 = 327,680 bytes, or roughly 320 KB. A sequence of 4,096 tokens therefore requires approximately 1.3 GB of KV cache, and a batch of 64 such sequences requires about 80 GB — already exceeding the memory of an A100 80GB GPU before accounting for model weights and activations.

2.2 Contiguous Pre-Allocation and Its Waste

The traditional approach to KV cache management pre-allocates a contiguous memory buffer for each sequence at the maximum supported sequence length. If the system supports sequences of up to 2,048 tokens, every sequence receives a 2,048-token buffer at request arrival, regardless of how many tokens it will actually generate. This mirrors the simplest memory allocation strategies from early computing: give each process a fixed partition and accept the waste.

The waste is severe. If the average generated sequence length is 256 tokens in a system configured for 2,048 tokens maximum, then on average 87.5 percent of allocated KV cache memory is unused at any given time. Even when sequences vary in length, the system must reserve memory for the worst case. This internal fragmentation directly limits the number of concurrent sequences the system can serve: GPU memory that is allocated but unused by one sequence cannot be given to another.

Additionally, because the system cannot predict how long a sequence will be at arrival time, it must either (a) allocate for the maximum length and accept the waste, or (b) attempt dynamic resizing of contiguous buffers, which is extremely expensive on GPUs because it requires copying potentially gigabytes of data to a new memory location whenever a buffer needs to grow. Neither option is acceptable at scale.

2.3 External Fragmentation

Beyond the internal waste within each sequence's buffer, contiguous allocation also suffers from external fragmentation. As sequences of different lengths arrive and complete, the GPU memory becomes peppered with free gaps that are individually too small to hold a new sequence's maximum-length buffer, even though the total free memory would be sufficient. This is the same problem that plagued early operating systems before the introduction of paging, and it has the same solution.

3\. The Virtual Memory Analogy

PagedAttention's core insight is that the KV cache memory management problem is structurally identical to the virtual memory problem that operating systems solved decades ago. The analogy is precise and worth examining in detail.

3.1 From OS Paging to KV Cache Paging

In an operating system, each process has a virtual address space that appears contiguous to the process but is mapped to physical memory frames that may be scattered anywhere in RAM. The mapping is maintained by a page table, and the unit of allocation is the page (typically 4 KB). This scheme eliminates both internal fragmentation (since pages are small) and external fragmentation (since any free physical frame can satisfy any virtual page request).

PagedAttention applies this model directly. Each sequence has a logical KV cache that appears as a contiguous array of token positions (0, 1, 2, ..., n). This logical cache is divided into blocks of a fixed size, typically 16 tokens per block, though this is a tunable parameter. Each block stores the key and value tensors for its tokens across all layers and all heads. The physical GPU memory is divided into a pool of identically sized physical blocks. A block table (the analog of a page table) maps each sequence's logical block indices to physical block addresses. Physical blocks need not be contiguous, and any free physical block can be assigned to any sequence's next logical block.

3.2 The Block Table

The block table is a lightweight per-sequence data structure, typically a simple array of integers. For a sequence currently occupying 200 tokens with a block size of 16, the block table contains ceiling(200 / 16) = 13 entries, each pointing to a physical block number. When the attention kernel needs to compute attention for this sequence, it reads the block table to find where each group of 16 key-value pairs is physically located and gathers them accordingly.

The block table itself is tiny compared to the KV cache data it indexes. For a sequence of 4,096 tokens with block size 16, the block table has 256 entries. At 4 bytes per entry (an int32 block index), this is just 1 KB — negligible overhead for managing potentially gigabytes of KV cache.

3.3 On-Demand Allocation

Just as an OS allocates physical pages on demand through page faults rather than pre-allocating the entire address space, PagedAttention allocates physical blocks only as they are needed. When a new sequence arrives, the system allocates only enough blocks for the initial prompt. As each new token is generated, if the current last block is full, a new physical block is allocated from the free pool and appended to the sequence's block table. When a sequence completes, all of its physical blocks are returned to the free pool immediately.

This on-demand approach means the system only uses memory proportional to the actual number of tokens stored, not the maximum possible number. The only waste is in the last block of each sequence, which may be partially filled. With a block size of 16, the average waste per sequence is 8 tokens — a fraction of a percent for any non-trivial sequence length. Kwon et al. reported that PagedAttention reduces KV cache waste to under 4 percent, compared to 60–80 percent with contiguous pre-allocation.

4\. How PagedAttention Works

4.1 Memory Layout

The physical block pool is organized as a large pre-allocated GPU memory region divided into uniformly sized blocks. Each physical block holds the key and value tensors for a fixed number of tokens (the block size) across all layers and heads. For a model with 80 layers, 8 KV heads, head dimension 128, and block size 16, each physical block stores 2 × 80 × 8 × 128 × 16 = 5,242,880 FP16 elements, or approximately 10 MB.

The total number of physical blocks available is determined by the GPU memory remaining after model weights, activations, and other overheads are accounted for. If 40 GB of GPU memory is available for the KV cache, this yields approximately 4,000 physical blocks — enough to store about 64,000 tokens worth of KV cache, distributed across however many sequences need them.

4.2 The Attention Kernel

The PagedAttention kernel is a modified attention implementation that reads key-value pairs from non-contiguous memory locations according to the block table. During the attention computation for a given query token, the kernel iterates over the block table entries for that sequence, fetching each physical block and computing the dot products between the query and the keys in that block. The partial results are accumulated across blocks using numerically stable online softmax.

This introduces some computational overhead compared to standard attention on contiguous memory, because the memory access pattern is less predictable and may not achieve optimal coalescing on the GPU. In practice, this overhead is small — typically less than 5 percent — because the attention computation is memory-bandwidth bound during the decode phase, and the dominant cost is reading the KV cache data regardless of whether it is contiguous. The memory savings far outweigh the slight computational penalty.

4.3 Block Size Selection

The block size is a tuning parameter that trades off between fragmentation and computational efficiency. Smaller blocks (e.g., 1 token) minimize waste but increase the block table size and may reduce GPU memory access efficiency due to smaller contiguous reads. Larger blocks (e.g., 64 or 128 tokens) improve memory access patterns but increase internal fragmentation in the last block.

The default block size of 16 tokens in vLLM was chosen as a balance point for typical workloads. For workloads with very short sequences, smaller blocks may be preferable to reduce last-block waste. For long-context workloads where sequences routinely span thousands of tokens, the last-block waste is negligible regardless of block size, and larger blocks can be used for better memory access performance. Some implementations support variable block sizes or hierarchical block schemes to adapt to mixed workloads.

5\. The vLLM System

5.1 Architecture Overview

vLLM (Virtual Large Language Model) is the open-source inference engine that introduced PagedAttention. Released in June 2023 by researchers at UC Berkeley, vLLM was designed from the ground up around the PagedAttention memory management paradigm. The system consists of several integrated components: a centralized scheduler, a KV cache manager (the block manager), model execution engines, and API-compatible serving endpoints.

The block manager maintains the global state of the physical block pool — which blocks are allocated, which are free, and the mapping from sequences to blocks. It handles allocation, deallocation, and the reference counting needed for memory sharing features like copy-on-write. The scheduler works with the block manager to determine which requests to admit, when to preempt sequences, and how to manage the memory budget across all active sequences.

5.2 Scheduling and Preemption

Because PagedAttention allows fine-grained memory management, the scheduler can make sophisticated decisions about resource allocation. When GPU memory runs low, instead of rejecting new requests or crashing, the system can preempt lower-priority sequences by swapping their KV cache blocks to CPU memory or even recomputing them later. Preemption is efficient because only the blocks for the preempted sequence need to be saved, and they can be restored exactly when the sequence is rescheduled.

vLLM supports two preemption strategies: swapping (moving blocks to CPU memory over PCIe) and recomputation (discarding blocks and rerunning prefill when the sequence resumes). Swapping is faster for sequences that will resume soon, while recomputation avoids the CPU memory overhead for workloads where preempted sequences may wait a long time before resuming.

5.3 Initial Performance Results

The original vLLM paper demonstrated dramatic throughput improvements over the state of the art at the time. On the OPT-13B and OPT-175B models, vLLM achieved 2–4x higher throughput than HuggingFace Transformers and 2–3x higher throughput than the FasterTransformer library, both measured at similar latency targets. The gains were larger on workloads with high variance in sequence lengths, where contiguous allocation wastes the most memory, and on workloads with shared prefixes, where copy-on-write deduplication provided additional savings.

These results established PagedAttention as a step-change improvement in LLM serving efficiency and catalyzed rapid adoption across the inference ecosystem.

6\. Memory Efficiency Gains

6.1 Near-Zero Internal Fragmentation

With contiguous allocation, internal fragmentation is proportional to the difference between the maximum supported sequence length and the actual sequence length. With PagedAttention, internal fragmentation is limited to the partially filled last block of each sequence. For a block size of 16, the expected waste per sequence is 8 token-slots, regardless of the maximum sequence length the system supports. For a system serving sequences that average 512 tokens, this represents waste of approximately 1.5 percent, compared to potentially 50–90 percent with contiguous allocation depending on the configured maximum length.

6.2 Zero External Fragmentation

Because any free physical block can be assigned to any sequence, there is no external fragmentation whatsoever. The system can use every free block regardless of its physical location. This is the same guarantee that paged virtual memory provides to operating systems, and it means the system achieves near-perfect memory utilization as long as there are free blocks available.

6.3 Higher Effective Batch Sizes

The practical consequence of eliminating memory waste is that the system can fit significantly more concurrent sequences in the same GPU memory. If contiguous allocation wastes 70 percent of KV cache memory on average, then PagedAttention effectively provides a 3.3x increase in usable KV cache capacity. This translates directly to higher batch sizes, which in turn improve GPU compute utilization and throughput. For workloads where the decode phase is memory-bandwidth bound, larger batch sizes are the primary lever for improving tokens-per-second-per-GPU.

In measured deployments, switching from contiguous allocation to PagedAttention has been shown to increase achievable batch sizes by 2–5x, with corresponding throughput improvements of 2–4x, depending on the workload's sequence length distribution.

7\. Copy-on-Write for Parallel Sampling

7.1 The Parallel Sampling Problem

Many LLM serving scenarios involve generating multiple output sequences from a single prompt. Examples include beam search (maintaining the top-k most likely partial sequences at each step), parallel sampling (generating n independent completions for the same prompt), and best-of-n sampling (generating n completions and selecting the best one by a reward model). In all these cases, the multiple sequences share the same prefix — the original prompt — and diverge only in the generated tokens.

With contiguous allocation, each of the n sequences requires its own full copy of the KV cache, including the shared prefix. This means the memory cost scales linearly with n, even though much of the data is identical across sequences. For a prompt of 1,000 tokens followed by generation of 100 tokens with n=8 parallel samples, the shared prefix represents 91 percent of each sequence's KV cache, and nearly 7x of the total KV cache memory is wasted on redundant copies.

7.2 Copy-on-Write Semantics

PagedAttention solves this with copy-on-write (CoW), another concept borrowed directly from OS virtual memory. When multiple sequences share a prefix, they share the same physical blocks for the prefix portion of their KV caches. Each sequence has its own block table, but the block table entries for the shared prefix point to the same physical blocks. A reference count tracks how many sequences reference each physical block.

When a sequence needs to append a new token to a shared block (i.e., a block whose reference count is greater than 1), the system copies the block to a new physical location and updates only that sequence's block table entry. Sequences that have not modified the block continue to reference the original. This ensures that each sequence sees its own consistent view of the KV cache while sharing the maximum amount of memory.

7.3 Memory Savings

The memory savings from copy-on-write scale with the ratio of shared prefix length to total sequence length and the number of parallel sequences. For beam search with beam width 8 and a prompt-to-generation ratio of 10:1, copy-on-write reduces KV cache memory usage by approximately 7x compared to full duplication. Even for parallel sampling with significant divergence, the savings from sharing the prompt's KV cache alone are substantial. Kwon et al. reported that for beam search workloads, vLLM's copy-on-write mechanism reduced memory usage by 55 percent compared to naive per-sequence allocation with PagedAttention (already much better than contiguous allocation).

8\. Prefix Caching and Cross-Request Sharing

8.1 The Prefix Sharing Opportunity

Many production LLM workloads exhibit high prefix overlap across requests. A chatbot service may prepend the same system prompt to every user message. A retrieval-augmented generation pipeline may share long document contexts across multiple questions about the same document. Few-shot prompting repeats the same examples across many requests. In all these cases, the KV cache for the shared prefix is identical and could be computed once and reused.

8.2 Automatic Prefix Caching

PagedAttention's block-based storage makes prefix caching natural and efficient. Since the KV cache for a prefix occupies a set of physical blocks, those blocks can be retained in memory after the original request completes and reused by future requests with the same prefix. The system computes a hash of the token content in each block and maintains a mapping from content hashes to physical block addresses. When a new request arrives, the system checks if its prompt's blocks match any cached blocks and, if so, skips the prefill computation for those tokens entirely.

vLLM's automatic prefix caching and SGLang's RadixAttention both implement this concept, with SGLang using a radix tree data structure to efficiently find the longest matching prefix between a new request and the cache. This approach supports partial matches: if a new request shares only part of its prefix with a cached request, the matching portion can still be reused.

8.3 Performance Impact

Prefix caching provides two distinct benefits: it reduces memory allocation (shared prefix blocks are not duplicated) and it reduces computation (prefill for cached tokens is skipped). The computation savings translate directly into lower time-to-first-token (TTFT), which is a critical latency metric for interactive applications.

For workloads with high prefix hit rates, the improvements are dramatic. SGLang's RadixAttention demonstrated 3–5x reductions in TTFT for workloads with long shared prefixes such as few-shot prompting and document Q&A. For multi-turn chat with a persistent system prompt, prefix caching typically saves 20–40 percent of total prefill computation, depending on the ratio of system prompt length to user message length.

9\. Integration with Continuous Batching

9.1 Complementary Optimizations

PagedAttention and continuous batching are distinct optimizations that reinforce each other powerfully. Continuous batching (introduced by the Orca system) allows sequences to enter and leave the active batch at each iteration step, rather than waiting for the entire batch to finish. PagedAttention provides the fine-grained memory management that makes continuous batching practical at scale.

Without PagedAttention, continuous batching is limited by the need to pre-allocate contiguous KV cache buffers for each sequence at its maximum possible length. This means new sequences can only be admitted when there is enough contiguous free memory for a full-length buffer, even if the system has plenty of fragmented free memory. PagedAttention removes this constraint: a new sequence only needs enough free blocks for its current tokens, and more blocks can be allocated incrementally.

9.2 The Combined Effect

The combination of continuous batching and PagedAttention enables serving systems to maintain near-constant GPU utilization across highly variable workloads. As short sequences complete, their blocks are freed immediately and become available for new arrivals. Long sequences grow their allocation gradually without blocking other sequences. The scheduler can pack sequences tightly because there is no contiguous allocation constraint, and can admit new requests aggressively because on-demand allocation means initial memory commitment is minimal.

In practice, systems running both continuous batching and PagedAttention consistently achieve 2–4x higher throughput than systems using only one of these techniques, and 5–10x higher throughput than systems using neither. This combined approach has become the baseline for all production LLM serving systems.

10\. Performance Benchmarks and Throughput Improvements

10.1 Throughput Comparisons

The performance impact of PagedAttention has been extensively benchmarked across different model sizes, hardware configurations, and workload patterns. The following summarizes representative results from published evaluations.

On a single A100 80GB GPU serving Llama 2 13B, vLLM with PagedAttention achieves approximately 14x higher throughput than naive HuggingFace Transformers serving and 2.2x higher throughput than text-generation-inference (TGI) at similar latency targets. For larger models requiring tensor parallelism, the gains persist: Llama 2 70B on 4x A100 shows 2–3x throughput improvement over TGI.

10.2 Workload Sensitivity

The magnitude of PagedAttention's improvement depends strongly on the workload's sequence length distribution. Workloads with high variance in sequence lengths benefit most, because contiguous allocation must reserve memory for the longest possible sequence while PagedAttention allocates proportionally to actual lengths. Workloads where all sequences are approximately the same length and close to the maximum see smaller (but still meaningful) improvements.

Shared-prefix workloads see the largest gains overall, because copy-on-write and prefix caching compound on top of the fragmentation elimination. A beam search workload with beam width 4 on Llama 2 13B showed 2.2x throughput improvement from PagedAttention's fragmentation elimination alone, and an additional 1.5x from copy-on-write block sharing, for a total of approximately 3.3x over contiguous allocation with the same continuous batching setup.

10.3 Latency Characteristics

PagedAttention has minimal impact on per-token latency. The modified attention kernel introduces a small overhead (typically 2–5 percent) due to the indirection through the block table and non-contiguous memory access. However, this per-token cost is overwhelmed by the throughput gains from being able to serve larger batches. For systems operating at high load, the reduction in queuing time from higher throughput more than compensates for any per-token overhead, resulting in lower end-to-end latency in practice.

Time-to-first-token is unaffected by PagedAttention per se, since the prefill phase processes the prompt in a single forward pass regardless of memory layout. However, when combined with prefix caching, TTFT can be reduced dramatically as discussed in Section 8.

11\. Adoption Across Frameworks

11.1 vLLM

vLLM remains the most widely used open-source PagedAttention implementation. Since its initial release, it has evolved significantly, adding support for speculative decoding, multi-modal models, pipeline parallelism, quantized KV caches, automatic prefix caching, and LoRA adapter serving. vLLM's PagedAttention implementation has been optimized for multiple hardware backends including NVIDIA GPUs (via custom CUDA kernels), AMD GPUs (via ROCm), and AWS Inferentia/Trainium. The project has a large open-source community and is used in production by numerous companies.

11.2 TensorRT-LLM

NVIDIA's TensorRT-LLM inference framework adopted paged KV cache management as a core feature. Their implementation follows the same conceptual framework as PagedAttention but integrates it with TensorRT's highly optimized attention kernels and execution engine. TensorRT-LLM's paged KV cache works seamlessly with their in-flight batching (continuous batching) system and is optimized for NVIDIA's hardware, including support for FP8 KV caches on Hopper and Blackwell architectures.

11.3 SGLang

SGLang (Structured Generation Language) built on PagedAttention and extended it with RadixAttention, which uses a radix tree to manage prefix sharing across requests at a more granular level. SGLang is particularly optimized for workloads involving structured generation (constrained decoding, JSON mode) and complex multi-call LLM programs where prefix sharing is especially valuable. Its radix tree approach enables more efficient prefix matching than simple hash-based lookup, particularly for partially overlapping prefixes.

11.4 Other Frameworks

The paged KV cache concept has been adopted broadly. HuggingFace text-generation-inference (TGI) supports paged attention. LMDeploy, an inference framework from the OpenMMLab ecosystem, implements paged attention with a focus on multi-modal model serving. DeepSpeed-FastGen includes paged memory management as part of its Dynamic SplitFuse scheduling system. The concept has also influenced proprietary serving systems at major AI companies that deploy their own inference infrastructure.

  ------------------------------ ------------------------------- ----------------------------------------
  **Framework**                  **PagedAttention Variant**      **Notable Extensions**

  vLLM                           Original PagedAttention         Prefix caching, CoW, LoRA, multi-modal

  TensorRT-LLM                  Paged KV Cache                  FP8 KV, in-flight batching, Blackwell

  SGLang                         RadixAttention                  Radix tree prefix matching, structured gen

  TGI                            Paged Attention                 HuggingFace ecosystem integration

  LMDeploy                       Paged KV Cache                  Multi-modal serving

  DeepSpeed-FastGen              Paged Memory                    Dynamic SplitFuse scheduling
  ------------------------------ ------------------------------- ----------------------------------------

12\. Limitations and Trade-Offs

12.1 Kernel Complexity and Overhead

The PagedAttention kernel is more complex than standard attention. The indirection through the block table adds control flow and non-contiguous memory accesses that can reduce computational efficiency, particularly for short sequences where the overhead is a larger proportion of total computation. Writing high-performance PagedAttention kernels for different hardware backends (NVIDIA, AMD, Intel, custom accelerators) requires significant engineering effort and deep knowledge of each architecture's memory subsystem.

12.2 Block Size Sensitivity

The choice of block size affects both memory efficiency and computational performance, and the optimal value depends on the workload. Systems that serve a mix of very short and very long sequences may find that no single block size is ideal. Adaptive or hierarchical block management adds complexity without a clear standard approach. In practice, the default block size of 16 works well for most workloads, but edge cases exist.

12.3 Metadata Overhead at Extreme Scale

While the block table is small for individual sequences, the aggregate metadata for managing millions of blocks across thousands of concurrent sequences can become non-trivial. The block manager must maintain free lists, reference counts, and hash tables (for prefix caching) that consume CPU memory and processing time. For extremely high-throughput systems processing tens of thousands of requests per second, the block management overhead can become a bottleneck if not carefully optimized.

12.4 Compatibility with Advanced Attention Mechanisms

Some advanced attention mechanisms, such as sliding window attention, sparse attention patterns, or cross-attention in encoder-decoder models, require modifications to the basic PagedAttention scheme. While these adaptations are generally straightforward, each new attention variant requires custom kernel development and testing. The rapid proliferation of attention mechanisms in the research community means that PagedAttention implementations must be continuously updated to support new architectures.

12.5 Hardware-Level Alternatives: vAttention

The kernel overhead introduced by PagedAttention's software-level block table indirection has motivated research into hardware-based alternatives. vAttention (Prabhu et al., 2024) observed that the 20–26 percent per-operation slowdown of PagedAttention kernels relative to non-paged alternatives stems fundamentally from the software indirection layer: block table lookups, extra branching, and non-contiguous memory access patterns that prevent the use of highly optimized attention kernels such as FlashAttention.

vAttention's key insight is that modern NVIDIA GPUs already possess the hardware needed to solve this problem. Through the CUDA Virtual Memory Management API, GPUs support hardware-managed page tables and demand paging at the memory management unit (MMU) level. vAttention allocates KV cache memory using GPU virtual memory, allowing the system to reserve a large contiguous virtual address range for each sequence while mapping physical GPU memory pages to that range on demand — exactly as an OS does for CPU memory. Because the KV cache appears contiguous to the attention kernel, standard optimized kernels (FlashAttention, FlashDecoding) can be used without modification. The GPU's MMU handles the virtual-to-physical translation transparently in hardware, with no software indirection overhead.

This approach achieves the same demand-paging benefits as PagedAttention — near-zero fragmentation, on-demand allocation, and efficient deallocation — while preserving full compatibility with the fastest available attention kernels. Copy-on-write semantics can similarly be implemented at the OS/driver level using the GPU's virtual memory remapping capabilities. vAttention thus represents a compelling alternative for deployments where kernel performance is critical and the target hardware supports the necessary virtual memory APIs. The trade-off is a tighter coupling to specific GPU hardware and driver versions, whereas PagedAttention's software approach is portable across any accelerator.

12.6 Not a Solution for the Fundamental Scaling Problem

PagedAttention eliminates waste in KV cache memory management, but it does not reduce the inherent memory requirements of the KV cache itself. A 128K-token sequence on a large model still requires tens of gigabytes of KV cache regardless of how efficiently that memory is allocated. For the fundamental problem of KV cache size scaling with sequence length, architectural approaches (GQA, MQA, cross-layer sharing) and compression techniques (quantization, token eviction) are necessary. PagedAttention is complementary to these techniques, not a substitute for them.

13\. Future Directions

13.1 Disaggregated KV Cache Storage

An emerging direction is to decouple KV cache storage from the GPU entirely, storing blocks in CPU memory, remote DRAM, or even distributed storage systems and fetching them on demand. This is the natural extension of the paging analogy — just as OS virtual memory can page to disk, KV cache blocks can be paged to cheaper, larger memory tiers. Systems like Mooncake and MemServe have explored this direction, using RDMA or high-speed interconnects to minimize the latency of remote block access. Disaggregated KV cache enables serving systems where KV cache capacity is no longer limited by GPU memory, at the cost of increased access latency that must be hidden through prefetching and scheduling.

13.2 Hardware-Aware Block Optimization

As GPU architectures evolve, there are opportunities to co-design block sizes and memory layouts with hardware features. Aligning blocks to GPU cache line boundaries, optimizing for the memory access patterns of specific tensor core configurations, and exploiting hardware support for gather/scatter operations can reduce the overhead of non-contiguous access. Future hardware might include direct support for indirection tables, further reducing the PagedAttention kernel's overhead.

13.3 Cross-Request and Cross-Model Sharing

Extending block sharing beyond single requests opens further optimization opportunities. A serving system handling multiple models that share a common tokenizer and partial architecture (such as a base model and its fine-tuned variants) could potentially share prefix KV cache blocks across models, amortizing prefill costs even further. Multi-tenant systems could share KV cache blocks for common system prompts across all users of a given application.

13.4 Integration with Linear and Hybrid Attention

As alternative attention mechanisms like linear attention (e.g., Lightning Attention) and hybrid architectures combining attention with state-space models (e.g., Jamba, Zamba) gain traction, the role of PagedAttention will evolve. These architectures reduce or eliminate the KV cache for some or all layers, potentially reducing the memory management burden. However, many hybrid models retain standard attention in a subset of layers, and PagedAttention remains valuable for efficiently managing the remaining KV cache in those layers.

13.5 Learned Allocation Policies

Current block allocation follows simple policies: allocate the next free block when needed. Future systems could use learned policies that predict a sequence's memory requirements based on the prompt content and generation parameters, pre-allocating blocks more intelligently to reduce allocation overhead during generation. Such policies could also inform preemption and eviction decisions, predicting which sequences are likely to complete soon and which will require long-running memory commitments.

14\. Practical Deployment Considerations

14.1 Monitoring KV Cache Health

For production deployments, monitoring KV cache memory behaviour is essential for capacity planning and performance tuning. Operators should track several key metrics.

Block utilization rate measures the proportion of allocated block slots that contain active token data, as opposed to empty slots in partially filled last blocks. A healthy system typically shows block utilization above 96 percent. Sustained utilization below 90 percent may indicate that the block size is too large for the workload's sequence length distribution and should be reduced.

Free pool size tracks the number of unallocated physical blocks available in the KV cache pool. This metric is the most direct indicator of memory pressure. When the free pool approaches zero, the system must begin preempting or rejecting requests. Operators should set alerts when the free pool drops below 10–15 percent of total blocks, as this threshold typically marks the onset of preemption-related latency increases.

Copy-on-write frequency measures how often the system performs block copies due to write conflicts on shared blocks. High CoW frequency during beam search or parallel sampling workloads is expected and healthy — it indicates that block sharing is working and that divergence is being handled correctly. However, unexpectedly high CoW frequency during single-sample workloads may indicate a misconfiguration in prefix caching or block sharing logic.

Shared-to-unique block ratio quantifies the memory savings from copy-on-write and prefix caching. A high ratio indicates effective sharing and suggests that the workload benefits significantly from these features. A low ratio on a workload expected to have high prefix overlap may indicate that prefix caching is not configured correctly or that the hash-matching mechanism is failing to detect shared prefixes.

Preemption rate tracks how often sequences are swapped to CPU memory or recomputed due to memory pressure. Any sustained preemption rate above zero indicates that the system is over-subscribed relative to its GPU memory capacity. While occasional preemption is acceptable, persistent preemption degrades tail latency and signals a need for either more GPU memory, a smaller maximum batch size, or KV cache compression techniques.

These metrics, taken together, give operators a comprehensive view of their serving system's memory efficiency and help guide decisions about fleet sizing, block size tuning, and workload routing.

15\. Conclusion

PagedAttention represents one of the most impactful systems innovations in the history of LLM inference. By recognizing that the KV cache memory management problem is structurally identical to the virtual memory problem that operating systems solved decades ago, it brought a well-understood solution — paging with block tables, on-demand allocation, and copy-on-write — to a domain where it was desperately needed. The results were immediate and dramatic: memory waste dropped from 60–80 percent to under 4 percent, achievable batch sizes increased by 2–5x, and serving throughput improved by a corresponding factor.

Beyond the direct efficiency gains, PagedAttention enabled a set of higher-level optimizations — prefix caching, cross-request sharing, fine-grained preemption, and memory-aware scheduling — that collectively transformed LLM serving from a memory-constrained, throughput-limited exercise into a mature systems engineering discipline. Its adoption across virtually every major inference framework confirms its status as foundational infrastructure.

The story of PagedAttention is also a reminder that some of the most impactful innovations in systems engineering come not from novel algorithms but from recognizing that a well-solved problem in one domain has a direct analog in another. Operating systems researchers solved the memory fragmentation problem in the 1960s. It took until 2023 for that solution to find its way into LLM serving, where it unlocked enormous practical value. As LLM inference continues to evolve, the systems community would do well to continue looking for these cross-domain analogies.
