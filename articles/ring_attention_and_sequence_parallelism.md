**Ring Attention and Sequence Parallelism**

Distributing Attention Across Devices for Long-Context Training and Inference

March 2026 • Technical Report

Table of Contents

1\. Introduction

As context windows expand from thousands to millions of tokens, the quadratic memory cost of attention becomes a fundamental barrier. Even with memory-efficient kernels like FlashAttention that reduce per-device memory from quadratic to linear, the linear term itself becomes prohibitive for very long sequences: the activation memory for the attention output, the KV cache, and the feedforward network outputs at each layer must all fit in a single GPU's memory. For a model with a hidden dimension of 4,096 processing 1 million tokens, the attention activations alone require tens of gigabytes per layer, far exceeding the high-bandwidth memory of any single GPU.

Sequence parallelism, also known as context parallelism, addresses this by distributing a single sequence across multiple GPUs so that each device holds only a fraction of the tokens. This is orthogonal to data parallelism (which distributes different sequences), tensor parallelism (which distributes the model's weight matrices), and pipeline parallelism (which distributes layers). Ring Attention, proposed by Liu et al. in 2023, is the most influential sequence parallelism technique for attention, enabling context lengths that scale linearly with the number of devices while maintaining mathematical equivalence to single-device attention.

This report covers Ring Attention's design, its relationship to FlashAttention, the complementary DeepSpeed Ulysses approach, and the hybrid methods that combine both for practical deployment.

2\. Why Sequence Parallelism Is Needed

2.1 The Activation Memory Wall

Consider training a Llama 3-class model (hidden dimension 8,192) on a sequence of 128K tokens. The attention output for a single layer is a tensor of shape (batch_size, sequence_length, hidden_dimension), which in BF16 is approximately 2 GB for a single sequence. Across 80 layers, this is 160 GB of activations (before accounting for gradients and optimizer states). The KV cache for inference is similarly large. No single GPU can hold this, and simply increasing HBM capacity is not viable due to physical limitations and manufacturing cost.

Tensor parallelism and pipeline parallelism address the weight and layer memory, respectively, but neither reduces the per-token activation memory. Data parallelism requires each device to process the full sequence. Only sequence parallelism directly reduces the activation memory per device by splitting the sequence itself.

2.2 Existing Parallelism Is Insufficient

NVIDIA's Megatron-LM introduced an early form of sequence parallelism that distributes dropout and layer normalisation along the sequence dimension to save activation memory, but it does not distribute the attention computation itself. Each device still requires the full sequence for computing attention, creating a memory bottleneck. The attention computation is the specific operation that must be parallelised to enable truly long contexts.

3\. Ring Attention

3.1 Core Design

Ring Attention distributes the input sequence across N devices arranged in a logical ring. Each device receives a contiguous chunk of 1/N of the sequence and computes the query, key, and value projections for its local chunk. The attention computation then proceeds as a series of N rounds. In each round, every device computes the attention between its local queries and the key-value pairs currently resident on it, using a memory-efficient kernel (FlashAttention) for the local computation. Between rounds, each device sends its key-value block to the next device in the ring and receives the block from the previous device. After N rounds, every device's queries have attended to all key-value pairs from the full sequence.

The critical insight is that the inter-device communication (sending KV blocks around the ring) can be overlapped with the local FlashAttention computation. While device i computes attention for its queries against the current KV block, it simultaneously sends that KV block to device i+1 and receives the next block from device i-1. If the computation time exceeds the communication time, the communication is fully hidden and the total wall-clock time is equivalent to computing FlashAttention on the full sequence on a single device.

3.2 Online Softmax Across Blocks

Ring Attention inherits the online softmax technique from FlashAttention. Each local FlashAttention computation produces a partial attention output along with the log-sum-exp statistics for its block. As new KV blocks arrive in subsequent rounds, the device updates its running statistics and rescales its accumulated output. After all N rounds, the final output is numerically identical to what would be computed if the full sequence were on a single device.

3.3 Communication Requirements

The minimum block size (and therefore the minimum sequence length per device) is determined by the ratio of compute throughput to interconnect bandwidth. For the computation to fully overlap communication, the local FlashAttention on each block must take at least as long as transferring a KV block to the next device. On modern hardware with NVLink (600 GB/s bidirectional on A100, 900 GB/s on H100), blocks of a few thousand tokens are typically sufficient. The original Ring Attention paper calculated that with H100 GPUs connected via NVLink, a minimum of a few thousand tokens per device is needed for full overlap. On slower interconnects (InfiniBand between nodes), larger block sizes are required.

3.4 Causal Masking and Load Balancing

For causal (autoregressive) attention, the workload is inherently unbalanced when each device holds a contiguous chunk: the device holding the last chunk must compute attention against all preceding tokens, while the device holding the first chunk does almost no work (since those tokens can only attend to themselves). This load imbalance can waste up to half the aggregate compute.

Striped Attention (Brandon et al., 2023) addresses this by distributing tokens across devices in an interleaved pattern rather than contiguous chunks. With zigzag assignment, device 0 gets tokens 0, 2N-1, 2N, 4N-1, and so on, ensuring that each device's workload is approximately equal regardless of the causal mask. This striped layout is compatible with RoPE as long as the position indices are correctly maintained (each token retains its original position index for the rotary embedding, even though it is physically located on a different device).

4\. DeepSpeed Ulysses

4.1 All-to-All Communication

DeepSpeed Ulysses, proposed by Jacobs et al. in 2023, takes a different approach to distributing attention. Instead of passing KV blocks around a ring, Ulysses partitions the Q, K, and V tensors along the head dimension (not the sequence dimension) across devices. Each device initially holds a portion of the sequence across all heads. Before the attention computation, an all-to-all collective redistributes the tensors so that each device holds the full sequence for a subset of attention heads. Each device then computes standard (non-distributed) attention for its assigned heads. After attention, another all-to-all restores the original partitioning.

4.2 Advantages and Limitations

Ulysses is generally faster than Ring Attention for moderate parallelism degrees because all-to-all collectives are highly optimised on modern interconnects and achieve higher bandwidth utilisation than point-to-point ring transfers. The local attention computation uses standard FlashAttention without modification, avoiding any overhead from distributed online softmax.

However, Ulysses has a fundamental scalability limitation: the maximum parallelism degree cannot exceed the number of attention heads. For a model with 32 query heads and grouped-query attention using 8 KV heads, Ulysses can parallelise across at most 8 devices (limited by the KV heads). For models with fewer heads, or when combined with tensor parallelism (which also partitions across heads), Ulysses may not provide enough parallelism for very long sequences.

5\. Hybrid Approaches

5.1 Unified Sequence Parallelism (USP)

Unified Sequence Parallelism combines Ring Attention and Ulysses to exploit their complementary strengths. Within a node (where NVLink provides high bandwidth), Ulysses partitions across heads for maximum efficiency. Across nodes (where interconnect bandwidth is lower), Ring Attention distributes along the sequence dimension and overlaps communication with computation. This two-level hierarchy matches the typical network topology of GPU clusters and achieves better performance than either method alone.

5.2 Context Parallelism in Practice

Meta's context parallelism implementation for Llama 3 uses a modified all-gather-based ring algorithm (pass-KV) that concurrently all-gathers KV shards while computing attention on local chunks. PyTorch provides a context_parallel API that automatically replaces standard scaled_dot_product_attention with a Ring Attention implementation, supporting both all-gather and all-to-all rotation methods. NVIDIA's Megatron-core and TransformerEngine support context parallelism as a first-class parallelism dimension.

6\. Performance Characteristics

6.1 Scaling Efficiency

The ringX family of methods (2024) demonstrated that optimised ring attention achieves up to 3.4 times speedup over naive ring attention on the Frontier supercomputer, reaching 38 percent model FLOPs utilisation (MFU) for training Llama 3 8B with a 1-million-token sequence length on 4,096 GPUs. This represents one of the highest training efficiencies reported for long-context learning on HPC systems.

For inference, Meta's context parallelism paper (2024) showed that pass-KV ring attention significantly outperforms tensor parallelism for multi-node long-context inference on Llama 3 405B, with the advantage growing as the number of nodes increases (tensor parallelism becomes bottlenecked by all-reduce latency across nodes, while ring attention overlaps communication with computation).

6.2 When to Use What

  ------------------------------------------------- -------------------------------------------- -------------------------------------------
  **Scenario**                                      **Recommended Approach**                      **Rationale**

  Single node, enough heads                         Ulysses                                      Highest bandwidth, no ring overhead

  Single node, few KV heads (GQA)                   Ring Attention                               Not limited by head count

  Multi-node, high parallelism                      Hybrid (Ulysses intra-node, Ring inter-node) Matches network hierarchy

  Very long context (\>1M tokens)                   Ring Attention (striped for causal)           Scales with device count, no head limit
  ------------------------------------------------- -------------------------------------------- -------------------------------------------

7\. Practical Considerations

For practitioners, sequence parallelism introduces several implementation details that require attention. RoPE position indices must be correctly maintained regardless of how tokens are physically distributed. If using striped or zigzag assignment, the position indices must correspond to the token's original position in the sequence, not its local index on the device. Frequency tensors (freq_cis in Llama-style implementations) must also be sharded consistently with the token distribution.

Padding is often necessary to ensure the sequence length is evenly divisible by the parallelism degree. For variable-length inputs in inference, the pass-Q variant of context parallelism (which sends queries around the ring rather than KV blocks) can be more efficient for prefill workloads where the number of new tokens is small relative to the cached context.

The interaction between sequence parallelism and other parallelism dimensions requires careful orchestration. Data parallelism, tensor parallelism, pipeline parallelism, and context parallelism can all be composed, but the communication patterns must be coordinated to avoid deadlocks and maximise overlap. Modern frameworks like Megatron-LM and TorchTitan provide abstractions (process group meshes) to manage this complexity.

8\. Conclusion

Sequence parallelism, particularly Ring Attention and its variants, has become essential infrastructure for training and deploying long-context LLMs. By distributing the attention computation across devices while overlapping communication with computation, Ring Attention enables context lengths that scale linearly with the number of GPUs without approximation or quality loss. The combination of Ulysses for intra-node efficiency and ring-based methods for inter-node scaling provides a practical framework that matches the hierarchical bandwidth structure of modern GPU clusters. As models push toward million-token and multi-million-token contexts, these techniques will remain the primary mechanism for making such contexts computationally feasible.
