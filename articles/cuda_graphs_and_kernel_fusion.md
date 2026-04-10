# CUDA Graphs and Kernel Fusion: Squeezing Performance from the Decode Loop

*April 2026 • Technical Report*

## 1. Introduction

The decode loop of LLM inference is a cruel optimization target. Each token requires running the entire model forward — dozens of layers, hundreds of kernel launches, billions of arithmetic operations — and producing a single token. The compute density is very low (one new token per pass, regardless of how complex the pass is), and the overheads of kernel launches, memory allocations, and CPU-GPU coordination add a constant per-step cost that quickly dominates the actual work for small batch sizes.

For a long time, this was just accepted. Frameworks dispatched kernels one at a time, the GPU executed them with whatever overhead the runtime imposed, and decode latency was what it was. As inference moved into production and per-token latency became a primary cost driver, this casual approach became unsustainable. Saving 100 microseconds per kernel launch on a model that does 500 launches per token saves 50 milliseconds per token — meaningful when you are trying to hit a 20 ms target for the first token of a streaming response.

Two complementary techniques have emerged as the standard tools for attacking this overhead. CUDA Graphs replace the dispatch-one-kernel-at-a-time pattern with a pre-recorded graph of operations that the GPU can execute in a single submission. Kernel fusion combines multiple small kernels into a single larger one, eliminating the launch overhead and increasing arithmetic intensity. Together they have transformed the low-batch decode regime, often delivering 1.5-3× throughput improvements with no algorithmic changes.

This report examines what CUDA Graphs and kernel fusion are, how they are used in modern LLM inference frameworks, and where they fit into the larger landscape of inference optimization.

## 2. The Kernel Launch Overhead

To understand why CUDA Graphs and fusion matter, it helps to understand what happens when a CUDA program launches a kernel. The application calls `cuLaunchKernel` (directly or through a higher-level API). The CUDA driver receives the call, validates the arguments, queues the kernel onto a stream, and returns control to the application. The GPU eventually picks up the kernel from the stream, sets up its execution state, runs the kernel, and signals completion.

This sequence has costs at every step. The CPU-side dispatch takes a few microseconds for argument marshaling and driver overhead. The PCIe transmission of the launch command takes more time. The GPU's command processor takes additional time to set up the kernel's execution context. For a kernel that performs only a few microseconds of useful work, the launch overhead can match or exceed the compute time.

For LLM inference, this matters acutely. Modern transformers have many small operations: layer norms, RoPE rotations, residual additions, activation functions. Each is a separate kernel launch. A single forward pass of a 70B model might involve 500-1000 kernel launches, each contributing 5-50 microseconds of overhead. The accumulated overhead can easily exceed 10-50 ms per token — comparable to the actual compute time on the GPU.

## 3. CUDA Graphs

### 3.1 What CUDA Graphs Are

CUDA Graphs (introduced in CUDA 10 in 2018) allow a sequence of kernel launches to be captured into a graph object that can be replayed many times. Instead of issuing each kernel launch separately, the application:

1. Captures a sequence of operations once, building a graph
2. Instantiates the graph (compiling it for the target device)
3. Launches the graph as a single submission to the GPU

After the initial capture, each subsequent execution of the graph requires only a single launch call. The CPU-side dispatch overhead is paid once for the entire graph rather than once per kernel. The driver and GPU command processor have full visibility into the graph structure, allowing them to schedule and pipeline the kernels more efficiently than they could when seeing them one at a time.

The performance improvement is significant. For LLM decode, where the same forward pass runs many times in a row, capturing the pass into a CUDA Graph can reduce the per-step overhead by 5-10× and the total per-token latency by 20-40% in low-batch scenarios.

### 3.2 The Capture and Replay Model

CUDA Graphs come in two flavors. **Stream capture** is the easier-to-use mode: the application puts a CUDA stream into capture mode, runs the operations as it normally would, and ends capture to produce a graph. The application code does not need to change much — only the surrounding capture/end-capture calls.

**Explicit graph construction** is the more powerful mode: the application directly creates graph nodes (kernel nodes, memcpy nodes, dependency edges) using API calls. This gives full control over the graph structure but requires more code. Most LLM frameworks use stream capture for its simplicity and rely on explicit construction only when stream capture cannot express what they need.

Once captured, a graph can be instantiated to produce an executable. The instantiation step does optimization and validation, after which the executable can be launched many times. Each launch is essentially free on the CPU side — a single API call instead of hundreds.

### 3.3 Why It's Hard for Inference

CUDA Graphs work best when the operations are static — the same kernels with the same arguments, in the same order, every time. LLM inference has several characteristics that make this challenging:

- **Variable sequence lengths**: Different requests have different prompt lengths, output lengths, and context sizes. A graph captured for one length cannot be reused for another.
- **Dynamic batching**: Continuous batching adds and removes requests from each step, changing the effective batch size.
- **Variable KV cache sizes**: The KV cache grows by one token per decode step, changing the size of every attention operation.

Several techniques have emerged to work around these issues. The most common is to capture multiple graphs, one for each commonly-occurring shape, and dispatch to the appropriate graph at runtime. A typical inference server might capture graphs for batch sizes 1, 2, 4, 8, 16, 32, and 64, padding requests up to the nearest captured size. This trades a small amount of wasted compute (from padding) for the launch-overhead savings.

Another approach is to use graphs only for the static parts of inference (the per-layer compute, with fixed shapes) and run the variable parts (attention with growing KV cache) outside the graph. This complicates the implementation but avoids the padding overhead.

vLLM, TensorRT-LLM, and SGLang all use CUDA Graphs aggressively in their decode paths. The CUDA Graph optimizations are usually configurable, but enabled by default for production deployments.

## 4. Kernel Fusion

### 4.1 What Kernel Fusion Is

Kernel fusion combines multiple small kernels into a single larger kernel. The fused kernel performs the same computation as the original sequence but with a single launch and (often) reduced memory traffic. The reduction in memory traffic comes from keeping intermediate results in registers or shared memory rather than writing them back to global memory between kernels.

The classic example is fusing a layer normalization with the linear projection that follows it. The unfused version writes the layer-norm output to global memory, then the linear projection reads it back. The fused version computes the layer-norm output into registers, applies the linear projection in the same kernel, and only writes the final result. The intermediate write and read are eliminated, saving memory bandwidth and reducing latency.

For LLM inference, fusion opportunities are everywhere. Attention can be fused with the input projection, the softmax, and the output projection (FlashAttention does exactly this). FFN layers can fuse the gate, up-projection, activation, and down-projection. RMSNorm can fuse with the residual addition that follows it. Each fusion eliminates a kernel launch and a round-trip through global memory.

### 4.2 Static Fusion vs. Dynamic Fusion

There are two main approaches to kernel fusion. **Static fusion** is performed at build time by a compiler. The model is traced through a representative input, fusion patterns are identified, and fused kernels are generated. The result is a fixed set of kernels that the inference engine uses without runtime fusion decisions. TensorRT-LLM, ONNX Runtime, and most production inference engines use static fusion.

**Dynamic fusion** is performed at runtime by a JIT compiler. As operations are dispatched, the JIT looks for fusion opportunities and generates fused kernels on the fly. The advantage is flexibility — operations that the static compiler missed can still be fused. The disadvantage is the JIT compilation cost, which is paid the first time each fused kernel is needed. PyTorch's `torch.compile` and JAX's XLA backend both use dynamic fusion (with caching to avoid repeated compilation).

In practice, both approaches are widely used. Static fusion provides predictable performance for known model architectures. Dynamic fusion provides flexibility for novel architectures and reduces the manual effort needed to support them.

### 4.3 The Role of Triton

Triton (covered in a separate report in this collection) is the most popular language for writing fused kernels for LLM workloads. The compiler-driven approach makes it easy to write a kernel that fuses several operations into one, and the resulting kernel is competitive with hand-written CUDA. Most modern LLM inference frameworks include a substantial library of Triton kernels for fused operations: attention variants, fused FFN, fused layer norms, RoPE, and so on.

The pattern in production code is usually:

1. Use cuBLAS or CUTLASS for the largest matrix multiplications
2. Use Triton for everything else, with aggressive fusion of the small operations
3. Use CUDA Graphs to amortize the launch overhead of whatever kernels remain

This combination — large GEMMs from libraries, fused custom kernels in Triton, graph-level batching of launches — is the standard recipe for high-performance inference in 2026.

## 5. Real-World Examples

### 5.1 vLLM

vLLM uses CUDA Graphs in its decode path for batch sizes that are common at runtime. The implementation captures graphs for several preset batch sizes during server initialization, then dispatches each decode step to the closest matching graph. For a server with batch size 32 as a typical case, the captured graphs might cover batch sizes 1, 2, 4, 8, 16, and 32, each pre-built and ready to launch.

The kernel fusion in vLLM is provided largely by Triton kernels for attention (FlashAttention, PagedAttention) and by custom CUDA kernels for the linear projections. The overall architecture aims to minimize the number of distinct kernel launches per decode step, with most operations either inside the attention kernel or inside the linear projection kernel.

The combined effect is that vLLM's decode path on H100 hardware achieves very low per-token latency for small batch sizes, where the launch overhead would otherwise dominate.

### 5.2 TensorRT-LLM

TensorRT-LLM is NVIDIA's optimized inference framework. It uses static graph compilation with aggressive kernel fusion, generating fused kernels for the entire model rather than running per-operation kernels. The compilation step is slow (several minutes for a large model) but the resulting engine is very fast at runtime.

TensorRT-LLM also uses CUDA Graphs internally for the static parts of the inference loop, layered on top of the kernel fusion. The combination allows it to achieve the lowest published per-token latencies for many models, particularly at small batch sizes where overheads matter most.

### 5.3 SGLang

SGLang takes a slightly different approach. Its RadixAttention scheme requires a more dynamic kernel dispatch than vLLM's, because the structure of the prefix cache changes from request to request. SGLang uses CUDA Graphs more selectively, capturing graphs for specific patterns rather than for the entire forward pass. Its kernel fusion is largely Triton-based, similar to vLLM.

The result is that SGLang's per-step overhead is higher than vLLM's for simple workloads but its cache hit rates are better, leading to better performance on workloads with significant prompt overlap.

## 6. Limitations and Pitfalls

### 6.1 The Capture-Replay Mismatch

The most common bug with CUDA Graphs is a mismatch between what was captured and what is replayed. Stream capture records the kernel arguments at capture time, including pointers to GPU memory. If the application changes those pointers between captures and replays, the graph still uses the old pointers, leading to incorrect results.

The fix is to use placeholder pointers during capture and update them via graph node parameters before each replay. Most frameworks handle this internally, but it remains a common gotcha for application developers writing custom graph code.

### 6.2 Over-Capture

Capturing too much in a single graph can backfire. A graph that includes operations that depend on host computation cannot be replayed without re-running the host code, defeating the purpose. A graph that includes synchronization with the host (waiting for an event, copying data back to CPU) becomes essentially useless for repeated replay.

The discipline is to capture only the GPU-side compute, with all host-side coordination happening outside the graph. Inference frameworks generally enforce this through their graph capture API, but custom code can run into the issue.

### 6.3 Fusion Beyond the Sweet Spot

Not all fusion opportunities are wins. Fusing two operations that have very different parallelism characteristics can produce a kernel that uses GPU resources inefficiently. Fusing a memory-bound operation with a compute-bound operation can leave one or the other underutilized. Fusing too aggressively can produce kernels that are too large to fit in registers, forcing spills to shared memory or global memory and erasing the benefit.

In practice, the right fusion granularity is determined by experimentation. A good rule of thumb is to fuse operations that are dispatched in immediate sequence and that share data, but to avoid fusing operations that have very different shapes or compute profiles.

## 7. Beyond CUDA Graphs and Fusion

CUDA Graphs and kernel fusion are essential techniques but not the only ones for reducing decode overhead. Several complementary approaches are common:

**Persistent kernels** that run for the entire duration of an inference workload, avoiding any per-step launch overhead. Implemented in some custom deployments but not in mainstream frameworks.

**Speculative decoding** (covered separately) reduces the number of decode steps needed by drafting multiple tokens at once and verifying them in parallel. This is a separate optimization axis from the kernel-level work but combines well with it.

**Custom inference engines** that bypass the standard CUDA stack entirely and talk directly to the GPU through CUDA driver APIs or even custom firmware. Not common but used in some edge deployments.

**Batched inference** that increases the per-step compute load enough that the launch overhead becomes a smaller fraction of the total. This is the simplest optimization but only works when latency is not critical.

## 8. The Big Picture

The story of CUDA Graphs and kernel fusion is the story of inference engineering catching up to model capability. For several years, the community focused on training optimizations (FlashAttention, ZeRO, mixed precision) while inference performance was an afterthought. As LLMs moved into production and inference cost became the dominant economic factor, the focus shifted. CUDA Graphs and fusion are part of a broader investment in inference performance that has driven 10× cost reductions over the past two years.

The remaining headroom in this area is shrinking. The big wins from kernel fusion have been captured, the launch overhead has been minimized, and the next generation of optimizations is increasingly about algorithmic improvements (speculative decoding, prompt caching, model architecture changes) rather than systems-level improvements. The kernel-level techniques are now table stakes for any serious inference deployment, and the differentiation between frameworks lies elsewhere.

## 9. Conclusion

CUDA Graphs and kernel fusion are unglamorous but essential. Together, they often deliver larger performance improvements than algorithmic changes, and they cost nothing in model quality. Any serious LLM inference deployment in 2026 uses both, and any framework that does not support them is at a significant performance disadvantage.

For practitioners, the practical guidance is straightforward: use an inference framework that supports CUDA Graphs and kernel fusion natively (vLLM, TensorRT-LLM, SGLang, TGI), and rely on the framework's defaults for the kernel-level optimizations. Custom kernel work is rarely worth the engineering effort for typical deployments — the frameworks handle it well enough. The optimization decisions that matter at the application level are about batching strategy, caching, and request scheduling. The kernel-level efficiency is solved at the framework level, and that is where it should stay.

The deeper lesson is that systems-level engineering matters as much as algorithmic engineering for production AI workloads. The same model architecture, served by two different inference stacks, can have throughput differences of 5× or more — entirely from differences in how kernels are launched, how data is moved, and how operations are fused. Investing in the right inference stack is one of the highest-leverage decisions in deploying LLMs at scale.
