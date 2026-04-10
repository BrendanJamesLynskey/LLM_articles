# Triton: GPU Kernel Programming for Machine Learning

*April 2026 • Technical Report*

## 1. Introduction

For most of CUDA's history, writing high-performance GPU kernels required deep expertise in the hardware: thread block configuration, shared memory layouts, register pressure, instruction-level scheduling, and the endless dance of memory coalescing and bank conflicts. The performance gap between expert-written CUDA and naive CUDA was routinely 10×, and the gap between a custom kernel and a generic library implementation often determined whether a research idea was practical. This expertise gap created a bottleneck: machine learning researchers wanted to experiment with new operations (custom attention variants, novel activation functions, fused training kernels) but lacked the time or skills to write the CUDA themselves.

Triton, developed at OpenAI and released in 2021, is an attempt to close this gap. It is a Python-embedded domain-specific language for GPU programming that abstracts away most of the low-level details — block-level memory management, thread scheduling, instruction selection — while still allowing programmers to control the things that matter for performance, like tile sizes and parallelization strategy. The result is a language in which a competent ML engineer can write kernels that match or exceed the performance of expert-written CUDA, in roughly an order of magnitude less code.

By 2025, Triton had become the dominant language for new GPU kernels in the open-source ML ecosystem. FlashAttention, Mamba, Liger Kernel, the bulk of the kernels in vLLM and TensorRT-LLM, and major contributions to PyTorch's `torch.compile` backend are all written in Triton. This report examines what Triton is, how it differs from raw CUDA, and why it has reshaped the GPU kernel development landscape for ML.

## 2. The Triton Programming Model

### 2.1 Block-Level Programming

The core abstraction in Triton is the block. Where CUDA exposes a grid of thread blocks, each containing many threads, Triton hides individual threads entirely and presents the programmer with a grid of program instances, each of which operates on a block of data. The programmer writes code that processes one block at a time, expressing operations in terms of tiles of values rather than individual scalars.

A simple Triton kernel for vector addition looks like:

```python
import triton
import triton.language as tl

@triton.jit
def add_kernel(x_ptr, y_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    output = x + y
    tl.store(output_ptr + offsets, output, mask=mask)
```

There are no thread indices, no shared memory declarations, no warp shuffles. The programmer specifies a `BLOCK_SIZE` (typically a power of two like 1024) and operates on arrays of that size as if they were scalar values. The Triton compiler decides how many threads to launch per block, how to distribute work across them, how to use shared memory, and how to vectorize loads and stores.

### 2.2 The Compiler

Triton's compiler is the secret sauce. It takes the Python source, traces it into an intermediate representation (Triton-IR), and applies a sequence of optimization passes that target specific GPU hardware features:

- **Memory coalescing**: Loads and stores are analyzed for access patterns and rearranged to ensure that consecutive threads access consecutive addresses.
- **Shared memory allocation**: Tiles that are reused across multiple operations are automatically allocated in shared memory, with bank conflict avoidance.
- **Pipeline scheduling**: Loops are software-pipelined to overlap compute with memory loads, achieving the same effect as hand-written `__pipeline_memcpy_async` in CUDA.
- **Tensor core dispatch**: Matrix multiplications are mapped to tensor core instructions where possible, with appropriate data layout transformations.
- **Register allocation**: Values are kept in registers when possible, with spill-to-shared-memory only when necessary.

The output is PTX (the NVIDIA intermediate assembly) or, on AMD hardware, LLVM IR for the ROCm backend. Triton was originally NVIDIA-only but has since gained AMD support and experimental backends for Intel and Apple GPUs.

## 3. Why Triton Changed the Game

### 3.1 The Productivity Argument

Writing a high-performance matrix multiplication in CUDA is a significant undertaking. NVIDIA's own CUTLASS library — a collection of templates for matmul kernels — is tens of thousands of lines of C++. The kernels are powerful and fast, but understanding them requires expertise that most ML engineers do not have. Modifying them for a new dtype, a new fusion pattern, or a new tile shape is hard.

A Triton matmul, by contrast, is approximately 50 lines of Python. It performs within 5-10% of CUTLASS for standard configurations and can be modified by anyone who knows Python and basic GPU concepts. This productivity gap matters enormously for research. A team that wants to experiment with a new attention variant can prototype the kernel in Triton in an afternoon, benchmark it, and iterate. The same task in CUDA might take weeks.

The FlashAttention story illustrates this. The original FlashAttention paper (2022) was implemented in CUDA, and the resulting code was a masterpiece of memory-optimized programming — but it was also famously difficult to modify. FlashAttention 2 and 3 maintained CUDA implementations, but all of the variants developed by the research community (FlashAttention with custom masks, with sliding windows, with sinks, with paged KV caches) were implemented in Triton. The Triton implementations were 5-10% slower than the CUDA reference but iterated 10× faster.

### 3.2 The Performance Argument

The productivity argument is only compelling if performance is competitive. Triton's design philosophy is that for most kernels, the difference between expert-written CUDA and Triton is in the single digits, while the difference between Triton and naive CUDA is much larger. In other words, Triton captures most of the optimizations that an expert would apply, automatically.

Empirical comparisons bear this out. On standard transformer kernels — matmul, softmax, layer norm, gelu, attention — Triton implementations achieve 85-95% of the performance of NVIDIA's hand-tuned cuBLAS and cuDNN, and on novel kernel patterns where no library equivalent exists, Triton routinely outperforms whatever CUDA the research community produces. The places where Triton falls short of expert CUDA tend to be very mature, well-optimized kernels (like dense matmul on NVIDIA's most recent generation), where the last few percent of performance requires hardware-specific tricks that Triton's optimizer hasn't yet absorbed.

### 3.3 Cross-Hardware Portability

A Triton kernel runs unchanged on multiple GPU architectures. The compiler emits different code for Volta, Turing, Ampere, Hopper, and Blackwell, taking advantage of the tensor cores and memory hierarchy of each. It also runs on AMD's CDNA and RDNA architectures via the ROCm backend, with similar performance characteristics. In an era when hyperscalers are diversifying away from NVIDIA-only deployments, this portability is increasingly valuable.

In practice, the portability is not perfect. Performance characteristics differ between vendors, and a kernel optimized for H100 may not achieve the same fraction of peak on MI300X without parameter tuning. But the source code does not need to change, and that is a meaningful improvement over the CUDA-only world.

## 4. Anatomy of a Triton Kernel: Matrix Multiplication

A simplified Triton matmul kernel illustrates the language's expressive power:

```python
@triton.jit
def matmul_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    a_ptrs = a_ptr + (offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn)

    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, K, BLOCK_K):
        a = tl.load(a_ptrs)
        b = tl.load(b_ptrs)
        accumulator += tl.dot(a, b)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    c_ptrs = c_ptr + (offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn)
    tl.store(c_ptrs, accumulator)
```

Several things deserve attention. The kernel operates on tiles of shape `(BLOCK_M, BLOCK_K)` and `(BLOCK_K, BLOCK_N)` — there is no notion of "this thread does this element." The `tl.dot` call expands into a tensor core matrix multiplication, with the compiler choosing the appropriate instruction (e.g., `wgmma` on Hopper or `mma` on Ampere) based on the target hardware. The accumulator is automatically kept in registers across loop iterations. The loop is software-pipelined by default, with the compiler issuing the next iteration's loads while the current iteration's compute is still in flight.

The same kernel in CUDA, achieving comparable performance, would require explicit shared memory tiling, double-buffering, async copy instructions on Hopper, careful synchronization, and dozens of optimization decisions. The Triton version makes those decisions implicitly through the compiler.

## 5. Triton in Production: Real Examples

### 5.1 FlashAttention Variants

The original FlashAttention is CUDA. Most of the variants in production use are Triton:

- **Tri Dao's Triton FlashAttention reference** is the basis for many downstream implementations and is competitive with the official CUDA version on Hopper.
- **FlashAttention with sliding windows** (used by Mistral models) is implemented in Triton in the official Mistral inference code.
- **PagedAttention with grouped query attention**, used by vLLM, is a Triton kernel that integrates with vLLM's KV cache page table.
- **Ring Attention** for very long contexts is typically implemented in Triton because the cross-device communication patterns make CUDA bookkeeping painful.

### 5.2 Mamba and State Space Models

The Mamba selective state space model requires a custom scan kernel that has no direct CUDA library equivalent. The reference implementation is a Triton kernel that performs a parallel prefix scan over the time dimension, with selective gating that depends on input-dependent parameters. The kernel was developed in Triton precisely because writing it in CUDA would have been a multi-month undertaking — and it would have been far less accessible to the broader research community.

### 5.3 Liger Kernel

Liger Kernel is an open-source library of Triton kernels for transformer training, focused on memory efficiency. It includes fused implementations of RMSNorm, RoPE, SwiGLU, cross-entropy loss with chunked computation, and several training-specific operations. Adopting Liger Kernel typically reduces training memory usage by 60% and improves throughput by 20% for large-model training, with no source code changes required from the user. The entire library is approximately 5,000 lines of Triton — comparable to a single non-trivial CUDA kernel in CUTLASS.

### 5.4 torch.compile

PyTorch's `torch.compile` (introduced in PyTorch 2.0) uses Triton as one of its primary code generation backends. When `torch.compile` is asked to optimize a function, it traces the operations, fuses them into larger compound operations, and emits Triton kernels for the fused groups. The result is that even users who never write Triton themselves benefit from it transparently when they use `torch.compile`.

The performance impact has been substantial. For a typical transformer training step, `torch.compile` with the Triton backend yields 1.5-2× throughput improvements over eager mode PyTorch, primarily through kernel fusion. This is particularly valuable for novel model architectures where hand-written CUDA kernels do not exist.

## 6. Limitations and Tradeoffs

Triton is not a panacea. There are several scenarios where raw CUDA remains the better choice.

**Maximum performance on highly optimized kernels.** For very mature kernels like dense matmul on Hopper, NVIDIA's CUTLASS templates take advantage of hardware features (warp specialization, asynchronous proxy operations, TMA loads) that Triton's compiler does not yet fully exploit. The gap is small — typically 5-10% — but for kernels that run for the entire duration of a training job, that gap can be worth the engineering effort to close.

**Operations with irregular control flow.** Triton's strength is regular tensor operations expressed as block-level computations. Operations with data-dependent control flow — sparse operations, dynamic graph operations, ragged tensors — are awkward to express in Triton and often require dropping back to CUDA.

**Fine-grained interaction with the hardware.** Some optimizations require precise control over warp scheduling, register allocation, or instruction selection. Triton's compiler makes most of these decisions for you, and there is no escape hatch when its choices are suboptimal. CUDA, with all its complexity, gives you that control.

**Compilation time.** Triton kernels are JIT-compiled the first time they are called. The compilation can take several seconds for complex kernels, which is fine for long-running training jobs but problematic for short scripts and interactive use. Triton has a kernel cache to avoid recompilation, but cache invalidation across hardware and PyTorch versions has been a source of operational headaches in some deployments.

## 7. The Triton Ecosystem

The Triton ecosystem has grown rapidly. Beyond the core compiler, several projects have emerged:

- **OpenAI Triton-MLIR**: A reimplementation of the Triton compiler on top of MLIR, providing better optimization passes and easier extensibility. As of 2025, this is the official upstream Triton.
- **Triton-CPU**: An experimental backend that targets CPUs, allowing the same Triton source to run on CPUs (with much lower performance, intended for testing and debugging).
- **Triton-IR tools**: Visualization and debugging tools for the intermediate representation, helpful for understanding what the compiler is doing.
- **Liger Kernel, Unsloth, xformers**: Libraries of pre-built Triton kernels for common transformer operations.

The community has also developed conventions and best practices: how to autotune block sizes for specific hardware, how to test kernels for numerical correctness, how to integrate Triton kernels into PyTorch's autograd, how to use Triton for backward passes.

## 8. Triton vs. Other Approaches

Several other languages and frameworks aim to make GPU kernel programming more accessible. **JAX/XLA** uses a compiler-driven approach where the user writes pure NumPy-like code and the compiler handles GPU execution. It is more abstract than Triton but offers less control over kernel-level decisions. **CUDA C++ with templates and CUTLASS** provides comparable performance to Triton but at much higher complexity. **CuPy and NumPy-on-GPU** offer ease of use but cannot fuse operations or generate custom kernels.

The closest competitor to Triton is **Pallas**, JAX's kernel programming language. Pallas is similar in spirit — block-level, compiler-driven, embedded in Python — but lives within the JAX ecosystem rather than PyTorch. The two languages are converging in terms of what they can express, and there is some hope of eventual interoperation.

## 9. The Future

Triton's trajectory is upward but not without challenges. The compiler needs to keep pace with new hardware features — Hopper's TMA, Blackwell's FP4 tensor cores, AMD's matrix accelerators on MI300 — and the gap between Triton and expert CUDA on these features tends to grow before subsequent compiler updates close it. The MLIR rewrite has helped here by providing a more extensible compilation infrastructure.

The longer-term question is whether Triton (or something like it) will become the default way to write GPU kernels for ML, displacing CUDA for everything except library-internal optimization. The trend lines suggest yes: by 2025, the majority of new GPU kernels in open-source ML are written in Triton, and the developer base of Triton-fluent ML engineers is growing rapidly. The CUDA ecosystem is not going away — NVIDIA's tooling and libraries are still essential — but the position of CUDA as the language one writes new kernels in is increasingly under pressure.

For practitioners, the lesson is straightforward: if you find yourself wanting to write a custom GPU kernel for an ML workload, start with Triton. The productivity gain is enormous, the performance is usually within a few percent of expert-written CUDA, and the resulting code is dramatically more maintainable. The cases where you need to drop down to CUDA are real but increasingly rare.

## 10. Conclusion

Triton represents a significant democratization of GPU kernel programming. It does not eliminate the need for hardware understanding — a programmer who does not know what shared memory is or why memory coalescing matters will still write slow Triton kernels — but it lowers the bar from "expert in CUDA, PTX, and SASS" to "competent in Python and basic GPU concepts." For ML researchers and engineers, this has unlocked a level of kernel-level experimentation that was previously the exclusive domain of a small priesthood of CUDA wizards.

The broader lesson is that domain-specific languages, when well-designed for the actual abstractions their users need, can substantially reduce the gap between novice and expert performance. CUDA was designed for a general audience of GPU programmers, and ML kernels are a specific subset with consistent patterns (block-level tile operations, regular memory access, tensor cores, fusion opportunities). Triton's contribution is recognizing those patterns and building a language around them, with a compiler smart enough to handle the details that previously required expert intervention. The result is fewer hand-optimized libraries and more researcher-written kernels — a healthier ecosystem and faster pace of innovation.
