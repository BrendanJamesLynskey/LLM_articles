# ML Compiler Stacks: XLA, MLIR, IREE, and TVM

*April 2026 • Technical Report*

## 1. Introduction

A modern machine learning model is, at its core, a computational graph. Inputs flow through layers of operations, each operation transforming tensors of numbers, until a final output is produced. From a compiler's perspective, this is just another program — and it has the same optimization opportunities as any other program: dead code elimination, common subexpression elimination, loop fusion, memory layout optimization, vectorization, and target-specific code generation. The role of an ML compiler is to take a high-level description of a model and produce efficient machine code for a specific target device.

The history of ML compilers is short but intense. The major frameworks (TensorFlow, PyTorch, JAX) have all developed compiler stacks of varying ambition. The hardware vendors (NVIDIA, Google, AMD, Intel) have built their own compilers tailored to their hardware. Independent projects (TVM, IREE) have emerged to cross-cut the framework-vendor matrix. The result is a landscape with multiple competing stacks, each optimized for different goals and supporting different combinations of frameworks and hardware.

This report examines the major ML compiler stacks — XLA, MLIR, IREE, and TVM — what they are, what they do, and how they fit into the broader ML infrastructure landscape. Understanding these tools is important for anyone trying to deploy models efficiently or to navigate the messy reality of getting a model from a research notebook to a production accelerator.

## 2. Why ML Needs Compilers

Before diving into the specific stacks, it is worth asking why ML even needs its own compilers. Why not just use LLVM, the dominant general-purpose compiler infrastructure?

The answer is that ML programs have specific structure that benefits from specialized compilation. Tensors are typed differently from scalars, operations on them have predictable shapes that can be reasoned about statically, and the dominant operations (matrix multiplications, convolutions, reductions) have hardware support that general-purpose compilers do not target. A general-purpose compiler can compile a matrix multiplication function, but it cannot easily emit a single tensor core instruction that does the whole operation in one cycle. The information needed for that optimization — that this multiply-accumulate-loop is a matrix multiplication, with these shapes, on these data types — is lost by the time the code reaches LLVM.

ML compilers preserve this high-level information through a hierarchy of intermediate representations (IRs). At the top, the IR represents tensor operations directly (matmul, conv, softmax). At the bottom, the IR represents target-specific instructions (PTX for NVIDIA, AMDGPU IR for AMD, vector intrinsics for CPUs). In between, multiple lowering passes progressively translate from high-level operations to low-level instructions, applying optimizations at each level where the relevant information is available.

The result is much better code than a general-purpose compiler could produce, at the cost of much more specialized infrastructure.

## 3. XLA

### 3.1 Origins and Position

XLA (Accelerated Linear Algebra) was originally developed by Google as the compiler backend for TensorFlow. It is now the primary compiler for JAX as well, and it is the only practical way to run JAX or TensorFlow code on Google's TPUs. XLA also targets NVIDIA GPUs and CPUs, with varying levels of optimization depending on the target.

XLA takes a graph of high-level operations (matmul, conv, softmax, reductions, etc.) and produces target-specific code. The graph is represented in HLO (High-Level Optimizer IR), an XLA-specific intermediate representation that captures the operations and their shapes. HLO is the canonical form on which most of XLA's optimizations operate.

### 3.2 What XLA Does

The XLA optimization pipeline includes:

- **Operator fusion**: Combine multiple small operations into single fused kernels. This is the most impactful optimization in practice, eliminating intermediate memory traffic and kernel launches.
- **Memory layout selection**: Choose data layouts (NHWC vs. NCHW, row-major vs. column-major) that match the target hardware's preferences.
- **Buffer assignment**: Plan which tensors live in which memory tiers and reuse buffers when possible.
- **Algebraic simplification**: Apply mathematical identities to simplify expressions.
- **Constant folding**: Pre-compute parts of the graph that depend only on constants.
- **Layout-aware operation selection**: Choose specialized implementations (e.g., specific tensor core paths) based on the operation's shape and the target hardware.

For TPUs, XLA is essentially the only path. The TPU's hardware does not support arbitrary CUDA-style programming; it executes programs that have been compiled by XLA into a specific instruction format. This tight coupling allows TPUs to achieve very high efficiency on supported workloads but constrains the flexibility of the programming model.

For GPUs, XLA competes with cuDNN, cuBLAS, and the various PyTorch and CUTLASS-based stacks. The competition is uneven: XLA on NVIDIA GPUs is often 10-30% slower than the best alternatives for production workloads, but it is improving steadily and offers the advantage of a unified compiler that targets multiple backends.

### 3.3 The TPU Connection

XLA's deep integration with TPUs is its defining characteristic. Google has invested heavily in making XLA the production-grade compiler for TPU workloads, and the result is that TPU performance is largely a function of XLA's quality. New TPU generations bring new XLA optimizations, and frontier models running on TPUs (Gemini, the open Gemma series, internal Google research) all depend on XLA for their inference and training performance.

This makes XLA strategically important even for users who never touch a TPU. If JAX or TensorFlow becomes the dominant ML framework for some workload, XLA's optimization quality directly affects performance. And the engineering investment Google puts into XLA shows up in the quality of the resulting binaries.

## 4. MLIR

### 4.1 What MLIR Is

MLIR (Multi-Level Intermediate Representation) is a compiler infrastructure project, not a compiler. It provides a framework for building IRs at multiple levels of abstraction, with the ability to mix levels in a single program and progressively lower from high-level abstractions to low-level instructions. MLIR was created at Google by Chris Lattner (the original author of LLVM) and is now part of the LLVM project.

MLIR is the foundational technology that several modern ML compilers are built on. XLA's newer code generation paths use MLIR internally. The Triton compiler uses MLIR. TensorFlow's TF-XLA bridge uses MLIR. PyTorch's torch.compile uses MLIR through several intermediate steps. The IREE project (covered next) uses MLIR as its core IR.

The reason MLIR has become foundational is that it solves a real problem: ML compilers historically used custom IRs that did not interoperate, making it hard to share optimization passes between compilers and to integrate new hardware backends. MLIR's flexibility — it can represent IRs at any level of abstraction, with custom operations and types — allows different teams to define their own IRs while still sharing infrastructure.

### 4.2 Dialects

MLIR's central concept is the dialect: a self-contained set of operations and types that represents a specific abstraction level or domain. There is a dialect for tensor operations (the "linalg" dialect), a dialect for affine loops, a dialect for vectorized operations, a dialect for SPIR-V code, and dozens of others. A program can mix dialects, allowing high-level tensor operations to live alongside low-level vector instructions.

The compiler's job is to lower from high-level dialects to low-level dialects, applying optimizations at each level. A typical lowering chain for a tensor operation might be: linalg → affine loops → SCF (structured control flow) → LLVM IR → machine code.

This compositional approach allows new optimization passes to be developed and shared between compilers. A pass that operates on the linalg dialect (e.g., a fusion algorithm) can be reused by any compiler that lowers through linalg, regardless of where the operation came from (XLA, IREE, custom).

### 4.3 The Stable HLO Effort

A specific MLIR-based effort worth mentioning is StableHLO. The HLO dialect that XLA uses internally has been informally exposed to other projects, but it is unstable — the operations and semantics can change with XLA versions. StableHLO is an effort to define a stable subset of HLO that can be used as a stable interchange format between frameworks and compilers. PyTorch, TensorFlow, JAX, and ONNX have all signaled support for StableHLO as a way to standardize the interface between frontends and backends.

If StableHLO succeeds, it will become the de facto interchange format for ML, replacing the patchwork of ONNX, custom HLO, and framework-specific representations that currently fragment the ecosystem.

## 5. IREE

### 5.1 What IREE Is

IREE (Intermediate Representation Execution Environment) is an open-source ML compiler and runtime, developed primarily by Google. It targets a wide range of hardware: CPUs, GPUs (CUDA, Vulkan, Metal), mobile NPUs, and embedded accelerators. Its goal is to be a universal deployment target — write the model once, deploy it anywhere with reasonable performance.

IREE uses MLIR as its core IR, with several IREE-specific dialects layered on top. The compilation pipeline takes a model from a frontend (TensorFlow, PyTorch via torch-mlir, JAX via stablehlo), lowers it through MLIR dialects, and produces an executable binary or VMFB (Virtual Machine Flatbuffer) that the IREE runtime can execute.

### 5.2 Why IREE Matters

The ML ecosystem has historically struggled with deployment portability. A model trained in PyTorch can be deployed easily on NVIDIA GPUs (through the PyTorch runtime) but is much harder to deploy on AMD GPUs, mobile NPUs, or embedded devices. ONNX Runtime addresses some of this but has its own limitations. IREE aims to be a unified deployment path that produces good performance on diverse hardware without forcing the user to maintain separate model versions.

In practice, IREE has gained traction primarily for mobile and embedded deployment, where the alternatives are weaker. The desktop/server side of the ML deployment landscape is dominated by framework-specific runtimes (PyTorch + cuDNN, ONNX Runtime, TensorRT), and IREE is competitive but not dominant in that space.

The most interesting recent IREE work has been on transformer LLM inference. The team has invested in making IREE generate competitive code for LLM workloads on diverse hardware, with reported wins on Apple Silicon, AMD GPUs, and Vulkan-based mobile GPUs. For users targeting non-NVIDIA hardware, IREE is increasingly a credible option for LLM deployment.

## 6. TVM

### 6.1 What TVM Is

TVM (Tensor Virtual Machine) is an open-source ML compiler originally developed at the University of Washington's SAMPL lab and now maintained by the Apache Software Foundation. It predates MLIR and IREE and has its own IR and optimization infrastructure. TVM was the first major ML compiler to demonstrate that automatic kernel generation could produce competitive performance with hand-tuned kernels for diverse hardware.

TVM's defining feature is its auto-scheduler, which automatically searches the space of kernel implementations to find one that performs well on the target hardware. Given a tensor operation and a target device, the auto-scheduler tries many different ways of implementing the operation (different tile sizes, parallelization strategies, memory layouts) and picks the best one through empirical measurement. The search is slow (hours for a complex model) but produces kernels that are competitive with manually-optimized libraries.

### 6.2 Position in the Landscape

TVM has a passionate user community but has not become dominant. Several factors contribute. The auto-scheduler's slow compilation makes it hard to use for rapid iteration. The optimizations are not always portable across hardware generations (a kernel optimized for Volta may not be optimal for Hopper). The ecosystem is fragmented, with multiple frontends (Relay, Relax, more recently TVM Unity) and competing approaches within the project.

For specific use cases — particularly edge deployment of fixed models where the slow compilation can be amortized over many runs — TVM remains a strong option. For general-purpose LLM deployment in 2026, it has been largely overshadowed by the more vertically-integrated stacks (XLA for JAX/TensorFlow, torch.compile for PyTorch, TensorRT-LLM for NVIDIA-targeted optimization).

## 7. PyTorch's torch.compile

PyTorch 2.0 introduced torch.compile, a JIT compilation pipeline for PyTorch models. The pipeline traces the user's Python code, captures the operations into a graph, and compiles the graph using one of several backends (Inductor by default, with options for XLA, OpenAI Triton, or others).

Inductor is PyTorch's homegrown compiler backend. It generates Triton kernels for most operations and falls back to library implementations (cuBLAS, cuDNN) for the largest matrix multiplications. Inductor uses MLIR internally for some code generation paths and is gradually moving more of its infrastructure onto MLIR.

For most PyTorch users, torch.compile is the relevant compiler stack. It is invoked transparently with a single decorator, produces meaningful speedups (typically 1.5-2× for transformer training and inference), and integrates seamlessly with the rest of PyTorch. The user does not need to know that XLA or MLIR exist — they just call `torch.compile(model)` and the speedup happens.

This is the modern face of ML compilation: invisible, transparent, and just enough configuration to expose to the user without overwhelming them with compiler internals.

## 8. The Specialized Inference Compilers

Beyond the general-purpose stacks, several specialized inference compilers exist for specific use cases.

### 8.1 TensorRT and TensorRT-LLM

NVIDIA's TensorRT is the most aggressive inference compiler for NVIDIA GPUs. It takes an ONNX model (or a model converted from PyTorch/TensorFlow) and produces a highly-optimized engine binary. The compilation process includes layer fusion, precision selection (FP16, INT8, FP8), kernel auto-tuning, and dynamic shape handling. The resulting engines achieve some of the highest inference throughputs available on NVIDIA hardware.

TensorRT-LLM is a more specialized version targeting large language models. It includes pre-built optimizations for transformer architectures, multi-GPU tensor parallelism, speculative decoding, and quantization. For NVIDIA GPU deployments of LLMs, TensorRT-LLM is one of the most performant options available, particularly when combined with FP4 on Blackwell hardware.

### 8.2 OpenVINO

Intel's OpenVINO is an inference compiler targeting Intel CPUs, GPUs, and NPUs. It supports models from PyTorch, TensorFlow, ONNX, and other frameworks, applying Intel-specific optimizations to produce executables that exploit AVX-512, AMX, and other Intel hardware features. OpenVINO is the de facto standard for inference deployment on Intel platforms.

### 8.3 ONNX Runtime

ONNX Runtime is Microsoft's cross-platform inference runtime. It accepts ONNX models and runs them on a wide range of hardware through pluggable execution providers. ONNX Runtime is not as optimized as the vendor-specific compilers but provides broader portability.

## 9. The Convergence Story

In 2026, the ML compiler landscape is converging on a few key patterns:

- **MLIR as the underlying framework**: Most new compiler projects use MLIR. The legacy projects (TVM, original XLA) are gradually moving to MLIR-based code generation.
- **Triton as the kernel language**: Most new fused kernels are written in Triton, regardless of the compiler stack that orchestrates them.
- **StableHLO as the interchange format**: A standard frontend-to-backend interface that lets different frameworks share compiler infrastructure.
- **Per-framework JIT compilers**: Each major framework has its own JIT (torch.compile for PyTorch, jax.jit for JAX, TensorFlow's tf.function), and these are increasingly the user-facing compilation interface.
- **Vendor-specific specialized compilers**: TensorRT-LLM, OpenVINO, and similar tools provide the last 10-30% of performance for production deployments.

The fragmentation is real but not chaotic. A typical production deployment might use torch.compile for development, TensorRT-LLM for the actual inference engine on NVIDIA, OpenVINO for the CPU fallback, and MLIR-based custom passes for any operations the standard tooling cannot handle. The user does not need to know all of this, but the people building the deployment infrastructure do.

## 10. Conclusion

ML compilers are increasingly invisible to end users but increasingly important to the performance of ML systems. The progress of the past five years — XLA's TPU dominance, MLIR's emergence as the common infrastructure, the maturation of torch.compile, the specialization of TensorRT-LLM — has been substantial. The compilers are now table stakes for any serious ML deployment, and the differences between them shape the practical choice of frameworks and hardware as much as the model architectures do.

For practitioners, the practical advice depends on your use case. If you are training models, the JIT compiler that comes with your framework (torch.compile, jax.jit, tf.function) is usually the right tool — use it without thinking about the layers underneath. If you are deploying models on NVIDIA hardware, TensorRT-LLM is worth the engineering investment for production workloads. If you are deploying on diverse hardware, IREE or ONNX Runtime are credible portable options. If you are building infrastructure or writing custom kernels, MLIR and Triton are the foundations to learn.

The deeper point is that ML compilers are no longer a niche concern. They are part of the critical infrastructure that determines how well models perform in practice, and the choice of compiler stack is as consequential as the choice of model architecture or hardware. Investing in compiler literacy — understanding what these tools do and when to reach for them — pays dividends in deployment efficiency and ability to navigate the increasingly complex ML hardware landscape.
