**Mixed-Precision Arithmetic for Practical Inferencing**

Numerical Format Strategies for Efficient LLM Deployment

March 2026 • Technical Report

Table of Contents

1\. Introduction

Large language model inference is a balancing act between numerical fidelity and computational efficiency. Every matrix multiplication, every attention score, and every activation in a transformer model is computed using a specific floating-point format, and the choice of that format has profound implications for memory consumption, throughput, latency, and output quality. Mixed-precision arithmetic is the practice of using different numerical precisions for different parts of the inference pipeline, assigning higher precision to operations that are sensitive to rounding error and lower precision to operations that can tolerate it. Rather than committing the entire model to a single bit width, mixed-precision strategies exploit the observation that not all tensors, layers, or operations contribute equally to model accuracy.

This report provides a practical guide to mixed-precision arithmetic as applied to LLM inference, covering the landscape of numerical formats from FP32 down to FP4, the hardware that accelerates them, the software frameworks that implement them, and the strategies practitioners use to select the right precision for each component of the inference pipeline.

2\. The Numerical Format Landscape

2.1 IEEE 754 and Beyond

Traditional deep learning relied on IEEE 754 FP32 (1 sign bit, 8 exponent bits, 23 mantissa bits) as the universal precision. FP32 provides a dynamic range spanning roughly 38 orders of magnitude with approximately 7 decimal digits of precision, far more than neural networks typically require. The transition to half-precision formats began with FP16 (1 sign, 5 exponent, 10 mantissa), which halves memory and doubles throughput on hardware with native FP16 tensor cores. However, FP16's limited exponent range can cause overflow and underflow during training, requiring loss scaling to maintain stability.

BF16 (Brain Float 16) emerged as an alternative that retains FP32's 8-bit exponent while truncating the mantissa to 7 bits. This preserves the full dynamic range of FP32 while sacrificing some precision. For inference, BF16's advantage is that models trained in BF16 can be served directly without any post-training conversion, and its wide dynamic range means it handles activation outliers gracefully. BF16 has become the de facto standard precision for LLM training and the baseline from which all lower-precision formats are measured.

2.2 FP8: The Production Workhorse

FP8 has rapidly become the preferred precision for production LLM serving on modern hardware. The format comes in two variants defined by the OFP (Open Floating Point) specification. E4M3 allocates 4 exponent bits and 3 mantissa bits, providing a moderate dynamic range with reasonable precision, and is typically used for weights and forward-pass activations. E5M2 allocates 5 exponent bits and 2 mantissa bits, offering a wider dynamic range at the cost of precision, and is preferred for gradients and backward-pass tensors. In the context of inference, E4M3 dominates because the forward pass does not involve gradients.

The practical appeal of FP8 for inference is that it provides a 2x memory reduction over BF16, enables hardware-accelerated matrix multiplication on NVIDIA Hopper and Blackwell tensor cores at the same throughput as INT8, and introduces negligible accuracy degradation for most models. FP8 quantization can often be applied as a simple post-training step with minimal calibration, making it the lowest-friction path to halving inference costs.

2.3 INT8 and INT4: Integer Precision

Integer formats remain important for LLM inference, particularly in weight-only quantization scenarios. INT8 provides 256 discrete levels with uniform spacing, which is well-suited to weight distributions that have been pre-processed to remove outliers. INT4 provides only 16 levels but achieves a 4x memory reduction over BF16, making it critical for edge deployment and cost-sensitive serving.

The key distinction between integer and floating-point low-precision formats is how they handle dynamic range. Integer formats distribute their precision uniformly across the quantization range, which means outlier values consume a disproportionate share of the representable range, compressing the resolution available for the majority of values clustered near zero. Floating-point formats, by contrast, provide logarithmically spaced levels that naturally concentrate precision near zero where most values lie. This is why FP8 often outperforms INT8 for activation quantization, while INT4 with careful per-group scaling remains competitive with FP4 for weight-only quantization.

2.4 FP4 and Microscaling Formats

The frontier of low-precision inference has moved to 4-bit floating point, driven by hardware support on NVIDIA Blackwell GPUs. Three FP4 variants are relevant for practitioners. Plain FP4 uses the E2M1 format (2 exponent bits, 1 mantissa bit), providing only 8 distinct magnitude levels. MXFP4, defined by the Open Compute Project's microscaling specification, groups 32 elements into a block sharing a single 8-bit (E8M0, power-of-two) scaling factor. NVFP4, introduced by NVIDIA for Blackwell, improves on MXFP4 by reducing the block size to 16 elements and using an E4M3 scaling factor instead of E8M0, plus an additional per-tensor FP32 scale. This two-level microscaling approach gives NVFP4 finer-grained adaptation to local value distributions, achieving accuracy close to FP8 while providing a 3.5x memory reduction over FP16 and a 1.8x reduction over FP8.

The microscaling concept is fundamental to making sub-8-bit floating point practical. By sharing a high-precision scale factor across a small block of low-precision elements, the effective precision of the representation is higher than the element bit width alone would suggest. The block's maximum-magnitude element is scaled to the FP4 maximum, and the scale factor preserves the information about the original magnitude. This means the largest value in each block is recovered at near-FP8 precision, even though it is stored in only 4 bits.

3\. Why Mixed Precision Matters for Inference

3.1 Not All Operations Need the Same Precision

A transformer model's inference pipeline contains operations with very different sensitivity to numerical precision. The large GEMM (General Matrix Multiply) operations in the feed-forward and attention projection layers account for the vast majority of compute and memory bandwidth consumption. These operations involve weight matrices with relatively well-behaved distributions and can typically tolerate aggressive quantization. In contrast, the softmax computation in attention, layer normalisation, residual additions, and the final logit computation are far more sensitive to precision because small numerical errors in these operations can compound or shift probability distributions.

Mixed-precision inference exploits this asymmetry by keeping sensitive operations in higher precision (typically BF16 or FP32) while running the dominant GEMM operations in lower precision (FP8, INT8, INT4, or FP4). The result is that most of the compute benefits from lower-precision acceleration while the numerically critical path maintains full fidelity.

3.2 The Prefill vs. Decode Asymmetry

LLM inference has two distinct phases with different computational characteristics. The prefill phase processes the entire input prompt in a single forward pass. This is compute-bound, dominated by large matrix multiplications, and benefits strongly from low-precision tensor core acceleration. The decode phase generates tokens one at a time, with each step reading the full model weights to produce a single output token. This is memory-bandwidth-bound, and the primary benefit of lower precision is reducing the volume of data that must be read from GPU memory.

This asymmetry creates an opportunity for phase-aware mixed precision. During prefill, both weight and activation quantization (W4A4 or W8A8) can accelerate the compute-bound matrix multiplications. During decode, weight-only quantization (W4A16 or W8A16) is often more effective because the bottleneck is reading weights, not computing with them. Some recent serving frameworks implement dynamic precision switching between prefill and decode phases, choosing the lowest precision that each phase can tolerate.

3.3 Layer-Wise Sensitivity

Not all layers in a transformer model respond equally to quantization. Empirically, the first and last few layers tend to be more sensitive than the middle layers, likely because the first layers establish the model's internal representation and the last layers must produce precise logit distributions. Embedding layers and the language modelling head (lm_head) are frequently kept at higher precision even when the rest of the model is aggressively quantized.

Mixed-precision strategies that assign different bit widths to different layers can achieve better accuracy-efficiency trade-offs than uniform quantization. Methods like SliM-LLM use salience metrics to determine which layers (or even which weight groups within a layer) should receive higher precision allocation. The Any-Precision LLM framework takes this further by encoding weights in a bitplane format that allows dynamic precision selection at runtime, serving different precision levels from a single model checkpoint.

4\. Hardware Support for Mixed Precision

4.1 NVIDIA GPU Generations

Each NVIDIA GPU generation has expanded the set of precisions that receive hardware acceleration. Volta (V100) introduced FP16 tensor cores. Ampere (A100) added BF16 and INT8 tensor core support, with TF32 (a 19-bit format using FP32 range with FP16 mantissa) as an implicit accumulation format. Hopper (H100) was the breakthrough for mixed-precision inference, adding native FP8 tensor cores that operate at the same throughput as INT8, along with the Transformer Engine that automates FP8 scaling factor management. Blackwell (B200/GB200) extends support to FP4 via fifth-generation tensor cores, with native NVFP4 and MXFP4 matrix multiplication support peaking at approximately 4.5 PFLOPS of dense FP4 compute on a single B200 GPU (9 PFLOPS with structured sparsity).

A critical architectural detail is that tensor cores compute in low precision but accumulate results in higher precision. FP8 tensor cores on Hopper accumulate in FP32, meaning the matrix multiplication is performed with FP8 inputs but the partial sums and output are maintained in FP32 before being optionally downcast. This accumulation strategy is essential to maintaining accuracy: without higher-precision accumulation, the rounding errors from thousands of multiply-accumulate operations would compound catastrophically.

4.2 AMD and Other Accelerators

AMD's MI300X supports FP8, INT8, and BF16 with hardware acceleration, and future AMD GPUs are expected to support MXFP4 as defined by the OCP microscaling specification. Google TPUs have long supported BF16 natively and have added INT8 support in recent generations. Intel's Gaudi accelerators support FP8 and BF16. The convergence of the industry on FP8 as a minimum standard for inference acceleration means that mixed-precision strategies built around FP8 are portable across hardware vendors, while FP4 strategies currently require NVIDIA Blackwell.

4.3 The Role of Memory Bandwidth

Modern GPUs have significantly more compute capability than memory bandwidth, especially at low precisions. An H100 SXM delivers 1,979 TFLOPS of FP8 compute but only 3.35 TB/s of HBM bandwidth. For a 70B-parameter model at FP8, the model weights occupy roughly 70 GB. Reading these weights once per token requires 70 GB of memory traffic, which at 3.35 TB/s takes approximately 21 milliseconds, yielding a theoretical maximum of about 48 tokens per second for single-batch decode. The actual compute for the matrix multiplications takes far less time than this, which is why decode is memory-bandwidth-bound and why reducing weight precision from 8 bits to 4 bits can nearly double single-batch decode throughput.

5\. Practical Mixed-Precision Strategies

5.1 FP8 Weight and Activation Quantization (W8A8)

The simplest and most widely deployed mixed-precision strategy for production serving is FP8 W8A8, where both weights and activations are quantized to FP8 E4M3. Weights are quantized statically during a post-training calibration step, with per-tensor or per-channel scaling factors computed from a small calibration dataset. Activations are quantized dynamically at runtime using per-token or per-tensor scaling. The softmax, layer norm, residual connections, and lm_head remain in BF16 or FP32.

This strategy is supported natively on Hopper and Blackwell hardware with minimal deployment friction. In vLLM, enabling FP8 inference is as simple as specifying the quantization parameter. The NVIDIA Transformer Engine automates the scaling factor management, maintaining a history of activation magnitudes and updating scaling factors to track the distribution over time. Accuracy degradation is typically negligible, with benchmark results showing less than 0.5% change in perplexity or downstream task accuracy.

5.2 INT4 Weight-Only with BF16 Activations (W4A16)

For scenarios where memory capacity is the primary constraint, INT4 weight-only quantization provides a 4x reduction in weight memory while keeping activations in full BF16 precision. Methods like GPTQ and AWQ produce high-quality INT4 weight matrices with per-group scaling (typically group size 128). At inference time, weights are dequantized to BF16 on-the-fly before matrix multiplication, using optimised kernels like Marlin that fuse the dequantization with the GEMM.

This strategy is particularly effective for single-batch or low-batch decode, where the bottleneck is reading weights from memory and the activations are too small to benefit from quantization. It is the dominant approach for running large models on consumer GPUs and for cost-sensitive cloud serving.

5.3 FP4 with Microscaling (W4A4 and W4A16)

On Blackwell hardware, NVFP4 enables true W4A4 inference where both weights and activations are in 4-bit floating point. This doubles the effective tensor core throughput compared to FP8 and further reduces memory bandwidth pressure. Early benchmarks on the RTX PRO 6000 and B200 show that NVFP4 W4A4 delivers the fastest time-to-first-token during prefill, where the compute savings are most impactful.

However, W4A4 does not always win during decode. In memory-bandwidth-bound decode scenarios, W4A16 (4-bit weights with 16-bit activations) can achieve lower per-token latency because the activation quantization overhead is not compensated by the bandwidth savings on the small activation tensors. The optimal strategy on Blackwell is therefore often phase-adaptive: W4A4 for prefill and W4A16 for decode.

5.4 KV Cache Quantization

The key-value cache, which stores attention state for all processed tokens, can grow to consume a significant fraction of GPU memory for long contexts or large batch sizes. Quantizing the KV cache from BF16 to FP8 halves its memory footprint, enabling 2-3x larger batch sizes or longer context windows without additional GPUs. FP8 KV cache quantization is generally preferred over INT8 because the floating-point format better handles the dynamic range of attention keys and values, with lower accuracy impact in most tested configurations.

More aggressive KV cache quantization to INT4 or lower is an active research area, with methods like KIVI demonstrating asymmetric 2-bit quantization of the KV cache with minimal quality degradation on certain models.

5.5 Attention-Level Mixed Precision

The attention mechanism itself presents mixed-precision opportunities. The query-key dot products and softmax computation are precision-sensitive because the softmax function amplifies small differences in its inputs. SageAttention and similar approaches keep the query and softmax computations in higher precision while quantizing the key and value projections more aggressively. This preserves attention pattern fidelity while still reducing the memory and compute cost of the attention operation.

6\. Calibration and Scaling Strategies

6.1 Static vs. Dynamic Scaling

Quantization scaling factors can be computed statically (once, during a calibration step) or dynamically (at runtime, for each input). Static scaling is simpler and has zero runtime overhead but requires the calibration data to be representative of the deployment distribution. Dynamic scaling adapts to each input's actual value range, providing better accuracy but adding the overhead of computing scaling factors at runtime. For weights, static scaling is always used since weights do not change. For activations, the choice depends on the deployment scenario: dynamic per-token scaling is more robust but slower, while static scaling is faster but may clip outlier activations.

6.2 Scaling Granularity

The granularity at which scaling factors are applied has a significant impact on quantization accuracy. Per-tensor scaling uses a single scale for the entire tensor, which is coarse-grained and can lose precision when value ranges vary significantly across dimensions. Per-channel scaling computes a separate scale for each output channel of a weight matrix, capturing inter-channel variance. Per-group scaling divides channels into small groups (32 or 128 elements) with independent scaling, providing the finest granularity. Per-block microscaling, as used in MXFP4 and NVFP4, is a variant of per-group scaling where the group structure is optimised for hardware tensor core layouts.

The trend is toward finer-grained scaling as hardware adds native support for per-group and per-block schemes. Finer granularity improves accuracy but increases the metadata overhead (more scaling factors to store and process). NVFP4's choice of 16-element blocks with E4M3 scaling factors represents a design point that balances accuracy against metadata overhead and hardware implementation complexity.

6.3 Outlier Handling

Activation outliers, values several standard deviations larger than the mean, are a persistent challenge for low-precision inference. These outliers appear in specific channels of the activation tensor and can dominate the quantization range, compressing the majority of values into a few quantization levels. Several strategies address this problem. SmoothQuant migrates the quantization difficulty from activations to weights by applying a mathematically equivalent per-channel scaling transformation. QuaRot applies Hadamard rotations to the weight and activation matrices, redistributing outlier energy across all channels and producing more uniform distributions that quantize well. For microscaling formats, the block structure inherently provides some outlier resilience because each block's scale factor adapts to the local maximum, preventing a single outlier from affecting the entire tensor.

7\. Software Ecosystem

7.1 Inference Engines

vLLM is the most widely used open-source LLM serving engine and provides comprehensive mixed-precision support. It integrates with GPTQ, AWQ, and FP8 quantized models, with optimised CUTLASS and Triton kernels for each format. vLLM's PagedAttention memory management works with quantized KV caches, and recent versions add NVFP4 support for Blackwell GPUs.

TensorRT-LLM is NVIDIA's optimised inference engine, offering the deepest integration with NVIDIA hardware features including FP8 Transformer Engine, NVFP4, and advanced kernel fusion. It provides the best single-GPU performance on NVIDIA hardware but is less portable.

llama.cpp targets CPU and consumer GPU inference, with extensive support for GGUF-format quantized models spanning from Q2 to Q8 precision levels. Its quantization formats use a mixture of block sizes and scaling strategies optimised for CPU SIMD and Apple Metal GPU execution.

7.2 Quantization Toolkits

The practical workflow for deploying a mixed-precision model typically involves selecting a base model, choosing a quantization method and target precision, running calibration with a representative dataset, and validating accuracy on relevant benchmarks before deployment. AutoGPTQ, llm-compressor, and NVIDIA's TensorRT Model Optimizer are the primary tools for this workflow, each supporting different combinations of quantization methods, precision formats, and target deployment engines.

8\. Accuracy Considerations

The following table summarises typical accuracy-efficiency characteristics for common mixed-precision inference configurations.

  --------------------- ----------------------- -------------------------------- -------------------------------- -------------------------------
  **Configuration**     **Memory Reduction**    **Accuracy Impact**              **Hardware Requirement**          **Best Use Case**

  BF16 (baseline)       1x                      None                             Any modern GPU                   Reference, quality-critical

  W8A8 FP8              2x                      Negligible (<0.5% ppl)           Hopper/Blackwell                 Production serving

  W8A8 INT8             2x                      Negligible--Minor                Ampere and newer                 Production, broad HW compat.

  W4A16 INT4 (GPTQ)     4x weights              Minor (1--3% ppl increase)       Any GPU with dequant kernels     Edge, consumer, cost-sensitive

  W4A16 INT4 (AWQ)      4x weights              Minor (1--2% ppl increase)       Any GPU with dequant kernels     Edge, consumer, cost-sensitive

  W4A4 NVFP4            ~3.5x effective         Minor (<1% with calibration)     Blackwell only                   High-throughput Blackwell

  W4A4 MXFP4            ~3.8x effective         Minor--Moderate                  Blackwell (NVIDIA), future AMD   Research, Blackwell serving

  FP8 KV + W4A16        4x weights, 2x KV       Minor                            Hopper/Blackwell                 Long-context, high-batch
  --------------------- ----------------------- -------------------------------- -------------------------------- -------------------------------

Accuracy impacts are approximate and vary by model architecture, size, and evaluation task. Larger models generally tolerate quantization better than smaller ones, and task-specific evaluation is always recommended before deployment.

9\. Emerging Directions

9.1 Phase-Adaptive Precision

As discussed in Section 3.2, the different computational profiles of prefill and decode suggest that dynamically switching precision between phases can improve overall efficiency. Recent work on progressive mixed precision during decode takes this further, starting generation at higher precision and gradually lowering precision as the sequence progresses, exploiting the observation that early tokens in a generation tend to be more influential on the final output than later tokens.

9.2 Format-Specialised Quantization

The introduction of hardware-native FP4 formats has revealed that existing quantization algorithms designed for integer formats do not transfer optimally to floating-point formats. MR-GPTQ (Micro-Rotated GPTQ) is an example of a format-specialised algorithm that tailors the quantization process to FP4's unique properties, using Hadamard rotations fused into the weight matrices and fast online rotation of activations. On Blackwell B200 GPUs, MR-GPTQ achieves up to 3.6x layer-wise speedup over FP16 and 2.2x end-to-end speedup while maintaining competitive accuracy.

9.3 Mixed-Precision for MoE Models

Mixture of Experts models present unique mixed-precision challenges because expert utilisation is highly non-uniform. Frequently activated experts may benefit from higher precision to maintain quality on common inputs, while rarely activated experts can tolerate more aggressive quantization. Expert-aware mixed-precision allocation is an emerging strategy that assigns precision based on activation frequency and sensitivity analysis.

9.4 One-Bit and Sub-Two-Bit Research

At the extreme end, BitNet and related approaches explore 1.58-bit (ternary) weight representations, where weights are constrained to {-1, 0, 1}. This eliminates multiplication entirely, replacing it with addition and subtraction. While current 1-bit models show noticeable accuracy degradation for large-scale LLMs, the approach is promising for on-device inference where power consumption is the primary constraint. Methods like BinaryMoS (Binary Mixture of Scales) use token-adaptive scaling factors to improve the representational capacity of binarised weights.

10\. Practical Decision Framework

Selecting a mixed-precision strategy for a given deployment involves balancing several constraints. For cloud serving on Hopper GPUs where quality is paramount, FP8 W8A8 with FP8 KV cache provides an excellent quality-to-cost ratio with minimal deployment complexity. For cloud serving on Blackwell where cost efficiency is the priority, NVFP4 W4A4 during prefill combined with W4A16 during decode maximises throughput. For edge deployment on consumer GPUs, INT4 weight-only quantization with GPTQ or AWQ remains the most practical choice given the broad hardware compatibility. For latency-critical applications where time-to-first-token matters most, NVFP4 W4A4 on Blackwell delivers the fastest prefill performance currently available.

In all cases, the embedding layer and language modelling head should be kept at higher precision unless memory constraints are extreme. The KV cache should be quantized to FP8 whenever possible, as the memory savings enable higher batch sizes that improve overall system throughput. Calibration data should be representative of the deployment domain, and accuracy should be validated on task-specific benchmarks rather than relying solely on perplexity measurements.

11\. Conclusion

Mixed-precision arithmetic has evolved from a training optimisation into a fundamental inference strategy that touches every component of the LLM serving stack. The key insight is that precision is not a single global setting but a resource to be allocated judiciously across weights, activations, KV cache, and different phases of inference. With hardware vendors now providing native support for formats spanning from FP32 down to FP4, and with software frameworks maturing to manage the complexity of multi-precision deployment, practitioners have an unprecedented degree of control over the accuracy-efficiency trade-off. The most effective deployments will be those that match precision to sensitivity at a fine granularity: high precision where it matters, low precision where it does not, and the intelligence to know the difference.
