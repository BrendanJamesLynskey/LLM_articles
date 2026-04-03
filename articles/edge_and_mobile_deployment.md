# Edge and Mobile LLM Deployment

*April 2026*

## 1. Introduction

Running large language models on edge devices—smartphones, tablets, laptops, embedded systems, and IoT hardware—has transitioned from a research curiosity to a production reality. Apple Intelligence runs on-device models for text rewriting, summarization, and notification prioritization on iPhones and Macs. Google's Gemini Nano powers on-device features in Pixel phones and Chrome. Samsung's Galaxy AI uses a mix of on-device and cloud models. Microsoft's Copilot+ PCs leverage NPUs for local AI workloads. These are not toy demonstrations; they are shipping features used by hundreds of millions of people.

The motivations for on-device deployment are compelling: zero-latency responses without network round-trips, privacy by keeping data on the device, offline availability, and elimination of per-query cloud costs. The challenges are equally formidable: mobile devices have 4-16 GB of RAM (shared between the model, the operating system, and all other applications), limited thermal headroom (sustained compute generates heat that throttles performance), battery constraints (GPU and NPU usage drains batteries), and heterogeneous hardware (different chips, different accelerators, different memory architectures across devices).

This report provides a comprehensive technical examination of edge and mobile LLM deployment. It covers the compression techniques that make models small enough to fit on devices, the hardware accelerators that run them, the inference engines that tie hardware and models together, and the practical considerations for building on-device AI features. The intended audience is engineers deploying LLMs on edge devices, product managers evaluating on-device vs. cloud trade-offs, and researchers working on model efficiency.

## 2. Why On-Device

### 2.1 Latency

Cloud-based LLM inference involves network round-trips: the query travels from the device to the cloud, the model processes it, and the response travels back. Even on fast connections, this adds 50-200ms of latency per request. For streaming responses, each token requires a round-trip or relies on a persistent connection that may be interrupted.

On-device inference eliminates network latency entirely. The model runs locally, and the first token can be generated within 50-200ms of the request (depending on model size and hardware). For applications where responsiveness is critical—predictive text, code completion, real-time transcription, interactive assistants—the latency difference between on-device and cloud is perceptible and meaningful.

### 2.2 Privacy

On-device inference keeps data on the device. The query, the model's reasoning, and the response never leave the device. This is a strong privacy guarantee—stronger than any cloud-based privacy policy, because the data exposure is zero by construction rather than by promise.

This matters for several categories of data:

**Personal communications.** Email summarization, message drafting, and writing assistance involve processing the user's private messages. On-device processing avoids sending these to a third party.

**Health and financial data.** Queries about symptoms, medications, bank transactions, or investments are sensitive. On-device processing provides privacy without requiring the user to trust a cloud provider.

**Enterprise data.** For corporate applications, on-device inference can process confidential documents without sending them to an external API, simplifying compliance with data residency requirements and confidentiality obligations.

**Ambient sensing.** Devices that continuously process audio (smart speakers, hearing aids) or images (cameras, AR glasses) generate data streams that are impractical and undesirable to send to the cloud. On-device models can process these streams locally, extracting relevant information and discarding the raw data.

### 2.3 Offline Availability

On-device models work without an internet connection. This is valuable in:

- Airplane mode and underground transit
- Rural areas with limited connectivity
- Developing regions where mobile data is expensive
- Industrial environments (factories, mines, ships) with restricted connectivity
- Military and emergency response scenarios where connectivity is unreliable

### 2.4 Cost

Cloud inference has a per-query cost that scales with usage. On-device inference has a one-time cost (distributing the model to the device) and a marginal cost of essentially zero per query (just the electricity consumed by the device, which is borne by the user). For applications with high query volumes or continuous background processing, the cost savings from on-device inference can be enormous.

For an application processing 100 queries per user per day across 100 million users, the cloud inference cost (even at $0.10 per million tokens) is substantial. On-device processing reduces this to zero marginal cost for the application provider.

## 3. Model Compression Techniques

### 3.1 Quantization

Quantization is the most important compression technique for on-device deployment. It reduces the numerical precision of model weights from 16-bit floating point (FP16/BF16) to lower precisions:

**INT8 quantization.** Reduces model size by 2x (16 bits → 8 bits). Quality loss is typically less than 1% on benchmarks. Widely supported by all major inference engines and hardware accelerators. This is the minimum level of quantization for on-device deployment.

**INT4 quantization.** Reduces model size by 4x (16 bits → 4 bits). Quality loss is 1-5%, depending on the model and quantization method. This is the sweet spot for most on-device deployments, balancing quality and size.

**Mixed-precision quantization.** Different layers or components of the model are quantized to different precisions. Attention layers (which are more sensitive to quantization) may use INT8, while feedforward layers (which are more robust) use INT4. This provides better quality than uniform INT4 at similar average precision.

**Quantization methods for on-device:**

- **GPTQ (post-training quantization):** Calibrates quantization parameters on a small dataset, minimizing quantization error layer by layer. Produces high-quality INT4 models with a one-time calibration cost.
- **AWQ (Activation-Aware Weight Quantization):** Identifies the most important weight channels (based on activation magnitudes) and preserves their precision, quantizing less important channels more aggressively. Often produces better quality than GPTQ at the same bit width.
- **GGUF format (llama.cpp):** A quantization format designed for CPU inference, supporting multiple quantization levels (Q2_K through Q8_0) with different quality-size trade-offs. The "K-quants" variants use per-block scaling factors that improve quality at low bit widths.
- **QAT (Quantization-Aware Training):** Simulates quantization during training, allowing the model to learn weights that are robust to quantization. Produces the best quality at a given bit width but requires a training phase, making it more expensive than post-training methods.

### 3.2 Pruning

Pruning removes weights or structures from the model that contribute least to output quality:

**Unstructured pruning.** Sets individual weights to zero based on magnitude or importance scores. Can achieve high sparsity (50-90%) with modest quality loss. However, unstructured sparsity is difficult to accelerate on most hardware—sparse matrix operations are slower than dense operations on GPUs and NPUs unless the sparsity pattern is structured.

**Structured pruning.** Removes entire neurons, attention heads, or layers. Structured pruning produces models with fewer parameters that can be served with standard dense inference, without requiring sparse computation support. The quality impact is typically higher than unstructured pruning at the same compression ratio.

**Layer dropping.** A specific form of structured pruning that removes entire transformer layers. Empirically, middle layers of deep transformers can often be removed with limited quality impact, because adjacent layers perform similar computations. Removing 25-30% of layers from a 32-layer model typically results in 5-10% quality degradation but a proportional reduction in inference cost and model size.

### 3.3 Knowledge Distillation

Knowledge distillation trains a smaller model (the student) to replicate the behavior of a larger model (the teacher). The student is trained on the teacher's output distributions rather than (or in addition to) the original training data. The student model can use a different architecture optimized for the target device.

Distillation is the primary technique used by model developers to create on-device model families:

- Google's Gemini Nano (1.8B and 3.25B parameters) is distilled from larger Gemini models.
- Apple's on-device models (~3B parameters) are distilled from larger Apple Foundation Models.
- Microsoft's Phi series uses a combination of distillation from larger models and training on curated "textbook-quality" data.

The quality gap between the teacher and student depends on the size ratio and the distillation method. A 3B student can typically capture 70-85% of a 70B teacher's quality on general benchmarks, with higher capture rates on tasks that do not require extensive world knowledge or complex reasoning.

### 3.4 Architecture Search and Design

Purpose-built small model architectures can outperform scaled-down versions of large model architectures at the same parameter count. Design choices that matter for on-device performance include:

**Depth vs. width trade-off.** On-device inference benefits from wider, shallower models because each layer adds a sequential computation step (increasing latency), while width can be parallelized on the device's GPU/NPU. On-device-optimized models tend to be wider relative to their depth compared to cloud-optimized models.

**Attention mechanism.** Grouped query attention (GQA) or multi-query attention (MQA) reduces the KV cache size, which is critical for fitting models in limited device memory. Most on-device models use GQA with 4-8 groups.

**Vocabulary size.** Larger vocabularies increase the embedding and output projection layers. On-device models sometimes use smaller vocabularies (32K-50K tokens vs. 100K+) to reduce these costs, at the expense of slightly less efficient tokenization.

**Activation functions.** SiLU/SwiGLU activations provide better quality than ReLU but are more expensive to compute. On-device models typically retain SiLU/SwiGLU because the compute cost of activations is small relative to matrix multiplications.

## 4. Mobile-Optimized Model Families

### 4.1 Phi (Microsoft)

The Phi series from Microsoft Research was specifically designed to demonstrate that small models trained on high-quality data can achieve disproportionate performance:

- **Phi-1 (1.3B):** Focused on code generation, trained on "textbook-quality" code data.
- **Phi-2 (2.7B):** General-purpose, trained on curated data mixing web content with synthetic "textbook" data.
- **Phi-3 Mini (3.8B):** Competitive with Llama 3 8B on many benchmarks despite being half the size.
- **Phi-3.5 Mini (3.8B):** Improved multilingual and long-context performance.
- **Phi-4 Mini (3.8B, 2025):** Further quality improvements through better training data and recipes.

The Phi family's key insight is that data quality can substitute for model size, making small models viable for on-device deployment without excessive quality sacrifice.

### 4.2 Gemma (Google DeepMind)

Google's Gemma models are open-weight models designed for efficient deployment:

- **Gemma 2B and 7B (2024):** Open-weight models trained on Google's data infrastructure.
- **Gemma 2 2B and 9B (2024):** Improved architecture with better per-parameter quality.
- **Gemma 3 (2025):** Further architectural improvements and multimodal capabilities.

Gemma models are designed to work well with Google's MediaPipe and TensorFlow Lite inference engines and are optimized for deployment on Android devices and in Chrome.

### 4.3 SmolLM (Hugging Face)

SmolLM is Hugging Face's series of small language models optimized for on-device inference:

- **SmolLM 135M, 360M, 1.7B:** A range of sizes targeting different device capabilities.
- **SmolLM2 (2024):** Improved versions with better training data and recipes.

SmolLM targets the extremely small end of the model spectrum, providing useful (if limited) capabilities in packages small enough for the most constrained devices—including IoT devices and older smartphones with limited RAM.

### 4.4 Apple OpenELM

Apple's OpenELM (Efficient Language Model) is a family of open-source models designed for on-device use:

- **OpenELM 270M, 450M, 1.1B, 3B:** Range of sizes for different Apple device capabilities.
- **Layer-wise scaling:** Different layer widths at different depths, with narrower early layers and wider later layers, optimized for the Apple Neural Engine's characteristics.

Apple's production on-device models (used in Apple Intelligence) are not open-source but are believed to use similar architectural principles, tuned for the specific capabilities of Apple Silicon's Neural Engine.

### 4.5 Qwen2.5 and Llama Small Variants

Both the Qwen (Alibaba) and Llama (Meta) families include small variants designed for efficient deployment:

- **Qwen2.5-0.5B, 1.5B, 3B:** Small Qwen models that are highly competitive for their size.
- **Llama 3.2 1B and 3B:** Meta's smallest Llama models, optimized for mobile deployment and supported by the llama.cpp and ExecuTorch ecosystems.

These models benefit from the scaling insights and training infrastructure of their larger siblings, providing small models that are trained on substantially more data than would be typical for their size class.

## 5. Inference Engines

### 5.1 llama.cpp

llama.cpp is the most widely used inference engine for on-device LLM deployment. Written in C/C++, it provides:

**CPU inference.** Highly optimized for CPU execution using AVX, AVX-512, and ARM NEON SIMD instructions. This is the primary execution mode for most on-device deployments, because every device has a CPU, while GPU/NPU support varies.

**Quantization support.** Native support for GGUF format quantizations from Q2 to Q8, with "K-quant" variants that provide better quality per bit than uniform quantization. The GGUF format has become the de facto standard for distributing quantized models.

**Metal backend (Apple).** GPU acceleration on Apple devices using the Metal API, leveraging the Apple GPU for matrix multiplications. Performance is 2-5x better than CPU-only on Apple Silicon.

**Vulkan backend.** GPU acceleration on Android and Windows devices using the Vulkan API. Performance varies by device but is typically 1.5-3x better than CPU-only.

**CUDA backend.** GPU acceleration on NVIDIA hardware, primarily for laptops and desktops with discrete GPUs.

llama.cpp's key advantage is portability—it runs on virtually any device with a C compiler, from a Raspberry Pi to a high-end workstation—and its community, which rapidly adds support for new models, quantization methods, and hardware backends.

### 5.2 MLC LLM

MLC LLM (Machine Learning Compilation for LLMs) is an inference engine built on the Apache TVM compiler framework. It compiles models to optimized code for specific hardware targets:

**Hardware-specific optimization.** MLC generates code optimized for the specific GPU, NPU, or CPU of the target device. This can provide better performance than generic inference engines, especially on hardware with unusual characteristics.

**Vulkan, Metal, and OpenCL support.** Supports GPU acceleration on all major mobile and desktop platforms.

**WebGPU support.** Enables running LLMs in the browser through WebGPU, the successor to WebGL that provides general-purpose GPU compute access from JavaScript.

**Android and iOS support.** Native mobile SDKs for integrating LLMs into mobile applications.

### 5.3 MediaPipe LLM Inference

Google's MediaPipe LLM Inference API is designed for deploying models on mobile devices with Google's ecosystem:

**Optimized for Gemma and Gemini Nano.** The inference engine is specifically tuned for Google's model architectures and quantization formats.

**GPU delegate.** Uses OpenGL ES (Android) or Metal (iOS) for GPU acceleration. The GPU delegate is highly optimized for the specific GPU architectures found in common mobile SoCs (Qualcomm Adreno, ARM Mali, Apple GPU).

**Cross-platform.** Supports Android, iOS, and web through a unified API.

### 5.4 Core ML (Apple)

Apple's Core ML framework provides optimized inference on Apple devices:

**Neural Engine integration.** Core ML can target the Apple Neural Engine (ANE), a dedicated NPU that provides high-throughput, energy-efficient inference. The ANE is optimized for matrix multiplications and convolutions at INT8 and FP16 precision.

**Model optimization pipeline.** Core ML includes tools for converting, quantizing, and optimizing models for Apple hardware. The `coremltools` Python package handles conversion from PyTorch, and the Core ML compiler generates optimized code for the target device.

**Integration with Apple Intelligence.** Apple's on-device models are served through a private variant of the Core ML runtime, with additional optimizations for the specific models used by Apple Intelligence features.

### 5.5 ONNX Runtime Mobile

Microsoft's ONNX Runtime includes a mobile variant optimized for edge deployment:

**Cross-platform.** Supports Android, iOS, Windows, and Linux on ARM.

**NNAPI support (Android).** Delegates computation to the Android Neural Networks API, which can target the device's NPU, GPU, or DSP.

**DirectML support (Windows).** Leverages DirectX for GPU acceleration on Windows devices, including Copilot+ PCs with NPUs.

**Quantized model support.** Native support for INT8 and INT4 quantized models in ONNX format.

### 5.6 ExecuTorch (Meta)

ExecuTorch is Meta's inference framework for deploying PyTorch models on edge devices:

**PyTorch-native.** Models are exported from PyTorch using `torch.export` and compiled for the target device. This provides a smooth workflow for researchers and developers who work in PyTorch.

**Backend delegation.** ExecuTorch delegates computation to platform-specific backends: Core ML on Apple, XNNPACK for CPU, Qualcomm QNN for Hexagon NPU, and others.

**Optimized for Llama.** ExecuTorch is specifically optimized for the Llama model family, with Llama 3.2 1B and 3B as primary deployment targets.

## 6. Hardware Accelerators on Mobile

### 6.1 Apple Neural Engine (ANE)

The Apple Neural Engine is a dedicated NPU present in all Apple Silicon chips (A-series for iPhones, M-series for Macs and iPads):

- **A17 Pro (iPhone 15 Pro):** 35 TOPS (INT8)
- **M4 (iPad Pro, Mac):** 38 TOPS (INT8)
- **M4 Pro/Max:** Up to 76 TOPS (INT8)

The ANE is optimized for specific operation patterns: batch matrix multiplications at INT8 and FP16, convolutions, and element-wise operations. It achieves very high energy efficiency—roughly 5-10 TOPS per watt, compared to 0.5-2 TOPS per watt for the GPU—making it the preferred accelerator for sustained inference workloads where battery life matters.

However, the ANE has constraints: it has limited memory (it accesses the unified memory pool, not a dedicated cache), supports a limited set of operations (some operations must fall back to the GPU or CPU), and has latency overhead for transferring data to and from the ANE execution pipeline.

### 6.2 Qualcomm Hexagon NPU

Qualcomm's Hexagon processor is the NPU in Snapdragon SoCs, found in most Android flagship phones:

- **Snapdragon 8 Gen 3:** 73 TOPS (INT8), with dedicated transformer acceleration
- **Snapdragon 8 Elite:** 75+ TOPS (INT8), improved transformer support

The Hexagon NPU includes dedicated hardware blocks for matrix multiplications and attention operations. Qualcomm's AI Engine Direct SDK provides access to the Hexagon NPU, and Qualcomm has worked with model developers (including Meta and Hugging Face) to optimize Llama and other models for Hexagon deployment.

### 6.3 MediaTek APU

MediaTek's AI Processing Unit (APU) is found in Dimensity SoCs, used in many mid-range and flagship Android phones:

- **Dimensity 9300:** Up to 46 TOPS (INT8)
- **Dimensity 9400:** Up to 50+ TOPS (INT8)

MediaTek's NeuroPilot SDK provides access to the APU, with support for common model formats and quantization schemes.

### 6.4 Mobile GPUs

In addition to dedicated NPUs, mobile GPUs provide general-purpose compute for LLM inference:

- **Apple GPU (M4):** ~4 TFLOP/s FP16, with unified memory architecture providing high memory bandwidth (120-800 GB/s depending on the chip).
- **Qualcomm Adreno GPU:** ~2-4 TFLOP/s FP16, with support for Vulkan compute shaders.
- **ARM Mali GPU:** ~1-3 TFLOP/s FP16, commonly found in Samsung Exynos and other SoCs.

Mobile GPUs are more flexible than NPUs (supporting arbitrary compute kernels) but less energy-efficient for the specific operation patterns used in LLM inference. In practice, the best performance often comes from a hybrid approach: NPU for the bulk of matrix multiplications, GPU for operations not supported by the NPU, and CPU for control flow and preprocessing.

### 6.5 Intel and AMD NPUs for Laptops

Windows laptops increasingly include NPUs for AI workloads:

- **Intel Meteor Lake/Lunar Lake NPU:** 10-48 TOPS (INT8)
- **AMD Ryzen AI (XDNA):** 16-50 TOPS (INT8)
- **Qualcomm Snapdragon X Elite (for Windows):** 45 TOPS (INT8)

These NPUs enable on-device LLM inference on laptops without requiring a discrete GPU. Microsoft's Copilot+ PC initiative requires a minimum of 40 TOPS NPU performance, establishing a baseline for on-device AI capability.

## 7. Memory Constraints

### 7.1 The Memory Budget

On mobile devices, the model must share memory with the operating system, other applications, and the inference engine's runtime overhead. The available memory for the model is a fraction of the total device RAM:

- **4 GB RAM (low-end phones):** ~1-2 GB available for the model. Only sub-1B models (heavily quantized) are practical.
- **6-8 GB RAM (mid-range phones):** ~2-4 GB available. 1-3B models at INT4 (0.5-1.5 GB) fit comfortably.
- **12-16 GB RAM (flagship phones):** ~6-10 GB available. 7-8B models at INT4 (3.5-4.5 GB) are feasible with careful memory management.
- **16-32 GB RAM (laptops):** ~8-24 GB available. 7-13B models at INT4, or even 70B models at aggressive quantization on high-end configurations.

### 7.2 KV Cache Memory on Device

The KV cache is an even tighter constraint on mobile than in the cloud, because the memory budget is smaller and the KV cache competes with the model weights for the same limited memory.

For a 3B model with GQA (8 KV heads, 128 head dim, 32 layers) at FP16, the KV cache for a 2048-token context is:

```
2 × 32 × 8 × 128 × 2048 × 2 bytes ≈ 268 MB
```

This is substantial—roughly 20-30% of the available memory for the model on a mid-range phone. Extending the context window to 8192 tokens would require ~1 GB of KV cache, leaving less memory for the model itself.

Techniques to manage KV cache on device:

**INT8 or INT4 KV cache quantization.** Reduces KV cache memory by 2-4x with modest quality impact (typically <1% degradation).

**Sliding window attention.** Limits the context window to a fixed size (e.g., 1024 or 2048 tokens), discarding older context. This caps the KV cache size but limits the model's ability to reference earlier conversation turns.

**Sparse attention patterns.** Use full attention for recent tokens and sparse attention for older tokens, reducing the memory required for the older portion of the KV cache.

### 7.3 Model Loading and Memory-Mapped Files

Loading a 3B INT4 model (~1.5 GB) into memory takes 1-3 seconds from flash storage on a modern phone. This latency is acceptable for explicit model invocations (user opens the assistant) but not for always-ready features (keyboard prediction, notification processing).

Memory-mapped (mmap) file loading addresses this by mapping the model file directly into virtual memory without copying it all to RAM immediately. Pages are loaded from storage on demand as they are accessed. This provides:

- **Near-instant startup.** The model is "loaded" in milliseconds (only the file mapping is created, not the data copy).
- **Memory pressure handling.** The OS can evict model pages under memory pressure and reload them when needed, preventing the model from causing out-of-memory conditions.
- **Shared memory.** Multiple processes can map the same model file, sharing the physical memory.

The trade-off is that the first inference may be slow (as pages are loaded from storage) and that sustained inference may cause page faults if the model is larger than available RAM, severely degrading performance. In practice, mmap works well when the model fits comfortably in available RAM and the pages have been "warmed" by a pre-loading pass.

## 8. Battery and Thermal Constraints

### 8.1 Power Consumption

LLM inference on mobile devices consumes significant power:

- **NPU inference (Apple ANE, Qualcomm Hexagon):** 2-5 watts during sustained inference.
- **GPU inference (Metal, Vulkan):** 3-8 watts during sustained inference.
- **CPU inference (NEON/AVX):** 4-10 watts during sustained inference.

For perspective, a typical smartphone battery is 4,000-5,000 mAh at 3.7V (roughly 15-18 Wh). Sustained inference at 5 watts would drain the battery in 3-3.5 hours. This means that on-device LLM inference is viable for interactive use (short bursts of inference followed by idle periods) but not for continuous background processing without significant battery impact.

### 8.2 Thermal Throttling

Mobile devices have limited thermal dissipation—no fans, small heat sinks, and plastic or glass enclosures that insulate rather than dissipate heat. Sustained high-power computation causes the device to heat up, triggering thermal throttling that reduces clock speeds and computation throughput.

Typical thermal throttling behavior:

- **0-30 seconds:** Full performance. The device can sustain peak compute because the thermal mass of the device absorbs the heat.
- **30-120 seconds:** Moderate throttling. The device reduces clock speeds by 10-30% as the temperature rises.
- **120+ seconds:** Significant throttling. Clock speeds may be reduced by 30-50%, halving sustained throughput compared to peak.

This means that on-device inference performance is not constant—it degrades over time during sustained use. Benchmarks that measure peak performance (first few seconds) will overstate real-world performance for applications that require sustained inference (e.g., processing a long document, generating a long response).

### 8.3 Optimizing for Power and Thermal

**Batch processing during charging.** Schedule compute-intensive tasks (document processing, model updates) when the device is charging, avoiding battery drain and taking advantage of better thermal management.

**NPU preference.** The NPU is typically 3-5x more energy-efficient than the GPU for the operations used in LLM inference. Prefer NPU execution for battery-conscious deployment.

**Aggressive quantization.** Lower-precision computation uses less energy per operation. INT4 inference uses roughly half the energy of INT8 at the same throughput.

**Dynamic computation budget.** Reduce the computation budget (shorter generation, smaller models) when the battery is low or the device is thermally throttled.

## 9. On-Device Fine-Tuning

### 9.1 Why Fine-Tune on Device

On-device fine-tuning adapts the model to the specific user's data and preferences without sending that data to the cloud. Use cases include:

- **Personalized text prediction.** Fine-tuning on the user's writing history to improve keyboard prediction quality.
- **Task-specific adaptation.** Fine-tuning on the user's specific task domain (e.g., a lawyer's legal terminology, a developer's codebase patterns).
- **Federated learning.** Fine-tuning on local data and aggregating updates across devices to improve the global model without centralizing data.

### 9.2 Technical Challenges

On-device fine-tuning faces severe resource constraints:

**Memory for gradients.** Full fine-tuning requires storing gradients and optimizer states, which typically require 3-4x the model weight memory. For a 3B model at FP16 (6 GB), full fine-tuning would require 18-24 GB of memory—far exceeding device capabilities.

**LoRA and QLoRA.** Parameter-efficient fine-tuning methods like LoRA add small adapter matrices (typically 0.1-1% of total parameters) and only compute gradients for these adapters. QLoRA combines this with quantized base weights, reducing the memory requirement to the model size plus a small overhead for the adapters and their gradients.

**Limited compute.** Fine-tuning even a small model on a mobile device is slow. A single fine-tuning epoch on 10,000 examples might take hours on a smartphone, compared to minutes on a cloud GPU.

**Training data quality.** On-device data is typically small, noisy, and unbalanced. Overfitting is a constant risk, requiring careful regularization and early stopping.

### 9.3 Practical Approaches

**Apple's approach.** Apple Intelligence uses a variant of on-device adaptation where user-specific signals (writing patterns, app usage, contact information) are incorporated into the model's context rather than through weight updates. This avoids the computational cost of fine-tuning while providing personalization.

**Federated fine-tuning.** Google's federated learning framework has been used for keyboard prediction models (Gboard) for years. The framework fine-tunes the model on each device's local data, sends only the gradient updates (not the data) to a central server, and aggregates updates from many devices to improve the global model. This provides privacy-preserving personalization at scale.

**Background fine-tuning.** Fine-tuning runs as a low-priority background task, processing a few examples whenever the device is idle and charging. Over days or weeks, the model accumulates sufficient fine-tuning to improve on the user's specific patterns.

## 10. Hybrid Edge-Cloud Architectures

### 10.1 The Hybrid Approach

Many production deployments use a hybrid architecture where the on-device model handles most queries and a cloud model handles queries that exceed the on-device model's capabilities:

1. Query enters the on-device model.
2. If the on-device model is confident and the query does not require external knowledge, return the on-device response.
3. If the on-device model is uncertain, the query is complex, or the query requires up-to-date information, escalate to the cloud model.

This is essentially model routing (as described in the companion article) where the small model is on-device and the large model is in the cloud.

### 10.2 Apple Intelligence as a Case Study

Apple Intelligence (introduced in 2024-2025) uses a three-tier architecture:

**Tier 1: On-device model (~3B parameters).** Handles simple tasks: text rewriting, brief summarization, notification prioritization, Smart Reply generation. Runs on the Apple Neural Engine with INT4 quantization.

**Tier 2: Private Cloud Compute.** For queries that exceed the on-device model's capabilities, Apple routes to its Private Cloud Compute (PCC) infrastructure—servers running Apple Silicon that process queries in a privacy-preserving enclave. The user's data is processed on Apple hardware, in encrypted memory, with no data persistence.

**Tier 3: Third-party cloud models.** For queries that the user explicitly requests be handled by a third-party model (e.g., ChatGPT integration), Apple provides the option to send the query to an external API, with user consent.

This architecture provides a graceful degradation: most queries are handled locally with zero latency and full privacy, complex queries are handled by more capable cloud models with strong privacy guarantees, and the most capable models are available with explicit user consent.

### 10.3 Latency-Aware Routing

In hybrid architectures, the routing decision must consider not just model capability but network conditions:

- **Good connectivity:** Route to the cloud model when it would provide a significantly better response.
- **Poor connectivity:** Prefer the on-device model even for harder queries, accepting lower quality to avoid latency spikes.
- **No connectivity:** All queries handled on-device.

Adaptive routing based on network quality requires monitoring RTT (round-trip time), bandwidth, and connection stability, and adjusting the routing threshold accordingly.

### 10.4 Prefill-Decode Splitting

An advanced hybrid pattern splits the inference workload between device and cloud:

**On-device prefill, cloud decode.** The device processes the input prompt (prefill) and sends the resulting KV cache to the cloud, which performs the token generation (decode). This keeps the input data on-device during prefill but requires sending the KV cache (which is large) to the cloud.

**Cloud prefill, on-device decode.** The cloud processes the prompt and sends the KV cache to the device, which generates tokens locally. This leverages the cloud's faster prefill performance while keeping the generated output local.

Both patterns are experimental and face practical challenges around KV cache transfer latency and size. They are not widely deployed as of 2026 but represent potential future optimizations for specific use cases.

## 11. WebLLM: Browser-Based Deployment

### 11.1 Running LLMs in the Browser

WebLLM (and related projects like Transformers.js) enables running LLMs directly in the web browser using WebGPU for GPU acceleration. The model weights are downloaded to the browser's cache, and inference runs entirely on the client side.

**WebGPU.** The successor to WebGL, WebGPU provides general-purpose GPU compute from JavaScript through a shader-based programming model. As of 2026, WebGPU is supported in Chrome, Edge, Firefox, and Safari, covering the majority of desktop and mobile browsers.

**Performance.** Browser-based inference is typically 30-60% slower than native inference (llama.cpp with Metal/Vulkan) due to WebGPU overhead, JavaScript runtime overhead, and limitations on memory management. However, for small models (1-3B), browser-based inference provides acceptable performance—3-10 tokens per second on a modern laptop.

### 11.2 Use Cases

Browser-based LLMs are useful for:

**Privacy-sensitive web applications.** A web application that processes sensitive data can run the LLM in the browser, ensuring that data never leaves the user's device, even though the application is delivered as a web page.

**Offline-capable web applications.** Using service workers and the Cache API, a web application can cache the model weights and run inference offline, providing an app-like experience without requiring installation.

**Demos and prototyping.** Running a model in the browser provides a zero-installation experience for demonstrations and prototypes.

### 11.3 Limitations

**Model size.** Browsers typically have more restrictive memory limits than native applications. Loading a 4 GB model into WebGPU memory may fail on devices with limited GPU memory or when other browser tabs are consuming resources.

**First-load latency.** Downloading a 1-4 GB model file on the first visit is a significant barrier. Subsequent visits can use the cached model, but the initial download is slow on typical consumer internet connections.

**WebGPU support gaps.** While WebGPU is supported in major browsers as of 2026, some devices (particularly older mobile devices) lack adequate WebGPU support, limiting coverage.

## 12. Practical Applications

### 12.1 Keyboard Prediction and Autocomplete

The oldest and most widespread on-device language model application is keyboard prediction. Modern smartphone keyboards use language models (traditionally small n-gram or LSTM models, increasingly small transformer models) to predict the next word, suggest completions, and correct errors. These models must be extremely fast (predictions must be available within 10-20ms of each keystroke) and extremely small (typically <100 MB, running as a background service).

### 12.2 On-Device Assistants

Apple's Siri, Google Assistant, and Samsung's Bixby increasingly use on-device LLMs for simple queries. The assistant recognizes the user's speech on-device (using an on-device speech recognition model), processes the query with an on-device LLM for simple requests, and escalates to the cloud for complex requests.

### 12.3 Email and Message Processing

Summarizing emails, generating smart replies, and prioritizing notifications are natural on-device tasks: the data is already on the device, the tasks are relatively simple (within a small model's capabilities), and privacy is important (email content should not be sent to a cloud model unnecessarily).

### 12.4 Document Processing

Processing local documents—summarization, question answering, search, and extraction—can be done on-device for short documents. For long documents, the context window limitations of small models and the computational cost of processing many tokens may require cloud escalation or chunked processing strategies.

### 12.5 Creative Tools

On-device models power creative tools: text rewriting (adjusting tone, length, formality), brainstorming assistance, grammar and style checking, and simple content generation. These tasks are well-suited to small models because they are relatively constrained (rewriting existing text is easier than generating novel content from scratch) and the quality bar is lower (the user will review and edit the output).

### 12.6 Accessibility

On-device LLMs enable accessibility features: real-time captioning of audio, description of images for visually impaired users, simplification of complex text for users with cognitive disabilities, and translation for users who do not speak the device's primary language. On-device processing is particularly valuable for accessibility because it works offline and provides consistent, low-latency responses.

## 13. Benchmarking On-Device Performance

### 13.1 Key Metrics

**Tokens per second (decode throughput).** The number of output tokens generated per second. This determines the "typing speed" of the model—how fast the user sees the response appearing.

- Acceptable: >5 tokens/second (readable streaming)
- Good: >15 tokens/second (comfortable conversational pace)
- Excellent: >30 tokens/second (faster than reading speed)

**Time to first token (TTFT).** The latency between submitting the query and receiving the first output token. This depends primarily on the prefill speed (processing the input prompt).

- Acceptable: <2 seconds
- Good: <500ms
- Excellent: <200ms

**Memory footprint.** The total memory used by the model, KV cache, and runtime. Must fit within the device's available memory without causing other applications to be evicted.

**Energy per token.** The energy consumed per generated token, determining the impact on battery life. Measured in millijoules per token.

**Model quality.** The quality of the model's outputs on relevant benchmarks. On-device quality is typically 60-85% of cloud model quality, depending on model size and quantization.

### 13.2 Representative Performance (2026)

Approximate performance for on-device inference on representative hardware:

| Model | Device | Quantization | Decode (tok/s) | TTFT (512 tokens) | Memory |
|-------|--------|-------------|----------------|-------------------|--------|
| Phi-3 Mini 3.8B | iPhone 15 Pro (ANE) | INT4 | 15-20 | ~400ms | ~2.5 GB |
| Llama 3.2 3B | iPhone 15 Pro (ANE) | INT4 | 12-18 | ~450ms | ~2.0 GB |
| Gemma 2 2B | Pixel 8 Pro (GPU) | INT4 | 10-15 | ~500ms | ~1.5 GB |
| Llama 3.2 1B | Snapdragon 8 Gen 3 (NPU) | INT4 | 20-30 | ~200ms | ~0.8 GB |
| Phi-3 Mini 3.8B | M4 MacBook Pro (ANE) | INT4 | 25-35 | ~250ms | ~2.5 GB |
| Llama 3.1 8B | M4 MacBook Pro (GPU) | INT4 | 30-50 | ~400ms | ~5.0 GB |

These numbers are approximate and vary with the inference engine, system load, thermal conditions, and specific model variant.

## 14. Distribution and Updates

### 14.1 Model Distribution

Distributing multi-gigabyte model files to millions of devices presents logistical challenges:

**Bundled with the app.** The model is included in the application binary. Simple but increases app download size dramatically (a 2 GB model makes the app unwieldy for most users).

**Downloaded on first use.** The app downloads the model the first time the AI feature is used. This keeps the initial app download small but requires a multi-gigabyte download before the feature works. The model can be cached locally for subsequent use.

**Delivered as a system component.** Apple and Google distribute on-device models as part of the operating system, downloaded in the background during system updates or initial device setup. This amortizes the download across all apps that use the model.

**Incremental updates.** When the model is updated, sending only the changed weights (as a diff from the previous version) can reduce download size by 50-90%. This is particularly effective for fine-tuned or adapted models where only the adapter weights change.

### 14.2 Model Updates and Versioning

On-device models must be updated periodically (for quality improvements, bug fixes, and capability additions). Update strategies include:

**OS-level updates.** Model updates are distributed with OS updates. This ensures consistency but limits update frequency to the OS release cadence.

**Background updates.** The model is updated in the background, independently of the OS. The new model is downloaded and verified before swapping with the old model, ensuring uninterrupted service.

**A/B testing on device.** Different users receive different model versions, allowing the developer to measure quality differences in production before rolling out a new version to all users.

## 15. Future Directions

### 15.1 Sub-1B Models with Useful Capabilities

As training techniques improve (better data curation, distillation, architecture search), sub-1B models are becoming increasingly capable. Models in the 100-500M parameter range can perform useful tasks (classification, extraction, simple generation) at extremely low resource cost, enabling AI features on even the most constrained devices.

### 15.2 Speculative Decoding on Device

Speculative decoding—using a tiny model (draft model) to generate candidate tokens that are verified by the larger model—can improve decode throughput on device by 1.5-2x. The draft model (50-200M parameters) generates several candidate tokens quickly, and the larger model verifies them in a single forward pass, accepting the correct ones and regenerating from the first incorrect one.

### 15.3 Hardware Evolution

Future mobile SoCs will include more powerful NPUs, more HBM-like high-bandwidth memory, and dedicated transformer acceleration hardware. Apple's roadmap suggests continued NPU improvements in the A-series and M-series chips. Qualcomm and MediaTek are investing in transformer-specific acceleration blocks. These improvements will enable larger and more capable on-device models with each hardware generation.

### 15.4 Multimodal On-Device Models

Current on-device models are primarily text-based, but multimodal capabilities (vision, audio) are arriving. On-device vision-language models can process photos and camera feeds locally, enabling features like visual question answering, image description, and real-time translation of text in images—all without cloud connectivity.

## 16. Conclusion

Edge and mobile LLM deployment has crossed the threshold from research to production. Models in the 1-8B parameter range, quantized to INT4, can run on modern smartphones and laptops at interactive speeds, providing useful capabilities for text processing, conversation, code assistance, and creative tasks. The key enablers are aggressive quantization (reducing model size by 4x with modest quality impact), mobile NPUs (providing efficient matrix multiplication at 35-75 TOPS), and optimized inference engines (llama.cpp, MLC LLM, Core ML, ExecuTorch) that bridge the gap between models and hardware.

The practical limitations remain significant. On-device models are substantially less capable than cloud models—a 3B model cannot match a 70B model on complex reasoning, extensive world knowledge, or nuanced instruction following. Memory constraints limit context windows. Thermal throttling degrades performance under sustained use. Battery drain is a concern for power-intensive applications.

The most effective deployment strategy is hybrid: on-device models for simple, privacy-sensitive, and latency-critical tasks; cloud models for complex tasks that exceed on-device capabilities; and intelligent routing between the two. This mirrors the broader industry trend toward model routing and cascading, with the added dimension of device capability and network conditions as routing inputs.

For practitioners, the key decisions are model selection (choosing the largest model that fits within the device's constraints), quantization level (INT4 as the default, INT8 when quality is critical), inference engine selection (matching the engine to the target hardware), and hybrid architecture design (defining the boundary between on-device and cloud processing). These decisions depend on the specific application, target device population, and quality requirements, but the tools and techniques described in this report provide the foundation for making them systematically.
