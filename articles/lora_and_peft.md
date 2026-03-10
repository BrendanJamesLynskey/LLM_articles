# LoRA and Parameter-Efficient Fine-Tuning

Adapting Large Language Models Without Full-Weight Updates

*March 2026* • Technical Report

Table of Contents

1\. Introduction

2\. The Cost of Full Fine-Tuning

3\. LoRA: Low-Rank Adaptation

4\. QLoRA: Quantized Base with LoRA

5\. Other PEFT Methods

6\. Comparison of Methods

7\. Multi-LoRA Serving

8\. Merging Adapters

9\. Practical Guidelines

10\. Conclusion

## 1. Introduction

As large language models have scaled to tens and hundreds of billions of parameters, the question of how to adapt them to specific tasks or domains has become a central engineering challenge. Full fine-tuning, which updates every parameter in the model, delivers strong results but demands GPU memory proportional to the full model size (for weights, gradients, and optimizer states), making it impractical for most organizations when working with frontier-scale models. Parameter-efficient fine-tuning (PEFT) methods address this by updating only a small fraction of the model's parameters while keeping the rest frozen, dramatically reducing memory requirements and training time while preserving most of the quality gains of full fine-tuning.

This report provides a detailed examination of LoRA and the broader family of PEFT techniques, covering their mechanisms, trade-offs, and practical deployment considerations for ML engineers working with modern LLMs.

## 2. The Cost of Full Fine-Tuning

### 2.1 Memory Breakdown

Full fine-tuning of a model with N parameters in mixed precision (BF16 weights, FP32 optimizer states) requires approximately 16N bytes of GPU memory during training: 2N for the BF16 model weights, 2N for the BF16 gradients, 4N for the FP32 master weights, and 8N for the Adam optimizer's first and second moment estimates. For a 70B-parameter model, this translates to roughly 1.12 TB of memory just for parameters and optimizer state, before accounting for activations and batch data. This exceeds the memory of even an 8-GPU node with 80 GB A100s unless aggressive sharding strategies like ZeRO-3 or FSDP are employed.

### 2.2 The Case for Parameter Efficiency

Research has consistently shown that the weight updates learned during fine-tuning occupy a low-dimensional subspace relative to the full parameter space. This observation, formalized in the intrinsic dimensionality literature, suggests that it should be possible to achieve effective adaptation by constraining updates to a low-rank manifold rather than updating every weight independently. PEFT methods exploit this insight, achieving 90-99% reductions in trainable parameters while retaining the vast majority of task performance.

## 3. LoRA: Low-Rank Adaptation

### 3.1 The Low-Rank Decomposition

LoRA, introduced by Hu et al. (2021), freezes all pre-trained weights and injects trainable low-rank decomposition matrices into selected layers. For a pre-trained weight matrix W of shape d_out x d_in, LoRA parameterizes the weight update as delta_W = B * A, where A is an r x d_in matrix and B is a d_out x r matrix, with r (the rank) being much smaller than both d_in and d_out. During the forward pass, the output becomes h = Wx + (alpha/r) * BAx, where alpha is a scaling hyperparameter. Matrix A is initialized with a random Gaussian distribution and B is initialized to zero, so the adapter begins as an identity modification and gradually learns task-specific adjustments.

### 3.2 Rank Selection

The rank r is the primary capacity knob for LoRA. Common values range from 4 to 256, with r=16 or r=32 being popular defaults for instruction tuning. Lower ranks are more parameter-efficient and act as stronger regularizers, which can be beneficial for small datasets or narrow tasks. Higher ranks provide more capacity and are preferred when the target task diverges significantly from the pre-training distribution. Empirically, the relationship between rank and performance shows diminishing returns: doubling the rank from 16 to 32 often yields a smaller improvement than doubling from 4 to 8.

### 3.3 Alpha Scaling

The scaling factor alpha controls the magnitude of the LoRA update relative to the pre-trained weights. The effective scaling is alpha/r, so increasing alpha amplifies the adapter's contribution. A common heuristic is to set alpha equal to twice the rank (e.g., alpha=32 for r=16), though the optimal value depends on the task and learning rate. When alpha is set equal to r, the scaling factor is 1, meaning the adapter output is added to the pre-trained output without rescaling.

### 3.4 Which Layers to Adapt

LoRA can be applied to any linear layer in the transformer, but the choice of target layers affects both parameter count and performance. In practice, adapting the query and value projection matrices (W_q, W_v) in the attention mechanism is the most common configuration and provides strong results. Extending LoRA to all four attention projections (Q, K, V, O) and the MLP layers (gate, up, down projections) increases trainable parameters but can yield meaningful improvements, particularly for complex tasks. The MLP layers contain the majority of the model's parameters, so including them substantially increases the adapter size but also its representational capacity.

## 4. QLoRA: Quantized Base with LoRA

### 4.1 Mechanism

QLoRA, proposed by Dettmers et al. (2023), combines 4-bit quantization of the base model with LoRA adapters trained in BF16. The base model weights are stored in a custom NormalFloat4 (NF4) format, which is information-theoretically optimal for normally distributed weights. During the forward pass, quantized weights are dequantized to BF16 on the fly for matrix multiplication. Gradients flow through the dequantization step to update only the LoRA parameters. QLoRA further introduces double quantization, where the quantization constants themselves are quantized to 8-bit, and paged optimizers that offload optimizer state to CPU memory when GPU memory is exhausted.

### 4.2 Impact

QLoRA enables fine-tuning a 65B-parameter model on a single 48 GB GPU, a task that would otherwise require a multi-node cluster with full fine-tuning. The quality penalty from 4-bit quantization of the base model is surprisingly small: QLoRA fine-tuned models typically match or come within 1-2% of full BF16 fine-tuning on downstream benchmarks. This technique was instrumental in democratizing LLM fine-tuning, making it accessible to researchers and practitioners without access to large GPU clusters.

## 5. Other PEFT Methods

### 5.1 Prefix Tuning and Prompt Tuning

Prefix tuning prepends a set of trainable continuous vectors (soft tokens) to the key and value sequences at every transformer layer, effectively steering the model's attention without modifying any weights. Prompt tuning is a simplified variant that prepends learnable embeddings only at the input layer. Both methods are extremely parameter-efficient (often fewer than 0.1% of model parameters) but tend to underperform LoRA on complex tasks, particularly as model size increases.

### 5.2 Adapters

Adapter modules insert small bottleneck layers (a down-projection, nonlinearity, and up-projection) between the existing layers of a transformer. The original adapter architecture by Houlsby et al. (2019) places two adapter modules per transformer block, one after the attention layer and one after the feed-forward layer. While adapters can match LoRA in quality, they add sequential computation to the forward pass, increasing inference latency unless the adapter is merged into the base weights, which is not always straightforward.

### 5.3 IA3: Infused Adapter by Inhibiting and Amplifying Inner Activations

IA3 learns element-wise rescaling vectors for the keys, values, and feed-forward intermediate activations. With only three learned vectors per transformer layer, IA3 introduces far fewer parameters than LoRA (typically 10-100x fewer) and adds negligible inference overhead. However, its limited capacity makes it best suited for tasks that are closely aligned with the base model's existing capabilities.

### 5.4 DoRA: Weight-Decomposed Low-Rank Adaptation

DoRA decomposes the pre-trained weight matrix into magnitude and direction components, then applies LoRA only to the directional component while learning a separate magnitude vector. This decomposition is motivated by the observation that full fine-tuning tends to make large directional changes with small magnitude adjustments, a pattern that standard LoRA struggles to replicate. DoRA consistently outperforms LoRA by 1-3% on instruction-following benchmarks at equivalent parameter counts, with minimal additional overhead.

## 6. Comparison of Methods

The choice of PEFT method involves trade-offs across several dimensions. LoRA offers the best balance of quality, parameter efficiency, and ecosystem support, making it the default choice for most fine-tuning scenarios. QLoRA extends this accessibility to memory-constrained environments. DoRA provides a quality premium over LoRA with modest additional complexity. Prefix tuning and IA3 are attractive when trainable parameter count must be minimized to an absolute minimum, such as when training thousands of task-specific adapters. Adapters remain relevant in research contexts but have largely been superseded by LoRA in production deployments due to their inference latency overhead.

## 7. Multi-LoRA Serving

### 7.1 The Multi-Tenant Challenge

In production environments, a single base model often serves many different use cases, each with its own LoRA adapter. Naively loading separate model instances per adapter wastes memory and precludes batching across users. Multi-LoRA serving systems address this by sharing the base model weights across all requests and dynamically applying the appropriate LoRA adapter per request within a single batch.

### 7.2 LoRAX and S-LoRA

LoRAX (LoRA eXchange) and S-LoRA are serving frameworks that implement efficient multi-adapter inference. S-LoRA introduces a unified paging mechanism that stores all adapters in a shared memory pool, loading adapter weights into GPU memory on demand. It uses custom CUDA kernels that apply different LoRA adapters to different requests within a single batched matrix multiplication, avoiding the overhead of separate adapter application. LoRAX builds on similar principles with a focus on production readiness, supporting dynamic adapter loading, adapter caching with LRU eviction, and seamless integration with the HuggingFace ecosystem. Both systems demonstrate that hundreds of LoRA adapters can be served concurrently with near-zero overhead compared to serving the base model alone.

## 8. Merging Adapters

### 8.1 Weight Merging

Because LoRA's contribution is a simple additive matrix (delta_W = BA), the adapter can be merged directly into the base model weights: W_new = W + (alpha/r) * BA. This produces a single set of weights with zero inference overhead, eliminating the need for adapter-aware serving infrastructure. Merging is irreversible in the sense that the merged model can no longer separate the adapter's contribution, but this is acceptable for single-purpose deployments.

### 8.2 Multi-Adapter Merging

When multiple LoRA adapters have been trained for complementary capabilities (e.g., one for coding, one for mathematical reasoning, one for instruction following), they can be merged together through linear combination: W_new = W + sum(lambda_i * delta_W_i). The mixing coefficients lambda_i can be tuned on a validation set. More sophisticated methods like TIES-Merging and DARE prune conflicting parameter updates before merging, reducing interference between adapters and improving the quality of the merged model. Task arithmetic approaches treat each adapter's delta as a task vector, enabling operations like negation (to remove a capability) and addition (to combine capabilities).

## 9. Practical Guidelines

### 9.1 Getting Started

For most fine-tuning scenarios, begin with LoRA applied to all attention and MLP projections at rank 16 with alpha 32. Use the HuggingFace PEFT library or similar framework that handles the boilerplate of freezing base weights, injecting adapters, and saving only the adapter parameters. Train with a learning rate in the range of 1e-4 to 3e-4, slightly higher than what you would use for full fine-tuning, since the smaller parameter count benefits from more aggressive updates.

### 9.2 Memory-Constrained Environments

When GPU memory is the binding constraint, use QLoRA with 4-bit NF4 quantization. The quality gap relative to BF16 LoRA is small and typically acceptable. Enable gradient checkpointing and use paged optimizers to further reduce peak memory usage. For a 7B model, QLoRA fine-tuning requires as little as 6-8 GB of GPU memory, making it feasible on consumer hardware.

### 9.3 When to Increase Rank

If validation loss plateaus early or the task involves substantial domain shift (e.g., adapting a general model to a specialized scientific domain), increase the rank to 64 or 128. Monitor the ratio of adapter parameters to training examples: if the adapter has more trainable parameters than training tokens divided by 1000, overfitting risk is high and a lower rank or stronger regularization may be warranted.

## 10. Conclusion

LoRA and the broader PEFT family have fundamentally changed the economics of LLM adaptation. By exploiting the low intrinsic dimensionality of fine-tuning updates, these methods reduce memory requirements by an order of magnitude while preserving the vast majority of task performance. QLoRA further extends accessibility to consumer hardware, while multi-LoRA serving systems enable efficient multi-tenant deployments. As models continue to grow, parameter-efficient methods will remain essential tools for any ML engineer working with large language models, bridging the gap between the scale of modern foundation models and the practical constraints of real-world training and deployment infrastructure.
