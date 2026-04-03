# Vision-Language Models: Unifying Visual and Textual Understanding in Large Models

*April 2026 · Technical Report*

## 1. Introduction

For decades, computer vision and natural language processing developed as separate disciplines with distinct architectures, training paradigms, and evaluation benchmarks. Convolutional neural networks dominated image classification and object detection, while recurrent networks and later transformers drove advances in text understanding and generation. The two fields occasionally intersected — image captioning, visual question answering — but these intersections relied on bolted-together pipelines where a vision encoder extracted features and a language model decoded them into text, with limited cross-modal reasoning.

The emergence of vision-language models (VLMs) has fundamentally changed this picture. Modern VLMs integrate visual and textual understanding into unified architectures capable of perceiving images, reading documents, interpreting charts, reasoning about spatial relationships, and generating coherent natural language responses grounded in visual evidence. The trajectory from CLIP's contrastive pre-training in 2021 through GPT-4V's multimodal capabilities in 2023 to the natively multimodal architectures of 2025–2026 represents one of the most consequential convergences in AI research.

This report provides a comprehensive technical examination of vision-language models: the architectures that encode and integrate visual information, the training strategies that align vision and language representations, the major commercial and open-source models, and the practical considerations — from resolution handling to inference cost to persistent limitations — that shape real-world deployment.

## 2. Evolution from Separate Models to Unified VLMs

### 2.1 The Pipeline Era

Early multimodal systems followed a strict pipeline architecture. An image was processed by a convolutional neural network (typically a ResNet or Inception variant) pre-trained on ImageNet, producing a fixed-dimensional feature vector. This vector was then fed into a language model — often an LSTM or GRU — that generated a text sequence conditioned on the visual features. The two components were trained separately (or fine-tuned end-to-end on small datasets), and the visual representation was fundamentally constrained by the classification-oriented pre-training of the CNN.

This pipeline approach had clear limitations. The visual encoder produced a single global feature vector that discarded spatial information. The language model had no mechanism to attend to different image regions for different parts of the generated text. And the two components operated in fundamentally different representation spaces, connected only by a thin linear projection layer.

### 2.2 Attention-Based Cross-Modal Fusion

The introduction of attention mechanisms to vision-language models marked the first major architectural advance. Models like Bottom-Up and Top-Down Attention (Anderson et al., 2018) used object detection to extract region-level features from images, then allowed the language model to attend selectively to different regions when generating each word. This enabled the model to "look at" the relevant part of an image when describing it — attending to a dog when generating the word "dog" and to a frisbee when generating "frisbee."

Cross-attention mechanisms extended this further. In models like VisualBERT, VilBERT, and LXMERT, transformer layers processed both visual and textual tokens, with cross-attention layers allowing each modality to attend to the other. These models could be pre-trained on image-text pairs with masked language modeling and image-text matching objectives, learning rich cross-modal representations.

### 2.3 The CLIP Revolution

OpenAI's CLIP (Contrastive Language-Image Pre-training), released in January 2021, represented a paradigm shift. Rather than training a vision model on a fixed set of ImageNet categories and a language model on text corpora, CLIP trained a vision encoder and a text encoder jointly on 400 million image-text pairs scraped from the internet, using a contrastive learning objective that pulled matching image-text pairs together in a shared embedding space while pushing non-matching pairs apart.

The contrastive pre-training objective was simple but powerful. Given a batch of N image-text pairs, CLIP computed cosine similarity between all N² possible pairings and trained the encoders to maximize similarity for the N correct pairs while minimizing similarity for the N²-N incorrect pairs. This produced visual representations that were inherently aligned with natural language — an image of a cat and the text "a photograph of a cat" would have high cosine similarity in the shared embedding space.

CLIP's impact was profound for three reasons. First, it demonstrated that web-scale image-text data, despite being noisy, could produce visual representations that matched or exceeded those trained on carefully curated datasets like ImageNet. Second, it enabled zero-shot image classification — classifying images into categories never seen during training by comparing image embeddings to text embeddings of category descriptions. Third, and most importantly for the VLM trajectory, it provided a pre-trained visual encoder whose representations were already aligned with language, making it a natural starting point for building models that could reason jointly about images and text.

### 2.4 SigLIP and Improved Contrastive Encoders

SigLIP (Sigmoid Loss for Language-Image Pre-training), introduced by Google in 2023, improved upon CLIP's contrastive training in a key way. While CLIP used a softmax-based contrastive loss that required computing similarities across all pairs in a batch (making it sensitive to batch size and requiring large batches for good performance), SigLIP replaced this with a sigmoid loss that operated independently on each image-text pair. Each pair was treated as a binary classification problem: is this image-text pair matched or not?

This change had practical benefits. SigLIP was more memory-efficient during training (no need for all-gather operations across devices to compute the full similarity matrix), scaled better to very large batch sizes, and achieved comparable or superior performance to CLIP with simpler distributed training. SigLIP encoders became the preferred visual backbone for many subsequent VLMs, including PaLI-X, PaLI-3, and several LLaVA variants.

### 2.5 From Dual Encoders to Generative VLMs

CLIP and SigLIP are dual-encoder models — they produce separate embeddings for images and text that can be compared but cannot generate text. The next major evolution was connecting these pre-trained visual encoders to large language models capable of open-ended text generation, creating models that could not just classify or retrieve but converse about images: answer questions, describe scenes, follow visual instructions, and reason through multi-step problems grounded in visual evidence.

This transition from discriminative to generative VLMs, which accelerated through 2023–2025, defines the current landscape.

## 3. Image Encoding Architectures

### 3.1 Vision Transformers (ViT)

The Vision Transformer (ViT), introduced by Dosovitskiy et al. (2020), adapted the transformer architecture from NLP to computer vision. An image is divided into fixed-size patches (typically 14×14 or 16×16 pixels), each patch is linearly projected into an embedding vector, positional embeddings are added, and the resulting sequence of patch tokens is processed by a standard transformer encoder.

For a 224×224 image with 14×14 patches, this produces a sequence of 256 patch tokens (16 patches per dimension, 16×16 = 256 tokens). A special [CLS] token is prepended to the sequence, and after transformer processing, the [CLS] token's representation serves as the global image embedding. For VLM applications, however, the full sequence of patch token representations (not just the [CLS] token) is typically used, as these retain spatial information about different image regions.

ViT comes in several sizes commonly used in VLMs:

| Model | Layers | Hidden Dim | Heads | Params | Patch Size |
|---|---|---|---|---|---|
| ViT-B/16 | 12 | 768 | 12 | 86M | 16×16 |
| ViT-L/14 | 24 | 1024 | 16 | 304M | 14×14 |
| ViT-H/14 | 32 | 1280 | 16 | 632M | 14×14 |
| ViT-G/14 | 40 | 1408 | 16 | 1.0B | 14×14 |
| ViT-22B | 48 | 6144 | 48 | 22B | 14×14 |

The CLIP ViT-L/14 variant, trained with CLIP's contrastive objective on 400M image-text pairs, became the de facto standard visual encoder for early VLMs. Later models adopted larger variants (ViT-G, ViT-22B) or SigLIP-trained encoders for improved visual understanding.

### 3.2 Feature Extraction Layers

An important but often overlooked detail is which layer's output is used as the visual representation. The final layer of a ViT encoder, particularly one trained with CLIP's contrastive objective, tends to produce representations optimized for global image-text similarity — useful for retrieval and classification but potentially losing fine-grained spatial details. Several VLMs have found that extracting features from intermediate layers (e.g., the second-to-last layer or even earlier layers) produces better results for tasks requiring detailed visual understanding, such as OCR or visual grounding.

LLaVA-1.5, for example, uses the output of the second-to-last layer of CLIP ViT-L/14, finding that this layer retains more spatial detail than the final layer. Some more recent architectures concatenate features from multiple layers or use learnable weighted combinations of different layers' outputs to capture both global semantic information and fine-grained spatial detail.

### 3.3 Convolutional Encoders and Hybrid Approaches

While ViT dominates the VLM landscape, some models use convolutional encoders or hybrid architectures. ConvNeXt, a modernized convolutional architecture that incorporates design elements from transformers (large kernel sizes, layer normalization, GELU activations), has been used as the visual encoder in some VLMs, particularly those targeting efficient deployment. Hybrid approaches that use convolutional layers for initial patch embedding followed by transformer layers for global reasoning combine the inductive biases of convolutions (translation invariance, local feature extraction) with the flexibility of attention.

### 3.4 Emerging Encoder Architectures

By 2025–2026, several newer visual encoding approaches have gained traction. DINOv2, a self-supervised vision model trained without any text supervision, produces visual features that excel at dense prediction tasks (segmentation, depth estimation) and have been used alongside CLIP/SigLIP encoders in dual-encoder VLM configurations. EVA-CLIP and InternViT push encoder scale to 4–6 billion parameters, demonstrating continued gains from scaling the visual backbone. SAM (Segment Anything Model) encoders have been incorporated into VLMs that require pixel-level understanding, such as visual grounding and referring expression segmentation.

## 4. Visual Token Integration Strategies

The central architectural question in VLM design is how to connect the visual encoder's output to the language model. Three main strategies have emerged, each with distinct tradeoffs.

### 4.1 Visual Tokens in the Sequence (Early Fusion)

The most straightforward approach treats visual features as additional tokens in the language model's input sequence. The visual encoder processes an image and produces a sequence of patch-level feature vectors. These are projected to the language model's embedding dimension (via a linear layer or small MLP) and prepended or interleaved with the text token embeddings. The language model then processes the combined sequence with its standard self-attention mechanism, allowing visual and textual tokens to attend to each other freely.

This approach is used by LLaVA, Qwen-VL, InternVL, and many other open-source VLMs. Its advantages include simplicity (no architectural modifications to the language model), the ability to leverage the language model's pre-trained attention patterns, and the natural handling of interleaved image-text inputs (important for multi-image and document understanding tasks). The primary disadvantage is cost: each image contributes hundreds or thousands of tokens to the sequence, directly increasing the computational cost of the language model's self-attention (which scales quadratically with sequence length).

For a 336×336 image with 14×14 patches, the visual encoder produces 576 tokens (24×24 patches). For a 672×672 image, this increases to 2,304 tokens. High-resolution document understanding might require 4,096 or more visual tokens per image. These visual tokens consume the same compute as text tokens in the language model's attention layers, making high-resolution visual understanding expensive.

### 4.2 Cross-Attention (Late Fusion)

An alternative approach adds dedicated cross-attention layers to the language model, where text tokens attend to visual features through a separate attention mechanism rather than sharing the main self-attention sequence. The visual features serve as keys and values in these cross-attention layers, while the text tokens provide the queries.

Flamingo (DeepMind, 2022) pioneered this approach, inserting gated cross-attention layers between the frozen layers of a pre-trained language model. The visual features are processed by a Perceiver Resampler that compresses them into a fixed number of visual tokens (typically 64), which then serve as the key-value pairs in the cross-attention layers. This approach has the advantage of keeping visual and textual processing partially separated, allowing the language model's self-attention to operate on text tokens alone (maintaining text generation quality) while still enabling deep cross-modal interaction through the cross-attention layers.

The cross-attention approach is also more flexible regarding the number of visual tokens. Because the visual features are only accessed through cross-attention (not occupying positions in the main sequence), the computational cost of visual processing is somewhat decoupled from the sequence length of the text. However, the cross-attention layers add parameters and complexity, and they require architectural modifications to the language model that complicate the use of pre-trained checkpoints.

### 4.3 Adapter-Based Approaches (Projection Modules)

A middle ground uses adapter modules — small neural networks that transform visual encoder outputs before feeding them into the language model. These adapters range from simple linear projections (as in the original LLaVA) to more sophisticated architectures:

**MLP Projectors**: LLaVA-1.5 replaced the original single linear layer with a two-layer MLP (with GELU activation), finding significant improvements in visual understanding. This simple change suggests that the mapping between visual and language representation spaces requires at least some nonlinear transformation.

**Perceiver Resampler**: Used in Flamingo and several successors, this architecture uses a set of learnable query tokens that attend to the full set of visual features through cross-attention, producing a fixed-size set of visual tokens regardless of input resolution. This compresses the visual information and provides a consistent token count to the language model.

**Q-Former**: Introduced in BLIP-2, the Q-Former (Querying Transformer) is a lightweight transformer that uses a set of learnable query tokens to extract the most relevant visual features through cross-attention with the visual encoder's output. It is pre-trained with multiple objectives (image-text contrastive learning, image-grounded text generation, and image-text matching) to learn an effective visual-to-language mapping.

**C-Abstractor and Spatial Adapters**: More recent adapters preserve spatial structure explicitly. The C-Abstractor (used in Honeybee and some LLaVA variants) applies convolutional operations to the visual feature grid before projection, maintaining the 2D spatial arrangement that is lost when features are flattened into a 1D sequence.

### 4.4 Natively Multimodal Architectures

The most recent architectural trend is natively multimodal models that are trained from scratch (or from early in pre-training) to process multiple modalities without a separate visual encoder. Google's Gemini models exemplify this approach: rather than using a pre-trained CLIP or SigLIP encoder connected to a pre-trained language model, Gemini processes images, text, audio, and video through a single transformer architecture that was multimodal from the start of training.

The advantage of native multimodality is deeper integration between modalities. In a two-stage VLM (frozen visual encoder + language model), the visual representations are fixed by the encoder's pre-training and cannot be refined based on the language model's needs. In a natively multimodal model, the visual and textual representations co-evolve during training, potentially enabling richer cross-modal reasoning.

The disadvantage is the training cost. Pre-training a natively multimodal model from scratch requires enormous compute and carefully curated multimodal training data. Most open-source VLMs therefore continue to use the two-stage approach (pre-trained visual encoder + pre-trained language model + alignment training), which allows leveraging the best available components from each modality.

## 5. Contrastive Pre-training: CLIP and Beyond

### 5.1 CLIP Architecture and Training

CLIP consists of two encoders: a vision encoder (either a ResNet or ViT variant) and a text encoder (a standard transformer). Both encoders project their respective inputs into a shared 512-dimensional (or 768-dimensional, depending on the variant) embedding space. Training uses the InfoNCE contrastive loss on batches of image-text pairs scraped from the internet.

The training data — WebImageText (WIT), a dataset of 400 million image-text pairs collected from the internet — was a key ingredient. The diversity and scale of this data, despite its noise, produced representations that generalized remarkably well across visual domains. CLIP could classify images from specialized domains (satellite imagery, medical scans, fine-grained species identification) with reasonable accuracy despite never being trained on labeled examples from those domains.

### 5.2 Limitations of CLIP Representations

Despite its success, CLIP has important limitations as a visual encoder for VLMs. CLIP was trained to align whole images with whole captions, optimizing for global image-text correspondence. This means CLIP representations are strong for capturing the overall semantic content of an image but can be weak for:

- **Fine-grained spatial reasoning**: Distinguishing "the cat is on the left of the dog" from "the dog is on the left of the cat" requires spatial understanding that CLIP's training signal does not directly encourage.
- **Counting**: Determining how many objects of a particular type are present in an image is difficult with CLIP features.
- **Text reading (OCR)**: CLIP was not specifically trained to read text in images, and its patch-based tokenization at typical resolutions often makes text illegible.
- **Detailed attribute understanding**: Fine distinctions (specific colors, textures, materials) can be lost in CLIP's global features.

These limitations have driven the development of improved visual encoders (SigLIP, InternViT, DFN-CLIP) and the use of higher resolutions and multi-crop strategies in VLMs.

### 5.3 Open CLIP and Reproducible Training

The OpenCLIP project, initiated by LAION, reproduced CLIP training with open data (LAION-400M, LAION-2B, LAION-5B) and open code, enabling the research community to study and improve contrastive visual pre-training. OpenCLIP checkpoints trained on LAION-2B achieved performance comparable to or exceeding OpenAI's original CLIP on many benchmarks, and several open-source VLMs use OpenCLIP or SigLIP encoders as their visual backbone.

DataComp, a benchmark for multimodal dataset design, emerged from this work and demonstrated that careful data curation (filtering, deduplication, quality scoring) could significantly improve CLIP training efficiency, achieving the same performance with fewer image-text pairs when the data was better curated.

## 6. LLaVA: Visual Instruction Tuning

### 6.1 LLaVA Architecture

LLaVA (Large Language and Vision Assistant), introduced by Liu et al. in April 2023, demonstrated that connecting a pre-trained CLIP visual encoder to a pre-trained large language model with a simple projection layer, then fine-tuning on visual instruction-following data, could produce a surprisingly capable VLM with relatively modest training cost.

The original LLaVA architecture was straightforward:

1. **Visual Encoder**: CLIP ViT-L/14 at 224×224 resolution, producing 256 patch tokens of dimension 1024.
2. **Projection Layer**: A single linear layer mapping from the visual encoder's dimension (1024) to the language model's dimension (4096 for LLaMA-based models).
3. **Language Model**: Vicuna (a fine-tuned LLaMA), which processes the projected visual tokens concatenated with the text tokens.

The visual tokens are inserted into the input sequence at the position of a special `<image>` token in the text. The language model then generates a response conditioned on both the visual tokens and the text instruction.

### 6.2 Two-Stage Training

LLaVA's training proceeds in two stages:

**Stage 1: Feature Alignment Pre-training**. The visual encoder and language model are both frozen. Only the projection layer is trained, using 558K image-caption pairs filtered from CC3M. This stage teaches the projection layer to map visual features into the language model's embedding space. Training takes approximately 4 hours on 8 A100 GPUs.

**Stage 2: Visual Instruction Tuning**. The projection layer and language model are fine-tuned end-to-end (the visual encoder remains frozen) on 158K visual instruction-following examples. These examples are generated using GPT-4 — given an image's caption and bounding box annotations, GPT-4 generates diverse question-answer pairs about the image, including detailed descriptions, multi-turn conversations, and complex reasoning tasks. Training takes approximately 10 hours on 8 A100 GPUs.

The total training cost (excluding the pre-trained components) was remarkably low — under $300 in compute — yet the resulting model demonstrated strong visual conversation abilities.

### 6.3 LLaVA-1.5 Improvements

LLaVA-1.5, released in October 2023, introduced several key improvements:

- **MLP Projector**: Replaced the single linear layer with a two-layer MLP (linear → GELU → linear), which improved visual understanding significantly across benchmarks.
- **Higher Resolution**: Increased input resolution from 224×224 to 336×336, producing 576 visual tokens instead of 256. This improved OCR and fine-grained detail recognition.
- **Academic Task Data**: Added training data from academic VQA datasets (VQAv2, GQA, OCR-VQA, TextVQA, VisualGenome) to improve performance on established benchmarks.
- **Stronger Language Model**: Upgraded from Vicuna-13B to variants using LLaMA-2 and eventually other base models.

### 6.4 LLaVA-NeXT and Beyond

LLaVA-NeXT (January 2024) introduced dynamic high-resolution processing. Instead of resizing all images to a fixed resolution, LLaVA-NeXT divides high-resolution images into multiple crops, processes each crop independently through the visual encoder, and concatenates the resulting visual tokens. This allows the model to process images at their native resolution (up to 672×672 or higher) without losing detail, at the cost of a proportionally higher number of visual tokens.

Subsequent LLaVA variants (LLaVA-OneVision, LLaVA-Video) extended the architecture to support video understanding (by processing sampled frames), multi-image inputs, and improved instruction-following. The LLaVA family demonstrated that the basic recipe of "frozen CLIP encoder + projection + LLM fine-tuning" was remarkably flexible and could be incrementally improved through better data, higher resolution, and stronger base models.

## 7. Major Commercial VLMs

### 7.1 GPT-4V and GPT-4o

OpenAI's GPT-4V (GPT-4 with Vision), launched in September 2023, was the first widely deployed commercial VLM. It demonstrated strong performance across a broad range of visual tasks: image description, visual question answering, chart and graph interpretation, OCR, code generation from screenshots, and multi-step visual reasoning.

GPT-4o (May 2024) unified text, vision, and audio into a single model trained end-to-end across modalities. Unlike GPT-4V, which processed images through a separate pipeline before feeding features to the language model, GPT-4o was described as natively multimodal — all modalities are processed by a single neural network. This architecture enables faster visual processing (reducing latency compared to GPT-4V) and potentially deeper cross-modal reasoning.

The specific architecture of GPT-4V and GPT-4o has not been publicly disclosed. Based on API behavior and external analysis, the models appear to process images at multiple resolutions (using a tiling strategy for high-resolution images), handle up to approximately 2048×2048 pixels (through tiling), and generate visual tokens that consume the context window alongside text tokens. High-resolution images in GPT-4o consume roughly 765 tokens for a 512×512 image and up to several thousand tokens for larger images, with the exact count depending on the tiling strategy.

### 7.2 Claude Vision

Anthropic's Claude models have supported vision since Claude 3 (March 2024). The Claude 3 family (Haiku, Sonnet, Opus) and subsequent Claude 3.5 and Claude 4 models process images as part of the input context, supporting tasks including document analysis, chart interpretation, image description, visual reasoning, and code generation from screenshots.

Claude's vision capabilities are notable for strong performance on document understanding tasks (reading dense text, interpreting tables, analyzing forms) and careful handling of uncertainty — the models tend to acknowledge when visual details are unclear rather than confabulating, though visual hallucination remains a general challenge. Claude supports images up to a configurable maximum resolution, with automatic resizing applied to images exceeding the limit. Visual tokens consume context window space, with the exact token count depending on image resolution.

### 7.3 Gemini: Native Multimodality

Google's Gemini models (December 2023 onward) represent the natively multimodal approach. Rather than connecting separate pre-trained vision and language models, Gemini was designed from the ground up to process text, images, audio, and video in a unified architecture. The Gemini family — Ultra, Pro, Flash, and Nano — spans a wide range of sizes and capability levels.

Gemini's native multimodality manifests in several practical advantages. The models handle interleaved image-text inputs naturally (e.g., "compare these three images" with images interspersed in the text). They support very long visual contexts — Gemini 1.5 Pro processes up to 1 million tokens, enabling analysis of hours of video or hundreds of images in a single context. The models also demonstrate strong performance on multilingual visual tasks, reflecting the multilingual nature of their training data.

Gemini 2.0 and subsequent models (through 2025–2026) have extended these capabilities with improved spatial understanding, video reasoning, and integration with tools and code execution for complex visual analysis tasks.

### 7.4 Other Commercial VLMs

Several other commercial VLMs have achieved notable capabilities:

- **Grok (xAI)**: Grok models include vision capabilities with particular strength in real-time visual analysis and integration with the X (formerly Twitter) platform.
- **Mistral**: Pixtral, Mistral's VLM, uses a novel visual encoder architecture and demonstrates competitive performance particularly on document understanding tasks.
- **Amazon Nova**: Amazon's Nova models include vision capabilities integrated into the AWS Bedrock platform.
- **Reka**: Reka's models have demonstrated strong multimodal performance, particularly on video understanding tasks.

## 8. Open-Source VLMs

### 8.1 InternVL Series

InternVL (Shanghai AI Lab) is one of the strongest open-source VLM families. InternVL 2 and 2.5 use InternViT-6B, a 6-billion-parameter vision encoder trained with a combination of contrastive and generative objectives, connected to InternLM2 language models of various sizes (2B, 8B, 26B, 76B parameters).

Key innovations in InternVL include:

- **Dynamic Resolution**: Images are divided into tiles of 448×448 pixels, with the number of tiles varying based on image resolution and aspect ratio (up to 12 tiles for very high-resolution images). This allows efficient processing of images with diverse aspect ratios without excessive padding.
- **Pixel Shuffle**: A downsampling operation applied after the visual encoder that reduces the number of visual tokens by a factor of 4 (merging 2×2 groups of adjacent tokens), making the model more efficient for high-resolution inputs.
- **Progressive Training**: The model is trained in stages — first aligning the visual encoder with the language model on image-caption data, then instruction tuning on diverse visual tasks, then further tuning on domain-specific data.

InternVL 2.5 Pro achieves performance competitive with GPT-4o on many visual understanding benchmarks while being fully open-source (weights, training code, and data recipes).

### 8.2 Qwen-VL Series

Alibaba's Qwen-VL series integrates vision capabilities into the Qwen language model family. Qwen2-VL (2024) introduced several technical innovations:

- **Naive Dynamic Resolution**: Rather than resizing images to a fixed resolution or using predetermined tile sizes, Qwen2-VL processes images at their native resolution by converting them to a variable number of visual tokens proportional to the image's pixel count. A minimum and maximum token count is enforced to bound computational cost.
- **Multimodal Rotary Position Embedding (M-RoPE)**: An extension of RoPE that encodes both temporal and spatial position information, enabling the model to understand the 2D spatial structure of images and the temporal ordering of video frames.
- **Video Support**: Qwen2-VL natively processes video by encoding frames at a configurable frame rate and interleaving them in the input sequence with temporal position encodings.

### 8.3 LLaVA Ecosystem

Beyond the original LLaVA models discussed in Section 6, the broader LLaVA ecosystem includes numerous community extensions:

- **LLaVA-OneVision**: A unified model supporting single-image, multi-image, and video understanding with a single set of weights.
- **LLaVA-Med**: A medical domain VLM fine-tuned on biomedical image-text data for clinical applications.
- **TinyLLaVA**: Compact LLaVA variants using small language models (1B–3B parameters) for edge deployment.
- **LLaVA-NeXT-Interleave**: Extended to handle arbitrarily interleaved image-text inputs.

### 8.4 Other Notable Open VLMs

**Idefics2 and Idefics3** (Hugging Face): Based on the Flamingo architecture with improvements, Idefics models demonstrate competitive multimodal performance while being fully open.

**Phi-3-Vision and Phi-3.5-Vision** (Microsoft): Small but capable VLMs (4B parameters) that achieve strong performance relative to their size, demonstrating that visual understanding does not necessarily require very large models.

**Molmo** (Allen AI): An open VLM family that includes novel pointing and grounding capabilities, allowing the model to indicate specific locations in an image as part of its response.

**Cambrian-1**: A research VLM that systematically studied the impact of different visual encoders, finding that combining multiple visual encoders (e.g., CLIP + DINOv2 + SAM) through a Spatial Vision Aggregator produced the best visual understanding across diverse tasks.

## 9. Resolution Handling

### 9.1 The Resolution-Cost Tradeoff

Resolution is one of the most important practical considerations in VLM design and deployment. Higher resolution preserves more visual detail — critical for reading small text, identifying fine-grained features, and understanding dense documents — but generates more visual tokens, increasing both the computational cost of inference and the consumption of the context window.

At a typical ViT patch size of 14×14 pixels:

| Input Resolution | Patches | Visual Tokens |
|---|---|---|
| 224 × 224 | 16 × 16 | 256 |
| 336 × 336 | 24 × 24 | 576 |
| 448 × 448 | 32 × 32 | 1,024 |
| 672 × 672 | 48 × 48 | 2,304 |
| 1344 × 1344 | 96 × 96 | 9,216 |

At GPT-4o-class pricing (roughly $5 per million input tokens), processing a single high-resolution image at 1344×1344 costs roughly 4.5 cents — comparable to processing several pages of text.

### 9.2 Dynamic Tiling

Dynamic tiling is the dominant approach to resolution handling in modern VLMs. Rather than resizing all images to a fixed resolution, the model:

1. Determines the optimal number of tiles based on the image's resolution and aspect ratio.
2. Divides the image into a grid of tiles (e.g., 2×2, 3×2, 1×4), each at the visual encoder's native resolution (e.g., 336×336 or 448×448).
3. Processes each tile independently through the visual encoder.
4. Concatenates the visual tokens from all tiles, often with a downsampled "thumbnail" of the full image to provide global context.
5. Feeds the combined visual tokens into the language model.

This approach allows the model to handle images of any resolution and aspect ratio while keeping each tile's processing efficient. A 1920×1080 image might be divided into a 4×2 grid of 480×540 tiles (each resized to 448×448 for the encoder), producing 8 × 1024 = 8,192 visual tokens plus additional tokens from the thumbnail.

### 9.3 Token Compression

Given the high cost of visual tokens, several techniques have been developed to compress them:

- **Pixel Shuffle / Token Merging**: Adjacent visual tokens are merged (averaged or concatenated and projected) to reduce the total count. InternVL's pixel shuffle reduces tokens by 4×.
- **Perceiver Resampling**: A fixed number of learnable queries attend to the full visual token sequence, producing a compressed representation. Flamingo uses 64 queries regardless of input resolution.
- **Adaptive Token Selection**: Some models learn to select or weight visual tokens based on their relevance to the text query, discarding uninformative background tokens. This is analogous to FastV and similar methods that prune visual tokens after the first few layers.
- **Progressive Resolution**: Process the image at low resolution first, then selectively process high-resolution crops of regions the model identifies as relevant.

## 10. OCR Capabilities

### 10.1 The Evolution of VLM OCR

Early VLMs had limited OCR capabilities — the standard 224×224 input resolution made most text illegible. As resolution increased and training data was enriched with document understanding examples, VLM OCR capabilities improved dramatically. By 2025–2026, frontier VLMs can read text from documents, receipts, signs, screenshots, and handwritten notes with accuracy approaching or matching dedicated OCR systems.

### 10.2 Document Understanding

Document understanding extends beyond simple OCR to structural comprehension. Modern VLMs can:

- Extract text from complex layouts (multi-column documents, tables, forms).
- Understand the hierarchical structure of documents (headers, sections, lists).
- Interpret tables, including multi-row/multi-column headers and merged cells.
- Read charts and graphs, extracting both the visual data and the labels/legends.
- Process handwritten text with reasonable accuracy (though performance drops for unusual handwriting styles).
- Handle mathematical notation, chemical formulas, and other specialized text.

Benchmarks like DocVQA, ChartQA, InfoVQA, and TextVQA measure these capabilities. Frontier VLMs (GPT-4o, Claude 3.5 Sonnet, Gemini 1.5 Pro, InternVL 2.5) achieve accuracy in the 85–95% range on DocVQA, compared to 95–98% for dedicated document AI systems that use specialized OCR pipelines.

### 10.3 Multilingual OCR

VLM OCR capabilities extend to non-Latin scripts, though with varying performance. Chinese, Japanese, and Korean text recognition is generally strong (reflecting the multilingual training data of major VLMs), while less-represented scripts (Arabic, Devanagari, Thai) may have lower accuracy. Models with specific multilingual training emphasis, such as Qwen-VL, tend to perform better on non-Latin OCR.

## 11. Visual Grounding

### 11.1 What is Visual Grounding?

Visual grounding refers to the ability to localize objects or regions in an image based on natural language descriptions — connecting the word "the red car on the left" to a specific bounding box or set of pixels in the image. Grounding is important for practical applications (robotic manipulation, visual assistants, image editing) and for verifying that a VLM's responses are actually based on correct visual understanding rather than learned priors.

### 11.2 Grounding Approaches in VLMs

Several approaches to visual grounding in VLMs have been explored:

**Bounding Box Coordinates as Text**: The simplest approach represents bounding boxes as text tokens in the model's output — e.g., generating `<box>(0.12, 0.34, 0.56, 0.78)</box>` to indicate a bounding box with normalized coordinates. This requires no architectural modifications; the model simply learns to generate coordinate tokens. Models like Qwen-VL, Shikra, and Ferret use this approach.

**Pointing and Clicking**: Molmo and similar models can output specific pixel coordinates (a "point" in the image) rather than bounding boxes, enabling more precise localization and natural "click here" interactions for visual assistants.

**Segmentation Masks**: Some VLMs (LISA, Grounded SAM integration) can generate pixel-level segmentation masks by connecting the VLM's output to a segmentation decoder (typically SAM). The VLM identifies what to segment based on the text query, and the segmentation model produces the mask.

### 11.3 Grounding Challenges

Visual grounding remains challenging for current VLMs. Common failure modes include:

- Generating plausible but incorrect bounding boxes (the box is in a reasonable location but doesn't precisely match the referred object).
- Confusing left/right and spatial relationships.
- Struggling with small objects or objects in cluttered scenes.
- Difficulty with counting-based references ("the third window from the left").

## 12. Benchmarks and Evaluation

### 12.1 General Visual Understanding

**MMMU (Massive Multi-discipline Multimodal Understanding)**: A challenging benchmark requiring college-level subject knowledge across 30 subjects (art, chemistry, engineering, etc.), with questions that require both visual perception and domain expertise. Frontier models score in the 60–75% range as of early 2026, compared to human expert performance of approximately 88%.

**MMBench**: A comprehensive benchmark covering perception, reasoning, and knowledge tasks with a carefully designed evaluation protocol (circular evaluation to mitigate position bias in multiple-choice answers).

**SEED-Bench**: Evaluates 12 aspects of multimodal understanding, from scene understanding to spatial reasoning to action prediction.

### 12.2 Mathematical and Scientific Reasoning

**MathVista**: Tests mathematical reasoning with visual inputs — interpreting graphs, solving geometry problems from diagrams, reading mathematical notation. This benchmark reveals significant variation between models, as it requires both strong visual perception and mathematical reasoning capability.

**AI2D**: Science diagram understanding, requiring interpretation of educational diagrams (biological processes, physical systems, chemical reactions).

### 12.3 Document and Chart Understanding

**DocVQA**: Question answering on scanned documents, testing OCR and document comprehension.

**ChartQA**: Question answering on charts and graphs, requiring both visual parsing and numerical reasoning.

**InfoVQA**: Questions about infographics, combining visual layout understanding with text extraction and reasoning.

**TextVQA**: Questions that require reading text visible in natural images (signs, product labels, book covers).

### 12.4 Hallucination Benchmarks

**POPE (Polling-based Object Probing Evaluation)**: Tests whether models hallucinate objects that are not present in images. The model is asked yes/no questions about object presence, with adversarial negative examples designed to trigger common hallucination patterns.

**CHAIR (Caption Hallucination Assessment with Image Relevance)**: Measures the proportion of objects mentioned in generated captions that are actually present in the image.

**HallusionBench**: A comprehensive hallucination benchmark with carefully designed visual illusions and ambiguous images.

### 12.5 Benchmark Limitations

VLM benchmarks have significant limitations. Many use multiple-choice format, which can be gamed by language-model priors (choosing the most "likely" answer without understanding the image). Benchmark contamination is a growing concern, as popular benchmark images may appear in training data. And many benchmarks test a narrow range of visual skills, failing to capture the breadth of real-world visual understanding needs.

## 13. Inference Cost of Visual Tokens

### 13.1 Token Economics

Visual tokens are processed identically to text tokens in the language model's transformer layers — each visual token participates in self-attention with all other tokens (both visual and textual) and passes through the same feed-forward networks. This means the computational cost of visual understanding scales linearly with the number of visual tokens per image and quadratically with the total sequence length.

For a typical deployment scenario, consider a VLM processing a document image:

| Configuration | Visual Tokens | Total Tokens (with 500 text tokens) | Relative Cost |
|---|---|---|---|
| Low resolution (224×224) | 256 | 756 | 1.0× |
| Medium resolution (336×336) | 576 | 1,076 | 2.0× |
| High resolution (672×672) | 2,304 | 2,804 | 13.8× |
| Dynamic tiling (6 tiles) | 6,144 | 6,644 | 77.5× |

The quadratic scaling of self-attention means that the cost increase from adding visual tokens is superlinear — doubling the number of visual tokens more than doubles the cost because each visual token also increases the cost of processing every other token.

### 13.2 KV Cache Considerations

Visual tokens also occupy space in the KV cache during autoregressive generation. For a model with 32 layers, 32 attention heads, and 128-dimensional head vectors, each token requires 32 × 32 × 128 × 2 (key + value) × 2 bytes (FP16) = 524,288 bytes ≈ 512 KB in the KV cache. A single high-resolution image with 2,304 visual tokens therefore occupies approximately 1.15 GB of KV cache — a significant allocation that reduces the available capacity for text tokens and limits batch sizes.

### 13.3 Cost Optimization Strategies

Several strategies mitigate the cost of visual tokens:

- **Token pruning**: Discard uninformative visual tokens (background, padding) after the first few transformer layers. FastV and similar methods can reduce visual token count by 50–90% with minimal quality loss on many tasks.
- **Resolution routing**: Use a lightweight classifier to determine the appropriate resolution for each image. Simple, large-text images can be processed at low resolution, while dense documents get high resolution.
- **Visual token caching**: For repeated analysis of the same image (e.g., multi-turn conversations about a document), cache the visual token representations to avoid re-encoding.
- **Efficient attention**: FlashAttention and other efficient attention implementations reduce the constant factor in attention computation, though they don't change the asymptotic scaling.

## 14. Limitations and Challenges

### 14.1 Visual Hallucination

Visual hallucination — generating text that describes objects, attributes, or relationships not present in the image — is the most persistent limitation of current VLMs. Common hallucination patterns include:

- **Object hallucination**: Describing objects that are not in the image, often based on contextual expectations (mentioning a keyboard when shown a desk, even if no keyboard is visible).
- **Attribute hallucination**: Incorrect descriptions of colors, sizes, counts, or spatial positions of objects that are present.
- **Relational hallucination**: Incorrectly describing the spatial relationships between objects ("the cat is sitting on the table" when the cat is under the table).
- **Text hallucination**: Generating plausible but incorrect transcriptions of text in images, particularly for partially visible or low-resolution text.

Hallucination rates vary significantly across models and tasks, but even frontier VLMs exhibit non-trivial hallucination rates. Mitigation strategies include training on negative examples (images paired with descriptions of what is not present), reinforcement learning from human feedback on visual tasks, and inference-time techniques like asking the model to verify its own descriptions.

### 14.2 Spatial Reasoning

Spatial reasoning — understanding relative positions, orientations, and geometric relationships — remains challenging. VLMs often struggle with:

- Left/right and above/below distinctions, particularly in complex scenes.
- Counting objects accurately, especially when objects overlap or are partially occluded.
- Understanding perspective and depth from 2D images.
- Interpreting maps, floor plans, and architectural diagrams.

These limitations stem partly from the training data (image-text pairs on the internet rarely describe precise spatial relationships) and partly from the patch-based tokenization, which processes images as sequences of local patches without explicit spatial structure.

### 14.3 Multi-Image Reasoning

While modern VLMs can process multiple images, their ability to reason across images — comparing details, tracking changes, identifying correspondences — is weaker than their single-image understanding. Tasks like "what changed between these two photos?" or "which of these four diagrams best illustrates the concept?" remain challenging, particularly when the relevant differences are subtle.

### 14.4 Temporal Understanding in Video

Even with video-capable VLMs, temporal reasoning (understanding the order of events, causal relationships, and processes that unfold over time) is limited. Most VLMs process video as a sparse set of sampled frames, losing information between frames and making it difficult to understand continuous actions, timing, and dynamics. Section 12 of the companion report on Video Understanding explores this limitation in detail.

### 14.5 Adversarial Robustness

VLMs are vulnerable to adversarial perturbations — small, carefully crafted modifications to images that cause the model to produce incorrect or manipulated outputs. These perturbations can be imperceptible to humans but dramatically change the model's behavior. Adversarial images can be used to bypass safety filters, cause incorrect classifications, or inject instructions through visual prompt injection (embedding text instructions within images that the model follows instead of the user's instructions).

### 14.6 Bias and Fairness

VLMs inherit biases from their training data, which can manifest in several ways: stereotypical associations between visual features and attributes (age, gender, ethnicity assumptions), differential performance across demographic groups (lower accuracy on underrepresented populations), and culturally biased interpretations of ambiguous visual content. These biases are being actively studied and mitigated, but they remain a significant concern for deployment in sensitive contexts.

## 15. Applications

### 15.1 Document Processing and Extraction

VLMs are increasingly deployed for document processing: extracting structured data from invoices, receipts, forms, contracts, and other business documents. The advantage over traditional OCR pipelines is the ability to understand document semantics, not just read text — a VLM can identify that a number is an invoice total, match line items to prices, and handle diverse document layouts without template-based programming.

### 15.2 Accessibility

Visual description capabilities enable accessibility applications: describing images for visually impaired users, generating alt-text for web content, and narrating visual content in videos. The quality of VLM-generated descriptions has improved to the point where they are genuinely useful for accessibility, though challenges remain with describing complex scenes, interpreting cultural context, and avoiding hallucinated details.

### 15.3 Medical Imaging

VLMs are being explored for medical imaging applications: interpreting X-rays, CT scans, pathology slides, and dermatological images. While not yet reliable enough for autonomous diagnosis, VLMs can assist radiologists by highlighting findings, generating preliminary reports, and answering questions about medical images. Specialized medical VLMs (LLaVA-Med, Med-PaLM M, BiomedCLIP) achieve better performance on medical tasks than general-purpose VLMs.

### 15.4 Autonomous Systems and Robotics

Vision-language models are increasingly integrated into robotic systems and autonomous agents, providing the visual understanding component that allows these systems to interpret their environment and follow natural language instructions. Models like RT-2 (Robotics Transformer 2) directly connect VLM-style architectures to robotic action outputs.

### 15.5 Content Moderation and Safety

VLMs are used for automated content moderation: detecting unsafe, harmful, or policy-violating visual content at scale. The advantage over traditional computer vision classifiers is the ability to understand context and nuance — a VLM can distinguish between educational medical content and graphic violence, or between artistic nudity and explicit content, based on contextual cues that simple classifiers miss.

## 16. Future Directions

### 16.1 Toward Unified Multimodal Models

The trend toward natively multimodal models — processing text, images, audio, video, and potentially other modalities (3D, tactile) in a single architecture — is accelerating. Rather than assembling VLMs from separately pre-trained components, future models will likely be multimodal from inception, trained on interleaved multimodal data that mirrors the rich sensory experience of the physical world.

### 16.2 World Models

VLMs are evolving toward "world models" that maintain internal representations of physical environments — understanding physics, spatial layout, object permanence, and causal relationships. This direction, still largely aspirational, would enable VLMs to reason about what will happen next in a scene, plan physical actions, and understand the consequences of interventions.

### 16.3 Efficient Visual Processing

Reducing the computational cost of visual understanding remains a key research direction. Techniques including learnable visual tokenization (training custom tokenizers that represent images more efficiently than fixed patch grids), dynamic compute allocation (spending more computation on visually complex regions and less on simple backgrounds), and architectural innovations that reduce the cost of cross-modal attention will all contribute to making VLMs more practical for deployment.

### 16.4 Grounding and Action

The integration of visual grounding (precise spatial understanding) with action capabilities (clicking, drawing, pointing) is enabling VLMs to serve as visual agents that can interact with graphical user interfaces, navigate web pages, and operate software. Models like CogAgent, SeeClick, and OS-Atlas demonstrate early versions of this capability, and rapid improvement is expected through 2026–2027.

## 17. Conclusion

Vision-language models have transformed the relationship between computer vision and natural language processing, evolving from clunky pipelines that connected separate vision and language components into unified systems capable of rich, nuanced visual understanding and generation. The field has progressed remarkably quickly: from CLIP's contrastive pre-training in 2021, through LLaVA's demonstration that simple architectures and instruction tuning could produce capable VLMs in 2023, to the frontier multimodal models of 2025–2026 that approach human-level performance on many visual understanding tasks.

Key architectural decisions — the choice of visual encoder, the integration strategy (early fusion vs. cross-attention vs. adapters), resolution handling, and token compression — have significant implications for model capability, inference cost, and practical deployability. The open-source ecosystem, led by models like InternVL, Qwen-VL, and the LLaVA family, has made VLM capabilities broadly accessible, while commercial models from OpenAI, Anthropic, and Google continue to push the capability frontier.

Persistent challenges — visual hallucination, spatial reasoning limitations, high inference costs for high-resolution images, and adversarial vulnerabilities — temper the impressive progress and define the active research frontier. As VLMs evolve toward natively multimodal architectures with stronger grounding and world-modeling capabilities, they are becoming not just tools for understanding images but foundations for AI systems that perceive and reason about the visual world.

## References

1. Dosovitskiy, A., et al. "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale." ICLR 2021.
2. Radford, A., et al. "Learning Transferable Visual Models From Natural Language Supervision." ICML 2021.
3. Zhai, X., et al. "Sigmoid Loss for Language Image Pre-Training." ICCV 2023.
4. Liu, H., et al. "Visual Instruction Tuning." NeurIPS 2023.
5. Liu, H., et al. "Improved Baselines with Visual Instruction Tuning." CVPR 2024.
6. Liu, H., et al. "LLaVA-NeXT: Improved Reasoning, OCR, and World Knowledge." 2024.
7. Alayrac, J.-B., et al. "Flamingo: a Visual Language Model for Few-Shot Learning." NeurIPS 2022.
8. Li, J., et al. "BLIP-2: Bootstrapping Language-Image Pre-training with Frozen Image Encoders and Large Language Models." ICML 2023.
9. Chen, Z., et al. "InternVL: Scaling up Vision Foundation Models and Aligning for Generic Visual-Linguistic Tasks." CVPR 2024.
10. Wang, P., et al. "Qwen2-VL: Enhancing Vision-Language Model's Perception of the World at Any Resolution." 2024.
11. OpenAI. "GPT-4V(ision) System Card." 2023.
12. Anthropic. "The Claude 3 Model Family." 2024.
13. Gemini Team, Google. "Gemini: A Family of Highly Capable Multimodal Models." 2024.
14. Anderson, P., et al. "Bottom-Up and Top-Down Attention for Image Captioning and Visual Question Answering." CVPR 2018.
15. Yue, X., et al. "MMMU: A Massive Multi-discipline Multimodal Understanding and Reasoning Benchmark for Expert AGI." CVPR 2024.
