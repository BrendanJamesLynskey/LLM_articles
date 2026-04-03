# Text-to-Image Diffusion Models: From DDPM to Flux

*April 2026 · Technical Report*

## 1. Introduction

The ability to generate photorealistic images from natural language descriptions — once the domain of science fiction — has become routine. A user types "a golden retriever wearing a space suit, floating above Earth, oil painting style" and, within seconds, receives a detailed, coherent image that matches the description. This capability, which emerged suddenly into public consciousness with DALL-E 2 and Stable Diffusion in 2022, is built on the mathematical framework of diffusion models: generative models that learn to create images by learning to reverse a gradual noising process.

Diffusion models have rapidly displaced earlier generative approaches (GANs, VAEs) as the dominant paradigm for image generation, and have been extended to video, audio, 3D content, and molecular design. The field has evolved from the foundational DDPM (Denoising Diffusion Probabilistic Models) paper in 2020 through a series of architectural innovations — latent diffusion, classifier-free guidance, the shift from U-Net to DiT (Diffusion Transformer) — and a succession of increasingly capable models: Stable Diffusion 1.5, 2, XL, and 3, DALL-E 2 and 3, Midjourney, and Flux.

This report provides a comprehensive technical examination of text-to-image diffusion models: the mathematical foundations, the key architectural components (VAE, denoising network, text encoder), the training and inference procedures, the major models and their innovations, fine-tuning techniques (ControlNet, LoRA), inference optimization, safety considerations, and the extension to video generation.

## 2. Diffusion Fundamentals

### 2.1 The Forward Process (Adding Noise)

The core idea of diffusion models is to define a gradual process that transforms data (images) into pure noise, then learn to reverse this process to generate new data from noise.

The forward (noising) process takes a clean image x₀ and progressively adds Gaussian noise over T timesteps:

```
q(x_t | x_{t-1}) = N(x_t; √(1-β_t) · x_{t-1}, β_t · I)
```

where β_t is a noise schedule that controls how much noise is added at each step. After T steps (typically T=1000), x_T is approximately pure Gaussian noise, with all structure of the original image destroyed.

A key mathematical property allows sampling x_t at any timestep directly from x₀ without iterating through intermediate steps:

```
q(x_t | x₀) = N(x_t; √(ᾱ_t) · x₀, (1-ᾱ_t) · I)
```

where ᾱ_t = ∏_{s=1}^t (1-β_s) is the cumulative product of the noise schedule. This means x_t can be written as:

```
x_t = √(ᾱ_t) · x₀ + √(1-ᾱ_t) · ε, where ε ~ N(0, I)
```

This reparameterization is essential for efficient training, as it allows sampling training examples at any noise level without sequential forward passes.

### 2.2 The Reverse Process (Removing Noise)

The reverse (denoising) process learns to recover x_{t-1} from x_t — to remove one step of noise. A neural network (the denoising model) is trained to predict the noise ε that was added, given the noisy image x_t and the timestep t:

```
ε_θ(x_t, t) ≈ ε
```

The denoising model is trained with a simple objective — minimize the mean squared error between the predicted noise and the actual noise:

```
L = E_{x₀, t, ε} [||ε - ε_θ(x_t, t)||²]
```

During generation, starting from pure noise x_T ~ N(0, I), the model iteratively denoises:

```
x_{t-1} = (1/√(α_t)) · (x_t - (β_t/√(1-ᾱ_t)) · ε_θ(x_t, t)) + σ_t · z
```

where z ~ N(0, I) is fresh noise (for stochastic sampling) and σ_t controls the stochasticity. After T denoising steps, the result x₀ is a clean generated image.

### 2.3 Score Matching Perspective

An alternative perspective views diffusion models through the lens of score matching. The "score" of a distribution is the gradient of its log-probability: ∇_x log p(x). A score-based model learns to estimate this gradient at each noise level, and generation proceeds by following the estimated score (gradient ascent on the log-probability) to move from low-probability noise toward high-probability images.

The score matching and denoising perspectives are mathematically equivalent: predicting the noise ε is proportional to predicting the negative score ∇_x log p(x_t). This equivalence connects diffusion models to the broader framework of score-based generative models (Song and Ermon, 2019; Song et al., 2021) and provides theoretical grounding for the design of noise schedules, sampling procedures, and training objectives.

### 2.4 Noise Schedules

The noise schedule β_1, β_2, ..., β_T controls the rate of noise addition and significantly affects generation quality.

**Linear schedule**: β_t increases linearly from β_1 = 10⁻⁴ to β_T = 0.02. Simple but allocates too many steps to the high-noise regime where the model learns mostly to generate coarse structure.

**Cosine schedule**: Proposed by Nichol and Dhariwal (2021), the cosine schedule provides a more uniform distribution of noise levels, allocating more denoising steps to the low-noise regime where fine details are generated. This produces noticeably better image quality, especially for fine details and textures.

**Shifted schedules**: For high-resolution images, the effective noise level at each step depends on the image resolution. The same β_t corresponds to a lower signal-to-noise ratio for higher-resolution images (because the noise is per-pixel while the signal has spatial structure). Resolution-dependent schedule shifting, used in some Stable Diffusion variants, adjusts the schedule based on the target resolution.

## 3. DDPM and DDIM

### 3.1 DDPM (Denoising Diffusion Probabilistic Models)

Ho et al. (2020) introduced DDPM, which made diffusion models practical for image generation. The key contributions were:

- A simplified training objective (predicting the noise ε rather than the full reverse distribution).
- A specific noise schedule and parameterization that produced stable training.
- Demonstration that diffusion models could generate high-quality images competitive with GANs, with the advantages of stable training and mode coverage (diffusion models do not suffer from mode collapse).

DDPM's limitation was the slow generation process: producing a single image required 1000 sequential denoising steps, each requiring a full forward pass through the denoising network. At the time, generating a single 256×256 image took minutes on a GPU.

### 3.2 DDIM (Denoising Diffusion Implicit Models)

Song et al. (2020) introduced DDIM, which reformulated the reverse process as a deterministic (non-Markovian) mapping from noise to images. The key insight was that the denoising process could skip timesteps — instead of denoising from t=1000 to t=999 to t=998..., DDIM could denoise from t=1000 to t=950 to t=900..., using far fewer steps.

DDIM enabled generation with as few as 50–100 steps (instead of 1000) with moderate quality loss. It also introduced the concept of deterministic generation: given the same starting noise, DDIM always produces the same image, enabling consistent image editing and interpolation in the noise space.

### 3.3 Further Sampling Improvements

Subsequent work produced increasingly efficient sampling methods:

**DPM-Solver**: An ODE solver-based approach that reduces generation to 10–25 steps with minimal quality loss.

**Euler and Euler Ancestral**: Simple first-order ODE solvers adapted for diffusion sampling.

**PNDM (Pseudo Numerical methods for Diffusion Models)**: Applies higher-order numerical integration methods for fewer-step sampling.

**UniPC**: A unified predictor-corrector framework that achieves high-quality generation in as few as 5–10 steps.

These sampling improvements are critical for practical deployment, as they reduce generation time from minutes to seconds.

## 4. Latent Diffusion and the Stable Diffusion Architecture

### 4.1 The Pixel Space Problem

Early diffusion models operated directly on pixel-space images. For a 512×512 RGB image, the denoising network processes a tensor of shape 512×512×3 = 786,432 dimensions. The U-Net architecture used for denoising must process this high-dimensional space at every denoising step, making both training and inference extremely computationally expensive.

### 4.2 The Latent Diffusion Insight

Latent Diffusion Models (LDM), introduced by Rombach et al. (2022), address this by performing the diffusion process in a lower-dimensional latent space rather than pixel space. A variational autoencoder (VAE) first compresses images into a compact latent representation, the diffusion process operates in this latent space, and the VAE decoder converts the generated latent back to pixel space.

This approach offers dramatic computational savings. A 512×512 image compressed by a factor of 8 in each spatial dimension produces a 64×64 latent (with typically 4 channels), reducing the dimensionality from 786,432 to 16,384 — a 48× reduction. The diffusion model operates in this compressed space, making both training and inference much faster.

### 4.3 The VAE (Variational Autoencoder)

The VAE component consists of:

**Encoder**: Compresses a pixel-space image into a latent representation. The encoder uses convolutional layers with downsampling (typically 8× spatial compression) and produces a mean and variance for each latent element.

**Decoder**: Reconstructs a pixel-space image from a latent representation. The decoder uses convolutional layers with upsampling to restore the original spatial resolution.

The VAE is trained separately from the diffusion model, with reconstruction loss (the decoded image should match the original) and KL divergence loss (the latent distribution should be close to a standard Gaussian). The VAE's quality directly impacts the overall image quality — any reconstruction artifacts in the VAE will appear in all generated images.

The Stable Diffusion VAE uses a downsampling factor of 8 and 4 latent channels, meaning a 512×512 pixel image maps to a 64×64×4 latent. Later versions (SDXL, SD3) use improved VAEs with 16 latent channels for better reconstruction quality.

### 4.4 The U-Net Denoising Network

The original Stable Diffusion models use a U-Net architecture for the denoising network. The U-Net is a convolutional architecture with:

**Encoder path**: A series of convolutional blocks with downsampling, producing feature maps at progressively lower spatial resolutions (64→32→16→8).

**Bottleneck**: Processing at the lowest spatial resolution with attention blocks for global context.

**Decoder path**: A series of convolutional blocks with upsampling, restoring the original latent resolution. Skip connections from the encoder to the decoder at each resolution level preserve fine-grained details.

**Attention layers**: Self-attention and cross-attention layers are interspersed throughout the U-Net, enabling global context modeling and conditioning on text embeddings.

**Timestep conditioning**: The current timestep t is embedded as a vector and injected into the network (typically through addition or modulation of intermediate features), telling the network the current noise level.

**Text conditioning**: The text prompt is encoded by a text encoder (CLIP or T5), and the resulting embeddings are injected through cross-attention layers. At each cross-attention layer, the image features serve as queries and the text embeddings serve as keys and values, allowing the image features to attend to relevant parts of the text description.

The U-Net for Stable Diffusion 1.5 has approximately 860 million parameters. For SDXL, it grows to approximately 2.6 billion parameters.

### 4.5 The Text Encoder

The text encoder converts the user's text prompt into a sequence of embeddings that condition the image generation. Different Stable Diffusion versions use different text encoders:

| Model | Text Encoder(s) | Embedding Dim | Max Tokens |
|---|---|---|---|
| SD 1.5 | CLIP ViT-L/14 | 768 | 77 |
| SD 2.0/2.1 | OpenCLIP ViT-H/14 | 1024 | 77 |
| SDXL | CLIP ViT-L + OpenCLIP ViT-bigG | 768 + 1280 | 77 |
| SD 3 | CLIP ViT-L + OpenCLIP ViT-bigG + T5-XXL | 768 + 1280 + 4096 | 77/77/256 |

The evolution toward larger and multiple text encoders reflects the importance of text understanding for image quality. The 77-token limit of CLIP-based encoders constrains the detail of prompts; the addition of T5-XXL in SD3 (with a 256-token limit) enables much more detailed and nuanced prompt understanding.

### 4.6 The Complete SD Architecture

The complete Stable Diffusion generation pipeline:

1. **Text encoding**: The prompt is encoded by the text encoder(s) into embedding vectors.
2. **Noise initialization**: A random noise tensor is sampled in the latent space (e.g., 64×64×4 for SD 1.5 at 512×512).
3. **Iterative denoising**: The U-Net processes the noisy latent, conditioned on the text embeddings and the current timestep, predicting the noise to remove. This is repeated for N steps (typically 20–50) using a chosen sampler (DDIM, Euler, DPM-Solver, etc.).
4. **VAE decoding**: The denoised latent is decoded by the VAE decoder into a pixel-space image.

## 5. Classifier-Free Guidance

### 5.1 The Concept

Classifier-free guidance (Ho and Salimans, 2022) is a technique that dramatically improves the quality and prompt-adherence of generated images. The key idea is to train the denoising model both with and without text conditioning (randomly dropping the text condition during training), then during generation, extrapolate between the unconditional and conditional predictions:

```
ε_guided = ε_uncond + w · (ε_cond - ε_uncond)
```

where w is the guidance scale (typically 5–15) and ε_cond / ε_uncond are the model's noise predictions with and without the text condition.

### 5.2 How It Works

The guided prediction amplifies the difference between the conditional and unconditional predictions. When w=1, the guidance has no effect (standard conditional generation). When w>1, the model is pushed to generate images that are more strongly aligned with the text prompt — producing sharper, more detailed, and more prompt-faithful images at the cost of reduced diversity and potentially oversaturated colors.

The guidance scale is one of the most important generation hyperparameters. Low guidance (w=1–3) produces diverse but potentially unfaithful images. High guidance (w=10–20) produces highly prompt-faithful images but may introduce artifacts, oversaturation, and reduced diversity. The sweet spot (w=7–12 for most models) balances fidelity and quality.

### 5.3 Computational Cost

Classifier-free guidance doubles the computation per denoising step because both the conditional and unconditional predictions must be computed. For N-step generation, this means 2N forward passes through the denoising network. In practice, the conditional and unconditional passes can be batched together, reducing the overhead to approximately 1.5-2× (depending on the batch size and hardware utilization).

### 5.4 Distilled Guidance

Recent work (SDXL Turbo, LCM) has distilled the guidance mechanism into the model itself, eliminating the need for dual forward passes. The distilled model is trained to produce, in a single pass, predictions that match what the guided model would produce with dual passes. This halves the inference computation while maintaining the quality benefits of guidance.

## 6. The U-Net to DiT Shift

### 6.1 Why DiT?

The Diffusion Transformer (DiT), introduced by Peebles and Xie (2023), replaces the convolutional U-Net with a standard transformer operating on patch tokens. Just as Vision Transformers (ViT) replaced CNNs for image classification, DiTs replace U-Nets for diffusion-based image generation.

The motivation is scaling: transformers have demonstrated more predictable and favorable scaling laws compared to convolutional architectures. As compute budgets for training generative models have increased, the transformer architecture provides a more efficient path to higher quality.

### 6.2 DiT Architecture

The DiT architecture:

1. **Patchify**: The noisy latent (e.g., 64×64×4) is divided into non-overlapping patches (e.g., 2×2), producing a sequence of patch tokens (1024 tokens for 64×64 with 2×2 patches).
2. **Positional embedding**: Learned or sinusoidal positional embeddings are added to encode spatial location.
3. **Transformer blocks**: Standard transformer blocks (self-attention + feed-forward) process the token sequence. Conditioning (timestep, text) is injected through adaptive layer normalization (adaLN-Zero) — the normalization parameters (scale and shift) are modulated by the conditioning information.
4. **Output projection**: The final tokens are projected to predict the noise (or velocity, or score) at each patch location.
5. **Unpatchify**: The predicted patches are rearranged back to the latent spatial layout.

### 6.3 Scaling Properties

DiT demonstrates clean scaling laws: image quality (measured by FID score) improves predictably with both model size and compute. The original DiT paper showed models from DiT-S (33M parameters) to DiT-XL (675M parameters), with consistent quality improvements at each scale.

This predictable scaling has made DiT the architecture of choice for frontier image generation models. SD3 and Flux both use DiT-based architectures, and Sora uses a "spacetime DiT" for video generation.

### 6.4 MM-DiT (Multimodal DiT)

Stable Diffusion 3 introduced MM-DiT, which extends DiT with separate transformer streams for image tokens and text tokens. Rather than injecting text through cross-attention (as in the U-Net), MM-DiT processes image and text tokens through dedicated transformer blocks with shared attention — allowing information to flow between modalities through the attention mechanism while maintaining separate feed-forward layers.

This joint processing produces better text-image alignment than cross-attention alone, particularly for complex prompts with multiple objects, spatial relationships, and attributes. MM-DiT also handles the different dimensionalities of image tokens (from the VAE) and text tokens (from the text encoders) through modality-specific projections.

## 7. Key Models

### 7.1 Stable Diffusion 1.5

Released in October 2022 by Stability AI (based on the latent diffusion work from CompVis/LMU Munich), SD 1.5 was the first widely available open-source text-to-image model. Key specifications:

- U-Net: ~860M parameters
- Text encoder: CLIP ViT-L/14
- VAE: 4-channel, 8× downsampling
- Native resolution: 512×512
- Training data: LAION-Aesthetics (a filtered subset of LAION-5B with aesthetic quality scores above a threshold)

SD 1.5 became the foundation of an enormous ecosystem of fine-tuned models, community extensions, and custom workflows. Its relatively small size (approximately 4 GB on disk) made it accessible to users with consumer GPUs.

### 7.2 Stable Diffusion 2 and 2.1

SD 2 (November 2022) upgraded the text encoder to OpenCLIP ViT-H/14 (larger, trained on more data) and supported higher native resolution (768×768). However, SD 2 was less popular than SD 1.5 in the community because:
- The new text encoder was less compatible with the prompt engineering techniques the community had developed for SD 1.5.
- NSFW content was more aggressively filtered from the training data, which some users perceived as reducing the model's expressiveness.
- Fine-tuned model weights from SD 1.5 were not compatible.

SD 2.1 partially addressed these issues with modified training, but the community largely continued to favor SD 1.5 and its fine-tuned derivatives.

### 7.3 Stable Diffusion XL (SDXL)

SDXL (July 2023) was a major upgrade:

- U-Net: ~2.6B parameters (3× larger than SD 1.5)
- Dual text encoders: CLIP ViT-L + OpenCLIP ViT-bigG
- VAE: Improved, 4-channel
- Native resolution: 1024×1024
- Refiner model: An optional second-stage model that refines images at the detail level

SDXL produced significantly higher quality images with better prompt adherence, particularly for complex scenes with multiple objects. The larger model required more GPU memory (approximately 7 GB for inference) but remained accessible on consumer hardware.

### 7.4 Stable Diffusion 3 and 3.5

SD3 (February 2024 announcement, models released through 2024) introduced the MM-DiT architecture and triple text encoders (CLIP + OpenCLIP + T5-XXL):

- Architecture: MM-DiT (transformer-based, replacing U-Net)
- Parameters: 2B (SD3 Medium) to 8B (SD3 Large)
- Text encoders: CLIP ViT-L + OpenCLIP ViT-bigG + T5-XXL (4.7B)
- VAE: 16-channel (improved reconstruction quality)
- Native resolution: 1024×1024

The T5-XXL text encoder dramatically improved prompt understanding, particularly for complex descriptions with precise spatial relationships and attribute bindings (e.g., "a red cube to the left of a blue sphere on a green table"). The MM-DiT architecture scaled more efficiently than the U-Net.

SD 3.5 (late 2024) refined the architecture with improved training and released Medium and Large variants.

### 7.5 DALL-E 2 and DALL-E 3

**DALL-E 2** (April 2022, OpenAI): Used a different approach than latent diffusion — a CLIP image embedding was generated from the text embedding using a diffusion "prior," then a diffusion "decoder" generated the image from the CLIP image embedding. This two-stage approach produced high-quality images but was superseded by the simpler latent diffusion approach.

**DALL-E 3** (October 2023, OpenAI): Integrated tightly with ChatGPT, using GPT-4 to rewrite user prompts into more detailed descriptions before passing them to the image generation model. This "prompt expansion" approach significantly improved the quality and prompt-adherence of generated images, as the model received more detailed and well-structured prompts. DALL-E 3 used a U-Net-based latent diffusion architecture with a T5 text encoder.

### 7.6 Midjourney

Midjourney (versions 1–6, 2022–2024) achieved the highest aesthetic quality among commercial text-to-image models for much of this period. Midjourney's specific architecture is not publicly documented, but the model is noted for:

- Exceptional aesthetic quality and artistic coherence
- Strong understanding of artistic styles and composition
- Excellent photorealism in its later versions (v5, v6)
- Accessed primarily through Discord bot interaction

Midjourney v6 (December 2023) and later versions achieved photorealism and text rendering capabilities that significantly narrowed the gap between generated and real photographs.

### 7.7 Flux

Flux (August 2024, Black Forest Labs — founded by former Stability AI researchers including Robin Rombach, the primary author of the latent diffusion paper) represents the current state of the art in open-source image generation:

- Architecture: Rectified flow transformer (a variant of DiT using flow matching instead of diffusion)
- Parameters: Flux.1-dev (~12B), Flux.1-schnell (distilled, faster)
- Text encoders: CLIP + T5-XXL
- Notable capability: Excellent text rendering within images

Flux uses "rectified flow" — a formulation that learns straight-line paths in latent space from noise to data, rather than the curved paths of traditional diffusion. This produces faster convergence and enables fewer-step generation. Flux.1-schnell can generate high-quality images in as few as 4 steps.

Flux quickly became the preferred open-source model for image generation, displacing SDXL in many workflows.

### 7.8 Other Notable Models

**Playground v2 and v2.5**: Fine-tuned SDXL variants with enhanced aesthetic quality.

**PixArt-α and PixArt-Σ**: Efficient DiT-based models that achieve competitive quality with significantly less training compute.

**Kandinsky**: A multi-stage model combining CLIP, a prior model, and a latent diffusion decoder.

**DeepFloyd IF**: A pixel-space diffusion model (not latent) that demonstrated strong text rendering capabilities.

## 8. ControlNet

### 8.1 The Problem

Standard text-to-image generation provides limited spatial control. The user can describe what they want but cannot precisely specify where objects should appear, what pose a person should have, or what composition the image should follow. ControlNet (Zhang and Agrawala, 2023) addresses this by adding spatial conditioning to pre-trained diffusion models.

### 8.2 Architecture

ControlNet creates a trainable copy of the encoder portion of the U-Net (the "ControlNet") that processes a conditioning image (edge map, depth map, pose skeleton, segmentation map, etc.) alongside the main U-Net. The ControlNet's outputs are added to the corresponding layers of the main U-Net through zero-initialized convolution layers, allowing the conditioning signal to gradually influence the generation process during training.

The key design choice — zero-initialization of the connection layers — ensures that at the start of training, the ControlNet has zero effect on the main U-Net, preserving the pre-trained model's quality. As training progresses, the connection layers learn to inject spatial conditioning information.

### 8.3 Conditioning Types

ControlNet supports diverse conditioning inputs:

| Condition Type | Input | What It Controls |
|---|---|---|
| Canny edges | Edge map | Object boundaries and shapes |
| Depth map | Estimated depth | 3D structure and composition |
| OpenPose | Skeleton keypoints | Human pose and body position |
| Segmentation | Semantic map | Layout and object placement |
| Normal map | Surface normals | 3D surface orientation |
| Scribble | Hand-drawn sketch | Rough composition |
| M-LSD lines | Straight lines | Architectural structure |
| Soft edge (HED) | Soft edges | Softer shape guidance |

Multiple ControlNets can be combined, allowing simultaneous control over pose, depth, and edges, for example.

### 8.4 IP-Adapter

IP-Adapter (Image Prompt Adapter) is a related technique that uses a reference image as conditioning, rather than a structural map. The reference image's features (extracted by CLIP or a similar model) are injected through cross-attention, enabling "image prompting" — generating new images that match the style, subject, or composition of a reference image.

## 9. LoRA Fine-Tuning

### 9.1 LoRA for Diffusion Models

Low-Rank Adaptation (LoRA), originally developed for language model fine-tuning, has been widely adopted for fine-tuning diffusion models. LoRA adds small, trainable low-rank matrices to the attention layers of the denoising network, allowing the model to learn new concepts, styles, or subjects with minimal parameter overhead.

A typical LoRA for Stable Diffusion adds 4–100 MB of parameters (compared to the full model's 2–12 GB), and can be trained on 10–50 images in 30 minutes to 2 hours on a consumer GPU.

### 9.2 Common Use Cases

**Style LoRAs**: Train on images in a specific artistic style (anime, watercolor, pixel art, a specific artist's style) to enable generating images in that style.

**Subject/character LoRAs**: Train on multiple images of a specific person, character, or object to enable generating new images of that subject in different contexts. This is the basis of personalized image generation.

**Concept LoRAs**: Train on examples of a specific concept (a particular type of clothing, architecture style, or visual effect) to add it to the model's vocabulary.

### 9.3 DreamBooth

DreamBooth (Ruiz et al., 2023) is a fine-tuning technique specifically designed for subject-driven generation. Given 3–5 images of a specific subject (a person, pet, object), DreamBooth fine-tunes the entire diffusion model (or a LoRA subset) to associate the subject with a unique identifier token (e.g., "a photo of [V] dog"). After fine-tuning, the model can generate images of the subject in novel contexts: "[V] dog wearing a chef's hat in a kitchen."

DreamBooth uses a prior preservation loss that regularizes the fine-tuning by generating images of the same class (dog, person, etc.) from the original model and including them in the training set. This prevents the model from forgetting the general concept while learning the specific subject.

### 9.4 Textual Inversion

Textual Inversion (Gal et al., 2023) is a lighter-weight personalization technique that learns a new text embedding for a concept rather than modifying the model weights. Given a few images of a concept, textual inversion optimizes a new token embedding that, when used in a prompt, causes the model to generate images of that concept. The model weights remain unchanged, making textual inversion very efficient but less expressive than LoRA or DreamBooth.

## 10. Inpainting and Image Editing

### 10.1 Inpainting

Inpainting — generating content for a masked (missing) region of an image while maintaining consistency with the surrounding unmasked regions — is a natural application of diffusion models. The denoising process is modified to preserve the known regions: at each denoising step, the known regions are replaced with their (appropriately noised) original values, while only the masked region is denoised by the model.

Dedicated inpainting models (e.g., SD Inpainting variants) are fine-tuned with an additional mask channel in the input, enabling the model to better understand the boundary between known and generated regions.

### 10.2 SDEdit and Image-to-Image

SDEdit (Meng et al., 2022) enables image editing by adding noise to an existing image (to a specified level), then denoising with a new text prompt. The noise level controls the tradeoff between fidelity to the original image (low noise = small changes) and adherence to the new prompt (high noise = major changes). This enables text-guided image transformation: turning a sketch into a photorealistic image, changing the style of an existing photo, or modifying specific aspects of an image.

### 10.3 Instruction-Based Editing

InstructPix2Pix (Brooks et al., 2023) and subsequent models enable editing images through natural language instructions ("make it sunny," "add a hat to the person," "change the color to blue"). These models are trained on pairs of (original image, edited image, editing instruction), typically generated synthetically using GPT-4 for instructions and Stable Diffusion for image pairs.

## 11. Inference Optimization

### 11.1 Distilled Models

Model distillation reduces the number of denoising steps required:

**SDXL Turbo**: A distilled version of SDXL that generates images in 1–4 steps using Adversarial Diffusion Distillation (ADD). ADD trains the student model with a combination of distillation loss (matching the teacher's trajectory) and adversarial loss (a discriminator distinguishes student outputs from real images).

**SDXL Lightning**: A distilled SDXL model using progressive distillation, generating images in 2–8 steps.

**LCM (Latent Consistency Models)**: A consistency distillation approach that trains the model to directly map from any noise level to the final image in a single step. LCM-LoRA provides a LoRA adapter that can be applied to any compatible model to enable 2–8 step generation.

### 11.2 Few-Step Generation Approaches

**Consistency Models** (Song et al., 2023): Train a model that directly maps any noise level to the clean image, enabling single-step generation. Consistency training enforces that the model's predictions from different noise levels along the same trajectory converge to the same output.

**Rectified Flow** (Liu et al., 2023): Learns straight-line trajectories from noise to data, enabling efficient few-step generation. Flux uses this approach.

**Flow Matching**: A generalization of diffusion that learns continuous normalizing flows between noise and data distributions. Flow matching simplifies training (the objective is simply to predict the velocity field) and enables more efficient sampling.

### 11.3 Hardware Optimization

**FlashAttention**: Reducing the memory footprint and computation of attention layers in the denoising network. Critical for high-resolution generation where attention operates on many tokens.

**Quantization**: INT8 and INT4 quantization of the U-Net/DiT reduces memory usage and increases throughput. The VAE decoder is typically kept in FP16 as it is more sensitive to quantization.

**Compilation**: torch.compile, TensorRT, ONNX Runtime, and other compilation frameworks optimize the execution graph for specific hardware.

**Token merging (ToMe)**: Merging similar tokens in the transformer-based denoising network to reduce computation. Applied to DiT models, ToMe can reduce inference time by 30–50% with minimal quality loss.

### 11.4 VRAM Requirements

| Model | Minimum VRAM | Recommended VRAM | Notes |
|---|---|---|---|
| SD 1.5 | 4 GB | 8 GB | Consumer GPU accessible |
| SDXL | 6 GB | 12 GB | With optimizations |
| SD3 Medium | 8 GB | 16 GB | T5-XXL can be offloaded |
| Flux.1-dev | 12 GB | 24 GB | FP8/FP16 quantized variants available |
| Flux.1-schnell | 10 GB | 16 GB | Faster, fewer steps |

Memory optimization techniques (CPU offloading, sequential attention, FP8 quantization) can reduce VRAM requirements at the cost of speed.

## 12. Safety and Ethics

### 12.1 NSFW Content Filtering

Most deployed text-to-image models include content safety measures:

**Safety classifiers**: A classifier checks the generated image for NSFW content before displaying it to the user. Stable Diffusion includes a safety checker based on CLIP embeddings that computes similarity to known NSFW concepts.

**Prompt filtering**: Certain words and phrases are blocked or modified before reaching the model.

**Model-level filtering**: Some models are trained with NSFW content removed from the training data, reducing (but not eliminating) the model's ability to generate explicit content.

The open-source nature of Stable Diffusion means that safety measures can be removed by users, which has been both a strength (enabling legitimate artistic freedom and research) and a concern (enabling generation of harmful content).

### 12.2 Watermarking and Provenance

Identifying AI-generated images is increasingly important for combating misinformation and deepfakes. Approaches include:

**Visible watermarks**: Adding a visible mark to generated images identifying them as AI-generated. Easily removed but provides a first line of identification.

**Invisible watermarks**: Embedding imperceptible patterns in generated images that can be detected by specialized tools. Stable Diffusion 2+ includes invisible watermarking. The robustness of invisible watermarks to image manipulation (cropping, compression, filtering) varies.

**C2PA (Coalition for Content Provenance and Authenticity)**: A metadata standard that embeds signed provenance information in image files, documenting how the image was created and modified. Adopted by major AI companies and camera manufacturers.

### 12.3 Bias and Representation

Text-to-image models inherit biases from their training data, which can manifest as:

- **Demographic bias**: Over-representing certain demographics in generated images (e.g., defaulting to lighter skin tones, younger ages, or specific body types).
- **Stereotypical associations**: Linking certain occupations, activities, or attributes to specific demographics.
- **Cultural bias**: Centering Western visual aesthetics and underrepresenting non-Western artistic and cultural traditions.

Mitigation strategies include diversifying training data, adjusting prompt processing to increase representation, and post-hoc filtering. These are active areas of research and product development.

### 12.4 Copyright Concerns

The use of copyrighted images in training data for diffusion models has been the subject of multiple lawsuits (Getty Images v. Stability AI, various artist class actions). Key questions include:

- Is training on copyrighted images fair use?
- Can models generate images that are substantially similar to training images?
- Do artists have the right to opt out of their work being used for training?

These legal questions remain unresolved as of early 2026, with different jurisdictions taking different approaches.

## 13. Video Generation

### 13.1 Extending Diffusion to Video

Video generation extends image diffusion by adding a temporal dimension. The noisy latent becomes a 3D tensor (time × height × width), and the denoising network must process this spatiotemporal volume.

### 13.2 Key Approaches

**Temporal attention**: Add temporal attention layers to the 2D denoising network, allowing the model to attend across frames at the same spatial location. The 2D spatial attention handles within-frame coherence while the temporal attention handles cross-frame consistency.

**3D convolutions/attention**: Replace 2D convolutions and attention with 3D counterparts that operate simultaneously across space and time. This is more computationally expensive but models spatiotemporal structure more directly.

**Frame-by-frame with conditioning**: Generate frames sequentially, conditioning each new frame on previously generated frames. This ensures temporal consistency but introduces artifacts from autoregressive error accumulation.

### 13.3 Sora

OpenAI's Sora (February 2024) demonstrated that a large-scale spacetime DiT, trained on massive video data, could generate coherent videos up to 60 seconds long with remarkable visual quality and physical realism. Sora operates on spacetime patches (3D patches of video latent), processes them through a transformer with spacetime attention, and generates video of variable resolutions and aspect ratios.

Sora's demonstrated capabilities include:
- Consistent 3D scenes maintained across camera movements.
- Realistic physics (to a degree — water flowing, cloth draping, objects falling).
- Extended temporal coherence (consistent characters and environments across a full minute).
- Multiple shots and camera angles within a single generated video.

### 13.4 Other Video Generation Models

**Runway Gen-2 and Gen-3**: Commercial video generation models with progressively improving quality and control.

**Pika**: A video generation platform focusing on short, high-quality video clips.

**Kling** (Kuaishou): A video generation model notable for its strong motion quality and physical realism.

**Mochi** (Genmo): An open-source video generation model.

**CogVideoX** (Zhipu): An open-source video generation model based on a 3D VAE and DiT architecture.

**Wan** (Alibaba): Open-source video generation with strong quality and temporal consistency.

### 13.5 Convergence with Autoregressive Approaches

An emerging trend is the convergence of diffusion-based and autoregressive approaches to image and video generation. While diffusion models generate images through iterative denoising of a full spatial grid, autoregressive models generate images token by token (like text generation), using discrete visual tokens from VQ-VAE/VQ-GAN codebooks.

Models like Parti (Google), Chameleon (Meta), and recent work from multiple labs demonstrate that autoregressive transformers can generate images of comparable quality to diffusion models. The advantages of autoregressive generation include:
- Unified architecture with text generation (the same transformer generates both text and images).
- Natural integration with language model capabilities (reasoning, instruction following).
- Straightforward scaling with established language model training infrastructure.

The disadvantages include:
- Sequential token generation is inherently slower than parallel denoising.
- Visual token vocabularies introduce quantization artifacts.
- The raster-scan ordering of token generation does not match the spatial structure of images.

Hybrid approaches that combine autoregressive planning (generating coarse structure) with diffusion refinement (filling in details) may offer the best of both worlds.

## 14. Conclusion

Text-to-image diffusion models have transformed visual content creation in a span of just four years. From DDPM's foundational demonstration in 2020, through Stable Diffusion's democratization of image generation in 2022, to Flux's state-of-the-art quality in 2024, the field has seen an extraordinary pace of innovation in architecture (U-Net to DiT), efficiency (1000 steps to 1–4 steps), quality (abstract compositions to photorealistic images), and control (text-only to ControlNet, IP-Adapter, and precise spatial conditioning).

The key architectural principles are now well established: latent diffusion (operating in a compressed space for efficiency), classifier-free guidance (extrapolating between conditional and unconditional predictions for quality), and transformer-based denoising networks (scaling predictably with compute). The frontier has shifted from these fundamentals to higher-order challenges: longer video generation, 3D consistency, physical realism, precise spatial control, and the convergence with autoregressive approaches that may ultimately produce unified models capable of generating text, images, video, audio, and 3D content from a single architecture.

The societal implications — deepfakes, copyright, creative displacement, misinformation — remain profound and unresolved. As the technology continues to improve, the gap between generated and real visual content narrows toward imperceptibility, making provenance tracking, watermarking, and media literacy increasingly important.

## References

1. Ho, J., Jain, A., and Abbeel, P. "Denoising Diffusion Probabilistic Models." NeurIPS 2020.
2. Song, J., Meng, C., and Ermon, S. "Denoising Diffusion Implicit Models." ICLR 2021.
3. Rombach, R., et al. "High-Resolution Image Synthesis with Latent Diffusion Models." CVPR 2022.
4. Ho, J. and Salimans, T. "Classifier-Free Diffusion Guidance." NeurIPS Workshop 2022.
5. Peebles, W. and Xie, S. "Scalable Diffusion Models with Transformers." ICCV 2023.
6. Podell, D., et al. "SDXL: Improving Latent Diffusion Models for High-Resolution Image Synthesis." ICLR 2024.
7. Esser, P., et al. "Scaling Rectified Flow Transformers for High-Resolution Image Synthesis." ICML 2024.
8. Black Forest Labs. "Announcing Flux.1." 2024.
9. Zhang, L. and Agrawala, M. "Adding Conditional Control to Text-to-Image Diffusion Models." ICCV 2023.
10. Ruiz, N., et al. "DreamBooth: Fine Tuning Text-to-Image Diffusion Models for Subject-Driven Generation." CVPR 2023.
11. Gal, R., et al. "An Image is Worth One Word: Personalizing Text-to-Image Generation using Textual Inversion." ICLR 2023.
12. Brooks, T., et al. "InstructPix2Pix: Learning to Follow Image Editing Instructions." CVPR 2023.
13. Meng, C., et al. "SDEdit: Guided Image Synthesis and Editing with Stochastic Differential Equations." ICLR 2022.
14. Song, Y., et al. "Consistency Models." ICML 2023.
15. OpenAI. "Video Generation Models as World Simulators." 2024.
