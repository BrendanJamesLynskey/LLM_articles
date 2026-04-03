# Video Understanding with Large Language Models

*April 2026 · Technical Report*

## 1. Introduction

Video is the dominant medium of digital information. Billions of hours of video are uploaded, shared, and consumed each day across platforms like YouTube, TikTok, and streaming services. Yet for large language models, video has remained the most challenging modality to integrate — far harder than static images or text. The challenge is fundamentally one of scale and complexity: a single minute of video at 30 frames per second contains 1,800 images, each of which would generate hundreds or thousands of visual tokens if processed by a standard vision encoder. The temporal dimension adds requirements for understanding motion, causality, event sequences, and narrative structure that static image understanding does not address.

Despite these challenges, the period from 2024 through 2026 has seen rapid progress in video understanding with LLMs. Models like Gemini 1.5 Pro (with its million-token context), GPT-4o, and open-source systems like LLaVA-Video and VideoLLaMA 2 have demonstrated capabilities ranging from summarizing long videos to answering detailed questions about specific moments to analyzing temporal relationships between events. These advances are driven by innovations in frame sampling, video tokenization, temporal modeling, and the scaling of both models and context windows.

This report provides a comprehensive technical examination of video understanding with LLMs: the fundamental challenges of video as a modality, the architectural approaches for encoding and reasoning about video, the major models and their capabilities, and the benchmarks, applications, and open problems that define the field.

## 2. The Challenge of Video

### 2.1 The Dimensionality Problem

A single 720p image (1280×720 pixels) processed by a ViT encoder with 14×14 patches produces approximately 3,700 visual tokens. A 10-minute video at 1 fps (a very sparse sampling rate) contains 600 frames, producing approximately 2.2 million visual tokens. At 30 fps, this increases to 66 million tokens for the same 10-minute video — far beyond the context window of any current language model.

Even with aggressive compression (lower resolution, token pooling, temporal merging), representing a video as a sequence of visual tokens requires orders of magnitude more tokens than representing a single image. This fundamental dimensionality challenge drives every architectural decision in video-language models.

### 2.2 Temporal Reasoning

Understanding video requires reasoning about time in ways that static images do not:

- **Temporal ordering**: Understanding the sequence of events (what happened first, what caused what).
- **Duration understanding**: Perceiving how long events take (distinguishing a quick glance from a sustained gaze, a brief touch from a long press).
- **Motion understanding**: Perceiving movement, speed, direction, and acceleration.
- **State changes**: Recognizing transformations (a door opening, water boiling, a person aging across a timelapse).
- **Causality**: Understanding cause-effect relationships between events (the ball was thrown, then it broke the window).
- **Anticipation**: Predicting what will happen next based on observed patterns.

These temporal reasoning capabilities require the model to maintain and reason about information across many frames, which is challenging when frames are processed as independent images or when temporal context is limited by the sampling strategy.

### 2.3 Redundancy and Information Density

Video contains enormous temporal redundancy. In a typical video, most of the visual content changes slowly — the background is static, objects move gradually, and significant events occupy only a fraction of the total frames. A 10-minute interview video might contain informative visual changes (gestures, expressions, visual aids) in perhaps 5% of its frames, with the remaining 95% being nearly identical.

This redundancy is both a problem (wasting compute on repeated information) and an opportunity (enabling aggressive frame sampling without significant information loss for many tasks). The key challenge is adaptive — knowing which frames contain the informative content and allocating more processing to those frames.

### 2.4 Audio-Visual Integration

Real-world video understanding often requires integrating visual and audio information. A cooking video's narration provides context for the visual actions; a news broadcast's speech content complements the visual imagery; a movie scene's emotional impact depends on both the visual composition and the soundtrack. Most current video-language models process only the visual track, missing the audio information that humans rely on heavily for video understanding.

## 3. Frame Sampling Strategies

### 3.1 Uniform Sampling

The simplest approach samples frames at regular intervals — every N-th frame or at a fixed rate (e.g., 1 fps, 0.5 fps). Uniform sampling is straightforward to implement and ensures coverage of the entire video, but it wastes computation on redundant frames (long static shots) while potentially missing brief but important events.

For a 10-minute video sampled at 1 fps, uniform sampling produces 600 frames. At 0.5 fps, 300 frames. At 0.1 fps (one frame every 10 seconds), just 60 frames. The choice of sampling rate directly trades off between temporal coverage and computational cost.

Common configurations in major models:

| Model | Typical Sampling | Frames for 10-min Video |
|---|---|---|
| GPT-4o | Adaptive, ~1-2 fps | ~600–1200 |
| Gemini 1.5 Pro | 1 fps | ~600 |
| LLaVA-Video | 1 fps (configurable) | ~600 |
| VideoLLaMA 2 | Uniform, task-dependent | 32–256 |

### 3.2 Keyframe Extraction

Keyframe extraction selects only frames where significant visual changes occur, based on:

- **Scene change detection**: Identifying cuts, transitions, and significant visual discontinuities.
- **Motion magnitude**: Selecting frames with high optical flow magnitude (significant movement).
- **Content diversity**: Using image embeddings to select frames that are maximally diverse (covering the most different visual content).
- **Shot boundary detection**: Selecting one representative frame per shot (a continuous camera segment).

Keyframe extraction is more efficient than uniform sampling for videos with varying information density. A 10-minute interview might yield only 20 keyframes (reflecting the limited visual variety), while a 10-minute action sequence might yield 200 keyframes (reflecting the rapid visual changes).

### 3.3 Adaptive Sampling

Adaptive sampling adjusts the sampling rate based on the content or the query. If the user asks "what happens at the 3:45 mark?", adaptive sampling concentrates frames around that timestamp. If the user asks "describe the overall content," adaptive sampling distributes frames uniformly but may increase density around detected events.

Adaptive sampling approaches include:

**Query-guided sampling**: Use the text query to determine which parts of the video are likely relevant, then sample more densely from those parts. This requires a coarse initial pass to identify relevant segments.

**Hierarchical sampling**: First sample very sparsely (one frame every 30 seconds) to get a global overview, then sample more densely from segments identified as relevant.

**Event-driven sampling**: Use a lightweight event detection model to identify timestamps where significant events occur, then sample densely around those events.

### 3.4 Multi-Resolution Temporal Sampling

Some approaches sample at multiple temporal resolutions simultaneously. A "global" stream samples every 30 seconds for overview context, while a "local" stream samples every 0.5 seconds within a focused temporal window. This enables both long-range temporal understanding and fine-grained event analysis.

## 4. Video Tokenization

### 4.1 Frame-Level Tokenization

The simplest video tokenization approach processes each sampled frame independently through a standard image encoder (ViT, SigLIP, etc.) and concatenates the resulting visual tokens into a long sequence. If each frame produces K visual tokens and N frames are sampled, the total video representation is N × K tokens.

This approach is used by most current video-language models (LLaVA-Video, VideoChat, mPLUG-Owl-Video). Its simplicity is both its strength (it leverages the same image encoder used for single-image understanding) and its weakness (it does not model temporal relationships between frames at the encoding level).

### 4.2 Temporal Token Compression

To reduce the token count, temporal compression merges tokens across frames:

**Temporal pooling**: For each spatial position, average or max-pool the tokens across adjacent frames. Pooling 4 adjacent frames reduces the total token count by 4×.

**Temporal attention pooling**: Use attention mechanisms to weight and merge tokens across time. A set of learnable query tokens attend to the visual tokens of multiple frames, producing a compressed representation. This is analogous to the Perceiver Resampler used in image-language models but applied across the temporal dimension.

**Slow-Fast temporal encoding**: Inspired by the SlowFast architecture for video recognition, process a sparse set of frames at high spatial resolution ("slow" pathway) and a dense set at low spatial resolution ("fast" pathway). The slow pathway captures fine spatial details, while the fast pathway captures temporal dynamics. The two pathways' tokens are combined and fed to the language model.

### 4.3 3D Patch Tokenization

Rather than processing each frame independently, 3D patch tokenization divides the video into spatiotemporal patches — cubes of pixels spanning space and time. A 3D patch of size 14×14×4 (14 pixels wide, 14 pixels tall, 4 frames deep) is projected into a single token, directly capturing local spatiotemporal features.

3D patch tokenization is used in video transformers like ViViT (Video Vision Transformer) and TimeSformer. The advantage is that temporal relationships are captured from the very first layer, rather than being learned only through the language model's self-attention over frame-level tokens. The disadvantage is that 3D patch models must be trained specifically for video (they cannot directly leverage pre-trained image encoders).

### 4.4 Video Tokenizers for Discrete Representations

Analogous to audio codecs (EnCodec, SoundStream) that tokenize audio into discrete tokens, video tokenizers compress video into discrete token sequences:

**Video VQVAE/VQGAN**: Extends the image VQVAE/VQGAN to video by adding temporal convolutional layers or 3D convolutions to the encoder and decoder. The codebook entries represent spatiotemporal patterns, and the resulting discrete token sequence can be processed by a language model.

**Cosmos Tokenizer** (NVIDIA): A video tokenizer that compresses video into a continuous or discrete latent representation with high compression ratios. The tokenizer uses a 3D causal architecture that ensures each latent token depends only on current and past frames, enabling streaming processing.

These discrete video tokenizations enable language models to "read" and potentially "write" video — using the same next-token prediction objective that drives text generation to generate video token sequences that can be decoded back to pixel-space.

### 4.5 Token Budget Management

Managing the total token budget for video is a critical engineering challenge. A practical framework:

1. **Determine available budget**: Based on the model's context window and the expected text input/output length, determine how many tokens can be allocated to video.
2. **Allocate across frames**: Divide the budget by the per-frame token count to determine the number of frames.
3. **Select sampling strategy**: Choose uniform, keyframe, or adaptive sampling to select the determined number of frames.
4. **Apply per-frame compression**: Use spatial token compression (pooling, perceiver resampling) if the per-frame token count is too high.

For example, with a 128K context model, 100K tokens allocated to video, and 256 tokens per frame, the budget supports approximately 390 frames — roughly 6.5 minutes at 1 fps.

## 5. Video-Language Architectures

### 5.1 VideoLLaMA and VideoLLaMA 2

VideoLLaMA (Zhang et al., 2023) extended the LLaVA architecture to video by processing sampled video frames through a visual encoder and connecting them to a language model. VideoLLaMA introduced an audio branch (processing the video's audio track through an audio encoder) alongside the visual branch, enabling audio-visual understanding.

VideoLLaMA 2 (2024) improved upon the original with:
- A spatial-temporal convolution (STC) connector that captures local temporal patterns before feeding tokens to the language model.
- Support for longer videos through more efficient temporal compression.
- Better audio-visual alignment through joint training on audio-visual data.

### 5.2 LLaVA-Video

LLaVA-Video (2024) adapted the LLaVA-OneVision architecture for video understanding. The key design decisions:

- **Frame sampling**: Uniform sampling at a configurable rate, typically 1 fps for long videos and higher rates for short clips.
- **Visual encoding**: Each frame is processed independently by a SigLIP encoder.
- **Temporal ordering**: Frame tokens are arranged in temporal order in the input sequence, with special tokens marking frame boundaries, allowing the language model's self-attention to learn temporal relationships.
- **Training data**: A large-scale video instruction tuning dataset combining existing video QA datasets with newly generated video conversation data.

LLaVA-Video demonstrated strong performance on video understanding benchmarks, achieving competitive results with commercial models on tasks ranging from video captioning to temporal question answering.

### 5.3 Gemini Video Understanding

Google's Gemini models handle video as a native modality, processing video frames (typically sampled at 1 fps) alongside text, images, and audio in their unified architecture. Gemini 1.5 Pro's million-token context window enables processing of very long videos — up to approximately one hour at 1 fps with room for text output.

Gemini's video capabilities include:
- Long video understanding (answering questions about hour-long videos).
- Temporal localization (identifying when specific events occur).
- Video summarization (producing structured summaries of video content).
- Cross-modal reasoning (answering questions that require integrating visual and audio information).
- Multi-video comparison (comparing content across multiple videos).

### 5.4 GPT-4o Video

GPT-4o processes video by sampling frames and encoding them as visual tokens, similar to its image processing but extended across time. The model supports video analysis through both the API (uploading video files) and real-time processing (in voice + video mode, processing the user's camera feed in real time).

GPT-4o's video capabilities are notable for their integration with other modalities — the model can discuss a video while simultaneously analyzing its audio content and responding through speech, enabling natural multimodal conversations about video content.

### 5.5 Specialized Video Models

**InternVideo2**: A large-scale video foundation model from Shanghai AI Lab that combines masked video modeling (self-supervised pre-training on video) with video-language alignment. InternVideo2 achieves strong performance on both video recognition (action classification) and video understanding (question answering) tasks.

**VideoChat and VideoChat2**: Dialogue-oriented video-language models that support multi-turn conversations about video content. VideoChat2 uses a progressive training strategy that builds video understanding capabilities incrementally.

**PLLaVA**: A pooling-based approach that compresses visual tokens from multiple frames using adaptive pooling, significantly reducing the token count while maintaining temporal information.

**Video-LLaVA**: Connects video and image encoders to a unified language model, enabling the same model to handle both image and video understanding tasks without separate architectures.

## 6. Temporal Reasoning

### 6.1 Short-Range Temporal Reasoning

Short-range temporal reasoning involves understanding events and relationships within a few seconds — the duration of individual actions, gestures, and interactions. This includes:

- Recognizing actions (running, cutting, pouring).
- Understanding object interactions (a person picking up a cup, a ball hitting a wall).
- Detecting fine-grained movements (facial expressions, hand gestures).

Short-range temporal reasoning is relatively well-handled by video-language models, particularly when frames are sampled densely enough to capture the action. Models trained on action recognition datasets (Kinetics, Something-Something) develop strong short-range temporal understanding.

### 6.2 Long-Range Temporal Reasoning

Long-range temporal reasoning requires understanding relationships across minutes, hours, or even the full duration of a video:

- **Narrative understanding**: Following a story arc across a movie or TV episode.
- **Process understanding**: Tracking multi-step procedures (cooking a recipe, assembling furniture).
- **Causal chains**: Understanding that event A (10 minutes ago) caused event B (5 minutes ago) which led to event C (now).
- **Temporal references**: Resolving references to earlier events ("like what she did before").

Long-range temporal reasoning is significantly harder because it requires maintaining and reasoning about information across many frames, which is limited by the model's context window and the frame sampling strategy.

### 6.3 Temporal Grounding

Temporal grounding is the video equivalent of spatial grounding in images — localizing when something happens in a video based on a natural language description. Given a query like "the moment when the dog catches the frisbee," the model should output a timestamp or time range.

Approaches include:
- **Direct timestamp prediction**: The model generates timestamps as text tokens (e.g., "3:42-3:45").
- **Frame selection**: The model identifies which sampled frames are relevant to the query.
- **Temporal scoring**: Each frame or segment receives a relevance score, and the highest-scoring segments are returned.

Temporal grounding performance varies significantly across models and query types. Simple events ("when does the person enter the room") are grounded with reasonable accuracy, while complex events ("when does the mood shift from happy to sad") remain challenging.

### 6.4 Temporal Hallucination

A distinctive failure mode of video-language models is temporal hallucination — generating descriptions of events that did not occur in the video, based on expectations from training data. If shown a video of someone in a kitchen, the model might describe cooking actions that are plausible but not actually present in the video. Temporal hallucination is particularly difficult to detect because the hallucinated events are contextually plausible.

## 7. Long Video Understanding

### 7.1 The Long Video Challenge

Long videos (10 minutes to hours) present extreme challenges for video-language models:

- **Token budget**: Even at 0.5 fps, a 1-hour video produces 1,800 frames. At 256 tokens per frame, this is 460,800 visual tokens — consuming most of a 512K context window.
- **Information density**: Important events may be scattered sparsely across the video's duration.
- **Narrative complexity**: Long videos often have complex narrative structures, multiple storylines, recurring characters, and evolving themes.
- **Memory requirements**: The KV cache for processing hundreds of thousands of visual tokens requires tens of gigabytes of memory.

### 7.2 Hierarchical Summarization

Hierarchical summarization addresses long videos by processing them in stages:

1. **Segment-level processing**: Divide the video into segments (e.g., 1-minute segments). Process each segment independently, generating a text summary.
2. **Summary aggregation**: Feed the segment-level summaries into the language model as text context, enabling reasoning across the full video.
3. **Targeted re-analysis**: If the user's query requires detailed information about a specific segment, re-process that segment at higher temporal resolution.

This hierarchical approach trades off between the fidelity of direct visual processing (which requires many tokens) and the efficiency of text summaries (which compress information aggressively). The text summaries lose visual details that might be relevant, but they enable reasoning over much longer videos than direct visual processing allows.

### 7.3 Memory-Augmented Approaches

Memory-augmented architectures maintain an external memory that accumulates information as the video is processed sequentially:

- **Streaming memory**: Process frames one at a time (or in small batches), updating a fixed-size memory bank with relevant information. The memory bank is then used as context for answering questions. This enables processing arbitrarily long videos with bounded memory.
- **Retrieval-augmented video**: Store frame features or segment summaries in a retrievable index. When a question is asked, retrieve the most relevant frames/summaries and process only those. This combines the efficiency of sparse processing with the ability to access detailed information when needed.

MovieChat (2023) exemplifies the memory-augmented approach, using a frame memory and a text memory that accumulate information across long videos, enabling question-answering on movies and TV episodes.

### 7.4 Gemini's Long-Context Approach

Gemini 1.5 Pro's million-token context window enables a more direct approach: simply process all sampled frames in a single context, relying on the model's long-context attention to reason across the full video. At 1 fps with ~250 tokens per frame, Gemini can process approximately 66 minutes of video in a single context.

This approach avoids the information loss of hierarchical summarization and the complexity of memory-augmented architectures, but it requires a model with a very large context window and the ability to attend over very long sequences effectively. The computational cost is also substantial — processing 500K+ visual tokens through a large transformer is expensive.

## 8. Streaming Video Analysis

### 8.1 Real-Time Video Processing

Streaming video analysis processes video frames as they arrive in real time, without access to future frames. This is essential for applications like:

- Live video surveillance and monitoring.
- Real-time sports analysis.
- Interactive video assistants (GPT-4o's live camera mode).
- Autonomous driving and robotics.
- Live event narration and accessibility.

### 8.2 Architectural Requirements

Streaming video processing requires:

- **Causal processing**: Each frame can only be processed using information from current and past frames, not future frames.
- **Bounded memory**: The system cannot store all past frames — it must maintain a fixed-size representation of video history.
- **Low latency**: Processing each frame must complete before the next frame arrives (typically 33ms for 30 fps video, 100ms for 10 fps).
- **Online decision-making**: The system must decide in real time what to remember, what to discard, and when to generate output.

### 8.3 Approaches

**Sliding window**: Maintain a fixed-size window of recent frames, discarding old frames as new ones arrive. Simple but loses all information about events outside the window.

**Memory bank + current frame**: Maintain a compressed memory bank of past video content alongside the current frame. The memory bank is updated at each step, adding information from the new frame and potentially forgetting old information.

**Event-triggered processing**: Process most frames minimally (updating a lightweight state tracker) and trigger full LLM-based analysis only when significant events are detected. This reduces the average computational cost while maintaining responsiveness to important events.

**Token recycling**: Reuse computed visual tokens from previous frames, only recomputing tokens for parts of the image that have changed. This exploits the temporal redundancy of video to reduce computation.

## 9. Benchmarks

### 9.1 VideoMME

VideoMME (Video Multi-Modal Evaluation) is a comprehensive benchmark for evaluating video understanding capabilities across multiple dimensions:

- **Short videos** (< 2 minutes): Testing perception and basic understanding.
- **Medium videos** (4–15 minutes): Testing multi-event understanding and temporal reasoning.
- **Long videos** (30–60 minutes): Testing long-range comprehension and narrative understanding.
- **With/without subtitles**: Testing whether models rely on visual information or text shortcuts.

VideoMME provides multiple-choice questions that test temporal perception, spatial reasoning, attribute understanding, and logical reasoning. Frontier models (GPT-4o, Gemini 1.5 Pro) score in the 60–75% range on the full benchmark, with performance degrading significantly on long videos.

### 9.2 EgoSchema

EgoSchema is a benchmark focused on egocentric (first-person) video understanding, derived from the Ego4D dataset. It contains 5,000 multiple-choice questions about 3-minute egocentric video clips, testing:

- Understanding of first-person activities and interactions.
- Temporal reasoning about sequences of actions.
- Object and environmental understanding from an egocentric perspective.

EgoSchema is particularly challenging because egocentric video is visually complex (rapid camera movement, occlusion, diverse environments) and requires understanding of the agent's intentions and actions from a first-person viewpoint.

### 9.3 Other Notable Benchmarks

**SEED-Bench-Video**: Extends the SEED-Bench image understanding benchmark to video, evaluating temporal understanding, action recognition, and scene comprehension.

**MVBench**: A multi-dimensional video understanding benchmark covering 20 tasks including action recognition, scene classification, temporal reasoning, and spatial reasoning.

**ActivityNet-QA**: Question answering on long activity videos, testing understanding of complex multi-step activities.

**NExT-QA**: A benchmark emphasizing causal and temporal reasoning about video events.

**Video-Bench**: A benchmark specifically for evaluating video-language models on diverse video understanding tasks.

### 9.4 Benchmark Challenges

Video benchmarks face several challenges:

- **Multiple-choice format bias**: As with image VLM benchmarks, multiple-choice format allows models to use elimination strategies and language priors.
- **Temporal shortcuts**: Some questions can be answered from a single frame, making them poor tests of temporal understanding.
- **Annotation quality**: Video annotation is expensive and error-prone, leading to noisy ground-truth labels.
- **Duration bias**: Most benchmarks focus on short or medium-length videos, leaving long video understanding underexamined.

## 10. Video Generation Context

### 10.1 The Relationship Between Understanding and Generation

Video understanding and video generation are increasingly interconnected. Models that can generate video (Sora, Runway Gen-3, Pika, Kling) implicitly develop representations of the visual world that may be useful for understanding. Conversely, models that understand video can guide and evaluate video generation.

### 10.2 Sora and World Models

OpenAI's Sora, announced in February 2024, demonstrated that a video generation model trained on large-scale video data develops emergent understanding of physics, perspective, and temporal dynamics. Sora can generate coherent videos up to 60 seconds long, maintaining consistent 3D scenes, realistic physics (to a degree), and smooth camera movements.

The "world model" hypothesis suggests that training a model to generate realistic video forces it to learn an internal model of how the world works — objects have permanence, gravity pulls things down, liquids flow, and actions have consequences. If true, video generation models could provide a foundation for video understanding that goes beyond surface-level pattern matching.

### 10.3 Implications for Video Understanding

The convergence of video understanding and generation suggests a future direction: unified models that can both understand existing video and generate new video, using a shared internal representation of the visual world. Early steps in this direction include models that can:

- Generate video continuations (given the first 5 seconds, predict what happens next).
- Edit videos based on natural language instructions.
- Generate videos from text descriptions while maintaining the realism learned from understanding real videos.

## 11. Applications

### 11.1 Content Moderation and Safety

Video platforms process billions of hours of user-uploaded video. Automated video understanding enables content moderation at scale — detecting policy violations, harmful content, misinformation, and copyright infringement. Video-language models can go beyond simple classification to understand context: distinguishing between educational content about dangerous topics and content that promotes dangerous behavior, for example.

### 11.2 Video Search and Retrieval

Video understanding enables semantic video search — finding specific moments in a video based on natural language queries. Rather than searching by metadata or tags, users can search by describing what they want to find: "the part where the speaker discusses climate change" or "all scenes showing the red car." This transforms video archives from collections of opaque files into searchable databases of visual information.

### 11.3 Accessibility

Video understanding supports accessibility through:
- **Audio description**: Generating descriptions of visual content for visually impaired viewers.
- **Enhanced captioning**: Going beyond speech transcription to describe visual actions, scene changes, and on-screen text.
- **Video summarization**: Creating concise text summaries of video content for people who cannot watch the full video.

### 11.4 Education and Training

Video-language models can analyze educational videos, generating study materials (summaries, quizzes, key point extraction), answering student questions about video content, and identifying the most informative segments of long lectures.

### 11.5 Surveillance and Security

Automated analysis of surveillance video — detecting anomalies, tracking individuals, identifying events — benefits from video-language models' ability to understand complex scenes and reason about temporal patterns. The ethical implications of surveillance applications require careful consideration and governance.

### 11.6 Sports Analysis

Video understanding enables automated sports analysis: tracking player movements, identifying key plays, generating statistical summaries, and providing real-time commentary. The temporal reasoning required (understanding game strategy, recognizing patterns across a full match) makes sports analysis a challenging and commercially valuable application.

### 11.7 Medical and Scientific Video Analysis

Analysis of medical procedures, laboratory experiments, and scientific observations benefits from video understanding models that can track multi-step processes, identify anomalies, and generate structured reports from video observations.

## 12. Compute Challenges

### 12.1 Memory Requirements

Video understanding imposes extreme memory requirements:

- **Visual encoding**: Processing N frames through a ViT-L encoder requires approximately N × 1.2 GB of GPU memory (for activation tensors at batch size 1).
- **KV cache**: Storing the KV cache for N × K visual tokens across L transformer layers requires N × K × L × 2 × D × 2 bytes (where D is the head dimension, factor of 2 for K and V, factor of 2 for FP16). For a 7B model processing 1,000 frames × 256 tokens/frame, this is approximately 17 GB.
- **Attention computation**: Self-attention over N × K visual tokens plus text tokens scales quadratically, dominating computation for long videos.

### 12.2 Latency

Video understanding tasks have diverse latency requirements:

- **Offline analysis** (content moderation, search indexing): Minutes to hours per video are acceptable.
- **Interactive analysis** (user asking questions about a video): Seconds per response is expected.
- **Real-time analysis** (live video monitoring, interactive assistants): Sub-second response times are required.

Meeting real-time requirements while processing visual tokens from video is the most challenging computational constraint.

### 12.3 Optimization Strategies

**Frame-level parallelism**: Process multiple frames through the visual encoder in parallel, exploiting the independence of per-frame encoding.

**Visual token caching**: Cache the visual encoder's output for previously processed frames, avoiding redundant computation when the model re-processes or re-analyzes video.

**Speculative video processing**: Begin processing video frames before the user asks a question, so that the visual tokens are pre-computed and ready when the query arrives.

**Efficient attention for long sequences**: FlashAttention, ring attention, and other efficient attention mechanisms are critical for handling the long sequences produced by video tokenization.

**Token pruning**: After the first few transformer layers, prune uninformative visual tokens (background, static regions), reducing the computational cost of subsequent layers. This is particularly effective for video, where much of the content is static or redundant.

### 12.4 Cost Analysis

Processing video through frontier VLMs is expensive:

| Scenario | Frames | Visual Tokens | Approximate Cost (API) |
|---|---|---|---|
| 1-minute clip, 1 fps | 60 | ~15K | $0.08–0.15 |
| 10-minute video, 1 fps | 600 | ~150K | $0.75–1.50 |
| 1-hour video, 1 fps | 3,600 | ~900K | $4.50–9.00 |
| 1-hour video, 0.1 fps | 360 | ~90K | $0.45–0.90 |

These costs assume approximately $5–10 per million input tokens. For applications that process large volumes of video (content moderation at platform scale), these costs are significant and drive the use of smaller, more efficient models and smarter frame sampling strategies.

## 13. Future Directions

### 13.1 Native Video Models

Most current video-language models process video as a sequence of independent frames, relying on the language model's self-attention to learn temporal relationships. Future models will likely process video more natively — with architectures specifically designed for spatiotemporal data, such as 3D transformers or state-space models that efficiently handle long temporal sequences.

### 13.2 Audio-Visual Integration

Integrating audio understanding with visual understanding for video is an active frontier. Models that can simultaneously process and reason about the visual track, the speech track, and the non-speech audio track (music, sound effects, environmental sounds) will achieve much richer video understanding than vision-only models.

### 13.3 Interactive Video Agents

The combination of real-time video understanding with voice interaction enables video agents that can participate in visual contexts — a user wearing smart glasses could have a conversation with an AI assistant that sees what they see and responds in real time. GPT-4o and Gemini Live have demonstrated early versions of this capability, and rapid improvement is expected.

### 13.4 Video-Centric Training

Current video-language models are typically adapted from image-language models, with video understanding added through additional training. Future models may be trained on video from the start, learning temporal dynamics as a fundamental aspect of their world model rather than as an adaptation of static image understanding.

### 13.5 Longer Videos

Extending video understanding to very long content — full movies, multi-hour meetings, days of surveillance footage — requires advances in both model architecture (longer context windows, more efficient attention) and processing strategies (hierarchical summarization, memory-augmented approaches, retrieval-based systems). The challenge is maintaining the ability to answer detailed questions about specific moments while reasoning about the global structure.

### 13.6 Embodied Video Understanding

For robotics and embodied AI, video understanding must go beyond passive observation to support active interaction with the environment. This requires understanding not just what is happening in a video but what actions are available, what their consequences would be, and how to achieve specific goals. The connection between video understanding and action planning is a key frontier for embodied AI systems.

## 14. Conclusion

Video understanding represents the most challenging frontier for multimodal language models, combining the difficulty of visual perception with the complexity of temporal reasoning, the scale of continuous data, and the computational cost of processing millions of visual tokens. The progress from 2024 through 2026 has been substantial — models can now summarize long videos, answer temporal questions, and engage in real-time video analysis — but significant gaps remain, particularly in long-range temporal reasoning, fine-grained action understanding, and efficient processing of extended video content.

The key architectural decisions — frame sampling strategy, visual tokenization approach, temporal modeling method, and token budget management — determine the tradeoff between temporal coverage, visual detail, and computational cost. No single approach dominates; the optimal choice depends on the video length, the task requirements, and the available computational budget.

As context windows continue to grow, video generation models advance, and hardware becomes more capable, video understanding will increasingly become a standard capability of multimodal AI systems. The models of the near future will not just watch video — they will understand its narrative, remember its details, reason about its implications, and engage in real-time visual interaction with the world.

## References

1. Maaz, M., et al. "VideoLLaMA 2: Advancing Spatial-Temporal Modeling and Audio Understanding in Video-LLMs." 2024.
2. Zhang, H., et al. "Video-LLaMA: An Instruction-tuned Audio-Visual Language Model for Video Understanding." 2023.
3. Zhang, Y., et al. "LLaVA-Video: Video Understanding with Large Multimodal Models." 2024.
4. Li, K., et al. "MVBench: A Comprehensive Multi-Modal Video Understanding Benchmark." 2024.
5. Fu, C., et al. "VideoMME: The First-Ever Comprehensive Evaluation Benchmark of Multi-Modal LLMs in Video Analysis." 2024.
6. Mangalam, K., et al. "EgoSchema: A Diagnostic Benchmark for Very Long-Form Video Language Understanding." NeurIPS 2023.
7. OpenAI. "Sora: Creating Video from Text." 2024.
8. Gemini Team, Google. "Gemini 1.5: Unlocking Multimodal Understanding across Millions of Tokens of Context." 2024.
9. Arnab, A., et al. "ViViT: A Video Vision Transformer." ICCV 2021.
10. Bertasius, G., et al. "Is Space-Time Attention All You Need for Video Understanding?" ICML 2021.
11. Song, E., et al. "MovieChat: From Dense Token to Sparse Memory for Long Video Understanding." CVPR 2024.
12. Wang, Y., et al. "InternVideo2: Scaling Foundation Models for Multimodal Video Understanding." 2024.
13. Lin, B., et al. "Video-LLaVA: Learning United Visual Representation by Alignment Before Projection." 2024.
14. Xu, H., et al. "PLLaVA: Parameter-free LLaVA Extension from Images to Videos for Video Dense Captioning." 2024.
15. Xiao, J., et al. "NExT-QA: Next Phase of Question-Answering to Explaining Temporal Actions." CVPR 2021.
