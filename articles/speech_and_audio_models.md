# Speech and Audio Models: From Whisper to Real-Time Voice Agents

*April 2026 · Technical Report*

## 1. Introduction

The integration of speech and audio processing into the large language model ecosystem represents one of the most rapidly evolving frontiers in AI. While text-based LLMs dominated attention through 2022–2023, the period from 2024 through 2026 has seen a convergence of speech recognition, speech synthesis, audio understanding, and language modeling into unified systems capable of real-time voice interaction. The release of GPT-4o with native voice capabilities in May 2024, followed by Gemini Live and a wave of open-source speech models, demonstrated that the era of natural, low-latency voice conversation with AI systems had arrived.

This transformation builds on decades of speech processing research, but the key enabling developments are recent: Whisper's demonstration that massive supervised pre-training could produce robust speech recognition, neural codec models (EnCodec, SoundStream) that tokenize audio into discrete representations compatible with language model architectures, and text-to-speech systems (VALL-E, Bark, XTTS) that generate natural-sounding speech from text prompts. Together, these technologies enable a new generation of voice-native AI agents that listen, understand, speak, and reason in real time.

This report provides a comprehensive technical examination of speech and audio models in the LLM era: the architectures for speech recognition, speech synthesis, and audio understanding; the tokenization schemes that bridge continuous audio and discrete language modeling; the engineering challenges of real-time voice agents; and the deployment infrastructure that makes low-latency voice interaction possible.

## 2. Whisper: Robust Speech Recognition at Scale

### 2.1 Architecture

Whisper, released by OpenAI in September 2022, is a transformer-based encoder-decoder model trained for automatic speech recognition (ASR). Its architecture is straightforward:

**Audio Encoder**: The input audio is resampled to 16 kHz and converted to an 80-channel log-Mel spectrogram with a window size of 25ms and a hop size of 10ms. The spectrogram is processed by two 1D convolutional layers (with GELU activation) that downsample the temporal dimension by a factor of 2, followed by a standard transformer encoder with sinusoidal positional embeddings. The encoder processes 30-second audio chunks, producing a sequence of encoder states.

**Text Decoder**: A standard autoregressive transformer decoder generates the transcription token by token, attending to the encoder states through cross-attention. The decoder uses learned positional embeddings and shares the token embedding with the output projection layer.

Whisper comes in multiple sizes:

| Model | Layers (Enc/Dec) | Width | Heads | Parameters |
|---|---|---|---|---|
| tiny | 4/4 | 384 | 6 | 39M |
| base | 6/6 | 512 | 8 | 74M |
| small | 12/12 | 768 | 12 | 244M |
| medium | 24/24 | 1024 | 16 | 769M |
| large | 32/32 | 1280 | 20 | 1.55B |
| large-v3 | 32/32 | 1280 | 20 | 1.55B |

### 2.2 Training Data and Approach

Whisper's key innovation is its training data, not its architecture. The model was trained on 680,000 hours of weakly supervised audio-text pairs collected from the internet — primarily audio with corresponding transcripts, subtitles, or descriptions. This is orders of magnitude more data than previous supervised ASR systems, which typically trained on 1,000–10,000 hours of carefully transcribed audio.

The "weak supervision" aspect is important: the training data includes imperfect transcriptions, machine-generated subtitles, and approximate alignments. Whisper learns to be robust to this noise through scale — with enough diverse data, the model learns to produce accurate transcriptions despite training on noisy labels.

The training is multitask: the same model handles speech recognition (audio → text), speech translation (audio in language A → text in language B), language identification, voice activity detection, and timestamp prediction. These tasks are distinguished by special tokens in the decoder's input sequence.

### 2.3 Robustness and Generalization

Whisper's most notable property is its robustness. Previous ASR systems often performed well on clean, in-domain audio but degraded dramatically on audio with background noise, accents, domain-specific vocabulary, or recording conditions different from the training data. Whisper, trained on diverse internet audio, generalizes much better to real-world conditions.

On standard ASR benchmarks (LibriSpeech), Whisper large-v3 achieves word error rates (WER) of approximately 2.5% on clean speech and 4.5% on noisy speech — competitive with or exceeding specialized ASR systems that were fine-tuned on the benchmark data. More importantly, Whisper maintains much more consistent performance across diverse conditions: different accents, background noise levels, recording equipment, and speaking styles.

### 2.4 Multilingual Capabilities

Whisper supports transcription in 99 languages and translation from any of these languages to English. Performance varies significantly by language, with high-resource languages (English, Spanish, French, German, Chinese, Japanese) achieving much better accuracy than low-resource languages. For the top 20–30 languages, Whisper provides usable ASR performance; for the remaining languages, accuracy is highly variable.

### 2.5 Limitations

Despite its strengths, Whisper has notable limitations:

- **30-second chunking**: The model processes audio in 30-second segments, with no cross-segment context. Long audio must be segmented, and errors at segment boundaries (mid-word splits, context loss) can occur.
- **Hallucination**: When processing silence or very quiet audio, Whisper can hallucinate transcriptions — generating plausible but nonexistent speech. This is a well-known issue that requires careful handling in production deployments (voice activity detection before transcription, confidence-based filtering).
- **Latency**: The encoder-decoder architecture processes complete 30-second chunks, making it unsuitable for real-time streaming without modification (see Section 8 on streaming ASR).
- **Repetition**: Whisper sometimes enters repetitive loops, generating the same phrase multiple times. This appears related to the attention mechanism and can be mitigated with repetition penalties or beam search constraints.

### 2.6 Whisper Derivatives and Improvements

The open-source community has produced numerous Whisper derivatives:

**Faster Whisper**: An optimized implementation using CTranslate2 that is 4–8× faster than the original with equivalent accuracy. Uses 8-bit quantization and optimized attention kernels.

**WhisperX**: Adds word-level timestamp alignment (using phoneme-based forced alignment after Whisper transcription) and speaker diarization (identifying who is speaking). Essential for applications like meeting transcription.

**Distil-Whisper**: A distilled version of Whisper that maintains 99% of the accuracy at 6× the speed, using a shallower decoder and knowledge distillation from the full model.

**Whisper-large-v3-turbo**: An official OpenAI release that reduces the decoder to 4 layers (from 32), dramatically reducing inference time while maintaining most of the accuracy of the full model.

**Insanely-Fast-Whisper**: Combines FlashAttention, batched processing, and speculative decoding for throughput-optimized transcription of long audio files.

## 3. Speech Tokenization

### 3.1 The Tokenization Challenge

Language models operate on discrete tokens, but speech is a continuous signal. Bridging this gap — converting continuous audio waveforms into discrete token sequences that language models can process — is a fundamental challenge that has driven much of the recent innovation in speech-language model integration.

Two main approaches have emerged: semantic tokens (derived from self-supervised speech models) and acoustic tokens (derived from neural audio codecs).

### 3.2 Semantic Tokens

Self-supervised speech models like HuBERT, wav2vec 2.0, and WavLM learn speech representations by predicting masked portions of audio spectrograms. The learned representations capture linguistic content (phonemes, words, meaning) but discard much of the acoustic detail (speaker identity, prosody, recording conditions).

To create discrete tokens, the continuous representations from these models are clustered (typically using k-means with 500–2000 clusters) to produce a codebook. Each frame of audio is mapped to its nearest cluster center, producing a sequence of discrete tokens at a rate of 25–50 tokens per second.

Semantic tokens are well-suited for tasks where linguistic content is the primary concern (ASR, speech-to-text) but are insufficient for speech synthesis, as they lack the acoustic detail needed to reconstruct natural-sounding audio.

### 3.3 EnCodec

Meta's EnCodec (2022) is a neural audio codec that compresses audio into discrete tokens while preserving enough acoustic detail for high-quality reconstruction. The architecture consists of:

**Encoder**: A convolutional neural network that converts raw audio waveforms to a latent representation, downsampling by a factor of 320 (for 24 kHz audio, this produces a latent frame every ~13ms).

**Residual Vector Quantization (RVQ)**: The continuous latent representation is quantized using multiple codebooks in a residual arrangement. The first codebook captures the coarsest features of the audio; each subsequent codebook captures the residual error from the previous codebooks. A typical configuration uses 8 codebooks, each with 1024 entries, producing 8 token streams at ~75 tokens per second per stream (600 total tokens per second for all 8 codebooks).

**Decoder**: A convolutional neural network that reconstructs the audio waveform from the quantized latent representation.

The residual structure is key: the first codebook captures the most important acoustic information (overall spectral shape, fundamental frequency), while later codebooks add progressively finer detail. This allows a quality-bitrate tradeoff — using fewer codebooks produces lower-quality but more compressible audio.

At 6 kbps (using 8 codebooks at 75 Hz), EnCodec achieves audio quality comparable to MP3 at 64 kbps — a remarkable compression ratio that reflects the power of neural codecs over traditional signal processing.

### 3.4 SoundStream

Google's SoundStream (2021) is architecturally similar to EnCodec: a convolutional encoder, residual vector quantization, and a convolutional decoder. SoundStream was one of the first neural codecs to achieve high-quality audio compression at very low bitrates and inspired much of the subsequent work, including EnCodec.

### 3.5 DAC (Descript Audio Codec)

DAC (2023) improves on EnCodec with better perceptual quality, particularly for music. It uses a similar architecture but with improved training objectives (including a multi-scale STFT discriminator and a perceptual loss) and higher codebook sizes.

### 3.6 Mimi and Beyond

Kyutai's Mimi codec (2024), used in the Moshi speech model, introduces a hybrid approach that combines semantic and acoustic tokenization. The first codebook is trained to capture semantic content (aligned with a distilled version of WavLM), while subsequent codebooks capture acoustic detail. This "semantic first, acoustic second" structure enables more efficient integration with language models — the language model can operate primarily on the semantic codebook while acoustic codebooks are handled by a separate, parallel decoder.

### 3.7 Token Rates and Vocabulary Sizes

The token rate of speech tokenization has significant implications for language model integration:

| Method | Tokens/sec | Vocabulary Size | Information Captured |
|---|---|---|---|
| HuBERT k-means | 50 | 500–2000 | Semantic content |
| EnCodec (1 codebook) | 75 | 1024 | Coarse acoustics |
| EnCodec (8 codebooks) | 600 | 1024 per book | Full acoustics |
| Mimi (semantic + acoustic) | 12.5 + 12.5×7 | 2048 per book | Semantic + acoustics |

For context, text language models process approximately 3–5 tokens per second of speech (the speaking rate of ~150 words per minute ÷ ~1.3 tokens per word). Speech tokenization produces 10–600× more tokens per second than text tokenization of the same content, creating significant computational challenges for language model processing.

## 4. Text-to-Speech

### 4.1 Traditional TTS Pipeline

Traditional text-to-speech systems followed a pipeline architecture: text normalization (expanding abbreviations, numbers) → phoneme conversion (grapheme-to-phoneme models) → acoustic model (predicting mel spectrograms from phonemes) → vocoder (converting spectrograms to waveforms). Each component was trained separately, and errors compounded through the pipeline.

### 4.2 VALL-E: Language Model TTS

Microsoft's VALL-E (January 2023) reconceived text-to-speech as a language modeling problem. Given a text prompt and a 3-second audio clip of the target speaker, VALL-E generates the speech tokens (using EnCodec) for the target text in the target speaker's voice.

The architecture uses two models:
1. **Autoregressive model**: Generates the first EnCodec codebook tokens autoregressively, conditioned on the text tokens and the speaker's audio prompt tokens.
2. **Non-autoregressive model**: Generates the remaining codebook tokens (2–8) in parallel, conditioned on the first codebook tokens and the text/speaker tokens.

This two-stage approach balances quality and speed: the autoregressive model captures the temporal structure (prosody, pacing, intonation) while the non-autoregressive model fills in the acoustic details efficiently.

VALL-E demonstrated remarkable zero-shot voice cloning — reproducing a speaker's voice characteristics from just 3 seconds of reference audio. The generated speech captured not just the speaker's timbre but their speaking style, accent, and emotional tone.

### 4.3 VALL-E 2 and Extensions

VALL-E 2 (2024) improved upon the original with:
- **Repetition-aware sampling**: A sampling strategy that prevents the repetitive artifacts common in autoregressive speech generation.
- **Grouped codec modeling**: Processing multiple codebook levels simultaneously for faster generation.
- **Improved prosody**: Better modeling of emphasis, rhythm, and intonation.

VALL-E 2 achieved human-parity speech synthesis on benchmark evaluations — the generated speech was indistinguishable from real human speech in controlled listening tests.

### 4.4 Bark

Suno's Bark (2023) is an open-source text-to-speech model that generates not just speech but also non-verbal sounds — laughter, sighs, music, background noise. Bark uses a GPT-like architecture operating on EnCodec tokens and is notable for its expressiveness and ability to generate audio with emotional nuance.

Bark's architecture processes text through three stages:
1. **Text to semantic tokens**: A GPT model converts text to HuBERT-derived semantic tokens.
2. **Semantic to coarse acoustic tokens**: A second GPT model converts semantic tokens to the first 2 EnCodec codebooks.
3. **Coarse to fine acoustic tokens**: A third model generates the remaining EnCodec codebooks.

### 4.5 XTTS and Coqui TTS

XTTS (Cross-Language TTS), developed by Coqui, supports text-to-speech in over 16 languages with voice cloning from a short reference clip. The model uses a GPT-2-like architecture conditioned on speaker embeddings extracted from the reference audio and text encoded by a character-level encoder.

XTTS v2 improved multilingual quality and reduced latency, achieving near-real-time generation on consumer GPUs. The model was open-sourced (with some restrictions) and became widely used in the developer community for building voice applications.

### 4.6 F5-TTS and E2 TTS

F5-TTS (2024) and E2 TTS represent a newer generation of text-to-speech models based on flow matching (a variant of diffusion modeling). Rather than generating discrete audio tokens autoregressively, these models generate mel spectrograms directly through an iterative denoising process conditioned on text embeddings.

F5-TTS uses a DiT (Diffusion Transformer) architecture with cross-attention to text representations. It achieves high-quality, natural-sounding speech with good prosody and can perform zero-shot voice cloning. The flow matching formulation enables a quality-speed tradeoff: more denoising steps produce higher quality at the cost of latency.

### 4.7 Parler-TTS and Controllable Generation

Parler-TTS (2024) introduces text-based control over speech characteristics. Instead of requiring a reference audio clip for voice specification, users describe the desired voice in natural language: "A young woman with a warm, friendly tone, speaking at a moderate pace with slight British accent." The model generates speech matching the text description, enabling precise control over voice characteristics without reference audio.

## 5. Speech-to-Speech Models

### 5.1 The Speech-to-Speech Paradigm

Traditional voice assistants followed a cascade architecture: ASR (speech → text) → LLM (text → text) → TTS (text → speech). Each stage adds latency and loses information. Speech-to-speech (S2S) models aim to bypass the text bottleneck, processing speech input directly and generating speech output without an intermediate text representation.

The advantages of speech-to-speech models include:
- **Lower latency**: Eliminating the ASR and TTS stages removes their latency contributions.
- **Preserved paralinguistic information**: Tone, emotion, emphasis, hesitation, and other non-textual vocal cues can be preserved and generated.
- **Natural turn-taking**: The model can process speech in real time, detecting natural pause points and generating responses without waiting for the speaker to finish.
- **Non-verbal communication**: The model can generate laughs, sighs, and other non-verbal vocalizations.

### 5.2 GPT-4o Voice Mode

GPT-4o, released in May 2024, was the first widely deployed speech-to-speech model from a major provider. Rather than the cascade approach of previous ChatGPT voice features (which used Whisper for ASR and a separate TTS system), GPT-4o processes audio natively — the same model that understands text also directly processes and generates audio tokens.

Key capabilities of GPT-4o's voice mode:
- **Low latency**: Response times of 200–400ms, approaching human conversational response times.
- **Emotional expressiveness**: The generated speech conveys appropriate emotion, emphasis, and tone.
- **Voice consistency**: Maintains a consistent voice identity across a conversation.
- **Interruption handling**: Users can interrupt the model mid-response, and it responds naturally.
- **Multilingual**: Supports voice conversations in multiple languages with natural-sounding pronunciation.

The specific architecture of GPT-4o's audio processing has not been fully disclosed, but it operates on audio tokens integrated into the same transformer architecture that processes text tokens. This native multimodality enables the model to reason about both the linguistic content and the paralinguistic features of speech simultaneously.

### 5.3 Gemini Live

Google's Gemini Live (launched late 2024) provides similar real-time voice conversation capabilities through the Gemini model family. Gemini's native multimodality — the model was designed from the start to process text, images, audio, and video — enables natural voice interaction with the same reasoning capabilities as the text-based model.

Gemini Live supports extended voice conversations with features like:
- Proactive responses (the model can speak unprompted based on context).
- Multi-turn memory (maintaining context across long conversations).
- Integration with Google's ecosystem (Search, Maps, Calendar) for tool-augmented voice interaction.

### 5.4 Moshi

Kyutai's Moshi (2024) is the first fully open-source speech-to-speech model. Moshi uses a dual-stream architecture:

- **Main stream**: A 7B-parameter language model (based on Helium) that operates on Mimi semantic tokens, generating both the model's speech and understanding the user's speech.
- **Depth decoder**: A smaller model that generates the acoustic codebook tokens conditioned on the semantic tokens from the main stream.

Moshi is notable for its full-duplex operation — it processes the user's speech and generates its own speech simultaneously, enabling natural overlapping speech and interruptions. The model maintains two token streams (user audio tokens and model audio tokens) that are processed in parallel by the main language model.

### 5.5 GLM-4-Voice and Other Open Models

GLM-4-Voice (Zhipu AI, 2024) is another open speech-to-speech model that integrates speech understanding and generation into a single language model. It uses CosyVoice as its speech tokenizer and can switch between speech and text modalities within a single conversation.

Other notable open speech models include:
- **SpeechGPT**: An early speech-language model that added speech tokens to the LLM vocabulary.
- **Spirit-LM**: Meta's model that interleaves text and speech tokens in a single sequence.
- **Qwen2-Audio**: Alibaba's audio-language model supporting both speech understanding and audio event detection.

## 6. Latency Requirements for Voice Agents

### 6.1 The 500ms Threshold

Human conversational dynamics set strict latency requirements for voice agents. In natural conversation, turn-taking gaps average 200–300ms, and gaps exceeding 500ms are perceived as awkward silences. For a voice agent to feel natural, the end-to-end latency — from the user finishing speaking to the agent beginning to respond — should be under 500ms, with 200–300ms being the target for a truly natural experience.

### 6.2 Latency Breakdown

In a cascade voice agent (ASR → LLM → TTS), the total latency is the sum of component latencies:

| Component | Typical Latency | Best-Case Latency |
|---|---|---|
| Voice Activity Detection | 50–100ms | 30ms |
| Endpoint Detection | 200–500ms | 100–200ms |
| ASR (Whisper) | 500–2000ms | 200ms (streaming) |
| LLM (first token) | 200–1000ms | 50–100ms |
| TTS (first audio) | 200–500ms | 50–100ms |
| Audio playback buffer | 50–100ms | 20ms |
| **Total** | **1200–4200ms** | **450–650ms** |

The largest latency contributors are endpoint detection (determining when the user has finished speaking), ASR processing, and LLM time-to-first-token. Reducing each of these is critical for achieving acceptable latency.

### 6.3 Endpoint Detection

Endpoint detection — determining when the user has finished speaking — is the most challenging latency component. The system must distinguish between:
- **Final pause**: The user has finished their turn (should trigger response generation).
- **Hesitation pause**: The user is thinking but will continue speaking (should not trigger response).
- **Breathing pause**: A natural break in continuous speech (should not trigger response).

Conservative endpoint detection (waiting longer to be sure the user is finished) increases latency but reduces false triggers. Aggressive endpoint detection (triggering quickly) reduces latency but risks interrupting the user.

Modern approaches use:
- **Acoustic features**: Falling intonation and sustained silence indicate turn completion.
- **Linguistic features**: Syntactic completeness of the transcribed text suggests turn completion.
- **LLM-based prediction**: Using the language model itself to predict whether the current utterance is complete.
- **Hybrid approaches**: Combining acoustic, linguistic, and contextual features for more accurate prediction.

### 6.4 Latency Optimization Strategies

**Streaming ASR**: Process audio incrementally rather than waiting for complete utterances. Streaming Whisper implementations and dedicated streaming ASR models (Canary, Parakeet from NVIDIA) produce partial transcriptions in real time.

**Speculative generation**: Begin LLM inference before the user finishes speaking, based on the partial transcription. If the speculation is correct, latency is significantly reduced; if not, generation restarts from the correct input.

**Chunked TTS**: Generate the first chunk of audio as soon as the first text tokens are available from the LLM, rather than waiting for the complete response. This pipelines TTS with LLM generation, overlapping their latencies.

**Edge deployment**: Running ASR and TTS models locally (on the user's device) eliminates network round-trip latency for these components. Whisper tiny/base and lightweight TTS models can run on modern smartphones and laptops.

**Speech-to-speech models**: Bypassing the cascade entirely (as in GPT-4o and Moshi) eliminates the ASR and TTS latency components, achieving the lowest possible end-to-end latency.

## 7. Streaming ASR and TTS

### 7.1 Streaming ASR

Traditional Whisper processes 30-second audio chunks non-incrementally — it must receive the complete chunk before producing any output. For real-time applications, streaming ASR models produce transcription incrementally as audio arrives.

**CTC-based models** (Connectionist Temporal Classification): Models like NVIDIA's Conformer-CTC process audio frame by frame, producing a probability distribution over characters or tokens at each frame. A CTC decoder then collapses the frame-level predictions into a transcription. CTC models are naturally streaming because they process audio left-to-right without future context.

**RNN-Transducer (RNNT)**: A streaming architecture that combines an audio encoder (processing audio frames) with a prediction network (conditioning on previously emitted tokens) and a joint network (combining both to produce the next token). RNNT models process audio in small chunks (10–40ms) and emit transcription tokens incrementally.

**Streaming Whisper implementations**: Several approaches adapt Whisper for streaming use:
- Process audio in overlapping windows, running Whisper on each window and merging the outputs.
- Use Whisper's encoder in streaming mode (processing audio incrementally) with a modified decoder that generates partial transcriptions.
- Distill Whisper into a streaming-native architecture (like a Conformer-RNNT) that retains Whisper's accuracy.

### 7.2 Streaming TTS

Streaming TTS generates audio incrementally as text becomes available, rather than waiting for the complete text input. This is essential for pipelining TTS with LLM generation:

1. LLM generates the first few tokens of the response.
2. TTS begins processing these tokens and generating audio.
3. Audio playback begins while the LLM and TTS continue generating.

This pipelining reduces perceived latency significantly. Instead of waiting for the complete LLM response and then the complete TTS generation, the user hears the beginning of the response while the rest is still being generated.

Key challenges in streaming TTS:
- **Prosody planning**: Natural-sounding speech requires planning prosody (intonation, emphasis, pacing) across the full sentence, but streaming TTS must begin generating audio before the full sentence is available. This can result in unnatural prosody, particularly at sentence boundaries.
- **Buffering tradeoffs**: Smaller audio buffers reduce latency but increase the risk of audio dropouts if generation cannot keep up with playback.
- **Cross-chunk consistency**: The voice quality and style must remain consistent across independently generated audio chunks.

## 8. Voice Activity Detection and Turn-Taking

### 8.1 Voice Activity Detection (VAD)

Voice activity detection determines whether audio contains speech or non-speech (silence, noise, music). VAD is the first processing stage in any voice pipeline and directly impacts both latency and accuracy.

**Silero VAD**: A compact neural network VAD model that achieves high accuracy with very low computational cost. It processes audio in 30ms frames and outputs a speech probability for each frame. Silero VAD has become the de facto standard for open-source voice applications due to its accuracy, speed, and permissive licensing.

**WebRTC VAD**: Google's voice activity detection, originally developed for the WebRTC framework. It uses Gaussian mixture models and is very fast but less accurate than neural approaches, particularly in noisy environments.

**Energy-based VAD**: The simplest approach, detecting speech based on audio energy (volume). Fast but unreliable in noisy environments.

### 8.2 Turn-Taking Models

Beyond simple voice activity detection, turn-taking models predict the structure of conversational turns — who is speaking, when they will stop, and when it is appropriate for the other party to begin speaking. These models enable more natural conversational dynamics in voice agents.

Turn-taking models use features including:
- **Prosodic features**: Pitch contour, energy envelope, and speaking rate changes that signal turn boundaries.
- **Lexical features**: Words and phrases that typically occur at turn boundaries ("so," "anyway," "you know").
- **Timing features**: Pause duration, speech rate changes, and rhythmic patterns.
- **Semantic completeness**: Whether the current utterance forms a complete thought.

### 8.3 Full-Duplex Interaction

The most natural voice interaction is full-duplex — both parties can speak and listen simultaneously, enabling overlapping speech, backchannels ("mm-hmm," "yeah"), and natural interruptions. Full-duplex voice agents must:

- **Echo cancellation**: Remove the agent's own voice from the captured audio so it doesn't interfere with user speech detection.
- **Barge-in detection**: Detect when the user begins speaking while the agent is still generating, and decide whether to stop, pause, or continue.
- **Backchannel generation**: Produce appropriate backchannel responses ("I see," "mm-hmm") during the user's speech to signal active listening.
- **Overlap management**: Handle overlapping speech gracefully, maintaining conversation coherence.

Moshi's dual-stream architecture explicitly supports full-duplex interaction by processing user and model audio streams in parallel. Most cascade-based systems achieve a simplified version of full-duplex by detecting user speech during agent output and triggering an interruption/restart.

## 9. Audio Understanding Beyond Speech

### 9.1 General Audio Understanding

While speech recognition focuses on converting spoken words to text, audio understanding encompasses a broader range of capabilities:

- **Audio event detection**: Identifying non-speech sounds (dog barking, car horn, glass breaking, music genres).
- **Audio captioning**: Generating natural language descriptions of audio content.
- **Sound source separation**: Isolating individual sound sources from a mixture.
- **Music understanding**: Identifying instruments, genres, mood, tempo, and musical structure.
- **Environmental sound classification**: Categorizing ambient audio environments (restaurant, office, outdoors).

### 9.2 Audio-Language Models

Models like Qwen2-Audio, SALMONN, and LTU (Listen, Think, Understand) extend the VLM paradigm to audio. These models connect an audio encoder (typically a Whisper encoder or a self-supervised audio model like BEATs or AST) to a language model, enabling open-ended question answering about audio content.

The architecture typically follows the same pattern as VLMs:
1. **Audio encoder**: Processes audio into a sequence of feature vectors.
2. **Adapter/projector**: Maps audio features to the language model's embedding space.
3. **Language model**: Processes the audio features alongside text tokens and generates text responses.

These models can answer questions like "What instruments are playing in this audio?" or "Describe the environment based on the background sounds" or "Is the speaker happy or sad?" — combining audio perception with language understanding and reasoning.

### 9.3 Multimodal Models with Native Audio

Gemini and GPT-4o process audio natively as part of their multimodal input. This enables audio understanding as part of broader multimodal reasoning — for example, watching a video with audio and answering questions that require integrating visual and auditory information, or analyzing a presentation that includes both slides and spoken narration.

## 10. Open Speech Models and Datasets

### 10.1 Open ASR Models

**Whisper** (OpenAI): The most widely used open ASR model, available in multiple sizes with permissive licensing.

**Canary and Parakeet** (NVIDIA): High-quality ASR models based on the Conformer architecture, optimized for specific languages and streaming applications. Part of NVIDIA's NeMo toolkit.

**MMS (Massively Multilingual Speech)** (Meta): Trained on over 1,100 languages, MMS extends ASR to languages far beyond the reach of Whisper. Quality varies significantly by language, but MMS provides some level of coverage for many underserved languages.

**SeamlessM4T** (Meta): A multimodal translation model that supports speech-to-speech translation, speech-to-text translation, and text-to-speech translation across nearly 100 languages.

### 10.2 Open TTS Models

**Piper**: A fast, lightweight TTS system designed for edge deployment. Supports multiple languages with pre-trained voice models.

**OpenVoice**: An open-source voice cloning model that can clone a speaker's voice from a short reference clip.

**CosyVoice** (Alibaba): A high-quality open TTS model with support for voice cloning and multilingual synthesis.

**Fish Speech**: An open-source TTS model based on VQGAN and language model architectures, supporting voice cloning and multiple languages.

**Mars5-TTS** (Camb AI): A two-stage TTS model using a language model to generate coarse acoustic codes followed by a diffusion model for refinement.

### 10.3 Datasets

**LibriSpeech**: 1,000 hours of English read speech from audiobooks, the standard ASR benchmark dataset.

**Common Voice** (Mozilla): A crowd-sourced multilingual speech dataset with over 20,000 hours across 100+ languages.

**GigaSpeech**: 10,000 hours of English audio from audiobooks, podcasts, and YouTube.

**WenetSpeech**: 10,000+ hours of Mandarin Chinese speech.

**People's Speech**: 31,000 hours of English speech from diverse sources.

**VCTK**: Multi-speaker English dataset commonly used for TTS development.

**Expresso**: An expressive speech dataset with multiple speaking styles (narration, conversation, whisper, etc.) for training more natural TTS systems.

## 11. Deployment Infrastructure

### 11.1 WebRTC

WebRTC (Web Real-Time Communication) is the standard protocol stack for real-time audio/video communication in web applications. For voice agents, WebRTC provides:

- **Low-latency audio streaming**: Peer-to-peer audio connections with sub-100ms transport latency.
- **Echo cancellation**: Built-in acoustic echo cancellation that prevents feedback loops.
- **Noise suppression**: Automatic background noise reduction.
- **Adaptive bitrate**: Automatic adjustment of audio quality based on network conditions.
- **Browser support**: Native support in all major web browsers without plugins.

WebRTC is the transport layer of choice for browser-based voice agents, providing the real-time audio infrastructure without requiring custom networking code.

### 11.2 LiveKit

LiveKit is an open-source real-time communication framework that builds on WebRTC and provides higher-level abstractions for building voice agent infrastructure:

- **Room-based architecture**: Participants (users and AI agents) join rooms, with automatic audio routing between participants.
- **Agent framework**: A Python/Node.js framework for building AI voice agents that connect to LiveKit rooms, process audio, and generate responses.
- **Pipeline components**: Pre-built integrations with ASR models (Whisper, Deepgram), LLMs (OpenAI, Anthropic, local models), and TTS models for building cascade voice agents.
- **Scalability**: Server-side media processing and routing, enabling deployment at scale without peer-to-peer connection management.
- **Telephony integration**: Bridge between WebRTC and traditional telephony (SIP/PSTN) for phone-based voice agents.

LiveKit has become the dominant open-source infrastructure for building AI voice agents, with its agent framework providing abstractions that handle the complex orchestration of streaming ASR, LLM inference, and streaming TTS.

### 11.3 Deepgram

Deepgram provides a commercial ASR API optimized for real-time voice applications, with features including:
- Streaming ASR with sub-200ms latency.
- End-of-utterance detection.
- Multiple concurrent language detection.
- Custom vocabulary and domain-specific models.

### 11.4 Daily and Pipecat

Daily.co provides WebRTC infrastructure as a service, and Pipecat (open-source, maintained by Daily) provides a pipeline framework for building voice agents. Pipecat abstracts the audio processing pipeline into composable components:

```
Audio Input → VAD → ASR → LLM → TTS → Audio Output
```

Each component can be swapped independently (different ASR models, different LLMs, different TTS voices), and the framework handles the streaming, buffering, and synchronization between components.

### 11.5 Telephony Integration

For phone-based voice agents, integration with the public switched telephone network (PSTN) is required. This is typically achieved through:

- **Twilio**: Provides programmable phone numbers and SIP trunking, with WebSocket-based media streaming that can be connected to voice agent pipelines.
- **Vonage (Nexmo)**: Similar programmable telephony services with real-time media streaming.
- **FreeSWITCH**: Open-source telephony platform that bridges PSTN and WebRTC.

The additional latency of telephony networks (20–80ms) must be accounted for in the total latency budget for phone-based voice agents.

## 12. Challenges and Future Directions

### 12.1 Expressiveness and Emotion

Current TTS systems can produce natural-sounding speech for neutral content, but generating speech with appropriate emotional expression, emphasis, and prosodic variety remains challenging. The most expressive systems (GPT-4o, VALL-E 2) achieve human-level naturalness in controlled evaluations, but consistent emotional appropriateness in open-ended conversations is still an active research area.

### 12.2 Speaker Consistency

Maintaining a consistent voice identity across long conversations, different emotional states, and diverse content types is difficult. Voice cloning models can reproduce a speaker's voice from a reference clip, but the cloned voice may drift or become inconsistent when generating content that is very different from the reference material.

### 12.3 Multilingual and Cross-Lingual

Voice agents that can seamlessly switch between languages, handle code-switching (mixing languages within a single utterance), and maintain consistent voice quality across languages are still emerging. SeamlessM4T and GPT-4o demonstrate progress, but quality varies significantly across languages.

### 12.4 Safety and Deepfakes

High-quality voice cloning raises significant safety concerns. The ability to generate speech in anyone's voice from a few seconds of reference audio enables deepfake audio — fake recordings of real people saying things they never said. Mitigations include:

- **Audio watermarking**: Embedding imperceptible watermarks in generated speech that identify it as synthetic.
- **Speaker verification**: Systems that can distinguish real speech from synthetic speech.
- **Consent mechanisms**: Requiring explicit consent from the speaker before their voice can be cloned.
- **Regulatory frameworks**: Laws restricting the creation and distribution of deepfake audio (several jurisdictions have enacted or proposed such laws).

### 12.5 Efficiency and Edge Deployment

Running complete voice agent pipelines on edge devices (smartphones, smart speakers, laptops) requires efficient models for each component. While individual components can run on edge hardware (Whisper tiny for ASR, lightweight TTS models), running the complete pipeline including a capable LLM remains challenging without cloud connectivity. Progress in on-device LLMs (Gemma, Phi, Llama 3.2) is gradually enabling fully local voice agents.

### 12.6 Multimodal Voice Agents

The next frontier is voice agents that are not just audio-aware but truly multimodal — agents that can see (through cameras), hear (through microphones), and respond through speech and visual displays simultaneously. Gemini Live and GPT-4o's vision + voice capabilities represent early steps, but fully integrated multimodal voice agents that can participate in face-to-face-like interactions are still developing.

## 13. Conclusion

The speech and audio landscape has been transformed by the convergence with large language models. Whisper established that robust speech recognition could be achieved through massive supervised training. Neural codec models (EnCodec, SoundStream, Mimi) provided the discrete tokenization bridge between continuous audio and discrete language modeling. Language-model-based TTS (VALL-E, Bark, F5-TTS) demonstrated that natural speech synthesis could be formulated as a token generation problem. And speech-to-speech models (GPT-4o, Gemini Live, Moshi) showed that the cascade architecture could be replaced by unified models that process and generate speech natively.

The engineering of real-time voice agents — achieving sub-500ms latency while maintaining natural conversation dynamics — remains a significant challenge that requires careful optimization across every component of the pipeline, from voice activity detection through endpoint detection, ASR, LLM inference, and TTS. The deployment infrastructure (WebRTC, LiveKit, Pipecat) has matured to support production voice agents, and the open-source ecosystem provides increasingly capable components for building custom voice applications.

As voice becomes a primary interface for AI interaction — complementing text-based interfaces for hands-free, eyes-free, and accessibility contexts — the models and infrastructure described in this report will form the foundation of a new generation of AI-powered voice experiences.

## References

1. Radford, A., et al. "Robust Speech Recognition via Large-Scale Weak Supervision." ICML 2023.
2. Défossez, A., et al. "High Fidelity Neural Audio Compression." 2022.
3. Zeghidour, N., et al. "SoundStream: An End-to-End Neural Audio Codec." IEEE/ACM Transactions on Audio, Speech, and Language Processing 2022.
4. Wang, C., et al. "Neural Codec Language Models are Zero-Shot Text to Speech Synthesizers." 2023.
5. Wang, C., et al. "VALL-E 2: Neural Codec Language Models are Human Parity Zero-Shot Text to Speech Synthesizers." 2024.
6. Kumar, R., et al. "High-Fidelity Audio Compression with Improved RVQGAN." 2023.
7. Défossez, A., et al. "Moshi: a Speech-Text Foundation Model for Real-Time Dialogue." 2024.
8. Pratap, V., et al. "Scaling Speech Technology to 1,000+ Languages." 2023.
9. Barrault, L., et al. "SeamlessM4T: Massively Multilingual & Multimodal Machine Translation." 2023.
10. OpenAI. "GPT-4o System Card." 2024.
11. Gemini Team, Google. "Gemini: A Family of Highly Capable Multimodal Models." 2024.
12. Chu, W., et al. "Qwen2-Audio Technical Report." 2024.
13. Chen, E., et al. "F5-TTS: A Fairytaler that Fakes Fluent and Faithful Speech with Flow Matching." 2024.
14. Lyth, D., and King, S. "Natural Language Guidance of High-Fidelity Text-to-Speech with Synthetic Annotations." 2024.
15. Suno. "Bark: Text-Prompted Generative Audio Model." 2023.
