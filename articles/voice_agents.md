# Voice Agents and Real-Time Conversational AI

*April 2026*

## 1. Introduction

For most of the history of language model applications, the interface has been text. Users type messages, models generate text responses. This is natural for many use cases—coding assistants, document analysis, search—but it is not how humans prefer to communicate in real time. Humans talk. The spoken word is faster, more natural, and more expressive than typing. It is the default human communication modality in customer service, healthcare consultations, sales calls, and everyday interactions.

Voice agents—AI systems that converse with humans through spoken language in real time—represent the convergence of several technologies that have individually matured: automatic speech recognition (ASR), large language models (LLMs), text-to-speech synthesis (TTS), and real-time audio transport. Each component has reached a level of quality where the combined system can hold conversations that, for straightforward interactions, are competitive with or indistinguishable from human agents.

The technical challenge is latency. Human conversation has strict timing expectations. A pause of more than 500-700 milliseconds after a speaker finishes feels unnatural. A pause of more than 1-2 seconds feels broken. Building a system that listens to human speech, understands it, generates an intelligent response, and speaks that response within this latency budget requires careful engineering across every component of the pipeline.

This report provides a comprehensive technical examination of voice agents and real-time conversational AI. It covers the architecture of voice systems, the latency budget and how it constrains design, the pipeline approach versus speech-to-speech models, the major platform offerings, open-source voice stacks, turn-taking and interruption handling, function calling during voice conversations, telephony integration, cost analysis, safety considerations, evaluation, and practical deployment patterns. The intended audience is engineers building voice agents and product teams evaluating voice as an interaction modality.

## 2. The Latency Budget

### 2.1 Human Expectations

Human conversational turn-taking operates on tight timelines. Research on conversational dynamics shows:

- The average gap between speakers in natural conversation is approximately 200 milliseconds.
- Gaps of 400-700ms are perceived as slightly slow but acceptable.
- Gaps beyond 700ms are perceived as noticeably delayed.
- Gaps beyond 1.5 seconds are perceived as the system being broken or unresponsive.

These expectations are non-negotiable. A voice agent that gives perfect answers but takes 3 seconds to start responding will be perceived as worse than one that gives adequate answers in 500ms. Latency is not a performance metric for voice agents—it is a correctness requirement.

### 2.2 Anatomy of the Latency Budget

The total latency from when the user stops speaking to when the agent starts speaking comprises:

**Voice activity detection (VAD): 100-200ms.** The system must detect that the user has finished speaking. This requires waiting long enough to distinguish a pause (the user is thinking) from a turn ending (the user expects a response). Too aggressive detection causes the agent to interrupt the user. Too conservative detection adds unnecessary latency.

**Audio transmission: 20-50ms.** The audio must travel from the user's device to the server. For WebRTC connections, this is typically 20-50ms depending on network conditions. For telephony, it may be higher.

**Speech recognition: 50-300ms.** The ASR system transcribes the user's speech to text. Streaming ASR can provide partial results as the user speaks, but the final transcription (including corrections and refinements) takes additional time after the speech ends.

**LLM inference (time to first token): 200-500ms.** The language model processes the conversation context and begins generating a response. This is the time to the first generated token, not the complete response. Time to first token depends on the model size, the context length, and the inference infrastructure.

**Text-to-speech: 50-200ms.** The TTS system converts the first chunk of text into audio. Streaming TTS can begin synthesizing audio from the first few words rather than waiting for the complete response.

**Audio transmission back: 20-50ms.** The synthesized audio travels from the server to the user's device.

**Total: 440-1,300ms.** With optimized components and favorable network conditions, the total latency can be below 500ms. With typical conditions, 600-900ms is common. With suboptimal conditions, 1-2 seconds.

### 2.3 Optimizing the Budget

Every component must be optimized to meet the budget:

**Streaming everywhere.** The most important optimization is streaming at every stage. Do not wait for complete ASR output before starting LLM inference—stream partial transcripts. Do not wait for complete LLM output before starting TTS—stream tokens to TTS as they are generated. Do not wait for complete TTS audio before starting playback—stream audio chunks to the client.

**Parallel processing.** ASR, LLM, and TTS can partially overlap. While the last words of the user's speech are being transcribed, the LLM can begin processing the already-transcribed portion.

**Speculative processing.** If the system can predict what the user is likely to say (based on partial transcript and conversation context), it can begin LLM inference speculatively. If the prediction is wrong, the speculative output is discarded.

**Edge deployment.** Running VAD on the client device eliminates the round-trip for voice activity detection. Some systems also run ASR on the edge, sending transcribed text instead of audio.

**Model selection.** Smaller, faster models reduce LLM inference time. For voice agents, a model with 200ms time-to-first-token is much more valuable than a model with 500ms TTFT, even if the faster model is slightly less capable. This is why many voice agent deployments use smaller models (GPT-4o-mini, Claude 3.5 Haiku, Gemini Flash) rather than frontier models.

**TTS caching.** Common phrases ("How can I help you today?", "Let me look that up for you") can be pre-synthesized and cached, eliminating TTS latency for these responses.

### 2.4 The First-Word Problem

The user's perception of latency is dominated by the time to the first word of the response. Even if the complete response takes 5 seconds to generate and speak, the conversation feels responsive if the first word arrives in 500ms. This is why streaming is so critical—it decouples the perceived latency (time to first word) from the actual processing time (time to complete response).

However, this creates a challenge: the model must begin generating a response before it has fully "thought" about the answer. For complex questions, the first few words of the response ("That's a great question. Let me...") may be filler while the model generates the substantive content. This is actually similar to how humans behave—we often begin speaking before we have fully formulated our response, using filler phrases to hold the conversational floor.

## 3. Pipeline Architecture: ASR + LLM + TTS

### 3.1 Overview

The traditional voice agent architecture is a pipeline of three components:

```
User Speech → ASR → Text → LLM → Text → TTS → Agent Speech
```

Each component is a separate system, and they communicate through text. The ASR system converts speech to text, the LLM processes the text and generates a response, and the TTS system converts the response text back to speech.

### 3.2 ASR Component

**Whisper and derivatives.** OpenAI's Whisper (Radford et al., 2022) established a new quality baseline for ASR. It is trained on 680,000 hours of multilingual audio and achieves near-human accuracy on many benchmarks. Whisper is open-source and runs locally, making it suitable for both cloud and edge deployment.

However, Whisper was designed for batch transcription, not streaming. Adaptations for streaming include:
- **Whisper streaming**: Processing audio in chunks and providing incremental results.
- **Faster Whisper**: An optimized implementation using CTranslate2 that achieves 4x faster-than-real-time processing.
- **Distil-Whisper**: A distilled version that is 6x faster with minimal accuracy loss.

**Deepgram.** Deepgram provides a streaming ASR API optimized for real-time applications. It offers sub-300ms latency for final transcripts, speaker diarization (identifying who is speaking), and language detection. Deepgram is widely used in production voice agent systems due to its low latency and high accuracy.

**Google Speech-to-Text.** Google's STT service offers streaming recognition with intermediate results. The Chirp model (2023) and its successors provide high accuracy across many languages.

**AssemblyAI.** AssemblyAI provides streaming ASR with features tailored for voice agent applications: real-time transcription, speaker labels, and content moderation.

### 3.3 LLM Component

The LLM in a voice pipeline operates similarly to a text-based chatbot, with additional constraints:

**Response style.** Voice responses must be conversational. Written prose does not sound natural when spoken. The system prompt must instruct the model to use short sentences, avoid complex constructions, and write for the ear rather than the eye.

**Response length.** Long responses are problematic in voice. A 500-word response that takes 3 minutes to speak is exhausting to listen to. Voice agents should generate concise responses—typically 1-3 sentences per turn for informational responses, with longer responses broken into interactive exchanges.

**Time-to-first-token.** As discussed, TTFT is critical. Model selection often prioritizes speed over capability.

**Structured output for actions.** The model must be able to generate tool calls (to look up information, process transactions, etc.) while maintaining conversational flow. This requires the model to decide whether to respond verbally, call a tool, or both.

### 3.4 TTS Component

**ElevenLabs.** ElevenLabs provides high-quality, low-latency TTS with support for voice cloning and emotional expression. Their streaming API delivers first audio within 100-300ms of receiving text. ElevenLabs has become the default choice for many production voice agents due to its quality-latency balance.

**OpenAI TTS.** OpenAI offers TTS through their API with multiple voice options. The voices are natural-sounding and consistent, though with less emotional range than ElevenLabs.

**Google Cloud TTS.** Google's TTS offers WaveNet and Neural2 voices with good quality and multi-language support. The streaming API supports low-latency applications.

**Cartesia.** Cartesia provides ultra-low-latency TTS (under 100ms to first audio) optimized specifically for real-time voice applications. Their Sonic model achieves near-human naturalness with extremely fast synthesis.

**XTTS and open-source alternatives.** Coqui's XTTS, StyleTTS2, and other open-source TTS models provide viable self-hosted alternatives. Quality has improved dramatically, though they typically require more compute to match the quality of commercial offerings.

### 3.5 Pipeline Advantages

The pipeline approach has several advantages:

**Modularity.** Each component can be independently selected, upgraded, and optimized. You can swap ASR providers without changing the LLM or TTS.

**Flexibility.** The LLM operates on text, which means all text-based LLM capabilities work unchanged: function calling, RAG, system prompts, conversation management.

**Transparency.** The intermediate text representation is inspectable. You can log the ASR transcript and the LLM response independently, making debugging straightforward.

**Provider diversity.** You can use the best-in-class provider for each component: Deepgram for ASR, Anthropic for the LLM, ElevenLabs for TTS.

### 3.6 Pipeline Disadvantages

**Information loss.** Converting speech to text discards paralinguistic information—tone, emphasis, emotion, speaking rate, hesitation. A user who says "I'm fine" sarcastically sounds the same in text as one who says it genuinely. The LLM cannot respond to emotional cues it does not receive.

**Error propagation.** ASR errors propagate through the pipeline. If the ASR misrecognizes a word, the LLM processes the wrong text. If the LLM generates text that is difficult for TTS (unusual names, technical terms, mixed languages), the TTS may mispronounce it.

**Latency accumulation.** Each component adds latency. The total latency is the sum of all components, and each component's latency is bounded by physics and engineering constraints.

**Unnatural prosody.** TTS generates speech from text without knowledge of the conversational context, leading to prosody (rhythm and intonation) that may not match the conversational situation. A response to a distressed user might be spoken with the same cheerful tone as a response to a casual question.

## 4. Speech-to-Speech Models

### 4.1 The Native Audio Approach

Instead of converting speech to text, processing text, and converting back to speech, speech-to-speech models process audio natively. The model receives audio tokens (representations of the user's speech) and generates audio tokens (representations of the agent's speech) directly, without an intermediate text representation.

### 4.2 GPT-4o Audio

GPT-4o (May 2024) was the first frontier model to support native audio input and output. It processes audio through:

1. **Audio tokenization.** The user's speech is converted to audio tokens by an audio encoder. These tokens capture not just the words but also the prosody, tone, and emotional content.

2. **Multi-modal processing.** The audio tokens are processed by the same transformer that handles text and image tokens. The model can attend to audio features alongside text context.

3. **Audio generation.** The model generates audio tokens directly, which are decoded into speech by an audio decoder.

This native approach preserves paralinguistic information—the model can "hear" the user's tone and respond with appropriate emotional tone in its own voice. It can also generate non-verbal sounds (laughter, hesitation, emphasis) that pipeline systems cannot.

### 4.3 GPT-4o Realtime API

The GPT-4o Realtime API (October 2024) provides WebSocket-based access to native audio capabilities for building voice agents:

**WebSocket connection.** The client establishes a WebSocket connection to the Realtime API. All communication happens through this persistent connection, eliminating per-request HTTP overhead.

**Audio streaming.** The client streams audio to the server in real-time. The server processes the audio incrementally and streams audio responses back.

**Server-side VAD.** The API includes built-in voice activity detection. The server detects when the user starts and stops speaking, handles turn-taking, and manages interruptions.

**Function calling.** The model can call functions during a voice conversation. When the model determines that it needs external information, it emits a function call event. The client executes the function, returns the result, and the model incorporates it into its spoken response.

**Event-based protocol.** The WebSocket protocol uses JSON events:

```json
// Client sends audio
{"type": "input_audio_buffer.append", "audio": "<base64 audio data>"}

// Server detects speech end
{"type": "input_audio_buffer.speech_stopped"}

// Server sends response audio
{"type": "response.audio.delta", "delta": "<base64 audio data>"}

// Server requests function call
{"type": "response.function_call_arguments.done", 
 "name": "lookup_order", 
 "arguments": "{\"order_id\": \"12345\"}"}
```

**Session configuration.** The session is configured with:
- Voice selection (alloy, echo, shimmer, etc.)
- System prompt (instructions for the agent)
- Tools (function definitions for tool calling)
- Turn detection settings (sensitivity, silence threshold)
- Temperature and other generation parameters

### 4.4 Gemini Live

Google's Gemini Live provides native voice conversation capabilities with Gemini models:

**Multimodal native processing.** Like GPT-4o, Gemini models process audio natively. The model can understand and generate speech directly.

**Multi-modal context.** Gemini Live can process audio alongside other modalities. A user can share their screen while talking, and the model can see what they see and hear what they say simultaneously.

**Streaming API.** Google provides a streaming API for building voice applications with Gemini. The API supports bidirectional audio streaming, server-side VAD, and function calling.

**Integration with Google services.** Gemini Live integrates with Google Search, Maps, and other Google services, allowing voice agents to access real-time information during conversations.

### 4.5 Speech-to-Speech Advantages

**Emotional understanding.** The model processes the full audio signal, including emotional cues. It can detect frustration, confusion, urgency, or satisfaction in the user's voice and respond appropriately.

**Natural prosody.** Generated speech has prosody that matches the conversational context because the model generates audio directly rather than converting text to speech.

**Lower latency potential.** By eliminating the ASR and TTS stages, native audio processing can reduce end-to-end latency. The model processes audio tokens directly, without the overhead of text conversion.

**Non-verbal communication.** The model can generate and understand non-verbal audio—laughter, sighs, emphasis, pauses—that enrich the conversation.

### 4.6 Speech-to-Speech Disadvantages

**Limited model options.** As of April 2026, only GPT-4o and Gemini offer native audio capabilities at scale. This means less provider diversity and no self-hosting options for the most capable models.

**Opacity.** Without an intermediate text representation, it is harder to inspect and debug what the model understood and what it generated. Logging audio is more expensive and less searchable than logging text.

**Hallucinated audio.** Models can generate speech that sounds confident but is incorrect, and the lack of a text intermediate means there is no easy point to inject fact-checking or validation.

**Cost.** Native audio models are more expensive than pipeline approaches because audio token processing consumes more compute than text processing. The GPT-4o Realtime API charges $100 per million input audio tokens and $200 per million output audio tokens (as of late 2025 pricing), making voice conversations significantly more expensive than text conversations.

**Content safety.** It is harder to apply content filters to audio than to text. Text-based content moderation is well-understood; audio content moderation is less mature.

## 5. Open-Source Voice Stacks

### 5.1 Pipecat

Pipecat (developed by Daily) is an open-source framework for building voice and multimodal AI agents. It provides:

**Pipeline architecture.** Pipecat implements the pipeline pattern with modular, composable components. You construct a pipeline by connecting processors:

```python
pipeline = Pipeline([
    transport.input(),       # Audio from WebRTC/telephony
    stt.processor(),         # Speech-to-text (Deepgram, Whisper, etc.)
    llm.processor(),         # Language model (OpenAI, Anthropic, etc.)
    tts.processor(),         # Text-to-speech (ElevenLabs, Cartesia, etc.)
    transport.output(),      # Audio back to user
])
```

**Transport layer.** Pipecat supports multiple transport protocols: WebRTC (through Daily), WebSocket, and telephony (through Twilio and other providers). This abstraction means the same agent logic works whether the user is connected through a web browser, a phone call, or a custom application.

**Processor ecosystem.** Pipecat has pre-built processors for most major ASR, LLM, and TTS providers. Adding a new provider requires implementing a small adapter, not rewriting the pipeline.

**Frame-based processing.** Data flows through the pipeline as "frames"—typed data units that carry audio, text, tool calls, or control signals. This design enables clean separation of concerns and supports streaming at every stage.

**Interruption handling.** Pipecat includes built-in support for barge-in (the user interrupting the agent mid-response). When a user starts speaking while the agent is talking, Pipecat cancels the current response, stops audio playback, and routes the user's speech to the ASR for processing.

### 5.2 LiveKit Agents

LiveKit Agents is an open-source framework built on LiveKit's real-time communication infrastructure:

**WebRTC-native.** LiveKit is a WebRTC platform, so LiveKit Agents inherits low-latency, peer-to-peer-style audio transport. Audio quality and latency are excellent.

**Worker-based architecture.** Agent logic runs in "workers" that can be scaled independently. Each worker handles one or more concurrent voice sessions. This architecture supports production deployment with load balancing and failover.

**Plugin system.** Similar to Pipecat's processors, LiveKit Agents uses plugins for STT, LLM, and TTS providers. Plugins handle the provider-specific integration, and the framework handles the orchestration.

**Multi-modal support.** LiveKit Agents supports video and screen sharing alongside voice, enabling agents that can see what the user sees while conversing with them.

**Telephony bridge.** LiveKit provides a SIP bridge for connecting voice agents to traditional phone networks, enabling deployment as phone-based agents.

### 5.3 Vocode

Vocode provides an open-source platform for voice agent development with a focus on telephony applications:

**Conversation management.** Vocode models voice conversations as state machines with defined states, transitions, and actions. This makes it easier to build structured conversations (menus, decision trees, multi-step workflows) that are common in telephony applications.

**Telephony-first.** While Vocode supports web-based voice (WebRTC), its primary focus is telephony. It integrates natively with Twilio, Vonage, and other telephony providers.

**Self-hosted.** Vocode is designed for self-hosted deployment, giving developers full control over data flow and privacy—important for applications that handle sensitive information (healthcare, financial services).

### 5.4 Comparison of Frameworks

| Feature | Pipecat | LiveKit Agents | Vocode |
|---|---|---|---|
| Primary transport | WebRTC (Daily) | WebRTC (LiveKit) | Telephony (Twilio) |
| Architecture | Pipeline/frames | Worker/plugins | State machine |
| Streaming support | Full | Full | Full |
| Interruption handling | Built-in | Built-in | Built-in |
| Telephony support | Via Twilio adapter | Via SIP bridge | Native |
| Multi-modal | Audio + video | Audio + video + screen | Audio |
| Deployment model | Self-hosted | Self-hosted or cloud | Self-hosted |
| Maturity | High | High | Medium |

## 6. Turn-Taking and Interruption Handling

### 6.1 The Turn-Taking Problem

Human conversation has complex turn-taking dynamics. Speakers signal when they are about to finish (declining intonation, completion of a syntactic unit, a pause). Listeners signal when they want to speak (inhaling, slight vocalization, body language). These signals are processed unconsciously and enable smooth turn transitions.

Voice agents must handle turn-taking without most of these cues. They have access to audio only—no body language, no visual signals. The primary mechanism is voice activity detection (VAD), which determines when the user is speaking and when they have stopped.

### 6.2 Voice Activity Detection

VAD is the first and most critical component of turn-taking. It must answer two questions:

1. **Is the user speaking?** Distinguish speech from background noise, music, TV audio, and other non-speech sounds.
2. **Has the user finished their turn?** Distinguish a mid-turn pause (the user is thinking) from a turn ending (the user expects a response).

**Energy-based VAD.** The simplest approach: speech is louder than silence. Threshold on audio energy (volume). This fails in noisy environments and cannot distinguish speech from other sounds.

**Model-based VAD.** Trained neural networks that classify audio frames as speech or non-speech. Silero VAD is the most widely used model-based VAD in voice agent applications. It provides frame-level speech probability with low latency and high accuracy.

**Endpointing.** The decision that the user has finished speaking is called endpointing. The endpoint is typically triggered by a silence duration threshold—if the user has been silent for N milliseconds, assume they have finished. The threshold is a critical tuning parameter:

- **Too short (200-300ms)**: The agent interrupts the user during natural pauses. Extremely annoying and breaks conversational flow.
- **Too long (1000-1500ms)**: The agent waits too long to respond, creating an unnatural gap.
- **Typical (400-700ms)**: A compromise that works for most conversations.

Some systems use adaptive endpointing—adjusting the threshold based on the conversation context. During a multi-part question, the threshold might be longer (allowing the user to pause between parts). During rapid back-and-forth, the threshold might be shorter.

### 6.3 Barge-In (User Interruption)

Barge-in occurs when the user starts speaking while the agent is still talking. This is common and normal in human conversation—a user might interrupt to:

- Correct a misunderstanding
- Provide additional information
- Express agreement or disagreement
- Skip a long-winded response

Handling barge-in requires:

1. **Detection**: While the agent is speaking, continue monitoring the audio for user speech. This is tricky because the agent's audio may be playing on the user's device and being picked up by the microphone (echo).

2. **Echo cancellation**: If the user's device plays the agent's audio through speakers (not headphones), the agent's voice will be picked up by the microphone. The system must separate the user's speech from the echo of the agent's speech. WebRTC includes acoustic echo cancellation (AEC) for this purpose, but it is not perfect.

3. **Response cancellation**: When barge-in is detected, the agent's current response must be stopped. This means stopping TTS generation, clearing the audio output buffer, and stopping playback.

4. **Context update**: The agent's conversation context must be updated to reflect that its response was interrupted. The context should include only the portion of the response that was actually spoken before the interruption.

5. **Processing the interruption**: The user's interrupting speech is processed normally through ASR and LLM, and a new response is generated that takes the interruption into account.

### 6.4 Agent Interruption (Proactive Speaking)

In some scenarios, the agent needs to speak proactively—not in response to the user's speech but due to an event:

- A timer expires (appointment reminder)
- A tool call completes (results are ready)
- A condition is met (an order status changes)

Proactive speaking requires checking whether the user is currently speaking and, if so, either waiting or using a polite interruption pattern ("Excuse me, I have an update on your order...").

### 6.5 Backchannel Signals

In human conversation, listeners provide backchannel signals—"uh-huh," "mmm," "right," "okay"—that indicate they are listening and understanding. These signals are important for conversational fluency but are challenging for voice agents:

- The agent must generate backchannels at appropriate times without interrupting the user.
- Backchannels should not trigger the user's VAD or endpointing logic.
- The timing must be natural—too frequent is annoying, too infrequent makes the agent seem unresponsive.

Some advanced voice agent implementations generate backchannels during the user's speech, but this remains an active area of development. Native audio models (GPT-4o, Gemini) are better positioned to handle backchannels because they can generate audio responses alongside the user's speech.

## 7. Emotional Tone and Prosody

### 7.1 Why Tone Matters

Voice communication is inherently emotional. The same words spoken with different tones convey very different meanings. A customer service agent that delivers bad news ("Unfortunately, your order has been delayed") in a cheerful, upbeat tone creates a jarring experience. A healthcare agent that discusses sensitive health information in a flat, robotic monotone feels uncaring.

### 7.2 Controlling TTS Emotion

TTS systems offer varying degrees of emotional control:

**Voice selection.** Different TTS voices have different base emotional characteristics. Some voices sound warm and empathetic; others sound professional and authoritative. Selecting the right base voice for the application is the first step.

**Explicit emotion tags.** Some TTS systems support emotional markup—tags that specify the desired emotion for a passage:

```xml
<speak>
  <prosody rate="slow" pitch="-2st">
    I'm sorry to hear about that. Let me see what I can do to help.
  </prosody>
</speak>
```

SSML (Speech Synthesis Markup Language) provides a standard for this, though support varies by provider.

**Implicit emotion from text.** More capable TTS models infer appropriate emotion from the text content. "I'm so excited to share this with you!" is spoken with enthusiasm without explicit markup. The quality of this inference varies significantly between providers.

**Model-based emotional control.** Native audio models (GPT-4o) can be instructed about emotional tone through the system prompt: "Speak in a warm, empathetic tone when delivering bad news." The model adjusts its audio output accordingly, though the degree of control is limited.

### 7.3 Detecting User Emotion

Pipeline voice agents (ASR + LLM + TTS) lose emotional information during ASR transcription. Native audio models can detect emotion directly from the audio signal.

For pipeline approaches, emotion can be partially recovered through:
- **Sentiment analysis** on the transcribed text
- **Acoustic feature extraction** (pitch, energy, speaking rate) from the raw audio, passed to the LLM alongside the transcript
- **Dedicated emotion detection models** that classify the user's emotional state from audio

### 7.4 Emotional Consistency

The agent's emotional responses should be consistent with the conversation context. If the user has been frustrated throughout the conversation, the agent should maintain an empathetic, patient tone—not suddenly switch to cheerful. This requires the LLM to track emotional context and generate responses with appropriate emotional markers for the TTS system.

## 8. WebRTC for Real-Time Audio Transport

### 8.1 Why WebRTC

WebRTC (Web Real-Time Communication) is the standard protocol for real-time audio and video communication in browsers. It was designed for video calling and has properties that make it ideal for voice agent applications:

**Low latency.** WebRTC uses UDP transport with mechanisms for handling packet loss and jitter. Typical one-way latency is 20-100ms, far lower than HTTP-based approaches.

**Echo cancellation.** WebRTC includes built-in acoustic echo cancellation, which is essential for full-duplex voice communication where the agent's speech may be picked up by the user's microphone.

**Noise suppression.** Built-in noise suppression filters out background noise from the user's environment.

**Automatic bandwidth adaptation.** WebRTC adjusts audio quality based on network conditions, maintaining low latency even on poor connections by sacrificing some audio quality.

**Browser native.** WebRTC is supported in all modern browsers without plugins, making it easy to deploy voice agents in web applications.

### 8.2 Architecture

A typical WebRTC-based voice agent architecture:

```
User's Browser
  ├── Microphone → WebRTC Audio Track → 
  └── Speaker   ← WebRTC Audio Track ←
                        |
                   WebRTC Server (SFU)
                        |
                   Voice Agent Server
                     ├── VAD
                     ├── ASR
                     ├── LLM
                     └── TTS
```

The user's browser captures microphone audio and sends it to a WebRTC server (Selective Forwarding Unit or SFU). The SFU forwards the audio to the voice agent server, which processes it through the pipeline and sends synthesized audio back through the SFU to the user's browser.

Services like Daily, LiveKit, and Agora provide the WebRTC infrastructure (SFU, TURN servers for NAT traversal, signaling), allowing developers to focus on the agent logic.

### 8.3 WebSocket Alternative

Some voice agent implementations use WebSockets instead of WebRTC for audio transport. WebSockets operate over TCP, which provides reliable delivery but with higher latency than WebRTC's UDP transport:

**Advantages of WebSocket**: Simpler to implement, works through firewalls without TURN servers, reliable delivery (no packet loss).

**Disadvantages of WebSocket**: Higher latency due to TCP head-of-line blocking, no built-in echo cancellation or noise suppression, no automatic bandwidth adaptation.

The GPT-4o Realtime API uses WebSockets, which provides simplicity at the cost of slightly higher latency compared to a WebRTC implementation.

## 9. Telephony Integration

### 9.1 Why Telephony Matters

Despite the ubiquity of web and mobile applications, the telephone remains a primary communication channel for many interactions: customer service, appointment booking, healthcare, government services, sales, and emergency services. Voice agents that can operate over phone calls have immediate practical value.

### 9.2 SIP and PSTN

Phone calls are routed through two networks:

**PSTN (Public Switched Telephone Network).** The traditional phone network that connects landlines and cell phones. Audio is encoded at 8kHz (narrowband) with G.711 or similar codecs. Quality is limited compared to VoIP.

**SIP (Session Initiation Protocol).** The protocol used for VoIP (Voice over IP) calls. SIP establishes call sessions, and RTP (Real-time Transport Protocol) carries the audio. Quality depends on the codec—G.711 (8kHz) for interoperability, Opus (up to 48kHz) for high quality.

Voice agents connect to these networks through telephony providers (Twilio, Vonage, Telnyx, Bandwidth) that provide:

- Phone numbers (local, toll-free, or international)
- Inbound call handling (answering incoming calls and routing them to the agent)
- Outbound calling (the agent initiates calls)
- SIP trunking (connecting the agent to the SIP network)
- Call recording and analytics

### 9.3 Twilio Integration

Twilio is the most widely used telephony provider for voice agents. The integration works through:

**TwiML (Twilio Markup Language).** When a call comes in, Twilio requests instructions from a webhook URL. The server responds with TwiML that tells Twilio what to do: play a greeting, connect to a WebSocket for streaming audio, etc.

**Media Streams.** Twilio's Media Streams feature provides real-time, bidirectional audio streaming over WebSocket. The voice agent receives raw audio from the caller and sends synthesized audio back:

```
Caller → PSTN → Twilio → WebSocket (audio stream) → Voice Agent
Caller ← PSTN ← Twilio ← WebSocket (audio stream) ← Voice Agent
```

**Call control.** Twilio provides APIs for call control: transferring to a human agent, placing the caller on hold, conferencing multiple parties, and recording the call.

### 9.4 Vonage, Telnyx, and Others

Other telephony providers offer similar capabilities:

**Vonage** (now part of Ericsson) provides WebSocket-based audio streaming with its Voice API. It offers features similar to Twilio with different pricing and geographic coverage.

**Telnyx** provides a developer-focused telephony platform with WebSocket audio streaming, SIP trunking, and competitive pricing for high-volume applications.

**Bandwidth** provides direct carrier connectivity, which can offer lower latency and better call quality than providers that aggregate across multiple carriers.

### 9.5 Telephony-Specific Challenges

Voice agents operating over telephone networks face additional challenges:

**Audio quality.** Narrowband phone audio (8kHz) has limited frequency range compared to wideband VoIP (16kHz+). ASR accuracy is lower on narrowband audio, particularly for certain phonemes.

**Latency.** Phone networks add 50-150ms of latency beyond what IP networks add. This makes the latency budget tighter.

**DTMF (touch-tone) handling.** Callers may press phone keys during a conversation (to select menu options, enter account numbers, etc.). The agent must detect and interpret DTMF tones.

**Call quality variation.** Cell phone calls, VoIP calls, and landline calls have different audio characteristics. Background noise, compression artifacts, and connection quality vary widely.

**Regulatory compliance.** Phone-based agents are subject to telecommunications regulations: call recording consent, do-not-call lists, emergency service requirements, and data protection laws that vary by jurisdiction.

## 10. Function Calling During Voice Conversations

### 10.1 The Challenge

Function calling in voice conversations introduces a unique challenge: the agent must call tools (to look up information, process transactions, etc.) while maintaining conversational flow. In a text conversation, the user sees a loading indicator while the tool executes. In a voice conversation, silence during tool execution feels like the connection dropped.

### 10.2 Handling Tool Execution Latency

When the model decides to call a tool, there is a gap while the tool executes. Strategies for filling this gap:

**Verbal acknowledgment.** The agent speaks a filler phrase before executing the tool: "Let me look that up for you..." or "One moment while I check on that." This sets the user's expectation that a response is coming.

**Background execution.** The tool call is initiated while the agent is still speaking its acknowledgment. By the time the agent finishes the filler phrase, the tool result may already be available.

**Streaming integration.** Some frameworks allow the model to begin generating a response while the tool call is in progress, then incorporate the tool result when it arrives. This requires careful coordination to ensure the response is coherent.

### 10.3 Multi-Tool Conversations

Complex voice agent tasks may require multiple tool calls in sequence. A flight booking conversation might require:

1. Search for flights (tool call)
2. Present options to the user (speech)
3. User selects an option (speech, ASR)
4. Check seat availability (tool call)
5. Present seat options (speech)
6. User selects a seat (speech, ASR)
7. Process payment (tool call)
8. Confirm booking (speech)

Each tool call introduces latency that must be managed with verbal fillers and expectations setting. The agent must maintain conversational context across all these steps, remembering what the user said earlier in the conversation and what information was returned by previous tool calls.

### 10.4 Parallel Tool Execution in Voice

When multiple independent tool calls are needed, executing them in parallel reduces total latency. While the agent speaks about the first topic, the tool results for the second topic are being fetched in the background. This requires the agent to plan its verbal responses around the expected availability of tool results.

### 10.5 Error Handling in Voice

Tool failures are particularly problematic in voice conversations. In a text interface, an error message can be displayed and the user can decide what to do. In a voice conversation, the agent must explain the error conversationally and propose alternatives:

- "I'm having trouble looking up your order right now. Could you give me your order number again?"
- "The system is taking longer than expected. Would you like me to keep trying or would you prefer to call back?"

The agent must be prepared to handle tool failures gracefully without losing the conversational thread.

## 11. Voice Agent Memory

### 11.1 Conversation Context

Voice agents must maintain context throughout a conversation, which may last from 30 seconds to 30 minutes or more. The context includes:

- What the user has said (transcripts of all user turns)
- What the agent has said (all agent responses)
- Tool call results (information retrieved during the conversation)
- Identified user information (name, account number, preferences)
- Conversation state (what topic we are discussing, what decisions have been made)

### 11.2 Context Window Management

Long voice conversations accumulate significant context. A 10-minute conversation at typical speaking rates produces approximately 1,500 words of transcript—not counting agent responses, tool results, system prompts, and tool definitions. A 30-minute conversation can approach or exceed the context window of smaller models.

Strategies for managing context:

**Summarization.** Periodically summarize the conversation history, replacing detailed transcripts with concise summaries. This reduces token usage but loses detail.

**Sliding window.** Keep the most recent N turns in full detail and summarize older turns. This preserves recent context while managing total size.

**Selective retention.** Keep key information (user identity, decisions made, important facts) in a structured format and discard the verbatim transcript of older turns.

### 11.3 Cross-Session Memory

Some voice agents need to remember information across multiple conversations:

- A user who called last week about an issue should not have to re-explain the problem.
- Preferences expressed in previous conversations should be remembered.
- The outcome of previous interactions should be available.

Cross-session memory is typically implemented through:

**User profiles.** A database of user information that is loaded into the agent's context at the start of each conversation.

**Conversation summaries.** Summaries of previous conversations stored in the user profile and provided as context.

**Memory systems.** More sophisticated memory systems that extract and store facts, preferences, and interaction patterns from conversations. These are loaded selectively based on relevance to the current conversation.

### 11.4 Memory and Privacy

Voice conversations often contain sensitive personal information—account numbers, health information, financial details, personal circumstances. Memory systems must:

- Comply with data protection regulations (GDPR, HIPAA, CCPA)
- Provide clear disclosure about what is remembered
- Allow users to request deletion of their data
- Encrypt stored information
- Limit access to authorized systems

## 12. Cost Analysis

### 12.1 Per-Minute Cost Breakdown

The cost of running a voice agent includes multiple components:

**ASR cost:** $0.003-0.01 per minute of audio (varies by provider; Deepgram charges approximately $0.0043/min for their Nova model)

**LLM cost:** $0.01-0.10 per minute of conversation (varies dramatically with model, context size, and response length; a conversation minute generates roughly 150-300 tokens of input and 100-200 tokens of output per turn, with 5-10 turns per minute)

**TTS cost:** $0.005-0.02 per minute of generated speech (varies by provider and voice quality; ElevenLabs charges approximately $0.018/min at scale)

**Telephony cost:** $0.005-0.02 per minute for PSTN connectivity (Twilio charges approximately $0.013/min for inbound calls)

**Infrastructure cost:** $0.002-0.01 per minute for compute, networking, and WebRTC/SIP infrastructure

**Total pipeline cost: approximately $0.03-0.15 per minute** for a typical pipeline agent with a mid-tier LLM.

### 12.2 Native Audio Model Cost

The GPT-4o Realtime API pricing (as of late 2025) is significantly higher than the pipeline approach:

- Input audio: $100 per million tokens (approximately $0.06/min of speech)
- Output audio: $200 per million tokens (approximately $0.12/min of speech)
- Plus text token costs for system prompts, tool definitions, and function calls

**Total native audio cost: approximately $0.20-0.40 per minute.**

This is 2-10x more expensive than the pipeline approach, which is a significant consideration for high-volume applications.

### 12.3 Cost Comparison with Human Agents

Human customer service agents cost approximately $0.30-1.00 per minute when fully loaded (salary, benefits, training, management, facilities, technology). Voice AI agents at $0.03-0.15 per minute represent a 70-95% cost reduction for tasks they can handle autonomously.

However, this comparison is simplistic. Human agents handle complex, ambiguous, and emotionally sensitive situations that AI agents cannot. The realistic deployment model is a hybrid: AI agents handle routine interactions (60-80% of volume) and escalate complex cases to human agents. The blended cost depends on the escalation rate.

### 12.4 Cost Optimization Strategies

**Model selection.** Use the smallest model that meets quality requirements. For many voice agent tasks, GPT-4o-mini, Claude 3.5 Haiku, or Gemini Flash provide adequate quality at 5-10x lower cost than frontier models.

**Context management.** Aggressively manage conversation context to reduce input token counts. Summarize history, prune irrelevant information, and keep tool definitions minimal.

**TTS caching.** Cache commonly spoken phrases (greetings, hold messages, error messages, closing statements). For agents with structured scripts, a significant fraction of speech can be pre-synthesized.

**Turn optimization.** Design conversations to minimize turns. Proactively provide information the user is likely to need rather than waiting for them to ask. This reduces the number of model calls per conversation.

**Batching tool calls.** When multiple pieces of information are needed, retrieve them in a single tool call rather than making separate calls. This reduces the number of LLM turns spent on tool execution.

## 13. Voice Cloning and Safety

### 13.1 Voice Cloning Technology

Modern TTS systems can clone voices from short audio samples—sometimes as little as 10-30 seconds of reference audio. This enables voice agents to speak in specific voices: a company executive, a brand mascot, or a custom-designed voice identity.

The technology works by:
1. Analyzing the reference audio to extract speaker characteristics (timbre, pitch range, speaking style).
2. Conditioning the TTS model on these characteristics.
3. Generating new speech that sounds like the reference speaker but says new content.

### 13.2 Applications

**Brand voice.** Companies can create a consistent voice identity for their AI agents, distinct from generic TTS voices. This voice becomes part of the brand, like a logo or color scheme.

**Accessibility.** People who have lost their voice due to medical conditions can use voice cloning to create a TTS voice that sounds like they did before, enabling more natural communication.

**Content creation.** Voice cloning enables automated content production (audiobooks, podcasts, training materials) in specific voices.

### 13.3 Safety Concerns

Voice cloning raises significant safety concerns:

**Impersonation and fraud.** Cloned voices can be used for social engineering attacks—calling a company's finance department using the CEO's cloned voice to authorize a wire transfer. Several high-profile cases of voice-cloning fraud have been reported.

**Misinformation.** Fake audio recordings of public figures saying things they never said can be used for political manipulation or reputational damage.

**Consent.** Using someone's voice without their consent raises ethical and legal issues. Many jurisdictions are developing regulations around voice cloning and synthetic media.

### 13.4 Safeguards

Responsible voice cloning requires:

**Consent verification.** Before cloning a voice, verify that the person whose voice is being cloned has given explicit consent. ElevenLabs, OpenAI, and other providers require consent verification for voice cloning.

**Watermarking.** Embed imperceptible audio watermarks in cloned speech that identify it as synthetic. This enables detection of cloned audio after distribution.

**Usage restrictions.** Limit what cloned voices can be used for. Prohibit use for impersonation, fraud, or deception.

**Detection tools.** Develop and deploy tools that can detect synthetic speech. This is an ongoing arms race between generation and detection capabilities.

**Disclosure.** In many jurisdictions, regulations require disclosure when a human is speaking with an AI agent. Voice agents that use cloned voices should identify themselves as AI at the start of the conversation.

## 14. Evaluation

### 14.1 Latency Metrics

**Time to first byte (TTFB).** The time from when the user stops speaking to when the first audio byte of the response is available at the server. This measures the processing pipeline latency without network transport.

**Time to first word.** The time from when the user stops speaking to when the first word of the response is audible to the user. This is the metric the user perceives and includes both processing and transport latency.

**End-to-end latency.** The time from when the user stops speaking to when the agent starts speaking, as measured at the user's device. This is the definitive latency metric.

**P50, P90, P99 latency.** Because latency varies across turns, percentile metrics are more informative than averages. A system with 500ms P50 latency but 3-second P99 latency has an inconsistent user experience—most turns are fast, but some are painfully slow.

### 14.2 Conversation Quality Metrics

**Task completion rate.** For task-oriented agents, what fraction of conversations result in successful task completion?

**Escalation rate.** What fraction of conversations are escalated to a human agent? High escalation rates indicate that the agent cannot handle the task distribution.

**User satisfaction.** Post-conversation surveys (CSAT, NPS) measure user perception. These are the ultimate quality metric but are expensive to collect and subject to response bias.

**Conversation efficiency.** How many turns does it take to complete a task? Fewer turns generally means a better experience (the user gets what they need faster).

**Error rate.** What fraction of agent responses contain errors? This includes factual errors, tool use errors, and misunderstandings of the user's intent.

### 14.3 ASR Quality Metrics

**Word Error Rate (WER).** The fraction of words that are incorrectly transcribed. WER is the standard ASR metric. Modern ASR systems achieve 5-10% WER on clean speech and 10-20% on noisy or accented speech.

**Real-time factor.** The ratio of processing time to audio duration. A real-time factor of 0.5 means the ASR processes audio twice as fast as real-time—important for streaming applications.

### 14.4 TTS Quality Metrics

**Mean Opinion Score (MOS).** Human listeners rate the naturalness of synthesized speech on a 1-5 scale. Top TTS systems achieve MOS of 4.0-4.5, compared to 4.5-4.8 for natural human speech.

**Character Error Rate in generated speech.** Some TTS systems occasionally mispronounce words or generate garbled audio. Measuring these errors automatically requires running the TTS output back through ASR and comparing to the input text.

### 14.5 End-to-End Evaluation

The most meaningful evaluation of a voice agent is end-to-end: have real or simulated users interact with the agent and measure outcomes.

**Simulated user testing.** Use a separate LLM to play the role of the user, conducting full voice conversations with the agent. The simulated user follows a script or persona and evaluates the agent's responses. This is cheaper and more scalable than human testing, though it may not capture all aspects of real user behavior.

**Human evaluation.** Recruit test users to interact with the agent on realistic tasks. Measure task completion, user satisfaction, and conversation quality. This is expensive but provides the most realistic assessment.

**A/B testing.** Deploy two agent variants to different user segments and compare metrics (task completion, satisfaction, escalation rate). This is the gold standard for evaluating changes to a production agent.

## 15. Practical Deployment Patterns

### 15.1 Customer Service

The most common voice agent deployment is customer service. The typical architecture:

1. Customer calls a phone number.
2. The IVR (Interactive Voice Response) system greets the caller and determines intent.
3. If the intent matches a category the AI agent can handle, the call is routed to the voice agent.
4. The voice agent handles the interaction, using tools to look up account information, process requests, and take actions.
5. If the agent cannot resolve the issue, it escalates to a human agent, providing a summary of the conversation so far.

Key design considerations:
- **Greeting and expectation-setting**: Clearly identify the agent as AI. Set expectations about what it can help with.
- **Authentication**: Verify the caller's identity through account number, security questions, or other mechanisms.
- **Scope definition**: Clearly define what the agent can and cannot do. Handle out-of-scope requests gracefully.
- **Escalation**: Make it easy for the user to reach a human agent if needed. Never force the user to stay in the AI loop.

### 15.2 Appointment Booking

Appointment booking is a high-value use case because it is structured (fixed set of information to collect) and high-volume (many businesses handle dozens to hundreds of booking calls per day):

1. Collect patient/client name.
2. Determine the type of appointment.
3. Check availability (tool call to scheduling system).
4. Present available slots.
5. Confirm the booking.
6. Send confirmation (email/SMS).

The structured nature of this task makes it well-suited for voice agents—the conversation follows a predictable pattern with limited branching.

### 15.3 Outbound Calling

Voice agents can make outbound calls for:
- Appointment reminders
- Payment reminders
- Survey collection
- Follow-up on service interactions
- Lead qualification

Outbound calling has additional considerations:
- **Regulatory compliance**: Many jurisdictions have strict rules about automated calls (TCPA in the US, GDPR in the EU). Consent, time-of-day restrictions, and do-not-call lists must be respected.
- **Answering machine detection**: The agent must determine whether a human or an answering machine answered. Strategies differ for each.
- **Call outcome tracking**: Track whether the call connected, whether the human engaged, and whether the objective was achieved.

### 15.4 Voice-Enabled Applications

Voice can be added to existing applications as an alternative interface:

- A banking app that allows users to check balances and transfer money by speaking.
- A smart home controller that processes natural language voice commands.
- A healthcare portal that allows patients to ask questions about their records.

The voice interface complements rather than replaces the visual interface, providing convenience for hands-free or eyes-free situations.

### 15.5 Internal Enterprise Agents

Voice agents within enterprises can handle:
- IT help desk inquiries
- HR policy questions
- Meeting scheduling
- CRM data entry (updating records by describing changes verbally)
- Status updates and reporting

Internal deployments often have more relaxed latency requirements (users are more tolerant of delay when using an internal tool) but stricter privacy requirements (enterprise data must not leave the corporate network).

## 16. Architecture Decisions

### 16.1 Pipeline vs. Native: When to Choose What

**Choose the pipeline approach when:**
- You need to use a specific LLM that does not support native audio (Claude, most open-weight models)
- Cost is a primary concern (pipeline is 2-10x cheaper)
- You need full control over each component (custom ASR, custom TTS)
- Transparency and debuggability are important (text transcripts at every stage)
- You are deploying in a regulated environment that requires text logging

**Choose native audio when:**
- Emotional understanding is critical to the use case
- Natural conversational flow and prosody are important
- You want the simplest possible architecture
- Latency must be minimized (eliminating ASR and TTS stages)
- The higher cost is acceptable

### 16.2 Self-Hosted vs. Cloud

**Cloud (API-based):**
- Faster to deploy
- No infrastructure management
- Access to frontier models
- Pay-per-use pricing
- Data leaves your network

**Self-hosted:**
- Full data control (important for HIPAA, GDPR)
- No per-request API costs (but fixed infrastructure costs)
- Custom model fine-tuning
- No dependency on external services
- Requires ML engineering expertise

A hybrid approach is common: self-hosted ASR and TTS (for data privacy and cost), cloud-based LLM (for quality).

### 16.3 Scaling Considerations

Voice agents require persistent connections (WebSocket or WebRTC) for each active conversation. Unlike HTTP APIs that handle stateless requests, voice sessions maintain state for the duration of the call. This affects scaling:

- **Connection management**: Each server can handle a limited number of concurrent WebSocket/WebRTC connections.
- **State management**: Conversation state (context, tool call history, user information) must be maintained for the duration of the call and cleaned up after.
- **Resource allocation**: Each active session consumes compute (ASR, LLM inference, TTS) and memory. Resource limits per session prevent individual calls from monopolizing shared infrastructure.
- **Load balancing**: Sticky sessions (routing a conversation to the same server for its duration) simplify state management but complicate load balancing.

## 17. The Future of Voice Agents

### 17.1 Convergence of Modalities

Voice agents are evolving toward multimodal agents. A customer service agent might start as a voice call, but the agent sends a link to the user's phone that opens a visual interface (photos, forms, confirmations) while the voice conversation continues. The agent interacts through both voice and visual channels simultaneously.

### 17.2 Improved Latency

Hardware improvements (faster inference chips), model improvements (smaller models with better quality), and architectural improvements (speculative decoding, multi-action generation) will continue to reduce latency. The goal is consistently sub-300ms response times, which would make AI voice agents indistinguishable from humans in terms of conversational pacing.

### 17.3 Better Emotional Intelligence

As native audio models improve, voice agents will become better at reading and responding to emotional cues. A frustrated customer will be met with a calmer, more empathetic tone. A confused caller will get clearer, simpler explanations. This emotional adaptation will significantly improve user satisfaction.

### 17.4 Multilingual and Code-Switching

Users in multilingual environments frequently switch languages mid-sentence. Future voice agents will handle code-switching naturally, recognizing language switches in real-time and responding in the appropriate language. Current systems handle this poorly—ASR trained on one language often fails when the user switches.

### 17.5 Personalization

Voice agents will increasingly personalize their behavior based on user history and preferences: speaking pace, vocabulary level, preferred language, conversation style, and communication preferences. A returning caller will be greeted by name, with the agent already aware of their history and likely needs.

## 18. Conclusion

Voice agents represent a fundamental shift in how humans interact with AI systems. The text interface that has dominated LLM applications is giving way to voice as the natural interface for real-time, interactive, and emotionally nuanced communication.

The technology stack is mature enough for production deployment. ASR achieves near-human accuracy. LLMs provide intelligent, context-aware responses. TTS generates natural-sounding speech. WebRTC and telephony integration enable deployment across web and phone channels. Open-source frameworks (Pipecat, LiveKit Agents, Vocode) provide production-quality scaffolding.

The primary challenges are latency, cost, and emotional intelligence. The latency budget is tight—under 700ms from user speech end to agent speech start—and every component in the pipeline must be optimized to meet it. Costs of $0.03-0.15 per minute for pipeline agents (or $0.20-0.40 for native audio) make voice agents dramatically cheaper than human agents for routine tasks, but the economics depend on the agent's ability to handle interactions autonomously without escalation. Emotional intelligence—understanding the user's emotional state and responding appropriately—remains an area where voice agents are noticeably less capable than humans.

For practitioners, the guidance is straightforward: start with a clear use case where voice is the natural modality (phone-based customer service, appointment booking, hands-free applications), choose a pipeline architecture for cost efficiency or a native audio model for emotional quality, use an open-source framework to avoid building infrastructure from scratch, and invest heavily in latency optimization—because in voice, latency is the user experience.
