# The Chinese Open-Source LLM Ecosystem: DeepSeek, Qwen, GLM, Yi, and the Geopolitics of Model Weights

*April 2026 • Technical Report*

## 1. Introduction

For the first three years of the LLM era, the conversation about open-source models centered on Western releases — Llama from Meta, Mistral from France, and assorted research releases from US universities. The Chinese AI ecosystem was assumed to be a follower, building Chinese-language alternatives but lagging the frontier. By late 2024, this assumption had collapsed. DeepSeek released a series of models that matched or exceeded the capabilities of the best closed Western models, while being open-weight, much smaller, and trained at a fraction of the cost. Qwen, Alibaba's flagship LLM family, became one of the most-downloaded model series on Hugging Face. GLM, Yi, InternLM, and several other Chinese open releases populated the leaderboards alongside the Western options.

By 2026, the Chinese open-source LLM ecosystem is one of the dominant forces in the global open AI landscape. The technical capability is real, the engineering quality is high, and the strategic implications are significant. For practitioners outside China, these models represent both opportunity (cutting-edge capability available under permissive licenses) and complexity (geopolitical concerns, supply chain considerations, and the question of how to evaluate models from organizations operating under different regulatory regimes).

This report examines the major Chinese open-source LLM families, the strategy behind their open releases, the technical innovations they have contributed, and the broader context in which they exist.

## 2. The Major Players

### 2.1 DeepSeek

DeepSeek is the most consequential Chinese open-source LLM organization to emerge in the past two years. Founded as a subsidiary of High-Flyer, a Chinese quantitative trading firm, DeepSeek has produced a series of models that have shaped the global conversation about what is possible with limited resources.

The DeepSeek-V3 model, released in late 2024, was a 671-billion-parameter Mixture-of-Experts model with 37 billion active parameters per token. Its quality matched GPT-4 class models on most benchmarks, and it was trained on roughly 14.8 trillion tokens for a reported cost of about $5.5 million — a fraction of what Western labs were spending on equivalent models. The training efficiency came from a combination of MoE architecture, careful precision management (FP8 training with custom gradient handling), Multi-head Latent Attention (MLA, an innovation that significantly reduces KV cache size), and aggressive optimization of the GPU communication patterns.

DeepSeek-R1, released in January 2025, was the more dramatic event. It was a reasoning model that matched OpenAI's o1 on math and coding benchmarks, and it was released as open-weights with a permissive license. The technical report described a remarkably simple training recipe: GRPO reinforcement learning with binary correctness rewards on math and code problems, with no supervised chain-of-thought data. The model learned to reason — including self-correction, multi-path exploration, and elaborate intermediate work — purely from outcome-level rewards.

The reception of R1 was unprecedented. The model topped the OpenLLM leaderboards for reasoning capability for several weeks, and the technical report became one of the most-cited LLM papers of 2025. The implication was that frontier reasoning capability was not the exclusive domain of well-funded American labs — a relatively small Chinese team had matched it with a published method that anyone could reproduce.

DeepSeek's subsequent releases have continued the pattern of strong models with detailed technical reports. The team's emphasis on transparency and reproducibility has made it a favorite of open-source researchers and a credible alternative to Western model providers.

### 2.2 Qwen (Alibaba)

Qwen is the LLM family from Alibaba Cloud. It is the largest and most extensively maintained Chinese open-source LLM project, with models ranging from 0.5 billion parameters (for edge deployment) to 72 billion parameters and beyond. The Qwen family includes general-purpose chat models, specialized coding models (Qwen-Coder), vision-language models (Qwen-VL), and reasoning models (Qwen-Reasoning).

The Qwen team has been consistent and prolific. Major releases have come every few months for the past two years, with each release improving on the previous in capability while maintaining the open-weights commitment. Qwen models support a broad range of languages (with particular strength in Chinese, English, and several Asian languages) and are widely used in production deployments inside and outside China.

By download volume and Hugging Face activity, Qwen is the most popular open Chinese LLM family. Many fine-tunes, distillations, and merged models in the open ecosystem are based on Qwen as the underlying foundation. The community engagement is comparable to the Llama community in the West.

### 2.3 GLM (Zhipu AI)

GLM (General Language Model) is the LLM family from Zhipu AI, a Tsinghua University spinoff. The GLM models predate the recent open-source LLM boom — the first GLM was released in 2021, before ChatGPT — and the family has evolved through several generations.

ChatGLM-6B, released in March 2023, was one of the first capable bilingual (Chinese-English) chat models available with open weights. It was widely used in Chinese-language NLP applications and helped establish Zhipu as a credible player in the Chinese LLM market. Subsequent releases (ChatGLM2, ChatGLM3, GLM-4) have steadily improved capability while remaining open.

Zhipu's commercial business is more focused than DeepSeek's; the company sells inference services and enterprise deployments built on top of GLM. The open releases serve as both technology demonstration and ecosystem building tool. Zhipu has received significant investment from Chinese tech giants and is one of the better-funded Chinese AI startups.

### 2.4 Yi (01.AI)

01.AI is the LLM startup founded by Kai-Fu Lee, the former Google China head and well-known AI investor. The Yi series of models has been notable for combining strong capability with careful licensing — Yi models have been released under licenses that explicitly allow commercial use, removing the legal ambiguity that surrounds some other Chinese model releases.

Yi-34B, released in late 2023, was one of the highest-quality open models of its time, performing comparably to the best Llama 2 fine-tunes. Subsequent Yi releases have continued to be competitive, though the family has been less prolific than Qwen or DeepSeek.

01.AI's positioning emphasizes Yi as a global model rather than a Chinese-specific one. The training data is multilingual, the documentation is in English, and the company actively engages with the international research community. This positioning has helped Yi gain traction outside China, though it has not been as widely adopted as Llama or Qwen.

### 2.5 InternLM (Shanghai AI Lab)

InternLM is the LLM family from the Shanghai AI Laboratory, a government-backed research institution. The InternLM models are research-grade releases with strong technical documentation and a focus on capabilities for scientific applications. They are less commercially-oriented than the other Chinese open LLMs but have produced solid models in the 7B-20B parameter range.

InternLM is part of a broader "Intern" family of open AI projects from the Shanghai AI Lab, which also includes vision models (InternVL), multimodal models, and specialized tools. The lab is well-funded and is positioned as a Chinese research counterweight to the major Western AI research organizations.

### 2.6 Others

Beyond the major players, the Chinese open-source ecosystem includes many smaller releases: Baichuan (a series of capable bilingual models), DeepSeek's code-specific releases (DeepSeek-Coder), Tencent's Hunyuan releases (originally closed but increasingly open-weight), MoonShot's Kimi (primarily a closed product but with some open variants), and various university and research lab releases.

The total volume of Chinese open model releases now rivals or exceeds the Western open output. The breadth of models — covering different sizes, domains, and capabilities — is one of the defining characteristics of the Chinese open LLM landscape.

## 3. Technical Innovations

The Chinese open-source LLM teams have not just kept up with Western releases — they have contributed major technical innovations that have shaped the field.

### 3.1 Multi-head Latent Attention (MLA)

DeepSeek's Multi-head Latent Attention is one of the most significant architectural innovations of the past two years. It compresses the KV cache by projecting keys and values into a lower-dimensional latent space, then projecting back at attention time. The compression ratio can be 4-10×, dramatically reducing the memory bandwidth required for long-context inference.

MLA was first introduced in DeepSeek-V2 and refined in V3. It has since been adopted by several other model developers (with credit to the DeepSeek team). The technique is particularly valuable for serving long-context models where KV cache size dominates the memory budget.

### 3.2 GRPO

Group Relative Policy Optimization, also from DeepSeek, has become the dominant method for training reasoning models. It is simpler and cheaper than PPO while producing comparable or better results for tasks with verifiable rewards. The DeepSeek-R1 paper that introduced GRPO is one of the most-cited LLM papers of 2025, and the technique has been replicated and extended by many other groups.

### 3.3 FP8 Training

DeepSeek-V3 demonstrated that training in FP8 (with careful precision management) was feasible at the 600B+ parameter scale. The technical report detailed the specific gradient scaling, loss scaling, and precision-aware optimizer states that made this work. The result was a roughly 2× reduction in training compute without quality loss, which has since been adopted by other labs working on FP8 training.

### 3.4 Efficient Mixture-of-Experts

The Chinese teams have invested heavily in MoE optimization. DeepSeek's MoE implementations, Qwen's MoE variants, and several other releases have explored different routing strategies, load balancing techniques, and expert sizes. The collective body of MoE work from Chinese labs is now larger than from Western labs and includes some of the most-cited recent papers on MoE training stability.

### 3.5 Long Context

Several Chinese releases have pushed the boundaries of long-context capability. Kimi from MoonShot supports 200K+ token contexts, and some of the more recent Qwen and GLM releases handle million-token contexts. The techniques used (combinations of attention modifications, position encoding tweaks, and continued pretraining on long sequences) are documented in papers and have influenced Western releases.

## 4. The Strategic Context

Why are Chinese organizations releasing competitive models as open-source? The motivations are complex and not entirely transparent.

### 4.1 Ecosystem Building

The most explicit motivation is ecosystem building. By releasing capable models with permissive licenses, Chinese organizations make their models the default choice for many practitioners — particularly in China and across Asia, where they often outperform Western alternatives on local-language tasks. The downstream effects include training the next generation of Chinese AI engineers on these models, building applications that depend on these models, and establishing the Chinese players as the natural collaborators for downstream work.

This is the same playbook that Meta has used with Llama: release strong open models, build an ecosystem around them, and benefit from the network effects without selling the models directly. Chinese organizations have learned the lesson well and are executing it at scale.

### 4.2 Compute and Export Controls

A more pragmatic motivation comes from US export controls on AI chips. The US government has progressively restricted the sale of advanced AI accelerators to China, with the H100, H200, and Blackwell chips all subject to export licensing. This has forced Chinese AI labs to be efficient — they cannot count on having unlimited access to the latest hardware, so they must make every chip count.

The DeepSeek story is the clearest example. DeepSeek-V3 was reportedly trained on a cluster of H800 chips (a downgraded H100 variant created specifically for the Chinese market). The team's emphasis on efficiency — FP8 training, MLA, optimized communication — was driven in part by the constraint of working with less compute than their American counterparts. The result was both a more efficient model and a demonstration that the export controls had not stopped Chinese AI progress.

This dynamic has produced a paradoxical outcome: the export controls intended to slow Chinese AI development have, in some ways, accelerated it, by forcing the Chinese teams to develop and publish techniques that are valuable to the entire field.

### 4.3 Government Priorities

Chinese government policy has explicitly prioritized AI as a strategic technology. The 2017 "New Generation Artificial Intelligence Development Plan" called for China to become the world leader in AI by 2030, with substantial state funding directed at AI research and infrastructure. The major Chinese AI labs receive direct or indirect government support, and the production of high-quality open-source models is consistent with the broader goal of establishing China as an AI leader.

The open-source release strategy serves multiple government objectives: it builds international goodwill, demonstrates Chinese AI capability, and reduces dependence on Western model providers. It is also less politically sensitive than commercial sales — open-source releases do not raise the same concerns about supply chain risk that commercial deployments do.

## 5. The Geopolitical Complications

For Western practitioners, using Chinese open-source models raises several questions that do not arise with Western releases.

### 5.1 Trust and Provenance

Can a model from a Chinese organization be trusted? The technical question — does the model contain backdoors or hidden behaviors — is hard to answer with certainty. Modern neural networks are too large to inspect comprehensively. There is no way to be sure that a model has not been trained to behave differently when it detects specific inputs (a "backdoor" trigger), or to subtly bias its outputs in ways that favor specific narratives.

In practice, the major Chinese open releases have not been found to contain backdoors or obvious manipulation. The technical community has examined them carefully, and the same kinds of evaluations that are applied to Western releases have been applied to Chinese ones. But the fundamental difficulty of comprehensively auditing a frontier model means that some uncertainty remains, and risk-averse organizations may prefer not to take it.

### 5.2 Censorship and Content Policies

Chinese models often refuse to discuss topics that are politically sensitive in China — Taiwan, Tiananmen, certain historical events, criticism of the Chinese government. This is not a hidden backdoor but an explicit training choice, sometimes documented in the model's system prompts or alignment training. For users who need to discuss these topics, Chinese models are unsuitable.

The censorship is usually limited to specific topics and does not affect general-purpose capabilities. A Chinese coding model is not censored — it codes. A Chinese reasoning model is not censored on math problems. The censorship is targeted at political content, not at general capability.

For users who do not need to discuss politically sensitive topics, the censorship is largely irrelevant. For users who do, Western models are a better fit.

### 5.3 Regulatory Compliance

Using a Chinese model in a Western enterprise context raises compliance questions. Some jurisdictions are considering or have passed regulations specifically about AI models from "unfriendly" countries, with the US being the most active. The legal landscape is in flux, and an organization that builds critical infrastructure on a Chinese model may face unexpected compliance obligations later.

The compliance picture is least complicated for research use, intermediate for non-critical commercial use, and most complicated for high-stakes or government-adjacent deployments. Organizations should consult with their legal and compliance teams before building production systems on models from any country whose government has adversarial relationships with their own.

## 6. The Practical Calculus

For most practical use cases, the question of whether to use a Chinese open-source LLM comes down to cost, capability, and licensing — the same factors that apply to any model choice.

**Cost**: Chinese open models are free to download and use. The inference costs depend on where you run them, but the model weights themselves cost nothing.

**Capability**: For most tasks in 2026, Chinese open models are competitive with Western open models. For some tasks (Chinese-language NLP, certain coding tasks, mathematical reasoning), they are the best open option. For other tasks (English-language nuance, certain reasoning tasks, latest features), Western models may be better.

**Licensing**: Most Chinese open models use permissive licenses. DeepSeek uses an MIT-based license. Qwen uses a custom license that is permissive for most uses but has some restrictions on very large deployments. Yi uses Apache 2.0. Always check the specific license for the specific model and version.

For practitioners with no specific concerns about Chinese-origin software, the decision should be made on technical merits. The Chinese options are often the right choice for cost-sensitive deployments, multilingual applications, or workloads where the Chinese teams have particular strength (reasoning, coding, long context). For practitioners with specific constraints (regulated industries, geopolitical concerns, dependency on Western tooling), the Western alternatives may be preferable.

## 7. Conclusion

The Chinese open-source LLM ecosystem is one of the most significant developments in global AI of the past several years. The models are real, the capabilities are genuine, and the technical contributions have shaped the field in ways that benefit everyone working with LLMs. DeepSeek's GRPO technique, MLA architecture, and FP8 training methodology are now part of the global LLM playbook. Qwen's prolific release schedule has provided high-quality open models for many use cases. The collective body of Chinese open-source LLM work has narrowed the gap between open and closed models in ways that no Western organization has matched.

For practitioners, the practical lesson is that the Chinese open models deserve serious consideration alongside the Western alternatives. They are not worse, they are not subservient followers, and in many cases they are the best option for specific use cases. The geopolitical and compliance considerations are real but limited to specific contexts. For most builders, the right approach is to evaluate Chinese models on their technical merits and choose them when they are the best fit.

The deeper significance is that AI is no longer a one-region story. The narrative of American dominance in AI development, which was largely accurate through 2023, has become increasingly complicated. Chinese teams have demonstrated frontier-class capability with open releases. European teams (Mistral, several research consortia) have produced competitive models. The locus of AI innovation is broadening, and the ecosystem of open models — across all geographies — is healthier and more diverse than ever. This is a positive development for everyone, regardless of national origin, who wants AI capability to be widely available rather than concentrated in the hands of a few well-funded organizations.
