# Pre-training Data Pipelines: From Raw Web Crawls to Trillion-Token Datasets

*April 2026 · Technical Report*

## 1. Introduction

The performance of a large language model is determined as much by its training data as by its architecture or training procedure. A model trained on 15 trillion carefully curated tokens can dramatically outperform a model of the same size trained on 15 trillion unfiltered tokens — not because of different architectures or hyperparameters, but because of differences in data quality, diversity, deduplication, and domain mixing. The preparation of pre-training data has evolved from a simple pipeline of "crawl the web, tokenize, train" into a sophisticated engineering discipline involving dozens of filtering stages, multiple deduplication strategies, careful domain balancing, and extensive decontamination procedures.

This report provides a comprehensive technical examination of modern pre-training data pipelines: the data sources, the deduplication and filtering techniques, the domain mixing strategies, and the quality-focused methodologies that have defined the state of the art through 2025–2026. We examine specific pipelines (FineWeb, RedPajama, Dolma, the Llama 3 data pipeline) in detail, alongside the broader principles and tradeoffs that guide pipeline design.

## 2. Data Sources

### 2.1 Common Crawl

Common Crawl is the foundational data source for virtually all large-scale pre-training datasets. This non-profit organization has been crawling the web since 2007 and makes its crawl archives freely available on Amazon S3. As of early 2026, Common Crawl contains over 250 billion web pages spanning roughly 100 petabytes of raw HTML.

Each monthly crawl snapshot (typically 2–4 billion pages) is published in three formats:
- **WARC (Web ARChive)**: The raw HTTP responses including headers and full HTML.
- **WET (WARC Encapsulated Text)**: Extracted plain text from the HTML.
- **WAT (Web ARChive Metadata)**: Metadata about each page (URL, headers, timestamps).

Most pre-training pipelines start from WARC files rather than WET files, because custom text extraction from HTML allows better control over content quality — the default Common Crawl text extraction loses structural information (headings, lists, tables) and can include navigation elements, advertisements, and boilerplate text.

### 2.2 FineWeb

FineWeb, released by Hugging Face in 2024, is a 15-trillion-token English web text dataset processed from 96 Common Crawl snapshots (2013–2024). FineWeb represents the state of the art in open pre-training data curation, with a carefully documented pipeline that includes text extraction, URL filtering, quality filtering, and deduplication.

**FineWeb-Edu** is a subset of FineWeb filtered for educational content. A classifier trained to predict the "educational value" of web pages (scored 0–5 by an LLM judge) is applied to FineWeb, retaining only pages scoring 3 or above. FineWeb-Edu contains approximately 1.3 trillion tokens and has been shown to produce models with significantly stronger reasoning and knowledge capabilities compared to models trained on unfiltered FineWeb, despite being nearly an order of magnitude smaller.

### 2.3 DCLM (DataComp-LM)

DCLM, from the DataComp project, takes a systematic approach to data curation by treating dataset construction as an optimization problem. The project defines a standardized evaluation protocol and invites submissions of data filtering strategies that maximize downstream model performance on a fixed evaluation suite.

The DCLM baseline pool starts from Common Crawl and applies a pipeline of deduplication, language identification, and quality filtering. The key innovation is the use of a fasttext classifier trained to distinguish high-quality web text (approximated by pages that appear in reference datasets like Wikipedia or curated corpora) from low-quality text. This simple classifier, applied at scale, produces a dataset (DCLM-Baseline, approximately 4 trillion tokens) that significantly outperforms previous open datasets on standardized evaluations.

### 2.4 The Stack (Code Data)

The Stack and its successor The Stack v2, curated by the BigCode project, provide code data for pre-training. The Stack v2 contains over 67 billion lines of code across 619 programming languages, sourced from permissively licensed GitHub repositories, with deduplication and near-duplicate detection applied.

Code data is a critical component of pre-training mixtures because it improves not just coding ability but also general reasoning, structured thinking, and instruction following. Models trained with a significant proportion of code data (typically 15–30% of the pre-training mixture) consistently outperform code-free models on non-coding benchmarks as well.

### 2.5 Curated Text Sources

Beyond web crawls, pre-training datasets include several curated sources:

- **Wikipedia**: High-quality encyclopedic content in over 300 languages. Typically contributes 1–3% of the pre-training mixture but has outsized impact on factual knowledge.
- **Books**: Open-access books from Project Gutenberg and Internet Archive. The Books3 dataset (approximately 197,000 books) was widely used before copyright concerns led to its removal from some datasets.
- **Academic papers**: ArXiv, PubMed, Semantic Scholar — particularly important for scientific and mathematical reasoning.
- **Government and legal documents**: Public court filings, legislation, government reports.
- **Technical documentation**: Software documentation, RFCs, technical standards.
- **Multilingual parallel corpora**: OPUS, CCAligned, and other parallel text collections for multilingual models.

### 2.6 Proprietary and Licensed Data

Commercial model developers increasingly supplement web-scraped data with proprietary or licensed content:

- **Publisher agreements**: Licensing deals with publishers, news organizations, and content providers for high-quality written content.
- **Internal data**: Enterprise-specific documents, knowledge bases, and communications (for custom models).
- **Crowdsourced data**: Purpose-built datasets created by paid annotators for specific domains or capabilities.

The use of copyrighted content for pre-training remains legally contested. Several major lawsuits (by the New York Times, Getty Images, book authors, and others) are challenging the fair-use arguments used by model developers. The outcome of these cases will significantly impact future data pipeline design.

## 3. Text Extraction

### 3.1 HTML to Text Conversion

Converting raw HTML to clean text is the first and one of the most impactful steps in the pipeline. The quality of text extraction affects every downstream filtering step.

Key challenges include:
- **Boilerplate removal**: Separating main content from navigation menus, sidebars, footers, advertisements, and cookie notices.
- **Structural preservation**: Maintaining meaningful structure (paragraphs, headings, lists) while removing HTML markup.
- **Table handling**: Converting HTML tables to a text representation that preserves their information content.
- **Script and style removal**: Eliminating JavaScript, CSS, and other non-content elements.
- **Character encoding**: Handling the diverse character encodings found across the web.

### 3.2 Text Extraction Tools

Several tools are commonly used for text extraction:

**trafilatura**: A Python library specifically designed for web content extraction. It uses a combination of heuristics and machine learning to identify main content, removing boilerplate with high accuracy. FineWeb uses trafilatura as its primary text extraction tool.

**jusText**: A heuristic-based tool that classifies text blocks as "good" (main content), "bad" (boilerplate), or "short" (ambiguous). It works well for news articles and blog posts but can struggle with non-standard layouts.

**Resiliparse**: A high-performance text extraction library used in the CCNet pipeline. Optimized for speed, it processes Common Crawl at scale but may sacrifice some extraction quality compared to trafilatura.

**readability-lxml**: Based on Mozilla's Readability algorithm (used in Firefox's Reader View), this tool extracts the main article content from web pages. It works well for article-style content but may miss content in non-article layouts.

### 3.3 The Impact of Extraction Quality

FineWeb's ablation studies demonstrated that text extraction quality has a measurable impact on downstream model performance. Switching from the default Common Crawl WET text to trafilatura-extracted text improved model quality on standard benchmarks by 2–4 percentage points — a significant gain achieved solely by changing the text extraction tool. This highlights an often-overlooked aspect of data pipeline design: the first step (extraction) compounds through all subsequent steps.

## 4. Deduplication

### 4.1 Why Deduplication Matters

Web crawl data contains enormous amounts of duplication. The same content appears on multiple URLs (syndicated articles, content farms, cross-posted content), the same page is crawled in multiple snapshots, and templates and boilerplate text repeat across millions of pages within the same site. Without deduplication, a model trained on raw Common Crawl data would memorize repeated passages rather than learning from diverse examples, wasting training compute and reducing generalization.

Deduplication has been shown to improve model quality significantly. Lee et al. (2022) demonstrated that deduplicating the C4 dataset improved perplexity by 10% and downstream task performance by several percentage points. The effect is consistent across model sizes and training durations.

### 4.2 Exact Deduplication

The simplest form of deduplication removes documents with identical content. This is typically implemented by hashing the normalized text of each document (after lowercasing, whitespace normalization, and punctuation removal) and removing documents with duplicate hashes.

Exact deduplication is fast (O(n) with hash tables) and removes a substantial fraction of duplicates, but it misses near-duplicates — documents that differ by small edits, different headers/footers, or minor reformatting. In Common Crawl data, exact deduplication typically removes 10–30% of documents.

### 4.3 URL-Based Deduplication

A complementary approach deduplicates based on URL rather than content. If the same URL appears in multiple crawl snapshots, only the most recent version is retained. This handles the common case of pages that are recrawled with minor updates (updated timestamps, different advertisements) but essentially identical content.

URL-based deduplication is very fast but imperfect — the same content can appear at different URLs (different domains, URL parameters, www vs. non-www, HTTP vs. HTTPS), and different content can appear at the same URL over time.

### 4.4 MinHash Deduplication

MinHash is the standard technique for fuzzy (near-duplicate) deduplication at scale. The algorithm:

1. **Shingling**: Convert each document to a set of character-level or word-level n-grams (shingles). Typical configurations use 5-grams of words or 13-grams of characters.
2. **MinHash signature**: For each document, compute a MinHash signature — a fixed-size vector of hash values that approximates the Jaccard similarity between documents. Each element of the signature is the minimum hash value of the shingle set under a different hash function.
3. **Locality-Sensitive Hashing (LSH)**: Group documents into buckets using bands of the MinHash signature. Documents that share a bucket are candidate near-duplicates.
4. **Verification**: For each candidate pair, compute the actual Jaccard similarity and mark pairs above a threshold (typically 0.7–0.8) as duplicates.
5. **Cluster resolution**: From the identified duplicate pairs, form clusters and retain only one document per cluster (typically the longest or most recent).

MinHash deduplication is computationally expensive at the scale of Common Crawl (billions of documents) but can be parallelized effectively. FineWeb performs MinHash deduplication independently within each Common Crawl snapshot, finding that cross-snapshot deduplication provides diminishing returns relative to its computational cost.

### 4.5 Suffix Array Deduplication

Suffix array-based methods (introduced by Lee et al., 2022) detect duplicate substrings at the passage level rather than the document level. This catches cases where different documents share long identical passages (copied paragraphs, standard disclaimers, embedded articles) even if the documents as a whole are not near-duplicates.

The process builds a suffix array over the concatenation of all documents, identifies repeated substrings above a length threshold (typically 50–100 tokens), and removes or truncates the duplicate passages. This is more computationally expensive than document-level deduplication but catches a different class of redundancy.

### 4.6 Embedding-Based Deduplication

A newer approach uses text embeddings to identify semantically similar documents even when they share little surface-level text overlap. Documents are encoded with an embedding model, and pairs with cosine similarity above a threshold are flagged as semantic duplicates. This catches paraphrases, translations, and reformulations that MinHash would miss.

Embedding-based deduplication is the most expensive approach (requiring inference through an embedding model for every document) but provides the most thorough deduplication. It is typically applied to smaller, curated datasets rather than the full web crawl.

### 4.7 Deduplication Ordering

The order in which deduplication is applied matters:

1. **URL deduplication first**: Fast, removes obvious duplicates.
2. **Exact deduplication second**: Also fast, catches content shared across different URLs.
3. **MinHash deduplication third**: Catches near-duplicates remaining after exact dedup.
4. **Suffix array deduplication (optional)**: Catches passage-level duplication.

Each stage typically removes 10–40% of the remaining data, with cumulative removal rates of 50–80% from raw Common Crawl.

## 5. Quality Filtering

### 5.1 The Quality Spectrum

Web crawl data spans an enormous quality spectrum, from carefully written Wikipedia articles and peer-reviewed scientific papers to spam, machine-generated filler text, cookie notices, and error pages. Quality filtering aims to retain the high-quality end of this spectrum while removing content that would not contribute positively to model training.

What constitutes "quality" for pre-training data is not straightforward. Academic prose is high quality in one sense, but a training set consisting entirely of academic prose would produce a model that cannot handle informal conversation. The goal is not to select only the "best" text but to select text that is useful for learning — text that demonstrates competent use of language across diverse contexts.

### 5.2 Heuristic Filters

Heuristic filters use simple, fast rules to remove obviously low-quality content. Common heuristics include:

**Length filters**: Remove documents that are too short (less than 100–200 characters) or too long (more than 100K characters). Very short documents are often error pages, navigation text, or boilerplate. Very long documents may be data dumps, log files, or machine-generated content.

**Language identification**: Use fastText or similar classifiers to identify the language of each document and remove documents that don't match the target language(s). This also removes garbled or mixed-language content that the classifier cannot confidently classify.

**Character-level filters**: Remove documents with excessive special characters, non-standard Unicode, HTML entities, or control characters. These indicate poorly extracted text or non-natural-language content.

**Word-level filters**: Remove documents where the average word length is unreasonable (too short or too long), where the type-token ratio is too low (extremely repetitive text), or where stop-word frequency is abnormal.

**Line-level filters**: Remove documents where most lines end without punctuation (indicating lists, code, or tables rather than prose), where most lines are very short (navigation elements), or where a high proportion of lines are duplicated (templates).

**"Dirty word" filters**: Remove or flag documents containing high concentrations of profanity, slurs, or explicit sexual content. The threshold is tunable — some models deliberately include diverse content while filtering the most extreme material.

### 5.3 Perplexity-Based Filtering

Perplexity filtering uses a language model to score how "natural" each document is. Documents with very high perplexity (the language model finds them surprising) are likely to be garbled, machine-generated, or non-natural-language content. Documents with very low perplexity may be extremely formulaic or templated.

The standard approach, introduced in the CCNet pipeline, trains a KenLM 5-gram language model on a high-quality reference corpus (typically Wikipedia) and computes the perplexity of each document. Documents with perplexity in a middle range (not too high, not too low) are retained.

An important consideration is that perplexity filtering biases toward text that resembles the reference corpus. A Wikipedia-trained perplexity model will favor encyclopedia-style text and penalize informal, conversational, or domain-specific text. This can inadvertently reduce the diversity of the training data. Some pipelines use multiple reference corpora or domain-specific perplexity models to mitigate this bias.

### 5.4 Classifier-Based Quality Filtering

The most effective quality filtering approach trains a classifier to distinguish high-quality text from low-quality text. The classifier is trained on a proxy task:

**Positive examples**: Text from known high-quality sources (Wikipedia, curated textbooks, high-quality news sites, pages linked from Wikipedia or educational resources).

**Negative examples**: Random web crawl text or text from known low-quality sources (content farms, spam sites, auto-generated pages).

The classifier (typically a fastText model for speed) assigns a quality score to each document, and documents below a threshold are removed. The threshold controls the quality-quantity tradeoff — a higher threshold produces a smaller, higher-quality dataset, while a lower threshold retains more data at the cost of lower average quality.

FineWeb-Edu's educational quality classifier exemplifies this approach. An LLM (Llama 3 70B) was used to score 500,000 web pages on a 0–5 scale for educational value. A fastText classifier was then trained on these scores and applied to all of FineWeb (15 trillion tokens), producing a 1.3-trillion-token educational subset. Models trained on FineWeb-Edu significantly outperformed models trained on the full FineWeb on reasoning and knowledge benchmarks.

DCLM's quality filtering uses a similar approach with a fastText classifier trained to distinguish text from curated reference corpora (OpenHermes 2.5, a high-quality instruction dataset) from random web text. This "instruction-aware" quality filter selects text that resembles the kind of content useful for instruction following, producing training data that improves both base model quality and downstream fine-tuning results.

### 5.5 Multi-Stage Filtering

Production pipelines apply multiple filtering stages in sequence:

1. URL blocklist filtering (remove known spam/adult domains)
2. Language identification
3. Heuristic filters (length, character, word-level)
4. Perplexity filtering
5. Classifier-based quality filtering
6. Deduplication (often interleaved with filtering)
7. Content safety filtering (PII, toxicity, NSFW)

Each stage removes a fraction of the data, with the cumulative effect typically reducing raw Common Crawl by 80–95%. The order matters — fast filters (URL blocklists, heuristics) are applied first to reduce the volume before expensive filters (classifiers, deduplication) are applied.

## 6. Content Filtering

### 6.1 PII Detection and Removal

Personally identifiable information (PII) — names, email addresses, phone numbers, social security numbers, addresses, and other identifying information — is pervasive in web crawl data. Removing or redacting PII is important for both legal compliance (GDPR, CCPA, and other privacy regulations) and ethical considerations.

Common PII detection approaches:
- **Regular expressions**: Pattern-matching for structured PII (email addresses, phone numbers, SSNs, credit card numbers). Fast but limited to well-defined patterns.
- **Named entity recognition (NER)**: NER models identify person names, organizations, and locations. More flexible than regex but also more expensive and prone to false positives.
- **Custom classifiers**: Specialized models trained on PII detection datasets that can identify less structured PII (partial addresses, dates of birth in context).

In practice, PII removal is imperfect. Web crawl data contains enormous amounts of PII in formats that resist automated detection. Most pipelines focus on removing the highest-risk PII (SSNs, credit card numbers, complete addresses) while accepting that some PII will remain.

### 6.2 Toxicity Filtering

Toxicity filtering removes or reduces content that is hateful, threatening, or harmful. Approaches include:

- **Keyword/phrase blocklists**: Fast but blunt, catching benign content that contains flagged words in non-harmful contexts.
- **Perspective API**: Google's toxicity scoring API, which provides fine-grained toxicity scores across multiple dimensions (toxicity, severe toxicity, insult, profanity, threat, identity attack).
- **Custom classifiers**: Models trained on toxicity datasets (Jigsaw, HateXplain) that classify text as toxic or non-toxic.
- **LLM-based evaluation**: Using a language model to assess whether content is harmful in context, providing more nuanced filtering than keyword or classifier approaches.

The degree of toxicity filtering is a design choice with significant implications. Aggressive filtering removes harmful content but can also remove content discussing sensitive topics in legitimate contexts (news articles about hate crimes, academic discussions of racism, medical descriptions of violence). This can result in a model that refuses to engage with important topics. Less aggressive filtering retains more diverse content but risks exposing the model to harmful patterns that could influence its outputs.

### 6.3 NSFW Filtering

Content that is sexually explicit, graphically violent, or otherwise inappropriate for general audiences is typically filtered from pre-training data. NSFW classifiers (trained on labeled datasets of explicit and non-explicit content) assign scores to each document, and documents above a threshold are removed.

Some pipelines distinguish between different categories of NSFW content. Sexual content may be filtered aggressively while violent content (which appears in news articles, historical accounts, and medical descriptions) may be filtered with a higher threshold. The filtering strategy depends on the intended deployment context of the model.

### 6.4 Copyright and Legal Filtering

Some pipelines implement copyright-aware filtering, removing or downweighting content from sources known to contain copyrighted material (books, academic papers behind paywalls, news articles). This is motivated by both legal risk reduction and ethical considerations.

The opt-out mechanism is another approach: respecting robots.txt directives and explicit requests from content creators to exclude their content from training data. Some datasets (like the AI2 Dolma) provide mechanisms for content creators to request removal.

## 7. Data Mixing

### 7.1 Domain Proportions

Pre-training datasets are not homogeneous — they are carefully composed mixtures of data from different domains, each contributing different capabilities to the model. The proportions of different domains in the training mixture significantly affect model behavior.

A typical domain mixture might include:

| Domain | Proportion | Primary Contribution |
|---|---|---|
| Web text (filtered) | 50–65% | General knowledge, diverse language |
| Code | 15–25% | Reasoning, structured thinking, coding |
| Books | 3–8% | Long-form coherence, literary knowledge |
| Wikipedia | 2–4% | Factual knowledge, encyclopedic style |
| Academic papers | 3–8% | Scientific reasoning, technical vocabulary |
| Conversations/forums | 2–5% | Conversational style, Q&A patterns |
| Math | 2–5% | Mathematical reasoning |
| Multilingual text | 5–15% | Multilingual capability |

### 7.2 Optimizing the Mixture

Finding the optimal data mixture is a significant research challenge. The mixture affects many downstream capabilities simultaneously, and improving one capability often comes at the cost of another (more code data improves coding but may reduce literary quality).

Approaches to mixture optimization include:

**Manual tuning**: Running small-scale training experiments with different mixtures and evaluating on a broad benchmark suite. This is the most common approach but is expensive and explores a limited search space.

**Scaling laws for data mixing**: Extending Chinchilla-style scaling laws to account for data composition. Doremi (2023) proposed Domain Reweighting with Minimax Optimization, which uses a small proxy model to estimate the optimal domain weights that minimize worst-case excess loss across domains.

**Online mixture adjustment**: Adjusting domain proportions during training based on observed loss curves. If the model is learning a particular domain quickly (low loss), reduce its proportion; if a domain has high loss and is learning slowly, increase its proportion.

**Data selection for downstream tasks**: Given a specific downstream task or evaluation suite, select the pre-training data that maximizes performance on that task. DSIR (Data Selection with Importance Resampling) selects pre-training examples that are distributionally similar to the target task data.

### 7.3 Curriculum Learning

Curriculum learning presents training data in a deliberate order — typically from easy to hard or from general to specific. For pre-training, this might mean:

- **Phase 1**: Train on a broad, diverse mixture of web text to learn general language patterns.
- **Phase 2**: Increase the proportion of high-quality, domain-specific data (code, math, scientific text) to build specialized capabilities.
- **Phase 3**: Include more challenging and nuanced content (complex reasoning, multi-step problems) to push the boundaries of capability.

Llama 3's training process followed this pattern, with the domain mixture evolving over the course of training. The initial phases used a broad web text mixture, and later phases upsampled high-quality sources (code, math, scientific text) to strengthen specific capabilities.

### 7.4 Data Repetition

When the amount of available high-quality data is less than the target token count for training, data must be repeated. The number of repetitions (epochs) has implications for model quality.

Research suggests that moderate repetition (2–4 epochs on high-quality data) is acceptable and can even be beneficial (the model extracts more information from high-quality examples on subsequent passes). Excessive repetition (10+ epochs) leads to overfitting and degraded generalization. The Chinchilla finding — that training tokens should scale roughly linearly with model parameters — implies that very large models may require more unique tokens than are available in existing datasets, creating pressure toward either synthetic data generation or more efficient data utilization.

Muennighoff et al. (2024) systematically studied the effects of data repetition, finding that the optimal number of epochs depends on data quality: high-quality data can be repeated more times than low-quality data before degradation sets in.

## 8. Specific Pipeline Case Studies

### 8.1 FineWeb Pipeline

The FineWeb pipeline, documented in detail by Hugging Face, processes 96 Common Crawl snapshots through the following stages:

1. **Text extraction**: trafilatura applied to WARC files, extracting main content and metadata.
2. **URL filtering**: Remove pages from blocklisted domains (spam, adult content, known low-quality sites).
3. **Language identification**: fastText language classifier, retaining only English text (for the English version).
4. **Heuristic filtering**: Apply C4-derived heuristics (minimum length, maximum line length, word count filters, bullet/ellipsis density, stop word ratio).
5. **Repetition removal**: Remove documents with excessive repeated lines, paragraphs, or n-grams (the "repetition filter" from Gopher/C4).
6. **MinHash deduplication**: Per-snapshot fuzzy deduplication with 5-gram shingles, 20 hash functions, and a Jaccard threshold of 0.75.
7. **Quality statistics**: Compute perplexity, quality classifier scores, and other metadata for each document.

The result is approximately 15 trillion tokens of English web text. FineWeb-Edu applies an additional educational quality classifier, reducing this to approximately 1.3 trillion tokens.

Key FineWeb findings:
- trafilatura text extraction was worth 2–4 points on benchmarks compared to default Common Crawl WET extraction.
- MinHash deduplication was essential — without it, model quality degraded significantly.
- The Gopher/C4 repetition filters were particularly impactful, catching repetitive content that MinHash missed.
- Removing too aggressively with quality classifiers hurt performance — there is a sweet spot where quality filtering improves average quality without removing too much useful diversity.

### 8.2 RedPajama

RedPajama, from Together AI, aimed to reproduce the training data composition described in the Llama technical report. RedPajama v1 collected data from the same seven sources as Llama:

1. Common Crawl (5 snapshots, processed with CCNet pipeline)
2. C4 (a cleaned version of Common Crawl from Google)
3. GitHub (code data)
4. Wikipedia
5. Books (Gutenberg, Books3)
6. ArXiv
7. StackExchange

RedPajama v2, released later, scaled to 30 trillion tokens from 84 Common Crawl snapshots with improved filtering. Importantly, RedPajama v2 provided quality signals (perplexity, deduplication status, classifier scores) as metadata rather than applying hard filters, allowing users to define their own quality thresholds.

### 8.3 Dolma

Dolma, from the Allen Institute for AI (AI2), is the pre-training dataset behind the OLMo models. It consists of approximately 3 trillion tokens from:

- Common Crawl (processed with a custom pipeline)
- The Stack (deduplicated)
- C4
- Reddit
- peS2o (AI2's semantic scholar open research corpus)
- Wikipedia and Wikibooks
- Project Gutenberg

Dolma's distinguishing feature is its emphasis on reproducibility and transparency. The entire pipeline is open-source, with detailed documentation of every filtering decision. Dolma also provides a data provenance system that tracks the origin of each document, enabling analysis of data composition and facilitating opt-out requests from content creators.

### 8.4 Llama 3 Data Pipeline

Meta's Llama 3 technical report (2024) provides the most detailed description of a frontier model's pre-training data pipeline. Key details:

**Scale**: Llama 3 was trained on approximately 15 trillion tokens, compared to 2 trillion for Llama 2 — a 7.5× increase that reflected the emphasis on data quality and quantity.

**Quality filtering**: Meta developed a series of quality classifiers:
- A text quality classifier trained on data labeled by Llama 2 (using the model to classify text as suitable for a "reference book" or not).
- A domain-specific classifier for code quality.
- A classifier for "knowledge density" — how much factual information a document contains.

**Deduplication**: URL-level deduplication, document-level MinHash deduplication, and line-level deduplication were all applied. The combination removed approximately 80% of the raw data.

**Heuristic filters**: Based on n-gram coverage analysis against training data from previous Llama versions, Meta identified and removed specific types of low-quality content (excessive repetition, formatting artifacts, adult content).

**Domain mixing**: The training mixture evolved over the course of training:
- Initial phase: Broad web text mixture.
- Later phases: Upsampled code, math, and scientific text.
- Final phase: Further upsampled the highest-quality data sources.

**Annealing**: In the final stage of training, the learning rate was annealed to zero while training on a small, very high-quality dataset. This "annealing" phase disproportionately improved performance on knowledge-intensive and reasoning tasks.

**PII filtering**: Implemented a PII detection pipeline that identified and redacted common PII patterns. Applied more aggressively to domains likely to contain PII (forums, personal blogs) than to domains less likely to contain it (Wikipedia, academic papers).

## 9. Decontamination

### 9.1 The Contamination Problem

Benchmark contamination occurs when test examples from evaluation benchmarks appear in the pre-training data. If a model has seen MMLU questions during pre-training, its performance on MMLU reflects memorization rather than genuine capability, rendering the benchmark meaningless.

Contamination is surprisingly common. Common Crawl contains web pages that publish benchmark datasets, discuss specific benchmark questions, or contain content similar to benchmark examples. Studies have found non-trivial contamination rates for nearly all popular benchmarks in standard web crawl datasets.

### 9.2 Decontamination Methods

**N-gram overlap**: Compute n-gram overlap between training documents and benchmark examples. Remove training documents with overlap above a threshold (typically 8–13 grams of matching text). This is the most common approach but can miss paraphrased contamination.

**Embedding similarity**: Compute embedding similarity between training documents and benchmark examples. Remove documents above a cosine similarity threshold. This catches paraphrased contamination but is more expensive and may have false positives.

**Exact match removal**: For benchmarks with known exact examples, remove those exact strings from the training data. Simple and precise but only catches verbatim contamination.

### 9.3 Limitations

Decontamination is inherently imperfect. Benchmarks test knowledge and capabilities that are genuinely present in the real world — removing all text related to benchmark topics would remove useful training data. The goal is to remove the specific benchmark examples (preventing memorization) while retaining the general knowledge that the benchmark tests.

Additionally, decontamination cannot account for contamination that occurs through model outputs. If a teacher model has memorized benchmark examples and generates synthetic training data that includes them, the contamination propagates through the synthetic data even if the original benchmark examples are removed from the web crawl data.

## 10. Tokenizer-Aware Filtering

### 10.1 The Tokenizer's Impact

The tokenizer converts raw text to the token sequences that the model actually trains on. Tokenizer behavior can interact with data quality in non-obvious ways:

- **Encoding efficiency**: Well-formed English text typically encodes at 3–4 characters per token with modern tokenizers (BPE, SentencePiece). Text that encodes at significantly different ratios (very high characters per token for garbled text, very low for repetitive content) may indicate quality issues.
- **Unknown token density**: Documents with many unknown or single-character tokens may contain unusual characters, formatting artifacts, or non-textual content.
- **Token sequence length**: The same character count can produce very different token counts depending on content — code, mathematical notation, and non-English text often produce longer token sequences, affecting training efficiency.

### 10.2 Token-Level Quality Metrics

Some pipelines compute quality metrics at the token level:
- **Tokens per character ratio**: Deviation from expected ratios flags unusual content.
- **Token entropy**: Very low entropy token sequences indicate repetitive content.
- **Special token concentration**: High density of special tokens, numbers, or punctuation tokens.

These metrics complement character-level and document-level quality measures.

## 11. Chinchilla Data Requirements

### 11.1 Scaling Laws for Data

Hoffmann et al. (2022) established that the optimal number of training tokens scales linearly with model parameters — a model with N parameters should be trained on approximately 20N tokens for compute-optimal training. This means:

| Model Size | Optimal Tokens | Approximate Dataset Size |
|---|---|---|
| 1B | 20B | ~40 GB text |
| 7B | 140B | ~280 GB text |
| 70B | 1.4T | ~2.8 TB text |
| 400B | 8T | ~16 TB text |

### 11.2 The Data Wall

As model sizes have increased, the demand for unique, high-quality training tokens has outpaced supply. The total amount of publicly accessible, high-quality English text on the internet is estimated at 5–15 trillion tokens (depending on quality thresholds). Models at the 70B+ parameter scale are already approaching or exceeding the Chinchilla-optimal token count for available data.

This "data wall" has driven several responses:
- **Training beyond Chinchilla-optimal**: Many recent models (Llama 3, Mistral, Qwen) train for significantly more tokens than the Chinchilla-optimal ratio suggests, accepting diminishing returns in exchange for continued improvement. Llama 3 8B, for example, was trained on 15T tokens — over 100× the Chinchilla-optimal amount for its size — because the resulting model is much better than one trained on the "optimal" 160B tokens.
- **Synthetic data**: Generating additional training data to supplement web crawl data (see the companion report on Synthetic Data Generation).
- **Multilingual data**: Training on text in multiple languages to increase the total pool of available data.
- **Data quality over quantity**: Investing in better quality filtering to extract more value from each token, rather than simply increasing token count.
- **Repeat and upsample**: Repeating high-quality data sources (Wikipedia, curated textbooks, high-quality code) multiple times while cycling through lower-quality web text once.

### 11.3 Beyond Chinchilla

The Chinchilla scaling laws optimize for compute-efficiency — minimum training cost for a given performance level. But for deployment, inference cost is often more important than training cost, since a model may serve billions of inference requests. This creates an incentive to "over-train" smaller models: training a 7B model on 15T tokens costs more compute per parameter than the Chinchilla optimum, but the resulting model is cheaper to deploy than a larger model trained optimally.

This insight, sometimes called "inference-optimal" scaling (as opposed to Chinchilla's "compute-optimal" scaling), has driven the trend toward training small models on massive datasets. The Phi models, Mistral, and Llama 3 all follow this pattern.

## 12. Emerging Trends

### 12.1 Model-Based Data Selection

Rather than using simple heuristics or classifiers, some recent approaches use language models themselves to evaluate and select training data. DSIR, Ask-LLM, and similar methods use either small proxy models or frontier models to score training examples based on their expected utility for downstream tasks.

QuRating (2024) uses LLMs to rate web pages on multiple quality dimensions (writing quality, educational value, factual content, required expertise) and provides multi-dimensional quality metadata that enables flexible filtering based on the desired training data profile.

### 12.2 Multimodal Data Pipelines

As models become multimodal, data pipelines must handle not just text but image-text pairs, video, audio, and other modalities. The challenges of deduplication, quality filtering, and domain mixing apply to each modality, with additional challenges specific to multimodal data (image-text alignment, visual quality assessment, audio transcription quality).

### 12.3 Continual Data Pipelines

Rather than treating pre-training data as a static artifact, some organizations are moving toward continual data pipelines that process new Common Crawl snapshots as they are released, apply the same filtering and quality procedures, and produce incrementally updated datasets. This enables continual pre-training on fresh data, keeping models up to date without full retraining.

### 12.4 Data Governance and Provenance

As legal and regulatory scrutiny of training data increases, data governance — tracking the provenance, licensing, and processing history of every document in the training set — is becoming more important. Tools like Dolma's provenance tracking system, DataComp's metadata approach, and emerging data governance frameworks aim to provide the documentation and auditability that regulators and content creators increasingly demand.

## 13. Conclusion

Pre-training data pipelines have evolved from simple "crawl and tokenize" workflows into sophisticated multi-stage engineering systems that rival the complexity of the model architectures they serve. The key insight of the past three years is that data quality is at least as important as data quantity — and that systematic investment in text extraction, deduplication, quality filtering, and domain mixing produces measurable improvements in model performance that would require significantly more compute to achieve through scale alone.

The open-source community has been instrumental in advancing the state of the art, with projects like FineWeb, RedPajama, Dolma, and DCLM providing not just datasets but detailed documentation of their methodologies, enabling the entire field to learn from each pipeline's design decisions and ablation results.

Looking ahead, the intersection of data pipeline engineering with legal, ethical, and regulatory considerations will shape the field's direction. The data wall — the finite supply of high-quality web text — is driving innovation in synthetic data generation, multilingual data utilization, and more efficient data curation. The models of 2027 will likely be trained on datasets that are smaller but dramatically higher quality than those of 2023, with sophisticated model-based quality selection, dynamic domain mixing, and careful provenance tracking as standard features.

## References

1. Penedo, G., et al. "The FineWeb Datasets: Decanting the Web for the Finest Text Data at Scale." 2024.
2. Li, J., et al. "DataComp-LM: In Search of the Next Generation of Training Sets for Language Models." 2024.
3. Wenzek, G., et al. "CCNet: Extracting High Quality Monolingual Datasets from Web Crawl Data." 2020.
4. Lee, K., et al. "Deduplicating Training Data Makes Language Models Better." ACL 2022.
5. Together Computer. "RedPajama: An Open Dataset for Training Large Language Models." 2023.
6. Soldaini, L., et al. "Dolma: An Open Corpus of Three Trillion Tokens for Language Model Pretraining Research." 2024.
7. Llama Team, Meta. "The Llama 3 Herd of Models." 2024.
8. Hoffmann, J., et al. "Training Compute-Optimal Large Language Models." NeurIPS 2022.
9. Rae, J., et al. "Scaling Language Models: Methods, Analysis & Insights from Training Gopher." 2022.
10. Raffel, C., et al. "Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer." JMLR 2020.
11. Kocetkov, D., et al. "The Stack: 3 TB of Permissively Licensed Source Code." 2022.
12. Muennighoff, N., et al. "Scaling Data-Constrained Language Models." NeurIPS 2024.
13. Xie, S., et al. "DoReMi: Optimizing Data Mixtures Speeds Up Language Model Pretraining." NeurIPS 2023.
14. Xie, S., et al. "Data Selection with Importance Resampling (DSIR)." 2023.
15. Barbaresi, A. "trafilatura: A Web Scraping Library and Command-Line Tool for Text Discovery and Extraction." ACL 2021.
