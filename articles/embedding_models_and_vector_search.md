# Embedding Models and Vector Search

*March 2026*

## 1. What Are Embeddings?

At their core, embeddings are dense vector representations of discrete objects -- tokens, sentences, passages, images, or entire documents -- projected into a continuous high-dimensional space where geometric proximity encodes semantic similarity. Unlike sparse representations such as TF-IDF or one-hot encodings, which inhabit spaces with dimensionality equal to the vocabulary size and consist almost entirely of zeros, dense embeddings pack meaning into relatively compact vectors of 256 to 4096 floating-point dimensions. The key insight is that a well-trained embedding model learns to position semantically related inputs near each other and unrelated inputs far apart, enabling downstream retrieval, clustering, classification, and recommendation without any task-specific fine-tuning.

## 2. Embedding Model Architectures

### 2.1 Bi-Encoders

The workhorse of modern embedding pipelines is the bi-encoder architecture. A bi-encoder processes each input independently through a shared or twin transformer encoder, producing a single vector per input. At query time, the query is encoded once and compared against a pre-computed index of document embeddings, making bi-encoders extremely efficient for large-scale retrieval. Models like Sentence-BERT established the pattern, and virtually every production embedding model today -- OpenAI's text-embedding-3, Cohere Embed v4, BGE, E5, GTE, and Nomic Embed -- follows this paradigm.

### 2.2 Cross-Encoders

Cross-encoders concatenate the query and document into a single input and pass the pair jointly through a transformer, allowing full token-level attention between them. This yields substantially higher relevance accuracy, but at a cost that is quadratic in the number of candidates, making cross-encoders impractical as first-stage retrievers. They are instead deployed as re-rankers, scoring the top-k results returned by a bi-encoder to refine the final ranking.

### 2.3 Late Interaction: ColBERT

ColBERT and its successors (ColBERTv2, ColPali) occupy a middle ground. Rather than compressing each input to a single vector, late interaction models retain per-token embeddings for both query and document, then compute relevance via a lightweight MaxSim operation -- each query token's maximum cosine similarity against all document tokens, summed across query tokens. This preserves fine-grained matching while still allowing document token embeddings to be pre-computed and indexed, achieving accuracy close to cross-encoders with latency closer to bi-encoders.

## 3. Training Objectives

### 3.1 Contrastive Learning and Hard Negatives

Modern embedding models are trained with contrastive objectives, most commonly InfoNCE, which pulls positive pairs together and pushes negatives apart in embedding space. The quality of negatives is critical: random negatives are too easy, so practitioners mine hard negatives -- passages that are topically similar but not actually relevant -- from BM25 results or from a previous-generation dense retriever. In-batch negatives, where every other positive in the mini-batch serves as a negative for a given query, provide a computationally free source of contrast and scale effectively with batch size.

### 3.2 Knowledge Distillation

A powerful training signal comes from distilling a cross-encoder teacher into a bi-encoder student. The cross-encoder scores query-document pairs with high accuracy; the bi-encoder is then trained to reproduce that score distribution. This technique, used in models like BGE and E5, lets the student inherit much of the teacher's ranking quality while retaining the efficiency of a bi-encoder at inference time.

## 4. Dimensionality and Matryoshka Embeddings

Embedding dimensionality controls the capacity-cost trade-off. Higher dimensions capture more nuance but consume more memory and slow similarity computation. Matryoshka Representation Learning (MRL) offers an elegant solution: the model is trained so that the first d dimensions of the full D-dimensional vector are themselves a useful embedding. Practitioners can truncate to 256, 512, or 1024 dimensions at deployment time, trading a small accuracy reduction for significant memory and latency savings. OpenAI's text-embedding-3 models and Nomic Embed both support MRL truncation natively.

## 5. Vector Similarity Metrics

Three similarity functions dominate. Cosine similarity measures the angle between vectors and is invariant to magnitude, making it the default when embeddings are L2-normalized. Dot product (inner product) is equivalent to cosine similarity for normalized vectors but can also capture magnitude information when vectors are unnormalized. Euclidean (L2) distance measures straight-line distance in embedding space; for normalized vectors, minimizing L2 distance is equivalent to maximizing cosine similarity. The choice often depends on how the embedding model was trained and whether normalization is applied at index time.

## 6. Vector Databases and Indexing

### 6.1 The Database Landscape

The vector database ecosystem has matured considerably. FAISS, developed by Meta, remains the gold standard library for in-process similarity search. Milvus and Qdrant are purpose-built distributed vector databases with filtering, replication, and multi-tenancy. Pinecone offers a fully managed service optimized for operational simplicity. On the relational side, pgvector brings approximate nearest neighbor search into PostgreSQL, allowing teams to avoid an entirely separate infrastructure component when scale requirements are moderate.

### 6.2 Indexing Algorithms

Exact brute-force search is O(n) per query and impractical beyond a few hundred thousand vectors. Approximate nearest neighbor (ANN) algorithms trade a small recall loss for orders-of-magnitude speedup. HNSW (Hierarchical Navigable Small World) builds a multi-layer proximity graph that supports logarithmic-time search and is the default index type in most vector databases due to its strong recall-latency profile. IVF (Inverted File Index) partitions the vector space into Voronoi cells using k-means, restricting search to only the nearest clusters. Product quantization (PQ) compresses vectors by decomposing them into sub-vectors and quantizing each independently, reducing memory footprint by 8-32x at the cost of some accuracy. These techniques are often combined -- IVF-PQ, HNSW-PQ -- to balance memory, speed, and recall.

## 7. Hybrid Search: Dense Meets Sparse

Pure dense retrieval can struggle with exact keyword matching, rare entities, and out-of-domain terminology. Hybrid search addresses this by fusing dense vector results with sparse lexical scores, typically BM25. Reciprocal rank fusion (RRF) and learned score combination are common merging strategies. Most production vector databases now support hybrid mode natively, and retrieval benchmarks consistently show that dense-plus-sparse outperforms either method alone, particularly on heterogeneous query distributions.

## 8. Quantized Embeddings

Full float32 embeddings consume 4 bytes per dimension; at 1024 dimensions and millions of documents, memory costs become substantial. Scalar quantization reduces each dimension to int8 (1 byte) or even int4, typically retaining 95-99% of full-precision recall. Binary quantization goes further, encoding each dimension as a single bit and using Hamming distance for comparison, achieving 32x compression relative to float32. Binary embeddings are particularly effective as a first-stage filter, with a full-precision re-scoring pass on the top candidates to recover accuracy. Cohere's Embed v4 and several open models now ship with built-in binary quantization support.

## 9. The Embedding Model Landscape

The field is rich with capable models. OpenAI's text-embedding-3-large offers strong general-purpose performance with native MRL support. Cohere Embed v4 excels at multilingual retrieval. On the open-source side, BGE (BAAI), E5 (Microsoft), and GTE (Alibaba) consistently rank at the top of the MTEB benchmark. Nomic Embed provides fully open-weight, open-data models with long-context support up to 8192 tokens. Newer entrants like Jina Embeddings v3 and Snowflake Arctic Embed push the Pareto frontier on quality versus model size. The choice depends on latency budget, language coverage, licensing constraints, and whether self-hosting is required.

## 10. Practical Considerations

### 10.1 Chunking Strategy

Embedding models have finite context windows, and retrieval granularity is determined by chunk boundaries. Fixed-size token chunking with overlap is the simplest approach, but semantic chunking -- splitting at paragraph or section boundaries -- tends to produce more coherent retrievable units. Recursive character splitting, used by LangChain and similar frameworks, offers a pragmatic middle ground. Chunk size should be tuned empirically: too small and context is lost, too large and the embedding becomes a diluted average over multiple topics.

### 10.2 Normalization and Domain Adaptation

Most embedding models produce L2-normalized output, but it is worth verifying this before choosing a similarity metric. For domain-specific corpora -- legal documents, biomedical literature, proprietary codebases -- general-purpose embeddings may underperform. Lightweight domain adaptation via continued contrastive fine-tuning on a modest set of in-domain query-passage pairs, sometimes as few as a thousand examples, can yield significant retrieval improvements without catastrophic forgetting of general capability.

## Conclusion

Embedding models and vector search have become foundational infrastructure for retrieval-augmented generation, semantic search, and a growing number of AI-powered applications. The design space spans architecture choices, training methodology, quantization strategy, indexing algorithms, and hybrid retrieval -- each with concrete trade-offs that ML engineers must navigate. As models grow more capable, context windows lengthen, and quantization techniques mature, the cost-quality Pareto frontier continues to shift, making high-quality semantic retrieval accessible at ever larger scales and tighter latency budgets. Understanding these building blocks is essential for anyone designing systems where language models must interact with external knowledge.
