# 📚 LLM Papers & Articles

A curated collection of articles, technical reports, and deep dives exploring large language models — from inference optimization to training techniques, evaluation methodology, and deployment at scale.

## About

This repository serves as a knowledge base for anyone working with or researching LLMs. Each article is a self-contained document covering a specific topic in depth, written to be accessible to engineers and researchers with a working knowledge of machine learning fundamentals.

## Articles

| Topic | Description |
|-------|-------------|
| [Batching in LLM Inference](articles/batching_report.pdf) | Static batching, continuous batching, disaggregated prefill/decode, speculative decoding, and PagedAttention |
| [KV Cache Optimization](articles/kv_cache_optimization.pdf) | Memory management strategies for key-value caches including paging, compression, and quantization |
| [Quantization for Inference](articles/quantization_for_inference.pdf) | Weight and activation quantization from FP16 down to INT4/FP4, calibration methods, and accuracy trade-offs |
| [Mixture of Experts](articles/mixture_of_experts.pdf) | Sparse MoE architectures, routing strategies, load balancing, and implications for serving infrastructure |
| [RLHF and Alignment](articles/rlhf_and_alignment.pdf) | Reinforcement learning from human feedback, DPO, reward modeling, and preference optimization |
| [Prompt Engineering](articles/prompt_engineering.pdf) | Chain-of-thought, few-shot prompting, system prompts, structured outputs, and tool use patterns |
| [Evaluation and Benchmarks](articles/evaluation_and_benchmarks.pdf) | LLM evaluation methodology, popular benchmarks, contamination risks, and human evaluation design |
| [Distributed Training](articles/distributed_training.pdf) | Data parallelism, tensor parallelism, pipeline parallelism, and ZeRO-style memory optimization |
| [Context Window Scaling](articles/context_window_scaling.pdf) | Extending context length through RoPE scaling, sparse attention, ring attention, and memory-augmented approaches |
| [Tokenization](articles/tokenization.pdf) | BPE, SentencePiece, vocabulary design, multilingual considerations, and the impact of tokenization on model behavior |
| [Distributed Inference on Small GPUs](articles/distributed_llm_inference_report.pdf) | Tensor, pipeline, sequence, and expert parallelism for multi-chip inference; communication bottlenecks; KV-cache management; quantization; continuous batching; and framework comparison |

## Repository Structure

```
├── articles/           # Long-form articles in Markdown
├── figures/            # Diagrams and illustrations referenced by articles
├── references/         # BibTeX files and supplementary citations
└── README.md
```

## Who This Is For

- **ML engineers** deploying LLMs in production and looking to understand optimization techniques
- **Researchers** wanting concise overviews of specific subfields with pointers to primary sources
- **Students** building foundational knowledge of modern LLM systems

## Contributing

Contributions are welcome. If you'd like to add an article or improve an existing one:

1. Fork the repository
2. Create a branch (`git checkout -b article/your-topic`)
3. Write your article in `articles/` following the existing format
4. Open a pull request with a brief summary of the topic and why it's a useful addition

Please keep articles technically grounded and cite primary sources where possible.

## License

This work is licensed under [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/). You are free to share and adapt the material with appropriate attribution.
