# 📚 LLM Papers & Articles

A curated collection of articles, technical reports, and deep dives exploring large language models — from inference optimization to training techniques, evaluation methodology, and deployment at scale.

## About

This repository serves as a knowledge base for anyone working with or researching LLMs. Each article is a self-contained document covering a specific topic in depth, written to be accessible to engineers and researchers with a working knowledge of machine learning fundamentals.

## Articles

| Topic | Format |
|-------|--------|
| Batching in LLM Inference | [PDF](articles/batching_report.pdf) · [MD](articles/batching_report.md) |
| KV Cache Optimization | [PDF](articles/kv_cache_optimization.pdf) · [MD](articles/kv_cache_optimization.md) |
| Quantization for Inference | [PDF](articles/quantization_for_inference.pdf) · [MD](articles/quantization_for_inference.md) |
| Mixture of Experts | [PDF](articles/mixture_of_experts.pdf) · [MD](articles/mixture_of_experts.md) |
| RLHF and Alignment | [PDF](articles/rlhf_and_alignment.pdf) · [MD](articles/rlhf_and_alignment.md) |
| Prompt Engineering | [PDF](articles/prompt_engineering.pdf) · [MD](articles/prompt_engineering.md) |
| Evaluation and Benchmarks | [PDF](articles/evaluation_and_benchmarks.pdf) · [MD](articles/evaluation_and_benchmarks.md) |
| Distributed Training | [PDF](articles/distributed_training.pdf) · [MD](articles/distributed_training.md) |
| Context Window Scaling | [PDF](articles/context_window_scaling.pdf) · [MD](articles/context_window_scaling.md) |
| Tokenization | [PDF](articles/tokenization.pdf) · [MD](articles/tokenization.md) |
| Distributed Inference on Small GPUs | [PDF](articles/distributed_llm_inference_report.pdf) · [MD](articles/distributed_llm_inference_report.md) |
| Speculative Decoding | [PDF](articles/speculative_decoding.pdf) · [MD](articles/speculative_decoding.md) |
| Power and Thermal Management | [PDF](articles/power_and_thermal_management.pdf) · [MD](articles/power_and_thermal_management.md) |
| Open-Weights LLM Landscape | [PDF](articles/open_weights_llm_landscape.pdf) · [MD](articles/open_weights_llm_landscape.md) |
| Memory Hierarchy and Offloading | [PDF](articles/memory_hierarchy_and_offloading.pdf) · [MD](articles/memory_hierarchy_and_offloading.md) |
| GPU Power Variability | [PDF](articles/gpu_power_variability.pdf) · [MD](articles/gpu_power_variability.md) |

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
