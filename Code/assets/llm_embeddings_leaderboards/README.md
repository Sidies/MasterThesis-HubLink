When deciding on which LLMs and Embedding models we use for our experiments, we used the following leaderboards as a reference:

1. *Chatbot Arena Leaderboard*[^3], proposed by [1]. This leaderboard ranks LLMs based on their performance in various tasks, providing a comprehensive overview of the current state-of-the-art models.
2. *Massive Multilingual Text Embedding Benchmark (MMTEB)*[^4], introduced by [2]. This leaderboard focuses on the performance of embedding models across multiple languages and tasks, allowing us to assess the effectiveness of different models in generating high-quality embeddings.

This folder contains the snapshots of the leaderboards as of February 16, 2025.

## Selection of LLMs and Embedding Models

As all retrievers are based on LLMs, the model selection is crucial for the performance of the retriever. Since HubLink, DiFaR, MindMap and FiDeLiS also work with embeddings, the selection of the embedding model is equally important.

For our experiments and the selection process, we implemented the following endpoints: *OpenAI* as a proprietary provider, *Ollama* and *Huggingface* for open source models which are run locally on the server. Furthermore, when choosing which models to use, we considered the following points:

1.  The OpenAI endpoint is proprietary and can introduce high costs if not managed carefully. As such, we considered the associated costs of the models and how many models from OpenAI we are using.
2.  Through testing, we found the HuggingFace models to be less optimized than the Ollama ones. This means that the amount of hardware memory resources required to run models on the HuggingFace endpoint is higher than on the Ollama endpoint, which may lead to "Out-Of-Memory" errors.
3.  We are restricted to the hardware resources available on our server. We have `32 GB` of GPU memory available which is enough to fit LLMs of the size of `32B` parameters on the GPU. However, running embedding models in parallel is then not feasible. Moreover, even if a large model fits on the GPU, its response time is likely too slow to be used in our experiments. Consequently, we chose to use smaller models.

To help in the selection process, we reviewed popular leaderboards to assess the performance of the models available. We examined two leaderboards, both reflecting the status as of February 16, 2025. For LLMs, we examined the *Chatbot Arena Leaderboard*[^3], proposed by [1]. For embedding models, we observed the *Massive Multilingual Text Embedding Benchmark (MMTEB)*[^4], introduced by [2].

**Selection of LLMs**
We selected the following LLMs for our experiments. *GPT-4o*, because the model is ranked at the top most position in the Chatbot Arena leaderboard via the OpenAI endpoint. *GPT-4o-mini*, ranked 24th yet delivering strong performance at a fraction of the cost. Also *O3-mini*, a newly released model that inherently implements chain-of-thought reasoning [19]. To include open-source options, we chose *Qwen2.5*, which is the Ollama endpoint model that performs the best on the leaderboard. However, due to our hardware constraints, we had to reduce the model to its `14B` parameter variant. Furthermore, we also selected *Llama3.1*, which represents the second-best Ollama model in the leaderboard. But we had to scale it down to the `8B` parameter model because of hardware constraints. We also evaluated *DeepSeek-R1* [20], a new open-source reasoning model with promising benchmarks, but its performance-to-runtime ratio was substantially worse than that of our selected models, so we excluded it.

**Selection of Embedding Models**
For embedding models, we included *text-embedding-3-large*, the largest embedding model available via the OpenAI API. With regard to open-source models, we used *Mxbai-Embed-Large* model, which is a fast and popular open-source model, ranked 41st on the MMTEB leaderboard. Because it is very fast with good performance, it makes it a good choice for the Base-Configurations in our selection process. We also evaluated *Granite-Embedding*, a new Ollama endpoint model that is not yet on the leaderboard. Still, it is a promising model that is fast and looks to have a good performance. Finally, we tested *gte-Qwen2-7B-instruct*, the top-ranked MMTEB model, but it exhibited slow inference and unexpectedly poor performance. We are not entirely sure why the models performance was poor, but we suspect that it may be due to the fact that it was used over the HuggingFace endpoint, which uses unoptimized models. Ollama, on the other hand, provides expert optimization for their models which makes them faster and could make them perform better which is the reason we opted to use models from Ollama over those provided on HuggingFace.

---
[^3]: [https://huggingface.co/spaces/lmarena-ai/chatbot-arena-leaderboard](https://huggingface.co/spaces/lmarena-ai/chatbot-arena-leaderboard) \[last accessed on 16.02.2025]

[^4]: [https://huggingface.co/spaces/mteb/leaderboard](https://huggingface.co/spaces/mteb/leaderboard) \[last accessed on 16.02.2025]

[1] Chatbot Arena: An Open Platform for Evaluating LLMs by Human Preference, Chiang et al. 2024: 10.48550/arXiv.2403.04132

[2] MMTEB: Massive Multilingual Text Embedding Benchmark, Enevoldsen et al. 2025: http://arxiv.org/abs/2502.13595