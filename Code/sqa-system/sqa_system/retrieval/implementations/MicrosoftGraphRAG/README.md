# GraphRAG retriever


The GraphRag retriever is a document based retriever, meaning that it expects to receive a list of documents as input. It has been proposed by Edge et al. [Edge2025](https://arxiv.org/pdf/2404.16130) and is available on [GitHub](https://github.com/microsoft/graphrag).

We have implemented the official GraphRAG retriever in the SQA system.

> Note: This retriever has not been used in any of our experiments as it is not a traditional Knowledge Graph based retriever. Our intention was to also test document based retrievers, but we did not have the time to do so.