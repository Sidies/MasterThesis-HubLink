# LightRAG Retriever

The LightRag retriever is a document based retriever, meaning that it expects to receive a list of documents as input. It has been proposed by Guo et al. [Guo2024](https://arxiv.org/abs/2410.05779) and since then has gone open source with a publically available repository on [GitHub](https://github.com/HKUDS/LightRAG). It is constantly evolving and shows state-of-the-art performance for Retrieval Augemented Generation (RAG) tasks.

We have implemented the official LightRAG retriever in the SQA system.

> Note: This retriever has not been used in any of our experiments as it is not a traditional Knowledge Graph based retriever. Our intention was to also test document based retrievers, but we did not have the time to do so. 