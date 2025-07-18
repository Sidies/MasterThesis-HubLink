# Direct Fact Retrieval Retriever (DiFaR)
Direct Fact Retrieval (DiFaR) is a knowledge graph-based retrieval approach by Baek et al.[Baek2023](https://arxiv.org/abs/2305.12416). They do not provide a repository for their code.

The implementation of DiFaR in this folder is based on their descriptions in the paper.

> This retriever has been used in our experiments

## Approach Explanation

The DiFaR approach first uses an indexing phase, where it converts all triples in the knowledge graph into a set of embeddings. Then when a question is given, the question is also embedded using the same embeddings model that was used for the conversion of the triples. Then a nearest neighbor search is performed to quickly search over potentially billions of the dense vectors to locate those triples whose embeddings are closest to the question embedding. These triples are the contexts that are retrieved for the question.

They also propose to rerank the retrieved triples to further refine the results which they call DiFaR2. This approach uses a language model that is given the question and the retrieved triples to rank the triples based on their relevance to the question.