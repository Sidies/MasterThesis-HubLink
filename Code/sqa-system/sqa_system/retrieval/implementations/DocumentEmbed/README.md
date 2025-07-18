# DocumentEmbed Retriever

## Overview
[Lewis et al.](https://arxiv.org/abs/2005.11401) have proposed Retrieval-Augmented-Generation (RAG) to allow Large Language Models (LLMs) to respond to questions by retrieving relevant information from a source of documents embedded into a low-dimensional vector space. The DocumentEmbed retriever is an implementation of this approach. It embeds the texts of publications in a pre-processing step and stores them into a database supporting vector retrieval. At query time, the retriever embeds the question and retrieves the most similar documents from the database. The retrieved documents are then passed to the LLM to generate the final answer. 

## Implementation
This is an implementation of the retriever following the descriptions of the paper.