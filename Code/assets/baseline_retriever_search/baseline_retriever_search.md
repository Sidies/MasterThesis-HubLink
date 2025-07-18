
# Baseline Retriever Search

To compare HubLink with state-of-the-art methods from the literature, we selected and implemented several approaches drawn from recent publications. The chosen baseline approaches represent established methods within the field previously evaluated on open-domain KGQA benchmarks. In the following, we outline the systematic process through which these methods were chosen.



## How we gather the papers:
Recently, surveys have been published that structure the current approaches found in the literature. [1] provide an in-depth analysis of the integration of LLMs with KGs, which includes KGQA but also goes beyond that. In their work, they propose a categorization of different integration strategies, assigning each examined paper to one of these categories. From this structure, we selected the categories relevant to KGQA, namely KG-Enhanced LLM - Inference, LLM-Augmented KG - Question Answering, and Synergized Reasoning, as these directly address the integration of LLMs and KGs for question answering tasks.
Furthermore, [2] propose a taxonomy for GRAG approaches, which classifies the methods in a range of dimensions. From this set, we include all publications covered by the survey, except those classified as Non-parametric, GNN-based retrievers, or those considered Training-based.
In addition to the surveys, we conducted a Google Scholar search to identify further KGQA approaches. Since both surveys were published in 2024, we limited our search to this year in order to find additional approaches not yet captured by the surveys.

[1] Unifying Large Language Models and Knowledge Graphs: A Roadmap, Pan et al. 2024: 10.1109/TKDE.2024.3352100
[2] Graph Retrieval-Augmented Generation: A Survey, Peng et al. 2024: 10.48550/arXiv.2408.08921


## Scholar Search:
To find papers that have been published after the publication of the surveys, we conducted a search on Google Scholar. We filtered by year to only include papers published in 2024. Moreover, we limited our search to the first 20 pages of the search results. We used the following queries:

- "Knowledge Graph Question Answering" [649 results]
- "Knowledge Graph" and "Question Answering" [8.020 results]
- "KGQA" [385 results]

To decide whether a paper from the scholarly search is relevant, we read the abstract to check whether the paper mentions a KGQA approach. If additionally the abstract does not indicate that the approach is training-based, we include the paper in our list. We also check whether the paper is already included in the surveys. If it is, we do not include it again. 

# Filtering Process:
Through the surveys and Google Scholar search, we collected an initial pool of 76 publications. The next step was to identify the KGQA approaches most relevant for comparison with HubLink by applying the following exclusion criteria:

1.  **[C1] LLM-Based:** The approach must employ a pre-trained LLM to support the retrieval process. Embedding models are also included under this criterion. This is relevant because our objective is to explore how LLMs can support literature search within a QA context.
2.  **[C2] KGQA Approach:** The method must represent a generalizable KGQA approach. Specifically, it should accept a natural language question and a KG as input, with the goal of extracting relevant information from the KG to answer the question.
3.  **[C3] Training-Free:** The approach must not require additional training or fine-tuning of pre-trained LLMs, nor the training of other models such as GNNs. Approaches that depend on a dataset of training examples are excluded, as we lack the resources for extensive training.

Applying these criteria to the set of 76 publications resulted in 13 relevant papers. Specifically, one paper was excluded for not using an LLM (**C1**), 21 were excluded as they did not represent a suitable KGQA approach (**C2**), and 41 were excluded for requiring model training (**C3**).

## Assessing Implementation Feasibility

From the 13 relevant papers, we evaluated the availability and applicability of the implementations provided by the authors for integration into the SQA system.

The approaches RoK and KSL were excluded as they do not provide source code, and their complexity made reimplementation impractical without it.

In the case of KG-GPT, after reviewing the code repository[^1], we found that the implementation corresponds to a claim verification pipeline rather than a traditional QA setting. It assumes a prior mapping of claim entities to graph entities, which is not available in our use case.

For ODA, although the authors provide a repository[^2], it contains only graph and dataset resources, lacking the implementation of the ODA approach itself.

DiFaR does not have a public implementation. However, its methodology closely resembles the RAG framework, differing mainly in embedding graph triples instead of documents. Given the experience of the thesis author with similar architectures, we deemed reimplementation feasible based on the description of the paper.

For the remaining methods: StructGPT, ToG, Mindmap, ToG-2, GoG, GRAPH-COT, and FiDeLiS, we found that the provided source code was generally adaptable for integration into the SQA system.

## Deciding on Final Implementations

After assessing implementation feasibility, eight of the 13 methods remained as candidates: StructGPT, ToG, Mindmap, ToG-2, GoG, GRAPH-COT, FiDeLiS, and DiFaR. To keep the scope of this work manageable, we ultimately selected five of these for implementation, guided by their methodological diversity. We categorized the eight candidates as follows:

* **Step-wise reasoning:** These approaches iteratively query the LLM to derive an answer step-by-step. This category included: StructGPT, ToG, ToG-2, GoG, GRAPH-COT, and FiDeLiS.
* **Subgraph construction:** These methods focus on building relevant subgraphs from which information is extracted. Mindmap was the sole candidate in this category.
* **Embedding-based:** These methods primarily use dense vector representations for retrieval. DiFaR was the only candidate here.

During the conceptual phase of this work, StructGPT and ToG were implemented to evaluate the general feasibility of the thesis. At the time, both were highly cited and had adaptable public code. However, they are similar in structure, and share a weakness particularly relevant to scholarly literature search: their entity selection can become random beyond a certain threshold, making correct entity identification reliant on chance.

Although ToG-2, a successor to ToG, was published during the conduction of this thesis, the issue described above was not resolved in the new version. For this reason, we decided not to implement this approach. Instead, we selected FiDeLiS from the Step-wise Reasoning category, as it specifically addresses the entity selection problem using dense vector similarity assessment. This brings the total number of implemented methods in the Step-wise Reasoning category to three.

In the Subgraph Construction and Embedding-based categories, only Mindmap and DiFaR remained after filtering. Therefore, both were implemented as baselines, bringing the total number of implemented baseline methods to five.

---
[^1]: [https://github.com/jiho283/KG-GPT](https://github.com/jiho283/KG-GPT) \[last accessed 24.11.2024]
[^2]: [https://github.com/Akirato/LLM-KG-Reasoning](https://github.com/Akirato/LLM-KG-Reasoning) \[last accessed 24.11.2024]
