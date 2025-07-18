# Think-on-Graph

> This retriever has not been used in our experiments because it showed that it is unable to retrieve our experimental data in the parameter selection process

## Overview
Think-on-Graph (ToG) is a knowledge graph-based retrieval approach by Sun et al. [Sun2024](https://arxiv.org/pdf/2307.07697.pdf). The corresponding repository for the code can be found [here](https://github.com/IDEA-FinAI/ToG/). 

ToG is one of the retrieval approaches we used as a baseline. For this to work, we adapted their code to work with our project.

## Approach Explanation
The retriever begins with the initialization phase, where the question is analyzed to identify key entities. A LLM extracts the main topic entities and generates an initial set of top-N entities. Once these entities are identified, the system proceeds to the exploration phase.

During exploration, the system iteratively traverses a KG to build reasoning paths. At the start of each iteration, the current set of reasoning paths includes all entities and relations discovered so far. The LLM identifies candidate relations by querying the KG for relations connected to the tail entities of the previous iteration. These relations are then ranked by their relevance to the question, and the top-N are selected in a LLM based pruning step to narrow the search space.

Next, the LLM uses the selected relations to find candidate entities, which are randomly pruned to stay within a predefined threshold given in the parameters. The reasoning paths are then updated with the newly discovered entities and relations, effectively increasing the depth of the reasoning paths by one with each iteration.

The reasoning phase follows, which involves evaluation and potential termination. The LLM evaluates whether the current reasoning paths contain enough information to answer the question. If so, it generates an answer using these paths. If not, the exploration continues until either an answer can be formulated or a predefined maximum depth is reached. If not sufficient information is found at that point, LLM resorts to its internal knowledge to produce a response.

## Implementation
We adapted their code with minimal necessary chances to work with the interface of the SQA system. During testing, we encountered several issues that required further adjustments to the original implementation. First, we had many executions that failed and some LLMs did not work with the retriever due to deviations from the expected output format. We found that the original parser was unable to handle variations in output formatting. To address this, we developed a more robust parser capable of extracting from a broader range of LLM outputs. Second, we parallelized the entity searching and scoring processes, as the original implementation proved to be inefficient in terms of execution speed.