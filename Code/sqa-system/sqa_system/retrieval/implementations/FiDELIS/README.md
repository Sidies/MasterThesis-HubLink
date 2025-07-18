# FiDeLiS Retriever

## Overview

The FiDeLiS approach combines beam search with embedding techniques. At query time, it accepts both a question and a topic entity, then incrementally traverses the graph starting from that entity. Initially, a keyword extraction is performed on the question using an LLM and the extracted keywords are transformed into low-dimensonal vectors via an embedding model. Next, the neighboring entities and relations of the topic entity are also converted into vectors using the same model, and the vectors most similar to these keywords are identified. Paths deemed relevant based on their score are then further expanded. By applying beam search, a path through the graph is iteratively planned and extended until either a final answer can be generated or the maximum number of steps is reached. 

The approach is authored by Sui et al. in the paper [FiDeLiS: Faithful Reasoning in Large Language Model for Knowledge Graph Question Answering](https://arxiv.org/abs/2405.13873)

We implemented the retriever by adapting the offical code from the [FiDeLiS repository](https://anonymous.4open.science/r/FiDELIS-E7FC).

> This retriever has been used in our experiments

## Approach Explanation

The FiDeLiS approach is initiated with an input consisting of a question and a corresponding topic entity, that is used as an entry to the graph. A large language model (LLM) is then employed to generate a strategic plan for addressing the question. This includes extracting relevant keywords and converting the interrogative query into a declarative statement. Subsequently, the extracted keywords are embedded using a pre-trained embedding model, after which the main iterative process begins. This process commences by retrieving all relational paths associated with the topic entity. Both the predicates and associated entities of these relations are embedded and subsequently scored based on their similarity to the keyword embeddings. The resulting relations are ranked according to their scores, and only the top-N relations are retained. These are added to a cumulative list of candidate paths. If the maximum path length has not yet been reached, the top-K candidates from the accumulated list are selected for further expansion, guided once again by the previously generated plan. At each iteration, a deductive termination check is conducted to determine whether the process should halt and a final answer should be synthesized from the candidate paths. The loop continues until either the predefined step limit is reached or a final answer is successfully produced. 


## Detailed Steps
1. The approach starts with a question and a list of topic entities
2. It Starts by querying an LLM
    1. to create a “Plan” on how the question should be approached
    2. to extract keywords from the question
    3. to transform the question into a declarative statement
3. It then embeds those keywords
4. Then a Loop starts that runs up until the maximum amount of steps defined
5. If it is not the first loop run, it does a `deductive termination` step where it checks whether it should be stopped
6. It starts with the topic entity and gets its “paths”. This is done by getting all relations of the topic and also embedding both the predicate and the entity of all the relations
7. It then scores these relations against the embedded keywords and sorts them by their score
8. It then prunes those paths to only the $top_n$ paths
9. These paths are then added to a total `list of candidate paths`
10. Then if the maximum length has not been reached it decides for the top $K$ candidates from the `total candidates list` which top $k$ paths should be further explored
    1. For this prompt it also uses the planning steps
11. It then proceeds from steps 5 with the selected candidates


## Implementation

We adapted the original code with minimal necessary chances to work with the interface of the SQA system. Still, some changes were necessary. First, we had to adapt the parsers to make them more robust when working with a variety of different LLMs. This is because the original parsers were only able to parse correctly when the output was exactly as expected. Second, we adapted the code to also return the output triples to be able to evaluate the retrieval performance of the method. Third, we encountered very long execution times when running the FiDeLiS retriever. To improve the performance, we implemented a caching mechanism to avoid redundant calls to the graph. We also parallelized the scoring of the entities to speed up the process.