## Review Manual

Thank you for participating in the code review. The following manual will guide you through the review process. You are participating in the second codereview.

The first review focused on the overall project structure, the [Core Package](../../sqa-system/sqa_system/core/) and an older version of the [HubLink Retriever](../../sqa-system/sqa_system/retrieval/implementations/HubLink/hub_link_retriever.py). The second review (this one) should focus on the new version of the [HubLink Retriever](../../sqa-system/sqa_system/retrieval/implementations/HubLink/hub_link_retriever.py) 

### <u>**1. Introduction**</u>

#### 1.0. Background

Finding relevant publications is a daunting task for researchers. The amount of scientific literature is growing exponentially and the statements and results are often stored in PDF format which makes it difficult to extract the information needed. Knowledge Graphs (KGs) are a promising approach to help researchers to find relevant publications more efficiently. KGs are a structured representation of knowledge that can be used to store and retrieve information.

A special type of KGs are Research Knowledge Graphs (RKG) that in addition to typical data such as people, documents, datasets,
and institutions, also contains statements from scientific articles stored as semantic resources. Using the information of such an RKG for literature search can help researchers to find relevant publications more efficiently. However, the navigation of such a graph is not trivial as it requires the users to have a good understanding of the graph structure. A promising approach to help with this is to use a Large Language Model (LLM) to allow users to ask question in natural language and receive answers from the graph without the need to understand the graph structure. The approach of applying LLMs to KGs for retrieval in a QA setting recently gained traction in the scientific community and its results are promising. However, it is a relatively new field. There is currently no established approach for this task but rather a classifcation of different approaches. I don't want to go into too much detail here, but a good taxonomy of current retrieval approaches is given in the paper [Graph Retrieval-Augmented Generation: A Survey](https://arxiv.org/abs/2408.08921):

<u>Taxonomy for Retrievers:</u>

1. Model Categorization
- Non-parametric Retrievers: Use heuristic rules or traditional graph search algorithms without deep-learning models.
- LM-based Retrievers: Utilize pre-trained language models like GPT-4 to process and interpret natural language queries.
- GNN-based Retrievers: Leverage Graph Neural Networks (GNNs) to encode and understand complex graph structures.
2. Retrieval Paradigm
- Once Retrieval: Access the graph with a single query to obtain all necessary information.
- Iterative Retrieval: Perform multiple sequential retrieval steps, with each step depending on previous results.
- Multi-stage Retrieval: Divide the retrieval process into distinct phases, each performing different tasks.
3. Retrieval Granularity
- Node-based: Retrieve individual nodes from the graph.
- Triplet-based: Retrieve subject-predicate-object relationships as tuples.
- Path-based: Retrieve sequences of relationships between nodes.
- Subgraph-based: Retrieve entire subgraphs for comprehensive relational contexts.
4. Training Strategy
- Training-Free Retrievers: Directly applied without requiring a training phase, often using pre-trained models.
- Training-Based Retrievers: Require a training or fine-tuning phase to adapt to specific tasks.

But the application of such retrievers on a RKG is currently an unexplored area. We hypothesize that utilizing the structure of RKGs in the retrieval process can improve the performance of the retrieval approach. 

#### 1.1. Goal
> Our goal with this master thesis is to create a new Retrieval Approach for the use in a Question Answering setting on Research Knowledge Graphs (RKG). 

<u>Requirements:</u>
We define the following requirements for our retrieval approach:
1. The approach should be applied to the graph to answer questions in natural language.
2. The approach should use a LLM to support the retrieval process.
3. The approach should operate without training or fine-tuning since no supervised dataset with sufficient data is available.
4. The approach should be able to function without a starting point in the graph.
5. The approach should be capable of answering meaningful questions that support literature search and consider multiple hops in the graph.
6. The answer of the approach should be transparent, meaning that the sources of the answer should be provided.

#### 1.2. Approach
As such we developed a new retrieval approach we call HubLink Retrieval which is implemented [here](../../sqa-system/sqa_system/retrieval/implementations/HubLink/hub_link_retriever.py). The naming comes from the concepts of hubs and links applied in the approach. 
- A **hub** is a special concept used in the retrieval approach that holds particular significance for a specific domain or research question. Hubs consolidate information and serve as pivotal points for searching relevant data.
- **A link** is a connection between a hub and the source of information that is used to further enrich the hub. Links are used to retrieve additional information about a hub with information outside of the graph.

**Categorization for the Taxonomy:**
Our retrieval approach has two strategies and as such slightly different categories
1. No Topic Entity Strategy: LM-based, Once Retrieval, Subgraph-based, Training-Free
2. Topic Entity Strategy: LM-based, Multi-stage Retrieval, Subgraph-based, Training-Free

#### 1.3. Evaluation
To evaluate the performance of our approach, we create a question answering dataset using a semi-automated approach that generates questions and answers that are further reviewed by a human. These questions are then used to evaluate the performance of the retrieval approach in our experimentation setting using the SQA System explained below.

<u>Metrics:</u>
We use several metrics to evaluate the performance of our approach. I already started preparing a Wiki page with the metrics we use [here](https://gitlab.com/software-engineering-meta-research/ak-theses/mastertheses/ma-marco-schneider/implementation/-/wikis/pages/Metrics) however the page is still work in progress and does not yet contain all metrics we are using.

<u>Baseline:</u>
We compare our approach to several baseline implementations. These can be subdivided into two categories:
1. **Document Focused**: These approaches receive as input a list of documents and a question. The performance that is evaluated is whether the correct information from the documents are retrieved.
    - **Document Embed**: This approach is the current state-of-the-art approach for document retrieval. It uses a pre-trained language model to encode the documents and the question and retrieves the most similar document. The implementation can be found [here](../../sqa-system/sqa_system/retrieval/implementations/DocumentEmbed/document_embed_retriever.py).
    - **LightRag**: This is a new retrieval approach which shows promising results for document retrieval. It uses a hybrid approach where text documents are both embedded and stored in a graph. After its paper publication the project became open-source and can be found [here](https://github.com/HKUDS/LightRAG). We implemented the approach [here](../../sqa-system/sqa_system/retrieval/implementations/LightRag/light_rag.py).
2. **Graph Focused**: These approaches receive as input a graph and a question. The performance that is evaluated is whether the correct information from the graph is retrieved.
    - **TripleEmbed**: This approach embeds the triples of the graph and the question and retrieves the most similar triple. The implementation can be found [here](../sqa-system/sqa_system/retrieval/implementations/TripleEmbed/triple_embed_retriever.py).
    - **StructGPT**: This approach uses a pre-trained language model to traverse the graph starting from a entry point named the "topic entity". There GitHub can be found [here](https://github.com/RUCAIBox/StructGPT). The implementation can be found [here](../../sqa-system/sqa_system/retrieval/implementations/StructGPT/struct_gpt_main.py).
    - **Think-on-Graph**: This approach uses a pre-trained language model to traverse the graph starting from a entry point named the "topic entity". There GitHub can be found [here](https://github.com/GasolSun36/ToG). The implementation can be found [here](../../sqa-system/sqa_system/retrieval/implementations/ToG/main_sparql.py).


#### 1.4. Framework
For the implementation and evaluation of this approach, we constructed the Scholarly Question Answering (SQA) System or Framework which is located in the [sqa-system](../../sqa-system) directory. This [image](../assets/overview-Architecture.svg) shows the architecture of the SQA System. 

At the core it provides the following functionalities:
1. **Experimentation**: The SQA System provides a way to evaluate different retrieval approaches [see experimentation](../../sqa-system/sqa_system/experimentation/).
2. **Knowledge Bases**: The SQA System provides implementations of Knowledge Graph and Vector Store knowledge bases [see knowledge_base](../../sqa-system/sqa_system/knowledge_base/).
3. **Configuration**: The SQA System provides a extensive configuration system that allows to configure the components of each run. Using this, each experiment can be easily reproduced [see configuration](../../sqa-system/data/configs/).
4. **Retrieval**: We implemented our new retrieval approach and baseline retrieval approaches [see retrieval](../../sqa-system/sqa_system/retrieval/).
5. **QA-Generation**: The SQA System provides implementations to generate questions and answers for the evaluation of the retrieval approaches [see qa_generation](../../sqa-system/sqa_system/qa_generation/).






### <u>**2. First Task - Read the Preparation Material**</u>
1. Read through the provided [PDF 1](./code_review_2_overview.pdf) to get an overview of our HubLink retriever and the SQA System.

### <u>**3. Second Task - Review the Pseudo Code of HubLink**</u>
2. Read through the provided [PDF 2](./hublink_retriever_chapter.pdf) which is the section for the HubLink retriever in the master thesis.
    1. Review the Pseudo-code structure, understandability, consistency, and best practices.
    2. Review the explanation of the Pseudo code for understandability and anything that might be missing for a reader to understand the code.




### <u>**4. Third Task - Review the Implementation Code**</u>

**Preparation:**
1. Clone the repository from [GitHub](https://gitlab.com/software-engineering-meta-research/ak-theses/mastertheses/ma-marco-schneider/implementation.git) 

2. Checkout the branch "codereview" by running `git checkout codereview` in the terminal.

2. Follow the installation instructions from the [README.md](../../README.md) file.

When using the VDL-SDQ-Kastel [Server](https://vdl.sdq.kastel.kit.edu/), you also need to do the following steps:

3. Open a terminal and make sure you are in your virtual environment.
4. Run `pip install ipykernel`
5. Run `python -m ipykernel install --user --name=venv --display-name "SQA-System virtual Env"`. This will prepare a kernel for the Jupyter Notebook. When running the Jupyter Notebook, make sure to select the kernel "SQA-System virtual Env" from the top right corner.

**How to review:**
Review the code with the following aspects where applicable: 
1. Understandability
2. Consistency
3. Best Practices
4. Extendability
5. Completeness


**What to review:**

<!-- Link stimmt nicht -->
1. Open the Codereview demonstration notebook and follow the instructions. Did everything work? If not what did not work?
    - We provide two notebooks. Choose <u>one</u> of the following:
        - [Notebook (requires OpenAI Key)](../../sqa-system/notebooks/codereview/codereview_openai.ipynb)
        - [Notebook (requires Huggingface Key and Nvidia GPU)](../../sqa-system/notebooks/codereview/codereview_huggingface.ipynb)


4. Review the code in the [sqa_system/retrieval/implementations/HubLink](../../sqa-system/sqa_system/retrieval/implementations/HubLink/hub_link_retriever.py) directory. 

    - It includes the implementation of the HubLink retriever. You should by now have a basic understanding on the functionality of the HubLink retriever from the PDFs provided above. Apply this knowledge while reviewing the code. Do you have further suggestions for improvement?

5. Review the ExperimentRunner 

    - It is located in [sqa_system/experimentation](../../sqa-system/sqa_system/experimentation/experiment_runner.py) and responsible for executing the experiments, creating the metrics and visualize the results. You may also look at the classes that the ExperimentRunner uses.


6. Review the Knowledge Graph Implementation. 

    - To implement KGs in the system, the SQA-System provides two base classes [Knowledge Graph](../../sqa-system/sqa_system/knowledge_base/knowledge_graph/storage/base/knowledge_graph.py) which is the interface from which all queries to the KG are performed from the system and [Knowledge Graph Factory](../../sqa-system/sqa_system/knowledge_base/knowledge_graph/storage/factory/base/knowledge_graph_factory.py) which is the interface for creating the KGs. Both of these classes need to be implemented when a new KG is added to the system.
    - We have two implementations for the `Knowledge Graph Base`: [RDFLib Graph](../../sqa-system/sqa_system/knowledge_base/knowledge_graph/storage/implementations/rdflib_knowledge_graph.py) and [ORKG Graph](../../sqa-system/sqa_system/knowledge_base/knowledge_graph/storage/implementations/orkg_remote_graph.py). Each of these implementations has a corresponding factory class that is responsible for creating the KGs: [RDFLib Graph Factory](../../sqa-system/sqa_system/knowledge_base/knowledge_graph/storage/factory/implementations/rdflib_knowledge_graph/rdflib_knowledge_graph_factory.py) and [ORKG Graph Factory](../../sqa-system/sqa_system/knowledge_base/knowledge_graph/storage/factory/implementations/orkg_knowledge_graph/orkg_knowledge_graph_factory.py).
    - After the implementation, the factory needs to be registered in the [Knowledge Graph Factory Registry](../../sqa-system/sqa_system/knowledge_base/knowledge_graph/storage/knowledge_graph_factory_registry.py) and that's it. The system will now be able to use the new KG by loading it through the [Knowledge Graph Manager](../../sqa-system/sqa_system/knowledge_base/knowledge_graph/storage/knowledge_graph_manager.py) which is responsible for the distribution of the KGs to the system.

7. Review the Retriever Implementation

    - To implement new Retrievers in the system, two base classes are provided depending on the type of the retriever. If the retriever expects a dataset as input in addition to the question, the retriever is `DocumentFocused` and should inherit from the [DocumentRetriever](../../sqa-system/sqa_system/retrieval/base/document_retriever.py) class. If the retriever retrieves its knowledge from a knowledge graph, it should inherit from the [KnowledgeGraphRetriever](../../sqa-system/sqa_system/retrieval/base/knowledge_graph_retriever.py) class.
    - When implementing retrievers, the SQA-System provides tools like [LLMAdapter](../../sqa-system/sqa_system/core/language_model/base/llm_adapter.py) for using an LLM, an [EmbeddingAdapter](../../sqa-system/sqa_system/core/language_model/base/embedding_adapter.py) for using embeddings, or the Knowledge Bases [KnowledgeGraph](../../sqa-system/sqa_system/knowledge_base/knowledge_graph/storage/base/knowledge_graph.py) and [VectorStore](../../sqa-system/sqa_system/knowledge_base/vector_store/storage/base/vector_store_adapter.py) that can be used for realising the retriever's functionality.
    - After the implementation the retriever needs to be registered in the appropriate [DocumentRetrieverFactory](../../sqa-system/sqa_system/retrieval/factory/document_retriever_factory.py) or [KnowledgeGraphRetrieverFactory](../../sqa-system/sqa_system/retrieval/factory/knowledge_graph_retriever_factory.py) depending on its type. The retriever can now be used in the system. For example it it automatically selectable during the creation of a new experiment configuration using the [CLIController](../../sqa-system/sqa_system/app/cli_controller.py).

    
#### 3. Conclusion
1. In short, what is your understanding of the HubLink retriever? (How does it work?)
2. Do you have any further suggestions for improving the HubLink retriever?
5. Do you have further suggestions for improving the codebase?

Thank you for your time and effort. Please send your review to marco.schneider@student.kit.edu