### Application on Research Questions

To validate the practical applicability and descriptive power of the proposed taxonomy, we utilize research questions extracted from the ICSA/ECSA publication dataset. Initially, the LLM-based extraction component of the SQA system was employed to extract research questions from the entire corpus of 153 publications. This process yielded a total of 231 research questions potentially relevant to the domain.

The selection of a subset of these questions for detailed analysis adhered to two primary criteria:
1.  **Diversity of Origin:** Ensuring questions are sourced from a wide range of publications to capture varied research intents.
2.  **Random Selection:** Employing a random sampling method to minimize selection bias.

To implement this selection, a script was developed to perform stratified random sampling. This script utilized the pre-existing classification of publications within the dataset based on their *research level* and *paper class* to establish sampling categories. Publications lacking identifiable research questions were excluded prior to sampling. Subsequently, 20 questions are drawn uniformly across the categories using a random seed. The random seed was set to "2025", the year of publication for this thesis to guarantee reproducibility of the selection process.

Following the selection, the 20 research questions were manually classified according to our proposed taxonomy ($\mathcal{T}_3$). To address research question **Q2.3** regarding the significance and added value of the taxonomy, a comparative analysis against existing classification schemes was necessary. To our knowledge, a standardized taxonomy does not exist for categorizing questions in the KGQA field. Therefore, we determined that existing KGQA benchmark datasets, which incorporate question classifications, serve as the most appropriate references. Consequently, the classification schemes from DBLP-QuAD [1] and LC-QuAD 2.0 [2] were selected for comparison. While SciQA represents another potential source, its classification categories were deemed too dataset-specific for general applicability in this context and were therefore excluded.

**Assumptions Regarding Underlying Knowledge Graph Structure**
The application and interpretation of the KGQA taxonomies inherently depends on assumptions about the underlying KG structure that a KGQA system would query to answer the classified questions. To apply the taxonomies, it is essential to understand the structure of the underlying KG on which the questions are based. Therefore, we made the following assumptions for the graph that contains the relevant data:

1.  **Granularity of Representation:** The KG is assumed to store information in semantically distinct nodes, facilitating a fine-grained representation. Consequently, answering most questions necessitates accessing and potentially combining information from multiple nodes unless the query explicitly targets a single atomic datum representable within one node.
2.  **Absence of Pre-computation:** It is assumed that answers, particularly for complex questions such as aggregation, comparison, or counting, are not pre-computed and stored within single nodes. Instead, deriving such answers involves retrieving and integrating information distributed across multiple nodes at query time.

These assumptions align with common practices in KG design where complex facts are decomposed, and imply that question complexity, as captured by the taxonomy, correlates with the complexity of the required graph traversal and data integration operations.

#### Validating the Significance

The question **Q2.3** from the GQM plan (provided in the taxonomy validation table - tab:gqm_taxonomy_validation), investigates whether the proposed KGQA retrieval taxonomy offers a significant improvement in descriptive precision compared to existing schemes. This is evaluated using the *classification delta* metric (**M2.3.1**), which quantifies the difference in classification granularity. Manual classification of the 20 selected research questions was performed using our taxonomy ($\mathcal{T}_3$) and the categories provided by DBLP-QuAD and LC-QuAD 2.0. As such we measure $classification\_delta(\mathcal{T}_3, \{\text{DBLP-QuAD}, \text{LC-QuAD 2.0}\}, \mathcal{C}) = \frac{8}{15} = 0.533$. This indicates, that applying $\mathcal{T}_3$ resulted in the use of 15 distinct classifications from our taxonomy to classify the 20 questions. In contrast, applying the classification schemes from DBLP-QuAD and LC-QuAD 2.0 to the same set of questions resulted in the utilization of a maximum of eight distinct classifications. According to metric **M2.3.1**, this demonstrates that $\mathcal{T}_3$ offers substantially higher granularity, enabling a more precise description and differentiation of the research questions based on their KGQA retrieval characteristics.

Moreover, we found that the classification of classes from DBLP-QuAD and LC-QuAD 2.0 possess a limited number of categories, leading to coarse-grained classifications that often fail to adequately distinguish the varying structural and semantic complexities in the research questions. Furthermore, the lack of a clear hierarchical structuring or categorization by concern within these schemes hinders the systematic analysis of questions.

These quantitative and qualitative findings support the conclusion that the proposed KGQA retrieval taxonomy addresses a notable gap. This is, because it provides a structured and fine-grained framework for classifying questions intended for KGQA systems, with each category addressing distinct retrieval characteristics and potential challenges. We expect such a classification to be valuable not only for understanding the nature of questions but also for assessing the capabilities and limitations of KGQA systems.


------------
[1]: # [1] DBLP-QuAD: A Question Answering Dataset over the DBLP Scholarly Knowledge Graph, Banerjee et al. 2023
[2]: # [2] LC-QuAD 2.0: A Large Dataset for Complex Question Answering over Wikidata and DBpedia, Dubey et al. 2019: 10.1007/978-3-030-30796-7_5