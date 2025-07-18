

## Identifying Clustering Relevance
In the following we are going to discuss for each of the extracted clusters whether we see it relevant to be added to our taxonomy. To support this analysis we are using the clustering summary from the [Cluster Analysis](./clustering/cluster_analysis.ipynb) notebook. After the relevance of each cluster is assessed, we will be creating our initial taxonomy based on these clusters.

The structure of the following analysis is as follows:
1. First we are going to list a summary of the cluster which comes from the mentioned Notebook
2. Then we are going to give an analysis on whether we see this cluster relevant for our taxonomy

This assessment is based on the established name and description to determine if they align with the goal of the taxonomy. In particular a category is seen as relevant if each of the following questions holds true:

1.  **GQ1.** "Does the category capture characteristics of questions to test the capabilities of KGQA retrieval systems, or does it reflect aspects specific to the scholarly literature search domain?"
2.  **GQ2.** "Can the category be generalized across different KGQA systems and within the literature search domain?"
3.  **GQ3.** "Can the classes within the category inform and guide the construction of diverse and meaningful evaluation datasets?"

```
Parent-Cluster ID: 1000
Parent-Cluster Name: Knowledge Graph Representation | Knowledge Graph Organization | Fact Granularity
Parent-Cluster Info: Groups those clusters that describe how the data that is asked for is organized in the Knowledge Graph. It distinguishes between the amount of facts that are needed to answer a question.
Amount of subclusters: 2
Amount of types in all subclusters: 11
Amount of different domains: 3
Amount of different years: 5
Amount of different categories: 2
Amount of different sources: 6
Names of Subclusters: ['Single Fact', 'Multi Fact']
Domains: 3x Scholarly, 2x General, 1x Vietnamese Language
Years: 2x 2023, 1x 2019, 1x 2020, 1x 2015, 1x 2017
Categories: 5x Knowledge Graph Question Answering Dataset, 1x Question Classifier
Sources:  ['DBLP-QuAD: A Question Answering Dataset over the DBLP Scholarly Knowledge Graph', 'The SciQA Scientific Question Answering Benchmark for Scholarly Knowledge', 'LC-QuAD 2.0: A Large Dataset for Complex Question Answering over Wikidata and Dbpedia', 'Question Answering on Scholarly Knowledge Graphs', 'Large-scale Simple Question Answering with Memory Networks', 'Ripple Down Rules for question answering']
```

**Cluster Description:** The first cluster is about the knowledge representation in the knowledge graph. It distinguishes between questions that require multiple facts to be answered and questions that require only a single fact. The metadata indicates, that this cluster originates from the development of KGQA datasets, with most contributions emerging the last five years. This indicates that the distinction between single fact and multi fact is a foundational aspect for many Knowledge Graph based QA systems. Moreover, the breadth of contributions covers both scholarly and general domains indicating that the distinction between single- and multi-fact is broadly applicable.

**Added to Taxonomy:** Yes

**Rationale:** This category captures structural aspects of question complexity by distinguishing between questions that require the retrieval of single versus multiple facts. This distinction directly impacts the complexity of the retrieval process and is broadly applicable across KGQA systems. Moreover, it is particularly relevant in the scholarly domain, where complex questions are common. The cluster also offers concrete guidance for the construction of datasets that vary in granularity with respect to graph traversal.

```
Parent-Cluster ID: 1001
Parent-Cluster Name: Answer Type | Answer Output Types & Formats
Parent-Cluster Info: Defines the expected answer format or data type. As such it classifies what the question expect to be in the answer.
Amount of subclusters: 32
Amount of types in all subclusters: 87
Amount of different domains: 10
Amount of different years: 17
Amount of different categories: 4
Amount of different sources: 25
Names of Subclusters: ['Generic Answer Type', 'Undefined', 'Date', 'Other', 'Distance Measurement', 'Actor', 'Technology', 'Expected Answer Type (EAT)', 'Asking Point Given', 'Definition', 'Time', 'Name', 'Title', 'Bibliometric Numbers', 'Manner', 'Software System', 'Monetary', 'Abbreviation', 'Procedural or Instructional', 'Organization', 'Duration', 'Boolean', 'Entity', 'Description', 'Properties', 'Human/Person', 'Location', 'Quantitative', 'Tool/Notation', 'Solution', 'Report', 'Theoretical Framework']
Domains: 11x General, 4x Scholarly, 3x Software Engineering, 1x Vietnamese Language, 1x Spoken Natural Language Processing, 1x Covid, 1x Healthcare, 1x Social Science, 1x Requirements Engineering, 1x Design Science
Years: 1x 2016, 2x 2000, 1x 2021, 1x 1999, 1x 1984, 1x 2024, 1x 2009, 1x 2007, 1x 2017, 2x 2015, 1x 2002, 1x 2003, 2x 2022, 3x 2023, 3x 2019, 2x 2020, 1x 2008
Categories: 9x Question Classifier, 8x Knowledge Graph Question Answering Dataset, 6x Research Questions, 2x Other
Sources:  ['The question answering systems: A survey', 'The Structure and Performance of an Open-Domain Question Answering System', 'What is in the KGQA Benchmark Datasets? Survey on Challenges in Datasets for Question Answering on Knowledge Graphs', 'AT&T at TREC-8', 'The Classification of Research Questions', 'A Rule-based Question Answering System for Reading Comprehension Tests', 'Hybrid-SQuAD: Hybrid Scholarly Question Answering Dataset', 'Learning foci for Question Answering over Topic Maps', 'The Future of Empirical Methods in Software Engineering Research', 'Ripple Down Rules for question answering', 'Linguistically Motivated Question Classification', 'Learning Question Classifiers', 'Writing good software engineering research papers', 'A Non-Factoid Question-Answering Taxonomy', 'DBLP-QuAD: A Question Answering Dataset over the DBLP Scholarly Knowledge Graph', 'The SciQA Scientific Question Answering Benchmark for Scholarly Knowledge', 'LC-QuAD 2.0: A Large Dataset for Complex Question Answering over Wikidata and Dbpedia', 'Question Answering on Scholarly Knowledge Graphs', 'A Comparative Study of Question Answering over Knowledge Bases', 'Selecting Empirical Methods for Software Engineering Research', 'Formulation of Research Question - Stepwise Approach', 'Types of research questions: descriptive, predictive, or causal', 'A Taxonomy for Classifying Questions Asked in Social Question and Answering', 'Divide and Conquer the EmpiRE: A Community-Maintainable Knowledge Graph of Empirical Research in Requirements Engineering', 'Construction of Design Science Research Questions']
```

**Cluster Description:** The second cluster is about the expected answer type or format. It classifies a given question by the expectation on how the answer should look like and what it should contain. The cluster spans a large number of 33 categories originating from 25 different sources. It covers a wide range of expected answer formats and types from undefined or ambiguous types to very concrete formats such as dates, monetary values, and bibliometric numbers. The metadata shows a broad temporal span ranging from 1984 to 2024. Furthermore, some concepts are considered by multiple sources while others are only represented by one or two sources. The category that was considered by the most sources is Boolean with 10 different sources, followed by Human/Person, Location, and Quantitative, which each appear in 7 sources. Date and Description are also common categories with 5 sources each. In addition, Undefined, Time, Organization, and Entity are considered multiple times with 4 sources each. The remaining categories are less commonly mentioned across multiple sources. Overall the cluster is the one most commonly cited in our reviewed literature covering a mix of domains that include the open domain, scholarly domain, software engineering domain, healthcare domain, and more. In addition, the cluster is widely covered in question classification, KGQA dataset creation, research question focus and other literature types.

**Added to Taxonomy:** Yes

**Rationale:** This category categorizes questions by the expected type of answer, such as dates, entities, quantities, or boolean values. These categories are helpful in describing the characteristics of questions in a literature search to construct diverse datasets. Furthermore, they are broadly applicable across domains and KGQA systems.


**Type Relevance:**
1. **Generic Answer Type**: 
    1. _Description:_ Generically defines the expected answer type.
    2. _Added to Taxonomy_: Yes
    3. _Rationale_: This type generally defines the expected answer type. As such it supports the naming of the cluster as `Answer Type`. However, it is too general to be used as a type inside of dimension.
2. **Undefined**: 
    1. _Description:_ Groups questions for which the expected answer type is unclear or cannot be determined by standard classification rules.
    2. _Added to Taxonomy_: Yes
    3. _Rationale_: This type could be useful for those cases where the question is unclear.
3. **Date**: 
    1. _Description:_ Classifies questions that expect a date as an answer.
    2. _Added to Taxonomy_: Yes
    3. _Rationale_: Relevant for those questions that ask for a specific date.
4. **Other**: 
    1. _Description:_ Classifies those questions that expect an answer type that does not fit in any of the other categories.
    2. _Added to Taxonomy_: Yes
    3. _Rationale_: Using this type would allow for a catch-all category for those questions that do not fit in any of the other categories. It is similar to the undefined type which is why we merge these.
5. **Distance Measurement**: 
    1. _Description:_ Classifies answers that expect a linear measure such as distance or length
    2. _Added to Taxonomy_: Yes
    3. _Rationale_: This type is relevant for those questions that ask for a distance measurement.
6. **Actor**: 
    1. _Description:_ Identifies questions that ask for an actor which is a individual, team, project, organization, or industry that performs an act.
    2. _Added to Taxonomy_: Yes
    3. _Rationale_: If a question asks for the actor in a specific context, this type is relevant.
7. **Technology**: 
    1. _Description:_ Classifies questions where the answer should be a technology which can be a process model, method, technique, tool, or language that is applied on the Software System
    2. _Added to Taxonomy_: Yes
    3. _Rationale_: In the SWA context, it is relevant to ask for technologies that are used in a specific context.
8. **Expected Answer Type (EAT)**:
    1. _Description:_ Relates to questions where the expected answer type is not explicitly given in the question but must be inferred from the context.
    2. _Added to Taxonomy_: No
    3. _Rationale_: Our classification task is intended to only provide the questions themselves without further context. Therefore, it is not possible to infer the expected answer type from an external context. It must be possible to determine the expected answer type from the question itself.
9. **Asking Point Given**: 
    1. _Description:_ Specifies those questions where the asking point is directly included in the question.
    2. _Added to Taxonomy_: No
    3. _Rationale_: As with the EAT type, every question that we consider with out taxonomy should be self-contained. Therefore, it is not necessary to have a type that specifies that the asking point is given in the question.
10. **Definition**: 
    1. _Description:_ Classifies questions that expect a definition or explanation, where the answer provides a clear definition or description of a concept.
    2. _Added to Taxonomy_: Yes
    3. _Rationale_: Relevant as asking for definitions is a common task in SWA literature research.
11. **Time**: 
    1. _Description:_ Classifies those questions that ask for a time.
    2. _Added to Taxonomy_: Yes
    3. _Rationale_: Relevant for those questions that ask for a specific time.
12. **Name**: 
    1. _Description:_ Focuses on answers that ask for names (or named entities) like persons, locations, or objects.
    2. _Added to Taxonomy_: Yes
    3. _Rationale_: Relevant for questions that ask for a specific name.
13. **Title**: 
    1. _Description:_ Covers answer types where the answer is expected to be a title such as the title of a publication or work.
    2. _Added to Taxonomy_: Yes
    3. _Rationale_: It is a common task in SWA literature research to ask for titles of publications.
14. **Bibliometric Numbers**:
    1. _Description:_ Specifies answers that involve bibliometric data such as publication counts, citation numbers, h-index, or i10-index.
    2. _Added to Taxonomy_: Yes
    3. _Rationale_: In the literature research task this type is relevant.
15. **Manner**: 
    1. _Description:_ Used when the answer is expected to detail the manner by which an action is performed.
    2. _Added to Taxonomy_: No
    3. _Rationale_: We consider this type to be too general for our taxonomy. 
16. **Software System**:
    1. _Description:_ Classifies answers that describe features of a software system, such as its size, complexity, or application domain.
    2. _Added to Taxonomy_: Yes
    3. _Rationale_: In the SWA context, it is relevant to ask for features of a software system.
17. **Monetary**: 
    1. _Description:_ Classifies questions that expect a monetary amount as an answer.
    2. _Added to Taxonomy_: Yes
    3. _Rationale_: Relevant for those questions that ask for a specific monetary amount.
18. **Abbreviation**: 
    1. _Description:_ Classifies questions that expect the long or short form of an abbreviation as an answer.
    2. _Added to Taxonomy_: Yes
    3. _Rationale_: To abbreviate terms is common in research literature. Therefore, this type is relevant.
19. **Procedural or Instructional**: 
    1. _Description_: Covers those questions that expect as answer step-by-step procedures, instructions, or techniques to achieve a specified task.
    2. _Added to Taxonomy_: Yes
    3. _Rationale_: This type defines the structure by which the answer should be given. As such it is relevant for our taxonomy.
20. **Organization**:
    1. _Description:_ Groups answers that identify organizations or human groups, including universities, research centers, laboratories, and other entities.
    2. _Added to Taxonomy_: Yes
    3. _Rationale_: This type is relevant for questions that ask for organizations like publishers or universities.
21. **Duration**: 
    1. _Description:_ Covers answers that express time durations such as 'three years' or 'few hours'.
    2. _Added to Taxonomy_: Yes
    3. _Rationale_: Relevant for those questions that ask for a specific duration.
22. **Boolean**:
    1. _Description:_ Classifies questions that expect an affirmative or negative response.
    2. _Added to Taxonomy_: Yes
    3. _Rationale_: This type reflects and overall expected type of an answer and as such is relevant for our taxonomy
23. **Entity**: 
    1. _Description:_ Used for questions where the answer is expected to name one or more entities or objects identified within the question.
    2. _Added to Taxonomy_: Yes
    3. _Rationale_: This type expects an entity as an answer. This is a rather broad definition and we consider merging it in the future but for the initial taxonomy we keep it as is.
24. **Description**: 
    1. _Description:_ Classifies questions that expect a definition or explanation, where the answer provides a clear definition or description of a concept.
    2. _Added to Taxonomy_: Yes
    3. _Rationale_: Relevant as asking for descriptions is a common task in SWA literature research.
25. **Properties**:
    1. _Description:_ Focuses on answers that provide descriptions, classifications, or overviews of processes or phenomena.
    2. _Added to Taxonomy_: Yes
    3. _Rationale_: That a question is asking for properties is a common task in SWA literature research.
26. **Human/Person**:
    1. _Description:_ Classifies questions that expect a human or person as an answer.
    2. _Added to Taxonomy_: Yes
    3. _Rationale_: Relevant for those questions that ask for a specific person.
27. **Location**:
    1. _Description:_ Classifies questions that expect a location as an answer.
    2. _Added to Taxonomy_: Yes
    3. _Rationale_: A rather broad term but still relevant for questions that ask for a specific location.
28. **Quantitative**:
    1. _Description:_ Groups answers that provide numerical information, counts, or quantitative measures.
    2. _Added to Taxonomy_: Yes
    3. _Rationale_: A broad term, relevant for those questions that ask for a specific quantity.
29. **Tool/Notation**:
    1. _Description:_ The question asks for a specific software tool that embodies a technique or a formal language that should have a calculus, semantics, or other basis for computing or inference.
    2. _Added to Taxonomy_: Yes
    3. _Rationale_: We see this relevant in the SWA context.
30. **Solution**:
    1. _Description:_ The question asks for a specific solution, design prototype, or evaluative judgment based on established software engineering principles.
    2. _Added to Taxonomy_: Yes
    3. _Rationale_: Relevant for those questions that ask for a specific solution.
31. **Report**:
    1. _Description:_ Interesting insights or general guidelines are provided in a report. The authors mention that the answer is not general or systematic enough to be categorized as a descriptive model.
    2. _Added to Taxonomy_: No
    3. _Rationale_: This type seeks to classify those answers that are structured like a report. We consider this type to be too general for our taxonomy.
32. **Theoretical Framework**:
    1. _Description:_ Questions that ask for theoretical statements that support a research approach. This is categorized into no specific theory, kernel theory, formal theory, or testable theory.
    2. _Added to Taxonomy_: Yes
    3. _Rationale_: Relevant for those questions that ask for a theoretical framework.

```
Parent-Cluster ID: 1003
Parent-Cluster Name: Question Type | Information Retrieval Actions | Question Processing Steps | Retrieval Operations
Parent-Cluster Info: Includes those clusters that indicate what activities are required from the retriever to be performed to arrive at the answer. As such it specifies the retrieval operations and query-processing steps required to derive an answer
Amount of subclusters: 18
Amount of types in all subclusters: 53
Amount of different domains: 7
Amount of different years: 11
Amount of different categories: 4
Amount of different sources: 17
Names of Subclusters: ['Question Type', 'Activity', 'Negation', 'Clause', 'Contingencies', 'Double Negation', 'Counts', 'Request', 'Superlatives', 'Comparison', 'Listing', 'String Operation', 'Multiple Intents', 'Disambiguation', 'Ranking', 'Temporal', 'Aggregation', 'Relationship']
Domains: 8x General, 2x Software Engineering, 3x Scholarly, 1x Vietnamese Language, 1x Healthcare, 1x Covid, 1x Requirements Engineering
Years: 1x 2016, 1x 2007, 4x 2023, 1x 2017, 1x 2008, 1x 1984, 2x 2019, 2x 2020, 1x 2021, 2x 2022, 1x 2015
Categories: 3x Question Classifier, 1x Other, 9x Knowledge Graph Question Answering Dataset, 4x Research Questions
Sources:  ['The question answering systems: A survey', 'The Future of Empirical Methods in Software Engineering Research', 'DBLP-QuAD: A Question Answering Dataset over the DBLP Scholarly Knowledge Graph', 'The SciQA Scientific Question Answering Benchmark for Scholarly Knowledge', 'Ripple Down Rules for question answering', 'Selecting Empirical Methods for Software Engineering Research', 'The Classification of Research Questions', 'Formulation of Research Question - Stepwise Approach', 'Types of research questions: descriptive, predictive, or causal', 'LC-QuAD 2.0: A Large Dataset for Complex Question Answering over Wikidata and Dbpedia', '10th Question Answering over Linked Data (QALD) Challenge', 'What is in the KGQA Benchmark Datasets? Survey on Challenges in Datasets for Question Answering on Knowledge Graphs', 'A Comparative Study of Question Answering over Knowledge Bases', 'A Non-Factoid Question-Answering Taxonomy', 'Question Answering on Scholarly Knowledge Graphs', 'Large-scale Simple Question Answering with Memory Networks', 'Divide and Conquer the EmpiRE: A Community-Maintainable Knowledge Graph of Empirical Research in Requirements Engineering']
```

**Cluster Description**: This cluster classifies the actions that need to be performed by the retriever to answer the question. It indicates the retrieval operations and query-processing steps required to derive an answer. It is the second most common cluster in our reviewed literature with a total of 17 different sources. It often originates from KGQA dataset literature but is also covered by question classifier studies and research question focus literature. The cluster is predominantly considered in the open and scholarly domain but also has contribution from the software engineering domain. It has also been considered by specialized domains such as requirements engineering or covid studies. The temporal span of the cluster is mixed. It contains older contributions dating back to 1984 but has a concentration of sources emerging in recent years.

**Added to Taxonomy**: Yes

**Rationale**: This category focuses on logical and computational operations implied by a question, such as comparison, counting, or negation. These operations are central in retrieval and directly relate to the functional requirements of a retriever. The cluster generalizes across domains and offers valuable guidance for the inclusion of questions targeting different retrieval abilities.

**Type Relevance**:
1. **Question Type**:
    1. _Description_: Generically defines the type of information request
    2. _Added to Taxonomy_: Yes
    3. _Rationale_: This type generally defines the type of information request. As such it supports the naming of the cluster as `Question Type`. However, it is too general to be used as a type inside of dimension.
2. **Activity**:
    1. _Description_: Generically defines the acitivies that are required to be performed to arrive at the answer.
    2. _Added to Taxonomy_: No
    3. _Rationale_: We consider this type to be too general for our taxonomy.
3. **Negation**:
    1. _Description_: Expects the retrieval process to include negations.
    2. _Added to Taxonomy_: Yes
    3. _Rationale_: Relevant for those questions that include negations.
4. **Clause**:
    1. _Description_: Defines questions where one part depends on another.
    2. _Added to Taxonomy_: Yes
    3. _Rationale_: Relevant for those questions that include clauses.
5. **Contingencies**:
    1. _Description_: Groups questions that involve cause effect relationships. It involves correlations but also causal relationships.
    2. _Added to Taxonomy_: Yes
    3. _Rationale_: Relevant for those questions that involve cause effect relationships.
6. **Double Negation**:
    1. _Description_: Requires to dobule negate parts of the query to arrive at the answer.
    2. _Added to Taxonomy_: No
    3. _Rationale_: We are of the opinion that a single negation type is sufficient for our taxonomy.
7. **Counts**:
    1. _Description_: Classifies questions that require to count information.
    2. _Added to Taxonomy_: Yes
    3. _Rationale_: Relevant for those questions that require to count information.
8. **Request**:
    1. _Description_: Generically classifies questions by the task that is specifically states like 'give' or 'list'
    2. _Added to Taxonomy_: No
    3. _Rationale_: We consider this type to be too general for our taxonomy.
9. **Superlatives**:
    1. _Description_: Groups questions asking for extreme values (maximum/minimum).
    2. _Added to Taxonomy_: Yes
    3. _Rationale_: Relevant for those questions that involve superlatives.
10. **Comparison**:
    1. _Description_:Classifies questions that compare two or more entities by extracting their similarities or differences
    2. _Added to Taxonomy_: Yes
    3. _Rationale_: Relevant for those questions that require to compare information.
11. **Listing**:
    1. _Description_: Classifies questions that require to retrieve multiple properties about a phenomenon and list them.
    2. _Added to Taxonomy_: Yes
    3. _Rationale_: Relevant for those questions that require to list information.
12. **String Operation**:
    1. _Description_: Specifies questions that demand operations on string data
    2. _Added to Taxonomy_: No
    3. _Rationale_: We decided to not include this type in the catalog. In natural language understanding it is required to understand words and their meanings, which revolves about finding strings in sentences or texts. This is a basic requirement for any NLP system. Therefore adding this type to the catalog would classify almost all questions as this type.
13. **Multiple Intents**:
    1. _Description_: Classifies questions that have multiple intentions which could be broken up into multiple different questions
    2. _Added to Taxonomy_: Yes
    3. _Rationale_: Relevant for those questions that have multiple intentions.
14. **Disambiguation**:
    1. _Description_: Focuses on the activity of resolving ambiguity in questions by identifying the correct subject or term among several possibilities before retrieval.
    2. _Added to Taxonomy_: No
    3. _Rationale_: We decided to not include this type as it is also a basic requirement for any NLP system.
15. **Ranking**:
    1. _Description_: Classifies questions that not only count or list facts but also rank or order them
    2. _Added to Taxonomy_: Yes
    3. _Rationale_: Relevant for those questions that require to rank information.
16. **Temporal**:
    1. _Description_: Classifies questions that that involve time series or trends over time, incorporating temporal aggregators and aspects into the retrieval process.
    2. _Added to Taxonomy_: Yes
    3. _Rationale_: Relevant for those questions that involve time series or trends over time.
17. **Aggregation**:
    1. _Description_: Focuses on queries that require aggregation functions (such as average, sum, or grouping) to synthesize data into a meaningful answer.
    2. _Added to Taxonomy_: Yes
    3. _Rationale_: Relevant for those questions that require to quantitatively aggregate information.
18. **Relationship**:
    1. _Description_: Targets questions that ask for the relationship between entities by identifying the connecting predicates or relationships in the knowledge graph.
    2. _Added to Taxonomy_: Yes
    3. _Rationale_: Relevant for those questions that require to identify relationships between entities.

```
Parent-Cluster ID: 1004
Parent-Cluster Name: WH-Patterns
Parent-Cluster Info: Groups questions by their interrogative form (Wh-questions), distinguishing between different wh-word patterns.
Amount of subclusters: 9
Amount of types in all subclusters: 28
Amount of different domains: 4
Amount of different years: 4
Amount of different categories: 2
Amount of different sources: 5
Names of Subclusters: ['What', 'Where', 'Which', 'When', 'Who', 'Why', 'How', 'Whose', 'Whom']
Domains: 1x Scholarly, 2x General, 1x Covid, 1x Vietnamese Language
Years: 1x 2023, 2x 2000, 1x 2022, 1x 2017
Categories: 2x Knowledge Graph Question Answering Dataset, 3x Question Classifier
Sources:  ['The SciQA Scientific Question Answering Benchmark for Scholarly Knowledge', 'A Rule-based Question Answering System for Reading Comprehension Tests', 'A Comparative Study of Question Answering over Knowledge Bases', 'Ripple Down Rules for question answering', 'The Structure and Performance of an Open-Domain Question Answering System']
```

**Cluster Description**: This cluster groups questions by their interrogative form (Wh-questions), distinguishing between different wh-word patterns. It is grounded in early question classification research and continues to play a role in modern QA systems. The cluster is sourced by five different sources, with the most recent contribution dating back to 2023. The cluster includes both scholarly and general domains, with some specialized contributions in the Covid research and the Vietnamese language. 

**Added to Taxonomy**: No

**Rationale**: Although WH-patterns are commonly used in question classification, they primarily reflect linguistic surface forms rather than the underlying retrieval characteristics. Furthermore, the classes are not general enough to provide meaningful characteristics in the scholarly literature search setting. Its utility for guiding the construction of a diverse dataset is also limited. While it may promote diversity with regard to WH-forms, it is not helpful to capture the meaning of questions or reliably correspond to distinct retrieval challenges.

```
Parent-Cluster ID: 1005
Parent-Cluster Name: Specialized Knowledge Base Types
Parent-Cluster Info: Contains classifications specific to querying specialized knowledge bases. Therefore to apply these classifications, it is required to run the question on that specific knowledge base.
Amount of subclusters: 2
Amount of types in all subclusters: 5
Amount of different domains: 2
Amount of different years: 3
Amount of different categories: 1
Amount of different sources: 3
Names of Subclusters: ['ORKG and SPARQL Queries', 'Wikidata Qualifiers']
Domains: 1x Scholarly, 2x General
Years: 1x 2023, 1x 2021, 1x 2019
Categories: 3x Knowledge Graph Question Answering Dataset
Sources:  ['The SciQA Scientific Question Answering Benchmark for Scholarly Knowledge', 'What is in the KGQA Benchmark Datasets? Survey on Challenges in Datasets for Question Answering on Knowledge Graphs', 'LC-QuAD 2.0: A Large Dataset for Complex Question Answering over Wikidata and Dbpedia']
```

**Cluster Description**: This cluster contains classifications specific to querying specialized knowledge bases. Therefore to apply these classifications, it is required to run the question on that specific knowledge base. The cluster is sourced by three different sources from recent years. The cluster includes both scholarly and general domains and is only covered by KGQA dataset literature.

**Added to Taxonomy**: No

**Rationale**: This category encompasses classes that are specific to certain knowledge bases, such as Wikidata qualifiers or ORKG-specific constructs. Although these distinctions are meaningful within tightly scoped contexts, they do not generalize across KGQA systems or domains. Consequently, the cluster offers limited value for constructing broadly applicable evaluation datasets and is more reflective of system-specific constraints than question-inherent properties.

```
Parent-Cluster ID: 1006
Parent-Cluster Name: Research-Orientation | Research-Focus | Scholarly Question Type
Parent-Cluster Info: Focuses on scholarly and research-oriented questions
Amount of subclusters: 8
Amount of types in all subclusters: 8
Amount of different domains: 2
Amount of different years: 2
Amount of different categories: 2
Amount of different sources: 2
Names of Subclusters: ['Research Output', 'Development Methods', 'Analytical Methods', 'Instance-Specific', 'Generalization ', 'Qualitative Modeling', 'Empirical Modeling', 'Analytic Modeling']
Domains: 1x Scholarly, 1x Software Engineering
Years: 1x 2024, 1x 2003
Categories: 1x Knowledge Graph Question Answering Dataset, 1x Research Questions
Sources:  ['Hybrid-SQuAD: Hybrid Scholarly Question Answering Dataset', 'Writing good software engineering research papers']
```

**Cluster Description**: This cluster focuses on classifying questions related to their research focus. This cluster is only sourced by two sources: one is a 2024 KGQA dataset focusing on scholarly content, and the other is a 2003 research-question-centric work in software engineering.

**Added to Taxonomy**: Yes

**Rationale**: This category categorizes questions based on their orientation towards research-related topics such as research methods or modeling approaches. Although the classes may be domain-specific, they align closely with our focus on scholarly literature search. In addition, the cluster is relevant for designing evaluation datasets that capture the diversity of scholarly questions.

**Type Relevance**:
1. **Research Output**:
    1. _Description_: Classifies questions focused on research outputs from publications
    2. _Added to Taxonomy_: No
    3. _Rationale_: We consider this type to be too general for our taxonomy.
2. **Development Methods**:
    1. _Description_: Covers questions that ask about the methods or means by which a software problem is solved.
    2. _Added to Taxonomy_: Yes
    3. _Rationale_: This is relevant as it can classify those question that ask for methods for development in SWA.
3. **Analytical Methods**:
    1. _Description_: Focuses on questions asking for methods used in the analysis or evaluation of software tools, technologies, or approaches.
    2. _Added to Taxonomy_: Yes
    3. _Rationale_: Relevant for those questions in SWA that seek for approaches related to analysis or evaluation
4. **Instance-Specific**:
    1. _Description_: The purpose of this is to classify questions that ask about the evaluation of a specific software instance
    2. _Added to Taxonomy_: No
    3. _Rationale_: We consider this type to be too specific for our taxonomy.
5. **Generalization**:
    1. _Description_: Classifies questions that seek broad concepts, taxonomies, or general principles in software engineering.
    2. _Added to Taxonomy_: Yes
    3. _Rationale_: We consider this to be relevant as it is common in SWA to seek for general principles.
6. **Qualitative Modeling**:
    1. _Description_: Groups questions expecting qualitative models, frameworks, or taxonomies that describe or structure a problem area.
    2. _Added to Taxonomy_: Yes
    3. _Rationale_: Relevant for those questions that ask for qualitative models.
7. **Empirical Modeling**:
    1. _Description_: Classifies questions that expect models grounded in observed data, employing statistical or empirical analysis.
    2. _Added to Taxonomy_: Yes
    3. _Rationale_: Relevant for those questions that ask for empirical models.
8. **Analytic Modeling**:
    1. _Description_: Focuses on questions expecting formal, mathematically defined models to facilitate analysis or computation.
    2. _Added to Taxonomy_: Yes
    3. _Rationale_: Relevant for those questions that ask for analytical models.

```
Parent-Cluster ID: 1007
Parent-Cluster Name: Answer Credibility | Answer Reliability
Parent-Cluster Info: Classifies questions by the credibility of the answer with regard to the truthfulness.
Amount of subclusters: 5
Amount of types in all subclusters: 9
Amount of different domains: 4
Amount of different years: 5
Amount of different categories: 3
Amount of different sources: 5
Names of Subclusters: ['Factual', 'Opinion', 'Debate', 'Conversational', 'Predictive']
Domains: 1x Requirements Engineering, 2x General, 1x Social Science, 1x Healthcare
Years: 1x 2023, 1x 2022, 1x 2015, 1x 1984, 1x 2020
Categories: 1x Knowledge Graph Question Answering Dataset, 2x Question Classifier, 2x Research Questions
Sources:  ['Divide and Conquer the EmpiRE: A Community-Maintainable Knowledge Graph of Empirical Research in Requirements Engineering', 'A Non-Factoid Question-Answering Taxonomy', 'A Taxonomy for Classifying Questions Asked in Social Question and Answering', 'The Classification of Research Questions', 'Types of research questions: descriptive, predictive, or causal']
```

**Cluster Description**: This cluster classifies questions by the credibility of the answer with regard to the truthfulness. It is covered by 5 sources from the requirements engineering, social science, healthcare, and open domain. Most sources have been from recent years with the oldest source dating back to 1984. The cluster is considered by KGQA dataset literature, question classifier studies, and research question focus literature.

**Added to Taxonomy**: Yes

**Rationale**: This category considers the credibility of expected answers, such as factual, predictive, or opinion-based. Because the literature search can include various credibility types, this classification is relevant to the scholarly literature search and generalizable across different domains. Moreover, considering this classification promotes diversity in the creation of question datasets.

**Type Relevance**:
1. **Factual**:
    1. _Description_: Groups questions that expect factual, evidence-based answers.
    2. _Added to Taxonomy_: Yes
    3. _Rationale_: Relevant for those questions that ask for information that is evidence-based.
2. **Opinion**:
    1. _Description_: Covers questions where the answer is expected to reflect personal experiences, opinions, or recommendations rather than strict facts.
    2. _Added to Taxonomy_: Yes
    3. _Rationale_: Relevant for those questions that ask for opinions or experiences of other researchers.
3. **Debate**:
    1. _Description_: Classifies questions that intend to provoke discussion or argumentation rather than factual answers.
    2. _Added to Taxonomy_: Yes
    3. _Rationale_: Relevant for those questions that ask for discussion or argumentation.
4. **Conversational**:
    1. _Description_: Identifies questions that are rhetorical or conversational in nature, where a definitive factual answer is not required.
    2. _Added to Taxonomy_: No
    3. _Rationale_: For a KGQA retrieval system, the goal is to provide factual answers and not engage in conversation. Therefore, we decided against this type.
5. **Predictive**:
    1. _Description_: Focuses on questions that ask for predictions or forecasts.
    2. _Added to Taxonomy_: Yes
    3. _Rationale_: Relevant for those questions that ask for predictions.

```
Parent-Cluster ID: 1008
Parent-Cluster Name: Question Goal | Question Intent | Question Objective
Parent-Cluster Info: Defines the overall goal or intent behind a question.
Amount of subclusters: 7
Amount of types in all subclusters: 14
Amount of different domains: 3
Amount of different years: 6
Amount of different categories: 2
Amount of different sources: 7
Names of Subclusters: ['Exploratory', 'Reasoning', 'Problem Solving', 'Gap Spotting', 'Problematization', 'Focus', 'Design']
Domains: 2x Software Engineering, 1x Design Science, 4x General
Years: 1x 2008, 2x 2019, 1x 2003, 1x 2022, 1x 2000, 1x 2016
Categories: 4x Research Questions, 3x Question Classifier
Sources:  ['Selecting Empirical Methods for Software Engineering Research', 'Construction of Design Science Research Questions', 'Formulation of Research Question - Stepwise Approach', 'Writing good software engineering research papers', 'A Non-Factoid Question-Answering Taxonomy', 'The Structure and Performance of an Open-Domain Question Answering System', 'The question answering systems: A survey']
```

**Cluster Description**: This cluster defines the overall goal or intent behind a question. It is covered by 7 sources from the software engineering, design science, and open domain. The publication years of the sources are mixed in between the years 2000 and 2022. The cluster is considered by question classifier studies and research question focus literature. 

**Added to Taxonomy**: Yes

**Rationale**: This category addresses the underlying intent behind a question, such as problem solving, reasoning, or exploration. These type of goals are relevant for scholarly literature search to capture the intent behind the question. As such, it promotes diversity in question dataset creation.

**Type Relevance**:

1. **Exploratory**:
    1. _Description_: Classifies questions aimed at exploring the existence, description, or comparative features of a phenomenon.
    2. _Added to Taxonomy_: Yes
    3. _Rationale_: Relevant for those questions that aim to explore a phenomenon.
2. **Reasoning**:
    1. _Description_: Classifies questions that seek reasons or causes behind events or phenomena.
    2. _Added to Taxonomy_: Yes
    3. _Rationale_: Relevant for those questions that ask for reasons or causes.
3. **Problem Solving**:
    1. _Description_: Focuses on questions that seek to find solutions or resolve issues.
    2. _Added to Taxonomy_: Yes
    3. _Rationale_: Relevant for those questions that ask for solutions.
4. **Gap Spotting**:
    1. _Description_: Classifies questions that aim to spot gaps in existing literature or research, thereby highlighting areas needing further investigation.
    2. _Added to Taxonomy_: Yes
    3. _Rationale_: Relevant for those questions that aim to spot gaps in existing literature.
5. **Problematization**:
    1. _Description_: Classifies questions that aim to articulate deficiencies or problems in current theories and practices suggesting a need for additional research.
    2. _Added to Taxonomy_: Yes
    3. _Rationale_: Relevant for those questions that aim to articulate deficiencies or problems in current theories.
6. **Focus**:
    1. _Description_: Generally defines the aim of a question as the focus, which should guide the extraction of most relevant information from the Knowledge Graph
    2. _Added to Taxonomy_: No
    3. _Rationale_: We consider this type to be too general for our taxonomy. 
7. **Design**:
    1. _Description:_ The question wants to discover improved methods for software design.
    2. _Added to Taxonomy_: Yes
    3. _Rationale_: It is relevant for the SWA task to identify improved methods in software design.

```
Parent-Cluster ID: 1009
Parent-Cluster Name: Application-Specific
Parent-Cluster Info: Groups those clusters where the type is specific to an application or use case and not useful for our task.
Amount of subclusters: 12
Amount of types in all subclusters: 12
Amount of different domains: 4
Amount of different years: 5
Amount of different categories: 3
Amount of different sources: 5
Names of Subclusters: ['Problem Statement Identification', 'Research Question Utilization', 'Research Question Typology', 'Inquiry Mode Classification', 'Outcome Artifact Classification', 'Missing Term Resolution', 'Three Term Resolution', 'Human Relationship Inquiry', 'Multi Category Classification', 'NNP Answer Type', 'Resource Typed', 'Resource Untyped']
Domains: 1x Design Science, 1x Vietnamese Language, 1x Spoken Natural Language Processing, 2x General
Years: 1x 2019, 1x 2017, 1x 2015, 1x 2000, 1x 2021
Categories: 1x Research Questions, 3x Question Classifier, 1x Knowledge Graph Question Answering Dataset
Sources:  ['Construction of Design Science Research Questions', 'Ripple Down Rules for question answering', 'Linguistically Motivated Question Classification', 'The Structure and Performance of an Open-Domain Question Answering System', 'What is in the KGQA Benchmark Datasets? Survey on Challenges in Datasets for Question Answering on Knowledge Graphs']
```

**Cluster Description**: This cluster groups those clusters where the type is specific to an application or use case and not useful for our task. It is covered by 5 sources from the design science, Vietnamese language, spoken natural language processing, and open domain. The publication years of the sources are mixed in between the years 2000 and 2021. The cluster is considered by question classifier studies, research question focused and KGQA dataset literature.

**Added to Taxonomy**: No

**Rationale**: This category includes classes that are defined in relation to specific applications or use contexts, such as outcome artifact classification or research question typologies. Although these categories may be useful within individual domains, they do not reflect generalizable properties of the questions themselves. As such, they provide limited value for constructing broadly applicable evaluation datasets.

