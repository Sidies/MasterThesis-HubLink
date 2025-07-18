import hashlib
from typing import List

from sqa_system.core.data.models.triple import Knowledge
from sqa_system.knowledge_base.knowledge_graph.storage.base.knowledge_graph import KnowledgeGraph
from sqa_system.knowledge_base.knowledge_graph.storage.utils.subgraph_builder import SubgraphBuilder, SubgraphOptions
from sqa_system.knowledge_base.knowledge_graph.storage.utils.graph_converter import GraphConverter
from sqa_system.core.language_model.base.embedding_adapter import EmbeddingAdapter
from sqa_system.app.cli.cli_progress_handler import ProgressHandler
from sqa_system.core.language_model.base.llm_adapter import LLMAdapter
from sqa_system.core.logging.logging import get_logger

from .vector_store import ChromaVectorStore

logger = get_logger(__name__)


class MindMapIndexer:
    """
    Indexer class for the MindMap retriever that indexes the entities
    in the knowledge graph. 
    
    The original code did not store their embeddings in a graph but rather
    in text files. To make the retriever more efficient, we store the embeddings
    in a graph database.
    """

    def __init__(self,
                 graph: KnowledgeGraph,
                 llm_adapter: LLMAdapter,
                 embedding_adapter: EmbeddingAdapter,
                 vector_store: ChromaVectorStore):
        self.graph = graph
        self.subgraph_builder = SubgraphBuilder(graph)
        self.graph_converter = GraphConverter(
            graph=graph,
            llm_adapter=llm_adapter
        )
        self.embedding_adapter = embedding_adapter
        self.vector_store = vector_store

    def run_indexing(self,
                     batch_size: int = 100):
        """
        Runs the indexing by retrieving the entities from the knowledge graph
        and indexing them in the vector store.
        
        Also checks whether the entities are already indexed in the vector store
        and skips those.
        
        Args:
            batch_size (int): The amount of entities to index in one batch.
        """

        publication_type = self.graph.paper_type
        root_entities = self.graph.get_entities_by_types(
            types=[publication_type]
        )

        if not root_entities:
            logger.warning("No entities found for indexing")
            return

        progress_handler = ProgressHandler()
        task_id = progress_handler.add_task(
            string_id="building_index",
            description="Building index...",
            total=len(root_entities)
        )

        batch_entities = []
        batch_metadatas = []
        unique_entities_with_doi = set()
        for entity in root_entities:

            _, subgraph = self.subgraph_builder.get_subgraph(
                root_entity=entity,
                options=SubgraphOptions(
                    go_against_direction=True,
                ))

            metadata = {
                "title": entity.text,
            }

            # Collect additional metadata from subgraph
            for triple in subgraph:
                if "doi" in triple.predicate:
                    metadata["doi"] = triple.entity_object.text
                elif "publisher" in triple.predicate:
                    metadata["publisher"] = triple.entity_object.text
                elif "datePublished" in triple.predicate:
                    metadata["datePublished"] = triple.entity_object.text
                if len(metadata) == 4:
                    break

            task2_id = progress_handler.add_task(
                string_id="adding_entities",
                description="Indexing entities...",
                total=len(subgraph)
            )

            for triple in subgraph:
                doi = metadata.get("doi", "")
                for i in range(2):
                    if i == 0:
                        text = triple.entity_object.text
                    else:
                        text = triple.entity_subject.text
                    entity_key = (text, doi)
                    if entity_key in unique_entities_with_doi:
                        progress_handler.update_task_by_string_id(task2_id, advance=1)
                        continue

                    unique_entities_with_doi.add(entity_key)

                    updated_metadata = metadata.copy()
                    updated_metadata["triple"] = triple.model_dump_json()
                    if i == 0:
                        updated_metadata["entity"] = triple.entity_object.model_dump_json()
                        batch_entities.append(triple.entity_object)
                    else:
                        updated_metadata["entity"] = triple.entity_subject.model_dump_json()
                        batch_entities.append(triple.entity_subject)
                    batch_metadatas.append(updated_metadata)

                if len(batch_entities) >= batch_size:
                    self._index_entities(batch_entities, batch_metadatas)
                    batch_entities.clear()
                    batch_metadatas.clear()
                progress_handler.update_task_by_string_id(task2_id, advance=1)

            progress_handler.finish_by_string_id(task2_id)
            progress_handler.update_task_by_string_id(task_id, advance=1)

        if batch_entities:
            self._index_entities(batch_entities, batch_metadatas)

        progress_handler.finish_by_string_id(task_id)

    def _index_entities(self,
                        entities: List[Knowledge],
                        metadatas: List[dict]):
        """
        Indexes a batch of triples.

        Args:
            triples (List[Triple]): The triples to index.
            metadatas (List[dict]): The metadatas corresponding to the triples.
        """
        entity_texts = [entity.text for entity in entities]
        dois = [metadata.get("doi", "") for metadata in metadatas]
        combined_texts = [f"{text} | DOI: {doi}" for text,
                          doi in zip(entity_texts, dois)]
        hash_ids = [hashlib.md5(combined_text.encode()).hexdigest()
                    for combined_text in combined_texts]

        # Check which ids already exist in the collection
        existing = self.vector_store.get_by_ids(hash_ids)
        if existing:
            existing_ids = set(existing["id"])
        else:
            existing_ids = set()

        # Filter out the entities that already exist
        new_entities = []
        new_metadatas = []
        new_hash_ids = []
        for i, hash_id in enumerate(hash_ids):
            if hash_id not in existing_ids:
                new_entities.append(entities[i])
                new_metadatas.append(metadatas[i])
                new_hash_ids.append(hash_id)

        if not new_entities:
            logger.debug("No new entities to index in this batch.")
            return

        new_entities_texts = [entity.text for entity in new_entities]

        embeddings = self.embedding_adapter.embed_batch(new_entities_texts)

        for i, _ in enumerate(new_entities):
            self.vector_store.store_data(
                key=new_hash_ids[i],
                embedding=embeddings[i],
                metadata=new_metadatas[i]
            )
        logger.info(f"Indexed {len(new_entities)} new entities.")
