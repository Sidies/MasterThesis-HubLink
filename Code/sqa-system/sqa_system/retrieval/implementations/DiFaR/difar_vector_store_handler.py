import hashlib
from typing import List, Optional, Tuple
from dataclasses import dataclass
import json
import chromadb


from sqa_system.core.data.models.triple import Triple
from sqa_system.knowledge_base.knowledge_graph.storage.base.knowledge_graph import KnowledgeGraph
from sqa_system.knowledge_base.knowledge_graph.storage.utils.subgraph_builder import SubgraphBuilder, SubgraphOptions
from sqa_system.knowledge_base.knowledge_graph.storage.utils.graph_converter import GraphConverter
from sqa_system.core.language_model.base.embedding_adapter import EmbeddingAdapter
from sqa_system.app.cli.cli_progress_handler import ProgressHandler
from sqa_system.core.data.file_path_manager import FilePathManager
from sqa_system.core.language_model.base.llm_adapter import LLMAdapter
from sqa_system.core.logging.logging import get_logger

logger = get_logger(__name__)


@dataclass
class VectorStoreHandlerOptions:
    """
    Options for the vector store handler.
    """
    distance_metric: str
    vector_store_name: str
    convert_to_text: bool = False


class DifarVectorStoreHandler:
    """
    This handler is responsible for managing the vector store for the DiFaR retriever.
    """

    def __init__(self,
                 graph: KnowledgeGraph,
                 embedding_adapter: EmbeddingAdapter,
                 llm_adapter: LLMAdapter,
                 options: VectorStoreHandlerOptions):
        self.distance_metric = options.distance_metric
        self.vector_store_name = options.vector_store_name
        self.progress_handler = ProgressHandler()
        self.convert_to_text = options.convert_to_text
        self.graph = graph
        self.embedding_adapter = embedding_adapter
        self.subgraph_builder = SubgraphBuilder(graph)
        self.graph_converter = GraphConverter(
            graph=graph,
            llm_adapter=llm_adapter
        )
        self.client = None
        self.collection = None
        self._initialize_vector_store(options.vector_store_name,
                                      options.distance_metric)

    def query(self,
              query: str,
              k: int,
              metadata_filter: Optional[dict] = None) -> List[Tuple[Triple, dict]]:
        """
        Queries the vector store for the given query.

        Args:
            query (str): The query to search for.
            k (int): The number of results to return.
        """
        embedding = self.embedding_adapter.embed(query)
        number_of_results = k
        retries = 3
        for attempt in range(retries):
            try:
                if metadata_filter:
                    results = self.collection.query(
                        query_embeddings=[embedding],
                        n_results=number_of_results,
                        include=["embeddings", "metadatas"],
                        where=metadata_filter
                    )
                else:
                    results = self.collection.query(
                        query_embeddings=[embedding],
                        n_results=number_of_results,
                        include=["embeddings", "metadatas"]
                    )
                break
            except Exception as e:
                logger.error(f"Error during query: {e}. Retrying query with less number of results.")
                number_of_results = max(1, number_of_results // 2)
                if attempt < retries - 1:
                    logger.info("Retrying...")
                else:
                    return []

        triples_with_metadata = []
        for metadata in results["metadatas"][0]:
            triple = Triple.model_validate_json(metadata["triple"])
            triples_with_metadata.append((triple, metadata))
        return triples_with_metadata

    def _initialize_vector_store(self,
                                 vector_store_name: str,
                                 distance_metric: str):
        storage_path = self._prepare_storage_path(vector_store_name)
        self._write_index_information(storage_path)
        logger.info(
            "Initializing DiFaR Vector Store with storage path: %s", storage_path)
        self.client = chromadb.PersistentClient(path=storage_path)
        # We encountered an error with chromadb that only happend after we scaled the dataset to its fullest.
        # The error that semingly randomly appeared was 'Cannot return the results in a contigious 2D array.
        # Probably ef or M is too small'. It appeared only at query time, indexing was fine.
        # The error is badly documented but the ressources suggest, that the issue appears, when the query
        # cant return the amount of results that are asked for. Therefore, we increase the parameters
        # here to reduce the chances of it happening. https://docs.trychroma.com/docs/collections/configure
        self.collection = self.client.get_or_create_collection(
            name="difar",
            metadata={
                "hnsw:space": distance_metric,
                "hnsw:search_ef": 200,
                "hnsw:construction_ef": 200,
                "hnsw:M": 32}
        )
        logger.info(
            f"Amount of embeddings in collection: {self.collection.count()}")

    def _prepare_storage_path(self, vector_store_name: str) -> str:
        # prepare the storage path the vector store will be saved to
        file_path_manager = FilePathManager()
        cache_dir = file_path_manager.CACHE_DIR
        storage_path = file_path_manager.combine_paths(
            cache_dir, "difar", vector_store_name)
        file_path_manager.ensure_dir_exists(storage_path)
        return storage_path

    def run_indexing(self,
                     batch_size: int = 100,
                     force_update: bool = False,
                     root_entity_ids: List[str] = None):
        """
        Runs the indexing process.

        Args:
            types (List[str]): The types of the root entities to index.
                Needs to be exactly the type.
            batch_size (int): The number of triples to process in each batch.
        """
        if force_update:
            logger.info(
                "Force update is enabled. Deleting existing collection data.")
            self._delete_collection_data()
            self._initialize_vector_store(
                self.vector_store_name, self.distance_metric
            )

        logger.info("Starting indexing process for DiFaR Retriever")

        root_entities = self._get_root_entities(root_entity_ids)

        root_neighbors = self._get_neighbor_entities(root_entities)

        if not root_neighbors:
            logger.warning("No entities found for indexing")
            return

        self._run_indexing_internal(root_neighbors, batch_size)

    def _delete_collection_data(self):
        self.client.delete_collection("difar")
        logger.info("Deleted collection data")

    def _write_index_information(self, storage_path: str):
        data = {
            "convert_to_text": self.convert_to_text,
            "distance_metric": self.distance_metric
        }
        data.update(self.graph_converter.llm_adapter.llm_config.model_dump())
        data.update(self.embedding_adapter.embedding_config.model_dump())
        file_storage_path = FilePathManager().combine_paths(
            storage_path, "index_info.json")
        with open(file_storage_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4)

    def _run_indexing_internal(self,
                               root_neighbors: List[Triple],
                               batch_size: int = 100):
        self.progress_handler.add_task(
            string_id="adding_subgraphs",
            description="Adding subgraphs to vector store",
            total=len(root_neighbors)
        )
        batch_triples = []
        batch_metadatas = []
        unique_triples_with_doi = set()
        for entity in root_neighbors:
            _, subgraph = self.subgraph_builder.get_subgraph(
                root_entity=entity,
                options=SubgraphOptions(
                    go_against_direction=True,
                ))
            metadata = self._get_metadata(subgraph)

            self.progress_handler.add_task(
                string_id="adding_triples",
                description="Adding triples to vector store",
                total=len(subgraph)
            )
            for triple in subgraph:
                triple_text = self._triple_to_text(triple)
                doi = metadata.get("doi", "")
                triple_key = (triple_text, doi)
                if triple_key in unique_triples_with_doi:
                    logger.debug(
                        f"Duplicate triple detected and skipped: {triple_text} | DOI: {doi}")
                    self.progress_handler.update_task_by_string_id(
                        "adding_triples", advance=1)
                    continue
                unique_triples_with_doi.add(triple_key)

                updated_metadata = metadata.copy()
                updated_metadata["triple"] = triple.model_dump_json()
                batch_triples.append(triple)
                batch_metadatas.append(updated_metadata)

                if len(batch_triples) >= batch_size:
                    self._index_triples(batch_triples, batch_metadatas)
                    batch_triples.clear()
                    batch_metadatas.clear()
                self.progress_handler.update_task_by_string_id(
                    "adding_triples", advance=1)

            self.progress_handler.finish_by_string_id("adding_triples")
            self.progress_handler.update_task_by_string_id(
                "adding_subgraphs", advance=1)

        if batch_triples:
            self._index_triples(batch_triples, batch_metadatas)

        self.progress_handler.finish_by_string_id("adding_subgraphs")

    def _index_triples(self,
                       triples: List[Triple],
                       metadatas: List[dict]):
        """
        Indexes a batch of triples.

        Args:
            triples (List[Triple]): The triples to index.
            metadatas (List[dict]): The metadatas corresponding to the triples.
        """
        triple_texts = [self._triple_to_text(triple) for triple in triples]
        dois = [metadata.get("doi", "") for metadata in metadatas]
        combined_texts = [f"{text} | DOI: {doi}" for text,
                          doi in zip(triple_texts, dois)]
        hash_ids = [hashlib.md5(combined_text.encode()).hexdigest()
                    for combined_text in combined_texts]

        # Check which ids already exist in the collection
        existing_ids = set(self.collection.get(ids=hash_ids)["ids"])

        # Filter out the triples that already exist
        new_triples = []
        new_metadatas = []
        new_hash_ids = []
        for i, hash_id in enumerate(hash_ids):
            if hash_id not in existing_ids:
                new_triples.append(triples[i])
                new_metadatas.append(metadatas[i])
                new_hash_ids.append(hash_id)

        if not new_triples:
            logger.debug("No new triples to index in this batch.")
            return

        new_triple_texts = self._prepare_new_triple_texts(new_triples)

        embeddings = self.embedding_adapter.embed_batch(new_triple_texts)

        self.collection.add(
            ids=new_hash_ids,
            embeddings=embeddings,
            metadatas=new_metadatas
        )
        logger.info(f"Indexed {len(new_triples)} new triples.")

    def _prepare_new_triple_texts(self, new_triples: List[Triple]) -> List[str]:
        new_triple_texts = []
        for triple in new_triples:
            if self.convert_to_text:
                triple_text = self.graph_converter.path_to_text([triple])
            else:
                triple_text = self._triple_to_text(triple)
            new_triple_texts.append(triple_text)
        return new_triple_texts

    def _get_root_entities(self, root_entity_ids: List[str]) -> List[Triple]:
        root_entities = []
        for root_entity_id in root_entity_ids:
            root_entity = self.graph.get_entity_by_id(root_entity_id)
            if root_entity:
                root_entities.append(root_entity)
            else:
                logger.warning(
                    f"Root entity with id {root_entity_id} not found in the graph.")
        return root_entities

    def _get_neighbor_entities(self, root_entities: List[Triple]) -> List[Triple]:
        root_neighbors = []
        for root_entity in root_entities:
            head_relations = self.graph.get_relations_of_head_entity(
                root_entity)
            for relation in head_relations:
                root_neighbors.append(relation.entity_object)
            tail_relations = self.graph.get_relations_of_tail_entity(
                root_entity)
            for relation in tail_relations:
                root_neighbors.append(relation.entity_subject)
        return root_neighbors

    def _get_metadata(self, subgraph: List[Triple]) -> dict:
        metadata = {}
        for triple in subgraph:
            if "doi" in triple.predicate:
                metadata["doi"] = triple.entity_object.text
            elif "publisher" in triple.predicate:
                metadata["publisher"] = triple.entity_object.text
            elif "datePublished" in triple.predicate or "publication year" in triple.predicate:
                metadata["datePublished"] = triple.entity_object.text
            elif "research field" in triple.predicate:
                metadata["research_field"] = triple.entity_object.text
            elif "venue" in triple.predicate:
                metadata["venue"] = triple.entity_object.text

            # Get the title
            head_types = self.graph.get_types_of_entity(triple.entity_object)
            if self.graph.paper_type in head_types:
                metadata["title"] = triple.entity_object.text
            tail_types = self.graph.get_types_of_entity(triple.entity_subject)
            if self.graph.paper_type in tail_types:
                metadata["title"] = triple.entity_subject.text
        return metadata

    def _triple_to_text(self, triple: Triple) -> str:
        """
        Converts a triple to a text representation.
        """
        return f"({triple.entity_subject.text}, {triple.predicate}, {triple.entity_object.text})"
