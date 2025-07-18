from typing import List, Optional, Dict
from threading import RLock
import chromadb
from chromadb import QueryResult

from sqa_system.core.data.file_path_manager import FilePathManager
from sqa_system.core.logging.logging import get_logger
from sqa_system.core.data.models.triple import Triple

logger = get_logger(__name__)


class ChromaVectorStore:

    def __init__(self, store_name: str, collection_name: str = "mindmap_retriever"):
        """
        Initializes the ChromaVectorStore by setting up the client and collection.

        Args:
            collection_name (str): Name of the Chroma collection.
        """
        self.store_name = store_name
        self.file_path_manager = FilePathManager()
        self.collection_name = collection_name
        self._lock = RLock()
        self.client = None
        self.collection = None
        self._initialize()

    def _initialize(self):
        with self._lock:
            if self.client is None:
                self.client = self._initialize_client()
            if self.collection is None:
                self.collection = self._initialize_collection()

    def _initialize_client(self):
        """
        Initializes the Chroma client with the specified persist directory.

        Returns:
            chromadb.Client: Configured Chroma client instance.
        """
        novel_retriever_cache_dir = self.file_path_manager.combine_paths(
            self.file_path_manager.CACHE_DIR, "mindmap_retriever", self.store_name
        )

        # Ensure the persist directory exists
        self.file_path_manager.ensure_dir_exists(novel_retriever_cache_dir)
        logger.info(
            f"Loading chroma vector store from: {novel_retriever_cache_dir}")

        client = chromadb.PersistentClient(path=novel_retriever_cache_dir)
        return client

    def _initialize_collection(self) -> chromadb.Collection:
        """
        Retrieves or creates the specified Chroma collection.

        Returns:
            chromadb.Collection: The Chroma collection instance.
        """
        if self.client is None:
            raise ValueError("Chroma client is not initialized.")
        collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"}
        )
        logger.info(f"Initialized collection: {self.collection_name}")
        logger.info(
            f"Amount of embeddings in collection: {collection.count()}")

        return collection

    def store_data(self,
                   key: str,
                   embedding: List[float],
                   metadata: Optional[Dict] = None):
        """
        Stores an embedding in the Chroma collection with the given key as its ID.

        Args:
            key (str): Unique identifier for the embedding.
            embedding (List[float]): The embedding vector.
            metadata (dict, optional): Additional metadata associated with the embedding.
        """
        if metadata is None:
            metadata = {}
        if self.collection is None:
            raise ValueError("Chroma collection is not initialized.")

        with self._lock:
            try:
                self.collection.upsert(
                    embeddings=[embedding],
                    metadatas=[metadata],
                    ids=[key]
                )
                logger.debug(f"Upserted embedding with ID: {key}")
            except Exception as e:
                logger.error(
                    f"Failed to upsert embedding with ID {key}: {e}")
                raise

    def similarity_search(self,
                          query_embedding: List[float],
                          n_results: int = 10,
                          exclude: List[str] = None) -> QueryResult:
        """
        Searches the Chroma collection for embeddings similar to the query embedding.

        Args:
            query_embedding (List[float]): The query embedding vector.
        """
        if self.collection is None:
            raise ValueError("Chroma collection is not initialized.")

        with self._lock:
            try:
                query_result = self.collection.query(
                    query_embeddings=[query_embedding],
                    n_results=n_results,
                    include=["embeddings", "metadatas", "distances"],
                    where={"entity": {"$nin": exclude}} if exclude else None
                )
                return query_result
            except Exception as e:
                logger.debug(f"Failed to query embeddings: {e}")
                raise 

    def get_by_ids(self,
                   keys: List[str]):
        """
        Retrieves an embedding and its metadata from the Chroma collection by its ID.

        Args:
            keys (List[str]): The IDs of the embeddings to retrieve.
        """
        if self.collection is None:
            raise ValueError("Chroma collection is not initialized.")

        with self._lock:
            try:
                results = self.collection.get(
                    ids=keys,
                    include=["embeddings", "metadatas"]
                )

                if not results['ids']:
                    return None

                return {
                    "id": results['ids'],
                    "embedding": results['embeddings'],
                    "metadata": results['metadatas']
                }
            except Exception as e:
                logger.error(
                    f"Failed to retrieve embeddings {e}")
                raise
