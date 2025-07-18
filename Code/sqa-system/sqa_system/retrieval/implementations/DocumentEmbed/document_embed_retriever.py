from typing import ClassVar, List
from typing_extensions import override

from sqa_system.core.data.models import RetrievalAnswer
from sqa_system.knowledge_base.vector_store.storage.vector_store_manager import VectorStoreManager
from sqa_system.core.logging.logging import get_logger
from sqa_system.core.config.models.additional_config_parameter import (
    AdditionalConfigParameter,
    RestrictionType
)
from sqa_system.core.config.models import (
    VectorStoreConfig,
    DocumentRetrievalConfig,
    ChunkingStrategyConfig,
    EmbeddingConfig,
    DatasetConfig
)

from ...base.document_retriever import DocumentRetriever

logger = get_logger(__name__)

DEFAULT_VECTOR_STORE_CONFIG = VectorStoreConfig(
    additional_params={
        "distance_metric": "cosine",
    },
    vector_store_type="chroma",
    chunking_strategy_config=ChunkingStrategyConfig(
        chunk_size=1000,
        chunk_overlap=100,
        additional_params={},
        chunking_strategy_type="RecursiveCharacterChunkingStrategy"
    ),
    dataset_config=DatasetConfig(
        additional_params={},
        file_name="merged_ecsa_icsa.json",
        loader="JsonPublicationLoader",
        loader_limit=-1
    ),
    embedding_config=EmbeddingConfig(
        additional_params={},
        endpoint="GoogleAI",
        name_model="models/text-embedding-004",
    )
)
DEFAULT_N_RESULTS = 15


class DocumentEmbedRetriever(DocumentRetriever):
    """
    A 'DocumentRetriever' class implementation that is responsible for retrieving document
    based data from a vector store.

    It is intended to be used as a Baseline for the State-of-the-Art embedding based retrieval
    approach.
    """

    ADDITIONAL_CONFIG_PARAMS: ClassVar[List[AdditionalConfigParameter]] = [
        AdditionalConfigParameter(
            name="n_results",
            description="The number of results to return.",
            param_type=int,
            default_value=DEFAULT_N_RESULTS,
            param_restriction=RestrictionType.GREATER_THAN_ZERO
        ),
        AdditionalConfigParameter(
            name="vector_store_config",
            description="The configuration of the vector store to use.",
            param_type=VectorStoreConfig,
            default_value=DEFAULT_VECTOR_STORE_CONFIG
        )
    ]

    def __init__(self, config: DocumentRetrievalConfig) -> None:
        super().__init__(config)
        self.settings = AdditionalConfigParameter.validate_dict(self.ADDITIONAL_CONFIG_PARAMS,
                                                                config.additional_params)
        self._prepare_vector_store(self.settings["vector_store_config"])

    def _prepare_vector_store(self, vector_store_config: VectorStoreConfig) -> None:
        vector_store_manager = VectorStoreManager()
        self.vector_store_adapter = vector_store_manager.get_item(
            vector_store_config)

    @override
    def retrieve(self, query_text: str) -> RetrievalAnswer:
        contexts = self.vector_store_adapter.query(query_text,
                                                   n_results=self.settings["n_results"])
        if not contexts:
            logger.info("No contexts found for the given query.")
            return RetrievalAnswer(contexts=[])
        return RetrievalAnswer(contexts=contexts)
