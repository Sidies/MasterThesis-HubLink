
from typing import List, Optional
from typing_extensions import override

from sqa_system.core.config.models import KGRetrievalConfig
from sqa_system.core.config.models.embedding_config import EmbeddingConfig
from sqa_system.core.data.models import Context
from sqa_system.core.data.models.context import ContextType
from sqa_system.core.language_model.llm_provider import LLMProvider
from sqa_system.retrieval import KnowledgeGraphRetriever
from sqa_system.core.data.models import RetrievalAnswer
from sqa_system.knowledge_base.knowledge_graph.storage import KnowledgeGraph
from sqa_system.core.config.models.additional_config_parameter import (
    AdditionalConfigParameter,
    RestrictionType
)
from sqa_system.core.logging.logging import get_logger

from .difar_vector_store_handler import DifarVectorStoreHandler, VectorStoreHandlerOptions

logger = get_logger(__name__)

DEFAULT_EMBEDDING_CONFIG = EmbeddingConfig(
    name="openai_text-embedding-3-small",
    additional_params={},
    endpoint="OpenAI",
    name_model="text-embedding-3-small"
)
DEFAULT_DISTANCE_METRIC = "l2"
DEFAULT_N_RESULTS = 15
DEFAULT_CONVERT_TO_TEXT = False
DEFAULT_FORCE_INDEX_UPDATE = False
DEFAULT_USE_TOPIC_ENTITY_IF_GIVEN = False


class DifarRetriever(KnowledgeGraphRetriever):
    """
    The DiFaR retriever has been proposed by Baek et al. in 
    'Direct Fact Retrieval from Knowledge Graphs without Entity Linking'.

    This is our implementation of the retriever which is based on the descriptions
    provided in their paper: https://arxiv.org/abs/2305.12416
    """
    ADDITIONAL_CONFIG_PARAMS = [
        AdditionalConfigParameter(
            name="distance_metric",
            description=("The distance metric to use. Available values are 'cosine'"
                         " for cosine similarity, 'l2' for squared L2, 'ip' for inner product."),
            param_type=str,
            available_values=['cosine', 'l2', 'ip'],
            default_value=DEFAULT_DISTANCE_METRIC
        ),
        AdditionalConfigParameter(
            name="embedding_config",
            description="The configuration for the embeddings model.",
            param_type=EmbeddingConfig,
            available_values=[],
            default_value=DEFAULT_EMBEDDING_CONFIG
        ),
        AdditionalConfigParameter(
            name="n_results",
            description=("The number of results to return."),
            param_type=int,
            available_values=[],
            default_value=DEFAULT_N_RESULTS,
            param_restriction=RestrictionType.GREATER_THAN_ZERO
        ),
        AdditionalConfigParameter(
            name="convert_to_text",
            description=(
                "Whether to convert each triple to text using the LLM"),
            param_type=bool,
            available_values=[],
            default_value=DEFAULT_CONVERT_TO_TEXT
        ),
        AdditionalConfigParameter(
            name="force_index_update",
            description=(
                "Whether to update the index each time the retriever is initialized."),
            param_type=bool,
            available_values=[],
            default_value=DEFAULT_FORCE_INDEX_UPDATE
        ),
        AdditionalConfigParameter(
            name="indexing_root_entity_ids",
            description=(
                "The ids of the root entities from which to index the triples."),
            param_type=str,
            available_values=[],
            default_value="['R659055']"
        )
    ]

    def __init__(self, config: KGRetrievalConfig, graph: KnowledgeGraph) -> None:
        super().__init__(config, graph)
        self.settings = AdditionalConfigParameter.validate_dict(
            self.ADDITIONAL_CONFIG_PARAMS, config.additional_params)
        self.llm_adapter = LLMProvider().get_llm_adapter(config.llm_config)
        self.embedding_model = LLMProvider().get_embeddings(
            self.settings["embedding_config"])
        
        if self.settings['convert_to_text']:
            vector_store_name = (f"{config.knowledge_graph_config.config_hash}_"
                            f"{self.settings['embedding_config'].config_hash}_"
                            f"{config.llm_config.config_hash}")
        else:
            vector_store_name = (f"{config.knowledge_graph_config.config_hash}_"
                                f"{self.settings['embedding_config'].config_hash}")
        self.vector_store_handler = DifarVectorStoreHandler(
            graph=graph,        
            embedding_adapter=self.embedding_model,
            llm_adapter=self.llm_adapter,
            options=VectorStoreHandlerOptions(
                convert_to_text=self.settings["convert_to_text"],   
                distance_metric=self.settings["distance_metric"],
                vector_store_name=vector_store_name,
            )
        )
        self.vector_store_handler.run_indexing(
            force_update=self.settings["force_index_update"],
            root_entity_ids=self.settings["indexing_root_entity_ids"]
        )

    @override
    def retrieve_knowledge(
        self,
        query_text: str,
        topic_entity_id: Optional[str],
        topic_entity_value: Optional[str]
    ) -> RetrievalAnswer:
        """
        Function to retrieve knowledge from the knowledge graph.
        """

        triples_with_metadata = None
 
        triples_with_metadata = self.vector_store_handler.query(
            query=query_text,
            k=self.settings["n_results"]
        )

        contexts: List[Context] = []
        for triple, metadata in triples_with_metadata:
            context = Context(
                context_type=ContextType.KG,
                text=(str(triple)),
                metadata=metadata
            )
            contexts.append(context)

        return RetrievalAnswer(contexts=contexts)

    def _prepare_settings(self, config: KGRetrievalConfig) -> dict:
        """
        Function to define the settings for the retriever.
        """
        def to_bool(value) -> bool:
            if isinstance(value, bool):
                return value
            if isinstance(value, str):
                return value.strip().lower() in ['true']
            return bool(value)

        embedding_config_json = config.additional_params.get(
            "embedding_config")
        if embedding_config_json and not isinstance(embedding_config_json, EmbeddingConfig):
            embedding_config = EmbeddingConfig.from_dict(embedding_config_json)
        else:
            embedding_config = DEFAULT_EMBEDDING_CONFIG

        settings = {
            "distance_metric": config.additional_params.get(
                "distance_metric", DEFAULT_DISTANCE_METRIC),
            "embedding_config": embedding_config,
            "n_results": int(config.additional_params.get(
                "n_results", DEFAULT_N_RESULTS)),
            "convert_to_text": to_bool(config.additional_params.get(
                "convert_to_text", DEFAULT_CONVERT_TO_TEXT)),
            "force_index_update": to_bool(config.additional_params.get(
                "force_index_update", DEFAULT_FORCE_INDEX_UPDATE)),
            "use_topic_entity_if_given": to_bool(config.additional_params.get(
                "use_topic_entity_if_given", DEFAULT_USE_TOPIC_ENTITY_IF_GIVEN))
        }
        return settings
