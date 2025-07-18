from typing import Optional
from typing_extensions import override

from sqa_system.core.language_model.llm_provider import LLMProvider
from sqa_system.core.config.models import (
    AdditionalConfigParameter,
    RestrictionType,
    LLMConfig,
    EmbeddingConfig,
    KGRetrievalConfig
)
from sqa_system.retrieval import KnowledgeGraphRetriever
from sqa_system.core.data.models import RetrievalAnswer, Context, ContextType
from sqa_system.knowledge_base.knowledge_graph.storage import KnowledgeGraph
from sqa_system.core.logging.logging import get_logger

from .utils.llm_navigator import LLMNavigator

logger = get_logger(__name__)

DEFAULT_EMBEDDING_CONFIG = EmbeddingConfig(
    additional_params={},
    endpoint="GoogleAI",
    name_model="models/text-embedding-004",
)

DEFAULT_LLM_CONFIG = LLMConfig(
    additional_params={},
    endpoint="OpenAI",
    name_model="gpt-4o-mini",
    temperature=0.1,
    max_tokens=-1,
)

class FidelisRetriever(KnowledgeGraphRetriever):
    """
    Implementation of the FiDELIS retriever based on the paper from Sui et al.
    "FiDeLiS: Faithful Reasoning in Large Language Model for Knowledge Graph Question Answering"
    URL: https://arxiv.org/abs/2405.13873
    Repo: https://anonymous.4open.science/r/FiDELIS-E7FC
    """
    
    ADDITIONAL_CONFIG_PARAMS = [
        AdditionalConfigParameter(
            name="embedding_config",
            description="The configuration for the embedding model.",
            default_value=DEFAULT_EMBEDDING_CONFIG,
            param_type=EmbeddingConfig,
        ),
        AdditionalConfigParameter(
            name="top_n",
            description=("The amount of rated paths that are considered for the next step per entity expanded"),
            default_value=30,
            param_type=int,
            param_restriction=RestrictionType.GREATER_THAN_ZERO
        ),
        AdditionalConfigParameter(
            name="top_k",
            description=("The actual amount of entities that are expanded in the next iteration."
                         " This is chosen by the llm based on the candidates from each path."),
            default_value=3,
            param_type=int,
            param_restriction=RestrictionType.GREATER_THAN_ZERO            
        ),
        AdditionalConfigParameter(
            name="max_length",
            description="The maximum depth of the path search.",
            default_value=3,
            param_type=int,
            param_restriction=RestrictionType.GREATER_THAN_ZERO
        ),
        AdditionalConfigParameter(
            name="alpha",
            description=("The weight of the path score in the final score "
                         "against the relation and neighbor scores."),
            default_value=0.3,
            param_type=float,
            param_restriction=RestrictionType.GREATER_THAN_ZERO
        ),
        AdditionalConfigParameter(
            name="prematurely_stop_when_paths_are_found",
            description=("By default the retriever will explore depths "
                         "to try to find answers. With this we stop early "
                         "if we found an answer on one depth"),
            default_value=False,
            param_type=bool
        ),
        AdditionalConfigParameter(
            name="use_deductive_reasoning",
            description=("Whether to use the deductive termination prompt "
                         "to determine whether the reasoning path is sufficient "
                        "to answer the question."),
            default_value=False,
            param_type=bool
        ),
        AdditionalConfigParameter(
            name="max_workers",
            description=("The maximum number of workers to use for parallelizing "
                         "the retrieval of relations."),
            default_value=8,
            param_type=int,
            param_restriction=RestrictionType.GREATER_THAN_ZERO
        )
    ]
    
    def __init__(self, config: KGRetrievalConfig, graph: KnowledgeGraph) -> None:
        super().__init__(config, graph)
        self.llm = LLMProvider().get_llm_adapter(config.llm_config)
        self.settings = AdditionalConfigParameter.validate_dict(self.ADDITIONAL_CONFIG_PARAMS,
                                                                config.additional_params)
        
    @override
    def retrieve_knowledge(
        self,
        query_text: str,
        topic_entity_id: Optional[str],
        topic_entity_value: Optional[str]
    ) -> RetrievalAnswer:
        
        args = {
            "top_n": self.settings["top_n"],
            "top_k": self.settings["top_k"],
            "max_length": self.settings["max_length"],
            "alpha": self.settings["alpha"],
            "llm_config": self.config.llm_config,
            "embedding_config": self.settings["embedding_config"],
            "max_workers": self.settings["max_workers"],
            "prematurely_stop_when_paths_are_found": self.settings["prematurely_stop_when_paths_are_found"],
            "use_deductive_reasoning": self.settings["use_deductive_reasoning"],
        }
        
        llm_navigator = LLMNavigator(self.graph, args)
        
        result = llm_navigator.beam_search(query_text, topic_entity_id)
        
        if not result:
            return RetrievalAnswer(contexts=[])
        
        logger.debug(f"Prediction: {result['prediction_llm']}")
        
        contexts = []
        for triple in result["prediction_direct_answer"]:
            context = Context(
                context_type=ContextType.KG,
                text=str(triple)
            )
            contexts.append(context)
        
        return RetrievalAnswer(
            contexts=contexts,
            retriever_answer=result["prediction_llm"])
        
        