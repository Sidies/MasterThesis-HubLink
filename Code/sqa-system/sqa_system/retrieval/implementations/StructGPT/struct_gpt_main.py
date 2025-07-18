import os
import argparse
import re
from typing import ClassVar, List
from typing_extensions import override

from sqa_system.core.data.models import RetrievalAnswer, Triple
from sqa_system.core.config.models import KGRetrievalConfig
from sqa_system.knowledge_base.knowledge_graph.storage.base.knowledge_graph import KnowledgeGraph
from sqa_system.core.data.models.context import Context, ContextType
from sqa_system.core.config.models.additional_config_parameter import (
    AdditionalConfigParameter,
    RestrictionType
)
from sqa_system.retrieval.base.knowledge_graph_retriever import KnowledgeGraphRetriever
from sqa_system.core.data.file_path_manager import FilePathManager
from sqa_system.core.logging.logging import get_logger

from .utils.struct_gpt_solver import Solver

logger = get_logger(__name__)


class StructGPTKnowledgeGraphRetriever(KnowledgeGraphRetriever):
    """
    The following class is a implementation of the StructGPT retriever.
    The repository can be found here: https://github.com/RUCAIBox/StructGPT/tree/main
    We used commit: 4bd220a

    Their code has been adapted to work with this project.
    """

    ADDITIONAL_CONFIG_PARAMS: ClassVar[List[AdditionalConfigParameter]] = [
        AdditionalConfigParameter(
            name="max_depth",
            description="The maximum depth of the search.",
            param_type=int,
            available_values=[],
            default_value=3,
            param_restriction=RestrictionType.GREATER_THAN_ZERO
        ),
        AdditionalConfigParameter(
            name="max_llm_serialization_tokens",
            description="The maximum tokens of the serialized data to pass to the LLM",
            param_type=int,
            available_values=[],
            default_value=1024,
            param_restriction=RestrictionType.GREATER_THAN_ZERO
        ),
        AdditionalConfigParameter(
            name="width",
            description=("The amount of relations from an entity that are explored. "
                         "This is forwarded during the LLM predicate selection and "
                         "LLM is asked to conform to this width."),
            param_type=int,
            available_values=[],
            default_value=5,
            param_restriction=RestrictionType.GREATER_THAN_ZERO
        ),
        AdditionalConfigParameter(
            name="bidirectional",
            description="If true, retrieval will consider both outgoing (tail) and incoming (head) relations.",
            param_type=bool,
            available_values=[],
            default_value=True
        ),
        AdditionalConfigParameter(
            name="replace_contribution_name",
            description=("If true, it replaces the string 'contribution' with 'paper content' "
                         "as this might be more meaningful for the LLM"),
            param_type=bool,
            available_values=[],
            default_value=True
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
        # initialize the args
        self.args = argparse.Namespace()
        current_directory = os.path.dirname(os.path.realpath(__file__))
        self.args.prompt_path = FilePathManager().combine_paths(
            current_directory,
            "utils",
            "prompt_for_webqsp.json")
        self.args.prompt_name = "chat_v1"
        self.args.llm_config = config.llm_config
        self.args.max_depth = self.config.additional_params["max_depth"]
        self.args.max_llm_input_tokens = self.config.additional_params["max_llm_serialization_tokens"]
        self.args.width = self.config.additional_params["width"]
        self.args.bidirectional = self.config.additional_params.get("bidirectional", True)
        self.args.replace_contribution_name = self.config.additional_params.get("replace_contribution_name", True)
        self.args.max_workers = self.config.additional_params.get("max_workers", 8)

    @override
    def retrieve_knowledge(self,
                           query_text: str,
                           topic_entity_id: str | None,
                           topic_entity_value: str | None) -> RetrievalAnswer:

        separator = "#" * 30
        logger.debug(
            f"SEPARATOR\n{separator}\n New StructGPT Retrieval\n{separator}")
        logger.debug("Starting StructGPT Retriever for query: %s, with topic entity: %s",
                     query_text, topic_entity_id)

        solver = Solver(self.args, self.graph)
        record = []
        prediction = ""
        chat_history = []
        final_filtered_triples = []
        try:
            question = query_text
            tpe_name = topic_entity_value
            tpe_id = topic_entity_id

            prediction, chat_history, record, final_filtered_triples = solver.forward_v2(
                question, tpe_name, tpe_id)
            logger.debug("prediction: %s, chat_history: %s, record: %s",
                         prediction, chat_history, record)
        except Exception as e:
            logger.error(f"Error in StructGPT retrieval: {e}")

        contexts = self._convert_result_to_contexts(final_filtered_triples)

        return RetrievalAnswer(contexts=contexts, retriever_answer=prediction)

    def _convert_result_to_contexts(self, triples: List[Triple]) -> List[Context]:
        contexts = []
        for triple in triples:
            context = Context(
                context_type=ContextType.KG,
                text=str(triple),
            )
            contexts.append(context)
        return contexts
