from typing import List
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate

from sqa_system.core.logging.logging import get_logger
from sqa_system.core.data.models import Triple
from sqa_system.core.language_model.base.llm_adapter import LLMAdapter
from sqa_system.core.language_model.prompt_provider import PromptProvider
from ..base.knowledge_graph import KnowledgeGraph

logger = get_logger(__name__)


class GraphConverter:
    """
    This class is responsible for converting subgraphs and paths into text representations.

    Args:
        llm_adapter (LLMAdapter): The LLM adapter that is used to convert the subgraph and path to text.
        graph (KnowledgeGraph, optional): The knowledge graph to be used. Defaults to None.
    """

    def __init__(self, llm_adapter: LLMAdapter, graph: KnowledgeGraph = None):
        self.graph = graph
        self.llm_adapter = llm_adapter
        self.prompt_provider = PromptProvider()
        self._load_prompts()

    def _load_prompts(self):
        """
        Loads the conversion prompts for subgraphs and paths from the prompt provider.
        """
        self.subgraph_template, self.subgraph_inputs, self.subgraph_partials = (
            self.prompt_provider.get_prompt("conversion/subgraph_to_text.yaml"))
        self.path_template, self.path_inputs, self.path_partials = (
            self.prompt_provider.get_prompt("conversion/path_to_text.yaml"))

    def subgraph_to_description(self, subgraph: List[Triple], llm_adapter: LLMAdapter) -> str:
        """
        Converts a subgraph to a text representation using an LLM.

        Args:
            subgraph (List[Triple]): A list of Triple objects representing the subgraph.
            llm_adapter (LLMAdapter): The LLM adapter that is used to convert the subgraph to text.

        Returns:
            str: A string representation of the subgraph, where the triples of the graph have been
                converted to a natural language description using a language model.
        """
        subgraph_text = self.convert_subgraph_for_llm(subgraph)
        parser = StrOutputParser()

        prompt = PromptTemplate(
            template=self.subgraph_template,
            input_variables=self.subgraph_inputs
        )

        llm_runnable = llm_adapter.llm
        if llm_runnable is None:
            raise ValueError("LLM has not been initialized correctly")
        chain = prompt | llm_runnable | parser

        response = chain.invoke({"subgraph": subgraph_text})
        return response

    def convert_subgraph_for_llm(self, subgraph: List[Triple]) -> str:
        """
        Converts a subgraph of relations into a string format suitable for use with
        language models (LLMs).

        Args:
            subgraph (List[Relation]): A list of Relation objects representing the subgraph.

        Returns:
            str: A string representation of the subgraph, where each relation is formatted
                 as a triple (head_entity_name, relation_desc, tail_entity_name) and each
                 triple is separated by a newline.
        """
        triples = []
        for relation in subgraph:
            if (relation.entity_subject is None or
                relation.entity_subject.uid is None or
                relation.entity_object is None or
                    relation.entity_object.uid is None):
                continue

            head_entity_name = relation.entity_subject.text
            tail_entity_name = relation.entity_object.text

            description = relation.predicate if relation.predicate else ""
            if description == "":
                continue
            if not description:
                continue
            triple = f"({head_entity_name}, {description}, {tail_entity_name}) "
            triples.append(triple)
        return "\n".join(triples)

    def path_to_text(self, path: List[Triple]) -> str:
        """
        Converts a path to a text representation using an LLM.

        Args:
            path (List[Triple]): A list of Triple objects representing the path.

        Returns:
            str: A string representation of the path, where the triples of the graph have been
                converted to a natural language description using a language model.
        """
        logger.debug(
            f"Converting path to text using LLM: {self.llm_adapter.llm_config.name_model}")

        if len(path) == 0:
            return ""

        path_text = self.convert_subgraph_for_llm(path)

        parser = StrOutputParser()

        prompt = PromptTemplate(
            template=self.path_template,
            input_variables=self.path_inputs
        )

        llm_runnable = self.llm_adapter.llm
        if llm_runnable is None:
            raise ValueError("LLM has not been initialized correctly")
        chain = prompt | llm_runnable | parser

        try:
            response = chain.invoke({"path": path_text})
        except Exception as e:
            logger.error(f"Error converting path to text: {e}")
            raise e
        return response
