from enum import Enum
import random
from typing import List, Optional
from typing_extensions import override
from concurrent.futures import ThreadPoolExecutor
from sqa_system.core.config.models.additional_config_parameter import (
    AdditionalConfigParameter,
    RestrictionType
)
from sqa_system.knowledge_base.knowledge_graph.storage.base.knowledge_graph import KnowledgeGraph
from sqa_system.core.config.models import KGRetrievalConfig
from sqa_system.app.cli.cli_progress_handler import ProgressHandler
from sqa_system.core.data.models.context import ContextType, Context
from sqa_system.core.data.models import RetrievalAnswer
from sqa_system.retrieval.base.knowledge_graph_retriever import KnowledgeGraphRetriever
from sqa_system.core.logging.logging import get_logger

from .utils.sparql_func import (
    generate_answer,
    relation_search_prune,
    entity_search,
    entity_score,
    update_history,
    entity_prune,
    reasoning,
    half_stop,
    id2entity_name_or_type,
)
from .utils.utils import generate_without_explored_paths, if_finish_list


logger = get_logger(__name__)


class ToGTechnique(Enum):
    """
    The available techniques for ToG
    """
    LLM = "llm"
    BM25 = "bm25"
    SENTENCE_BERT = "sentence_bert"


class ToGKnowledgeGraphRetriever(KnowledgeGraphRetriever):
    """
    The following class is a implementation of the Think-on-Graph retriever.
    The repository can be found here: https://github.com/IDEA-FinAI/ToG/tree/7ccbb92e17579f934bb778386230de47eca0ab67
    We used commit: 934064c

    Their code has been adapted to work with this project.
    """

    ADDITIONAL_CONFIG_PARAMS = [
        AdditionalConfigParameter(
            name="width",
            description="The amount of relations from an entity that are explored.",
            param_type=int,
            available_values=[],
            default_value=5,
            param_restriction=RestrictionType.GREATER_THAN_ZERO
        ),
        AdditionalConfigParameter(
            name="max_depth",
            description="The maximum depth of the search.",
            param_type=int,
            available_values=[],
            default_value=3,
            param_restriction=RestrictionType.GREATER_THAN_ZERO
        ),
        AdditionalConfigParameter(
            name="retain_threshold",
            description="If the retriever uses an LLM and the amount of entities is larger than this threshold, the entities are reduced.",
            param_type=int,
            available_values=[],
            default_value=20,
            param_restriction=RestrictionType.GREATER_THAN_ZERO
        ),
        AdditionalConfigParameter(
            name="n_retain_entity",
            description="If the retriever uses an LLM and the amount of entities is larger than a threshold, the entities are reduced to this amount.",
            param_type=int,
            available_values=[],
            default_value=5,
            param_restriction=RestrictionType.GREATER_THAN_ZERO
        ),
        AdditionalConfigParameter(
            name="technique",
            description="The technique that is used for searching in the knowledge graph.",
            param_type=str,
            available_values=["llm", "bm25", "sentence_bert"],
            default_value="llm"
        ),
        AdditionalConfigParameter(
            name="bidirectional",
            description="If the search is bidirectional.",
            param_type=bool,
            available_values=[True, False],
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
        self.tog_technique = self.config.additional_params.get("technique", ToGTechnique.LLM.value)
        self.width = self.config.additional_params.get("width", 5)
        self.max_depth = self.config.additional_params.get("max_depth", 10)
        self.retain_threshold = self.config.additional_params.get("retain_threshold", 20)
        self.n_retain_entity = self.config.additional_params.get("n_retain_entity", 10)
        self.temperature_reasoning = config.llm_config.temperature
        self.model_name = config.llm_config.name_model
        self.temperature_exploration = config.llm_config.temperature
        self.bidirectional = self.config.additional_params.get("bidirectional", True)
        self.replace_contribution = self.config.additional_params.get("replace_contribution_name", True)
        self.max_workers = self.config.additional_params.get("max_workers", 8)

    @override
    def retrieve_knowledge(self,
                           query_text: str,
                           topic_entity_id: str | None,
                           topic_entity_value: str | None) -> RetrievalAnswer:

        cluster_chain_of_entities, answer = self.run(
            query_text, topic_entity_id, topic_entity_value)
        contexts = self.convert_cluster_chain_of_entities_to_contexts(
            cluster_chain_of_entities)
        return RetrievalAnswer(contexts=contexts, retriever_answer=answer)

    def run(self,
            query_text: str,
            topic_entity_id: Optional[str],
            topic_entity_value: Optional[str]):
        """
        This runs the main function of the ToG Retriever.
        This implementation is adapted from the original code. 
        The original logic stays the same, we changed the calls to work
        with the SQA system.
        """

        separator = "#" * 30
        logger.debug(
            f"SEPARATOR\n{separator}\n New ToG Retrieval\n{separator}")
        logger.debug("Starting ToG Retriever for query: %s, with topic entity: %s",
                     query_text, topic_entity_id)
        logger.debug("Parameters: width: %s, max_depth: %s, retain_threshold: %s, n_retain_entity: %s, temperature_reasoning: %s, temperature_exploration: %s",
                     self.width,
                     self.max_depth,
                     self.retain_threshold,
                     self.n_retain_entity,
                     self.temperature_reasoning,
                     self.temperature_exploration)

        question = query_text
        topic_entity = {topic_entity_id: topic_entity_value}
        cluster_chain_of_entities = []
        if topic_entity_id is None or topic_entity is None or len(topic_entity) == 0:
            logger.debug(
                "No topic entity found for query: %s. Returning empty result.", query_text)

            results = generate_without_explored_paths(
                question=question,
                temperature_reasoning=self.temperature_reasoning,
                llm_config=self.config.llm_config
            )
            return cluster_chain_of_entities, results
            # # save_2_jsonl(question, results, [], file_name=file_save_path)
            # print_results(question, results, cluster_chain_of_entities)

        pre_relations = []
        pre_heads = [-1] * len(topic_entity)
        flag_printed = False
        progress_handler = ProgressHandler()
        progress_task = progress_handler.add_task(
            description="Searching depth",
            string_id="searching_depth",
            total=self.max_depth,
            reset=True,
        )
        for depth in range(1, self.max_depth + 1):
            current_entity_relations_list = []
            i = 0
            for entity, value in topic_entity.items():
                if not isinstance(entity, str) or not isinstance(topic_entity[entity], str):
                    continue
                if entity != "[FINISH_ID]":
                    if topic_entity[entity] is not None:
                        retrieve_relations_with_scores = relation_search_prune(entity,
                                                                               topic_entity[entity],
                                                                               pre_relations,
                                                                               pre_heads[i],
                                                                               question,
                                                                               self.tog_technique,
                                                                               self.width,
                                                                               self.temperature_reasoning,
                                                                               self.model_name,
                                                                               self.graph,
                                                                               self.config.llm_config,
                                                                               self.bidirectional,
                                                                               self.replace_contribution)
                    else:
                        retrieve_relations_with_scores = []
                    current_entity_relations_list.extend(retrieve_relations_with_scores)
                i += 1
            total_candidates = []
            total_scores = []
            total_relations = []
            total_entities_id = []
            total_topic_entities = []
            total_head = []
            total_triples = []
            

            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                for result in executor.map(self._process_relation, 
                                           current_entity_relations_list, 
                                           [question] * len(current_entity_relations_list)):
                    if not result:
                        continue

                    entity_candidates, entity_relation, scores, entity_candidates_id = result
                    (total_candidates, total_scores, total_relations,
                    total_entities_id, total_topic_entities,
                    total_head, total_triples) = update_history(
                        entity_candidates,
                        entity_relation,
                        scores,
                        entity_candidates_id,
                        total_candidates,
                        total_scores,
                        total_relations,
                        total_entities_id,
                        total_topic_entities,
                        total_head,
                        total_triples
                    )

            if len(total_candidates) == 0:
                logger.debug(
                    "No new knowledge added during search depth %d, stop searching.", depth)
                answer = half_stop(question,
                                   cluster_chain_of_entities,
                                   depth,
                                   temperature_reasoning=self.temperature_reasoning,
                                   llm_config=self.config.llm_config)
                progress_handler.finish_by_string_id(progress_task)
                return cluster_chain_of_entities, answer

                # flag_printed = True
                # return answer

            flag, chain_of_entities, entities_id, pre_relations, pre_heads, triples = entity_prune(
                total_triples, total_entities_id, total_relations, total_candidates, total_topic_entities, total_head, total_scores, self.graph, width=self.width)
            cluster_chain_of_entities.append(chain_of_entities)
            if flag:
                stop, results = reasoning(question,
                                          cluster_chain_of_entities=cluster_chain_of_entities,
                                          temperature_reasoning=self.temperature_reasoning,
                                          llm_config=self.config.llm_config)
                if stop:
                    logger.debug("ToG stoped at depth %d.", depth)
                    progress_handler.finish_by_string_id(progress_task)
                    return cluster_chain_of_entities, results

                    # # save_2_jsonl(
                    # #     question, results, cluster_chain_of_entities, file_name=file_save_path)
                    # print_results(question, results, cluster_chain_of_entities)
                    # flag_printed = True
                    # return str(results)
                else:
                    logger.debug("depth %d still not find the answer.", depth)
                    flag_finish, entities_id = if_finish_list(entities_id)
                    if flag_finish:
                        logger.debug(
                            "No new knowledge added during search depth %d, stop searching.", depth)

                        answer = half_stop(question,
                                           cluster_chain_of_entities=cluster_chain_of_entities,
                                           depth=depth,
                                           llm_config=self.config.llm_config,
                                           temperature_reasoning=self.temperature_reasoning)
                        progress_handler.finish_by_string_id(progress_task)
                        return cluster_chain_of_entities, answer
                        # # flag_printed = True
                        # return answer
                    else:
                        topic_entity = {entity: id2entity_name_or_type(
                            entity, self.graph) for entity in entities_id}
                        progress_handler.update_task_by_string_id(progress_task)
                        continue
            else:
                logger.debug(
                    "No new knowledge added during search depth %d, stop searching.", depth)

                answer = half_stop(question,
                                   cluster_chain_of_entities=cluster_chain_of_entities,
                                   depth=depth,
                                   llm_config=self.config.llm_config,
                                   temperature_reasoning=self.temperature_reasoning)
                progress_handler.finish_by_string_id(progress_task)
                return cluster_chain_of_entities, answer
                # # flag_printed = True
                # return answer
            

        if not flag_printed:

            results = generate_without_explored_paths(question=question,
                                                      llm_config=self.config.llm_config,
                                                      temperature_reasoning=self.temperature_reasoning)
            return [], results
            # # save_2_jsonl(question, results, [], file_name=file_save_path)
            # print_results(question, results, cluster_chain_of_entities)
            # return str(results)

        logger.debug("A answer was not found in the knowledge base.")
        return [], "A answer was not found in the knowledge base."
    
    def _process_relation(self, 
                          entity_relation, 
                          question):
        """
        NEW: Added parallel implementation to speed up the process.
        """
        if entity_relation['head']:
            entity_candidates_id = entity_search(
                entity_relation['entity'], entity_relation['relation'], self.graph, True)
        elif self.bidirectional:
            entity_candidates_id = entity_search(
                entity_relation['entity'], entity_relation['relation'], self.graph, False)
        else:
            return None

        if self.tog_technique == ToGTechnique.LLM.value and len(entity_candidates_id) >= self.retain_threshold:
            entity_candidates_id = random.sample(entity_candidates_id, self.n_retain_entity)

        if not entity_candidates_id:
            return None

        scores, entity_candidates, entity_candidates_id = entity_score(
            question,
            entity_candidates_id,
            entity_relation['score'],
            entity_relation['relation'],
            width=self.width,
            temperature_exploration=self.temperature_exploration,
            technique=self.tog_technique,
            graph=self.graph,
            llm_config=self.config.llm_config
        )

        return entity_candidates, entity_relation, scores, entity_candidates_id

    def convert_cluster_chain_of_entities_to_contexts(self, cluster_chain_of_entities: list) -> List[Context]:
        """
        Converts the cluster chain of entities to a list of contexts.
        """
        contexts: List[Context] = []
        for sublist in cluster_chain_of_entities:
            for chain in sublist:
                for triple_list in chain:
                    for triple in triple_list[3]:
                        triple_str = str(triple)
                        contexts.append(Context(
                            context_type=ContextType.KG,
                            text=triple_str
                        ))
        # reverse the contexts
        contexts = contexts[::-1]
        return contexts
