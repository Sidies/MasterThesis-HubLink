from typing import List, Optional, Tuple
from typing_extensions import override


from sqa_system.core.data.models import RetrievalAnswer
from sqa_system.core.logging.logging import get_logger

from ..models import (
    IsHubOptions,
    EntityWithDirection,
    Hub,
    ProcessedQuestion
)
from ..utils.hub_finder import HubFinder
from .base_retrieval_strategy import BaseRetrievalStrategy

logger = get_logger(__name__)


class TraversalRetrievalStrategy(BaseRetrievalStrategy):
    """
    This retrieval strategy traverses the graph from a given topic entity
    and retrieves considers all hub entities that are reachable from the 
    topic entity on a per level basis. The hubs are then used to generate
    partial answers which are then consolidated into a final answer.

    Args:
        topic_entity_id (str): The ID of the topic entity to start the traversal from.
    """

    def __init__(self, topic_entity_id: str, **kwargs):
        super().__init__(**kwargs)
        self.topic_entity_id = topic_entity_id

    @override
    def _run_retrieval(self, processed_question: ProcessedQuestion) -> Optional[RetrievalAnswer]:
        """
        This is the main loop of the retrieval strategy. First all HubPaths of the current
        level (the depth from the topic entity to the current traversal) are retrieved.
        Then the hubs are pruned and the remaining hubs are used to generate partial answers.
        If no final answer is found, the next level is traversed and the process is repeated
        until the maximum level is reached or no more hubs are found.
        
        Args:
            processed_question (ProcessedQuestion): The processed question containing the
                question, components and embeddings.
                
        Returns:
            Optional[RetrievalAnswer]: The final answer generated from the partial answers
                or None if no final answer was found.
        """

        # Prepare the topic entity by retrieving the Node from the graph
        start_entity = self.graph.get_entity_by_id(self.topic_entity_id)
        if not start_entity:
            logger.error("Topic entity not found in the knowledge graph")
            return None

        # Add the topic entity to the list which is used for finding the hub paths
        # We are traversing the graph in both directions, so we need to add the topic entity
        # as a root entity in both directions
        root_entities = [
            EntityWithDirection(entity=start_entity,
                                left=True, path_from_topic=[]),
            EntityWithDirection(entity=start_entity,
                                left=False, path_from_topic=[])
        ]

        # Add Progress Bar to UI
        level_task = self.progress_handler.add_task(
            description="Traversing levels",
            total=self.settings.max_level,
            string_id="traversing_levels",
            reset=True
        )

        # Start the Loop
        level = 1
        while level <= self.settings.max_level:

            logger.debug("Traversing level %s", level)
            if len(root_entities) == 0:
                logger.debug("No more entities to traverse")
                break

            # Get the Hub candidates that are reachable from the root entities on the current level
            candidate_hubs, next_traversal_candidates = self._get_hubs_at_current_level(
                processed_question=processed_question,
                entities=root_entities,
                level=level
            )
            
            logger.debug(f"Found {len(candidate_hubs)} hubs on level {level} before pruning.")
            
            # Prepare the next traversal entities if we find no final answer on this level
            root_entities = next_traversal_candidates

            # Continue to the next level if no hubs are found on this level
            if not candidate_hubs:
                logger.debug("No hub paths found on level %s", level)
                self.progress_handler.update_task_by_string_id(
                    level_task, advance=1)
                level += 1
                continue

            # Prune the hubs by their total score to only use the specified amount of hubs
            relevant_hubs = self._prune_hubs(
                hubs=candidate_hubs, alpha=self.settings.path_weight_alpha)
            
            logger.debug(f"Found {len(relevant_hubs)} relevant hubs on level {level} after pruning.")

            # Based on the remaining relevant hubs, get the partial answers
            partial_hub_answers = self._get_partial_answers(
                processed_question=processed_question,
                hub_scoring=relevant_hubs
            )

            # If we found at least one partial answer, we try to generate a final answer
            if len(partial_hub_answers) > 0:
                logger.debug("Found answers in hubs: %s", [
                             answer.hub_answer for answer in partial_hub_answers])

                final_answer = self.answer_generator.get_final_answer(
                    question=processed_question.question,
                    hub_answers=partial_hub_answers,
                    settings=self.settings
                )

                # If a final answer was generated, we return it
                # else we proceed to the next level
                if final_answer:
                    logger.debug("Final answer found: %s",
                                 final_answer.retriever_answer)
                    self.progress_handler.finish_by_string_id(level_task)
                    return final_answer
                logger.debug("Insufficient information in hubs")

            self.progress_handler.update_task_by_string_id(
                level_task, advance=1)
            level += 1

        self.progress_handler.finish_by_string_id(level_task)
        return RetrievalAnswer(contexts=[], retriever_answer=None)

    def _get_hubs_at_current_level(
            self,
            processed_question: ProcessedQuestion,
            entities: List[EntityWithDirection],
            level: int) -> Tuple[List[Hub], List[EntityWithDirection]]:
        """
        This method retrieves all Hubs that are reachable from the given root entities
        It first retrieved the root nodes from the graph that are classified as hubs.
        It then retrieves the HubPaths for that node from the vector store by ranking
        the HubPaths by their similarity to the processed question.
        
        If for a given root node of a hub the HubPaths are empty, the hub is updated
        and the HubPaths are retrieved again.
        
        Args:
            processed_question (ProcessedQuestion): The processed question containing the
                question, components and embeddings.
            entities (List[EntityWithDirection]): The list of entities
                to get the HubPaths from.
            level (int): The current level of the traversal.
            
        Returns:
            Tuple[List[Hub], List[EntityWithDirection]]: A tuple
                containing the list of Hub objects representing the candidate hubs
                and the list of next traversal candidates.
        """

        candidate_hub_entities, entities = self._get_hub_roots_at_current_level(
            next_traversal_candidates=entities
        )
        
        logger.debug("Found %s hub roots at level %s", len(candidate_hub_entities), level)

        if not candidate_hub_entities:
            logger.debug("No hubs found at level %s", level)
            return [], entities

        hubs_that_need_processing, hubs = self._get_hub_paths_from_hubs(
            processed_question=processed_question,
            candidate_hub_entities=candidate_hub_entities
        )

        # If the index of a hub needs to be updated, we update them here
        next_entities, fresh_processed_hubs = self.hub_builder.build_hubs(
            hub_entities=hubs_that_need_processing,
            update_cached_hubs=self.settings.check_updates_during_retrieval
        )
        entities.extend(next_entities)

        # For those hubs that were processed, we need to get the hub paths again
        # as they might have changed or added new paths
        for hub in fresh_processed_hubs:
            hub_paths_with_score = self._get_hub_paths_for_hub(
                processed_question=processed_question,
                hub_id=hub.root_entity.entity.uid
            )
            if len(hub_paths_with_score) == 0:
                logger.warning(
                    "No query result found for hub even after processing: %s",
                    hub.root_entity.entity.uid)
                continue
            hubs.append(Hub(
                root_entity=hub.root_entity,
                paths=hub_paths_with_score
            ))
        return hubs, entities

    def _get_hub_roots_at_current_level(
            self,
            next_traversal_candidates: List[EntityWithDirection]
    ) -> Tuple[List[EntityWithDirection], List[EntityWithDirection]]:
        """
        Given the current traversal canidates, this function traverses the graph in the
        directions given in the traversal candidates list to find for each possible path
        those entities that are classified as root entities of a hub.
        
        Args:
            next_traversal_candidates (List[EntityWithDirection]): The list of entities
                to traverse the graph from.
                
        Returns:
            Tuple[List[EntityWithDirection], List[EntityWithDirection]]: A tuple
                containing the list of candidate hub entities and the list of
                next traversal candidates.
        """
        hub_finder = HubFinder(
            graph=self.graph,
            is_hub_options=IsHubOptions(
                hub_edges=self.settings.hub_edges,
                types=self.settings.hub_types
            ),
        )

        # Find the roots of the hubs reachable on the current level
        candidate_hub_entities, next_entities = hub_finder.find_hub_root_entities(
            root_entities=next_traversal_candidates,
            need_same_hop_amount=self.settings.compare_hubs_with_same_hop_amount
        )
        # Prepare the next traversal level
        next_traversal_candidates = next_entities

        return candidate_hub_entities, next_traversal_candidates

    def _get_hub_paths_from_hubs(
            self,
            processed_question: ProcessedQuestion,
            candidate_hub_entities: List[EntityWithDirection]
    ) -> Tuple[List[EntityWithDirection], List[Hub]]:
        """
        This function collects for each root entity of a hub, the HubPaths for that specific
        Hub. If no HubPaths are found, the hub is added to the list of hubs that need
        processing, meaning that their index needs to be updated.
        
        Args:
            processed_question (ProcessedQuestion): The processed question containing the
                question, components and embeddings.
            candidate_hub_entities (List[EntityWithDirection]): The list of entities
                to get the HubPaths from.
                
        Returns:
            Tuple[List[EntityWithDirection], List[Hub]]: A tuple
                containing the list of hubs that need processing and the list of
                Hub objects representing the candidate hubs.
        """

        hubs_that_need_processing: List[EntityWithDirection] = []
        hubs: List[Hub] = []
        if not self.settings.check_updates_during_retrieval:
            # If we don't force the hub update, we can directly do the
            # similarity search on the hubs
            for entity_with_direction in candidate_hub_entities:
                hub_paths = self._get_hub_paths_for_hub(
                    processed_question=processed_question,
                    hub_id=entity_with_direction.entity.uid
                )
                
                if len(hub_paths) == 0:
                    # In case the hub candidate does not return any results,
                    # we need to process the hub
                    hubs_that_need_processing.append(entity_with_direction)
                    continue
                
                hubs.append(Hub(
                    root_entity=entity_with_direction,
                    paths=hub_paths
                ))
        else:
            hubs_that_need_processing = candidate_hub_entities

        return hubs_that_need_processing, hubs
