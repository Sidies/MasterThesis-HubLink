from collections import deque
import itertools
from typing import Optional, List, Tuple
import pandas as pd
from typing_extensions import override
import re
from concurrent.futures import ThreadPoolExecutor, as_completed

from langchain_core.prompts import (
    PromptTemplate
)
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.schema import (
    AIMessage,
    HumanMessage,
    SystemMessage,
)

from sqa_system.app.cli.cli_progress_handler import ProgressHandler
from sqa_system.core.config.models import (
    EmbeddingConfig,
    KGRetrievalConfig
)
from sqa_system.core.language_model.llm_provider import LLMProvider
from sqa_system.core.config.models import AdditionalConfigParameter, RestrictionType

from sqa_system.retrieval import KnowledgeGraphRetriever
from sqa_system.core.data.models import RetrievalAnswer, Knowledge, Triple, Context, ContextType
from sqa_system.knowledge_base.knowledge_graph.storage import KnowledgeGraph
from sqa_system.core.logging.logging import get_logger

from .utils.vector_store import ChromaVectorStore
from .utils.mindmap_indexer import MindMapIndexer

logger = get_logger(__name__)

DEFAULT_EMBEDDING_CONFIG = EmbeddingConfig(
    additional_params={},
    endpoint="GoogleAI",
    name_model="models/text-embedding-004",
)


class MindMapRetriever(KnowledgeGraphRetriever):
    """
    Implementation of the MindMap Retriever based on the paper from Wang et al.
    "MindMap: Knowledge Graph Prompting Sparks Graph of Thoughts in Large Language Models"
    
    URL: https://arxiv.org/pdf/2308.09729
    Repo: https://github.com/wyl-willing/MindMap/tree/main Commit: 0411a54
    """

    ADDITIONAL_CONFIG_PARAMS = [
        AdditionalConfigParameter(
            name="embedding_config",
            description="The configuration for the embedding model.",
            default_value=DEFAULT_EMBEDDING_CONFIG,
            param_type=EmbeddingConfig,
        ),
        AdditionalConfigParameter(
            name="final_paths_to_keep",
            description=("Determines how many of the computed evidence "
                         "paths from the graph are retained for the final "
                         "answer generation."),
            default_value=5,
            param_type=int,
            param_restriction=RestrictionType.GREATER_THAN_ZERO
        ),
        AdditionalConfigParameter(
            name="neighbor_entities_to_keep",
            description=("The maximum number of one hop neighbor relationships "
                         "to include when building the prompt for the LLM"),
            default_value=5,
            param_type=int,
            param_restriction=RestrictionType.GREATER_THAN_ZERO
        ),
        AdditionalConfigParameter(
            name="shortest_paths_to_keep",
            description=("The maximum number of candidate shortests paths that are "
                         "retained during the shortest path search between entties. "
                         "In other words, during the search only this number of "
                         "paths are kept if there are many candidates."),
            default_value=1,
            param_type=int,
            param_restriction=RestrictionType.GREATER_THAN_ZERO
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
        self.embedding_model = LLMProvider().get_embeddings(
            self.settings["embedding_config"])
        vector_store_name = (f"{self.graph.config.config_hash}_"
                             f"{self.settings['embedding_config'].config_hash}")
        self.vector_store = ChromaVectorStore(
            store_name=vector_store_name)
        self._index_entities()
        
    def _index_entities(self):
        indexer = MindMapIndexer(
            graph=self.graph,
            llm_adapter=self.llm,
            embedding_adapter=self.embedding_model,
            vector_store=self.vector_store,
        )
        indexer.run_indexing()

    @override
    def retrieve_knowledge(
        self,
        query_text: str,
        topic_entity_id: Optional[str],
        topic_entity_value: Optional[str]
    ) -> RetrievalAnswer:
        """
        The main function to retrieve knowledge from the graph as defined
        by the retrieval interface of the SQA System.
        
        We adapted the original implementation to work with the SQA system 
        without modifying the logic of the original implementation.
        """

        question_kg = self.extract_entities(query_text)
        if len(question_kg) == 0:
            logger.warning("No entities extracted from the question.")
            return RetrievalAnswer(contexts=[])

        # Get the embeddings of the entities in the question
        entity_embeddings = self.embedding_model.embed_batch(question_kg)

        match_kg = []

        # get most similar entities from vector store
        collected_entities = []
        for emb in entity_embeddings:
            query_result = self.vector_store.similarity_search(
                emb, n_results=1, exclude=collected_entities)
            metadata = query_result["metadatas"][0][0]
            match_kg.append(metadata)
            collected_entities.append(metadata["entity"])

        # Find the Shortest Paths between the start entity and the end entity
        if len(match_kg) != 1 or 0:
            start_entity: Knowledge = Knowledge.model_validate_json(match_kg[0]["entity"])
            start_metadata = match_kg[0]
            candidates: List[dict] = match_kg[1:]

            result_path_list = []
            while 1:
                flag = 0
                paths_list = []
                while candidates != []:
                    end_metadata = candidates[0]
                    end_entity = Knowledge.model_validate_json(candidates[0]["entity"])
                    candidates.remove(end_metadata)
                    candidates_text = [Knowledge.model_validate_json(candidate["entity"]).text for candidate in candidates]
                    paths, exist_entity = self.find_shortest_path(
                        start_metadata, candidates_text, end_metadata)
                    
                    path_list = paths
                    if paths == []:
                        flag = 1
                        if candidates == []:
                            flag = 0
                            break
                        # Change the start entity to the next entity in the list
                        start_entity = candidates[0]
                        candidates.remove(start_entity)
                        break
                    else:
                        if path_list != []:
                            paths_list.append(path_list)
                    if exist_entity is not None:
                        for i, candidate in enumerate(candidates[:]): 
                            cand_entity = Knowledge.model_validate_json(candidate["entity"])
                            if cand_entity.text == exist_entity:
                                candidates.pop(i)
                                break

                    # Set the start entity to the end entity of the last path
                    start_entity = end_entity
                # Generate all possible combinations
                final_result_paths = self.combine_lists(*paths_list)

                if final_result_paths != []:
                    result_path_list.extend(final_result_paths)
                if flag == 1:
                    continue
                else:
                    break
                
            
            # Iterate over every path and collect the first element of each
            # non empty path to find out the different starting points among the paths
            start_tmp = []
            for path_new in result_path_list:
            
                if path_new == []:
                    continue
                if path_new[0] not in start_tmp:
                    start_tmp.append(path_new[0])
            
            # Check if no valid candidate paths are found
            if len(start_tmp) == 0:
                    final_result_paths = {}
                    single_path = {}
            else:
                # If there is only one unique starting node, it takes the top five paths from
                # the full list 
                if len(start_tmp) == 1:
                    final_result_paths = result_path_list[:self.settings["final_paths_to_keep"]]
                else:
                    # If there are multiple unique starting nodes the code aims to select
                    # up to the defined number of paths in a way that tries to cover different starting entities
                    final_result_paths = []          
                    if len(start_tmp) >= self.settings["final_paths_to_keep"]:
                        # If there are more than the defined number of unique starting nodes it iterates them
                        # and adds the path only if the starting node is not already in the result
                        # list. If the result list has reached the defined number of paths it stops the iteration
                        for path_new in result_path_list:
                            if path_new == []:
                                continue
                            if path_new[0] in start_tmp:
                                final_result_paths.append(path_new)
                                start_tmp.remove(path_new[0])
                            if len(final_result_paths) == self.settings["final_paths_to_keep"]:
                                break
                    else:
                        # If there are less than the defined number of unique starting nodes, the code tries to
                        # distribute the paths equally among the starting nodes
                        count = self.settings["final_paths_to_keep"] // len(start_tmp)
                        remind = self.settings["final_paths_to_keep"] % len(start_tmp)
                        count_tmp = 0
                        for path_new in result_path_list:
                            if len(final_result_paths) < self.settings["final_paths_to_keep"]:
                                if path_new == []:
                                    continue
                                if path_new[0] in start_tmp:
                                    if count_tmp < count:
                                        final_result_paths.append(path_new)
                                        count_tmp += 1
                                    else:
                                        start_tmp.remove(path_new[0])
                                        count_tmp = 0
                                        if path_new[0] in start_tmp:
                                            final_result_paths.append(path_new)
                                            count_tmp += 1

                                    if len(start_tmp) == 1:
                                        count = count + remind
                            else:
                                break

                try:
                    single_path = result_path_list[0]
                except:
                    single_path = result_path_list
                
        else:
            final_result_paths = {}
            single_path = {}                  

        if len(match_kg) != 1 or 0:
            response_of_KG_list_path = []
            if final_result_paths == {}:
                response_of_KG_list_path = []
            else:
                result_new_path = []
                for total_path_i in final_result_paths:
                    path_input = ""
                    for triple in total_path_i:
                        path_input += triple.entity_subject.text + "->" + triple.predicate + "->" + triple.entity_object.text + "\n"
                    result_new_path.append(path_input)
                
                path = "\n".join(result_new_path)
                response_of_KG_list_path = self.prompt_path_finding(path)
        else:
            response_of_KG_list_path = '{}'
            
        # response_single_path = self.prompt_path_finding(single_path) # 
        
                    
        # Get One Hop neighbors for each entity in the match_kg
        neighbor_list: List[List[Triple]] = []
        for match_entity in match_kg:
            # Because it can happen that an entity is a literal which has no doi, in that
            # case we get the subject of the triple the entity was coming from
            entity_knowledge = Knowledge.model_validate_json(match_entity["entity"])
            if not self.graph.is_intermediate_id(entity_knowledge.uid):
                entity_triple = Triple.model_validate_json(match_entity["triple"])
                entity_knowledge = entity_triple.entity_subject
                
            neighbors = self.graph.get_relations_of_head_entity(entity_knowledge)            
            neighbors.extend(self.graph.get_relations_of_tail_entity(entity_knowledge))
            
            neighbor_list.extend(neighbors)

        # 7. knowledge gragh neighbor entities based prompt generation   
        neighbor_new_list = []
        for neighbor_i in neighbor_list:
            neighbor = "->".join([neighbor_i.entity_subject.text, neighbor_i.predicate, neighbor_i.entity_object.text])
            neighbor_new_list.append(neighbor)
        if len(neighbor_new_list) > self.settings["neighbor_entities_to_keep"]:
            neighbor_input = "\n".join(neighbor_new_list[:self.settings["neighbor_entities_to_keep"]])
        else:
            neighbor_input = "\n".join(neighbor_new_list)
            
        response_of_KG_neighbor = self.prompt_neighbor(neighbor_input)
        
        # 8. Answer generation
        output_all = self.final_answer(query_text, response_of_KG_list_path, response_of_KG_neighbor)
        logger.debug('\nMindMap:\n', output_all)
        
        # Generate the final response
        return RetrievalAnswer(
            contexts=self._convert_final_result_paths_to_context(final_result_paths),
            retriever_answer=self._extract_final_answer(output_all)
        )
        
    def _convert_final_result_paths_to_context(self, 
                                               final_result_paths: List[List[Triple]]) -> List[Context]:
        contexts = []
        for path in final_result_paths:
            for triple in path:
                context = Context(
                    context_type=ContextType.KG,
                    text=str(triple),
                )
                contexts.append(context)
        return contexts
        
    def _extract_final_answer(self, output_all: str) -> str | None:
        if "Output 1:".lower() in output_all.lower() and "Output 2:".lower() in output_all.lower():
            extraction_regex = re.compile(r"Output 1:(.*)Output 2:", re.DOTALL)
                
            match = extraction_regex.search(output_all)
            if match:
                extraction = match.group(1).strip()
                if "#" in extraction:
                    extraction = extraction.replace("#", "")
                return extraction
        logger.warning("The LLM returned a unexpected formatting. Using generation LLM for answer generation.")
        return None

    def extract_entities(self, query_text: str) -> List[str]:

        prompt = """
        Your task is to extract entities from the following questions. You are first given examples of the task and then you are given the question:
        
        Input: What is the research level of the paper titled 'How Developers Discuss Architecture Smells: An Exploratory Study on Stack Overflow'?
        Output: Research level, Paper Title, How Developers Discuss Architecture Smells: An Exploratory Study on Stack Overflow
        
        Input: Which publications discuss the research object known as 'Architecture Extraction'?
        Output: Publications, Research Object, Architecture Extraction
        
        Input: Which publications discussing service orientation as a research object were published before 2018?
        Output: Publications, Research Object, Service orientation, Year, 2018
        
        Input: {Question}
        Output:        
        """
        prompt = prompt.format(Question=query_text)

        answer = self.llm.generate(prompt)

        pattern = re.compile(
            r"(?:Output:|Extracted entities:)\s*(.*)", re.IGNORECASE | re.DOTALL)
        match = pattern.search(answer.content)
        try:
            if match:
                entities_text = match.group(1).strip()
            else:
                # Fallback: if no explicit marker is found, assume the entire answer is the entities string.
                entities_text = answer.content
            entities_text = entities_text.split("\n")[0].strip()

            entities = [entity.strip() for entity in re.split(
                r",\s*", entities_text) if entity.strip()]

            logger.debug(f"Extracted entities: {entities}")
        except Exception as e:
            logger.error(f"Error extracting entities: {e}")
            entities = []

        return entities

    def _fetch_neighbors(self, node: Knowledge):
            """
            NEW: Helper to fetch neighbors from the graph in parallel.
            """
            # Each call below presumably waits on an external API:
            if not self.graph.is_intermediate_id(node.uid):
                return []
            relations = []
            relations.extend(self.graph.get_relations_of_head_entity(node))
            relations.extend(self.graph.get_relations_of_tail_entity(node))
            return relations

    def find_shortest_path(self,
                           start_metadata: dict,
                           candidate_list: list,
                           end_metadata: dict,
                           max_depth: int = 10):
        """
        NEW: The original implementation utilized the shortest paths algorithm provided by the
        graph implementation. We do not have this functionality for our graph implementation.

        Therefore we implemented BFS based on the Pseudocode on Wikipedia:
        https://en.wikipedia.org/wiki/Breadth-first_search
        and adapted the code to work with our graph implementation.
        
        Because traditional BFS is slow, we further adapted the code to work bidirectional based on:
        https://medium.com/@zdf2424/discovering-the-power-of-bidirectional-bfs-a-more-efficient-pathfinding-algorithm-72566f07d1bd
        """

        start_triple = Triple.model_validate_json(start_metadata["triple"])
        start_nodes = [start_triple.entity_subject, start_triple.entity_object]
        
        end_triple = Triple.model_validate_json(end_metadata["triple"])
        end_nodes = [end_triple.entity_subject, end_triple.entity_object]
        
        # For the forward (start) search:
        f_visited = {}
        f_queue = deque()
        for node in start_nodes:
            f_visited[start_triple.model_dump_json()] = (node, [start_triple])
            f_queue.append((node, start_triple))
        
        # For the backward (goal) search:
        b_visited = {}
        b_queue = deque()
        for node in end_nodes:
            b_visited[end_triple.model_dump_json()] = (node, [end_triple])
            b_queue.append((node, end_triple))

        valid_paths = [] # All valid paths found
        candidate_found = None
        found_depth = None
        
        progress_handler = ProgressHandler()
        task = progress_handler.add_task(
            string_id="shortest_path",
            description="Finding shortest paths...",
            total=len(b_queue) + len(f_queue)
        )
        
        while f_queue and b_queue:
            logger.debug(f"Finding shortest path, Forward Queue length: {len(f_queue)}, Backward Queue length: {len(b_queue)}")
            if candidate_found is not None:
                break
            
            # To speed up the search we search in the direction with the smaller queue
            if len(f_queue) < len(b_queue):
                # Forward search
                f_queue_size = len(f_queue)
                
                forward_relations: List[Tuple[Knowledge, Triple, List[Triple]]] = []
                for _ in range(f_queue_size):
                    current_node, current_triple = f_queue.popleft()
                    _, current_path = f_visited[current_triple.model_dump_json()]
                    current_depth = len(current_path)
                    if current_depth >= max_depth:
                        progress_handler.update_task_by_string_id(task, 1)
                        continue
                    # additionally if we already have found a path, we can stop the traversal
                    # of paths that are longer than the shortest path
                    if found_depth is not None and current_depth > found_depth:
                        progress_handler.update_task_by_string_id(task, 1)
                        continue
                    
                    if current_triple.model_dump_json() in b_visited:
                        forward_path = current_path
                        backward_path = b_visited[current_triple.model_dump_json()][1]
                        backward_path = backward_path[::-1]
                        path = forward_path + backward_path
                        total_depth = len(path)

                        # to check that it is actually the shortest
                        if found_depth is None:
                            found_depth = total_depth
                        elif total_depth < found_depth:
                            found_depth = total_depth
                            valid_paths = []
                            
                        if total_depth == found_depth:
                            logger.debug(f"Found match in forward search with {current_triple.model_dump_json()}")
                            logger.debug(f"Forward path {forward_path}")
                            logger.debug(f"Backward path {backward_path}")
                            logger.debug(f"Found path {path}")
                            # Check every node in the complete path for a candidate match
                            for triple in path:
                                if triple.entity_subject.text in candidate_list:
                                    candidate_found = triple.entity_subject.text
                                    
                                    progress_handler.finish_by_string_id(task)
                                    return (path, candidate_found)
                                if triple.entity_object.text in candidate_list:
                                    candidate_found = triple.entity_object.text
                                    progress_handler.finish_by_string_id(task)
                                    return (path, candidate_found)
                            # No candidate found in this path; add it to our valid paths
                            valid_paths.append(path)
                            if len(valid_paths) >= self.settings["shortest_paths_to_keep"]:
                                progress_handler.finish_by_string_id(task)
                                return (valid_paths[:self.settings["shortest_paths_to_keep"]], candidate_found)
                    else:
                        forward_relations.append((current_node, current_triple, current_path))
                      
                neighbors_map = self._get_all_neighbors_in_parallel(
                    forward_relations,
                    max_workers=self.settings["max_workers"])
                
                # Expand the forward search
                for (current_node, current_triple, current_path) in forward_relations:
                    neighbors = neighbors_map[(current_node, current_triple)]
                    progress_handler.update_task_by_string_id(task, 1)
                    for relation in neighbors:
                        
                        neighbor = (relation.entity_object 
                                if relation.entity_subject.uid == current_node.uid 
                                else relation.entity_subject)
                        
                        if relation.model_dump_json() in f_visited:
                            progress_handler.update_task_by_string_id(task, 1)
                            continue
                        
                        new_path = current_path + [relation]
                        if len(new_path) > max_depth:
                            progress_handler.update_task_by_string_id(task, 1)
                            continue
                        f_visited[relation.model_dump_json()] = (neighbor, new_path)
                        f_queue.append((neighbor, relation))
                        progress_handler.update_task_length(task, len(b_queue) + len(f_queue))
            
                        # Now we need to check if the node has already been reached by the backward search
                        if relation.model_dump_json() in b_visited:
                            # We found a path we now need to merge them and make sure that
                            # we have no duplicates
                            forward_path = new_path
                            forward_path = forward_path[:-1] # Remove the last element
                            backward_path = b_visited[relation.model_dump_json()][1]      
                            backward_path = backward_path[::-1]
                            
                            path = forward_path + backward_path
                            total_depth = len(path)
                            
                            # to check that it is actually the shortest
                            if found_depth is None:
                                found_depth = total_depth
                            elif total_depth < found_depth:
                                found_depth = total_depth
                                valid_paths = []
                                
                            if total_depth == found_depth:
                                logger.debug(f"Found match in forward search with {relation.model_dump_json()}")
                                logger.debug(f"Forward path {forward_path}")
                                logger.debug(f"Backward path {backward_path}")
                                logger.debug(f"Found path {path}")
                                # Check every node in the complete path for a candidate match
                                for triple in path:
                                    if triple.entity_subject.text in candidate_list:
                                        candidate_found = triple.entity_subject.text
                                        progress_handler.finish_by_string_id(task)
                                        return (path, candidate_found)
                                    if triple.entity_object.text in candidate_list:
                                        candidate_found = triple.entity_object.text
                                        progress_handler.finish_by_string_id(task)
                                        return (path, candidate_found)
                                # No candidate found in this path; add it to our valid paths
                                valid_paths.append(path)
                                if len(valid_paths) >= self.settings["shortest_paths_to_keep"]:
                                    progress_handler.finish_by_string_id(task)
                                    return (valid_paths[:self.settings["shortest_paths_to_keep"]], candidate_found)
            # Backward search
            else:
                b_queue_size = len(b_queue)
                
                backward_relations: List[Tuple[Knowledge, Triple, List[Triple]]] = []
                for _ in range(b_queue_size):
                    current_node, current_triple = b_queue.popleft()
                    _, current_path = b_visited[current_triple.model_dump_json()]
                    current_depth = len(current_path)
                    if current_depth >= max_depth:
                        progress_handler.update_task_by_string_id(task, 1)
                        continue
                    # additionally if we already have found a path, we can stop the traversal
                    # of paths that are longer than the shortest path
                    if found_depth is not None and current_depth > found_depth:
                        progress_handler.update_task_by_string_id(task, 1)
                        continue
                    
                    if current_triple.model_dump_json() in f_visited:
                        forward_path = f_visited[current_triple.model_dump_json()][1]
                        backward_path = current_path
                        backward_path = backward_path[::-1]

                        path = forward_path + backward_path
                        
                        total_depth = len(path)

                        # to check that it is actually the shortest
                        if found_depth is None:
                            found_depth = total_depth
                        elif total_depth < found_depth:
                            found_depth = total_depth
                            valid_paths = []
                            
                        if total_depth == found_depth:
                            logger.debug(f"Found match in forward search with {current_triple.model_dump_json()}")
                            logger.debug(f"Forward path {forward_path}")
                            logger.debug(f"Backward path {backward_path}")
                            logger.debug(f"Found path {path}")
                            # Check every node in the complete path for a candidate match
                            for triple in path:
                                if triple.entity_subject.text in candidate_list:
                                    candidate_found = triple.entity_subject.text
                                    progress_handler.finish_by_string_id(task)
                                    return (path, candidate_found)
                                if triple.entity_object.text in candidate_list:
                                    candidate_found = triple.entity_object.text
                                    progress_handler.finish_by_string_id(task)
                                    return (path, candidate_found)
                            # No candidate found in this path; add it to our valid paths
                            valid_paths.append(path)
                            if len(valid_paths) >= self.settings["shortest_paths_to_keep"]:
                                progress_handler.finish_by_string_id(task)
                                return (valid_paths[:self.settings["shortest_paths_to_keep"]], candidate_found)
                    else: 
                        backward_relations.append((current_node, current_triple, current_path))
                
                neighbors_map = self._get_all_neighbors_in_parallel(
                    backward_relations, max_workers=self.settings["max_workers"])
                        
                for (current_node, current_triple, current_path) in backward_relations:
                    neighbors = neighbors_map[(current_node, current_triple)]
                    progress_handler.update_task_by_string_id(task, 1)

                    for relation in neighbors:
                        neighbor = (relation.entity_object 
                                        if relation.entity_subject.uid == current_node.uid 
                                        else relation.entity_subject)
                        
                        if relation.model_dump_json() in b_visited:
                            progress_handler.update_task_by_string_id(task, 1)
                            continue
                        
                        new_path = current_path + [relation]
                        if len(new_path) > max_depth:
                            progress_handler.update_task_by_string_id(task, 1)
                            continue
                        b_visited[relation.model_dump_json()] = (neighbor, new_path)
                        b_queue.append((neighbor, relation))
                        progress_handler.update_task_length(task, len(b_queue) + len(f_queue))
            
                        # Now we need to check if the node has already been reached by the forward search
                        if relation.model_dump_json() in f_visited:
                            # We found a path, we now need to merge them and make sure that
                            # we have no duplicates
                            forward_path = f_visited[relation.model_dump_json()][1]
                            forward_path = forward_path[:-1] # Remove the last element 
                            backward_path = new_path[::-1]
                            path = forward_path + backward_path
                            total_depth = len(path)

                            # to check that it is actually the shortest
                            if found_depth is None:
                                found_depth = total_depth
                            elif total_depth < found_depth:
                                found_depth = total_depth
                                valid_paths = []
                                
                            if total_depth == found_depth:
                                logger.debug(f"Found match in forward search with {relation.model_dump_json()}")
                                logger.debug(f"Forward path {forward_path}")
                                logger.debug(f"Backward path {backward_path}")
                                logger.debug(f"Found path {path}")
                                # Check every node in the complete path for a candidate match
                                for triple in path:
                                    if triple.entity_subject.text in candidate_list:
                                        candidate_found = triple.entity_subject.text
                                        progress_handler.finish_by_string_id(task)
                                        return (path, candidate_found)
                                    if triple.entity_object.text in candidate_list:
                                        candidate_found = triple.entity_object.text
                                        progress_handler.finish_by_string_id(task)
                                        return (path, candidate_found)
                                # No candidate found in this path; add it to our valid paths
                                valid_paths.append(path)
                                if len(valid_paths) >= self.settings["shortest_paths_to_keep"]:
                                    progress_handler.finish_by_string_id(task)
                                    return (valid_paths[:self.settings["shortest_paths_to_keep"]], candidate_found)
        progress_handler.finish_by_string_id(task)
        return (valid_paths[:self.settings["shortest_paths_to_keep"]], candidate_found)
    
    def _get_all_neighbors_in_parallel(
        self, 
        relations_to_expand: List[Tuple[Knowledge, Triple, List[Triple]]], 
        max_workers: int = 10) -> dict[Tuple[Knowledge, Triple], List[Triple]]:
        # Get all the neighbors
        neighbors_map = {}
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_key = {}
            for (node, triple, path) in relations_to_expand:
                future_work = executor.submit(self._fetch_neighbors, node)
                future_to_key[future_work] = (node, triple, path)

            for future_work in as_completed(future_to_key):
                node, triple, path = future_to_key[future_work]
                try:
                    neighbors_map[(node, triple)] = future_work.result()
                except Exception as e:
                    logger.error(f"Error fetching neighbors for {node.uid}: {e}")
                    neighbors_map[(node, triple)] = []
        return neighbors_map

    def combine_lists(self, *lists):
        combinations = list(itertools.product(*lists))
        results = []
        for combination in combinations:
            new_combination = []
            for sublist in combination:
                if isinstance(sublist, list):
                    new_combination += sublist
                else:
                    new_combination.append(sublist)
            results.append(new_combination)
        return results
    
    def prompt_path_finding(self, path_input):
        """
        NEW: We adapted the grammar of the prompt as we found the original 
        prompt to have potential translation issues.
        
        We also changed the examples to our use case.
        """
        template = """
        You are given paths from a Knowledge Graph. The paths follow the 'entity->relationship->entity' format. Use this Knowledge Graph Path information to convert them to natural language, respectively. Use single quotation marks for entity name and relation name. And name them as Path-based Evidence 1, Path-based Evidence 2,...
        \n\n
        \n\n
        Example Output:
        Path-based Evidence 1: The publication 'Semantic Differencing for Message-Driven Component & Connector Architectures' has an author linked through the relationship 'http://predicate.org/authors' to the entity 'http://ressource.org/authors_28'.\n\nPath-based Evidence 2: The publication 'Semantic Differencing for Message-Driven Component & Connector Architectures' is associated with the venue type 'Conference' through the relationship 'http://predicate.org/venue_type'.
        \n\n
        \n\n
        Input: 
        {Path}
        \n\n
        \n\n
        Your Output:
        """

        prompt = PromptTemplate(
            template = template,
            input_variables = ["Path"]
        )

        system_message_prompt = SystemMessagePromptTemplate(prompt = prompt)
        system_message_prompt.format(Path = path_input)

        human_template = "{text}"
        human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)

        chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt,human_message_prompt])
        chat_prompt_with_values = chat_prompt.format_prompt(Path = path_input,\
                                                            text={})

        response_of_KG_path = self.llm.generate(str(chat_prompt_with_values))
        return response_of_KG_path.content

    def prompt_neighbor(self, neighbor):
        """
        NEW: We adapted the grammar of the prompt as we found the original 
        prompt to have potential translation issues.
        
        We also changed the examples to our use case.
        """
        template = """
        You are given a Knowledge Graph. The format is of the graph is a list of 'entity->relationship->entity'. Use the knowledge graph information and convert it to natural language, respectively. Use single quotation marks for entity name and relation name. And name them as Neighbor-based Evidence 1, Neighbor-based Evidence 2,...
        \n\n
        \n\n
        Example Output:
        **Neighbor-based Evidence 1:** The author 'Arvid Butting' is related to the resource 'http://ressource.org/authors_28' through the relationship 'http://predicate.org/entry_1'.\n\n**Neighbor-based Evidence 2:** The author 'Oliver Kautz' is related to the resource 'http://ressource.org/authors_28' through the relationship 'http://predicate.org/entry_2'.\n\n**Neighbor-based Evidence 3:** The author 'Bernhard Rumpe' is related to the resource 'http://ressource.org/authors_28' through the relationship 'http://predicate.org/entry_3'.\n\n**Neighbor-based Evidence 4:** The author 'Andreas Wortmann' is related to the resource 'http://ressource.org/authors_28' through the relationship 'http://predicate.org/entry_4'.\n\n**Neighbor-based Evidence 5:** The resource 'http://ressource.org/authors_28' is related to the work 'Semantic Differencing for Message-Driven Component & Connector Architectures' through the relationship 'http://predicate.org/authors'.
        \n\n
        \n\n
        Input:
        {neighbor}
        \n\n
        \n\n
        Your Output:
        """

        prompt = PromptTemplate(
            template = template,
            input_variables = ["neighbor"]
        )

        system_message_prompt = SystemMessagePromptTemplate(prompt = prompt)
        system_message_prompt.format(neighbor = neighbor)

        human_template = "{text}"
        human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)

        chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt,human_message_prompt])
        chat_prompt_with_values = chat_prompt.format_prompt(neighbor = neighbor,\
                                                            text={})

        response_of_KG_neighbor = self.llm.generate(str(chat_prompt_with_values))

        return response_of_KG_neighbor.content
    
    def final_answer(self, question, response_of_KG_list_path, response_of_KG_neighbor):
        """
        NEW: We adapted the grammar of the prompt as we found the original 
        prompt to have potential translation issues.
        
        We also changed the examples to our use case.
        """
        messages  = [
                    SystemMessage(content="You are an excellent researcher. You can answer questions based on the contexts given to you."),
                    HumanMessage(content="I need your help in answering the following question:" + question + "What are the relevant contexts to answer this question?"),
                    AIMessage(content="In the following you get some context that might help you answering the question:\n\n" +  '###'+ response_of_KG_list_path + '\n\n' + '###' + response_of_KG_neighbor),
                    HumanMessage(content= "Ok lets answer the question:" + question + "\n\nThink step by step.\n\n\n"
                                + "Output1: This should be the answer to the given question\n\n"
                                +"Output2: This should be the inference process as a string \n Transport the inference process into the following format:\n Path-based Evidence number('entity name'->'relation name'->...)->Path-based Evidence number('entity name'->'relation name'->...)->Neighbor-based Evidence number('entity name'->'relation name'->...)->Neighbor-based Evidence number('entity name'->'relation name'->...)->result number('entity name')->Path-based Evidence number('entity name'->'relation name'->...)->Neighbor-based Evidence number('entity name'->'relation name'->...). \n\n"
                                +"Output3: Draw a decision tree. The entity or relation in single quotes in the inference process is added as a node with the source of evidence, which is followed by the entity in parentheses.\n\n"
                                + "Here is an example:\n"
                                + """
                                    Output 1:
                                    Based on the provided context, the publications on SpringerLink that focus on research related to architecture extraction are: 1. Determination and Enforcement of Least-Privilege Architecture in Android

                                    Output 2:
                                    Path-based Evidence 5('Determination and Enforcement of Least-Privilege Architecture in Android'->'has a research object related to'->'Architecture Extraction')->Path-based Evidence 9('Determination and Enforcement of Least-Privilege Architecture in Android'->'has a research object related to'->'Architecture Extraction')->result 1('Determination and Enforcement of Least-Privilege Architecture in Android')->Path-based Evidence 8('Determination and Enforcement of Least-Privilege Architecture in Android'->'is published at'->'SpringerLink').

                                    Output 3: 
                                    Determination and Enforcement of Least-Privilege Architecture in Android(Path-based Evidence 5)
                                    └── has a research object related to(Path-based Evidence 5)
                                        └── Architecture Extraction(Path-based Evidence 5)(result 1)
                                            └── is published at(Path-based Evidence 8)
                                                └── SpringerLink(Path-based Evidence 8)
                                        """
                                )]
            
        prompt = ChatPromptTemplate.from_messages(messages)
        result = self.llm.generate(str(prompt))
        output_all = result.content
        return output_all